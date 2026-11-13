"""Ordered frozen SigLIP features for original sentence spans and token windows.

This module never writes new contest descriptions or changes their wording.
Punctuation boundaries are deterministic, not semantic action boundaries:
abbreviations can be split and one sentence can describe several actions.
The existing whole-description truncate64 representation is kept separately.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

from data_contract import load_contracts
from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, verified_file
from prepare_features import load_features
from prepare_team_validation import validate_cache
from siglip_model import (FrozenSiglip, MAX_TOKENS, MODEL_ID, add_siglip_specials,
                          pad_special_tokens, token_windows)

VERSION = "original-sentence-spans-complete-token-windows-v1"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
BOUNDARY_PATTERN = r"(?<=[.!?])\s+(?=\S)"


def sentence_spans(text):
    """Return original nonempty substrings, discarding boundary whitespace only."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Original description must be nonempty text")
    starts, ends = [0], []
    for match in re.finditer(BOUNDARY_PATTERN, text):
        ends.append(match.start())
        starts.append(match.end())
    ends.append(len(text))
    spans = []
    for start, end in zip(starts, ends):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append({"start": start, "end": end, "text": text[start:end]})
    # A change to the boundary logic must not silently omit non-whitespace data.
    if "".join("".join(x["text"].split()) for x in spans) != "".join(text.split()):
        raise ValueError("Sentence splitting lost original content")
    return spans


def sentence_plan(tokenizer, text):
    """Plan all ordered model inputs; never truncate a sentence's tail.

    Token counts of separately encoded sentences need not equal the token count
    of the whole paragraph, because tokenizer boundaries may differ. Coverage
    is checked against each source sentence's own complete token sequence.
    """
    original = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    segments, sentences = [], []
    for sentence_index, span in enumerate(sentence_spans(text)):
        ids = list(tokenizer.encode(span["text"], add_special_tokens=False, truncation=False))
        windows = token_windows(ids, special_count, MAX_TOKENS)
        if [token for window in windows for token in window] != ids:
            raise ValueError("Token windows lost or duplicated sentence tokens")
        sentences.append({**span, "content_token_count": len(ids),
                          "content_token_ids": ids, "window_count": len(windows)})
        for window_index, window in enumerate(windows):
            padded = pad_special_tokens(tokenizer, window, MAX_TOKENS)
            segments.append({"text": span["text"], "start": span["start"], "end": span["end"],
                             "sentence_index": sentence_index, "window_index": window_index,
                             "token_ids": padded, "content_token_ids": window,
                             "weight": max(1, len(window))})
    denominator = max(len(segments) - 1, 1)
    for index, segment in enumerate(segments):
        segment["position"] = index / denominator
    full_count = len(add_siglip_specials(tokenizer, original))
    return {"version": VERSION, "original_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "original_content_token_count": len(original),
            "original_full_token_count": full_count,
            "original_truncated": full_count > MAX_TOKENS,
            "sentence_count": len(sentences), "segment_count": len(segments),
            "sentence_content_token_count": sum(x["content_token_count"] for x in sentences),
            "encoded_content_token_count": sum(len(x["content_token_ids"]) for x in segments),
            "all_sentence_tokens_encoded": True, "sentences": sentences, "segments": segments}


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("sentence_features.py", "siglip_model.py")}


def _expected_ids(manifest, split, bank):
    ids = [row["episode_id"] for row in manifest["episodes"]]
    if (not ids or len(set(ids)) != len(ids) or
            set(split["train_ids"]) & set(split["val_ids"]) or
            set(split["train_ids"] + split["val_ids"]) != set(ids) or
            set(bank["frames"]) != set(ids) or not set(ids) <= set(bank["texts"])):
        raise ValueError("Need exactly the original train/validation frame coverage")
    return ids


def _audit_row(episode_id, plan):
    keys = ("original_full_token_count", "original_truncated", "sentence_count", "segment_count",
            "sentence_content_token_count", "encoded_content_token_count", "all_sentence_tokens_encoded")
    return {"episode_id": episode_id, **{key: plan[key] for key in keys},
            "plan_sha256": digest(plan),
            "token_inputs_sha256": digest([s["token_ids"] for s in plan["segments"]])}


def _encode_plans(encoder, plans):
    """Encode unique fixed-length inputs once per shard, without chunk pooling."""
    rows, lookup, selections = [], {}, []
    for plan in plans:
        selected = []
        for segment in plan["segments"]:
            key = tuple(segment["token_ids"])
            if key not in lookup:
                lookup[key] = len(rows)
                rows.append(list(key))
            selected.append(lookup[key])
        selections.append(selected)
    vectors = []
    with encoder.torch.inference_mode():
        for start in range(0, len(rows), encoder.text_batch_size):
            inputs = encoder.torch.tensor(rows[start:start + encoder.text_batch_size],
                                          dtype=encoder.torch.long, device=encoder.device)
            vectors.append(encoder._unit(encoder.model.get_text_features(input_ids=inputs)).cpu())
    all_vectors = encoder.torch.cat(vectors).numpy().astype(np.float32)
    return [all_vectors[selection].copy() for selection in selections]


def _check_arrays(arrays, ids, dimension=768):
    if list(arrays["episode_ids"]) != list(ids):
        raise ValueError("Sentence feature shard IDs/order differ")
    offsets = arrays["offsets"]
    if (offsets.shape != (len(ids) + 1,) or offsets.dtype.kind not in "iu" or
            offsets[0] != 0 or np.any(np.diff(offsets) < 1)):
        raise ValueError("Invalid sentence feature offsets")
    size = int(offsets[-1])
    vectors, positions, weights = arrays["vectors"], arrays["positions"], arrays["weights"]
    if (vectors.shape != (size, dimension) or positions.shape != (size,) or weights.shape != (size,) or
            not np.isfinite(vectors).all() or
            not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4) or
            not np.isfinite(positions).all() or not np.isfinite(weights).all() or
            np.any(weights < 1) or not np.equal(weights, np.floor(weights)).all()):
        raise ValueError("Sentence features must be finite unit vectors with positive token weights")
    for start, end in zip(offsets[:-1], offsets[1:]):
        expected = np.arange(int(end - start), dtype=np.float32) / max(int(end - start) - 1, 1)
        if not np.allclose(positions[int(start):int(end)], expected, atol=1e-7):
            raise ValueError("Sentence segments must have exact ordered relative positions")


def prepare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.model_revision != MODEL_REVISION:
        raise ValueError("Sentence comparison requires the same pinned SigLIP revision")
    if min(args.text_batch_size, args.shard_size) < 1:
        raise ValueError("Sentence batch and shard sizes must be positive")
    manifest, split, _ = load_contracts(args.contract_dir)
    bank = load_features(args.feature_dir)
    validate_cache(bank, manifest, args.model_revision)
    ids = _expected_ids(manifest, split, bank)
    request = {"schema": 1, "version": VERSION,
               "source_fingerprint": manifest["source"]["fingerprint"],
               "split_hash": digest(split), "features": bank["index"]["fingerprint"],
               "model_id": MODEL_ID, "revision": args.model_revision, "episode_ids": ids,
               "descriptions_sha256": digest([(r["episode_id"], r["description"]) for r in manifest["episodes"]]),
               "boundary_pattern": BOUNDARY_PATTERN, "max_tokens": MAX_TOKENS,
               "overflow": "ordered non-overlap windows within each source sentence; no tail discarded",
               "position": "segment ordinal divided by max(number of segments minus one, 1)",
               "weight": "number of content tokens in the segment, minimum one",
               "device": args.device, "text_batch_size": args.text_batch_size, "shard_size": args.shard_size,
               "sources": _source_hashes()}
    immutable_json(output / "request.json", request)
    if (output / "status.json").is_file() and read_json(output / "status.json").get("status") == "complete":
        loaded = load_sentence_features(output, bank, manifest, split)
        print(f"Reuse complete sentence features: {len(ids)} original descriptions", flush=True)
        return loaded["index"]
    context = {"request_hash": digest(request), "attempt_id": uuid.uuid4().hex}
    atomic_json(output / "status.json", {**context, "status": "running"})
    started = time.perf_counter()
    try:
        encoder = FrozenSiglip(args.device, text_batch_size=args.text_batch_size, revision=args.model_revision)
        if (encoder.feature_dim != 768 or encoder.provenance.get("model_id") != MODEL_ID or
                encoder.provenance.get("snapshot_commit") != args.model_revision):
            raise ValueError("Sentence encoder does not match the fixed original encoder")
        old_tokenizer = bank["index"]["model"].get("tokenizer", {}).get("backend_sha256")
        new_tokenizer = encoder.provenance.get("tokenizer", {}).get("backend_sha256")
        if old_tokenizer and old_tokenizer != new_tokenizer:
            raise ValueError("Sentence tokenizer differs from original feature tokenizer")
        provenance = {**encoder.provenance, "sentence_feature_version": VERSION,
                      "sentence_features": "individual unit features; source order retained; no learned encoder"}
        immutable_json(output / "encoder_provenance.json", provenance)
        fingerprint = digest({"request": request, "encoder": provenance})
        records, audits, reused, encode_seconds = [], [], 0, 0.0
        for start in range(0, len(ids), args.shard_size):
            batch = manifest["episodes"][start:start + args.shard_size]
            selected_ids = [r["episode_id"] for r in batch]
            plans = [sentence_plan(encoder.tokenizer, r["description"]) for r in batch]
            expected_audit = [_audit_row(ep, plan) for ep, plan in zip(selected_ids, plans)]
            target = output / f"shard-{start:05d}.npz"
            marker = output / (target.name + ".json")
            if marker.exists():
                meta = read_json(marker)
                if (meta["fingerprint"] != fingerprint or meta["ids"] != selected_ids or
                        meta["audit"] != expected_audit):
                    raise ValueError("Sentence shard belongs to different source/tokenization")
                verified_file(target, meta["file"])
                with np.load(target, allow_pickle=False) as arrays:
                    _check_arrays(arrays, selected_ids, encoder.feature_dim)
                reused += len(batch)
            else:
                before = time.perf_counter()
                features = _encode_plans(encoder, plans)
                encode_seconds += time.perf_counter() - before
                arrays = {"episode_ids": np.asarray(selected_ids),
                          "vectors": np.concatenate(features).astype(np.float32),
                          "offsets": np.asarray([0] + list(np.cumsum([len(x) for x in features])), dtype=np.int64),
                          "positions": np.asarray([s["position"] for p in plans for s in p["segments"]], dtype=np.float32),
                          "weights": np.asarray([s["weight"] for p in plans for s in p["segments"]], dtype=np.int64)}
                _check_arrays(arrays, selected_ids, encoder.feature_dim)
                temporary = target.with_name(target.name + "." + context["attempt_id"] + ".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(temporary, target)
                meta = {"fingerprint": fingerprint, "ids": selected_ids, "audit": expected_audit,
                        "file": file_record(target)}
                atomic_json(marker, meta)
            records.append({"name": target.name, **meta})
            audits.extend(expected_audit)
            print(f"Sentence features {min(start+len(batch),len(ids))}/{len(ids)}; reused={reused}", flush=True)
        audit = {"version": VERSION, "episodes": len(ids),
                 "original_truncated_descriptions": sum(r["original_truncated"] for r in audits),
                 "sentence_count": sum(r["sentence_count"] for r in audits),
                 "segment_count": sum(r["segment_count"] for r in audits),
                 "max_segment_count": max(r["segment_count"] for r in audits),
                 "all_sentence_tokens_encoded": all(r["all_sentence_tokens_encoded"] and
                     r["sentence_content_token_count"] == r["encoded_content_token_count"] for r in audits),
                 "plans_sha256": digest([(r["episode_id"], r["plan_sha256"]) for r in audits]),
                 "note": "Punctuation splitting is not semantic action annotation; global truncate64 path remains separate.",
                 "episodes_detail": audits}
        atomic_json(output / "sentence_audit.json", audit)
        index = {**context, "schema": 1, "version": VERSION, "fingerprint": fingerprint,
                 "source_fingerprint": manifest["source"]["fingerprint"], "split_hash": digest(split),
                 "features": bank["index"]["fingerprint"], "feature_dim": encoder.feature_dim,
                 "episode_ids": ids, "model": provenance, "request": request,
                 "shards": records, "audit": file_record(output / "sentence_audit.json"),
                 "timing": {"elapsed_seconds": time.perf_counter() - started,
                            "text_encoding_seconds": encode_seconds, "cached_episodes": reused,
                            "note": "Feature preparation for original train+validation, not an inference benchmark."}}
        atomic_json(output / "index.json", index)
        atomic_json(output / "status.json", {**context, "status": "complete", "index": file_record(output / "index.json")})
        return index
    except BaseException as exc:
        atomic_json(output / "status.json", {**context, "status": "failed", "error": str(exc)})
        raise


def load_sentence_features(folder, bank, manifest, split):
    """Validate and load original caption features without mutating the frame bank."""
    import torch
    folder = Path(folder)
    expected_ids = _expected_ids(manifest, split, bank)
    validate_cache(bank, manifest, MODEL_REVISION)
    status = read_json(folder / "status.json")
    if status.get("status") != "complete":
        raise ValueError("Sentence feature preparation is incomplete")
    index = read_json(verified_file(folder / "index.json", status["index"]))
    if any(index[k] != status[k] for k in ("attempt_id", "request_hash")):
        raise ValueError("Stale sentence feature index")
    request = read_json(folder / "request.json")
    if request != index["request"] or digest(request) != index["request_hash"]:
        raise ValueError("Sentence feature request changed")
    if (index["source_fingerprint"] != manifest["source"]["fingerprint"] or
            index["split_hash"] != digest(split) or index["features"] != bank["index"]["fingerprint"] or
            index["episode_ids"] != expected_ids or index["feature_dim"] != 768 or
            index["version"] != VERSION or index["model"].get("model_id") != MODEL_ID or
            index["model"].get("snapshot_commit") != MODEL_REVISION or
            request["descriptions_sha256"] != digest([(r["episode_id"], r["description"]) for r in manifest["episodes"]])):
        raise ValueError("Sentence features differ from original data, encoder, split or frame cache")
    provenance = read_json(folder / "encoder_provenance.json")
    if provenance != index["model"] or digest({"request": request, "encoder": provenance}) != index["fingerprint"]:
        raise ValueError("Sentence encoder provenance or fingerprint changed")
    verified_file(folder / "sentence_audit.json", index["audit"])
    features, positions, weights, audits = {}, {}, {}, []
    for shard in index["shards"]:
        if shard["fingerprint"] != index["fingerprint"]:
            raise ValueError("Sentence shard fingerprint differs")
        # Bind the sidecar too: a resumed prepare must see the same immutable shard.
        marker = read_json(folder / (shard["name"] + ".json"))
        if marker != {k: v for k, v in shard.items() if k != "name"}:
            raise ValueError("Sentence shard sidecar changed")
        with np.load(verified_file(folder / shard["name"], shard["file"]), allow_pickle=False) as stored:
            arrays = {k: stored[k] for k in stored.files}
        _check_arrays(arrays, shard["ids"], index["feature_dim"])
        if [r["episode_id"] for r in shard["audit"]] != shard["ids"]:
            raise ValueError("Sentence shard audit coverage differs")
        for i, ep in enumerate(shard["ids"]):
            if ep in features:
                raise ValueError("Duplicate sentence episode")
            start, end = map(int, arrays["offsets"][i:i+2])
            if shard["audit"][i]["segment_count"] != end - start:
                raise ValueError("Sentence segment audit differs")
            features[ep] = torch.from_numpy(arrays["vectors"][start:end].copy())
            positions[ep] = torch.from_numpy(arrays["positions"][start:end].copy())
            weights[ep] = torch.from_numpy(arrays["weights"][start:end].copy())
        audits.extend(shard["audit"])
    if list(features) != expected_ids:
        raise ValueError("Sentence features do not cover original episodes in order")
    audit = read_json(folder / "sentence_audit.json")
    if audit["episodes_detail"] != audits or not audit["all_sentence_tokens_encoded"]:
        raise ValueError("Sentence audit differs or reports incomplete token coverage")
    return {"features": features, "positions": positions, "weights": weights, "index": index}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=64)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
