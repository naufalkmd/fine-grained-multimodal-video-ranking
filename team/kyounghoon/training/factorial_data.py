"""One controlled, reproducible sentence-order negative for the E0--E3 study.

The rule is the first candidate of team.negatives.NegativeGenerator._reorder:
keep the opening sentence and reverse the remaining sentences. It also matches
reversed_description in team07-starterkit main at 28e97e96f56e0f761f322ec0e13c92fe481cdf62.
https://github.com/naufalkmd/team07-starterkit/blob/28e97e96f56e0f761f322ec0e13c92fe481cdf62/team/negatives.py

Only supplied descriptions are rearranged, in code. No teacher, external text
generation, clause shuffle, semantic verification, or new handwritten caption
is involved. A changed order is a weak negative, not a verified contradiction.
"""
from __future__ import annotations

from collections import Counter
import copy
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np

from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, verified_file
from sentence_features import MODEL_REVISION, _encode_plans, sentence_plan
from siglip_model import FrozenSiglip, MODEL_ID, make_text_plan
from team.negatives import split_sentences

VERSION = "paired-original-reorder-one-v1"
RULE = "keep_first_sentence_reverse_remaining_min_three_no_clause_shuffle"
BASE_MIX = {"random": 2, "lexical": 1, "similar_episode": 1}


def _normal(text):
    return " ".join(text.split()).casefold()


def reorder_text(text):
    """Return the team's first reorder candidate, or None when unchanged."""
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return None
    changed = " ".join([sentences[0], *reversed(sentences[1:])])
    return changed if _normal(changed) != _normal(text) else None


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("factorial_data.py", "sentence_features.py", "siglip_model.py", "team/negatives.py")}


def _full_input_key(paragraph, sentence):
    return digest({"paragraph": paragraph["token_inputs"]["truncate64"],
                   "segments": [{k: row[k] for k in ("token_ids", "position", "weight")}
                                for row in sentence["segments"]]})


def _cue_flags(text):
    # These are audit flags, never an automatic claim that an item is false.
    return sorted({word.casefold() for word in re.findall(
        r"\b(?:finally|then|next|before|after|first|last|previously|subsequently)\b", text, flags=re.I)})


def _validate_original_inputs(manifest, split, bank, sentence_bank):
    ids = [r["episode_id"] for r in manifest["episodes"]]
    train, valid = set(split["train_ids"]), set(split["val_ids"])
    if len(ids) != len(set(ids)) or train & valid or train | valid != set(ids):
        raise ValueError("Reorder preparation requires disjoint complete original splits")
    if set(bank["frames"]) != set(ids) or set(sentence_bank["features"]) != set(ids):
        raise ValueError("Reorder preparation needs original frame and sentence coverage")
    return ids, train, valid


def _validate_arrays(arrays, metadata, dimension):
    rows = list(metadata["by_source"].values())
    ids = [r["aug_id"] for r in rows]
    if arrays["aug_ids"].tolist() != ids:
        raise ValueError("Reorder feature IDs differ from generated text records")
    offsets = arrays["offsets"]
    if offsets.shape != (len(ids) + 1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0:
        raise ValueError("Invalid reordered sentence offsets")
    if len(ids) and np.any(np.diff(offsets) < 1):
        raise ValueError("Every reorder needs at least one sentence feature")
    total = int(offsets[-1])
    if arrays["texts"].shape != (len(ids), dimension) or arrays["sentences"].shape != (total, dimension):
        raise ValueError("Reorder feature dimensions differ")
    for name in ("texts", "sentences"):
        values = arrays[name]
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=-1), 1, atol=2e-4):
            raise ValueError("Reordered features must be finite unit vectors")
    for name in ("positions", "weights"):
        if arrays[name].shape != (total,) or not np.isfinite(arrays[name]).all():
            raise ValueError("Invalid reorder sentence metadata")
    if np.any(arrays["weights"] < 1):
        raise ValueError("Reorder sentence weights must be positive")
    for row, start, end in zip(rows, offsets[:-1], offsets[1:]):
        start, end = int(start), int(end)
        if row["segment_count"] != end - start:
            raise ValueError("Reorder segment count does not match its feature range")
        expected = np.arange(end - start, dtype=np.float32) / max(end - start - 1, 1)
        if not np.allclose(arrays["positions"][start:end], expected, atol=1e-7):
            raise ValueError("Reordered sentence positions were not recomputed")


def _extend(bank, sentence_bank, arrays, metadata):
    import torch
    # Frames and immutable original indexes are shared; no original mapping is mutated.
    new_bank = {**bank, "texts": dict(bank["texts"]), "token_keys": dict(bank["token_keys"]),
                "full_token_counts": dict(bank.get("full_token_counts", {}))}
    new_sentences = {**sentence_bank, **{key: dict(sentence_bank[key])
                                      for key in ("features", "positions", "weights")}}
    _validate_arrays(arrays, metadata, bank["index"]["feature_dim"])
    for i, row in enumerate(metadata["by_source"].values()):
        aug = row["aug_id"]
        if aug in new_bank["texts"] or aug in new_sentences["features"]:
            raise ValueError("Generated ID collides with original feature ID")
        start, end = map(int, arrays["offsets"][i:i+2])
        new_bank["texts"][aug] = torch.from_numpy(arrays["texts"][i].copy())
        new_bank["token_keys"][aug] = row["base_token_key"]
        new_bank["full_token_counts"][aug] = row["full_token_count"]
        new_sentences["features"][aug] = torch.from_numpy(arrays["sentences"][start:end].copy())
        new_sentences["positions"][aug] = torch.from_numpy(arrays["positions"][start:end].copy())
        new_sentences["weights"][aug] = torch.from_numpy(arrays["weights"][start:end].copy())
    return new_bank, new_sentences, metadata


def _load_prepared(output, request, bank, sentence_bank):
    status = read_json(output / "status.json")
    if status.get("status") != "complete" or status.get("request_hash") != digest(request):
        raise ValueError("Reorder cache is incomplete or belongs to another request")
    for name, record in status["artifacts"].items():
        verified_file(output / name, record)
    metadata = read_json(output / "augmentation.json")
    if (metadata["request"] != request or metadata["fingerprint"] !=
            digest({k: v for k, v in metadata.items() if k != "fingerprint"})):
        raise ValueError("Reorder augmentation manifest checksum differs")
    with np.load(output / "features.npz", allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    return _extend(bank, sentence_bank, arrays, metadata)


def prepare_reorder(manifest, split, bank, sentence_bank, output_dir, *, device="cuda", text_batch_size=16):
    """Make all split-local reorder texts once and cache their frozen features.

    Validation rearrangements never enter training schedules. Cache checks bind
    the full supplied text, split, original features, encoder and source code.
    """
    if type(text_batch_size) is not int or text_batch_size < 1:
        raise ValueError("text_batch_size must be a positive integer")
    ids, train, valid = _validate_original_inputs(manifest, split, bank, sentence_bank)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    request = {"version": VERSION, "rule": RULE, "source_fingerprint": manifest["source"]["fingerprint"],
               "split_hash": digest(split), "train_ids": list(split["train_ids"]), "validation_ids": list(split["val_ids"]),
               "descriptions_sha256": digest([(r["episode_id"], r["description"]) for r in manifest["episodes"]]),
               "original_features": bank["index"]["fingerprint"],
               "original_sentence_features": sentence_bank["index"]["fingerprint"],
               "feature_dim": bank["index"]["feature_dim"], "model_id": MODEL_ID, "revision": MODEL_REVISION,
               "device": device, "text_batch_size": text_batch_size, "sources": _source_hashes()}
    immutable_json(output / "request.json", request)
    if (output / "status.json").exists() and read_json(output / "status.json").get("status") == "complete":
        print("Reuse verified reorder text/features", flush=True)
        return _load_prepared(output, request, bank, sentence_bank)
    context = {"request_hash": digest(request), "attempt_id": uuid.uuid4().hex}
    atomic_json(output / "status.json", {**context, "status": "running"})
    encoder = None
    started = time.perf_counter()
    try:
        encoder = FrozenSiglip(device, text_batch_size=text_batch_size, revision=MODEL_REVISION)
        if (encoder.feature_dim != bank["index"]["feature_dim"] or encoder.provenance.get("model_id") != MODEL_ID or
                encoder.provenance.get("snapshot_commit") != MODEL_REVISION):
            raise ValueError("Reorder encoder differs from the original frozen encoder")
        old_tokenizer = bank["index"].get("model", {}).get("tokenizer", {}).get("backend_sha256")
        new_tokenizer = encoder.provenance.get("tokenizer", {}).get("backend_sha256")
        if old_tokenizer and old_tokenizer != new_tokenizer:
            raise ValueError("Reorder tokenizer differs from the original cache")
        records, by_source, plans, texts = [], {}, [], []
        for source in manifest["episodes"]:
            ep, original = source["episode_id"], source["description"]
            changed = reorder_text(original)
            row = {"source_episode_id": ep, "split": "train" if ep in train else "validation", "rule": RULE}
            if changed is None:
                records.append({**row, "eligible": False, "reason": "fewer_than_three_sentences_or_normalized_identity"})
                continue
            original_text = make_text_plan(encoder.tokenizer, original)
            original_sentence = sentence_plan(encoder.tokenizer, original)
            paragraph = make_text_plan(encoder.tokenizer, changed)
            sentence = sentence_plan(encoder.tokenizer, changed)
            old_key = digest(original_text["token_inputs"]["truncate64"])
            if old_key != bank["token_keys"][ep]:
                raise ValueError("Original token inputs differ from frozen features")
            full_key = _full_input_key(paragraph, sentence)
            if full_key == _full_input_key(original_text, original_sentence):
                records.append({**row, "eligible": False, "reason": "identical_full_model_input"})
                continue
            aug_id = "reorder_" + ep + "_" + digest(changed)[:16]
            row.update({"eligible": True, "aug_id": aug_id, "text": changed,
                        "text_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                        "base_token_key": digest(paragraph["token_inputs"]["truncate64"]),
                        "base_token_collision_with_positive": paragraph["token_inputs"]["truncate64"] == original_text["token_inputs"]["truncate64"],
                        "full_input_key": full_key, "full_token_count": paragraph["full_token_count"],
                        "segment_count": sentence["segment_count"], "sentence_count": sentence["sentence_count"],
                        "all_sentence_tokens_encoded": sentence["all_sentence_tokens_encoded"],
                        "language_cue_flags": _cue_flags(changed), "semantic_falsehood_verified": False})
            records.append(row)
            by_source[ep] = row
            plans.append(sentence)
            texts.append(changed)
        encoded_texts, encoded_sentences = [], []
        for start in range(0, len(texts), 64):
            choices = [{"id": str(i), "text": text} for i, text in enumerate(texts[start:start+64])]
            vectors, _ = encoder.text_features(choices)
            encoded_texts.extend(vectors["truncate64"])
            encoded_sentences.extend(_encode_plans(encoder, plans[start:start+64]))
            print(f"Reorder features {min(start+64,len(texts))}/{len(texts)}", flush=True)
        dim = bank["index"]["feature_dim"]
        arrays = {"aug_ids": np.asarray([r["aug_id"] for r in by_source.values()], dtype=str),
                  "texts": np.asarray(encoded_texts, dtype=np.float32).reshape(-1, dim),
                  "sentences": (np.concatenate(encoded_sentences).astype(np.float32) if encoded_sentences else np.empty((0, dim), dtype=np.float32)),
                  "offsets": np.asarray([0] + list(np.cumsum([len(x) for x in encoded_sentences])), dtype=np.int64),
                  "positions": np.asarray([s["position"] for p in plans for s in p["segments"]], dtype=np.float32),
                  "weights": np.asarray([s["weight"] for p in plans for s in p["segments"]], dtype=np.float32)}
        audit = {"total": len(ids), "eligible": len(by_source), "excluded": len(ids)-len(by_source),
                 "eligible_train": sum(ep in train for ep in by_source), "eligible_validation": sum(ep in valid for ep in by_source),
                 "base_collisions_retained": sum(r["base_token_collision_with_positive"] for r in by_source.values()),
                 "full_input_collisions_excluded": sum(r.get("reason") == "identical_full_model_input" for r in records),
                 "language_cue_episodes": sum(bool(r["language_cue_flags"]) for r in by_source.values()),
                 "note": "Rule-generated weak negatives; order changes are not guaranteed to make a caption false."}
        metadata = {"version": VERSION, "request": request, "rule": RULE, "by_source": by_source,
                    "records": records, "audit": audit, "encoder": encoder.provenance}
        metadata["fingerprint"] = digest(metadata)
        _validate_arrays(arrays, metadata, dim)
        temporary = output / ("features.npz." + context["attempt_id"] + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output / "features.npz")
        atomic_json(output / "augmentation.json", metadata)
        atomic_json(output / "status.json", {**context, "status": "complete", "elapsed_seconds": time.perf_counter()-started,
                    "artifacts": {n: file_record(output / n) for n in ("augmentation.json", "features.npz")}})
        return _extend(bank, sentence_bank, arrays, metadata)
    except BaseException as exc:
        atomic_json(output / "status.json", {**context, "status": "failed", "error": str(exc)})
        raise
    finally:
        if encoder is not None:
            del encoder
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def build_order_validation(original_random_rows, augmentation_metadata):
    """Replace only the second negative in an eligible original five-choice row."""
    result = []
    for original in original_random_rows:
        ep = original["episode_id"]
        aug = augmentation_metadata["by_source"].get(ep)
        if aug is None:
            continue
        if aug["split"] != "validation" or aug["source_episode_id"] != ep:
            raise ValueError("Order validation may only use its own validation episode")
        row = copy.deepcopy(original)
        choices = row["choices"]
        if len(choices) != 5 or len({c["id"] for c in choices}) != 5:
            raise ValueError("Order validation needs five uniquely identified choices")
        negatives = [i for i, c in enumerate(choices) if c["id"] != row["correct_choice_id"]]
        if len(negatives) != 4:
            raise ValueError("Original validation positive must occur exactly once")
        slot = negatives[1]
        if _normal(aug["text"]) in {_normal(c["text"]) for i, c in enumerate(choices) if i != slot}:
            continue
        row["replaced_source_episode_id"] = choices[slot]["source_episode_id"]
        choices[slot] = {**choices[slot], "text": aug["text"], "source_episode_id": aug["aug_id"],
                         "negative_type": "reorder", "original_source_episode_id": ep}
        row["reorder_choice_id"] = choices[slot]["id"]
        row["order_negative_verified"] = False
        result.append(row)
    return result


class PairedReorderSchedules:
    """Keep the original schedule; replace precisely one random slot for E2/E3."""
    def __init__(self, base_store, augmentation_metadata, output_dir):
        self.base_store = base_store
        self.train_ids = list(base_store.train_ids)
        self.by_source = augmentation_metadata["by_source"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if dict(base_store.mix) != BASE_MIX or list(base_store.mix) != list(BASE_MIX):
            raise ValueError("Paired study requires ordered random=2,lexical=1,similar_episode=1")
        train = set(self.train_ids)
        for ep in train & self.by_source.keys():
            if self.by_source[ep]["split"] != "train" or self.by_source[ep]["source_episode_id"] != ep:
                raise ValueError("Training reorder comes from outside its own training episode")
        self.metadata = {"version": VERSION, "base_fingerprint": base_store.fingerprint,
                         "augmentation_fingerprint": augmentation_metadata["fingerprint"], "train_ids": self.train_ids,
                         "replace": "second random negative; retain all other original slots and episode order",
                         "fallback": "unchanged original schedule when reorder unavailable or duplicates a choice",
                         "pool": "training_only", "k": 4, "source_sha256": _source_hashes()}
        self.fingerprint = digest(self.metadata)
        immutable_json(self.output_dir / "request.json", {**self.metadata, "fingerprint": self.fingerprint})

    def epoch(self, use_reorder, seed, epoch):
        if type(use_reorder) is not bool:
            raise ValueError("use_reorder must be boolean")
        base, base_audit = self.base_store.epoch("team", seed, epoch)
        if not use_reorder:
            return base, base_audit
        base_path = self.base_store.output_dir / base_audit["file"]["path"]
        verified_file(base_path, base_audit["file"])
        raw = json.loads(gzip.decompress(base_path.read_bytes()))
        records = raw["records"]
        if [r["episode_id"] for r in records] != [r[0] for r in base]:
            raise ValueError("Base type records do not match the actual schedule")
        converted, used, changed, reasons = [], [], 0, Counter()
        for (ep, choices), source in zip(base, records):
            negatives = source["negative_sources"]
            random_slots = [i+1 for i, r in enumerate(negatives) if r["type"] == "random"]
            if len(random_slots) < 2 or choices != [ep] + [r["source"] for r in negatives]:
                raise ValueError("Cannot identify two original random slots in base schedule")
            slot = random_slots[1]
            new = list(choices)
            aug = self.by_source.get(ep)
            record = {"episode_id": ep, "choices_before": choices, "replaced_choice_index": None}
            if aug is None:
                reasons["ineligible"] += 1
            elif _normal(aug["text"]) in {_normal(self.base_store.descriptions[x]) for i, x in enumerate(choices) if i != slot}:
                reasons["duplicate_with_retained_choice"] += 1
            else:
                new[slot] = aug["aug_id"]
                changed += 1
                record.update({"replaced_choice_index": slot, "aug_id": aug["aug_id"], "text": aug["text"],
                               "type": "reorder", "source_episode_id": ep,
                               "base_token_collision_with_positive": aug["base_token_collision_with_positive"]})
            record["choices_after"] = new
            used.append(record)
            converted.append((ep, new))
        payload = {"version": VERSION, "config_fingerprint": self.fingerprint, "seed": seed, "epoch": epoch,
                   "base_schedule_hash": digest(base), "base_file": base_audit["file"], "records": used}
        path = self.output_dir / "reorder" / f"seed{seed}" / f"epoch-{epoch:03d}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = path.with_name(path.name + ".json")
        if marker.exists():
            verified_file(path, read_json(marker)["file"])
        if path.exists():
            try:
                old = json.loads(gzip.decompress(path.read_bytes()))
            except (OSError, EOFError, ValueError) as exc:
                raise ValueError("Corrupt paired reorder schedule") from exc
            if old != payload:
                raise ValueError("Paired reorder schedule differs from immutable original/augmentation inputs")
        else:
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(gzip.compress(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
            os.replace(temporary, path)
        record = {"path": path.relative_to(self.output_dir).as_posix(), **file_record(path)}
        audit = {"mode": "reorder", "seed": seed, "epoch": epoch, "episodes": len(base),
                 "schedule_hash": digest(converted), "base_schedule_hash": digest(base), "config_fingerprint": self.fingerprint,
                 "pool": "training_only", "replaced_episodes": changed, "fallback_episodes": len(base)-changed,
                 "fallback_reasons": dict(reasons), "base_token_collisions_retained": sum(r.get("base_token_collision_with_positive", False) for r in used),
                 "file": record}
        immutable_json(marker, {"audit": audit, "file": record})
        return converted, audit
