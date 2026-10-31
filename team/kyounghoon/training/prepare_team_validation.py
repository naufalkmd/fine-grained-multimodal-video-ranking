"""Freeze teammate-generated VALIDATION choices and any required text features."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid
import zipfile

import numpy as np
import torch
from data_contract import load_contracts
from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, verified_file
from prepare_features import load_features
from siglip_model import FrozenSiglip, MODEL_ID, make_text_plan
from team_validation import build_team_candidates

UPSTREAM_COMMIT = "00afd4ccad21fa8a4f62108583de1bd700cb43aa"
UPSTREAM_REPO = "https://github.com/naufalkmd/team07-starterkit"
UPSTREAM_BRANCH = "feat/lexical-negatives"


def read_action_labels(train_zip, manifest, episode_ids):
    """Retain repeated labels, unlike manifest.label_ids which is a set."""
    records = {r["episode_id"]: r for r in manifest["episodes"]}
    result = {}
    with zipfile.ZipFile(train_zip) as archive:
        for ep in episode_ids:
            record = records[ep]
            member = str(PurePosixPath(record["image_members"][0]).parent.parent / "episode.json")
            data = archive.read(member)
            if hashlib.sha256(data).hexdigest() != record["metadata_sha256"]:
                raise ValueError(f"Action metadata differs from fixed data contract: {ep}")
            actions = json.loads(data)["actions"]
            result[ep] = [r["label"] for r in actions]
    return result


def team_source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("team/train_negatives.py", "team/negatives.py", "team/lexical.py")}


def validate_cache(bank, manifest, model_revision):
    index = bank["index"]
    if index["source_fingerprint"] != manifest["source"]["fingerprint"]:
        raise ValueError("Feature cache belongs to different original data")
    if index.get("text_mode") != "truncate64" or index["feature_dim"] != 768:
        raise ValueError("Need the ordered per-frame truncate64 SigLIP cache")
    model = index["model"]
    if model.get("model_id") != MODEL_ID or model.get("snapshot_commit") != model_revision:
        raise ValueError("Feature encoder ID/revision differs")
    if model.get("image_aggregation") != "none; one unit vector per original frame, numeric order retained":
        raise ValueError("Mean-only cache cannot be used for temporal comparison")


def prepare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, split, original = load_contracts(args.contract_dir)
    bank = load_features(args.feature_dir)
    validate_cache(bank, manifest, args.model_revision)
    request = {"schema": 1, "source_fingerprint": manifest["source"]["fingerprint"],
               "split_hash": digest(split), "features": bank["index"]["fingerprint"],
               "mix": args.mix, "fill": args.fill, "seed": args.seed, "epoch": 0, "k": 4,
               "revision": args.model_revision, "upstream_commit": UPSTREAM_COMMIT,
               "upstream_source_hashes": team_source_hashes()}
    immutable_json(output / "request.json", request)
    context = {"request_hash": digest(request), "attempt_id": uuid.uuid4().hex}
    atomic_json(output / "status.json", {**context, "status": "running"})
    try:
        labels = read_action_labels(args.train_zip, manifest, split["val_ids"])
        # Only augmentation types need a tokenizer beyond original-caption hashes.
        types = {part.partition("=")[0].strip() for part in args.mix.split(",") if part.strip()} | {args.fill}
        tokenizer = None
        if types & {"reorder", "added_action", "swapped_action"}:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID, revision=args.model_revision)
        key_fn = (lambda text: digest(make_text_plan(tokenizer, text)["token_inputs"]["truncate64"])) if tokenizer else None
        candidates = build_team_candidates(manifest, split, bank["token_keys"], mix_text=args.mix,
                                            fill=args.fill, seed=args.seed, epoch=0,
                                            action_labels=labels, token_key_fn=key_fn)
        candidates["source_snapshot"] = {"repository": UPSTREAM_REPO, "branch": UPSTREAM_BRANCH,
                                          "commit": UPSTREAM_COMMIT, "sha256": team_source_hashes()}
        immutable_json(output / "candidates.json", candidates)
        all_choices = [c for row in candidates["episodes"] for c in row["choices"]]
        extras = {c["source_episode_id"]: c["text"] for c in all_choices
                  if c["source_episode_id"] not in bank["texts"]}
        extra_record = None
        if extras:
            encoder = FrozenSiglip(args.device, text_batch_size=args.text_batch_size, revision=args.model_revision)
            ids = sorted(extras)
            vectors, plans = [], []
            for start in range(0, len(ids), 128):
                selected = ids[start:start + 128]
                encoded, chunk_plans = encoder.text_features([{"id": source, "text": extras[source]} for source in selected])
                vectors.extend(encoded["truncate64"])
                plans.extend(chunk_plans)
                print(f"Extra validation texts {min(start+128,len(ids))}/{len(ids)}", flush=True)
            target = output / "extra_texts.npz"
            temporary = target.with_name(target.name + "." + context["attempt_id"] + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, ids=np.asarray(ids), vectors=np.asarray(vectors, dtype=np.float32),
                                    token_keys=np.asarray([digest(p["token_inputs"]["truncate64"]) for p in plans]),
                                    full_token_counts=np.asarray([p["full_token_count"] for p in plans], dtype=np.int64))
            os.replace(temporary, target)
            extra_record = file_record(target)
        index = {**context, "source_fingerprint": manifest["source"]["fingerprint"],
                 "split_hash": digest(split), "features": bank["index"]["fingerprint"],
                 "candidates": file_record(output / "candidates.json"), "extra_texts": extra_record,
                 "extra_text_count": len(extras), "episode_count": len(candidates["episodes"]),
                 "request": request}
        atomic_json(output / "index.json", index)
        atomic_json(output / "status.json", {**context, "status": "complete", "index": file_record(output / "index.json")})
        print(json.dumps({"validation_episodes": len(candidates["episodes"]), "extra_texts": len(extras),
                          "mix": args.mix, "fill": args.fill, "seed": args.seed}, indent=2))
        return candidates
    except BaseException as exc:
        atomic_json(output / "status.json", {**context, "status": "failed", "error": str(exc)})
        raise


def load_team_validation(folder, bank, manifest, split):
    """Verify frozen candidates; new text vectors are evaluation-only, never in train pool."""
    folder = Path(folder)
    status = read_json(folder / "status.json")
    if status["status"] != "complete":
        raise ValueError("Team validation preparation is incomplete")
    verified_file(folder / "index.json", status["index"])
    index = read_json(folder / "index.json")
    if any(index[k] != status[k] for k in ("request_hash", "attempt_id")):
        raise ValueError("Stale team-validation index")
    if (index["source_fingerprint"] != manifest["source"]["fingerprint"] or
            index["split_hash"] != digest(split) or index["features"] != bank["index"]["fingerprint"]):
        raise ValueError("Team validation data/split/features differ")
    candidates = read_json(verified_file(folder / "candidates.json", index["candidates"]))
    by_id = {r["episode_id"]: r for r in manifest["episodes"]}
    if [r["episode_id"] for r in candidates["episodes"]] != split["val_ids"]:
        raise ValueError("Team candidates must cover the same validation episodes in order")
    if index["extra_texts"] is not None:
        path = verified_file(folder / "extra_texts.npz", index["extra_texts"])
        with np.load(path, allow_pickle=False) as stored:
            arrays = {k: stored[k] for k in stored.files}
        ids = arrays["ids"].tolist()
        values = arrays["vectors"]
        if (len(set(ids)) != len(ids) or len(ids) != index["extra_text_count"] or
                values.shape != (len(ids), bank["index"]["feature_dim"]) or not np.isfinite(values).all() or
                not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-4)):
            raise ValueError("Invalid extra validation text vectors")
        for i, source in enumerate(ids):
            if source in bank["texts"] or not source.startswith("text:"):
                raise ValueError("Extra validation vectors must never replace original text features")
            bank["texts"][source] = torch.from_numpy(values[i].copy())
            bank["token_keys"][source] = str(arrays["token_keys"][i])
            bank["full_token_counts"][source] = int(arrays["full_token_counts"][i])
    for row in candidates["episodes"]:
        ids = [c["id"] for c in row["choices"]]
        if len(ids) != 5 or len(set(ids)) != 5 or row["correct_choice_id"] not in ids:
            raise ValueError("Team validation requires five unique choice IDs")
        for c in row["choices"]:
            source = c["source_episode_id"]
            if source not in bank["texts"]:
                raise ValueError("Missing validation text features")
            if source in by_id:
                if source not in split["val_ids"] or c["text"] != by_id[source]["description"]:
                    raise ValueError("Original candidate must match its validation source")
            elif source != "text:" + hashlib.sha256(c["text"].encode()).hexdigest():
                raise ValueError("Generated candidate content hash differs")
            if c["id"] == row["correct_choice_id"]:
                if source != row["episode_id"] or c["text"] != by_id[source]["description"]:
                    raise ValueError("Correct candidate must be original description for this episode")
    return candidates, {"request": index["request"], "extra_text_count": index["extra_text_count"],
                        "candidates_sha256": index["candidates"]["sha256"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-zip", required=True)
    p.add_argument("--contract-dir", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--mix", default="lexical=2,similar_episode=2")
    p.add_argument("--fill", default="random")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--text-batch-size", type=int, default=16)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
