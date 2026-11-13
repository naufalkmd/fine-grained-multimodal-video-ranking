"""Cache every ORIGINAL frame, with frozen permitted SigLIP. No feature learning."""
from __future__ import annotations
import argparse
import io
import os
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np
from data_contract import load_contracts, read_manifest
from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, sources_hash, verified_file
from siglip_model import FrozenSiglip, numeric_image_order


class FrameEncoder(FrozenSiglip):
    def frames(self, archive, members):
        from PIL import Image
        result = []
        with self.torch.inference_mode():
            members = numeric_image_order(members)
            for start in range(0, len(members), self.image_batch_size):
                images = []
                try:
                    for member in members[start:start + self.image_batch_size]:
                        with Image.open(io.BytesIO(archive.read(member))) as source:
                            images.append(source.convert("RGB"))
                    pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
                    pixels = pixels.to(device=self.device, dtype=self.dtype)
                    result.append(self._unit(self.model.get_image_features(pixel_values=pixels)).cpu().numpy())
                finally:
                    for image in images:
                        image.close()
        return np.concatenate(result, axis=0).astype(np.float32)


def check_arrays(arrays, ids, dimension=768):
    if list(arrays["episode_ids"]) != list(ids):
        raise ValueError("Feature shard IDs/order differ")
    offsets = arrays["offsets"]
    if offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 1):
        raise ValueError("Invalid frame offsets")
    frames, texts = arrays["frames"], arrays["texts"]
    if frames.shape != (int(offsets[-1]), dimension) or texts.shape != (len(ids), dimension):
        raise ValueError("Feature dimensions differ")
    for features in (frames, texts):
        if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=2e-4):
            raise ValueError("Features must be finite unit vectors")
    if arrays["token_keys"].shape != (len(ids),) or arrays["full_token_counts"].shape != (len(ids),):
        raise ValueError("Tokenizer audit shape differs")


def prepare(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, split, candidates = load_contracts(args.contract_dir)
    if read_manifest(args.train_zip) != manifest:
        raise ValueError("ZIP differs from the fixed original data")
    episodes = manifest["episodes"]
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    request = {"schema": 1, "source": manifest["source"], "revision": args.model_revision,
               "device": args.device, "image_batch": args.image_batch_size, "text_batch": args.text_batch_size,
               "episode_ids": [r["episode_id"] for r in episodes], "shard_size": args.shard_size,
               "sources": sources_hash(), "text_mode": "truncate64"}
    immutable_json(output / "request.json", request)
    request_hash, attempt = digest(request), uuid.uuid4().hex
    status_base = {"request_hash": request_hash, "attempt_id": attempt}
    atomic_json(output / "status.json", {**status_base, "status": "running"})
    started = time.perf_counter()
    try:
        encoder = FrameEncoder(args.device, args.image_batch_size, args.text_batch_size, revision=args.model_revision)
        encoder.provenance["cached_image_representation"] = "all individual unit frame features in numeric order"
        encoder.provenance["cached_text_representation"] = "truncate64 unit features; chunk features discarded"
        encoder.provenance["image_aggregation"] = "none; one unit vector per original frame, numeric order retained"
        encoder.provenance["position_information"] = "sequence order retained; no learned temporal encoder in cache"
        # A runtime or resolved-model change must never silently mix cached features.
        immutable_json(output / "encoder_provenance.json", encoder.provenance)
        fingerprint = digest({"request": request, "encoder": encoder.provenance})
        shard_records, image_seconds, text_seconds, reused = [], 0.0, 0.0, 0
        with zipfile.ZipFile(args.train_zip) as archive:
            for start in range(0, len(episodes), args.shard_size):
                batch = episodes[start:start + args.shard_size]
                ids = [r["episode_id"] for r in batch]
                filename = f"shard-{start:05d}.npz"
                target, marker = output / filename, output / (filename + ".json")
                if marker.exists():
                    meta = read_json(marker)
                    if meta["fingerprint"] != fingerprint or meta["ids"] != ids:
                        raise ValueError("Cached shard belongs to another feature configuration")
                    verified_file(target, meta["file"])
                    with np.load(target, allow_pickle=False) as cached:
                        check_arrays(cached, ids, encoder.feature_dim)
                    reused += len(ids)
                else:
                    before = time.perf_counter()
                    frames = [encoder.frames(archive, r["image_members"]) for r in batch]
                    image_seconds += time.perf_counter() - before
                    before = time.perf_counter()
                    # Only original text and neutral IDs reach the frozen encoder.
                    text, plans = encoder.text_features([{"id": str(i), "text": r["description"]}
                                                        for i, r in enumerate(batch)])
                    text_seconds += time.perf_counter() - before
                    arrays = {"episode_ids": np.asarray(ids), "frames": np.concatenate(frames),
                              "offsets": np.asarray([0] + list(np.cumsum([len(f) for f in frames])), dtype=np.int64),
                              "texts": np.asarray(text["truncate64"], dtype=np.float32),
                              "token_keys": np.asarray([digest(p["token_inputs"]["truncate64"]) for p in plans]),
                              "full_token_counts": np.asarray([p["full_token_count"] for p in plans], dtype=np.int64)}
                    check_arrays(arrays, ids, encoder.feature_dim)
                    tmp = target.with_name(filename + "." + attempt + ".tmp")
                    with tmp.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    os.replace(tmp, target)
                    meta = {"fingerprint": fingerprint, "ids": ids, "file": file_record(target)}
                    atomic_json(marker, meta)
                shard_records.append({"name": filename, **meta})
                print(f"Features {min(start + len(batch), len(episodes))}/{len(episodes)}; reused={reused}", flush=True)
        index = {"schema": 1, **status_base, "fingerprint": fingerprint, "feature_dim": encoder.feature_dim,
                 "episode_ids": request["episode_ids"], "source_fingerprint": manifest["source"]["fingerprint"],
                 "shards": shard_records, "text_mode": "truncate64", "model": encoder.provenance,
                 "timing": {"elapsed_seconds": time.perf_counter() - started, "image_seconds": image_seconds,
                            "text_seconds": text_seconds, "cached_episodes": reused,
                            "note": "Shared train+validation feature preparation; not submission inference benchmark."}}
        atomic_json(output / "index.json", index)
        atomic_json(output / "status.json", {**status_base, "status": "complete", "index": file_record(output / "index.json")})
        return index
    except BaseException as exc:
        atomic_json(output / "status.json", {**status_base, "status": "failed", "error": str(exc)})
        raise


def load_features(folder):
    import torch
    folder = Path(folder)
    status = read_json(folder / "status.json")
    if status["status"] != "complete":
        raise ValueError("Feature preparation is incomplete")
    verified_file(folder / "index.json", status["index"])
    index = read_json(folder / "index.json")
    if index["attempt_id"] != status["attempt_id"] or index["request_hash"] != status["request_hash"]:
        raise ValueError("Stale feature index")
    frames, texts, token_keys, counts = {}, {}, {}, {}
    for shard in index["shards"]:
        verified_file(folder / shard["name"], shard["file"])
        with np.load(folder / shard["name"], allow_pickle=False) as arrays:
            arrays = {key: arrays[key] for key in arrays.files}
            check_arrays(arrays, shard["ids"], index["feature_dim"])
            for i, episode_id in enumerate(shard["ids"]):
                if episode_id in frames:
                    raise ValueError("Duplicate episode in feature cache")
                a, b = int(arrays["offsets"][i]), int(arrays["offsets"][i + 1])
                frames[episode_id] = torch.from_numpy(arrays["frames"][a:b].copy())
                texts[episode_id] = torch.from_numpy(arrays["texts"][i].copy())
                token_keys[episode_id] = str(arrays["token_keys"][i])
                counts[episode_id] = int(arrays["full_token_counts"][i])
    if list(frames) != index["episode_ids"]:
        raise ValueError("Feature index coverage/order differs")
    return {"frames": frames, "texts": texts, "token_keys": token_keys, "full_token_counts": counts, "index": index}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-zip", required=True)
    p.add_argument("--contract-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-batch-size", type=int, default=8)
    p.add_argument("--text-batch-size", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=32)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--max-episodes", type=int)
    args = p.parse_args()
    if args.shard_size < 1 or (args.max_episodes is not None and args.max_episodes < 1):
        p.error("Sizes must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
