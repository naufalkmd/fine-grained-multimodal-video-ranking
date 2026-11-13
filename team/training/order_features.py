"""Encode every frame and every event once with frozen SigLIP, for training the order aligner.

Saves, per episode, the normalized frame features (frames in time order) and the normalized
features of the description's events (team/evaluation/order_check.py `events`). Validation candidate texts
(`--candidates`, `--generated`) are encoded too, so local checks need no SigLIP calls. Frames are
encoded in large batches in float16; each episode is read once.

    python -m team.training.order_features --output cuda/outputs/order/features.pt
"""

import argparse
import json
import time
from pathlib import Path

import torch

from common import ROOT, load_config
from dataset import load_episodes, load_images
from team.evaluation.order_check import OrderChecker, events


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "cuda/models/siglip")
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "cuda/outputs/validation/candidates_qwen.json")
    parser.add_argument("--generated", type=Path,
                        default=ROOT / "cuda/outputs/qwen/long_rewrite.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    episodes = load_episodes(load_config(args.config)["data_dir"])
    checker = OrderChecker(args.model_dir, "cuda")

    texts = {e.description for e in episodes}
    if args.candidates and args.candidates.exists():
        for entry in json.loads(args.candidates.read_text(encoding="utf-8")):
            for candidate in entry["sets"].values():
                texts.update(choice["text"] for choice in candidate["choices"])
    if args.generated and args.generated.exists():
        for line in args.generated.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            texts.update([record["description"], *record["negatives"].values()])
    unique_events = sorted({part for text in texts for part in events(text)})
    started = time.perf_counter()
    event_features = torch.cat([checker.texts(unique_events[i:i + args.batch_size]).half().cpu()
                                for i in range(0, len(unique_events), args.batch_size)])
    print(f"Events: {len(unique_events)} in {time.perf_counter() - started:.0f}s", flush=True)

    frames, pending, pending_ids = {}, [], []

    def flush():
        features = checker.frames(pending).half().cpu()
        start = 0
        for episode_id, count in pending_ids:
            frames[episode_id] = features[start:start + count]
            start += count
        for image in pending:
            image.close()
        pending.clear()
        pending_ids.clear()

    started = time.perf_counter()
    for done, episode in enumerate(episodes, 1):
        images = load_images(episode)
        pending.extend(images)
        pending_ids.append((episode.episode_id, len(images)))
        if len(pending) >= args.batch_size:
            flush()
        if done % 500 == 0:
            print(f"Episodes: {done}/{len(episodes)}, {time.perf_counter() - started:.0f}s",
                  flush=True)
    if pending:
        flush()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"events": unique_events, "event_features": event_features, "frames": frames,
                "model": str(args.model_dir)}, args.output)
    print(f"Saved {args.output}: {len(frames)} episodes, {len(unique_events)} events", flush=True)


if __name__ == "__main__":
    main()
