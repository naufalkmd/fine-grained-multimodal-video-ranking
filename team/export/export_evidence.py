"""Add fixed reference videos to an exported ranker submission, so that it ranks by image evidence.

The references are chosen from the training split with a fixed seed, and stored as frame features.

    python -m team.export.export_evidence --ranker-submission cuda/temporal_word_swap_v1_submission \
        --frame-cache cuda/outputs/frames/features.pt --splits cuda/outputs/baseline/splits.json \
        --output-dir cuda/evidence_v1_submission
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from common import ROOT


def export(ranker_submission, frame_cache, splits, output_dir, count=32, seed=2026):
    ranker_submission, output_dir = Path(ranker_submission), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    shutil.copytree(ranker_submission, output_dir, dirs_exist_ok=True)
    train = json.loads(Path(splits).read_text(encoding="utf-8"))["train"]
    references = random.Random(seed).sample(train, count)
    cached = torch.load(frame_cache, map_location="cpu", weights_only=True)
    features = [cached[episode] for episode in references]
    longest = max(len(item) for item in features)
    frames = torch.zeros(count, longest, features[0].size(-1), dtype=features[0].dtype)
    mask = torch.zeros(count, longest, dtype=torch.bool)
    for index, item in enumerate(features):
        frames[index, :len(item)] = item
        mask[index, :len(item)] = True
    torch.save({"frames": frames, "mask": mask, "episode_ids": references}, output_dir / "weights/reference_frames.pt")
    shutil.copyfile(ROOT / "team/inference/submission_ranker.py", output_dir / "submission_ranker.py")
    shutil.copyfile(ROOT / "team/inference/submission_evidence.py", output_dir / "submission.py")
    print(f"Exported: {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranker-submission", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--references", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    export(args.ranker_submission, args.frame_cache, args.splits, args.output_dir, args.references, args.seed)


if __name__ == "__main__":
    main()
