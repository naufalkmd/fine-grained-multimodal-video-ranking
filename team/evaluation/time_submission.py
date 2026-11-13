"""Time an exported submission as the server would run it, and project the full test set.

Loads the submission once, then ranks `--episodes` training episodes, each with its own description
and four other episodes' descriptions as candidates. Reports the loading time, seconds per episode
and the projected time for 1,910 episodes against the 15-minute limit. Run it on a Colab T4 to
check the limit on the hardware the rules name.

    python -m team.evaluation.time_submission --submission-dir cuda/smolvlm_confidence_v1_submission
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from common import ROOT, load_config
from dataset import load_episodes, load_images
from team.evaluation.rank_eval import load_submission

TEST_EPISODES = 1910
LIMIT_SECONDS = 15 * 60


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    episodes = load_episodes(load_config(args.config)["data_dir"])
    rng = random.Random(args.seed)
    sample = rng.sample(episodes, args.episodes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start = time.perf_counter()
    predictor = load_submission(args.submission_dir.resolve(), device)
    loading = time.perf_counter() - start
    timings = []
    for episode in sample:
        others = rng.sample([e for e in episodes if e is not episode], 4)
        choices = [{"id": f"c{i}", "text": item.description}
                   for i, item in enumerate([episode, *others])]
        images = load_images(episode)
        try:
            start = time.perf_counter()
            predictor.predict(images, choices)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        finally:
            for image in images:
                image.close()
    timings = timings[3:]  # The first calls include one-off warm-up.
    mean = statistics.fmean(timings)
    projected = loading + mean * TEST_EPISODES
    print(json.dumps({
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "loading_seconds": round(loading, 1), "seconds_per_episode": round(mean, 3),
        "worst_seconds": round(max(timings), 3),
        "projected_minutes": round(projected / 60, 1),
        "limit_minutes": LIMIT_SECONDS / 60, "fits": projected < LIMIT_SECONDS,
        "note": "Image loading by the server isn't included; leave a margin."}, indent=2))


if __name__ == "__main__":
    main()
