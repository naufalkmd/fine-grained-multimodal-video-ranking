"""Action pairing: does reversing the frames flip a submission's choice between a short positive and
its reversed-action sentence ("folds" / "unfolds")? Chance is exactly 0.5.

A preference that comes from the wording alone shows up on both frame directions and cancels, so
only a model that reads the direction of the action from the video can score above 0.5. Uses the
validation records from team.candidates.qwen_negatives and any exported ranker submission.

    python -m team.evaluation.action_pairing --config cuda/config.json \
        --generated cuda/outputs/qwen/candidates_v3.jsonl --submission-dir cuda/q2_submission
"""

import argparse
import json
import sys
from pathlib import Path

from common import ROOT, choose_device, load_config
from dataset import load_episodes, load_images
from team.evaluation.rank_eval import load_submission

TIE = 1e-4


def action_pairing(predictor, episodes, records):
    wins = 0.0
    for record in records:
        images = load_images(episodes[record["episode_id"]])
        choices = [{"id": "positive", "text": record["positive"]},
                   {"id": "reversed", "text": record["negatives"]["reversed_action"]}]
        try:
            forward = predictor._logits(images, choices).tolist()
            backward = predictor._logits(list(reversed(images)), choices).tolist()
        finally:
            for image in images:
                image.close()
        gain = (forward[0] - forward[1]) - (backward[0] - backward[1])
        wins += 1.0 if gain > TIE else 0.5 if abs(gain) <= TIE else 0.0
    return {"score": round(wins / len(records), 4), "episodes": len(records), "chance": 0.5}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--generated", type=Path, required=True, help="team.candidates.qwen_negatives output.")
    parser.add_argument("--submission-dir", type=Path, required=True,
                        help="An exported ranker submission (submission_ranker.py).")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    episodes = {episode.episode_id: episode for episode in load_episodes(config["data_dir"])}
    records = [json.loads(line) for line in args.generated.read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r["split"] == args.split and r["positive"]
               and "reversed_action" in r["negatives"]]
    predictor = load_submission(args.submission_dir.resolve(), choose_device(args.device))
    if not hasattr(predictor, "_logits"):
        sys.exit("The submission needs submission_ranker.py's _logits method.")
    print(json.dumps(action_pairing(predictor, episodes, records), indent=2))


if __name__ == "__main__":
    main()
