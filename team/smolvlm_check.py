"""Quick checks of SmolVLM likelihood scoring (team/smolvlm_scorer.py) before any training.

Scores a sample of validation episodes on fixed candidate sets, with the right video and with
another episode's video, at prior weights 0 (plain likelihood) and 1 (pointwise mutual
information), and runs action pairing (reversing the frames should flip the preference between a
short positive and its reversed-action sentence; chance 0.5). `word_swap` is built here from the
validation descriptions, because the saved candidate files don't contain it. Saves every raw
score so that other combinations can be scored without rerunning the model.

    python -m team.smolvlm_check --episodes 120
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from common import ROOT, load_config, save_json
from dataset import load_episodes, load_images
from team.negatives import NegativeGenerator, candidate_set, load_action_labels
from team.rank_eval import RANK_SCORES
from team.smolvlm_scorer import SmolVLMScorer
from team.submission_central import Submission as Centrality

SETS = ("word_swap", "qwen", "added_action", "swapped_action", "similar_episode")


def rank_of_correct(scores, ids, correct):
    order = sorted(range(len(ids)), key=lambda index: -scores[index])
    return [ids[index] for index in order].index(correct)


def add_word_swap(candidates, episodes, seed):
    descriptions = [episodes[entry["episode_id"]].description for entry in candidates]
    generator = NegativeGenerator(descriptions, [load_action_labels(episodes[entry["episode_id"]])
                                                 for entry in candidates])
    rng = random.Random(seed)
    for index, entry in enumerate(candidates):
        negatives = generator.generate(index, "word_swap", 4, rng)
        if len(negatives) == 4:
            entry["sets"]["word_swap"] = candidate_set(descriptions[index], negatives, rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "cuda/models/smolvlm-256m-instruct")
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "cuda/outputs/validation/candidates_qwen.json")
    parser.add_argument("--generated", type=Path, default=ROOT / "cuda/outputs/qwen/candidates_v3.jsonl")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=ROOT / "cuda/outputs/smolvlm/check.json")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = load_config(args.config)
    episodes = {episode.episode_id: episode for episode in load_episodes(config["data_dir"])}
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    add_word_swap(candidates, episodes, args.seed)
    sample = random.Random(args.seed).sample(
        [entry for entry in candidates if "qwen" in entry["sets"]], args.episodes)
    scorer = SmolVLMScorer(args.model_dir, "cuda", args.max_frames, args.image_size)
    centrality = Centrality(None, "cpu")

    rows, timings = [], []
    for done, entry in enumerate(sample):
        wrong = sample[(done + 1) % len(sample)]["episode_id"]
        right_images = load_images(episodes[entry["episode_id"]])
        wrong_images = load_images(episodes[wrong])
        for name in SETS:
            if name not in entry["sets"]:
                continue
            candidate = entry["sets"][name]
            texts = [choice["text"] for choice in candidate["choices"]]
            start = time.perf_counter()
            right, prior = scorer.components(right_images, texts)
            timings.append(time.perf_counter() - start)
            wrong_scores, _ = scorer.components(wrong_images, texts)
            rows.append({"episode_id": entry["episode_id"], "set": name,
                         "ids": [choice["id"] for choice in candidate["choices"]],
                         "correct": candidate["correct"], "right": right.tolist(),
                         "wrong": wrong_scores.tolist(), "prior": prior.tolist(),
                         "central": centrality.score(None, candidate["choices"])})
        for image in right_images + wrong_images:
            image.close()
        if (done + 1) % 20 == 0:
            print(f"Episodes: {done + 1}/{len(sample)}", flush=True)

    records = [json.loads(line) for line in args.generated.read_text(encoding="utf-8").splitlines()]
    sampled = {entry["episode_id"] for entry in sample}
    records = [r for r in records if r["episode_id"] in sampled and r["positive"]
               and "reversed_action" in r["negatives"]]
    pairs = []
    for record in records:
        images = load_images(episodes[record["episode_id"]])
        texts = [record["positive"], record["negatives"]["reversed_action"]]
        forward, _ = scorer.components(images, texts)
        backward, _ = scorer.components(list(reversed(images)), texts)
        pairs.append({"episode_id": record["episode_id"], "forward": forward.tolist(),
                      "backward": backward.tolist()})
        for image in images:
            image.close()

    report = {"settings": vars(args) | {"model_dir": str(args.model_dir)},
              "seconds_per_call": round(statistics.fmean(timings), 4),
              "summary": summarize(rows, pairs), "rows": rows, "pairs": pairs}
    report["settings"] = {key: str(value) for key, value in report["settings"].items()}
    save_json(args.output, report)
    print(json.dumps({key: report[key] for key in ["seconds_per_call", "summary"]}, indent=2))


def summarize(rows, pairs):
    summary = {}
    for name in SETS:
        selected = [row for row in rows if row["set"] == name]
        if not selected:
            continue
        result = {"episodes": len(selected)}
        for weight in (0.0, 1.0):
            for video in ("right", "wrong"):
                result[f"w{weight:g}_{video}"] = round(statistics.fmean(
                    RANK_SCORES[rank_of_correct(
                        [a - weight * b for a, b in zip(row[video], row["prior"])],
                        row["ids"], row["correct"])] for row in selected), 4)
        result["centrality"] = round(statistics.fmean(
            RANK_SCORES[rank_of_correct(row["central"], row["ids"], row["correct"])]
            for row in selected), 4)
        summary[name] = result
    if pairs:
        gains = [(p["forward"][0] - p["forward"][1]) - (p["backward"][0] - p["backward"][1])
                 for p in pairs]
        summary["action_pairing"] = {"pairs": len(pairs), "score": round(statistics.fmean(
            1.0 if gain > 0 else 0.5 if gain == 0 else 0.0 for gain in gains), 4), "chance": 0.5}
    return summary


if __name__ == "__main__":
    main()
