"""Score an exported submission on fixed 5-candidate validation sets, as the server would call it.

Also checks the checks: the scoring code against known scorers, whether generated negatives can be
spotted from text alone, and whether the model's scores respond to frame order. See
docs/validation.md. These scores depend on our generated negatives and are not leaderboard scores.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

from common import ROOT, choose_device, load_config, save_json
from dataset import load_episodes, load_images
from team.negatives import (
    MIXED_TYPES,
    NEGATIVE_TYPES,
    NegativeGenerator,
    build_validation_sets,
    load_action_labels,
)

RANK_SCORES = (1.0, 0.5, 0.25, 0.125, 0.0)
RANDOM_SCORE = sum(RANK_SCORES) / len(RANK_SCORES)
# About three standard errors above chance for an 834-episode validation split.
LEAK_THRESHOLD = 0.42
ORDER_BLIND_DIFFERENCE = 1e-3


def rank_score(ranking, correct):
    return RANK_SCORES[ranking.index(correct)]


def summarize(scores):
    error = statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
    return {"score": round(statistics.fmean(scores), 4), "standard_error": round(error, 4),
            "episodes": len(scores)}


def set_names(candidates):
    return sorted({name for episode in candidates for name in episode["sets"]})


def score_known_scorers(candidates, seed=2026):
    """Raise if the scoring code gives the wrong answer for scorers with a known result."""
    rng = random.Random(seed)
    oracle, worst, chance = [], [], []
    for episode in candidates:
        for candidate in episode["sets"].values():
            ids = [choice["id"] for choice in candidate["choices"]]
            others = [item for item in ids if item != candidate["correct"]]
            oracle.append(rank_score([candidate["correct"], *others], candidate["correct"]))
            worst.append(rank_score([*others, candidate["correct"]], candidate["correct"]))
            chance.append(rank_score(rng.sample(ids, len(ids)), candidate["correct"]))
    result = {"oracle": statistics.fmean(oracle), "worst": statistics.fmean(worst),
              "random": round(statistics.fmean(chance), 4), "expected_random": RANDOM_SCORE,
              "sets": len(chance)}
    # Four standard errors of a random scorer, whose per-set standard deviation is about 0.36.
    tolerance = 4 * 0.36 / math.sqrt(len(chance))
    if (result["oracle"] != 1 or result["worst"] != 0
            or abs(result["random"] - RANDOM_SCORE) > tolerance):
        raise RuntimeError(f"Scoring code failed its sanity check: {result}")
    return result


def leak_check(train_descriptions, train_labels, candidates, seed=2026):
    """Text-only scores per set. Well above chance means a negative gives itself away.

    A diagnostic only: this classifier's outputs must never select negatives or train a model.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # The start marker lets the classifier use cues such as a description opening with "Finally".
    mark = lambda text: "__start__ " + text
    generator = NegativeGenerator(train_descriptions, train_labels)
    results = {}
    for name in set_names(candidates):
        rng = random.Random(f"{seed}:leak:{name}")
        types = MIXED_TYPES if name == "mixed" else [name]
        negatives = [text for index in range(len(train_descriptions))
                     for text in generator.generate(index, rng.choice(types), 1, rng)]
        texts = [mark(text) for text in [*train_descriptions, *negatives]]
        labels = [1] * len(train_descriptions) + [0] * len(negatives)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
        classifier.fit(vectorizer.fit_transform(texts), labels)
        scores = []
        for episode in candidates:
            candidate = episode["sets"].get(name)
            if candidate is None:
                continue
            values = classifier.decision_function(vectorizer.transform(
                [mark(choice["text"]) for choice in candidate["choices"]]))
            order = sorted(range(len(values)), key=lambda index: -values[index])
            scores.append(rank_score([candidate["choices"][i]["id"] for i in order],
                                     candidate["correct"]))
        results[name] = {**summarize(scores), "leak": statistics.fmean(scores) > LEAK_THRESHOLD}
    return results


def load_submission(artifact, device):
    sys.path.insert(0, str(artifact))
    spec = importlib.util.spec_from_file_location("participant_submission", artifact / "submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Submission(artifact_dir=str(artifact), device=device)


def order_check(predictor, episodes, count, seed=2026):
    """Compare each correct description's score on forward and reversed frames."""
    differences, prefers_forward = [], 0
    for episode in random.Random(seed).sample(episodes, min(count, len(episodes))):
        images = load_images(episode)
        choices = [{"id": "c0", "text": episode.description}]
        try:
            forward = predictor.score(images, choices)[0]
            backward = predictor.score(list(reversed(images)), choices)[0]
        finally:
            for image in images:
                image.close()
        differences.append(abs(forward - backward))
        prefers_forward += forward > backward
    mean_difference = statistics.fmean(differences)
    return {"episodes": len(differences), "mean_abs_difference": round(mean_difference, 6),
            "prefers_forward": round(prefers_forward / len(differences), 4),
            "order_blind": mean_difference < ORDER_BLIND_DIFFERENCE}


def evaluate(predictor, episodes_by_id, candidates):
    """Rank every set through predict(), timing only the prediction calls."""
    scores = {name: [] for name in set_names(candidates)}
    timings = []
    for done, entry in enumerate(candidates, 1):
        images = load_images(episodes_by_id[entry["episode_id"]])
        try:
            for name, candidate in entry["sets"].items():
                start = time.perf_counter()
                ranking = predictor.predict(images, candidate["choices"])
                timings.append(time.perf_counter() - start)
                scores[name].append(rank_score(ranking, candidate["correct"]))
        finally:
            for image in images:
                image.close()
        if done % 50 == 0 or done == len(candidates):
            print(f"Ranked episodes: {done}/{len(candidates)}", flush=True)
    timings.sort()
    return ({name: summarize(values) for name, values in scores.items()},
            {"mean": round(statistics.fmean(timings), 4), "worst": round(timings[-1], 4),
             "p95": round(timings[int(0.95 * (len(timings) - 1))], 4)})


def load_candidates(path, valid_ids, descriptions, labels, seed):
    """Reuse the saved candidates so that scores stay comparable across experiments."""
    path = Path(path)
    if path.exists():
        candidates = json.loads(path.read_text(encoding="utf-8"))
        if [entry["episode_id"] for entry in candidates] != valid_ids:
            raise ValueError(f"Saved candidates use a different split; use a new path: {path}")
        return candidates
    candidates = build_validation_sets(valid_ids, descriptions, labels, seed)
    save_json(path, candidates)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run with splits.json.")
    parser.add_argument("--submission-dir", type=Path,
                        help="Exported submission. Omit to run only the scoring and leak checks.")
    parser.add_argument("--trained-on", required=True,
                        help=f"Comma-separated negative types used in training: {NEGATIVE_TYPES}")
    parser.add_argument("--candidates", type=Path, help="Defaults to validation/candidates.json "
                        "beside the config's output directory.")
    parser.add_argument("--output", type=Path, help="Defaults to rank_eval.json in the run directory.")
    parser.add_argument("--order-episodes", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    trained_on = sorted(args.trained_on.split(","))
    if unknown := set(trained_on) - set(NEGATIVE_TYPES):
        parser.error(f"Unknown negative types: {sorted(unknown)}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = load_config(args.config)
    splits = json.loads((args.run_dir / "splits.json").read_text(encoding="utf-8"))
    episodes = {episode.episode_id: episode for episode in load_episodes(config["data_dir"])}
    labels = {episode_id: load_action_labels(episode) for episode_id, episode in episodes.items()}
    train, valid = splits["train"], splits["valid"]
    if set(train) & set(valid):
        raise ValueError("Train and validation episodes overlap.")
    candidates = load_candidates(
        args.candidates or Path(config["output_dir"]).parent / "validation/candidates.json",
        valid, [episodes[i].description for i in valid], [labels[i] for i in valid], config["seed"])
    report = {"trained_on": trained_on,
              "held_out": [name for name in NEGATIVE_TYPES if name not in trained_on],
              "note": "Scores use our generated negatives; they are not leaderboard scores.",
              "known_scorers": score_known_scorers(candidates, config["seed"]),
              "text_only_leak": leak_check([episodes[i].description for i in train],
                                           [labels[i] for i in train], candidates, config["seed"])}
    print(json.dumps(report, indent=2), flush=True)
    if args.submission_dir:
        predictor = load_submission(args.submission_dir.resolve(), choose_device(args.device))
        report["scores"], report["predict_seconds"] = evaluate(predictor, episodes, candidates)
        report["order_check"] = order_check(predictor, [episodes[i] for i in valid],
                                            args.order_episodes, config["seed"])
    output = args.output or args.run_dir / "rank_eval.json"
    save_json(output, report)
    print(json.dumps({key: report[key] for key in ["scores", "predict_seconds", "order_check"]
                      if key in report}, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
