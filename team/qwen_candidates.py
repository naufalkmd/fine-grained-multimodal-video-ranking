"""Add the test-like `qwen` set to the validation candidates, and check it for text-only cues.

Each validation episode with four accepted negatives in team.qwen_negatives output becomes one
5-candidate item: its short positive and those four negatives, shuffled with a fixed seed. The leak
check trains a text-only classifier on the training split's generated candidates, and also scores
fixed heuristics such as picking the shortest candidate. Well above 0.375 means a model could pass
the set without the video. The classifier is a diagnostic only: never select or train with it.

    python -m team.qwen_candidates --generated cuda/outputs/qwen/candidates_v3.jsonl \
        --candidates cuda/outputs/validation/candidates_lexical_csls.json \
        --output cuda/outputs/validation/candidates_qwen.json
"""

import argparse
import difflib
import json
import random
import statistics
from pathlib import Path

from common import save_json
from team.negatives import candidate_set
from team.qwen_negatives import NEGATIVE_KINDS, words
from team.rank_eval import RANDOM_SCORE, rank_score

SET_NAME = "qwen"


def load_generated(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[record["episode_id"]] = record  # A rerun's line replaces an earlier one.
    return records


def complete(record):
    return record["positive"] is not None and len(record["negatives"]) == len(NEGATIVE_KINDS)


def add_qwen_set(candidates, records, seed):
    added = 0
    for entry in candidates:
        record = records.get(entry["episode_id"])
        entry["sets"].pop(SET_NAME, None)
        if record and complete(record):
            rng = random.Random(f"{seed}:{entry['episode_id']}:{SET_NAME}")
            negatives = [record["negatives"][kind] for kind in NEGATIVE_KINDS]
            entry["sets"][SET_NAME] = candidate_set(record["positive"], negatives, rng)
            added += 1
    return added


def centrality(item):
    """Each choice's mean word-level similarity to the other choices in the same item.

    When every negative is one edit of the positive, the positive sits in the middle of the set.
    """
    texts = [words(choice["text"]) for choice in item["choices"]]
    return {choice["id"]: statistics.fmean(
        difflib.SequenceMatcher(a=texts[i], b=texts[j], autojunk=False).ratio()
        for j in range(len(texts)) if j != i) for i, choice in enumerate(item["choices"])}


def heuristic_scores(items):
    """Text-only rules with no training. Well above chance means the set has a text-only cue."""
    rules = {"shortest": lambda item: {c["id"]: -len(words(c["text"])) for c in item["choices"]},
             "longest": lambda item: {c["id"]: len(words(c["text"])) for c in item["choices"]},
             "most_central": centrality}
    results = {}
    for name, rule in rules.items():
        scores = []
        for item in items:
            values = rule(item)
            order = sorted(item["choices"], key=lambda choice: -values[choice["id"]])
            scores.append(rank_score([choice["id"] for choice in order], item["correct"]))
        results[name] = round(statistics.fmean(scores), 4)
    return results


def classifier_score(train_records, items):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts, labels = [], []
    for record in train_records:
        texts.append(record["positive"])
        labels.append(1)
        for negative in record["negatives"].values():
            texts.append(negative)
            labels.append(0)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(vectorizer.fit_transform(texts), labels)
    scores = []
    for item in items:
        values = classifier.decision_function(vectorizer.transform([c["text"] for c in item["choices"]]))
        order = sorted(range(len(values)), key=lambda index: -values[index])
        scores.append(rank_score([item["choices"][i]["id"] for i in order], item["correct"]))
    return round(statistics.fmean(scores), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True, help="Existing candidates file.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.resolve() == args.candidates.resolve():
        parser.error("Write to a new file so earlier runs stay comparable.")
    records = load_generated(args.generated)
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    added = add_qwen_set(candidates, records, args.seed)
    save_json(args.output, candidates)
    items = [entry["sets"][SET_NAME] for entry in candidates if SET_NAME in entry["sets"]]
    train = [r for r in records.values() if r["split"] == "train" and r["positive"] is not None]
    report = {"validation_items": added, "validation_episodes": len(candidates),
              "chance": RANDOM_SCORE, "heuristics": heuristic_scores(items),
              "text_only_classifier": classifier_score(train, items) if train else None,
              "train_records_with_positive": len(train), "output": str(args.output)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
