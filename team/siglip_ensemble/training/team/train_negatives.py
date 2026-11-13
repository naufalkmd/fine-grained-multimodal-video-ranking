"""Training negatives: k per episode, with a chosen count from each type, resampled every epoch.

Built only from the training split, so validation descriptions never become training negatives.
The saved file is the record of negatives actually used, kept for the organizers' inspection.

    python -m team.train_negatives --config cuda/config.json --run-dir cuda/outputs/baseline \
        --k 4 --mix lexical=2,similar_episode=2 --epochs 20
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import ROOT, load_config, save_json
from dataset import load_episodes
from team.negatives import NEGATIVE_TYPES, NegativeGenerator, load_action_labels


def parse_mix(text):
    """'lexical=2,similar_episode=2' -> {'lexical': 2, 'similar_episode': 2}."""
    mix = {}
    for part in filter(None, text.split(",")):
        name, _, count = part.partition("=")
        name = name.strip()
        if name not in NEGATIVE_TYPES:
            raise ValueError(f"Unknown negative type {name!r}; choose from {NEGATIVE_TYPES}")
        if name in mix:
            raise ValueError(f"Negative type listed twice: {name}")
        mix[name] = int(count)
        if mix[name] < 0:
            raise ValueError(f"Negative counts must not be negative: {part}")
    return mix


def check_mix(mix, k, fill):
    if k < 1:
        raise ValueError("k must be at least 1.")
    if sum(mix.values()) > k:
        raise ValueError(f"The mix asks for {sum(mix.values())} negatives but k is {k}.")
    if fill not in NEGATIVE_TYPES:
        raise ValueError(f"Unknown fill type {fill!r}; choose from {NEGATIVE_TYPES}")


def sample_negatives(generator, index, mix, k, fill, rng):
    """k distinct (type, text) pairs: the counts in `mix`, then `fill` for the rest.

    A type that runs short is topped up from `fill`, so the result can hold fewer of that type
    than asked; the returned types record what was actually used.
    """
    chosen = {}

    def take(negative_type, count):
        if count <= 0:
            return
        added = 0
        for text in generator.generate(index, negative_type, count + len(chosen), rng):
            if text not in chosen:
                chosen[text] = negative_type
                added += 1
                if added == count:
                    return

    for negative_type, count in mix.items():
        take(negative_type, count)
    take(fill, k - len(chosen))
    if len(chosen) < k:
        raise ValueError(f"Could not find {k} distinct negatives for description {index}.")
    return [(negative_type, text) for text, negative_type in chosen.items()]


class TrainingNegatives:
    def __init__(self, episode_ids, descriptions, action_labels, mix, k, fill="random", seed=2026):
        check_mix(mix, k, fill)
        self.episode_ids = list(episode_ids)
        self.generator = NegativeGenerator(descriptions, action_labels)
        self.mix, self.k, self.fill, self.seed = dict(mix), k, fill, seed

    def epoch(self, epoch):
        """One list of (type, text) pairs per episode; the same epoch always gives the same list."""
        return [sample_negatives(self.generator, index, self.mix, self.k, self.fill,
                                 random.Random(f"{self.seed}:{episode_id}:train:{epoch}"))
                for index, episode_id in enumerate(self.episode_ids)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run with splits.json.")
    parser.add_argument("--k", type=int, default=4, help="Negatives per episode.")
    parser.add_argument("--mix", default="", help="Counts per type, e.g. lexical=2,similar_episode=2.")
    parser.add_argument("--fill", default="random", help="Type used for the rest of the k.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        help="Defaults to train_negatives.json in the run directory.")
    args = parser.parse_args()
    mix = parse_mix(args.mix)
    check_mix(mix, args.k, args.fill)
    config = load_config(args.config)
    train_ids = json.loads((args.run_dir / "splits.json").read_text(encoding="utf-8"))["train"]
    episodes = {episode.episode_id: episode for episode in load_episodes(config["data_dir"])}
    descriptions = [episodes[i].description for i in train_ids]
    sampler = TrainingNegatives(train_ids, descriptions,
                                [load_action_labels(episodes[i]) for i in train_ids],
                                mix, args.k, args.fill, config["seed"])
    # Negatives copied from another description are stored by episode ID to keep the file small.
    source = {text: episode_id for episode_id, text in zip(train_ids, descriptions)}
    record, used = [], Counter()
    for epoch in range(args.epochs):
        negatives = sampler.epoch(epoch)
        record.append({episode_id: [{"type": t, "source": source[x]} if x in source
                                    else {"type": t, "text": x} for t, x in items]
                       for episode_id, items in zip(train_ids, negatives)})
        used.update(t for items in negatives for t, _ in items)
        print(f"Epoch {epoch + 1}/{args.epochs} sampled", flush=True)
    output = args.output or args.run_dir / "train_negatives.json"
    save_json(output, {"k": args.k, "mix": mix, "fill": args.fill, "seed": config["seed"],
                       "episodes": len(train_ids), "epochs": record})
    print(json.dumps({"types_used": dict(used), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
