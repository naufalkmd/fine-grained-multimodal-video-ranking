"""Train a small event-to-frame order aligner on frozen SigLIP features (team/order_features.py).

Positives are the training descriptions, whose events happen in the frames' order. Negatives are
the same events in other orders, made by code, and, with the frames reversed, the original order
(the reversed order is then the positive). Every candidate in a group shares the same events, so
only order can separate them.

The aligner starts as the untrained order check (team/order_check.py): low-rank corrections to the
event and frame features start at zero, and so does the weight of a prior that places early
events in early frames. Its score is the best total when events take frames in order, minus the
total when each event takes its best frame anywhere, per event. Checked on validation episodes,
with LLM-written order changes (Qwen `changed_order`, a type never trained on) as the main measure.

    python -m team.train_order --features cuda/outputs/order/features.pt \
        --splits cuda/outputs/baseline/splits.json --output cuda/outputs/order/aligner.pt
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from common import ROOT, load_config, save_json
from dataset import load_episodes
from team.order_check import events
from team.rank_eval import RANK_SCORES


class OrderAligner(torch.nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.event_down = torch.nn.Linear(dim, rank, bias=False)
        self.event_up = torch.nn.Linear(rank, dim, bias=False)
        self.frame_down = torch.nn.Linear(dim, rank, bias=False)
        self.frame_up = torch.nn.Linear(rank, dim, bias=False)
        torch.nn.init.zeros_(self.event_up.weight)
        torch.nn.init.zeros_(self.frame_up.weight)
        self.prior_weight = torch.nn.Parameter(torch.zeros(()))
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(50.0)))

    def similarity(self, event_features, frame_features):
        e = F.normalize(event_features + self.event_up(self.event_down(event_features)), dim=-1)
        f = F.normalize(frame_features + self.frame_up(self.frame_down(frame_features)), dim=-1)
        return e @ f.T

    def fit(self, similarity, orders):
        """Order fit of each event order (rows of `orders`) against the frames: (count,)."""
        rows = similarity[orders]                                   # (count, K, T)
        count, events_, frames = rows.shape
        if events_ > 1 and frames > 1:
            k = torch.arange(events_, device=rows.device).float() / (events_ - 1)
            t = torch.arange(frames, device=rows.device).float() / (frames - 1)
            rows = rows - self.prior_weight * (k[:, None] - t[None, :]).abs()
        free = rows.max(dim=2).values.sum(dim=1)
        best = rows[:, 0]
        for step in range(1, events_):
            best = rows[:, step] + torch.cummax(best, dim=1).values
        return (best.max(dim=1).values - free) / events_


def lookup(features, texts):
    return torch.stack([features["event_features"][features["index"][p]] for p in texts]).float()


def permutations(count, how_many, rng, keep_first=True, swap_two=True):
    """The original order first, then up to `how_many` other orders of the same events. By default
    the opening event stays in place and two later events trade places, as in the order changes
    seen locally; shuffling everything lets the opening summary sentence give the order away."""
    original = tuple(range(count))
    seen, result = {original}, [list(original)]
    movable = list(range(1 if keep_first else 0, count))
    if len(movable) < 2:
        return result
    for _ in range(50):
        if len(result) > how_many:
            break
        order = list(original)
        if swap_two:
            a, b = rng.sample(movable, 2)
            order[a], order[b] = order[b], order[a]
        else:
            tail = [order[i] for i in movable]
            rng.shuffle(tail)
            for i, value in zip(movable, tail):
                order[i] = value
        if tuple(order) not in seen:
            seen.add(tuple(order))
            result.append(order)
    return result


def candidate_fit(model, features, frames, texts, device):
    """Order fit for candidates with different events (validation sets)."""
    fits = []
    for text in texts:
        parts = events(text)
        sim = model.similarity(lookup(features, parts).to(device), frames)
        fits.append(model.fit(sim, torch.arange(len(parts), device=device)[None])[0].item())
    return fits


@torch.no_grad()
def evaluate(model, features, valid_ids, candidates, generated, device):
    model.eval()
    results = {}
    reorder = [e for e in candidates if e["episode_id"] in valid_ids and "reorder" in e["sets"]]
    for video in ("right", "wrong", "reversed"):
        scores = []
        for position, entry in enumerate(reorder):
            candidate = entry["sets"]["reorder"]
            episode = entry["episode_id"] if video != "wrong" else \
                reorder[(position + 1) % len(reorder)]["episode_id"]
            frames = features["frames"][episode].float().to(device)
            frames = frames.flip(0) if video == "reversed" else frames
            fits = candidate_fit(model, features, frames, [c["text"] for c in candidate["choices"]],
                                 device)
            correct = [c["id"] for c in candidate["choices"]].index(candidate["correct"])
            order = sorted(range(len(fits)), key=lambda i: -fits[i])
            scores.append(RANK_SCORES[order.index(correct)])
        results[f"reorder_{video}"] = round(statistics.fmean(scores), 4)
    pairs = [r for r in generated if r["episode_id"] in valid_ids and "changed_order" in r["negatives"]]
    for video in ("right", "wrong", "reversed"):
        wins = []
        for position, record in enumerate(pairs):
            episode = record["episode_id"] if video != "wrong" else \
                pairs[(position + 1) % len(pairs)]["episode_id"]
            frames = features["frames"][episode].float().to(device)
            frames = frames.flip(0) if video == "reversed" else frames
            a, b = candidate_fit(model, features, frames,
                                 [record["description"], record["negatives"]["changed_order"]], device)
            wins.append(1.0 if a > b else 0.5 if a == b else 0.0)
        results[f"llm_order_{video}"] = round(statistics.fmean(wins), 4)
    results["reorder_episodes"], results["llm_order_pairs"] = len(reorder), len(pairs)
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "cuda/outputs/validation/candidates_qwen.json")
    parser.add_argument("--generated", type=Path,
                        default=ROOT / "cuda/outputs/qwen/long_rewrite.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--move-first", action="store_true",
                        help="Let the opening event move too (it gives the order away locally).")
    parser.add_argument("--shuffle-all", action="store_true",
                        help="Shuffle the movable events instead of swapping two.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.load(args.features, weights_only=False)
    features["index"] = {text: i for i, text in enumerate(features["events"])}
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    descriptions = {e.episode_id: e.description
                    for e in load_episodes(load_config(ROOT / "cuda/config.json")["data_dir"])}
    train = [i for i in splits["train"] if len(events(descriptions[i])) >= 3]
    valid_ids = set(splits["valid"])
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    generated = [json.loads(line) for line in args.generated.read_text(encoding="utf-8").splitlines()]

    model = OrderAligner(features["event_features"].shape[1], args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = [{"epoch": 0, **evaluate(model, features, valid_ids, candidates, generated, device)}]
    print(json.dumps(history[-1]), flush=True)
    best = (history[-1]["llm_order_right"], 0)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        losses, correct_first = [], []
        for start in range(0, len(train), args.batch_size):
            loss = 0.0
            for episode_id in train[start:start + args.batch_size]:
                parts = events(descriptions[episode_id])
                frames = features["frames"][episode_id].float().to(device)
                event_features = lookup(features, parts).to(device)
                orders = permutations(len(parts), args.negatives, rng, not args.move_first,
                                      not args.shuffle_all)
                if len(orders) < 2:
                    continue
                for reverse in (False, True):
                    seen_frames = frames.flip(0) if reverse else frames
                    group = [list(reversed(o)) for o in orders] if reverse else orders
                    sim = model.similarity(event_features, seen_frames)
                    fit = model.fit(sim, torch.tensor(group, device=device))
                    logits = fit * model.log_scale.exp()
                    loss = loss + F.cross_entropy(logits[None], torch.zeros(1, dtype=torch.long,
                                                                              device=device))
                    correct_first.append(float(logits.argmax().item() == 0))
            loss = loss / (2 * len(train[start:start + args.batch_size]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, features, valid_ids, candidates, generated, device)
        history.append({"epoch": epoch, "train_loss": round(statistics.fmean(losses), 4),
                        "train_correct_first": round(statistics.fmean(correct_first), 4),
                        "prior_weight": round(model.prior_weight.item(), 4), **metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics["llm_order_right"] > best[0]:
            best = (metrics["llm_order_right"], epoch)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "dim": features["event_features"].shape[1],
                        "rank": args.rank}, args.output)
    save_json(args.output.with_suffix(".json"), {
        **{key: str(value) for key, value in vars(args).items()}, "history": history,
        "best_epoch": best[1], "train_episodes": len(train),
        "trained_on": "training descriptions and code-made reorderings of their events"})
    print(f"Best epoch {best[1]} (LLM order pairs, right video {best[0]}); saved {args.output}")


if __name__ == "__main__":
    main()
