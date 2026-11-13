"""Train a ranking model on generated negatives, using cached frames and frozen encoders.

Every run selects its checkpoint with the same rule, so that runs stay comparable: the clean rank
score (the 5-candidate score on SELECTION_SETS, which have no detected text leak), averaged with
order pairing. These scores depend on our generated negatives and are not leaderboard scores. See
docs/experiments/validation.md.
"""

import argparse
import gzip
import json
import random
import statistics
import time
from pathlib import Path

import torch
from baseline_model import FrozenEncoders
from common import ROOT, choose_device, load_config, save_json, seed_everything
from dataset import load_episodes, load_images
from team.candidates.negatives import (
    NEGATIVE_TYPES,
    NegativeGenerator,
    load_action_labels,
    reversed_description,
)
from team.inference.ranker_model import MODELS
from team.training.train_negatives import check_mix, parse_mix
from team.training.train_negatives import sample_negatives as sample_mixed
from team.evaluation.rank_eval import rank_score, summarize
from torch import nn

SELECTION_SETS = ("lexical", "similar_episode")


@torch.inference_mode()
def encode_frames(encoders, episode):
    """One float16 CLS feature per frame, matching what CUDA inference computes."""
    images = load_images(episode)
    try:
        features = []
        for start in range(0, len(images), encoders.image_batch_size):
            inputs = encoders.image_processor(
                images=images[start:start + encoders.image_batch_size], return_tensors="pt")
            pixels = inputs["pixel_values"].to(encoders.device, dtype=encoders.dtype)
            features.append(encoders.image_model(pixel_values=pixels).last_hidden_state[:, 0].cpu())
    finally:
        for image in images:
            image.close()
    return torch.cat(features).half()


def load_frame_features(path, episodes, encoders):
    path = Path(path)
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        for episode in episodes:
            if len(cached[episode.episode_id]) != len(episode.frames):
                raise ValueError(f"Frame cache doesn't match the data: {episode.episode_id}")
        print(f"Loaded cached frames: {path}", flush=True)
        return cached
    cached = {}
    for index, episode in enumerate(episodes):
        cached[episode.episode_id] = encode_frames(encoders, episode)
        if (index + 1) % 200 == 0 or index + 1 == len(episodes):
            print(f"Encoded frames: {index + 1}/{len(episodes)} episodes", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, path)
    return cached


@torch.no_grad()  # Not inference_mode: training layers need to use these tensors in autograd.
def encode_tokens(encoders, texts):
    inputs = encoders.tokenizer(texts, padding=True, truncation=True,
                                max_length=encoders.max_text_length, return_tensors="pt").to(encoders.device)
    tokens = encoders.text_model(**inputs).last_hidden_state.float()
    return tokens, inputs["attention_mask"].bool()


def collate_frames(features, device):
    lengths = [len(item) for item in features]
    frames = torch.zeros(len(features), max(lengths), features[0].size(-1), device=device)
    mask = torch.zeros(len(features), max(lengths), dtype=torch.bool, device=device)
    for index, item in enumerate(features):
        frames[index, :len(item)] = item.to(device).float()
        mask[index, :len(item)] = True
    return frames, mask


def reverse_frames(features):
    return [item.flip(0) for item in features]


def sample_negatives(generator, index, types, count, rng):
    """Text -> type. Each type is drawn separately; descriptions short of options get random ones."""
    negatives = {}
    for _ in range(count * 5):
        negative_type = rng.choice(types)
        for text in generator.generate(index, negative_type, 1, rng):
            negatives.setdefault(text, negative_type)
        if len(negatives) >= count:
            break
    for text in generator.generate(index, "random", count, rng):
        if len(negatives) < count:
            negatives.setdefault(text, "random")
    return dict(list(negatives.items())[:count])


def draw_negatives(generator, index, types, count, rng, mix=None, fill="random"):
    """Text -> type. With a mix, the counts per type are fixed (team.training.train_negatives); otherwise types are drawn."""
    if mix:
        return {text: kind for kind, text in sample_mixed(generator, index, mix, count, fill, rng)}
    return sample_negatives(generator, index, types, count, rng)


@torch.inference_mode()
def score_sets(model, encoders, frames_by_episode, candidates, name, device, batch_size=32,
               wrong_video=False):
    """The 5-candidate score of one set. With wrong_video, any preference comes from text alone."""
    model.eval()
    entries = [entry for entry in candidates if name in entry["sets"]]
    scores = []
    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        video_entries = ([entries[(start + i + 1) % len(entries)] for i in range(len(batch))]
                         if wrong_video else batch)
        frames, frame_mask = collate_frames(
            [frames_by_episode[entry["episode_id"]] for entry in video_entries], device)
        sets = [entry["sets"][name] for entry in batch]
        tokens, token_mask = encode_tokens(
            encoders, [choice["text"] for item in sets for choice in item["choices"]])
        logits = model.logits(frames, frame_mask, tokens, token_mask, 5).float().cpu()
        for item, row in zip(sets, logits):
            order = row.argsort(descending=True, stable=True).tolist()
            scores.append(rank_score([item["choices"][i]["id"] for i in order], item["correct"]))
    return summarize(scores)


@torch.inference_mode()
def order_check(model, encoders, frames_by_episode, episodes, device, count=200, seed=2026):
    model.eval()
    picked = random.Random(seed).sample(episodes, min(count, len(episodes)))
    differences = []
    for start in range(0, len(picked), 32):
        batch = picked[start:start + 32]
        features = [frames_by_episode[episode.episode_id] for episode in batch]
        tokens, token_mask = encode_tokens(encoders, [episode.description for episode in batch])
        forward = model.logits(*collate_frames(features, device), tokens, token_mask, 1)
        backward = model.logits(*collate_frames(reverse_frames(features), device), tokens, token_mask, 1)
        differences.extend((forward - backward).squeeze(-1).tolist())
    return {"episodes": len(differences),
            "mean_abs_logit_difference": round(statistics.fmean(abs(d) for d in differences), 6),
            "prefers_forward": round(sum(d > 0 for d in differences) / len(differences), 4)}


@torch.inference_mode()
def order_pairing(model, encoders, frames_by_episode, episodes, device, count=400, seed=2026,
                  batch_size=32):
    """Does reversing the frames change which text order the model prefers? Chance is exactly 0.5.

    A preference between the two texts that ignores the video appears on both sides of the
    difference and cancels, so text cues can't raise this score.
    """
    model.eval()
    usable = [e for e in episodes if reversed_description(e.description) != e.description]
    picked = random.Random(seed).sample(usable, min(count, len(usable)))
    wins = 0.0
    for start in range(0, len(picked), batch_size):
        batch = picked[start:start + batch_size]
        features = [frames_by_episode[episode.episode_id] for episode in batch]
        texts = [text for episode in batch
                 for text in (episode.description, reversed_description(episode.description))]
        tokens, token_mask = encode_tokens(encoders, texts)
        forward = model.logits(*collate_frames(features, device), tokens, token_mask, 2)
        backward = model.logits(*collate_frames(reverse_frames(features), device), tokens, token_mask, 2)
        gains = (forward[:, 0] - forward[:, 1]) - (backward[:, 0] - backward[:, 1])
        wins += (gains > 1e-4).sum().item() + 0.5 * (gains.abs() <= 1e-4).sum().item()
    return {"episodes": len(picked), "score": round(wins / len(picked), 4)}


def load_qwen_records(path, train_ids):
    """Training-split records from team.candidates.qwen_negatives with a positive and at least one negative."""
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["episode_id"] in train_ids and record["positive"] and record["negatives"]:
            records[record["episode_id"]] = record
    return records


def qwen_candidates(record, count, flip_probability, short_pool, rng, reverse_actions=False):
    """A short positive with its single-change negatives, topped up with other short positives.

    With a swapped-order negative, flipping works as for full descriptions: on reversed frames the
    swapped sentence is correct and the original wording becomes a negative. With reverse_actions,
    the reversed-action sentence can play that role too, since playing "folds" backwards looks like
    "unfolds". Each such text is then correct on one frame direction and wrong on the other, so
    wording alone can't tell them apart.
    """
    correct = record["positive"]
    negatives = {text: f"qwen_{kind}" for kind, text in record["negatives"].items()}
    kinds = ("swapped_order", "reversed_action") if reverse_actions else ("swapped_order",)
    targets = [record["negatives"][kind] for kind in kinds if record["negatives"].get(kind)]
    flipped = bool(targets) and rng.random() < flip_probability
    if flipped:
        target = targets[0] if len(targets) == 1 else rng.choice(targets)
        negatives = {record["positive"]: "qwen_flipped_original",
                     **{text: kind for text, kind in negatives.items() if text != target}}
        correct = target
    for _ in range(count * 20):
        if len(negatives) >= count:
            break
        text = rng.choice(short_pool)
        if text != correct and text not in negatives:
            negatives[text] = "qwen_other_episode"
    return correct, dict(list(negatives.items())[:count]), flipped


def video_contrast_loss(model, frames, frame_mask, tokens, token_mask, rows, limit):
    """Given a correct text, find its video among the batch's videos. Only the video can answer this."""
    count = min(limit, len(rows))
    rows = rows[:count]
    logits = model.logits(frames[:count], frame_mask[:count], tokens[rows].repeat(count, 1, 1),
                          token_mask[rows].repeat(count, 1), count)
    return nn.functional.cross_entropy(logits.T, torch.arange(count, device=logits.device))


def run_epoch(model, optimizer, encoders, frames_by_episode, generator, episodes, types, device,
              epoch, seed, batch_size, reverse_weight, flip_probability, count=4, mix=None,
              fill="random", record=None, qwen=None, qwen_probability=0.0,
              qwen_reverse_actions=False, video_contrast_weight=0.0, video_contrast_limit=12):
    """Train for one epoch. `record` receives the candidates used, for the organizers' inspection."""
    model.train()
    order = list(range(len(episodes)))
    random.Random(f"{seed}:order:{epoch}").shuffle(order)
    ranking_loss = nn.CrossEntropyLoss()
    order_loss = nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "correct_first": 0, "episodes": 0}
    short_pool = [item["positive"] for item in qwen.values()] if qwen else []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        features, texts, targets = [], [], []
        for i in indices:
            rng = random.Random(f"{seed}:negatives:{epoch}:{episodes[i].episode_id}")
            description = episodes[i].description
            frames = frames_by_episode[episodes[i].episode_id]
            qwen_record = qwen.get(episodes[i].episode_id) if qwen else None
            if qwen_record and rng.random() < qwen_probability:
                correct, negatives, flipped = qwen_candidates(qwen_record, count, flip_probability,
                                                              short_pool, rng, qwen_reverse_actions)
                if flipped:
                    frames = frames.flip(0)
                features.append(frames)
                choices = [correct, *negatives]
                rng.shuffle(choices)
                if record is not None:
                    record.write(json.dumps({"episode_id": episodes[i].episode_id, "flipped": flipped,
                                             "correct": correct, "negatives": negatives}) + "\n")
                texts.extend(choices)
                targets.append(choices.index(correct))
                continue
            correct = reversed_description(description)
            negatives = draw_negatives(generator, i, types, count, rng, mix, fill)
            if correct != description and rng.random() < flip_probability:
                # Reverse frames and text together, so the reordered wording is sometimes the
                # right answer. Only the video can then show which text order is correct.
                frames = frames.flip(0)
                others = {text: kind for text, kind in negatives.items() if text != correct}
                negatives = dict(list({description: "flipped_original", **others}.items())[:count])
                flipped = True
            else:
                correct = description
                flipped = False
            features.append(frames)
            choices = [correct, *negatives]
            rng.shuffle(choices)
            if record is not None:
                record.write(json.dumps({"episode_id": episodes[i].episode_id, "flipped": flipped,
                                         "correct": correct, "negatives": negatives}) + "\n")
            texts.extend(choices)
            targets.append(choices.index(correct))
        tokens, token_mask = encode_tokens(encoders, texts)
        frames, frame_mask = collate_frames(features, device)
        logits = model.logits(frames, frame_mask, tokens, token_mask, count + 1)
        targets = torch.tensor(targets, device=device)
        loss = ranking_loss(logits, targets)
        if video_contrast_weight > 0 and len(indices) > 1:
            rows = torch.arange(len(indices), device=device) * (count + 1) + targets
            loss = loss + video_contrast_weight * video_contrast_loss(
                model, frames, frame_mask, tokens, token_mask, rows, video_contrast_limit)
        if reverse_weight > 0:
            # The correct text should fit forward frames better than reversed frames.
            rows = torch.arange(len(indices), device=device) * (count + 1) + targets
            backward = model.logits(*collate_frames(reverse_frames(features), device),
                                    tokens[rows], token_mask[rows], 1)
            margin = logits.gather(1, targets.unsqueeze(1)) - backward
            loss = loss + reverse_weight * order_loss(margin, torch.ones_like(margin))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals["loss"] += loss.item() * len(indices)
        totals["correct_first"] += (logits.argmax(dim=1) == targets).sum().item()
        totals["episodes"] += len(indices)
    return {"loss": totals["loss"] / totals["episodes"],
            "correct_first": totals["correct_first"] / totals["episodes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--train-negatives",
                        help=f"Comma-separated types from {NEGATIVE_TYPES}, drawn at random for each "
                             "negative; the rest are held out. Use this or --mix.")
    parser.add_argument("--mix", default="",
                        help="Fixed counts per type instead, e.g. lexical=3,similar_episode=3,random=2. "
                             "Types used, including --fill, count as trained; the rest are held out.")
    parser.add_argument("--k", type=int, default=4,
                        help="Negatives per training episode; the ranking list holds 1 + k candidates.")
    parser.add_argument("--fill", default="random", help="With --mix, the type for the rest of the k.")
    parser.add_argument("--splits", type=Path, required=True, help="splits.json from an earlier run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, help="Defaults to frames/features.pt beside the "
                        "config's output directory.")
    parser.add_argument("--candidates", type=Path, help="Defaults to validation/candidates.json "
                        "beside the config's output directory.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--reverse-weight", type=float, default=0.1,
                        help="Weight of the forward-over-reversed frames loss; some actions look "
                             "the same reversed, so keep this small.")
    parser.add_argument("--no-cross-attention", action="store_true",
                        help="Temporal model only: average text instead of cross-attending.")
    parser.add_argument("--seed", type=int, help="Overrides the config's seed for repeat runs.")
    parser.add_argument("--flip-probability", type=float, default=0.0,
                        help="Share of episodes trained with frames and text reversed together.")
    parser.add_argument("--video-contrast-weight", type=float, default=0.0,
                        help="Weight of also finding each correct text's video among the batch's videos.")
    parser.add_argument("--save-negatives", action="store_true",
                        help="Save every epoch's candidates in the run's negatives/ folder.")
    parser.add_argument("--qwen-candidates", type=Path,
                        help="team.candidates.qwen_negatives output; its training-split records add short "
                             "positives with single-change negatives.")
    parser.add_argument("--qwen-probability", type=float, default=0.5,
                        help="Share of episodes with a usable record trained on it instead.")
    parser.add_argument("--qwen-reverse-actions", action="store_true",
                        help="When flipping a Qwen episode, the reversed-action sentence can be the "
                             "correct text for the reversed frames, as the swapped-order one is.")
    parser.add_argument("--max-train-episodes", type=int, help="Small subset for a smoke run.")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip training and rebuild report.json from the run's best.pt.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if bool(args.train_negatives) == bool(args.mix):
        parser.error("Give exactly one of --train-negatives and --mix.")
    mix = parse_mix(args.mix)
    if mix:
        check_mix(mix, args.k, args.fill)
        train_types = {name for name, count in mix.items() if count}
        if sum(mix.values()) < args.k:
            train_types.add(args.fill)
        train_types = sorted(train_types)
    else:
        train_types = sorted(args.train_negatives.split(","))
    if unknown := set(train_types) - set(NEGATIVE_TYPES):
        parser.error(f"Unknown negative types: {sorted(unknown)}")
    held_out = [name for name in NEGATIVE_TYPES if name not in train_types]
    if not held_out:
        parser.error("Hold at least one negative type out of training (docs/experiments/validation.md).")
    if (args.output_dir / "best.pt").exists() and not args.report_only:
        raise FileExistsError(f"Use a new --output-dir; a run exists: {args.output_dir}")
    config = load_config(args.config)
    candidate_seed = config["seed"]
    if args.seed is not None:
        config["seed"] = args.seed
    seed_everything(config["seed"], config["cpu_threads"])
    device = choose_device(args.device)
    workspace = Path(config["output_dir"]).parent
    episodes = {e.episode_id: e for e in load_episodes(config["data_dir"])}
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    if set(splits["train"]) & set(splits["valid"]):
        raise ValueError("Train and validation episodes overlap.")
    train = [episodes[i] for i in splits["train"]][:args.max_train_episodes]
    valid = [episodes[i] for i in splits["valid"]]
    candidates = json.loads((args.candidates or workspace / "validation/candidates.json")
                            .read_text(encoding="utf-8"))
    if [entry["episode_id"] for entry in candidates] != splits["valid"]:
        raise ValueError("Candidates use a different validation split.")
    encoders = FrozenEncoders(config["image_model_dir"], config["text_model_dir"], device,
                              config["image_batch_size"], config["text_batch_size"],
                              config["max_text_length"])
    frames_by_episode = load_frame_features(args.frame_cache or workspace / "frames/features.pt",
                                            list(episodes.values()), encoders)
    generator = NegativeGenerator([e.description for e in train],
                                  [load_action_labels(e) for e in train])
    qwen = (load_qwen_records(args.qwen_candidates, {e.episode_id for e in train})
            if args.qwen_candidates else None)
    model_config = {"hidden_dim": config["hidden_dim"], "dropout": config["dropout"]}
    if args.model == "temporal":
        model_config["cross_attention"] = not args.no_cross_attention
    model = MODELS[args.model](**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    reverse_weight = args.reverse_weight if args.model == "temporal" else 0.0
    run_info = {"model": args.model, "model_config": model_config, "trained_on": train_types,
                "negatives_per_episode": args.k, "mix": mix or None,
                "fill": args.fill if mix else None,
                "qwen_candidates": str(args.qwen_candidates) if qwen else None,
                "qwen_probability": args.qwen_probability if qwen else 0.0,
                "qwen_train_records": len(qwen) if qwen else 0,
                "qwen_reverse_actions": bool(qwen) and args.qwen_reverse_actions,
                "held_out": held_out, "reverse_weight": reverse_weight,
                "flip_probability": args.flip_probability, "save_negatives": args.save_negatives,
                "video_contrast_weight": args.video_contrast_weight,
                "train_episodes": len(train), "valid_episodes": len(valid),
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "seed": config["seed"], "candidate_seed": candidate_seed,
                "selection_sets": SELECTION_SETS, "epochs": args.epochs, "patience": args.patience,
                "note": "Scores use our generated negatives; they are not leaderboard scores."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        save_json(args.output_dir / "run_info.json", run_info)
        save_json(args.output_dir / "splits.json", splits)
    best, stale, history = -1.0, 0, []
    for epoch in range(0 if args.report_only else args.epochs):
        started = time.perf_counter()
        record = None
        if args.save_negatives:
            (args.output_dir / "negatives").mkdir(exist_ok=True)
            record = gzip.open(args.output_dir / f"negatives/epoch_{epoch + 1:03d}.jsonl.gz", "wt",
                               encoding="utf-8")
        try:
            train_metrics = run_epoch(model, optimizer, encoders, frames_by_episode, generator, train,
                                      train_types, device, epoch, config["seed"], args.batch_size,
                                      reverse_weight, args.flip_probability, args.k, mix, args.fill,
                                      record, qwen, args.qwen_probability if qwen else 0.0,
                                      args.qwen_reverse_actions, args.video_contrast_weight)
        finally:
            if record is not None:
                record.close()
        selected = {name: score_sets(model, encoders, frames_by_episode, candidates, name, device)
                    for name in SELECTION_SETS}
        pairing = order_pairing(model, encoders, frames_by_episode, valid, device)
        selection = (statistics.fmean(item["score"] for item in selected.values()) + pairing["score"]) / 2
        metrics = {"epoch": epoch + 1, "train": train_metrics, "selection_sets": selected,
                   "order_pairing": pairing,
                   "selection": round(selection, 4),
                   "seconds": round(time.perf_counter() - started, 1)}
        history.append(metrics)
        print(json.dumps({"epoch": epoch + 1, "loss": round(train_metrics["loss"], 4),
                          "selection": metrics["selection"], "seconds": metrics["seconds"],
                          "order_pairing": pairing["score"],
                          **{name: item["score"] for name, item in selected.items()}}), flush=True)
        save_json(args.output_dir / "history.json", history)
        if selection > best:
            best, stale = selection, 0
            torch.save({"format_version": 2, "model": args.model, "model_config": model_config,
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "config": config, "epoch": epoch + 1, "run_info": run_info},
                       args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.", flush=True)
                break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    names = sorted({name for entry in candidates for name in entry["sets"]})
    if args.report_only:
        run_info = checkpoint["run_info"]
    report = {**run_info, "best_epoch": checkpoint["epoch"],
              "order_pairing": order_pairing(model, encoders, frames_by_episode, valid, device),
              "scores": {name: score_sets(model, encoders, frames_by_episode, candidates, name, device)
                         for name in names},
              # Near 0.375 means the model needs the video; well above means it reads text cues.
              "wrong_video_scores": {name: score_sets(model, encoders, frames_by_episode, candidates,
                                                      name, device, wrong_video=True)
                                     for name in names},
              "order_check": order_check(model, encoders, frames_by_episode, valid, device)}
    save_json(args.output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in
                      ["best_epoch", "trained_on", "held_out", "order_pairing", "scores",
                       "wrong_video_scores", "order_check"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
