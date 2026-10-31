"""Train small A-D heads on frozen SigLIP; fixed local proxies, never submission."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import math
import os
from pathlib import Path
import random
import statistics
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F
from comparison_protocol import metric_rows, paired_reward_interval, rank_prediction
from data_contract import load_contracts
from experiment_io import (atomic_json, digest, file_record, immutable_json, read_json,
                           sources_hash, verified_file)
from prepare_features import load_features
from prepare_team_validation import load_team_validation, read_action_labels
from training_schedules import TrainingSchedules
from temporal_models import TemporalRanker, collate_frames, make_segments

KINDS = ("A", "B", "C", "D")
TRAINING_MODES = ("random", "team")
LABELS = {"A": "global_mean_MLP", "B": "uniform_bins_MLP",
          "C": "adjacent_similarity_merge_MLP", "D": "all_frames_transformer"}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def training_schedule(train_ids, token_keys, seed, epoch):
    """Shared epoch/seed candidate schedule; no validation IDs or annotations."""
    pool = list(train_ids)
    rng = random.Random(seed * 1_000_003 + epoch)
    order = list(pool)
    rng.shuffle(order)
    rows = []
    if len({token_keys[x] for x in pool}) < 5:
        raise ValueError("Need at least five distinct training text inputs")
    for positive in order:
        choices, used = [positive], {token_keys[positive]}
        # Rejection avoids allocating a 3336-item list for every training pair.
        attempts = 0
        while len(choices) < 5:
            candidate = rng.choice(pool)
            attempts += 1
            if token_keys[candidate] not in used:
                choices.append(candidate)
                used.add(token_keys[candidate])
            if attempts > 1000 and len(choices) < 5:
                remaining = [x for x in pool if token_keys[x] not in used]
                if not remaining:
                    raise ValueError("Insufficient noncolliding training negatives")
                candidate = rng.choice(remaining)
                choices.append(candidate)
                used.add(token_keys[candidate])
        rows.append((positive, choices))
    return rows


def prepared_frames(bank, kind, bins):
    if kind not in ("B", "C"):
        return None
    return {ep: make_segments(frames, kind, bins) for ep, frames in bank["frames"].items()}


def batch_tensors(episode_ids, choice_sources, bank, device, segments=None):
    if segments is None:
        frames, mask = collate_frames([bank["frames"][ep] for ep in episode_ids])
        position = None
    else:
        frames = torch.stack([segments[ep][0] for ep in episode_ids])
        mask = torch.stack([segments[ep][1] for ep in episode_ids])
        position = torch.stack([segments[ep][2] for ep in episode_ids]).to(device)
    text = torch.stack([torch.stack([bank["texts"][source] for source in choices])
                        for choices in choice_sources]).to(device)
    return frames.to(device), mask.to(device), text, position


def score_batch(model, batch, bank, device, segments):
    args = batch_tensors([r["episode_id"] for r in batch],
                         [[c["source_episode_id"] for c in r["choices"]] for r in batch],
                         bank, device, segments)
    if model is None:
        frames, mask, text, _ = args
        mean = (frames * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return (F.normalize(mean, dim=-1).unsqueeze(1) * text).sum(-1)
    return model(args[0], args[1], args[2], segment_positions=args[3])


def evaluate(model, candidates, bank, device, batch_size, segments=None):
    if model is not None:
        model.eval()
    predictions, loss_sum = [], 0.0
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for start_index in range(0, len(candidates), batch_size):
            batch = candidates[start_index:start_index + batch_size]
            scores = score_batch(model, batch, bank, device, segments)
            targets = torch.tensor([[c["id"] for c in r["choices"]].index(r["correct_choice_id"])
                                    for r in batch], device=device)
            loss_sum += float(F.cross_entropy(scores, targets, reduction="sum"))
            for row, values in zip(batch, scores.cpu().tolist()):
                predictions.append(rank_prediction(row["episode_id"], row["choices"], values, row["correct_choice_id"]))
    synchronize(device)
    metrics = metric_rows(predictions)
    metrics.update({"cross_entropy": loss_sum / len(candidates),
                    "head_evaluation_seconds": time.perf_counter() - start})
    return {"metrics": metrics, "predictions": predictions}


def disturbed_bank(bank, episode_ids, operation):
    frames = {}
    for ep in episode_ids:
        values = bank["frames"][ep]
        if operation == "reverse":
            order = list(reversed(range(len(values))))
        elif operation == "shuffle":
            order = list(range(len(values)))
            random.Random(int(digest({"episode": ep, "seed": 2030})[:16], 16)).shuffle(order)
        else:
            raise ValueError(operation)
        frames[ep] = values[order]
    return {**bank, "frames": frames}


def order_diagnostics(model, original_predictions, candidates, bank, device, batch_size, kind, bins):
    """No truth label or 'accuracy' is assigned to altered videos."""
    baseline = {p["episode_id"]: p for p in original_predictions}
    result = {}
    if model is not None:
        model.eval()
    for operation in ("reverse", "shuffle"):
        disturbed = disturbed_bank(bank, [r["episode_id"] for r in candidates], operation)
        segments = prepared_frames(disturbed, kind, bins)
        rows = []
        with torch.no_grad():
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start:start + batch_size]
                values = score_batch(model, batch, disturbed, device, segments).cpu().tolist()
                for source, scores in zip(batch, values):
                    ids = [c["id"] for c in source["choices"]]
                    ranking = sorted(ids, key=lambda c: (-scores[ids.index(c)], c))
                    before = baseline[source["episode_id"]]
                    difference = max(abs(scores[i] - before["scores"][c]) for i, c in enumerate(ids))
                    rows.append({"episode_id": source["episode_id"], "top1_changed": ranking[0] != before["ranking"][0],
                                 "ranking_changed": ranking != before["ranking"], "max_abs_score_change": difference,
                                 "ranking": ranking})
        result[operation] = {"episodes": len(rows),
                             "top1_changed_fraction": statistics.mean(r["top1_changed"] for r in rows),
                             "ranking_changed_fraction": statistics.mean(r["ranking_changed"] for r in rows),
                             "mean_max_abs_score_change": statistics.mean(r["max_abs_score_change"] for r in rows),
                             "note": "Sensitivity only; original truth may not apply to the changed video.",
                             "episodes_detail": rows}
    return result


def token_audit(candidates, bank):
    counts = bank["full_token_counts"]
    collisions = []
    total = truncated = 0
    for row in candidates:
        positive = next(c["source_episode_id"] for c in row["choices"] if c["id"] == row["correct_choice_id"])
        matching = []
        for choice in row["choices"]:
            source = choice["source_episode_id"]
            total += 1
            truncated += counts[source] > 64
            if choice["id"] != row["correct_choice_id"] and bank["token_keys"][source] == bank["token_keys"][positive]:
                matching.append(choice["id"])
        if matching:
            collisions.append({"episode_id": row["episode_id"], "colliding_negative_ids": matching})
    return {"candidate_occurrences": total, "truncated_occurrences": truncated,
            "truncation_rate": truncated / total, "positive_negative_collision_episodes": collisions,
            "note": "All models use identical truncate64 text. Tail-only differences may be invisible."}


def atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def initial_parameters_hash(model):
    value = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def train_head(kind, seed, config, bank, train_ids, validation, team, folder, device, run_hash, segments,
               *, training_mode, schedule_store):
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "complete.json"
    if marker.exists():
        done = read_json(marker)
        if done["run_hash"] != run_hash or done["training_mode"] != training_mode:
            raise ValueError("Completed head belongs to a different request")
        for name, metadata in done["artifacts"].items():
            verified_file(folder / name, metadata)
        cached = read_json(folder / "result.json")
        for used in cached["used_negative_schedules"]:
            _, checked = schedule_store.epoch(training_mode, seed, used["epoch"])
            if checked != used:
                raise ValueError("Saved negative schedule differs from the completed head's training record")
        print(f"Reuse completed {training_mode}/{kind} seed={seed}", flush=True)
        return cached
    # An interrupted small head is restarted reproducibly; completed heads are retained.
    seed_all(seed)
    model = TemporalRanker(kind, **config["model"]).to(device)
    initial_hash = initial_parameters_hash(model)
    optimizer_steps = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best, stale, history = None, 0, []
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        schedule, schedule_audit = schedule_store.epoch(training_mode, seed, epoch)
        loss_sum = 0.0
        for start in range(0, len(schedule), config["batch_size"]):
            batch = schedule[start:start + config["batch_size"]]
            frames, mask, text, position = batch_tensors([r[0] for r in batch], [r[1] for r in batch],
                                                        bank, device, segments)
            optimizer.zero_grad(set_to_none=True)
            scores = model(frames, mask, text, segment_positions=position)
            # Independent per-choice scorer has no candidate-position input.
            loss = F.cross_entropy(scores, torch.zeros(len(batch), device=device, dtype=torch.long))
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss in {kind}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * len(batch)
        val = evaluate(model, validation, bank, device, config["batch_size"], segments)
        metrics = val["metrics"]
        selection = (metrics["mean_rank_reward"], -metrics["cross_entropy"])
        history.append({"epoch": epoch + 1, "train_cross_entropy": loss_sum / len(schedule),
                        "random_validation": metrics, "training_schedule_hash": digest(schedule), "negative_schedule": schedule_audit})
        if best is None or selection > best:
            best, stale = selection, 0
            atomic_checkpoint(folder / "best.pt", {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                                   "kind": kind, "seed": seed, "epoch": epoch + 1,
                                                   "training_mode": training_mode, "initial_parameters_hash": initial_hash,
                                                   "model_config": config["model"], "run_hash": run_hash})
        else:
            stale += 1
        atomic_json(folder / "history.json", history)
        print(f"{training_mode}/{kind} seed={seed} epoch={epoch+1} trainCE={loss_sum/len(schedule):.4f} "
              f"random_reward={metrics['mean_rank_reward']:.6f} top1={metrics['top1_accuracy']:.4f}", flush=True)
        if stale >= config["patience"]:
            break
    synchronize(device)
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    random_eval = evaluate(model, validation, bank, device, config["batch_size"], segments)
    team_eval = evaluate(model, team, bank, device, config["batch_size"], segments)
    diagnostics = order_diagnostics(model, random_eval["predictions"], validation, bank, device,
                                    config["batch_size"], kind, config["model"]["bins"])
    result = {"run_hash": run_hash, "kind": kind, "seed": seed, "best_epoch": checkpoint["epoch"],
              "training_mode": training_mode, "initial_parameters_hash": initial_hash,
              "optimizer_steps": optimizer_steps,
              "used_negative_schedules": [r["negative_schedule"] for r in history],
              "validation_hashes": {"random": config["random_candidates_hash"], "team": config["team_candidates_hash"]},
              "epochs_run": len(history), "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "training_and_checkpoint_validation_seconds": training_seconds,
              "random": random_eval, "team": team_eval,
              "diagnostics_summary": {op: {k: v for k, v in data.items() if k != "episodes_detail"}
                                      for op, data in diagnostics.items()}}
    atomic_json(folder / "order_diagnostics.json", diagnostics)
    atomic_json(folder / "result.json", result)
    atomic_json(marker, {"run_hash": run_hash, "training_mode": training_mode, "artifacts": {name: file_record(folder / name) for name in
                         ("result.json", "order_diagnostics.json", "best.pt", "history.json")}})
    return result


def distribution(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values}


def episode_reward_means(runs, split_name):
    mapped = [{r["episode_id"]: r["rank_reward"] for r in run[split_name]["predictions"]} for run in runs]
    if any(set(rows) != set(mapped[0]) for rows in mapped):
        raise ValueError("Runs do not evaluate identical episodes")
    return {ep: statistics.mean(rows[ep] for rows in mapped) for ep in mapped[0]}



def comparison(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, split, random_candidates = load_contracts(args.contract_dir)
    bank = load_features(args.feature_dir)
    if bank["index"]["source_fingerprint"] != manifest["source"]["fingerprint"]:
        raise ValueError("Features and contracts refer to different data")
    if set(bank["frames"]) != set(split["train_ids"] + split["val_ids"]):
        raise ValueError("Full train and validation features are required")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    team_candidates, team_provenance = load_team_validation(args.team_validation_dir, bank, manifest, split)
    immutable_json(output / "team_validation_candidates.json", team_candidates)
    train_ids, validation, team = split["train_ids"], random_candidates["episodes"], team_candidates["episodes"]
    if args.smoke:
        train_ids, validation, team = train_ids[:32], validation[:8], team[:8]
    original = {r["episode_id"]: r for r in manifest["episodes"]}
    labels = read_action_labels(args.train_zip, manifest, train_ids)
    schedules = TrainingSchedules(
        train_ids, {ep: original[ep]["description"] for ep in train_ids},
        {ep: labels[ep] for ep in train_ids}, bank["token_keys"], output / "negative_schedules",
        mix_text=args.train_mix, fill=args.train_fill, random_schedule_fn=training_schedule)
    config = {"seeds": args.seeds, "epochs": args.epochs, "patience": args.patience,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": 0.01,
              "model": {"feature_dim": bank["index"]["feature_dim"], "projection_dim": 128,
                        "hidden_dim": 128, "bins": 10, "dropout": 0.1},
              "loss": "five-choice cross entropy; positive plus four negatives within training split",
              "training_modes": list(TRAINING_MODES), "training_schedules": schedules.metadata,
              "schedule_fingerprint": schedules.fingerprint,
              "selection": "max random validation rank reward; then min CE; then earlier epoch",
              "smoke": args.smoke, "device": device, "train_ids_hash": digest(train_ids),
              "random_candidates_hash": digest(random_candidates), "team_candidates_hash": digest(team_candidates),
              "features": bank["index"]["fingerprint"], "sources": sources_hash(),
              "team_validation": team_provenance,
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__}}
    immutable_json(output / "run_config.json", config)
    run_hash, attempt = digest(config), uuid.uuid4().hex
    context = {"run_hash": run_hash, "attempt_id": attempt}
    atomic_json(output / "status.json", {**context, "status": "running"})
    start = time.perf_counter()
    try:
        atomic_json(output / "tokenization_audit.json", {"random": token_audit(validation, bank),
                                                        "team": token_audit(team, bank)})
        reference = {"random": evaluate(None, validation, bank, device, args.batch_size),
                     "team": evaluate(None, team, bank, device, args.batch_size)}
        atomic_json(output / "zero_shot_reference.json", reference)
        results = {mode: {} for mode in TRAINING_MODES}
        for kind in KINDS:
            print(f"Prepare {kind}: {LABELS[kind]}", flush=True)
            segments = prepared_frames(bank, kind, config["model"]["bins"])
            for mode in TRAINING_MODES:
                results[mode][kind] = []
                for seed in args.seeds:
                    result = train_head(kind, seed, config, bank, train_ids, validation, team,
                                        output / "heads" / mode / f"{kind}-seed{seed}", device, run_hash, segments,
                                        training_mode=mode, schedule_store=schedules)
                    results[mode][kind].append(result)
            for random_run, team_run in zip(results["random"][kind], results["team"][kind]):
                if random_run["initial_parameters_hash"] != team_run["initial_parameters_hash"]:
                    raise ValueError("Paired training conditions did not start from identical parameters")
                if random_run["validation_hashes"] != team_run["validation_hashes"]:
                    raise ValueError("Paired training conditions used different validation choices")
        summary = {mode: {} for mode in TRAINING_MODES}
        for mode in TRAINING_MODES:
            for kind, runs in results[mode].items():
                data = {"name": LABELS[kind], "training_mode": mode, "seeds": args.seeds,
                        "trainable_parameters": runs[0]["trainable_parameters"],
                        "best_epochs": [r["best_epoch"] for r in runs],
                        "epochs_run": [r["epochs_run"] for r in runs],
                        "optimizer_steps": [r["optimizer_steps"] for r in runs],
                        "total_training_seconds": sum(r["training_and_checkpoint_validation_seconds"] for r in runs)}
                for label in ("random", "team"):
                    data[label] = {metric: distribution([r[label]["metrics"][metric] for r in runs])
                                   for metric in ("mean_rank_reward", "top1_accuracy", "head_evaluation_seconds")}
                summary[mode][kind] = data
        comparisons, training_effects = {}, {}
        for label in ("random", "team"):
            comparisons[label], training_effects[label] = {}, {}
            for mode in TRAINING_MODES:
                comparisons[label][mode] = {}
                for base, candidate in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C")):
                    comparisons[label][mode][f"{candidate}_minus_{base}"] = paired_reward_interval(
                        episode_reward_means(results[mode][base], label),
                        episode_reward_means(results[mode][candidate], label))
            for kind in KINDS:
                training_effects[label][kind] = paired_reward_interval(
                    episode_reward_means(results["random"][kind], label),
                    episode_reward_means(results["team"][kind], label))
        report = {**context, "result_status": "complete", "smoke_subset": args.smoke,
                  "train_episodes": len(train_ids), "validation_episodes": len(validation),
                  "metric": "mean reward [1,.5,.25,.125,0]; random primary; team validation diagnostic",
                  "zero_shot": {k: v["metrics"] for k, v in reference.items()},
                  "team_validation": team_provenance, "training_schedules": schedules.metadata,
                  "trained_on": {"random": "four random original training captions",
                                 "team": {"mix": args.train_mix, "fill": args.train_fill, "pool": "training_only"}},
                  "models": summary, "paired_comparisons": comparisons, "training_effects": training_effects,
                  "completed_fits": sum(len(runs) for mode in results.values() for runs in mode.values()),
                  "feature_preparation": bank["index"]["timing"], "elapsed_seconds": time.perf_counter() - start,
                  "limitations": ["Local development proxies, not leaderboard prediction.",
                                  "Checkpoints use the same 834 development episodes, so there is model-selection bias.",
                                  "Team negatives may be semantically valid; training/validation type families overlap.",
                                  "Token-input collisions are retained and audited for exact teammate sampler fidelity.",
                                  "Shared maximum epochs/patience; early stopping can give different optimizer-step counts.",
                                  "Bootstrap covers episode sampling conditional on selected models, not seed or selection uncertainty.",
                                  "Order disturbance measures sensitivity, not correct temporal reasoning.",
                                  "Head timings exclude frozen encoding and cannot prove server inference limits."]}
        atomic_json(output / "comparison.json", report)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["training_mode", "model", "random_reward_mean", "random_reward_std", "team_reward_mean",
                         "team_reward_std", "random_top1_mean", "team_top1_mean", "parameters"])
        writer.writerow(["none", "SigLIP_zero_shot", reference["random"]["metrics"]["mean_rank_reward"], 0,
                         reference["team"]["metrics"]["mean_rank_reward"], 0,
                         reference["random"]["metrics"]["top1_accuracy"],
                         reference["team"]["metrics"]["top1_accuracy"], 0])
        for mode, models in summary.items():
            for kind, data in models.items():
                writer.writerow([mode, kind, data["random"]["mean_rank_reward"]["mean"],
                                 data["random"]["mean_rank_reward"]["std"], data["team"]["mean_rank_reward"]["mean"],
                                 data["team"]["mean_rank_reward"]["std"], data["random"]["top1_accuracy"]["mean"],
                                 data["team"]["top1_accuracy"]["mean"], data["trainable_parameters"]])
        (output / "comparison.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
        atomic_json(output / "status.json", {**context, "status": "complete",
                                            "comparison_file": file_record(output / "comparison.json")})
        print(buffer.getvalue(), flush=True)
        return report
    except BaseException as exc:
        atomic_json(output / "status.json", {**context, "status": "failed", "error": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--train-zip", required=True)
    p.add_argument("--train-mix", default="lexical=2,similar_episode=2")
    p.add_argument("--train-fill", default="random")
    p.add_argument("--team-validation-dir", required=True)
    p.add_argument("--contract-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("Training parameters must be positive")
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 or seed > 2**32-1 for seed in args.seeds):
        p.error("Seeds must be distinct integers in [0,2**32-1]")
    comparison(args)


if __name__ == "__main__":
    main()
