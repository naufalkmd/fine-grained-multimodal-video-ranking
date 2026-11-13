"""Frozen SigLIP residual comparison; local proxies, never server submissions.

Only official positive indices supervise CE. The frozen cosine is an additive
model path, not a teacher target. Original captions supply every negative.
"""
from __future__ import annotations

import argparse
import csv
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
from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, sources_hash, verified_file
from prepare_features import load_features
from prepare_team_validation import load_team_validation, read_action_labels
from run_comparison import (atomic_checkpoint, distribution, episode_reward_means,
                            initial_parameters_hash, seed_all, synchronize, training_schedule)
from sentence_features import load_sentence_features
from sentence_models import FrozenSiglipResidual
from temporal_models import collate_frames
from training_schedules import TrainingSchedules

KINDS = ("global", "sentence")
TRAINING_MODES = ("random", "mixed")
MIX = "random=2,lexical=1,similar_episode=1"
SELECTION = "max team reward; then max random reward; then min team CE; then earlier epoch (epoch 0 eligible)"


def batch_tensors(episode_ids, choice_sources, bank, sentence_bank, device):
    """Keep every frame and every original sentence/window; pad within batch."""
    frames, frame_mask = collate_frames([bank["frames"][ep] for ep in episode_ids])
    text = torch.stack([torch.stack([bank["texts"][ep] for ep in row]) for row in choice_sources])
    maximum = max(len(sentence_bank["features"][ep]) for row in choice_sources for ep in row)
    sentences = text.new_zeros((len(episode_ids), len(choice_sources[0]), maximum, text.shape[-1]))
    sentence_mask = torch.zeros(sentences.shape[:-1], dtype=torch.bool)
    positions = text.new_zeros(sentences.shape[:-1])
    weights = text.new_zeros(sentences.shape[:-1])
    for b, choices in enumerate(choice_sources):
        if len(choices) != 5:
            raise ValueError("Every comparison must have five choices")
        for c, ep in enumerate(choices):
            count = len(sentence_bank["features"][ep])
            sentences[b, c, :count] = sentence_bank["features"][ep]
            sentence_mask[b, c, :count] = True
            positions[b, c, :count] = sentence_bank["positions"][ep]
            weights[b, c, :count] = sentence_bank["weights"][ep]
    return tuple(value.to(device) for value in (frames, frame_mask, text, sentences, sentence_mask, positions, weights))


def score_components(model, batch, bank, sentence_bank, device):
    tensors = batch_tensors([r["episode_id"] for r in batch],
                            [[c["source_episode_id"] for c in r["choices"]] for r in batch],
                            bank, sentence_bank, device)
    if model is not None:
        return model.components(*tensors)
    frames, mask, text = tensors[:3]
    mean = (frames * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    base = (F.normalize(mean, dim=-1).unsqueeze(1) * text).sum(-1)
    return {"base": base, "scores": base, "correction": torch.zeros_like(base)}


def score_statistics(components, targets, temperature):
    scores, correction = components["scores"].detach(), components["correction"].detach()
    positive = scores.gather(1, targets[:, None]).squeeze(1)
    negatives = scores.clone().scatter_(1, targets[:, None], -torch.inf)
    return {
        "positive_margin": float((positive - negatives.max(1).values).mean()),
        "max_softmax": float((scores / temperature).softmax(-1).max(-1).values.mean()),
        "score_rms": float(scores.square().mean().sqrt()),
        "correction_rms": float(correction.square().mean().sqrt()),
        "correction_max_abs": float(correction.abs().max()),
    }


def evaluate(model, candidates, bank, sentence_bank, device, batch_size, temperature):
    if model is not None:
        model.eval()
    predictions, loss_sum, statistics_sum = [], 0.0, {}
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(candidates), batch_size):
            rows = candidates[offset:offset + batch_size]
            components = score_components(model, rows, bank, sentence_bank, device)
            scores = components["scores"]
            targets = torch.tensor([[c["id"] for c in r["choices"]].index(r["correct_choice_id"])
                                    for r in rows], device=device)
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite validation scores")
            loss_sum += float(F.cross_entropy(scores / temperature, targets, reduction="sum"))
            for key, value in score_statistics(components, targets, temperature).items():
                if key.endswith("max_abs"):
                    statistics_sum[key] = max(statistics_sum.get(key, 0), value)
                else:
                    statistics_sum[key] = statistics_sum.get(key, 0) + value * len(rows)
            predictions.extend(rank_prediction(r["episode_id"], r["choices"], values, r["correct_choice_id"])
                               for r, values in zip(rows, scores.cpu().tolist()))
    synchronize(device)
    metrics = metric_rows(predictions)
    metrics.update({"cross_entropy": loss_sum / len(candidates), "loss_temperature": temperature,
                    "head_evaluation_seconds": time.perf_counter() - start,
                    **{key: value if key.endswith("max_abs") else value / len(candidates)
                       for key, value in statistics_sum.items()}})
    return {"metrics": metrics, "predictions": predictions}


def selection_key(evaluations):
    return (evaluations["team"]["metrics"]["mean_rank_reward"],
            evaluations["random"]["metrics"]["mean_rank_reward"],
            -evaluations["team"]["metrics"]["cross_entropy"])


def zero_gate(evaluations, reference):
    """Check every score in both sets, not just aggregate accuracy."""
    maxima = {}
    for label in ("random", "team"):
        baseline = {r["episode_id"]: r for r in reference[label]["predictions"]}
        current = evaluations[label]["predictions"]
        if set(baseline) != {r["episode_id"] for r in current}:
            raise ValueError("Initial baseline gate episode coverage differs")
        maxima[label] = max(abs(value - baseline[row["episode_id"]]["scores"][choice])
                            for row in current for choice, value in row["scores"].items())
        if maxima[label] > 1e-6:
            raise ValueError("Epoch-0 residual does not preserve the frozen SigLIP scores")
    return {"passed": True, "tolerance": 1e-6, "max_abs_score_difference": maxima}


def input_diagnostics(model, predictions, candidates, bank, sentence_bank, device, batch_size):
    """Perturbation sensitivity only: never label altered videos as correct/incorrect."""
    model.eval()
    before = {r["episode_id"]: r for r in predictions}
    episode_ids = sorted(before)
    swapped = {ep: episode_ids[(i + 1) % len(episode_ids)] for i, ep in enumerate(episode_ids)}
    result = {}
    for operation in ("reverse", "shuffle", "swap_video"):
        frames = {}
        for ep in episode_ids:
            values = bank["frames"][ep]
            order = list(range(len(values)))
            if operation == "reverse":
                order.reverse()
            elif operation == "shuffle":
                random.Random(int(digest({"episode": ep, "seed": 2030})[:16], 16)).shuffle(order)
            frames[ep] = bank["frames"][swapped[ep]] if operation == "swap_video" else values[order]
        changed_bank = {**bank, "frames": frames}
        details = []
        with torch.no_grad():
            for offset in range(0, len(candidates), batch_size):
                batch = candidates[offset:offset + batch_size]
                scores = score_components(model, batch, changed_bank, sentence_bank, device)["scores"].cpu().tolist()
                for row, values in zip(batch, scores):
                    ids = [c["id"] for c in row["choices"]]
                    ranking = sorted(ids, key=lambda c: (-values[ids.index(c)], c))
                    previous = before[row["episode_id"]]
                    details.append({"episode_id": row["episode_id"],
                                    "top1_changed": ranking[0] != previous["ranking"][0],
                                    "ranking_changed": ranking != previous["ranking"],
                                    "max_abs_score_change": max(abs(values[i] - previous["scores"][c]) for i, c in enumerate(ids)),
                                    **({"video_source": swapped[row["episode_id"]]} if operation == "swap_video" else {})})
        result[operation] = {"episodes": len(details),
                             "top1_changed_fraction": statistics.mean(r["top1_changed"] for r in details),
                             "ranking_changed_fraction": statistics.mean(r["ranking_changed"] for r in details),
                             "mean_max_abs_score_change": statistics.mean(r["max_abs_score_change"] for r in details),
                             "note": "Input sensitivity only; no accuracy or truth label is assigned to altered videos.",
                             "episodes_detail": details}
    return result


def train_head(kind, seed, config, bank, sentence_bank, validation_sets, reference, folder, device,
               run_hash, *, training_mode, schedule_store):
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "complete.json"
    schedule_mode = "team" if training_mode == "mixed" else "random"
    if marker.exists():
        done = read_json(marker)
        expected_identity = {"run_hash": run_hash, "training_mode": training_mode, "kind": kind, "seed": seed}
        if any(done.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("Completed head belongs to a different request")
        for name, metadata in done["artifacts"].items():
            verified_file(folder / name, metadata)
        cached = read_json(folder / "result.json")
        if any(cached.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("Completed head result belongs to a different request")
        for used in cached["used_negative_schedules"]:
            _, checked = schedule_store.epoch(schedule_mode, seed, used["epoch"])
            if checked != used:
                raise ValueError("Saved negative schedule differs from the completed head record")
        print(f"Reuse completed {training_mode}/{kind} seed={seed}", flush=True)
        return cached

    # Interrupted small heads restart from identical seed; completed heads reuse.
    seed_all(seed)
    model = FrozenSiglipResidual(kind, **config["model"]).to(device)
    initial_hash = initial_parameters_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    temperature, batch_size = config["loss_temperature"], config["batch_size"]
    optimizer_steps, stale = 0, 0
    used_schedules, history = [], []
    started = time.perf_counter()

    def all_evaluations():
        return {label: evaluate(model, rows, bank, sentence_bank, device, batch_size, temperature)
                for label, rows in validation_sets.items()}

    def checkpoint(epoch, path):
        atomic_checkpoint(folder / path, {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                         "kind": kind, "seed": seed, "epoch": epoch, "training_mode": training_mode,
                                         "initial_parameters_hash": initial_hash, "model_config": config["model"], "run_hash": run_hash})

    evaluations = all_evaluations()
    gate = zero_gate(evaluations, reference)
    atomic_json(folder / "initial_debug.json", {"zero_gate": gate, "initial_parameters_hash": initial_hash,
                                               "metrics": {k: v["metrics"] for k, v in evaluations.items()}})
    best = selection_key(evaluations)
    random_best = (evaluations["random"]["metrics"]["mean_rank_reward"], -evaluations["random"]["metrics"]["cross_entropy"])
    checkpoint(0, "initial.pt")
    checkpoint(0, "best.pt")
    checkpoint(0, "random_best.pt")
    history.append({"epoch": 0, "train_cross_entropy": None, "gradient_norm": None,
                    "random_validation": evaluations["random"]["metrics"], "team_validation": evaluations["team"]["metrics"],
                    "selected_primary": True, "selected_random": True, "baseline_candidate": True})
    atomic_json(folder / "history.json", history)
    first_gradient = None
    for epoch_index in range(config["epochs"]):
        epoch = epoch_index + 1
        model.train()
        schedule, schedule_audit = schedule_store.epoch(schedule_mode, seed, epoch_index)
        used_schedules.append(schedule_audit)
        totals, total_items = {}, 0
        for offset in range(0, len(schedule), batch_size):
            batch = schedule[offset:offset + batch_size]
            tensors = batch_tensors([r[0] for r in batch], [r[1] for r in batch], bank, sentence_bank, device)
            optimizer.zero_grad(set_to_none=True)
            components = model.components(*tensors)
            targets = torch.zeros(len(batch), device=device, dtype=torch.long)
            # ONLY official positive choice index supervises training. No teacher target.
            loss = F.cross_entropy(components["scores"] / temperature, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss: {training_mode}/{kind}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient norm")
            if first_gradient is None:
                first_gradient = {name: float(p.grad.detach().norm()) if p.grad is not None else None
                                  for name, p in model.named_parameters()}
            optimizer.step()
            optimizer_steps += 1
            stats = {**score_statistics(components, targets, temperature),
                     "train_cross_entropy": float(loss.detach()), "gradient_norm": float(norm)}
            for key, value in stats.items():
                if key.endswith("max_abs"):
                    totals[key] = max(totals.get(key, 0), value)
                else:
                    totals[key] = totals.get(key, 0) + value * len(batch)
            total_items += len(batch)
        evaluations = all_evaluations()
        key = selection_key(evaluations)
        selected = key > best
        if selected:
            best, stale = key, 0
            checkpoint(epoch, "best.pt")
        elif epoch >= config["min_epochs"]:
            stale += 1
        random_key = (evaluations["random"]["metrics"]["mean_rank_reward"], -evaluations["random"]["metrics"]["cross_entropy"])
        selected_random = random_key > random_best
        if selected_random:
            random_best = random_key
            checkpoint(epoch, "random_best.pt")
        history.append({"epoch": epoch,
                        **{k: v if k.endswith("max_abs") else v / total_items for k, v in totals.items()},
                        "random_validation": evaluations["random"]["metrics"], "team_validation": evaluations["team"]["metrics"],
                        "selected_primary": selected, "selected_random": selected_random,
                        "training_schedule_hash": digest(schedule), "negative_schedule": schedule_audit,
                        "stale_epochs_after_minimum": stale})
        atomic_json(folder / "history.json", history)
        checkpoint(epoch, "last.pt")
        print(f"{training_mode}/{kind} seed={seed} epoch={epoch} CE={totals['train_cross_entropy']/total_items:.4f} "
              f"random={evaluations['random']['metrics']['mean_rank_reward']:.6f} "
              f"team={evaluations['team']['metrics']['mean_rank_reward']:.6f}", flush=True)
        if epoch >= config["min_epochs"] and stale >= config["patience"]:
            break
    synchronize(device)
    training_seconds = time.perf_counter() - started
    selected_checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected_checkpoint["state_dict"])
    evaluations = all_evaluations()
    diagnostics = input_diagnostics(model, evaluations["team"]["predictions"], validation_sets["team"],
                                    bank, sentence_bank, device, batch_size)
    initial_debug = read_json(folder / "initial_debug.json")
    initial_debug["first_batch_gradient_norms"] = first_gradient
    atomic_json(folder / "initial_debug.json", initial_debug)
    result = {"run_hash": run_hash, "kind": kind, "seed": seed, "training_mode": training_mode,
              "best_epoch": selected_checkpoint["epoch"], "baseline_fallback": selected_checkpoint["epoch"] == 0,
              "initial_parameters_hash": initial_hash, "zero_gate": gate, "optimizer_steps": optimizer_steps,
              "used_negative_schedules": used_schedules, "epochs_run": len(history) - 1,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training_and_checkpoint_validation_seconds": training_seconds,
              "validation_hashes": {"random": config["random_candidates_hash"], "team": config["team_candidates_hash"]},
              **evaluations,
              "diagnostics_summary": {op: {k: v for k, v in data.items() if k != "episodes_detail"}
                                      for op, data in diagnostics.items()}}
    atomic_json(folder / "input_diagnostics.json", diagnostics)
    atomic_json(folder / "result.json", result)
    artifacts = ("result.json", "input_diagnostics.json", "initial_debug.json", "history.json",
                 "initial.pt", "best.pt", "random_best.pt", "last.pt")
    atomic_json(marker, {"run_hash": run_hash, "training_mode": training_mode, "kind": kind, "seed": seed,
                         "artifacts": {name: file_record(folder / name) for name in artifacts}})
    return result


def comparison(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, split, random_candidates = load_contracts(args.contract_dir)
    bank = load_features(args.feature_dir)
    if bank["index"]["source_fingerprint"] != manifest["source"]["fingerprint"]:
        raise ValueError("Features and contracts refer to different data")
    expected = set(split["train_ids"] + split["val_ids"])
    if set(bank["frames"]) != expected:
        raise ValueError("Full train and validation frame features are required")
    sentence_bank = load_sentence_features(args.sentence_feature_dir, bank, manifest, split)
    if set(sentence_bank["features"]) != expected:
        raise ValueError("Full original-caption sentence features are required")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    team_candidates, team_provenance = load_team_validation(args.team_validation_dir, bank, manifest, split)
    if [r["episode_id"] for r in team_candidates["episodes"]] != [r["episode_id"] for r in random_candidates["episodes"]]:
        raise ValueError("The two validation sets must cover exactly the same videos in the same order")
    if any(c["source_episode_id"] not in expected for group in (random_candidates, team_candidates)
           for row in group["episodes"] for c in row["choices"]):
        raise ValueError("This comparison supports unedited original-caption negatives only")
    train_ids = split["train_ids"][:32] if args.smoke else split["train_ids"]
    validation_sets = {"random": random_candidates["episodes"], "team": team_candidates["episodes"]}
    if args.smoke:
        validation_sets = {k: v[:8] for k, v in validation_sets.items()}
    seeds = [2026] if args.smoke else args.seeds
    models = list(KINDS) if args.smoke else args.models
    original = {r["episode_id"]: r for r in manifest["episodes"]}
    labels = read_action_labels(args.train_zip, manifest, train_ids)
    schedules = TrainingSchedules(train_ids, {ep: original[ep]["description"] for ep in train_ids},
                                   {ep: labels[ep] for ep in train_ids}, bank["token_keys"], output / "negative_schedules",
                                   mix_text=args.train_mix, fill=args.train_fill, random_schedule_fn=training_schedule)
    root = Path(__file__).resolve().parent
    code_hashes = {**sources_hash(), **{p.relative_to(root).as_posix(): file_record(p)["sha256"]
                                     for p in sorted((root / "team").glob("*.py"))}}
    config = {"version": "siglip-sentence-residual-v1", "seeds": seeds, "models": models,
              "epochs": 1 if args.smoke else args.epochs, "min_epochs": 1 if args.smoke else args.min_epochs,
              "patience": 1 if args.smoke else args.patience, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "weight_decay": 0.01, "loss_temperature": args.loss_temperature,
              "model": {"feature_dim": bank["index"]["feature_dim"], "hidden_dim": args.hidden_dim,
                        "attention_temperature": args.attention_temperature, "residual_limit": args.residual_limit, "dropout": 0.1},
              "loss": "CE(final_scores / temperature, official_positive_index); no teacher targets or distillation",
              "training_modes": list(TRAINING_MODES), "training_schedules": schedules.metadata,
              "schedule_fingerprint": schedules.fingerprint, "selection": SELECTION,
              "early_stopping": "Count stale epochs only from min_epochs onward; improvements reset count",
              "smoke": args.smoke, "device": device, "train_ids_hash": digest(train_ids), "split_hash": digest(split),
              "random_candidates_hash": digest(random_candidates), "team_candidates_hash": digest(team_candidates),
              "evaluated_candidates_hash": {k: digest(v) for k, v in validation_sets.items()},
              "features": bank["index"]["fingerprint"], "sentence_features": sentence_bank["index"]["fingerprint"],
              "sources": code_hashes, "team_validation": team_provenance,
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
              "submission_limit_user_reported": 30}
    immutable_json(output / "run_config.json", config)
    immutable_json(output / "random_validation_candidates.json", random_candidates)
    immutable_json(output / "team_validation_candidates.json", team_candidates)
    run_hash, attempt = digest(config), uuid.uuid4().hex
    context = {"run_hash": run_hash, "attempt_id": attempt}
    if (output / "status.json").exists():
        previous_status = read_json(output / "status.json")
        if previous_status.get("run_hash") != run_hash:
            raise ValueError("Status belongs to a different request")
        if previous_status.get("status") == "complete":
            for name, metadata in previous_status["artifacts"].items():
                verified_file(output / name, metadata)
    atomic_json(output / "status.json", {**context, "status": "running"})
    start = time.perf_counter()
    try:
        reference = {label: evaluate(None, rows, bank, sentence_bank, device, args.batch_size, args.loss_temperature)
                     for label, rows in validation_sets.items()}
        atomic_json(output / "zero_shot_reference.json", reference)
        results = {mode: {} for mode in TRAINING_MODES}
        for kind in models:
            for mode in TRAINING_MODES:
                results[mode][kind] = [train_head(kind, seed, config, bank, sentence_bank, validation_sets,
                                                 reference, output / "heads" / mode / f"{kind}-seed{seed}",
                                                 device, run_hash, training_mode=mode, schedule_store=schedules)
                                       for seed in seeds]
            for a, b in zip(results["random"][kind], results["mixed"][kind]):
                if a["initial_parameters_hash"] != b["initial_parameters_hash"] or a["validation_hashes"] != b["validation_hashes"]:
                    raise ValueError("Paired modes must share initialization and validation choices")
        summary = {mode: {} for mode in TRAINING_MODES}
        for mode, model_runs in results.items():
            for kind, runs in model_runs.items():
                summary[mode][kind] = {"training_mode": mode, "seeds": seeds,
                    "trainable_parameters": runs[0]["trainable_parameters"], "best_epochs": [r["best_epoch"] for r in runs],
                    "baseline_fallbacks": sum(r["baseline_fallback"] for r in runs),
                    "epochs_run": [r["epochs_run"] for r in runs], "optimizer_steps": [r["optimizer_steps"] for r in runs],
                    **{label: {metric: distribution([r[label]["metrics"][metric] for r in runs])
                               for metric in ("mean_rank_reward", "top1_accuracy", "cross_entropy", "head_evaluation_seconds")}
                       for label in validation_sets}}
        effects, gaps, structures = {}, {}, {}
        for label in validation_sets:
            effects[label] = {kind: paired_reward_interval(episode_reward_means(results["random"][kind], label),
                                                          episode_reward_means(results["mixed"][kind], label)) for kind in models}
            reference_rewards = {r["episode_id"]: r["rank_reward"] for r in reference[label]["predictions"]}
            gaps[label] = {mode: {kind: paired_reward_interval(reference_rewards, episode_reward_means(runs, label))
                                  for kind, runs in model_runs.items()} for mode, model_runs in results.items()}
            if set(models) == set(KINDS):
                structures[label] = {mode: paired_reward_interval(episode_reward_means(model_runs["global"], label),
                                                                  episode_reward_means(model_runs["sentence"], label))
                                     for mode, model_runs in results.items()}
        report = {**context, "result_status": "complete", "smoke_subset": args.smoke,
                  "train_episodes": len(train_ids), "validation_episodes": len(validation_sets["random"]),
                  "metric": "mean reward [1,.5,.25,.125,0]; fixed local proxies, NOT leaderboard score", "selection": SELECTION,
                  "zero_shot": {k: v["metrics"] for k, v in reference.items()}, "models": summary,
                  "training_effects": effects, "versus_zero_shot": gaps, "sentence_minus_global": structures,
                  "team_validation": team_provenance, "training_schedules": schedules.metadata,
                  "trained_on": {"random": "four random original training captions",
                                 "mixed": {"mix": args.train_mix, "fill": args.train_fill, "pool": "training_only"}},
                  "completed_fits": sum(len(runs) for mode in results.values() for runs in mode.values()),
                  "feature_preparation": bank["index"].get("timing", {}), "sentence_preparation": sentence_bank["index"].get("timing", {}),
                  "elapsed_seconds": time.perf_counter() - start, "submission_limit_user_reported": 30,
                  "limitations": ["Local development sets are reused for checkpoint/model selection; selection bias remains.",
                     "Epoch-0 fallback prevents lower selected TEAM reward on this development set, not hidden-test regression.",
                     "Random validation can decline when primary team reward improves; inspect both scores separately.",
                     "Team negatives may be semantically valid; sampler type families overlap between train and validation.",
                     "Bootstrap is conditional on selected models; it excludes seed/model-selection uncertainty and hidden-test shift.",
                     "Reverse/shuffle sensitivity does not establish correct temporal reasoning.",
                     "Head timings exclude frozen encoders; this run does not verify the server 15-minute limit.",
                     "No submission is made; GPU training must run on the user's free Colab/Kaggle T4."]}
        atomic_json(output / "comparison.json", report)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["training_mode", "model", "random_reward_mean", "random_reward_std", "team_reward_mean",
                         "team_reward_std", "random_top1_mean", "team_top1_mean", "parameters", "baseline_fallbacks"])
        writer.writerow(["none", "SigLIP_zero_shot", reference["random"]["metrics"]["mean_rank_reward"], 0,
                         reference["team"]["metrics"]["mean_rank_reward"], 0, reference["random"]["metrics"]["top1_accuracy"],
                         reference["team"]["metrics"]["top1_accuracy"], 0, "n/a"])
        for mode, model_results in summary.items():
            for kind, data in model_results.items():
                writer.writerow([mode, kind, data["random"]["mean_rank_reward"]["mean"], data["random"]["mean_rank_reward"]["std"],
                                 data["team"]["mean_rank_reward"]["mean"], data["team"]["mean_rank_reward"]["std"],
                                 data["random"]["top1_accuracy"]["mean"], data["team"]["top1_accuracy"]["mean"],
                                 data["trainable_parameters"], data["baseline_fallbacks"]])
        (output / "comparison.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
        artifacts = ("comparison.json", "comparison.csv", "zero_shot_reference.json", "run_config.json",
                     "random_validation_candidates.json", "team_validation_candidates.json")
        atomic_json(output / "status.json", {**context, "status": "complete",
                                            "artifacts": {name: file_record(output / name) for name in artifacts}})
        print(buffer.getvalue(), flush=True)
        return report
    except BaseException as exc:
        atomic_json(output / "status.json", {**context, "status": "failed", "error": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("contract-dir", "feature-dir", "sentence-feature-dir", "team-validation-dir", "train-zip", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", choices=KINDS, nargs="+", default=list(KINDS))
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--loss-temperature", type=float, default=0.07)
    p.add_argument("--train-mix", default=MIX)
    p.add_argument("--train-fill", default="random")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--residual-limit", type=float, default=0.1)
    p.add_argument("--attention-temperature", type=float, default=0.07)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.min_epochs, args.patience, args.batch_size) < 1 or (not args.smoke and args.min_epochs > args.epochs):
        p.error("Positive epochs/patience/batch-size required, with min-epochs <= epochs")
    if any(not math.isfinite(v) or v <= 0 for v in (args.learning_rate, args.loss_temperature, args.residual_limit, args.attention_temperature)):
        p.error("Learning rate, temperatures, and residual limit must be finite and positive")
    if args.hidden_dim < 4 or args.hidden_dim % 4:
        p.error("Hidden dimension must be a positive multiple of 4")
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 or s > 2**32 - 1 for s in args.seeds):
        p.error("Seeds must be distinct integers in [0,2**32-1]")
    if len(args.models) != len(set(args.models)):
        p.error("Models must be distinct")
    comparison(args)


if __name__ == "__main__":
    main()
