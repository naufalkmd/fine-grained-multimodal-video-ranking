"""E0--E3 paired local experiment. No leaderboard submission or teacher targets.

The two original validation sets select checkpoints. Reordered-caption diagnostics
are auxiliary and never used to select an epoch. Completed E0 checkpoints may be
reused only after provenance, schedules, initialization and predictions agree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
from pathlib import Path
import shutil
import statistics
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from comparison_protocol import paired_reward_interval
from data_contract import load_contracts
from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, sources_hash, verified_file
from factorial_data import prepare_reorder, build_order_validation, PairedReorderSchedules
from factorial_models import build_model
from prepare_features import load_features
from prepare_team_validation import load_team_validation, read_action_labels
from run_comparison import (atomic_checkpoint, distribution, episode_reward_means,
                            initial_parameters_hash, seed_all, synchronize, training_schedule)
from run_sentence_comparison import (batch_tensors, evaluate, input_diagnostics, score_statistics,
                                     selection_key, zero_gate, MIX, SELECTION)
from sentence_features import load_sentence_features
from training_schedules import TrainingSchedules

EXPERIMENTS = {
    "E0": {"frame_transformer": False, "reorder": False},
    "E1": {"frame_transformer": True, "reorder": False},
    "E2": {"frame_transformer": False, "reorder": True},
    "E3": {"frame_transformer": True, "reorder": True},
}
VERSION = "siglip-frame-reorder-factorial-v1"
LEGACY_EXCLUDED = {"factorial_models.py", "factorial_data.py", "run_factorial_comparison.py", "siglip_factorial_continue.py"}


def _state_hash(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def common_initial_hash(model, model_config):
    with torch.random.fork_rng(devices=[]):
        common_keys = set(build_model(False, model_config).state_dict())
    state = model.state_dict()
    if not common_keys <= set(state):
        raise ValueError("Frame model must preserve the common sentence module keys")
    return _state_hash({name: state[name] for name in common_keys})


def pair_metrics(evaluation, candidates):
    """Compare original and reordered text without assigning truth to altered video."""
    by_id = {row["episode_id"]: row for row in candidates}
    details = []
    for prediction in evaluation["predictions"]:
        row = by_id[prediction["episode_id"]]
        margin = prediction["scores"][row["correct_choice_id"]] - prediction["scores"][row["reorder_choice_id"]]
        win = 1.0 if margin > 1e-6 else 0.0 if margin < -1e-6 else 0.5
        details.append({"episode_id": row["episode_id"], "margin": margin, "win": win})
    if not details:
        raise ValueError("Order metrics require at least one eligible episode")
    return {"episodes": len(details), "win_rate": statistics.mean(row["win"] for row in details),
            "mean_margin": statistics.mean(row["margin"] for row in details),
            "ties": sum(row["win"] == 0.5 for row in details), "tie_tolerance": 1e-6,
            "note": "Auxiliary original-vs-rule-reordered caption proxy; semantic falsity is not guaranteed.",
            "predictions": details}


def _evaluate_all(model, sets, bank, sentence_bank, device, config):
    return {label: evaluate(model, rows, bank, sentence_bank, device, config["batch_size"], config["loss_temperature"])
            for label, rows in sets.items() if rows}


def _same_predictions(actual, saved):
    maximum = 0.0
    for label in ("random", "team"):
        a = {row["episode_id"]: row for row in actual[label]["predictions"]}
        b = {row["episode_id"]: row for row in saved[label]["predictions"]}
        if set(a) != set(b):
            raise ValueError(f"E0 {label} prediction episode coverage differs")
        for episode, row in a.items():
            old = b[episode]
            if row["ranking"] != old["ranking"] or row["correct_choice_id"] != old["correct_choice_id"]:
                raise ValueError(f"E0 {label} prediction rankings differ")
            if set(row["scores"]) != set(old["scores"]):
                raise ValueError("E0 prediction candidate IDs differ")
            maximum = max(maximum, *(abs(value - old["scores"][choice]) for choice, value in row["scores"].items()))
    if maximum > 1e-6:
        raise ValueError(f"E0 recomputed scores differ by {maximum}; tolerance is 1e-6")
    return maximum


def _without_file(audit):
    return {key: value for key, value in audit.items() if key != "file"}


def check_legacy_e0(folder, seed, config, model, base_schedules, validation_sets,
                    bank, sentence_bank, device):
    """Read-only validation. Raise on mismatch; caller records rejection and retrains."""
    folder = Path(folder)
    old = read_json(folder / "run_config.json")
    root = Path(__file__).resolve().parent
    names = {p.name for p in root.glob("*.py")
             if p.name not in LEGACY_EXCLUDED and not p.name.startswith("factorial_")}
    names |= {p.relative_to(root).as_posix() for p in (root / "team").glob("*.py")}
    if set(old["sources"]) != names:
        raise ValueError("E0 legacy source inventory differs")
    for name, expected in old["sources"].items():
        # Notebooks write source with LF; normalize Windows checkout newlines.
        actual = hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
        if expected not in {actual, file_record(root / name)["sha256"]}:
            raise ValueError(f"E0 legacy source differs: {name}")
    if old.get("version") != "siglip-sentence-residual-v1" or old.get("smoke"):
        raise ValueError("E0 source must be a completed full sentence-residual run")
    required = ("model", "epochs", "min_epochs", "patience", "batch_size", "learning_rate",
                "weight_decay", "loss_temperature", "selection", "early_stopping", "versions",
                "train_ids_hash", "split_hash", "random_candidates_hash", "team_candidates_hash",
                "features", "sentence_features")
    for key in required:
        if old.get(key) != config.get(key):
            raise ValueError(f"E0 configuration differs: {key}")
    if old.get("training_schedules") != base_schedules.metadata or old.get("schedule_fingerprint") != base_schedules.fingerprint:
        raise ValueError("E0 original negative sampling protocol differs")
    expected_hashes = {k: config["evaluated_candidates_hash"][k] for k in ("random", "team")}
    if old.get("evaluated_candidates_hash") != expected_hashes:
        raise ValueError("E0 evaluated candidate rows differ")
    if seed not in old.get("seeds", []) or "sentence" not in old.get("models", []):
        raise ValueError("E0 required seed/model was not run")
    status = read_json(folder / "status.json")
    if status.get("status") != "complete" or status.get("run_hash") != digest(old):
        raise ValueError("E0 comparison completion identity differs")
    for name, metadata in status["artifacts"].items():
        verified_file(folder / name, metadata)
    source = folder / "heads" / "mixed" / f"sentence-seed{seed}"
    marker = read_json(source / "complete.json")
    result = read_json(source / "result.json")
    identity = {"run_hash": digest(old), "training_mode": "mixed", "kind": "sentence", "seed": seed}
    for record in (marker, result):
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("E0 head completion identity differs")
    required_artifacts = {"initial.pt", "best.pt", "last.pt", "random_best.pt", "history.json",
                          "initial_debug.json", "result.json", "input_diagnostics.json"}
    if not required_artifacts <= set(marker["artifacts"]):
        raise ValueError("E0 requires actual completed checkpoint artifacts, not a result ZIP")
    for name, metadata in marker["artifacts"].items():
        verified_file(source / name, metadata)
    fresh_hash = initial_parameters_hash(model)
    initial = torch.load(source / "initial.pt", map_location="cpu", weights_only=True)
    best = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    for checkpoint in (initial, best):
        if any(checkpoint.get(key) != value for key, value in identity.items()):
            raise ValueError("E0 checkpoint identity differs")
        if checkpoint.get("model_config") != config["model"] or checkpoint.get("initial_parameters_hash") != fresh_hash:
            raise ValueError("E0 checkpoint model or initial state differs")
    if initial.get("epoch") != 0 or _state_hash(initial["state_dict"]) != fresh_hash:
        raise ValueError("E0 actual initial tensors differ")
    if result.get("initial_parameters_hash") != fresh_hash or best.get("epoch") != result.get("best_epoch"):
        raise ValueError("E0 selected checkpoint metadata differs")
    if result.get("validation_hashes") != {k: config[k + "_candidates_hash"] for k in ("random", "team")}:
        raise ValueError("E0 result validation hashes differ")
    if [item["epoch"] for item in result["used_negative_schedules"]] != list(range(result["epochs_run"])):
        raise ValueError("E0 training epoch coverage differs")
    for saved in result["used_negative_schedules"]:
        _, actual = base_schedules.epoch("team", seed, saved["epoch"])
        if _without_file(saved) != _without_file(actual):
            raise ValueError("E0 actual original negative schedules differ")
        saved_path = folder / "negative_schedules" / saved["file"]["path"]
        verified_file(saved_path, saved["file"])
    model.load_state_dict(best["state_dict"], strict=True)
    evaluation = _evaluate_all(model, {k: validation_sets[k] for k in ("random", "team")}, bank, sentence_bank, device, config)
    maximum = _same_predictions(evaluation, result)
    return {"status": "reused", "source_comparison": str(folder), "source_head": str(source),
            "source_run_hash": digest(old), "source_complete": file_record(source / "complete.json"),
            "source_best_checkpoint": file_record(source / "best.pt"), "max_abs_score_difference": maximum,
            "legacy_result": result, "legacy_source": source}


def _save_checkpoint(path, model, config, experiment, seed, epoch, run_hash, initial_hash, common_hash):
    atomic_checkpoint(path, {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                             "experiment": experiment, "seed": seed, "epoch": epoch,
                             "model_config": config["model"], "run_hash": run_hash,
                             "initial_parameters_hash": initial_hash, "common_initial_parameters_hash": common_hash,
                             **EXPERIMENTS[experiment]})


def train_experiment(experiment, seed, config, bank, sentence_bank, sets, reference, folder,
                     device, run_hash, schedules, base_schedules, reuse_e0_dir=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"run_hash": run_hash, "experiment": experiment, "seed": seed}
    use_frame, use_reorder = (EXPERIMENTS[experiment][k] for k in ("frame_transformer", "reorder"))
    marker = folder / "complete.json"
    if marker.exists():
        completed = read_json(marker)
        if any(completed.get(key) != value for key, value in identity.items()):
            raise ValueError("Completed factorial head belongs to a different request")
        for name, metadata in completed["artifacts"].items():
            verified_file(folder / name, metadata)
        result = read_json(folder / "result.json")
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError("Completed factorial result identity differs")
        for used in result["used_negative_schedules"]:
            _, actual = schedules.epoch(use_reorder, seed, used["epoch"])
            if used != actual:
                raise ValueError("Completed factorial negative schedule differs")
        print(f"Reuse completed {experiment} seed={seed}", flush=True)
        return result

    seed_all(seed)
    model = build_model(use_frame, config["model"]).to(device)
    initial_hash, common_hash = initial_parameters_hash(model), common_initial_hash(model, config["model"])
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    start = time.perf_counter()
    evaluations = _evaluate_all(model, {k: sets[k] for k in ("random", "team")}, bank, sentence_bank, device, config)
    gate = zero_gate(evaluations, reference)
    _save_checkpoint(folder / "initial.pt", model, config, experiment, seed, 0, run_hash, initial_hash, common_hash)
    initial_debug = {"zero_gate": gate, "initial_parameters_hash": initial_hash,
                     "common_initial_parameters_hash": common_hash,
                     "metrics": {key: value["metrics"] for key, value in evaluations.items()}}
    history = [{"epoch": 0, "train_cross_entropy": None, "gradient_norm": None,
                "random_validation": evaluations["random"]["metrics"], "team_validation": evaluations["team"]["metrics"],
                "selected_primary": True, "baseline_candidate": True}]
    used_schedules, steps, first_gradient, reuse = [], 0, None, {"status": "not_requested"}
    if experiment == "E0" and reuse_e0_dir and not config["smoke"]:
        try:
            reuse = check_legacy_e0(reuse_e0_dir, seed, config, model, base_schedules, sets, bank, sentence_bank, device)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            reuse = {"status": "rejected", "reason": str(exc), "source_comparison": str(reuse_e0_dir)}
            print(f"E0 seed={seed}: cannot reuse ({exc}); train from the common initialization.", flush=True)
            model.load_state_dict(initial_state)
            # Read-only validation can consume RNG while rebuilding schedules; reset it.
            seed_all(seed)
            model = build_model(False, config["model"]).to(device)
        else:
            source = reuse.pop("legacy_source")
            legacy = reuse.pop("legacy_result")
            for saved in legacy["used_negative_schedules"]:
                _, audit = schedules.epoch(False, seed, saved["epoch"])
                used_schedules.append(audit)
            history = read_json(source / "history.json")
            initial_debug["legacy_initial_debug"] = read_json(source / "initial_debug.json")
            steps, best_epoch = legacy["optimizer_steps"], legacy["best_epoch"]
            _save_checkpoint(folder / "best.pt", model, config, experiment, seed, best_epoch, run_hash, initial_hash, common_hash)
            # Keep the original last checkpoint separately, with its original provenance.
            shutil.copyfile(source / "last.pt", folder / "legacy_last.pt")
            print(f"E0 seed={seed}: verified and reused epoch {best_epoch}.", flush=True)

    if reuse["status"] != "reused":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        best, stale, best_epoch = selection_key(evaluations), 0, 0
        _save_checkpoint(folder / "best.pt", model, config, experiment, seed, 0, run_hash, initial_hash, common_hash)
        for epoch_index in range(config["epochs"]):
            epoch = epoch_index + 1
            model.train()
            schedule, audit = schedules.epoch(use_reorder, seed, epoch_index)
            used_schedules.append(audit)
            totals, count = {}, 0
            for offset in range(0, len(schedule), config["batch_size"]):
                batch = schedule[offset:offset + config["batch_size"]]
                tensors = batch_tensors([row[0] for row in batch], [row[1] for row in batch], bank, sentence_bank, device)
                optimizer.zero_grad(set_to_none=True)
                components = model.components(*tensors)
                target = torch.zeros(len(batch), dtype=torch.long, device=device)
                loss = F.cross_entropy(components["scores"] / config["loss_temperature"], target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite factorial training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite factorial gradient norm")
                if first_gradient is None:
                    first_gradient = {name: float(p.grad.detach().norm()) if p.grad is not None else None
                                      for name, p in model.named_parameters()}
                optimizer.step()
                steps += 1
                stats = {**score_statistics(components, target, config["loss_temperature"]),
                         "train_cross_entropy": float(loss.detach()), "gradient_norm": float(norm),
                         "correction_saturation_fraction": float((components["correction"].detach().abs() >= 0.95 * config["model"]["residual_limit"]).float().mean())}
                for key, value in stats.items():
                    totals[key] = max(totals.get(key, 0), value) if key.endswith("max_abs") else totals.get(key, 0) + value * len(batch)
                count += len(batch)
            evaluations = _evaluate_all(model, {k: sets[k] for k in ("random", "team")}, bank, sentence_bank, device, config)
            key = selection_key(evaluations)
            improved = key > best
            if improved:
                best, stale, best_epoch = key, 0, epoch
                _save_checkpoint(folder / "best.pt", model, config, experiment, seed, epoch, run_hash, initial_hash, common_hash)
            elif epoch >= config["min_epochs"]:
                stale += 1
            history.append({"epoch": epoch, **{key: value if key.endswith("max_abs") else value / count for key, value in totals.items()},
                            "random_validation": evaluations["random"]["metrics"], "team_validation": evaluations["team"]["metrics"],
                            "selected_primary": improved, "training_schedule_hash": digest(schedule), "negative_schedule": audit,
                            "stale_epochs_after_minimum": stale})
            atomic_json(folder / "history.json", history)
            _save_checkpoint(folder / "last.pt", model, config, experiment, seed, epoch, run_hash, initial_hash, common_hash)
            print(f"{experiment} seed={seed} epoch={epoch} CE={totals['train_cross_entropy']/count:.4f} "
                  f"random={evaluations['random']['metrics']['mean_rank_reward']:.6f} "
                  f"team={evaluations['team']['metrics']['mean_rank_reward']:.6f}", flush=True)
            if epoch >= config["min_epochs"] and stale >= config["patience"]:
                break
        initial_debug["first_batch_gradient_norms"] = first_gradient

    synchronize(device)
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise ValueError("Selected factorial checkpoint identity differs")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    evaluations = _evaluate_all(model, sets, bank, sentence_bank, device, config)
    if "order" in evaluations:
        evaluations["order_pair"] = pair_metrics(evaluations["order"], sets["order"])
    diagnostics = input_diagnostics(model, evaluations["team"]["predictions"], sets["team"], bank,
                                    sentence_bank, device, config["batch_size"])
    result = {**identity, **EXPERIMENTS[experiment], "best_epoch": checkpoint["epoch"],
              "baseline_fallback": checkpoint["epoch"] == 0, "initial_parameters_hash": initial_hash,
              "common_initial_parameters_hash": common_hash, "zero_gate": gate,
              "epochs_run": len(history) - 1, "optimizer_steps": steps,
              "used_negative_schedules": used_schedules, "reuse": reuse,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training_and_checkpoint_validation_seconds": training_seconds,
              "validation_hashes": config["evaluated_candidates_hash"], **evaluations,
              "diagnostics_summary": {op: {k: v for k, v in data.items() if k != "episodes_detail"} for op, data in diagnostics.items()}}
    atomic_json(folder / "result.json", result)
    atomic_json(folder / "initial_debug.json", initial_debug)
    atomic_json(folder / "history.json", history)
    atomic_json(folder / "input_diagnostics.json", diagnostics)
    atomic_json(folder / "reuse.json", reuse)
    names = ["result.json", "initial_debug.json", "history.json", "input_diagnostics.json", "reuse.json", "initial.pt", "best.pt"]
    names += ["legacy_last.pt"] if reuse["status"] == "reused" else ["last.pt"]
    atomic_json(marker, {**identity, "artifacts": {name: file_record(folder / name) for name in names}})
    return result


def interaction_interval(rewards, bootstrap_samples=2000):
    ids = sorted(rewards["E0"])
    if not ids or any(set(values) != set(ids) for values in rewards.values()):
        raise ValueError("Interaction requires identical episode coverage")
    delta = np.asarray([rewards["E3"][ep] - rewards["E2"][ep] - rewards["E1"][ep] + rewards["E0"][ep] for ep in ids])
    rng = np.random.default_rng(2029)
    samples = []
    for start in range(0, bootstrap_samples, 256):
        draws = rng.integers(0, len(ids), size=(min(256, bootstrap_samples - start), len(ids)))
        samples.extend(delta[draws].mean(axis=1).tolist())
    return {"episodes": len(ids), "mean_delta": float(delta.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "contrast": "(E3-E2)-(E1-E0)", "bootstrap_samples": bootstrap_samples,
            "method": "paired_episode_bootstrap_of_seed_mean_reward_interaction",
            "limitations": "Conditional on fitted models; excludes seed uncertainty, model selection, multiple comparisons and hidden-test shift."}


def comparison(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest, split, random_candidates = load_contracts(args.contract_dir)
    bank = load_features(args.feature_dir)
    expected = set(split["train_ids"] + split["val_ids"])
    if set(bank["frames"]) != expected or bank["index"]["source_fingerprint"] != manifest["source"]["fingerprint"]:
        raise ValueError("Original feature and contract coverage differs")
    sentence_bank = load_sentence_features(args.sentence_feature_dir, bank, manifest, split)
    team_candidates, team_provenance = load_team_validation(args.team_validation_dir, bank, manifest, split)
    if [row["episode_id"] for row in random_candidates["episodes"]] != [row["episode_id"] for row in team_candidates["episodes"]]:
        raise ValueError("Original validation video coverage/order differs")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    train_ids = split["train_ids"][:32] if args.smoke else split["train_ids"]
    labels = read_action_labels(args.train_zip, manifest, train_ids)
    original = {row["episode_id"]: row for row in manifest["episodes"]}
    base_schedules = TrainingSchedules(train_ids, {ep: original[ep]["description"] for ep in train_ids},
                                       {ep: labels[ep] for ep in train_ids}, bank["token_keys"], output / "base_negative_schedules",
                                       mix_text=MIX, fill="random", random_schedule_fn=training_schedule)
    augmentation_dir = Path(args.augmentation_dir) if getattr(args, "augmentation_dir", "") else output / "augmentation"
    bank, sentence_bank, augmentation = prepare_reorder(manifest, split, bank, sentence_bank, augmentation_dir,
                                                       device=args.device, text_batch_size=args.text_batch_size)
    schedules = PairedReorderSchedules(base_schedules, augmentation, output / "negative_schedules")
    sets = {"random": random_candidates["episodes"], "team": team_candidates["episodes"]}
    if args.smoke:
        sets = {name: rows[:8] for name, rows in sets.items()}
    order = build_order_validation(sets["random"], augmentation)
    if order:
        sets["order"] = order
    seeds = [2026] if args.smoke else args.seeds
    root = Path(__file__).resolve().parent
    hashes = {**{name: value for name, value in sources_hash().items() if name != "siglip_factorial_continue.py"},
              **{p.relative_to(root).as_posix(): file_record(p)["sha256"] for p in sorted((root / "team").glob("*.py"))}}
    config = {"version": VERSION, "seeds": seeds, "experiments": EXPERIMENTS,
              "epochs": 1 if args.smoke else args.epochs, "min_epochs": 1 if args.smoke else args.min_epochs,
              "patience": 1 if args.smoke else args.patience, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "weight_decay": .01, "loss_temperature": args.loss_temperature,
              "model": {"feature_dim": bank["index"]["feature_dim"], "hidden_dim": args.hidden_dim,
                        "attention_temperature": .07, "residual_limit": .1, "dropout": .1},
              "frame_module": {"hidden_dim": 64, "heads": 4, "layers": 1, "base_path_unchanged": True},
              "selection": SELECTION,
              "early_stopping": "Count stale epochs only from min_epochs onward; improvements reset count",
              "smoke": args.smoke, "device": args.device, "train_ids_hash": digest(train_ids), "split_hash": digest(split),
              "random_candidates_hash": digest(random_candidates), "team_candidates_hash": digest(team_candidates),
              "evaluated_candidates_hash": {name: digest(rows) for name, rows in sets.items()},
              "features": bank["index"]["fingerprint"], "sentence_features": sentence_bank["index"]["fingerprint"],
              "augmentation": augmentation["fingerprint"], "training_schedules": schedules.metadata,
              "schedule_fingerprint": schedules.fingerprint, "base_schedule_fingerprint": base_schedules.fingerprint,
              "sources": hashes, "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
              "reuse_e0_dir": str(Path(args.reuse_e0_dir)) if args.reuse_e0_dir else None,
              "loss": "CE(final score / .07, provided positive); no distillation or teacher labels",
              "submission_limit_user_reported": 30}
    immutable_json(output / "run_config.json", config)
    immutable_json(output / "random_validation_candidates.json", random_candidates)
    immutable_json(output / "team_validation_candidates.json", team_candidates)
    immutable_json(output / "order_validation_candidates.json", {"episodes": order, "selection_use": False,
                                                                 "source": "fixed random candidates, one negative replaced"})
    run_hash = digest(config)
    context = {"run_hash": run_hash, "attempt_id": uuid.uuid4().hex}
    if (output / "status.json").exists():
        previous = read_json(output / "status.json")
        if previous.get("run_hash") != run_hash:
            raise ValueError("Output belongs to another factorial request")
        if previous.get("status") == "complete":
            for name, meta in previous["artifacts"].items():
                verified_file(output / name, meta)
    atomic_json(output / "status.json", {**context, "status": "running"})
    start = time.perf_counter()
    try:
        reference = _evaluate_all(None, sets, bank, sentence_bank, args.device, config)
        if order:
            reference["order_pair"] = pair_metrics(reference["order"], order)
        atomic_json(output / "zero_shot_reference.json", reference)
        results = {}
        for experiment in EXPERIMENTS:
            results[experiment] = [train_experiment(experiment, seed, config, bank, sentence_bank, sets, reference,
                                                   output / "heads" / experiment / f"seed{seed}", args.device, run_hash,
                                                   schedules, base_schedules, args.reuse_e0_dir) for seed in seeds]
        for i, seed in enumerate(seeds):
            if len({results[experiment][i]["common_initial_parameters_hash"] for experiment in EXPERIMENTS}) != 1:
                raise ValueError(f"Shared sentence parameters were not initialized identically for seed {seed}")
            for a, b in (("E0", "E1"), ("E2", "E3")):
                left, right = results[a][i]["used_negative_schedules"], results[b][i]["used_negative_schedules"]
                if left[:min(len(left), len(right))] != right[:min(len(left), len(right))]:
                    raise ValueError("Paired structures used different epoch candidate schedules")
        summary = {}
        for experiment, runs in results.items():
            summary[experiment] = {**EXPERIMENTS[experiment], "seeds": seeds, "best_epochs": [r["best_epoch"] for r in runs],
                                   "epochs_run": [r["epochs_run"] for r in runs],
                                   "baseline_fallbacks": sum(r["baseline_fallback"] for r in runs),
                                   "reused_fits": sum(r["reuse"]["status"] == "reused" for r in runs),
                                   "trainable_parameters": runs[0]["trainable_parameters"],
                                   **{name: {metric: distribution([r[name]["metrics"][metric] for r in runs])
                                             for metric in ("mean_rank_reward", "top1_accuracy", "cross_entropy", "head_evaluation_seconds")}
                                      for name in sets}}
            if order:
                summary[experiment]["order_pair"] = {metric: distribution([r["order_pair"][metric] for r in runs])
                                                      for metric in ("win_rate", "mean_margin")}
        effects, gaps = {}, {}
        for name in sets:
            rewards = {experiment: episode_reward_means(runs, name) for experiment, runs in results.items()}
            effects[name] = {f"{b}_minus_{a}": paired_reward_interval(rewards[a], rewards[b])
                             for a, b in (("E0", "E1"), ("E0", "E2"), ("E2", "E3"), ("E1", "E3"))}
            effects[name]["interaction"] = interaction_interval(rewards)
            base = {row["episode_id"]: row["rank_reward"] for row in reference[name]["predictions"]}
            gaps[name] = {experiment: paired_reward_interval(base, value) for experiment, value in rewards.items()}
        if order:
            pair_rewards = {}
            for experiment, runs in results.items():
                maps = [{row["episode_id"]: row["win"] for row in run["order_pair"]["predictions"]} for run in runs]
                if any(set(values) != set(maps[0]) for values in maps):
                    raise ValueError("Order-pair seed episode coverage differs")
                pair_rewards[experiment] = {ep: statistics.mean(values[ep] for values in maps) for ep in maps[0]}
            effects["order_pair"] = {f"{b}_minus_{a}": {**paired_reward_interval(pair_rewards[a], pair_rewards[b]),
                                                        "metric": "original_vs_reorder_win_with_half_ties"}
                                     for a, b in (("E0", "E1"), ("E0", "E2"), ("E2", "E3"), ("E1", "E3"))}
            effects["order_pair"]["interaction"] = {**interaction_interval(pair_rewards),
                                                     "metric": "original_vs_reorder_win_with_half_ties"}
        report = {**context, "result_status": "complete", "smoke_subset": args.smoke,
                  "train_episodes": len(train_ids), "validation_episodes": len(sets["random"]), "order_validation_episodes": len(order),
                  "metric": "mean reward [1,.5,.25,.125,0]; fixed development proxies, NOT leaderboard scores",
                  "selection": SELECTION, "models": summary, "effects": effects, "versus_zero_shot": gaps,
                  "zero_shot": {name: value["metrics"] for name, value in reference.items() if "metrics" in value},
                  "zero_shot_order_pair": {k: v for k, v in reference.get("order_pair", {}).items() if k != "predictions"},
                  "team_validation": team_provenance, "augmentation": {k: v for k, v in augmentation.items() if k != "by_source"},
                  "completed_fits": sum(len(runs) for runs in results.values()),
                  "reused_fits": sum(row["reused_fits"] for row in summary.values()),
                  "elapsed_seconds": time.perf_counter() - start, "submission_limit_user_reported": 30,
                  "limitations": ["Reused development set; checkpoint/model selection bias remains.",
                                  "Rule-reordered captions are weak negatives, not guaranteed false statements.",
                                  "Frame sensitivity and reorder proxy scores do not prove correct temporal reasoning.",
                                  "Frozen base scores are preserved; post-training rankings or hidden-test accuracy are not guaranteed.",
                                  "Reported bootstrap intervals are conditional on selected fitted models.",
                                  "Feature-cached head timings do not verify the 15-minute offline inference limit."]}
        atomic_json(output / "comparison.json", report)
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["experiment", "random_reward_mean", "random_reward_std", "team_reward_mean", "team_reward_std",
                         "random_top1_mean", "team_top1_mean", "parameters", "baseline_fallbacks", "reused_fits",
                         "order_reward_mean", "order_pair_win_mean"])
        writer.writerow(["SigLIP_zero_shot", reference["random"]["metrics"]["mean_rank_reward"], 0,
                         reference["team"]["metrics"]["mean_rank_reward"], 0,
                         reference["random"]["metrics"]["top1_accuracy"], reference["team"]["metrics"]["top1_accuracy"],
                         0, "n/a", 0, reference["order"]["metrics"]["mean_rank_reward"] if order else "",
                         reference["order_pair"]["win_rate"] if order else ""])
        for experiment, row in summary.items():
            writer.writerow([experiment, row["random"]["mean_rank_reward"]["mean"], row["random"]["mean_rank_reward"]["std"],
                             row["team"]["mean_rank_reward"]["mean"], row["team"]["mean_rank_reward"]["std"],
                             row["random"]["top1_accuracy"]["mean"], row["team"]["top1_accuracy"]["mean"],
                             row["trainable_parameters"], row["baseline_fallbacks"], row["reused_fits"],
                             row["order"]["mean_rank_reward"]["mean"] if order else "",
                             row["order_pair"]["win_rate"]["mean"] if order else ""])
        (output / "comparison.csv").write_text(stream.getvalue(), encoding="utf-8-sig")
        names = ["comparison.json", "comparison.csv", "zero_shot_reference.json", "run_config.json",
                 "random_validation_candidates.json", "team_validation_candidates.json", "order_validation_candidates.json"]
        atomic_json(output / "status.json", {**context, "status": "complete", "artifacts": {name: file_record(output / name) for name in names}})
        print(stream.getvalue(), flush=True)
        return report
    except BaseException as exc:
        atomic_json(output / "status.json", {**context, "status": "failed", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract-dir", "feature-dir", "sentence-feature-dir", "team-validation-dir", "train-zip", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reuse-e0-dir", default="")
    parser.add_argument("--augmentation-dir", default="", help="Optional shared reorder feature cache for smoke/full runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--loss-temperature", type=float, default=.07)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.min_epochs, args.patience, args.batch_size, args.text_batch_size) < 1:
        parser.error("Epochs, patience and batch sizes must be positive")
    if not args.smoke and args.min_epochs > args.epochs:
        parser.error("min-epochs must not exceed epochs")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed <= 2**32 - 1 for seed in args.seeds):
        parser.error("Seeds must be distinct uint32 values")
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.loss_temperature)):
        parser.error("Learning rate and loss temperature must be positive and finite")
    comparison(args)


if __name__ == "__main__":
    main()
