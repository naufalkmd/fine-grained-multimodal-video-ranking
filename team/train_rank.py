"""Train the kit's concat MLP on 1 correct + k negatives per episode, choosing the loss.

`--loss listwise` is a softmax cross-entropy over the 1 + k candidates, so the correct description
must outscore every negative, as in the 5-candidate ranking. `--loss bce` scores each pair on its
own, as the kit does, with the positive weighted k times so both classes count equally.
Checkpoints are chosen by the 5-candidate validation score on the trained types, not by BCE.

The model, features and checkpoint format match the kit, so export.py and team.rank_eval work on
the result unchanged. Negatives come from team.train_negatives and only from the training split.

    python -m team.train_rank --config cuda/config.json --output-dir cuda/outputs/listwise_lex2_sim2 \
        --k 4 --mix lexical=2,similar_episode=2 --loss listwise
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from baseline_model import IMAGE_MODEL_ID, TEXT_MODEL_ID, FrozenEncoders, SimilarityMLP
from common import ROOT, choose_device, load_config, save_json, seed_everything
from dataset import extract_features, feature_signature, load_episodes, split_indices, write_splits
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from team.negatives import load_action_labels
from team.rank_eval import RANK_SCORES, load_candidates
from team.train_negatives import TrainingNegatives, check_mix, epoch_record, parse_mix

LOSSES = ("listwise", "bce")


def candidate_loss(logits, loss):
    """logits: [episodes, 1 + k] with the correct description in column 0."""
    if loss == "listwise":
        return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long,
                                                   device=logits.device))
    labels = torch.zeros_like(logits)
    labels[:, 0] = 1
    negatives = logits.shape[1] - 1
    weights = torch.ones_like(logits)
    weights[:, 0] = negatives
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights,
                                              reduction="sum") / weights.sum()


def correct_ranks(scores, correct):
    """1-based rank of the correct column under a stable descending sort, as predict() does."""
    own = scores.gather(1, correct[:, None])
    columns = torch.arange(scores.shape[1], device=scores.device)[None, :]
    ahead = (scores > own) | ((scores == own) & (columns < correct[:, None]))
    return ahead.sum(dim=1) + 1


class TextBank:
    """Text features by string, encoding only texts not seen before."""

    def __init__(self, texts, features):
        self.index = {text: i for i, text in enumerate(texts)}
        self.features = features

    def add(self, texts, encoders):
        new = list(dict.fromkeys(text for text in texts if text not in self.index))
        if new:
            if encoders is None:
                raise ValueError("New negative texts need the text encoder.")
            self.features = torch.cat([self.features, encoders.encode_texts(new)])
            self.index.update({text: len(self.index) + i for i, text in enumerate(new)})

    def ids(self, texts):
        return torch.tensor([self.index[text] for text in texts])


def validation_sets(candidates, valid_rows):
    """Per set name: image rows [n], choice texts [n][5] and correct columns [n]."""
    sets = {}
    for entry in candidates:
        for name, candidate in entry["sets"].items():
            rows, texts, correct = sets.setdefault(name, ([], [], []))
            ids = [choice["id"] for choice in candidate["choices"]]
            rows.append(valid_rows[entry["episode_id"]])
            texts.append([choice["text"] for choice in candidate["choices"]])
            correct.append(ids.index(candidate["correct"]))
    return sets


@torch.inference_mode()
def validate(model, images, bank, sets, device):
    model.eval()
    results = {}
    for name, (rows, texts, correct) in sets.items():
        text_ids = torch.stack([bank.ids(choice_texts) for choice_texts in texts])
        image = images[torch.tensor(rows)].to(device)
        text = bank.features[text_ids].to(device)
        scores = model.logits(image[:, None].expand(-1, text_ids.shape[1], -1), text)
        ranks = correct_ranks(scores, torch.tensor(correct, device=device))
        results[name] = torch.tensor(RANK_SCORES, device=device)[ranks - 1].mean().item()
    return results


def train(config, device, candidates_path, select_on):
    seed_everything(config["seed"], config["cpu_threads"])
    device = choose_device(device)
    mix, k, fill, loss = config["negative_mix"], config["k"], config["fill"], config["loss"]
    check_mix(mix, k, fill)
    episodes = load_episodes(config["data_dir"])
    train_indices, valid_indices = split_indices(episodes, config["validation_fraction"],
                                                 config["seed"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "history.json").exists():
        raise FileExistsError(f"Use a new --output-dir; an existing run is present: {output}")
    save_json(output / "config.json", config)
    write_splits(output / "splits.json", episodes, train_indices, valid_indices)
    encoders = FrozenEncoders(
        config["image_model_dir"], config["text_model_dir"], device,
        config["image_batch_size"], config["text_batch_size"], config["max_text_length"],
    )
    signature = feature_signature(episodes, config)
    images, texts = extract_features(episodes, encoders, output / "features.pt", signature)
    images, texts = images.clone(), texts.clone()
    descriptions = [episode.description for episode in episodes]
    bank = TextBank(descriptions, texts)

    valid_ids = [episodes[i].episode_id for i in valid_indices]
    candidates = load_candidates(candidates_path, valid_ids, [descriptions[i] for i in valid_indices],
                                 [load_action_labels(episodes[i]) for i in valid_indices],
                                 config["seed"])
    sets = validation_sets(candidates, {episodes[i].episode_id: i for i in range(len(episodes))})
    # Picking checkpoints on held-out types would make their scores no longer held out.
    trained = {name for name, count in mix.items() if count}
    if sum(mix.values()) < k:
        trained.add(fill)
    select_on = select_on or sorted(trained)
    if unknown := set(select_on) - set(sets):
        raise ValueError(f"Validation sets not in the candidates file: {sorted(unknown)}")
    bank.add([text for _, choice_texts, _ in sets.values() for row in choice_texts for text in row],
             encoders)

    train_ids = [episodes[i].episode_id for i in train_indices]
    train_descriptions = [descriptions[i] for i in train_indices]
    sampler = TrainingNegatives(train_ids, train_descriptions,
                                [load_action_labels(episodes[i]) for i in train_indices],
                                mix, k, fill, config["seed"])
    copies_only = set(mix) | {fill} <= {"lexical", "similar_episode", "random"}
    model_config = {"image_dim": encoders.image_dim, "text_dim": encoders.text_dim,
                    "hidden_dim": config["hidden_dim"], "dropout": config["dropout"]}
    if copies_only:  # Every negative is a cached description, so the encoders are done.
        del encoders
        encoders = None
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    model = SimilarityMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    positives = torch.tensor(train_indices)
    provenance = {
        "image_model_id": IMAGE_MODEL_ID, "text_model_id": TEXT_MODEL_ID,
        "total_available_episodes": len(episodes), "used_episodes": len(episodes),
        "train_episodes": len(train_indices), "valid_episodes": len(valid_indices),
        "demo_subset": False, "feature_signature": signature,
        "negative_sampling": f"team.train_negatives k={k} mix={mix} fill={fill}, resampled per epoch",
        "loss": loss, "selected_on": select_on,
        "validation_candidates": hashlib.sha256(
            Path(candidates_path).read_bytes()).hexdigest()[:16],
    }
    best, stale, history, record = -1.0, 0, [], []
    for epoch in range(config["epochs"]):
        negatives = sampler.epoch(epoch)
        record.append(epoch_record(train_ids, train_descriptions, negatives))
        bank.add([text for items in negatives for _, text in items], encoders)
        negative_ids = torch.stack([bank.ids([text for _, text in items]) for items in negatives])
        model.train()
        loss_sum, count = 0.0, 0
        for batch in DataLoader(torch.arange(len(train_indices)), batch_size=config["batch_size"],
                                shuffle=True, generator=generator):
            text_ids = torch.cat([positives[batch, None], negative_ids[batch]], dim=1)
            image = images[positives[batch]].to(device)
            text = bank.features[text_ids].to(device)
            logits = model.logits(image[:, None].expand(-1, text_ids.shape[1], -1), text)
            batch_loss = candidate_loss(logits, loss)
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()
            loss_sum += batch_loss.item() * len(batch)
            count += len(batch)
        valid = validate(model, images, bank, sets, device)
        selection = sum(valid[name] for name in select_on) / len(select_on)
        metrics = {"epoch": epoch + 1, "train": {loss: loss_sum / count},
                   "valid": {**valid, "selection": selection}}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if selection > best:
            best, stale = selection, 0
            torch.save({
                "format_version": 1, "model_config": model_config,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": config, "epoch": epoch + 1, "metrics": metrics, "provenance": provenance,
            }, output / "best.pt")
        else:
            stale += 1
        save_json(output / "history.json", history)
        if stale >= config["patience"]:
            print("Early stopping.", flush=True)
            break
    save_json(output / "provenance.json", provenance)
    save_json(output / "train_negatives.json", {"k": k, "mix": mix, "fill": fill,
                                                 "seed": config["seed"], "epochs": record})
    print(f"Saved best checkpoint: {output / 'best.pt'}", flush=True)
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, default=4, help="Negatives per episode.")
    parser.add_argument("--mix", default="", help="Counts per type, e.g. lexical=2,similar_episode=2.")
    parser.add_argument("--fill", default="random", help="Type used for the rest of the k.")
    parser.add_argument("--loss", choices=LOSSES, default="listwise")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--candidates", type=Path, help="Validation candidates; defaults to the "
                        "same file as team.rank_eval.")
    parser.add_argument("--select-on", default="",
                        help="Comma-separated validation sets averaged to pick the checkpoint; "
                        "defaults to the negative types used in training.")
    args = parser.parse_args()
    config = load_config(args.config)
    candidates = args.candidates or Path(config["output_dir"]).parent / "validation/candidates.json"
    config["output_dir"] = str(args.output_dir.resolve())
    config.update({"k": args.k, "negative_mix": parse_mix(args.mix), "fill": args.fill,
                   "loss": args.loss})
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train(config, args.device, candidates, [name for name in args.select_on.split(",") if name])


if __name__ == "__main__":
    main()
