"""Fine-tune SmolVLM to write the organizers' description from an episode's frames.

Uses only the correct descriptions, so no generated negatives are needed. On `--prior-probability`
of the batches the frames are left out, so that the same model also learns how likely a
description is with no video, the prior that team/inference/smolvlm_scorer.py subtracts. The vision encoder
stays frozen. Mixed precision is float16, so the run also works on a Colab T4.

    python -m team.training.train_smolvlm --config cuda/config.json \
        --splits cuda/outputs/baseline/splits.json --output-dir cuda/outputs/smolvlm/ft_v1
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from common import ROOT, load_config, save_json, seed_everything
from dataset import load_episodes, load_images
from team.inference.smolvlm_scorer import SmolVLMScorer, sample_frames


def batch_inputs(scorer, episodes, with_frames, rng, jitter):
    """Token ids, attention mask and labels (description tokens only) for a batch."""
    rows, pixel_values, pixel_masks = [], [], []
    for episode in episodes:
        images = load_images(episode)
        try:
            frames = sample_frames(images, scorer.max_frames)
            if jitter and len(images) > scorer.max_frames:
                # Random frames, still in time order, so the model doesn't learn one spacing.
                picks = sorted(rng.sample(range(len(images)), scorer.max_frames))
                frames = [images[index] for index in picks]
            frames = [image.convert("RGB") for image in frames]
        finally:
            for image in images:
                image.close()
        if with_frames:
            prompt = scorer.processor(text=scorer._prompt(len(frames)), images=[frames],
                                      return_tensors="pt")
            pixel_values.append(prompt["pixel_values"])
            pixel_masks.append(prompt["pixel_attention_mask"])
        else:
            prompt = scorer.tokenizer(scorer._prompt(0), return_tensors="pt")
        prefix = prompt["input_ids"][0].tolist()
        target = scorer._candidate_ids([episode.description])[0]
        rows.append((prefix + target, [-100] * len(prefix) + target))
    width = max(len(ids) for ids, _ in rows)
    pad = scorer.tokenizer.pad_token_id or 0
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for row, (values, targets) in enumerate(rows):
        ids[row, :len(values)] = torch.tensor(values)
        labels[row, :len(values)] = torch.tensor(targets)
        mask[row, :len(values)] = 1
    inputs = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    if with_frames:
        inputs["pixel_values"] = torch.cat(pixel_values)
        inputs["pixel_attention_mask"] = torch.cat(pixel_masks)
    return inputs


@torch.no_grad()
def validation_loss(model, scorer, episodes, batch_size, device):
    model.eval()
    losses = {}
    for with_frames in (True, False):
        total, count = 0.0, 0
        for start in range(0, len(episodes), batch_size):
            inputs = batch_inputs(scorer, episodes[start:start + batch_size], with_frames,
                                  None, False)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                total += model(**inputs).loss.item() * len(episodes[start:start + batch_size])
            count += len(episodes[start:start + batch_size])
        losses["with_frames" if with_frames else "without_frames"] = round(total / count, 4)
    model.train()
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "cuda/models/smolvlm-256m-instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prior-probability", type=float, default=0.2)
    parser.add_argument("--valid-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda")
    episodes = {episode.episode_id: episode for episode in load_episodes(config["data_dir"])}
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    train = [episodes[i] for i in splits["train"]]
    valid = [episodes[i] for i in splits["valid"]][:args.valid_episodes]

    scorer = SmolVLMScorer(args.model_dir, device, args.max_frames, args.image_size,
                           dtype=torch.float32)
    model = scorer.model
    for parameter in model.model.vision_model.parameters():
        parameter.requires_grad = False
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / (args.batch_size * args.accumulation))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 50) * max(0.0, 1 - step / steps))
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(args.seed)
    history = [{"epoch": 0, **validation_loss(model, scorer, valid, args.batch_size, device)}]
    print(json.dumps(history[-1]), flush=True)

    for epoch in range(1, args.epochs + 1):
        order = train[:]
        rng.shuffle(order)
        started, running = time.time(), []
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        for index, batch in enumerate(batches, 1):
            with_frames = rng.random() >= args.prior_probability
            inputs = batch_inputs(scorer, batch, with_frames, rng, True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs).loss / args.accumulation
            scaler.scale(loss).backward()
            running.append(loss.item() * args.accumulation)
            if index % args.accumulation == 0 or index == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
            if index % 100 == 0:
                print(f"epoch {epoch} batch {index}/{len(batches)} loss "
                      f"{sum(running[-100:]) / len(running[-100:]):.4f} "
                      f"{time.time() - started:.0f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": round(sum(running) / len(running), 4),
                        **validation_loss(model, scorer, valid, args.batch_size, device)})
        print(json.dumps(history[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.to(torch.float16).save_pretrained(args.output_dir / "model")
    scorer.processor.save_pretrained(args.output_dir / "model")
    save_json(args.output_dir / "training_info.json",
              {**{key: str(value) for key, value in vars(args).items()}, "history": history,
               "train_episodes": len(train), "trained_on": "correct descriptions only"})
    print(f"Saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
