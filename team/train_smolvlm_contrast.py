"""Continue fine-tuning SmolVLM so that the video, not the wording, separates one-word edits.

For each episode the correct description T and a one-word edit T' (team/word_swap.py, with a
replacement table built from the training descriptions only) are scored on the episode's frames V
and on another batch episode's frames V2. The loss asks for

    [log p(T | V) - log p(T' | V)] - [log p(T | V2) - log p(T' | V2)] > 0,

so a preference that comes from the text alone, such as T' reading less naturally, cancels and
only the video can lower it. The captioning loss on (V, T) and on T with no frames is kept, so that
the model still writes the descriptions and its no-frame likelihood stays usable as a prior.

    python -m team.train_smolvlm_contrast --splits cuda/outputs/baseline/splits.json \
        --model-dir cuda/outputs/smolvlm/ft_v1/model --output-dir cuda/outputs/smolvlm/contrast_v1
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import ROOT, load_config, save_json, seed_everything
from dataset import load_episodes, load_images
from team.smolvlm_scorer import SmolVLMScorer
from team.word_swap import replacement_table, swap_one_word


def episode_frames(scorer, episode, rng):
    images = load_images(episode)
    try:
        picks = sorted(rng.sample(range(len(images)), min(scorer.max_frames, len(images))))
        return [images[index].convert("RGB") for index in picks]
    finally:
        for image in images:
            image.close()


def pad_rows(rows, pad):
    width = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    targets = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for row, (values, start) in enumerate(rows):
        ids[row, :len(values)] = torch.tensor(values)
        targets[row, start:len(values)] = 1
        mask[row, :len(values)] = 1
    return ids, targets, mask


def row_log_likelihoods(model, ids, targets, mask, extra=None):
    """Summed log-probability of each row's target tokens, and the token count."""
    inputs = {"input_ids": ids, "attention_mask": mask, **(extra or {})}
    logits = model(**inputs).logits[:, :-1].float()
    scores = -F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
    weights = targets[:, 1:].float()
    return (scores * weights).sum(dim=1), weights.sum(dim=1)


def contrast_step(model, scorer, batch, table, rng, device):
    """Losses for one batch: video contrast, captioning with frames, captioning without."""
    pad = scorer.tokenizer.pad_token_id or 0
    frames = [episode_frames(scorer, episode, rng) for episode in batch]
    prompts = [scorer.processor(text=scorer._prompt(len(f)), images=[f], return_tensors="pt")
               for f in frames]
    edits = []
    for episode in batch:
        edited = None
        for _ in range(10):
            edited = swap_one_word(episode.description, table, rng)
            if edited and edited != episode.description:
                break
        edits.append(edited if edited != episode.description else None)

    rows, pixel_values, pixel_masks = [], [], []
    for index, episode in enumerate(batch):
        if edits[index] is None:
            continue
        other = (index + 1) % len(batch)
        texts = scorer._candidate_ids([episode.description, edits[index]])
        for video in (index, other):
            prefix = prompts[video]["input_ids"][0].tolist()
            for text in texts:
                rows.append((prefix + text, len(prefix)))
                pixel_values.append(prompts[video]["pixel_values"])
                pixel_masks.append(prompts[video]["pixel_attention_mask"])
    ids, targets, mask = (tensor.to(device) for tensor in pad_rows(rows, pad))
    extra = {"pixel_values": torch.cat(pixel_values).to(device),
             "pixel_attention_mask": torch.cat(pixel_masks).to(device)}
    scores, counts = row_log_likelihoods(model, ids, targets, mask, extra)
    scores = scores.view(-1, 4)
    margin = (scores[:, 0] - scores[:, 1]) - (scores[:, 2] - scores[:, 3])
    contrast = F.softplus(-margin).mean()
    caption = -(scores[:, 0] / counts.view(-1, 4)[:, 0]).mean()

    prefix = scorer.tokenizer(scorer._prompt(0))["input_ids"]
    text_rows = [(prefix + ids_, len(prefix))
                 for ids_ in scorer._candidate_ids([e.description for e in batch])]
    ids, targets, mask = (tensor.to(device) for tensor in pad_rows(text_rows, pad))
    prior_scores, prior_counts = row_log_likelihoods(model, ids, targets, mask)
    prior = -(prior_scores / prior_counts).mean()
    accuracy = (margin > 0).float().mean().item()
    return contrast, caption, prior, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Usually team.train_smolvlm's model/ folder.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--caption-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda")
    episodes = {episode.episode_id: episode for episode in load_episodes(config["data_dir"])}
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    train = [episodes[i] for i in splits["train"]]
    table = replacement_table([episode.description for episode in train])

    scorer = SmolVLMScorer(args.model_dir, device, args.max_frames, dtype=torch.float32)
    model = scorer.model
    for parameter in model.model.vision_model.parameters():
        parameter.requires_grad = False
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / (args.batch_size * args.accumulation))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 30) * max(0.0, 1 - step / steps))
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(args.seed)
    history = []

    for epoch in range(1, args.epochs + 1):
        order = train[:]
        rng.shuffle(order)
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        batches = [batch for batch in batches if len(batch) > 1]
        started, log = time.time(), []
        for index, batch in enumerate(batches, 1):
            with torch.autocast("cuda", dtype=torch.float16):
                contrast, caption, prior, accuracy = contrast_step(model, scorer, batch, table,
                                                                   rng, device)
                loss = (args.contrast_weight * contrast + args.caption_weight * caption
                        + args.prior_weight * prior) / args.accumulation
            scaler.scale(loss).backward()
            log.append((contrast.item(), caption.item(), prior.item(), accuracy))
            if index % args.accumulation == 0 or index == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
            if index % 50 == 0:
                recent = log[-50:]
                means = [sum(item[k] for item in recent) / len(recent) for k in range(4)]
                print(f"epoch {epoch} batch {index}/{len(batches)} contrast {means[0]:.4f} "
                      f"caption {means[1]:.4f} prior {means[2]:.4f} margin>0 {means[3]:.3f} "
                      f"{time.time() - started:.0f}s", flush=True)
        history.append({"epoch": epoch, **{name: round(sum(item[k] for item in log) / len(log), 4)
                        for k, name in enumerate(["contrast", "caption", "prior", "margin_positive"])}})
        print(json.dumps(history[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.to(torch.float16).save_pretrained(args.output_dir / "model")
    scorer.processor.save_pretrained(args.output_dir / "model")
    save_json(args.output_dir / "training_info.json",
              {**{key: str(value) for key, value in vars(args).items()}, "history": history,
               "train_episodes": len(train),
               "trained_on": "correct descriptions and one-word edits of them (word_swap), "
                             "with another training episode's frames as the comparison"})
    print(f"Saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
