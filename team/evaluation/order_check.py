"""Does a candidate's order of events match the order of the frames? A SigLIP check, no training.

Each candidate is split into events (sentences, clauses and "then" steps). SigLIP scores every
event against every frame. `aligned` is the best total when events must take frames in their
stated order (each event no earlier than the one before), `free` the total when each event takes
its best frame anywhere. `aligned - free` is zero when the stated order fits the frames, and more
negative the more it has to be bent. Candidates that only reorder the same events share every
event, so any text-only preference cancels and only the order counts.

`main` checks this locally on `reorder` sets and on Qwen's `changed_order` negatives, with the
right video, another episode's video and the right video reversed.

    python -m team.evaluation.order_check --episodes 200
"""

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

from common import ROOT, load_config
from dataset import load_episodes, load_images
from team.evaluation.rank_eval import RANK_SCORES

MAX_TOKENS = 64
SPLIT = re.compile(r"(?<=[.!?;])\s+|,\s+(?:and\s+)?(?:then\s+)?|\s+(?:and\s+)?then\s+")


def events(text):
    parts = [part.strip(" .") for part in SPLIT.split(text.strip())]
    return [part for part in parts if len(part.split()) >= 2] or [text]


class OrderChecker:
    def __init__(self, model_dir, device):
        root = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(root, local_files_only=True,
                                                            use_fast=False)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "tokenizer.json"), eos_token="</s>", pad_token="</s>",
            unk_token="<unk>")
        self.model = AutoModel.from_pretrained(root, local_files_only=True, dtype=torch.float32)
        self.model.requires_grad_(False).to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def frames(self, images):
        pixels = self.processor(images=[image.convert("RGB") for image in images],
                                return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        return torch.nn.functional.normalize(
            self.model.get_image_features(pixel_values=pixels).float(), dim=-1)

    @torch.inference_mode()
    def texts(self, sentences):
        tokens = self.tokenizer(sentences, padding="max_length", truncation=True,
                                max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(
            self.model.get_text_features(input_ids=tokens["input_ids"]).float(), dim=-1)

    def scores(self, frame_features, texts):
        """(aligned, free) mean per event for each text."""
        parts = [events(text) for text in texts]
        unique = list(dict.fromkeys(p for group in parts for p in group))
        similarity = (self.texts(unique) @ frame_features.T).cpu()
        index = {text: i for i, text in enumerate(unique)}
        results = []
        for group in parts:
            rows = similarity[[index[p] for p in group]]
            free = rows.max(dim=1).values.sum().item()
            best = rows[0].clone()
            for row in rows[1:]:
                best = row + torch.cummax(best, dim=0).values
            results.append((best.max().item() / len(group), free / len(group)))
        return results


def rank_score(values, correct):
    order = sorted(range(len(values)), key=lambda i: -values[i])
    return RANK_SCORES[order.index(correct)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "cuda/models/siglip")
    parser.add_argument("--candidates", type=Path,
                        default=ROOT / "cuda/outputs/validation/candidates_qwen.json")
    parser.add_argument("--generated", type=Path,
                        default=ROOT / "cuda/outputs/qwen/long_rewrite.jsonl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    episodes = {e.episode_id: e for e in load_episodes(load_config(args.config)["data_dir"])}
    checker = OrderChecker(args.model_dir, "cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    reorder = [e for e in candidates if "reorder" in e["sets"]][:args.episodes]
    results = {key: [] for key in ("order right", "order wrong", "order reversed",
                                   "aligned right", "aligned wrong")}
    started = time.perf_counter()
    for position, entry in enumerate(reorder):
        candidate = entry["sets"]["reorder"]
        texts = [c["text"] for c in candidate["choices"]]
        correct = [c["id"] for c in candidate["choices"]].index(candidate["correct"])
        other = reorder[(position + 1) % len(reorder)]["episode_id"]
        for video, episode_id, reverse in (("right", entry["episode_id"], False),
                                           ("wrong", other, False),
                                           ("reversed", entry["episode_id"], True)):
            images = load_images(episodes[episode_id])
            features = checker.frames(images[::-1] if reverse else images)
            for image in images:
                image.close()
            scores = checker.scores(features, texts)
            results[f"order {video}"].append(rank_score([a - f for a, f in scores], correct))
            if video != "reversed":
                results[f"aligned {video}"].append(rank_score([a for a, _ in scores], correct))
    print(f"reorder sets: {len(reorder)} episodes, "
          f"{(time.perf_counter() - started) / len(reorder):.3f} s per episode (3 videos each)")
    for key, values in results.items():
        print(f"  {key:16s} {statistics.fmean(values):.3f}")

    records = [json.loads(line) for line in args.generated.read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if "changed_order" in r["negatives"]][:args.episodes]
    wins = {"right": [], "wrong": [], "reversed": []}
    for position, record in enumerate(records):
        texts = [record["description"], record["negatives"]["changed_order"]]
        other = records[(position + 1) % len(records)]["episode_id"]
        for video, episode_id, reverse in (("right", record["episode_id"], False),
                                           ("wrong", other, False),
                                           ("reversed", record["episode_id"], True)):
            images = load_images(episodes[episode_id])
            features = checker.frames(images[::-1] if reverse else images)
            for image in images:
                image.close()
            (a0, f0), (a1, f1) = checker.scores(features, texts)
            gap = (a0 - f0) - (a1 - f1)
            wins[video].append(1.0 if gap > 0 else 0.5 if gap == 0 else 0.0)
    print(f"Qwen changed_order pairs: {len(records)}; correct text preferred:")
    for key, values in wins.items():
        print(f"  {key:16s} {statistics.fmean(values):.3f}")


if __name__ == "__main__":
    main()
