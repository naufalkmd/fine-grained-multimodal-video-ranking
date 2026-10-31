"""Server API v2 submission: rank candidates by how well their order of events fits the frames.

Each candidate is split into events (sentences, clauses and "then" steps). SigLIP scores every
event against every frame. A candidate's score is the best total when its events take frames in
the stated order, minus the total when each event takes its best frame anywhere, per event: zero
when the order fits, more negative the more it has to be bent. Ties go to centrality
(submission_central.py). No trained layers; SigLIP is stored in float16 and its tokenizer is read
from tokenizer.json, because sentencepiece isn't in the evaluation environment. No state is kept
between episodes. Same scoring as team/order_check.py.
"""

import re
import statistics
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

from submission_central import similarity_matrix

MAX_TOKENS = 64
SPLIT = re.compile(r"(?<=[.!?;])\s+|,\s+(?:and\s+)?(?:then\s+)?|\s+(?:and\s+)?then\s+")


def events(text):
    parts = [part.strip(" .") for part in SPLIT.split(text.strip())]
    return [part for part in parts if len(part.split()) >= 2] or [text]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir) / "models/siglip"
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
    def order_fit(self, images, texts):
        pixels = self.processor(images=[image.convert("RGB") for image in images],
                                return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        frames = torch.nn.functional.normalize(
            self.model.get_image_features(pixel_values=pixels).float(), dim=-1)
        parts = [events(text) for text in texts]
        unique = list(dict.fromkeys(p for group in parts for p in group))
        tokens = self.tokenizer(unique, padding="max_length", truncation=True,
                                max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
        sentences = torch.nn.functional.normalize(
            self.model.get_text_features(input_ids=tokens["input_ids"]).float(), dim=-1)
        similarity = (sentences @ frames.T).cpu()
        index = {text: i for i, text in enumerate(unique)}
        fits = []
        for group in parts:
            rows = similarity[[index[p] for p in group]]
            free = rows.max(dim=1).values.sum().item()
            best = rows[0].clone()
            for row in rows[1:]:
                best = row + torch.cummax(best, dim=0).values
            fits.append((best.max().item() - free) / len(group))
        return fits

    def ranking(self, images, texts):
        if not images:
            raise ValueError("An episode must contain at least one image.")
        fits = self.order_fit(images, texts)
        similarity = similarity_matrix(texts)
        central = [statistics.fmean(row[j] for j in range(len(texts)) if j != i)
                   for i, row in enumerate(similarity)]
        return sorted(range(len(texts)), key=lambda i: (-fits[i], -central[i]))

    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty.")
        if len(choices) == 1:
            return [1.0]
        ranking = self.ranking(images, [choice["text"] for choice in choices])
        return [1.0 - ranking.index(i) / (len(choices) - 1) for i in range(len(choices))]

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
