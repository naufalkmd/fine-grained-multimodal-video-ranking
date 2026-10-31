"""Server API v2: rank candidates by SigLIP's image-text match, with no trained layers.

Each candidate is split into sentences, because SigLIP reads at most 64 tokens. A sentence scores
its best match over the frames, and a candidate scores the mean over its sentences. Frame order is
not used. Its tokenizer loads from tokenizer.json because sentencepiece isn't in the evaluation
environment. No state is kept between episodes.
"""

import re
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

MAX_TOKENS = 64


def split_sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part] or [text]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir) / "models/siglip"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(root, local_files_only=True, use_fast=False)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "tokenizer.json"), eos_token="</s>", pad_token="</s>", unk_token="<unk>")
        self.model = AutoModel.from_pretrained(root, local_files_only=True, dtype=torch.float32)
        self.model.requires_grad_(False).to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def _images(self, images, batch_size=16):
        features = []
        for start in range(0, len(images), batch_size):
            pixels = self.processor(images=[image.convert("RGB") for image in images[start:start + batch_size]],
                                    return_tensors="pt")["pixel_values"].to(self.device, dtype=self.dtype)
            features.append(self.model.get_image_features(pixel_values=pixels).float())
        return torch.nn.functional.normalize(torch.cat(features), dim=-1)

    @torch.inference_mode()
    def _texts(self, sentences, batch_size=32):
        features = []
        for start in range(0, len(sentences), batch_size):
            tokens = self.tokenizer(sentences[start:start + batch_size], padding="max_length", truncation=True,
                                    max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
            features.append(self.model.get_text_features(input_ids=tokens["input_ids"]).float())
        return torch.nn.functional.normalize(torch.cat(features), dim=-1)

    def _matches(self, images, choices):
        if not images:
            raise ValueError("An episode must contain at least one image.")
        if not choices:
            raise ValueError("choices must not be empty.")
        sentences = [split_sentences(choice["text"]) for choice in choices]
        # Candidates that are edits of each other repeat sentences, so encode each one once.
        unique = list(dict.fromkeys(text for group in sentences for text in group))
        best = (self._texts(unique) @ self._images(images).T).amax(dim=1)
        lookup = dict(zip(unique, best.tolist()))
        return [sum(lookup[text] for text in group) / len(group) for group in sentences]

    def score(self, images, choices):
        """Mean best-frame cosine similarity, mapped from [-1, 1] to [0, 1]."""
        return [(match + 1) / 2 for match in self._matches(images, choices)]

    def predict(self, images, choices):
        matches = self._matches(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -matches[index])
        return [choices[index]["id"] for index in order]
