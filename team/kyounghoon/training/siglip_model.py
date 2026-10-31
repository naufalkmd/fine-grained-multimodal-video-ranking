"""Frozen SigLIP features; no training, labels, cross-episode updates, or submission API.

API checked against Transformers 4.57.6's modeling_siglip.py and
tokenization_siglip.py. SigLIP pools the last position, including padding, so
all inputs use exactly 64 positions and EOS padding, without an attention mask.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import PurePosixPath

MODEL_ID = "google/siglip-base-patch16-224"
TEXT_MODES = ("truncate64", "token_chunks_weighted_mean")
MAX_TOKENS = 64
IMPLEMENTATION_VERSION = "frozen-siglip-v2-fast-weighted-chunks"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def numeric_image_order(members):
    """Refuse ambiguous/non-numeric names instead of lexicographically sorting 10 before 2."""
    keyed = []
    for member in members:
        stem = PurePosixPath(member).stem
        if not stem.isdecimal():
            raise ValueError(f"Expected numeric frame filename: {member}")
        keyed.append((int(stem), str(member)))
    if len({number for number, _ in keyed}) != len(keyed):
        raise ValueError("Duplicate numeric frame IDs in one episode")
    return [member for _, member in sorted(keyed)]


def token_windows(content_ids, special_token_count, max_length=MAX_TOKENS):
    """Non-overlapping token chunks reserving the actual tokenizer special-token budget."""
    budget = max_length - special_token_count
    if budget < 1:
        raise ValueError("Tokenizer special tokens exhaust the text context")
    return [list(content_ids[start:start + budget])
            for start in range(0, len(content_ids), budget)] or [[]]


def pad_special_tokens(tokenizer, content_ids, max_length=MAX_TOKENS):
    ids = add_siglip_specials(tokenizer, content_ids)
    if len(ids) > max_length:
        raise ValueError("Chunk plus tokenizer special tokens exceeds the model context")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    return ids + [int(tokenizer.pad_token_id)] * (max_length - len(ids))


def add_siglip_specials(tokenizer, content_ids):
    """The generic fast wrapper lacks SiglipTokenizer's Python X </s> override."""
    if getattr(tokenizer, "is_fast", False):
        if tokenizer.num_special_tokens_to_add(pair=False) != 1 or tokenizer.eos_token_id is None:
            raise ValueError("Unexpected SigLIP backend special-token template")
        ids = list(content_ids)
        return ids if ids and ids[-1] == tokenizer.eos_token_id else ids + [tokenizer.eos_token_id]
    return list(tokenizer.build_inputs_with_special_tokens(list(content_ids)))


def make_text_plan(tokenizer, text, max_length=MAX_TOKENS):
    """Pure tokenization plan also used for truncation/collision audits."""
    raw = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    full_length = len(add_siglip_specials(tokenizer, raw))
    truncated = tokenizer(text, padding="max_length", max_length=max_length,
                          truncation=True, return_attention_mask=False)["input_ids"]
    if len(truncated) != max_length:
        raise ValueError("Tokenizer did not produce the fixed 64-token input")
    windows = token_windows(raw, specials, max_length)
    chunks = [pad_special_tokens(tokenizer, ids, max_length) for ids in windows]
    return {"content_token_count": len(raw), "full_token_count": full_length,
            "truncated": full_length > max_length, "chunk_count": len(chunks),
            "chunk_content_weights": [max(1, len(ids)) for ids in windows],
            "token_inputs": {"truncate64": [list(truncated)], "token_chunks_weighted_mean": chunks}}


def dot_score(image_features, text_features):
    if len(image_features) != len(text_features) or not image_features:
        raise ValueError("Feature dimensions do not match")
    score = math.fsum(float(a) * float(b) for a, b in zip(image_features, text_features))
    if not math.isfinite(score):
        raise ValueError("Non-finite matching score")
    return score


class FrozenSiglip:
    """Runtime imports are lazy so scoring and tokenizer tests need no ML dependencies."""

    def __init__(self, device="auto", image_batch_size=8, text_batch_size=16,
                 revision="main", local_files_only=False):
        import torch
        import transformers
        from PIL import __version__ as pillow_version
        from transformers import AutoImageProcessor, PreTrainedTokenizerFast, SiglipModel

        if min(image_batch_size, text_batch_size) < 1:
            raise ValueError("Batch sizes must be positive")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        torch.manual_seed(2026)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2026)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
        kwargs = {"revision": revision, "local_files_only": local_files_only}
        self.model = SiglipModel.from_pretrained(MODEL_ID, dtype=self.dtype,
                                                attn_implementation="eager", **kwargs)
        commit = getattr(self.model.config, "_commit_hash", None)
        if not commit:
            raise RuntimeError("HF snapshot commit could not be resolved; refusing ambiguous cache")
        pinned = {"revision": commit, "local_files_only": local_files_only}
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False, **pinned)
        # The official tokenizer.json is loaded directly: no SentencePiece/protobuf conversion.
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID, **pinned)
        if not self.tokenizer.is_fast or self.tokenizer.eos_token_id != self.tokenizer.pad_token_id:
            raise ValueError("Expected the official fast backend with EOS padding")
        probe = "Tokenizer parity check, with punctuation!"
        raw_probe = self.tokenizer.encode(probe, add_special_tokens=False)
        if self.tokenizer.encode(probe, add_special_tokens=True) != add_siglip_specials(self.tokenizer, raw_probe):
            raise ValueError("Official tokenizer.json does not apply the expected X </s> template")
        self.model.requires_grad_(False).to(self.device).eval()
        self.feature_dim = int(self.model.config.text_config.projection_size or
                               self.model.config.text_config.hidden_size)
        context = self.model.config.text_config.max_position_embeddings
        if context != MAX_TOKENS:
            raise ValueError(f"Unexpected SigLIP context: {context}")
        self.provenance = {
            "model_id": MODEL_ID, "requested_revision": revision, "snapshot_commit": commit,
            "model_config": self.model.config.to_dict(),
            "image_processor": self.processor.to_dict(),
            "tokenizer": {"class": type(self.tokenizer).__name__,
                          "backend_sha256": hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                          "model_max_length": self.tokenizer.model_max_length,
                          "special_tokens": self.tokenizer.special_tokens_map,
                          "padding_side": self.tokenizer.padding_side,
                          "special_token_count": self.tokenizer.num_special_tokens_to_add(pair=False)},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "pillow": pillow_version},
            "device": str(self.device), "dtype": str(self.dtype),
            "cuda_device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "image_batch_size": image_batch_size, "text_batch_size": text_batch_size,
            "implementation": IMPLEMENTATION_VERSION,
            "image_aggregation": "unit-normalize every frame; mean all frames; unit-normalize mean",
            "text_modes": {"truncate64": "official tokenizer truncation and fixed padding to 64",
                           "token_chunks_weighted_mean": "non-overlap token chunks up to 64 minus special budget; unit chunks; content-token-weighted mean; unit mean"},
            "attention_implementation": "eager",
            "attention_mask": "omitted, matching fixed-padding SigLIP tokenizer contract",
            "scores": "cosine similarity, not calibrated probability; shared monotone logit scale omitted",
            "training": False, "position_information": False,
        }

    def _unit(self, features):
        features = features.float()
        norm = features.norm(dim=-1, keepdim=True)
        if not self.torch.isfinite(features).all() or (norm <= 1e-12).any():
            raise ValueError("Invalid or zero feature norm")
        return features / norm

    def image_features(self, archive, image_members):
        from PIL import Image

        members = numeric_image_order(image_members)
        if not members:
            raise ValueError("Episode contains no images")
        total, count = None, 0
        with self.torch.inference_mode():
            for start in range(0, len(members), self.image_batch_size):
                images = []
                try:
                    for member in members[start:start + self.image_batch_size]:
                        with Image.open(io.BytesIO(archive.read(member))) as image:
                            images.append(image.convert("RGB"))
                    inputs = self.processor(images=images, return_tensors="pt")
                    pixels = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
                    frames = self._unit(self.model.get_image_features(pixel_values=pixels))
                    subtotal = frames.sum(dim=0)
                    total = subtotal if total is None else total + subtotal
                    count += len(images)
                finally:
                    for image in images:
                        image.close()
            return self._unit((total / count).unsqueeze(0))[0].cpu().tolist()

    def text_features(self, choices):
        """Only public [{id,text}] inputs enter the encoder, never correctness or origin."""
        plans = [make_text_plan(self.tokenizer, choice["text"]) for choice in choices]
        rows, ranges = [], []
        # Deduplicate exact model inputs, including short-caption overlap across modes.
        row_index = {}
        for plan in plans:
            indices = {}
            for mode in TEXT_MODES:
                selected = []
                for row in plan["token_inputs"][mode]:
                    key = tuple(row)
                    if key not in row_index:
                        row_index[key] = len(rows)
                        rows.append(row)
                    selected.append(row_index[key])
                indices[mode] = selected
            ranges.append(indices)
        encoded = []
        with self.torch.inference_mode():
            for start in range(0, len(rows), self.text_batch_size):
                ids = self.torch.tensor(rows[start:start + self.text_batch_size],
                                        dtype=self.torch.long, device=self.device)
                encoded.append(self._unit(self.model.get_text_features(input_ids=ids)).cpu())
            features = self.torch.cat(encoded)
            result = {}
            for mode in TEXT_MODES:
                vectors = []
                for item, plan in zip(ranges, plans):
                    selected = features[item[mode]]
                    weights = plan["chunk_content_weights"] if mode == "token_chunks_weighted_mean" else [1]
                    weight_tensor = self.torch.tensor(weights, dtype=self.torch.float32).unsqueeze(-1)
                    pooled = (selected * weight_tensor).sum(dim=0, keepdim=True) / weight_tensor.sum()
                    vectors.append(self._unit(pooled)[0].tolist())
                result[mode] = vectors
        return result, plans
