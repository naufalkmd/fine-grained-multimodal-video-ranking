"""Frozen SigLIP cosine plus a bounded, zero-initialized supervised residual.

No teacher targets are used. Inputs are cached frozen encoder embeddings; every
valid frame participates in the sentence alignment. Masks must be valid prefixes.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrozenSiglipResidual(nn.Module):
    def __init__(self, kind="sentence", feature_dim=768, hidden_dim=64,
                 attention_temperature=0.07, residual_limit=0.1, dropout=0.1):
        super().__init__()
        if kind not in {"global", "sentence"}:
            raise ValueError("kind must be global or sentence")
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if kind == "sentence" and hidden_dim % 4:
            raise ValueError("sentence hidden_dim must be divisible by four")
        if not math.isfinite(attention_temperature) or attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive and finite")
        if not math.isfinite(residual_limit) or residual_limit <= 0:
            raise ValueError("residual_limit must be positive and finite")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.kind = kind
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.attention_temperature = float(attention_temperature)
        self.residual_limit = float(residual_limit)
        descriptor_dim = 2 * feature_dim + (10 if kind == "sentence" else 0)
        self.descriptor = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        if kind == "sentence":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.sentence_encoder = nn.TransformerEncoder(
                layer, num_layers=1, enable_nested_tensor=False,
            )
            self.sentence_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        # Only this projection is zero; gradients reach it on the first update.
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def _prefix_counts(mask):
        counts = mask.sum(-1)
        expected = torch.arange(mask.shape[-1], device=mask.device) < counts[..., None]
        if (counts == 0).any() or not torch.equal(mask, expected):
            raise ValueError("masks must be nonempty valid prefixes")
        return counts

    def _validated(self, frames, frame_mask, text, sentences, sentence_mask, positions, weights):
        if frames.ndim != 3 or text.ndim != 3 or sentences.ndim != 4:
            raise ValueError("expected frames[B,N,D], text[B,C,D], sentences[B,C,M,D]")
        b, n, d = frames.shape
        if n == 0 or d != self.feature_dim or text.shape[0] != b or text.shape[-1] != d:
            raise ValueError("invalid frame or text dimensions")
        c = text.shape[1]
        if c == 0 or sentences.shape[:2] != (b, c) or sentences.shape[-1] != d:
            raise ValueError("invalid sentence dimensions")
        m = sentences.shape[2]
        if m == 0 or frame_mask.shape != (b, n) or sentence_mask.shape != (b, c, m):
            raise ValueError("invalid mask shape")
        if positions.shape != sentence_mask.shape or weights.shape != sentence_mask.shape:
            raise ValueError("position/weight shape must match sentence mask")
        if any(x.dtype != torch.bool or x.device != frames.device
               for x in (frame_mask, sentence_mask)):
            raise ValueError("masks must be bool on the feature device")
        if not frames.is_floating_point() or any(
            not x.is_floating_point() or x.dtype != frames.dtype or x.device != frames.device
            for x in (text, sentences, positions, weights)
        ):
            raise ValueError("floating inputs must share dtype and device")
        counts = self._prefix_counts(frame_mask)
        self._prefix_counts(sentence_mask)
        if not all(torch.isfinite(x).all() for x in (
            frames[frame_mask], text, sentences[sentence_mask],
            positions[sentence_mask], weights[sentence_mask],
        )):
            raise ValueError("valid inputs must be finite")
        if (weights[sentence_mask] <= 0).any():
            raise ValueError("valid segment weights must be positive")
        if ((positions[sentence_mask] < 0) | (positions[sentence_mask] > 1)).any():
            raise ValueError("valid segment positions must be in [0, 1]")
        if m > 1 and ((positions[..., 1:] < positions[..., :-1]) & sentence_mask[..., 1:]).any():
            raise ValueError("segment positions must be nondecreasing")
        # masked_fill, unlike multiplication by zero, removes NaNs in padding.
        return (
            frames.masked_fill(~frame_mask[..., None], 0),
            sentences.masked_fill(~sentence_mask[..., None], 0),
            positions.masked_fill(~sentence_mask, 0),
            weights.masked_fill(~sentence_mask, 0), counts,
        )

    def components(self, frames, frame_mask, text, sentences, sentence_mask, positions, weights):
        frames, sentences, positions, weights, counts = self._validated(
            frames, frame_mask, text, sentences, sentence_mask, positions, weights
        )
        b, n, _ = frames.shape
        c, m = sentences.shape[1:3]
        video = F.normalize(frames.sum(1) / counts[:, None], dim=-1)
        # Keep the existing zero-shot score exactly, including truncate64 text.
        base = (video[:, None, :] * text).sum(-1)
        diagnostics = {}
        if self.kind == "global":
            expanded = video[:, None, :].expand_as(text)
            pooled = self.descriptor(torch.cat((expanded * text, (expanded - text).abs()), -1))
        else:
            similarity = torch.einsum("bnd,bcmd->bcnm", frames, sentences)
            valid_frames = frame_mask[:, None, :, None]
            attention = (similarity / self.attention_temperature).masked_fill(
                ~valid_frames, float("-inf")
            ).softmax(dim=2)
            aligned = torch.einsum("bcnm,bnd->bcmd", attention, frames)
            mean_similarity = similarity.sum(2) / counts[:, None, None]
            max_similarity = similarity.masked_fill(~valid_frames, float("-inf")).amax(2)
            min_similarity = similarity.masked_fill(~valid_frames, float("inf")).amin(2)
            attended_similarity = (attention * similarity).sum(2)
            first = similarity[:, :, 0, :]
            last_indices = (counts - 1)[:, None, None, None].expand(b, c, 1, m)
            last = similarity.gather(2, last_indices).squeeze(2)
            frame_positions = torch.arange(n, device=frames.device, dtype=frames.dtype)[None, :]
            frame_positions = frame_positions / (counts - 1).clamp_min(1)[:, None]
            expected_position = (attention * frame_positions[:, None, :, None]).sum(2)
            variance = (
                attention * (frame_positions[:, None, :, None] - expected_position[:, :, None, :]).square()
            ).sum(2)
            # Epsilon keeps optional input gradients finite for single-frame data.
            position_std = (variance.clamp_min(0) + 1e-8).sqrt()
            scalars = torch.stack((
                mean_similarity, max_similarity, min_similarity, attended_similarity,
                first, last, last - first, expected_position, position_std, positions,
            ), dim=-1)
            tokens = self.descriptor(torch.cat((
                aligned * sentences, (aligned - sentences).abs(), scalars,
            ), dim=-1)).masked_fill(~sentence_mask[..., None], 0)
            tokens = self.sentence_encoder(
                tokens.reshape(b * c, m, self.hidden_dim),
                src_key_padding_mask=~sentence_mask.reshape(b * c, m),
            ).reshape(b, c, m, self.hidden_dim)
            tokens = self.sentence_norm(tokens).masked_fill(~sentence_mask[..., None], 0)
            pooled = (tokens * weights[..., None]).sum(2) / weights.sum(2)[..., None]
            diagnostics["attention_entropy"] = (
                -(attention * attention.clamp_min(1e-12).log()).sum(2)
            ).masked_fill(~sentence_mask, 0)
            diagnostics["aligned_position"] = expected_position.masked_fill(~sentence_mask, 0)
        raw = self.output(self.dropout(pooled)).squeeze(-1)
        correction = self.residual_limit * raw.tanh()
        return {
            "base": base, "scores": base + correction, "correction": correction,
            "raw_correction": raw, "pooled_representation": pooled, **diagnostics,
        }

    def forward(self, frames, frame_mask, text, sentences, sentence_mask, positions, weights):
        return self.components(frames, frame_mask, text, sentences, sentence_mask, positions, weights)["scores"]
