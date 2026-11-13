"""Small trainable scoring heads over frozen, per-frame SigLIP features.

The input does not include labels, episode IDs, or candidate source IDs. Each
candidate is scored independently. B and C have identical trainable structures;
only their deterministic, adjacent-frame grouping differs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _check_frame_list(frames):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 2:
        raise ValueError("Frames must be a [time, feature] tensor")
    if min(frames.shape) < 1 or not frames.is_floating_point():
        raise ValueError("Frames must be a nonempty floating-point tensor")
    if not torch.isfinite(frames).all():
        raise ValueError("Frame features must be finite")


def collate_frames(frame_list):
    """Return zero-padded [B,T,D] features and a valid-prefix boolean mask."""
    if not frame_list:
        raise ValueError("At least one episode is required")
    for frames in frame_list:
        _check_frame_list(frames)
    first = frame_list[0]
    if any(x.shape[1] != first.shape[1] or x.device != first.device or
           x.dtype != first.dtype for x in frame_list):
        raise ValueError("Feature dimension, device, and dtype must match")
    maximum = max(x.shape[0] for x in frame_list)
    padded = first.new_zeros((len(frame_list), maximum, first.shape[1]))
    mask = torch.zeros((len(frame_list), maximum), dtype=torch.bool, device=first.device)
    for index, frames in enumerate(frame_list):
        padded[index, :len(frames)] = frames
        mask[index, :len(frames)] = True
    return padded, mask


def _group_ranges(frames, kind, bins):
    """Return disjoint, ordered, half-open frame ranges covering the episode."""
    n = len(frames)
    if n <= bins:
        return [(i, i + 1) for i in range(n)]
    if kind == "B":
        # array_split semantics: group sizes differ by at most one, no overlap.
        quotient, remainder = divmod(n, bins)
        groups, start = [], 0
        for index in range(bins):
            end = start + quotient + (index < remainder)
            groups.append((start, end))
            start = end
        return groups
    if kind != "C":
        raise ValueError("Only B and C use fixed-size segment grouping")

    # Merge decisions require no gradients. Keeping this tiny calculation on
    # CPU avoids a GPU synchronization for every adjacent-pair argmax.
    values = frames.detach().to(device="cpu", dtype=torch.float32)
    groups = [(i, i + 1) for i in range(n)]
    totals = [row.clone() for row in values]
    while len(groups) > bins:
        directions = F.normalize(torch.stack(totals), dim=-1)
        similarities = (directions[:-1] * directions[1:]).sum(dim=-1)
        # torch.argmax chooses the first maximum: ties are resolved leftward.
        index = int(similarities.argmax())
        groups[index:index + 2] = [(groups[index][0], groups[index + 1][1])]
        totals[index:index + 2] = [totals[index] + totals[index + 1]]
    return groups


def make_segments(frame_list, kind, bins):
    """Compress one [N,D] sequence into padded [K,D], [K], and [K,2].

    Positions are [original center index / max(N-1,1), frame_count / N].
    Each output feature is the unit-normalized mean of ALL original frames in
    its contiguous interval. Repeated merges never average unit centroids or
    give a one-frame group the same weight as a larger group.
    """
    _check_frame_list(frame_list)
    if kind not in {"B", "C"}:
        raise ValueError("kind must be B or C")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    n, dimension = frame_list.shape
    groups = _group_ranges(frame_list, kind, bins)
    pooled = frame_list.new_zeros((bins, dimension))
    mask = torch.zeros(bins, dtype=torch.bool, device=frame_list.device)
    positions = frame_list.new_zeros((bins, 2))
    for index, (start, end) in enumerate(groups):
        pooled[index] = F.normalize(frame_list[start:end].mean(dim=0), dim=0)
        mask[index] = True
        positions[index, 0] = ((start + end - 1) / 2) / max(n - 1, 1)
        positions[index, 1] = (end - start) / n
    return pooled, mask, positions


class TemporalRanker(nn.Module):
    """A: mean; B: uniform bins; C: adjacent merging; D: all-frame Transformer.

    ``forward(frames, mask, text)`` accepts [B,T,D], [B,T], [B,C,D].
    For B/C, supplying ``segment_positions=[B,K,2]`` means the input frames and
    mask are already the outputs of make_segments; this skips repeated pooling.
    All other forward inputs are frozen features. The returned [B,C] values are
    logits for ranking, not independently calibrated probabilities.
    """

    def __init__(self, kind, feature_dim=768, projection_dim=128,
                 hidden_dim=128, bins=10, dropout=0.1):
        super().__init__()
        if kind not in {"A", "B", "C", "D"}:
            raise ValueError("kind must be A, B, C, or D")
        if min(feature_dim, projection_dim, hidden_dim, bins) < 1:
            raise ValueError("Model dimensions and bins must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if kind == "D" and projection_dim % 4:
            raise ValueError("D requires projection_dim divisible by four heads")
        self.kind, self.feature_dim = kind, feature_dim
        self.projection_dim, self.bins = projection_dim, bins
        self.image_projection = nn.Sequential(nn.Linear(feature_dim, projection_dim), nn.GELU())
        self.text_projection = nn.Sequential(nn.Linear(feature_dim, projection_dim), nn.GELU())
        if kind in {"B", "C", "D"}:
            self.position_projection = nn.Linear(2, projection_dim)
        if kind == "D":
            layer = nn.TransformerEncoderLayer(
                d_model=projection_dim, nhead=4, dim_feedforward=2 * projection_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.temporal_norm = nn.LayerNorm(projection_dim)
        video_dim = bins * projection_dim if kind in {"B", "C"} else projection_dim
        self.scorer = nn.Sequential(
            nn.Linear(video_dim + projection_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _check_inputs(self, frames, mask):
        if frames.ndim != 3 or mask.shape != frames.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Expected frames [B,T,D] and boolean mask [B,T]")
        if frames.shape[-1] != self.feature_dim or min(frames.shape[:2]) < 1:
            raise ValueError("Invalid frame count or feature dimension")
        if frames.device != mask.device or not frames.is_floating_point():
            raise ValueError("Frames and mask must share a device; frames must be floating point")
        counts = mask.sum(dim=1)
        expected = torch.arange(frames.shape[1], device=frames.device)[None] < counts[:, None]
        if (counts == 0).any() or not torch.equal(mask, expected):
            raise ValueError("Each episode needs a nonempty valid prefix")
        if not torch.isfinite(frames[mask]).all():
            raise ValueError("Valid frame features must be finite")
        return counts

    def encode_video(self, frames, mask, segment_positions=None):
        counts = self._check_inputs(frames, mask)
        # masked_fill rather than multiplication also excludes NaN padding.
        clean = frames.masked_fill(~mask[..., None], 0)
        if self.kind == "A":
            if segment_positions is not None:
                raise ValueError("A does not accept segment positions")
            mean = clean.sum(dim=1) / counts[:, None]
            return self.image_projection(F.normalize(mean, dim=-1))
        if self.kind in {"B", "C"}:
            if segment_positions is None:
                grouped = [make_segments(clean[i, :int(n)], self.kind, self.bins)
                           for i, n in enumerate(counts)]
                clean = torch.stack([row[0] for row in grouped])
                mask = torch.stack([row[1] for row in grouped])
                positions = torch.stack([row[2] for row in grouped])
            else:
                if clean.shape[1] != self.bins or segment_positions.shape != (*mask.shape, 2):
                    raise ValueError("Precomputed segments must have shapes [B,K,D], [B,K,2]")
                if segment_positions.device != frames.device or not torch.isfinite(segment_positions[mask]).all():
                    raise ValueError("Segment positions must be finite on the feature device")
                positions = segment_positions.masked_fill(~mask[..., None], 0)
            encoded = self.image_projection(clean) + self.position_projection(positions)
            # Both projections have biases; padding MUST be cleared afterwards.
            return encoded.masked_fill(~mask[..., None], 0).flatten(start_dim=1)
        if segment_positions is not None:
            raise ValueError("D computes positions from all supplied frames")
        ordinal = torch.arange(clean.shape[1], device=frames.device, dtype=frames.dtype)[None]
        centers = ordinal / (counts - 1).clamp_min(1)[:, None]
        fractions = torch.ones_like(centers) / counts[:, None]
        positions = torch.stack((centers, fractions), dim=-1)
        encoded = self.image_projection(clean) + self.position_projection(positions)
        encoded = encoded.masked_fill(~mask[..., None], 0)
        contextual = self.temporal(encoded, src_key_padding_mask=~mask)
        contextual = self.temporal_norm(contextual).masked_fill(~mask[..., None], 0)
        return contextual.sum(dim=1) / counts[:, None]

    def forward(self, frames, mask, text, segment_positions=None):
        if (text.ndim != 3 or text.shape[0] != frames.shape[0] or
                text.shape[2] != self.feature_dim or text.shape[1] < 1):
            raise ValueError("Text must have shape [B,choices,feature_dim]")
        if text.device != frames.device or not torch.isfinite(text).all():
            raise ValueError("Text must be finite and on the feature device")
        video = self.encode_video(frames, mask, segment_positions)
        captions = self.text_projection(text)
        pairs = torch.cat((video[:, None, :].expand(-1, text.shape[1], -1), captions), dim=-1)
        return self.scorer(pairs).squeeze(-1)
