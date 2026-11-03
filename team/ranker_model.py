"""Models that score candidates using per-frame features and text tokens.

Both share one interface so that training and evaluation can compare them directly.
Candidates are flattened: an episode with C candidates supplies C rows of text tokens.
"""

import math

import torch
from torch import nn


def masked_mean(values, mask):
    mask = mask.unsqueeze(-1).to(values.dtype)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class MeanPoolRanker(nn.Module):
    """The kit's concat MLP: averaging frames means it can't see their order."""

    def __init__(self, image_dim=768, text_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def logits(self, frames, frame_mask, tokens, token_mask, candidates):
        video = masked_mean(frames, frame_mask).repeat_interleave(candidates, dim=0)
        pairs = torch.cat([video, masked_mean(tokens, token_mask)], dim=-1)
        return self.network(pairs).squeeze(-1).view(-1, candidates)


class TemporalRanker(nn.Module):
    """A temporal transformer over frames, with text tokens cross-attending to the result.

    Without cross-attention, text is averaged as in MeanPoolRanker, so that an ablation can
    measure the temporal layers on their own.
    """

    TIME_FREQUENCIES = 8

    def __init__(self, image_dim=768, text_dim=768, hidden_dim=256, heads=4,
                 video_layers=2, cross_layers=2, dropout=0.1, cross_attention=True, patch_grid=0):
        super().__init__()
        self.cross_attention = cross_attention
        self.patch_grid = patch_grid
        if patch_grid:
            self.patch_projection = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            self.patch_position = nn.Parameter(torch.zeros(patch_grid * patch_grid, hidden_dim))
        self.frame_projection = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.token_projection = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.time_projection = nn.Linear(2 * self.TIME_FREQUENCIES, hidden_dim)
        self.video_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, heads, 4 * hidden_dim, dropout,
                                       batch_first=True, norm_first=True),
            video_layers, enable_nested_tensor=False)
        self.cross_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, heads, 4 * hidden_dim, dropout,
                                       batch_first=True, norm_first=True),
            cross_layers)
        self.token_attention = nn.Linear(hidden_dim, 1)
        self.score = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def time_features(self, frame_mask):
        """Position as a fraction of the episode, because frame counts differ between episodes."""
        lengths = frame_mask.sum(dim=1, keepdim=True).clamp_min(2)
        indices = torch.arange(frame_mask.size(1), device=frame_mask.device).unsqueeze(0)
        fraction = (indices / (lengths - 1)).unsqueeze(-1)
        frequencies = 2 ** torch.arange(self.TIME_FREQUENCIES, device=frame_mask.device) * math.pi
        angles = fraction * frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def logits(self, frames, frame_mask, tokens, token_mask, candidates):
        patches = None
        if self.patch_grid:
            frames, patches = frames[:, :, 0], frames[:, :, 1:]
        time = self.time_projection(self.time_features(frame_mask))
        video = self.frame_projection(frames) + time
        video = self.video_encoder(video, src_key_padding_mask=~frame_mask)
        # Encode each episode's video once, then share it between that episode's candidates.
        memory = video.repeat_interleave(candidates, dim=0)
        memory_mask = frame_mask.repeat_interleave(candidates, dim=0)
        if patches is not None:
            detail = self.patch_projection(patches) + self.patch_position + time.unsqueeze(2)
            detail_mask = frame_mask.unsqueeze(2).expand(-1, -1, patches.size(2))
            detail = detail.flatten(1, 2).repeat_interleave(candidates, dim=0)
            detail_mask = detail_mask.flatten(1, 2).repeat_interleave(candidates, dim=0)
            attention_memory = torch.cat([memory, detail], dim=1)
            attention_mask = torch.cat([memory_mask, detail_mask], dim=1)
        else:
            attention_memory, attention_mask = memory, memory_mask
        text = self.token_projection(tokens)
        if self.cross_attention:
            text = self.cross_decoder(text, attention_memory, tgt_key_padding_mask=~token_mask,
                                      memory_key_padding_mask=~attention_mask)
            weights = self.token_attention(text).masked_fill(~token_mask.unsqueeze(-1), -1e4).softmax(dim=1)
            pooled_text = (weights * text).sum(dim=1)
        else:
            pooled_text = masked_mean(text, token_mask)
        pooled_video = masked_mean(memory, memory_mask)
        features = torch.cat([pooled_text, pooled_video, pooled_text * pooled_video], dim=-1)
        return self.score(features).squeeze(-1).view(-1, candidates)


MODELS = {"mean_pool": MeanPoolRanker, "temporal": TemporalRanker}
