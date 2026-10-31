"""Controlled frame-Transformer addition to the existing sentence residual.

E0/E2 are the unchanged sentence model. E1/E3 add a small frame adapter only
inside the learned correction path. The original frozen cosine path always
uses the unmodified input features and the original truncate64 text feature.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from sentence_models import FrozenSiglipResidual


class FrameTransformerResidual(FrozenSiglipResidual):
    """All-frame temporal adapter followed by the existing sentence residual.

The common sentence layers are initialized before the new frame layers. A
forked CPU RNG preserves the RNG state following common-layer initialization,
so adding these parameters does not change common initial weights or the next
caller RNG draw. Training dropout draws naturally differ between structures.

Both the original score correction and the new frame adapter start at zero.
Consequently gradients first reach the score output, then the frame output,
and then the frame Transformer. No teacher outputs are learning targets.
"""

    TIME_FREQUENCIES = 8

    def __init__(self, feature_dim=768, hidden_dim=64,
                 attention_temperature=0.07, residual_limit=0.1, dropout=0.1,
                 frame_hidden_dim=64):
        if (not isinstance(frame_hidden_dim, int) or isinstance(frame_hidden_dim, bool)
                or frame_hidden_dim <= 0 or frame_hidden_dim % 4):
            raise ValueError("frame_hidden_dim must be positive and divisible by four")
        super().__init__(kind="sentence", feature_dim=feature_dim, hidden_dim=hidden_dim,
                         attention_temperature=attention_temperature,
                         residual_limit=residual_limit, dropout=dropout)
        self.frame_hidden_dim = frame_hidden_dim
        with torch.random.fork_rng(devices=[]):
            self.frame_projection = nn.Sequential(
                nn.Linear(feature_dim, frame_hidden_dim), nn.LayerNorm(frame_hidden_dim),
            )
            self.frame_time_projection = nn.Linear(2 * self.TIME_FREQUENCIES, frame_hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=frame_hidden_dim, nhead=4, dim_feedforward=2 * frame_hidden_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.frame_encoder = nn.TransformerEncoder(
                layer, num_layers=1, enable_nested_tensor=False,
            )
            self.frame_norm = nn.LayerNorm(frame_hidden_dim)
            self.frame_output = nn.Linear(frame_hidden_dim, feature_dim, bias=False)
            nn.init.zeros_(self.frame_output.weight)

    def _contextual_frames(self, frames, frame_mask, counts):
        """Accept validated, zero-padded features; retain every valid frame."""
        index = torch.arange(frames.shape[1], device=frames.device, dtype=frames.dtype)
        relative = index[None, :] / (counts - 1).clamp_min(1)[:, None]
        frequencies = 2 ** torch.arange(
            self.TIME_FREQUENCIES, device=frames.device, dtype=frames.dtype
        ) * math.pi
        angles = relative[..., None] * frequencies
        time = torch.cat((angles.sin(), angles.cos()), dim=-1)
        hidden = self.frame_projection(frames) + self.frame_time_projection(time)
        hidden = hidden.masked_fill(~frame_mask[..., None], 0)
        hidden = self.frame_encoder(hidden, src_key_padding_mask=~frame_mask)
        hidden = self.frame_norm(hidden).masked_fill(~frame_mask[..., None], 0)
        change = self.frame_output(hidden)
        return F.normalize(frames + change, dim=-1).masked_fill(~frame_mask[..., None], 0)

    def components(self, frames, frame_mask, text, sentences, sentence_mask, positions, weights):
        frames, sentences, positions, weights, counts = self._validated(
            frames, frame_mask, text, sentences, sentence_mask, positions, weights
        )
        # Compute the baseline BEFORE the learned frame transformation. This
        # remains identical to FrozenSiglipResidual even after adapter training.
        video = F.normalize(frames.sum(1) / counts[:, None], dim=-1)
        original_base = (video[:, None, :] * text).sum(-1)
        adapted = self._contextual_frames(frames, frame_mask, counts)
        result = super().components(
            adapted, frame_mask, text, sentences, sentence_mask, positions, weights
        )
        result["base"] = original_base
        result["scores"] = original_base + result["correction"]
        return result


def build_model(use_frame_transformer: bool, model_config: dict):
    """Build one architecture after the caller seeds PyTorch.

The standard model_config contains the existing sentence-model arguments. An
optional frame_hidden_dim is accepted only for the frame-Transformer condition.
    """
    if type(use_frame_transformer) is not bool:
        raise ValueError("use_frame_transformer must be a bool")
    config = dict(model_config)
    if use_frame_transformer:
        return FrameTransformerResidual(**config)
    if "frame_hidden_dim" in config:
        raise ValueError("frame_hidden_dim belongs only to the frame-Transformer condition")
    return FrozenSiglipResidual(kind="sentence", **config)
