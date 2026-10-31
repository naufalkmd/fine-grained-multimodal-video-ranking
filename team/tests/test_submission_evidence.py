"""Check image-evidence scoring without any real models."""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "team"))

import ranker_model  # noqa: F401, needed by submission_ranker's import
sys.modules.setdefault("ranker_model", ranker_model)

from submission_evidence import evidence_scores


class ScriptedModel(nn.Module):
    """Score = text preference per candidate + video_gain x (first frame value) per candidate."""

    def __init__(self, text, video):
        super().__init__()
        self.text, self.video = torch.tensor(text), torch.tensor(video)

    def logits(self, frames, frame_mask, tokens, token_mask, candidates):
        strength = frames[:, 0, 0].unsqueeze(1)
        return self.text.unsqueeze(0) + strength * self.video.unsqueeze(0)


def run(model):
    frames = torch.full((1, 2, 4), 2.0)
    references = torch.zeros(4, 2, 4)
    masks = torch.ones(1, 2, dtype=torch.bool), torch.ones(4, 2, dtype=torch.bool)
    tokens = torch.zeros(3, 2, 4)
    return evidence_scores(model, frames, masks[0], references, masks[1], tokens,
                           torch.ones(3, 2, dtype=torch.bool), 3)


def test_a_preference_that_ignores_the_video_cancels_exactly():
    scores = run(ScriptedModel([5.0, 1.0, -2.0], [0.0, 0.0, 0.0]))
    assert torch.allclose(scores, torch.zeros(3))


def test_what_the_video_adds_decides_the_order_despite_text_preference():
    scores = run(ScriptedModel([5.0, 1.0, -2.0], [0.0, 3.0, 1.0]))
    assert scores.argmax().item() == 1
    assert torch.allclose(scores, torch.tensor([0.0, 6.0, 2.0]))
