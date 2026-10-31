"""Check ranking models and training helpers without any real models or data."""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.negatives import NegativeGenerator, reversed_description
from team.ranker_model import MODELS, MeanPoolRanker, TemporalRanker
from team.train_ranker import collate_frames, order_pairing, reverse_frames, sample_negatives


class PairingEpisode:
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.description = "Opening. Middle part. Final part."


class PairingModel(torch.nn.Module):
    """Prefers the first text by `text_bias`, and by `video_gain` more on forward frames."""

    def __init__(self, text_bias, video_gain):
        super().__init__()
        self.text_bias, self.video_gain = text_bias, video_gain

    def logits(self, frames, frame_mask, tokens, token_mask, candidates):
        forward = (frames[:, 0, 0] < frames[:, -1, 0]).float()
        first = self.text_bias + self.video_gain * forward
        return torch.stack([first, torch.zeros_like(first)], dim=1)


@pytest.mark.parametrize("text_bias,video_gain,expected", [(5.0, 0.0, 0.5), (5.0, 1.0, 1.0), (0.0, 0.0, 0.5)])
def test_order_pairing_ignores_text_bias_and_rewards_video_order(monkeypatch, text_bias, video_gain, expected):
    monkeypatch.setattr("team.train_ranker.encode_tokens",
                        lambda encoders, texts: (torch.zeros(len(texts), 1, 1), torch.ones(len(texts), 1, dtype=torch.bool)))
    episodes = [PairingEpisode(f"e{i}") for i in range(4)]
    frames = {e.episode_id: torch.tensor([[1.0], [2.0], [3.0]]).half() for e in episodes}
    result = order_pairing(PairingModel(text_bias, video_gain), None, frames, episodes, "cpu")
    assert result == {"episodes": 4, "score": expected}


def test_reversed_description_keeps_the_opening():
    assert reversed_description("A. B. C.") == "A. C. B."
    assert reversed_description("A. B.") == "A. B."


def inputs(frames=6, episodes=2, candidates=5, seed=0):
    torch.manual_seed(seed)
    video = torch.randn(episodes, frames, 768)
    tokens = torch.randn(episodes * candidates, 9, 768)
    return (video, torch.ones(episodes, frames, dtype=torch.bool), tokens,
            torch.ones(episodes * candidates, 9, dtype=torch.bool), candidates)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_logits_have_one_score_per_candidate_and_accept_any_candidate_count(name):
    model = MODELS[name](hidden_dim=32, dropout=0.0).eval()
    assert model.logits(*inputs(candidates=5)).shape == (2, 5)
    assert model.logits(*inputs(candidates=3)).shape == (2, 3)


def test_temporal_model_responds_to_frame_order_but_mean_pool_does_not():
    frames, frame_mask, tokens, token_mask, candidates = inputs()
    backward = frames.flip(1)
    temporal = TemporalRanker(hidden_dim=32, dropout=0.0).eval()
    assert not torch.allclose(temporal.logits(frames, frame_mask, tokens, token_mask, candidates),
                              temporal.logits(backward, frame_mask, tokens, token_mask, candidates))
    mean_pool = MeanPoolRanker(hidden_dim=32, dropout=0.0).eval()
    assert torch.allclose(mean_pool.logits(frames, frame_mask, tokens, token_mask, candidates),
                          mean_pool.logits(backward, frame_mask, tokens, token_mask, candidates))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_padding_does_not_change_scores(name):
    model = MODELS[name](hidden_dim=32, dropout=0.0).eval()
    frames, frame_mask, tokens, token_mask, candidates = inputs(episodes=1)
    expected = model.logits(frames, frame_mask, tokens, token_mask, candidates)
    padded_frames = torch.cat([frames, torch.full((1, 3, 768), 9.0)], dim=1)
    padded_frame_mask = torch.cat([frame_mask, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
    padded_tokens = torch.cat([tokens, torch.full((5, 4, 768), 9.0)], dim=1)
    padded_token_mask = torch.cat([token_mask, torch.zeros(5, 4, dtype=torch.bool)], dim=1)
    actual = model.logits(padded_frames, padded_frame_mask, padded_tokens, padded_token_mask, candidates)
    assert torch.allclose(expected, actual, atol=1e-4)


def test_without_cross_attention_text_is_averaged_but_frame_order_still_counts():
    frames, frame_mask, tokens, token_mask, candidates = inputs()
    model = TemporalRanker(hidden_dim=32, dropout=0.0, cross_attention=False).eval()
    expected = model.logits(frames, frame_mask, tokens, token_mask, candidates)
    shuffled = tokens[:, torch.randperm(tokens.size(1))]
    assert torch.allclose(expected, model.logits(frames, frame_mask, shuffled, token_mask, candidates), atol=1e-4)
    assert not torch.allclose(expected, model.logits(frames.flip(1), frame_mask, tokens, token_mask, candidates))


def test_time_features_scale_with_each_episodes_length():
    model = TemporalRanker(hidden_dim=32)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    features = model.time_features(mask)
    assert torch.allclose(features[0, 2], features[1, 4], atol=1e-5)


def test_collate_and_reverse_keep_each_episodes_own_frames():
    features = [torch.tensor([[1.0], [2.0]]).half(), torch.tensor([[3.0]]).half()]
    frames, mask = collate_frames(reverse_frames(features), "cpu")
    assert frames.squeeze(-1).tolist() == [[2, 1], [3, 0]]
    assert mask.tolist() == [[True, True], [True, False]]


def test_sampled_negatives_are_distinct_and_fall_back_when_a_type_runs_out():
    descriptions = ["Only one sentence here.", *[f"Another one number {i}." for i in range(9)]]
    generator = NegativeGenerator(descriptions, [["a"]] * 10)
    negatives = sample_negatives(generator, 0, ["reorder"], 4, random.Random(1))
    assert len(negatives) == 4 == len(set(negatives))
    assert descriptions[0] not in negatives
