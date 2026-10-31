"""Check ranking models and training helpers without any real models or data."""

import io
import json
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.negatives import NegativeGenerator, reversed_description
from team.ranker_model import MODELS, MeanPoolRanker, TemporalRanker
from team.train_ranker import (
    SELECTION_SETS,
    collate_frames,
    draw_negatives,
    order_pairing,
    reverse_frames,
    run_epoch,
    sample_negatives,
)


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
    assert set(negatives.values()) <= {"reorder", "random"}

def test_draw_negatives_follows_a_mix_or_falls_back_to_drawn_types():
    descriptions = [f"Person {i} opens box {i % 5}. They lift cup {i % 3}. They wave." for i in range(30)]
    generator = NegativeGenerator(descriptions, [["open"] if i % 2 else ["show"] for i in range(30)])
    mixed = draw_negatives(generator, 0, None, 8, random.Random(0),
                           {"lexical": 3, "similar_episode": 3}, "random")
    assert len(mixed) == 8 == len(set(mixed))
    assert descriptions[0] not in mixed
    assert (draw_negatives(generator, 0, ["random"], 4, random.Random(2))
            == sample_negatives(generator, 0, ["random"], 4, random.Random(2)))


class TrainEpisode:
    def __init__(self, index):
        self.episode_id = f"e{index}"
        self.description = f"Person {index} opens a box. They lift cup {index}. They wave {index}."


def test_run_epoch_ranks_one_correct_and_k_negatives_also_when_flipped(monkeypatch):
    seen = []

    def fake_tokens(encoders, texts):
        seen.append(list(texts))
        return torch.randn(len(texts), 3, 768), torch.ones(len(texts), 3, dtype=torch.bool)

    monkeypatch.setattr("team.train_ranker.encode_tokens", fake_tokens)
    episodes = [TrainEpisode(i) for i in range(12)]
    generator = NegativeGenerator([e.description for e in episodes], [["a"]] * 12)
    frames = {e.episode_id: torch.randn(4, 768).half() for e in episodes}
    model = TemporalRanker(hidden_dim=32, dropout=0.0, cross_attention=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_epoch(model, optimizer, None, frames, generator, episodes, None, "cpu", 0, 1,
                        batch_size=4, reverse_weight=0.1, flip_probability=1.0, count=8,
                        mix={"lexical": 4, "random": 4})
    assert metrics["loss"] > 0
    batches = [texts for texts in seen if len(texts) % 9 == 0 and len(texts) > 4]
    assert sum(len(texts) for texts in batches) == 9 * len(episodes)
    # Every episode is flipped, so its original wording must be among the 9 as a negative.
    flat = [text for texts in batches for text in texts]
    assert all(e.description in flat for e in episodes)
    assert all(reversed_description(e.description) in flat for e in episodes)


def test_selection_uses_only_sets_without_a_detected_text_leak():
    assert set(SELECTION_SETS) == {"lexical", "similar_episode"}


def test_run_epoch_records_the_candidates_it_used(monkeypatch):
    monkeypatch.setattr("team.train_ranker.encode_tokens", lambda encoders, texts: (
        torch.randn(len(texts), 3, 768), torch.ones(len(texts), 3, dtype=torch.bool)))
    episodes = [TrainEpisode(i) for i in range(4)]
    generator = NegativeGenerator([e.description for e in episodes], [["a"]] * 4)
    frames = {e.episode_id: torch.randn(4, 768).half() for e in episodes}
    model = TemporalRanker(hidden_dim=32, dropout=0.0)
    record = io.StringIO()
    run_epoch(model, torch.optim.AdamW(model.parameters()), None, frames, generator, episodes, ["random"],
              "cpu", 0, 1, batch_size=2, reverse_weight=0.0, flip_probability=1.0, record=record)
    rows = [json.loads(line) for line in record.getvalue().splitlines()]
    assert {row["episode_id"] for row in rows} == {e.episode_id for e in episodes}
    for row, episode in zip(sorted(rows, key=lambda r: r["episode_id"]), episodes):
        assert row["flipped"] and len(row["negatives"]) == 4
        assert row["correct"] == reversed_description(episode.description)
        assert row["negatives"][episode.description] == "flipped_original"


QWEN_RECORD = {"episode_id": "e1", "positive": "A person folds a towel and places it down.",
               "negatives": {"reversed_action": "A person unfolds a towel and places it down.",
                             "swapped_order": "A person places a towel down and folds it."}}


def test_qwen_candidates_fill_to_k_and_flip_with_the_swapped_sentence():
    from team.train_ranker import qwen_candidates

    pool = [f"Someone does thing {i}." for i in range(10)]
    correct, negatives, flipped = qwen_candidates(QWEN_RECORD, 8, 0.0, pool, random.Random(0))
    assert not flipped and correct == QWEN_RECORD["positive"]
    assert len(negatives) == 8 and correct not in negatives
    assert negatives[QWEN_RECORD["negatives"]["reversed_action"]] == "qwen_reversed_action"
    correct, negatives, flipped = qwen_candidates(QWEN_RECORD, 4, 1.0, pool, random.Random(0))
    assert flipped and correct == QWEN_RECORD["negatives"]["swapped_order"]
    assert negatives[QWEN_RECORD["positive"]] == "qwen_flipped_original"
    assert len(negatives) == 4 and correct not in negatives


def test_qwen_records_come_only_from_the_training_split(tmp_path):
    from team.train_ranker import load_qwen_records

    path = tmp_path / "generated.jsonl"
    rows = [{**QWEN_RECORD, "episode_id": "train_ok"}, {**QWEN_RECORD, "episode_id": "valid_one"},
            {**QWEN_RECORD, "episode_id": "train_empty", "negatives": {}}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert set(load_qwen_records(path, {"train_ok", "train_empty"})) == {"train_ok"}


def test_submission_centrality_and_standardize():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # submission.py imports ranker_model
    from team.submission_ranker import centrality, standardize

    texts = ["A man is buttoning his shirt.", "A man is unbuttoning his shirt.",
             "A man is buttoning his shirt and then sits down.", "A dog runs in a park."]
    scores = centrality(texts)
    assert max(range(4), key=lambda i: scores[i]) == 0
    assert min(range(4), key=lambda i: scores[i]) == 3
    values = standardize([1.0, 2.0, 3.0])
    assert abs(sum(values)) < 1e-9 and values[0] < values[1] < values[2]
    assert standardize([5.0, 5.0]) == [0.0, 0.0]


def test_qwen_reverse_actions_lets_the_reversed_sentence_be_correct_on_reversed_frames():
    from team.train_ranker import qwen_candidates

    record = {**QWEN_RECORD, "negatives": {
        "reversed_action": "A person unfolds a towel and places it down."}}
    pool = [f"Someone does thing {i}." for i in range(10)]
    correct, negatives, flipped = qwen_candidates(record, 4, 1.0, pool, random.Random(0))
    assert not flipped and correct == record["positive"]
    correct, negatives, flipped = qwen_candidates(record, 4, 1.0, pool, random.Random(0), True)
    assert flipped and correct == record["negatives"]["reversed_action"]
    assert negatives[record["positive"]] == "qwen_flipped_original"



def test_action_pairing_ignores_wording_and_rewards_video_direction(monkeypatch):
    from team import action_pairing as module

    class Episode:
        frames = ()

    class Frame(int):
        def close(self):
            pass

    monkeypatch.setattr(module, "load_images", lambda episode: [Frame(1), Frame(2), Frame(3)])

    class Predictor:
        def __init__(self, text_bias, video_gain):
            self.text_bias, self.video_gain = text_bias, video_gain

        def _logits(self, images, choices):
            forward = 1.0 if images[0] < images[-1] else 0.0
            return torch.tensor([self.text_bias + self.video_gain * forward, 0.0])

    records = [{"episode_id": "e", "positive": "folds", "negatives": {"reversed_action": "unfolds"}}]
    episodes = {"e": Episode()}
    assert module.action_pairing(Predictor(5.0, 0.0), episodes, records)["score"] == 0.5
    assert module.action_pairing(Predictor(5.0, 1.0), episodes, records)["score"] == 1.0
