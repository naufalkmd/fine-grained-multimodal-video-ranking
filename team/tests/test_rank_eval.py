"""Check generated negatives and the 5-candidate score without any models or real data."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.negatives import (
    NEGATIVE_TYPES,
    NegativeGenerator,
    build_validation_sets,
    candidate_set,
    reorder_clauses,
    split_sentences,
)
from team.rank_eval import RANDOM_SCORE, order_check, rank_score, score_known_scorers

DESCRIPTIONS = [
    f"Person {i} opens box {i}, then lifts it. They move cup {i}. They wave {i}, then sit. "
    f"Finally, they close door {i}."
    for i in range(30)
]
LABELS = [["open", "close"] if i % 2 else ["show", "show", "pour"] for i in range(30)]
IDS = [f"episode_{i}" for i in range(30)]


@pytest.mark.parametrize("negative_type", NEGATIVE_TYPES)
def test_negatives_are_reproducible_distinct_and_never_the_description(negative_type):
    generator = NegativeGenerator(DESCRIPTIONS, LABELS)
    first = generator.generate(0, negative_type, 4, random.Random(7))
    assert first == generator.generate(0, negative_type, 4, random.Random(7))
    assert len(first) == 4 == len(set(first))
    assert DESCRIPTIONS[0] not in first


def test_reorder_keeps_the_opening_and_starts_by_reversing_the_rest():
    generator = NegativeGenerator(DESCRIPTIONS, LABELS)
    negatives = generator.generate(0, "reorder", 4, random.Random(1))
    opening, *rest = split_sentences(DESCRIPTIONS[0])
    assert negatives[0] == " ".join([opening, *reversed(rest)])
    assert all(text.startswith(opening) for text in negatives)


def test_donor_sentences_are_never_openings():
    generator = NegativeGenerator(DESCRIPTIONS, LABELS)
    openings = {split_sentences(text)[0] for text in DESCRIPTIONS}
    for negative_type in ["added_action", "swapped_action"]:
        for text in generator.generate(0, negative_type, 4, random.Random(6)):
            assert not (set(split_sentences(text)[1:]) & openings)


def test_clause_reorder_moves_capitalization_to_the_new_start():
    rng = random.Random(0)
    results = {reorder_clauses("Opens a box, then lifts it.", rng) for _ in range(20)}
    assert results == {"Opens a box, then lifts it.", "Then lifts it, opens a box."}


def test_added_and_swapped_actions_use_another_description():
    generator = NegativeGenerator(DESCRIPTIONS, LABELS)
    own = split_sentences(DESCRIPTIONS[0])
    for text in generator.generate(0, "added_action", 4, random.Random(2)):
        sentences = split_sentences(text)
        assert sentences[0] == own[0] and len(sentences) == len(own) + 1
        assert set(own) < set(sentences)
    for text in generator.generate(0, "swapped_action", 4, random.Random(2)):
        sentences = split_sentences(text)
        assert len(sentences) == len(own) and len(set(own) & set(sentences)) == len(own) - 1


def test_single_sentence_cannot_swap_an_action():
    generator = NegativeGenerator(["One sentence only.", "Another one."], [["a"], ["b"]])
    assert generator.generate(0, "swapped_action", 1, random.Random(3)) == []


def test_similar_episode_uses_action_label_overlap():
    descriptions = [f"Description {i}." for i in range(40)]
    labels = [["open", "close"]] * 21 + [["pour"]] * 19
    generator = NegativeGenerator(descriptions, labels)
    for text in generator.generate(0, "similar_episode", 4, random.Random(4)):
        assert labels[descriptions.index(text)] == ["open", "close"]


def test_candidate_positions_vary_and_the_correct_id_points_to_the_description():
    positions = set()
    for seed in range(40):
        candidate = candidate_set("right", ["a", "b", "c", "d"], random.Random(seed))
        chosen = next(c for c in candidate["choices"] if c["id"] == candidate["correct"])
        assert chosen["text"] == "right"
        positions.add(candidate["correct"])
    assert len(positions) == 5


def test_validation_sets_have_five_unique_candidates_and_are_fixed():
    first = build_validation_sets(IDS, DESCRIPTIONS, LABELS, seed=5)
    assert first == build_validation_sets(IDS, DESCRIPTIONS, LABELS, seed=5)
    for entry in first:
        assert set(entry["sets"]) == {*NEGATIVE_TYPES, "mixed"}
        for candidate in entry["sets"].values():
            assert len({choice["text"] for choice in candidate["choices"]}) == 5


def test_rank_score_matches_the_competition_formula():
    ranking = ["a", "b", "c", "d", "e"]
    assert [rank_score(ranking, item) for item in ranking] == [1, 0.5, 0.25, 0.125, 0]
    assert RANDOM_SCORE == 0.375


def test_known_scorers_give_their_expected_results():
    result = score_known_scorers(build_validation_sets(IDS, DESCRIPTIONS, LABELS, seed=5))
    assert result["oracle"] == 1 and result["worst"] == 0
    assert result["random"] == pytest.approx(0.375, abs=4 * 0.36 / result["sets"] ** 0.5)


class FakePredictor:
    def __init__(self, order_matters):
        self.order_matters = order_matters

    def score(self, images, choices):
        return [images[0] / 10 if self.order_matters else sum(images) / 100 for _ in choices]


class FakeEpisode:
    description = "text"


def test_order_check_detects_a_model_that_ignores_frame_order(monkeypatch):
    monkeypatch.setattr("team.rank_eval.load_images", lambda episode: _FakeImages([9, 1, 3]))
    episodes = [FakeEpisode() for _ in range(5)]
    assert order_check(FakePredictor(False), episodes, 5)["order_blind"]
    sensitive = order_check(FakePredictor(True), episodes, 5)
    assert not sensitive["order_blind"] and sensitive["prefers_forward"] == 1


class _FakeImages(list):
    """Frames stand in as numbers; order_check only closes and reverses them."""

    def __iter__(self):
        return iter([_Frame(value) for value in list.__iter__(self)])


class _Frame(int):
    def close(self):
        pass
