"""Check mixed training negatives without any models or real data."""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.candidates.negatives import NegativeGenerator
from team.training.train_negatives import TrainingNegatives, parse_mix, sample_negatives

DESCRIPTIONS = [
    f"Person {i} opens box {i % 7}, then lifts it. They move cup {i % 5}. Finally, they wave."
    for i in range(40)
]
LABELS = [["open", "close"] if i % 2 else ["show", "pour"] for i in range(40)]
IDS = [f"episode_{i}" for i in range(40)]


def test_parse_mix():
    assert parse_mix("lexical=2, similar_episode=1") == {"lexical": 2, "similar_episode": 1}
    assert parse_mix("") == {}
    for bad in ["nonsense=1", "lexical=1,lexical=2", "lexical=-1"]:
        with pytest.raises(ValueError):
            parse_mix(bad)


def test_counts_follow_the_mix_and_fill_the_rest():
    generator = NegativeGenerator(DESCRIPTIONS, LABELS)
    for index in range(len(DESCRIPTIONS)):
        result = sample_negatives(generator, index, {"lexical": 2, "similar_episode": 1}, 4,
                                  "random", random.Random(index))
        texts = [text for _, text in result]
        assert len(texts) == 4 == len(set(texts))
        assert DESCRIPTIONS[index] not in texts
        assert Counter(t for t, _ in result) == {"lexical": 2, "similar_episode": 1, "random": 1}


def test_a_short_type_is_topped_up_from_fill():
    descriptions = ["One sentence only.", "Another one.", "A third.", "A fourth.", "A fifth."]
    generator = NegativeGenerator(descriptions, [["a"]] * 5)
    result = sample_negatives(generator, 0, {"reorder": 2}, 3, "random", random.Random(0))
    assert Counter(t for t, _ in result) == {"random": 3}


def test_epochs_are_reproducible_and_resampled():
    sampler = TrainingNegatives(IDS, DESCRIPTIONS, LABELS, {"lexical": 2, "similar_episode": 2}, 4)
    first = sampler.epoch(0)
    assert first == sampler.epoch(0)
    assert first != sampler.epoch(1)


def test_rejects_a_mix_larger_than_k():
    with pytest.raises(ValueError):
        TrainingNegatives(IDS, DESCRIPTIONS, LABELS, {"lexical": 3, "similar_episode": 2}, 4)
