"""Check the fill-in-the-blank margins without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

from submission_fill import margins, pair_margin, ranks


def test_shared_tokens_do_not_count():
    first = ([1, 2, 3, 4], [9.0, 9.0, 1.0, 9.0])
    second = ([1, 2, 5, 4], [-9.0, -9.0, 3.0, -9.0])
    assert pair_margin(first, second) == 1.0 - 3.0


def test_added_tokens_count_for_the_candidate_that_has_them():
    first = ([1, 2], [0.5, 0.5])
    second = ([1, 2, 3], [0.5, 0.5, 2.0])
    assert pair_margin(first, second) == -2.0


def test_margins_sum_to_zero_and_favour_the_supported_word():
    values = margins([([1, 2], [0.0, 3.0]), ([1, 3], [0.0, 1.0]), ([1, 4], [0.0, -1.0])])
    assert values[0] > values[1] > values[2]
    assert abs(sum(values)) < 1e-9


def test_unfamiliar_favours_the_least_expected_word_and_ignores_length():
    from submission_unfamiliar import pair_margin as mean_margin, unfamiliar_scores
    values = unfamiliar_scores([([1, 2], [0.0, -1.0]), ([1, 3], [0.0, -5.0]), ([1, 4], [0.0, -3.0])])
    assert values[1] > values[2] > values[0]
    assert mean_margin(([1, 2], [0.0, -2.0]), ([1, 3, 4, 5], [0.0, -2.0, -2.0, -2.0])) == 0.0


def test_ranks_share_ties():
    assert ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2]
