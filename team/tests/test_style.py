"""Check the writing-style rule without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "team"))

from submission_style import build_counts, combined_scores, ranks, style_scores

TRAIN = ["a person opens the box", "a person closes the box", "a person lifts the cup", "they open the door"] * 5


def test_unusual_word_pairs_score_lower_than_familiar_ones():
    scores = style_scores(["a person opens the box", "a person opens the cup box"], build_counts(TRAIN))
    assert scores[0] > scores[1]


def test_unknown_words_stay_finite_and_score_low():
    scores = style_scores(["a person opens the box", "zebra quantum foo"], build_counts(TRAIN))
    assert scores[0] > scores[1] > float("-inf")


def test_style_is_per_word_so_length_alone_does_not_win():
    counts = build_counts(["a b c d e f"] * 10)
    short, long = style_scores(["a b c", "a b c d e f"], counts)
    assert abs(short - long) < 1.0


def test_central_weight_adds_similarity_rank():
    counts = build_counts(TRAIN)
    texts = ["a person opens the box", "a person opens the door", "they lift a cup"]
    combined = combined_scores(texts, counts, 3)
    assert combined == [a + 3 * b for a, b in zip(ranks(style_scores(texts, counts)), ranks(combined_scores_similarity(texts)))]


def test_negative_sign_reverses_the_style_order():
    counts = build_counts(TRAIN)
    texts = ["a person opens the box", "a person opens the cup box", "zebra quantum foo"]
    assert combined_scores(texts, counts, 0, -1) == [-value for value in style_scores(texts, counts)]


def combined_scores_similarity(texts):
    from submission_style import similarity_scores
    return similarity_scores(texts)
