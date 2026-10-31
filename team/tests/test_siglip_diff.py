"""Check difference-focused matching without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from submission_siglip_diff import difference_phrases, pairwise_margins, ranks


def test_a_replaced_word_gives_a_phrase_on_each_side_with_context():
    first, second = difference_phrases("a man opens the box today".split(), "a man closes the box today".split(), 1)
    assert first == ["man opens the"] and second == ["man closes the"]


def test_an_added_action_still_has_context_on_the_shorter_side():
    first, second = difference_phrases("he buttons his shirt.".split(), "he buttons his shirt and sits.".split(), 2)
    assert first == ["his shirt."] and second == ["his shirt and sits."]


def test_identical_candidates_have_no_differences():
    assert difference_phrases(["same"], ["same"]) == ([], [])


def test_the_word_the_video_supports_wins_even_with_many_shared_words():
    shared = "a person in a room slowly"
    texts = [f"{shared} opens the box", f"{shared} closes the box", f"{shared} lifts the box"]
    match = lambda phrases: [1.0 if "opens" in phrase else 0.0 for phrase in phrases]
    margins = pairwise_margins(texts, match)
    assert margins[0] > margins[1] and margins[0] > margins[2]
    assert abs(sum(margins)) < 1e-9


def test_margins_for_a_single_candidate_are_zero():
    assert pairwise_margins(["only one"], lambda phrases: [1.0] * len(phrases)) == [0.0]


def test_ranks_share_ties():
    assert ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2]
