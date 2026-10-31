"""Check the text-only centrality submission without any models or data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.submission_central import Submission


def test_the_one_edited_text_ranks_first_and_every_id_is_returned():
    choices = [
        {"id": "a", "text": "A man is unbuttoning his shirt in front of the sofa."},
        {"id": "b", "text": "A man is buttoning his shirt in front of the sofa."},
        {"id": "c", "text": "A man is buttoning his shirt in front of the sofa and then sits down."},
        {"id": "d", "text": "A man is buttoning his coat in front of the sofa."},
        {"id": "e", "text": "A woman is buttoning his shirt in front of the sofa."},
    ]
    ranking = Submission(".", "cpu").predict([], choices)
    assert ranking[0] == "b"
    assert sorted(ranking) == list("abcde")


def test_a_single_choice_is_returned_as_is():
    assert Submission(".", "cpu").predict([], [{"id": "x", "text": "Anything."}]) == ["x"]
