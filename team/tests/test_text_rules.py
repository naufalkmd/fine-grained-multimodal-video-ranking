"""Check the text rules without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "team"))

from submission_text_rules import consensus_scores, ranks, sequence_scores

EDITS = ["he buttons his shirt", "he unbuttons his shirt", "he buttons his shirt and sits",
         "he buttons his jacket", "he buttons his shirt slowly"]


def test_consensus_prefers_the_text_the_others_are_edits_of():
    scores = consensus_scores(EDITS)
    assert scores.index(max(scores)) == 0


def test_consensus_does_not_reward_length():
    scores = consensus_scores(["a b", "a b c d e f", "a b"])
    assert scores[0] > scores[1]


def test_sequence_scores_prefer_the_central_text():
    scores = sequence_scores(EDITS)
    assert scores.index(max(scores)) == 0


def test_ranks_share_ties():
    assert ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2]
