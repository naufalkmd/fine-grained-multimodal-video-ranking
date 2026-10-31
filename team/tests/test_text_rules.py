"""Check the text rules without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "team"))

from submission_style import build_counts, descending, interleaved, style_scores
from submission_text_rules import Submission, consensus_scores, ranks, sequence_scores

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


def merged_submission(merge="interleave"):
    submission = object.__new__(Submission)
    submission.rule = "consensus"
    submission.merge = merge
    submission.counts = build_counts(["he buttons his shirt", "he buttons his jacket slowly"] * 3)
    return submission


def test_merge_takes_the_rule_and_the_least_familiar_in_turn():
    submission = merged_submission()
    rule_order = descending(consensus_scores(EDITS))
    unfamiliar = descending([-value for value in style_scores(EDITS, submission.counts)])
    assert submission._order(EDITS) == interleaved(rule_order, unfamiliar)
    assert submission._order(EDITS)[0] == rule_order[0]


def test_without_a_merge_the_order_is_the_rule_alone():
    assert merged_submission("none")._order(EDITS) == descending(consensus_scores(EDITS))


def test_merged_scores_follow_the_ranking():
    submission = merged_submission()
    choices = [{"id": f"c{i}", "text": text} for i, text in enumerate(EDITS)]
    scores = submission.score([], choices)
    assert all(0 <= score <= 1 for score in scores)
    assert [choices[i]["id"] for i in descending(scores)] == submission.predict([], choices)


def test_ranks_share_ties():
    assert ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2]
