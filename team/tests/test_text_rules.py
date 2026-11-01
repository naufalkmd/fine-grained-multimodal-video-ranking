"""Check the text rules without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "team"))

from submission_style import build_counts, descending, interleaved, style_scores
from submission_text_rules import (Submission, consensus_scores, lowest_excess, own_word_scores, ranks, sequence_scores,
                                   standardized, unique_words)

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


BASE = "a b c d e f g h i j k l"
FROM_BASE = [BASE, "a x c d e f g h i j k l", "a b c x e f g h i j k l", "a b c d e f x h i j k l", "a b c d e f g h i x k l"]
FROM_HIDDEN = ["a p c d e f g h i j k l", "a b c q e f g h i j k l", "a b c d e f r h i j k l",
               "a b c d e f g h i s k l", "a b c d t f g h i j k l"]
REORDERED = ["a b c d e f g h i j k l", "d e f a b c g h i j k l", "a b c g h i d e f j k l",
             "a b c d e f j k l g h i", "d e f a b c j k l g h i"]


def gated_submission(gate_excess=None):
    submission = merged_submission()
    submission.gate = True
    submission.gate_excess = gate_excess
    return submission


def test_excess_separates_a_base_from_a_hidden_original():
    assert lowest_excess(FROM_BASE) < 0.1
    assert lowest_excess(FROM_HIDDEN) > 0.3


def test_every_candidate_of_an_order_change_has_no_unique_words():
    assert unique_words(REORDERED) == [0, 0, 0, 0, 0]


def test_gate_leaves_a_base_alone():
    assert not gated_submission()._no_base_candidate(FROM_BASE)
    assert not gated_submission(0.3)._no_base_candidate(FROM_BASE)


def test_gate_fires_when_every_candidate_has_a_unique_word():
    assert gated_submission()._no_base_candidate(FROM_HIDDEN)


def test_only_the_excess_gate_fires_on_order_changes():
    assert not gated_submission()._no_base_candidate(REORDERED)
    assert gated_submission(0.3)._no_base_candidate(REORDERED)


def test_a_fired_gate_puts_the_least_familiar_first():
    submission = gated_submission()
    unfamiliar = descending([-value for value in style_scores(FROM_HIDDEN, submission.counts)])
    assert submission._order(FROM_HIDDEN)[0] == unfamiliar[0]


def test_own_word_scores_ignore_the_shared_wording():
    counts = build_counts(["a y c d"] * 5 + ["a b c d"] * 5)
    shared = "a b c d a b c d "
    scores = own_word_scores([shared + "a y c d", shared + "a z c d"], counts)
    assert scores[1] < scores[0]


def test_a_fired_gate_can_take_its_first_pick_from_own_words():
    submission = gated_submission()
    submission.gate_own_words = True
    own = descending([-value for value in own_word_scores(FROM_HIDDEN, submission.counts)])
    assert submission._order(FROM_HIDDEN)[0] == own[0]


def test_own_words_change_nothing_when_the_gate_does_not_fire():
    submission = gated_submission()
    plain = submission._order(FROM_BASE)
    submission.gate_own_words = True
    assert submission._order(FROM_BASE) == plain


def test_a_fired_gate_can_average_both_measures():
    submission = gated_submission()
    submission.gate_mean = True
    whole = [-value for value in style_scores(FROM_HIDDEN, submission.counts)]
    own = [-value for value in own_word_scores(FROM_HIDDEN, submission.counts)]
    mean = [a + b for a, b in zip(standardized(whole), standardized(own))]
    assert submission._order(FROM_HIDDEN)[0] == descending(mean)[0]


def test_the_mean_changes_nothing_when_the_gate_does_not_fire():
    submission = gated_submission()
    plain = submission._order(FROM_BASE)
    submission.gate_mean = True
    assert submission._order(FROM_BASE) == plain


def test_base_first_moves_a_text_with_no_own_words_ahead_of_the_rule():
    submission = gated_submission()
    submission.gate_base_first = True
    texts = [BASE, "a p c d e f g h i j k l", "a b c q e f g h i j k l", "a b c d e f r h i j k l", "a b c d e f g h i s k l"]
    submission._values = lambda texts: [0.0, 4.0, 3.0, 2.0, 1.0]
    assert unique_words(texts) == [0, 1, 1, 1, 1]
    assert submission._order(texts)[0] == 0


def test_ranks_share_ties():
    assert ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2]
