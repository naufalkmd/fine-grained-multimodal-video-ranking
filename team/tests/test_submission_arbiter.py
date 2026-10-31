"""Check when the video reorders the two text rules' top picks, without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from submission_arbiter import Submission
from submission_style import build_counts, descending, interleaved, style_scores

TEXTS = ["a person lifts the red cup and sets it down", "a person lifts the red cup and sets it up",
         "a person lifts the red box and sets it down", "a person lifts the red cup and walks away",
         "zebra quantum lifts the red cup and sets it down"]
COUNTS = build_counts(["a person lifts the red cup and sets it down", "a person sets the box down"] * 5)
RULE_ORDER = [1, 0, 2, 3, 4]


class FakeScorer:
    def __init__(self, margin):
        self.margin = margin
        self.calls = []

    def token_scores(self, images, texts):
        self.calls.append(texts)
        return [([1, 2], [0.0, 0.0]), ([1, 3], [0.0, self.margin])]


class FakeRules:
    """The rule's order merged in turn with the least familiar wording, as team/submission_text_rules.py does."""

    counts = COUNTS

    def __init__(self, rule_order):
        self.rule_order = rule_order

    def _order(self, texts):
        return interleaved(self.rule_order, descending([-value for value in style_scores(texts, COUNTS)]))


def predictor(margin, threshold=3.0, rule_order=None):
    submission = object.__new__(Submission)
    submission.threshold = threshold
    submission.rules = FakeRules(rule_order or RULE_ORDER)
    submission.scorer = FakeScorer(margin)
    return submission


def unfamiliar_order():
    return descending([-value for value in style_scores(TEXTS, COUNTS)])


def test_the_picks_differ_in_this_example():
    assert RULE_ORDER[0] != unfamiliar_order()[0]


def test_a_margin_at_or_below_the_threshold_keeps_the_rule_first():
    assert predictor(3.0)._order([], TEXTS) == interleaved(RULE_ORDER, unfamiliar_order())


def test_a_margin_above_the_threshold_only_swaps_the_top_two():
    expected = interleaved(RULE_ORDER, unfamiliar_order())
    expected[0], expected[1] = expected[1], expected[0]
    assert predictor(3.5)._order([], TEXTS) == expected


def test_the_model_compares_the_rule_pick_with_the_least_familiar():
    submission = predictor(0.0)
    submission._order([], TEXTS)
    assert submission.scorer.calls == [[TEXTS[RULE_ORDER[0]], TEXTS[unfamiliar_order()[0]]]]


def test_no_model_call_when_both_rules_pick_the_same_text():
    least = unfamiliar_order()[0]
    submission = predictor(5.0, rule_order=[least] + [i for i in range(5) if i != least])
    order = submission._order([], TEXTS)
    assert order[0] == least
    assert submission.scorer.calls == []


def test_predict_returns_every_id_once():
    choices = [{"id": f"c{i}", "text": text} for i, text in enumerate(TEXTS)]
    assert sorted(predictor(0.0).predict([], choices)) == sorted(c["id"] for c in choices)


def test_scores_follow_the_ranking():
    choices = [{"id": f"c{i}", "text": text} for i, text in enumerate(TEXTS)]
    submission = predictor(0.0)
    scores = submission.score([], choices)
    assert all(0 <= score <= 1 for score in scores)
    assert [choices[i]["id"] for i in descending(scores)] == submission.predict([], choices)
