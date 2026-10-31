"""Check rank combination without any models."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import submission_combined
from submission_combined import Submission, ranks

CHOICES = [{"id": name, "text": name} for name in "abc"]


def combined(first, second):
    submission = Submission.__new__(Submission)
    submission.ranker = SimpleNamespace(_logits=lambda images, choices: torch.tensor(first))
    submission.siglip = SimpleNamespace(_matches=lambda images, choices: second)
    return submission


def test_ranks_order_scores_and_share_ties():
    assert ranks([0.1, 5.0, 2.0]) == [0, 2, 1]
    assert ranks([1.0, 1.0, 3.0]) == [0.5, 0.5, 2]


def test_a_large_score_scale_does_not_dominate():
    submission = combined([1000.0, 0.0, -1000.0], [0.1, 0.3, 0.2])
    assert submission.predict(None, CHOICES)[0] == "b"


def test_predict_returns_every_id_and_scores_stay_in_range():
    submission = combined([3.0, 2.0, 1.0], [0.3, 0.2, 0.1])
    assert submission.predict(None, CHOICES) == ["a", "b", "c"]
    assert submission.score(None, CHOICES) == [1.0, 0.5, 0.0]
