"""Check shortlist ordering without any models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

from submission_shortlist import shortlist_order


def test_evidence_decides_first_place_within_the_shortlist():
    central = [0.9, 0.8, 0.7, 0.2, 0.1]
    evidence = [0.0, 5.0, 1.0, 9.0, 9.0]
    assert shortlist_order(central, evidence, 3) == [1, 2, 0, 3, 4]


def test_a_strong_evidence_score_outside_the_shortlist_cannot_win():
    assert shortlist_order([0.9, 0.8, 0.1], [0.0, 1.0, 99.0], 2)[0] == 1


def test_a_shortlist_of_one_keeps_the_centrality_order():
    assert shortlist_order([0.1, 0.9, 0.5], [9.0, 0.0, 5.0], 1) == [1, 2, 0]


def test_a_shortlist_of_all_uses_evidence_alone():
    assert shortlist_order([0.1, 0.9, 0.5], [3.0, 1.0, 2.0], 3) == [0, 2, 1]
