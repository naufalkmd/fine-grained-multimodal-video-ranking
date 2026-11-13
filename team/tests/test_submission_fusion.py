"""Check the configurable text-signal submission without any models or data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

from team.export.export_fusion import char_counts
from team.inference.submission_fusion import Submission

CHOICES = [
    {"id": "a", "text": "A man is unbuttoning his shirt in front of the sofa."},
    {"id": "b", "text": "A man is buttoning his shirt in front of the sofa."},
    {"id": "c", "text": "A man is buttoning his shirt in front of the sofa and then sits down."},
    {"id": "d", "text": "A man is buttoning his coat in front of the sofa."},
    {"id": "e", "text": "a man buttoning shirt by sofa"},
]


def make(tmp_path, rule, descriptions=("A man is unbuttoning his shirt in front of the sofa.",)):
    (tmp_path / "weights").mkdir(parents=True)
    (tmp_path / "weights/rule.json").write_text(json.dumps(rule), encoding="utf-8")
    (tmp_path / "weights/char_counts.json").write_text(
        json.dumps({"order": 3, "counts": char_counts(descriptions, 3)}), encoding="utf-8")
    return Submission(str(tmp_path), "cpu")


def test_single_signals_rank_as_described(tmp_path):
    shortest = make(tmp_path, {"signals": ["short"], "method": "single"}).predict([], CHOICES)
    assert shortest[0] == "e" and shortest[-1] == "c" and sorted(shortest) == list("abcde")


def test_character_familiarity_prefers_the_counted_writing(tmp_path):
    submission = make(tmp_path, {"signals": ["unfamiliar_chars"], "method": "single"})
    familiar = submission.chars.mean_log_prob(CHOICES[0]["text"])
    assert familiar > submission.chars.mean_log_prob(CHOICES[4]["text"])
    assert submission.predict([], CHOICES)[-1] == "a"


def test_interleave_and_bayes_return_every_candidate_once(tmp_path):
    rule = {"signals": ["centrality", "short"], "method": "interleave"}
    ranking = make(tmp_path, rule).predict([], CHOICES)
    assert ranking[:2] == ["b", "e"] and sorted(ranking) == list("abcde")
    shares = {"centrality": [0.4, 0.2, 0.2, 0.1, 0.1], "short": [0.2] * 5}
    bayes = make(tmp_path / "x", {"signals": ["centrality", "short"], "method": "bayes",
                                  "rank_shares": shares})
    assert bayes.predict([], CHOICES)[0] == "b"
