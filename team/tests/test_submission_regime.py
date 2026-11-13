"""Check the structure-gated text submission without any models or data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

from team.inference.submission_regime import Submission

SHIRT = "A man is buttoning his shirt in front of the sofa."


def make_submission(tmp_path, familiar_text):
    words = ["<s>", *familiar_text.lower().rstrip(".").split(), "</s>"]
    counts = {"prior": 1.0, "unigrams": {w: 10 for w in words},
              "bigrams": {f"{a} {b}": 10 for a, b in zip(words, words[1:])}}
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/bigram_counts.json").write_text(json.dumps(counts), encoding="utf-8")
    return Submission(str(tmp_path), "cpu")


def test_edits_of_one_text_keep_centrality_first(tmp_path):
    choices = [
        {"id": "a", "text": "A man is unbuttoning his shirt in front of the sofa."},
        {"id": "b", "text": SHIRT},
        {"id": "c", "text": "A man is buttoning his shirt in front of the sofa and then sits down."},
        {"id": "d", "text": "A man is buttoning his coat in front of the sofa."},
        {"id": "e", "text": "A woman is buttoning his shirt in front of the sofa."},
    ]
    submission = make_submission(tmp_path, "A man is unbuttoning his shirt in front of the sofa.")
    assert submission.unique_words([c["text"] for c in choices])[1] == 0
    ranking = submission.predict([], choices)
    assert ranking[0] == "b"
    assert sorted(ranking) == list("abcde")


def test_without_a_source_text_the_least_familiar_goes_first(tmp_path):
    choices = [
        {"id": "a", "text": "A man buttons a shirt near the sofa."},
        {"id": "b", "text": "Someone fastens the blue jacket by a window."},
        {"id": "c", "text": "A woman opens a drawer and takes out a cup."},
    ]
    submission = make_submission(tmp_path, "A man buttons a shirt near the sofa.")
    assert 0 not in submission.unique_words([c["text"] for c in choices])
    ranking = submission.predict([], choices)
    assert ranking[-1] != "a" and ranking[0] in "bc"
    assert sorted(ranking) == list("abc")


def test_without_the_gate_centrality_always_goes_first(tmp_path):
    choices = [
        {"id": "a", "text": "A man buttons a shirt near the sofa."},
        {"id": "b", "text": "A man buttons a coat near the sofa."},
        {"id": "c", "text": "Someone fastens the blue jacket by a window."},
    ]
    submission = make_submission(tmp_path, "A man buttons a shirt near the sofa.")
    submission.gate = False
    assert submission.predict([], choices)[0] in "ab"
