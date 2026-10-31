"""Check the code rules that accept or reject generated candidates, without any model."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.qwen_negatives import check, checked_record, parse

POSITIVE = "A person folds a towel and places it on the table."


def test_single_word_changes_pass_and_bigger_rewrites_fail():
    assert check("reversed_action", POSITIVE, "A person unfolds a towel and places it on the table.") == "ok"
    assert check("changed_detail", POSITIVE, "A person folds a shirt and places it on the table.") == "ok"
    assert check("reversed_action", POSITIVE, "Someone unfolds a blue towel and drops it on the floor.") != "ok"
    assert check("reversed_action", POSITIVE, POSITIVE) == "same_as_positive"
    assert check("changed_detail", POSITIVE, None) == "missing"


def test_added_action_must_extend_the_positive():
    assert check("added_action", POSITIVE,
                 "A person folds a towel and places it on the table and then sits down.") == "ok"
    assert check("added_action", POSITIVE, "A person folds a towel and then sits down.") != "ok"
    assert check("added_action", POSITIVE, POSITIVE + " Then") != "ok"


def test_swapped_order_must_reuse_the_same_words():
    assert check("swapped_order", POSITIVE,
                 "A person places a towel on the table and then folds it.") == "ok"
    assert check("swapped_order", POSITIVE, "A child throws a red ball against the wall.") != "ok"


def test_records_keep_only_passing_distinct_negatives():
    raw = "Sure:\n" + json.dumps({
        "positive": POSITIVE,
        "reversed_action": "A person unfolds a towel and places it on the table.",
        "added_action": "A person folds a towel.",
        "swapped_order": None,
        "changed_detail": "A person unfolds a towel and places it on the table.",
    })
    episode = SimpleNamespace(episode_id="episode_1", description="Long original text.")
    record = checked_record(episode, "valid", raw)
    assert record["positive"] == POSITIVE
    assert set(record["negatives"]) == {"reversed_action"}
    assert record["checks"]["changed_detail"] == "duplicate"
    assert record["checks"]["swapped_order"] == "missing"
    assert checked_record(episode, "valid", "no json here")["checks"] == {"positive": "unparsed"}


def test_parse_finds_the_json_object():
    assert parse('text {"positive": "x"} more') == {"positive": "x"}
    assert parse("[1, 2]") is None
