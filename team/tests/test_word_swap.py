"""Check word-swap negatives without any models or real data."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team.candidates.negatives import NegativeGenerator
from team.candidates.word_swap import replacement_table, swap_one_word

PEOPLE = ["man", "woman", "child"]
VERBS = ["opens", "closes", "lifts"]
OBJECTS = ["box", "door", "lid"]
DESCRIPTIONS = [f"A {person} {verb} the {thing} slowly."
                for person in PEOPLE for verb in VERBS for thing in OBJECTS]
DESCRIPTIONS += [f"Filler{i} words{i} here{i}." for i in range(250)]


def test_words_in_the_same_contexts_become_replacements():
    table = replacement_table(DESCRIPTIONS)
    assert set(table["opens"]) >= {"closes", "lifts"}
    assert "box" not in table["opens"]


def test_very_common_words_are_not_replaced():
    assert "the" not in replacement_table(DESCRIPTIONS)


def test_swap_changes_exactly_one_word_and_keeps_the_rest():
    table = replacement_table(DESCRIPTIONS)
    text = "A man opens the box slowly."
    swapped = swap_one_word(text, table, random.Random(3))
    differences = [(a, b) for a, b in zip(text.split(), swapped.split()) if a != b]
    assert len(text.split()) == len(swapped.split()) and len(differences) == 1


def test_swap_returns_none_when_no_word_has_a_replacement():
    assert swap_one_word("Nothing known here.", {}, random.Random(0)) is None


def test_generator_gives_reproducible_distinct_word_swaps():
    generator = NegativeGenerator(DESCRIPTIONS, [["a"]] * len(DESCRIPTIONS))
    first = generator.generate(0, "word_swap", 4, random.Random(1))
    assert first == generator.generate(0, "word_swap", 4, random.Random(1))
    assert len(first) == 4 == len(set(first)) and DESCRIPTIONS[0] not in first
