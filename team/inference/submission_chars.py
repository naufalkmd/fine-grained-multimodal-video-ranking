"""How familiar a text's characters are from the training descriptions, case and punctuation kept.

A Witten-Bell interpolated character n-gram model, counted by `build_char_counts` from the training
descriptions. Word-pair familiarity (team/inference/submission_style.py) lowercases and drops punctuation; this keeps
writing habits that words alone miss. Used by team/inference/submission_text_rules.py when `chars` is set.
"""

import math
import statistics
from collections import Counter, defaultdict

ORDER = 5
START, END = "\x02", "\x03"


def build_char_counts(descriptions, order=ORDER):
    """Counts of the character after each history of up to order - 1 characters, as a JSON-ready dictionary."""
    counts = defaultdict(Counter)
    for text in descriptions:
        padded = START * (order - 1) + text + END
        for i in range(order - 1, len(padded)):
            for length in range(order):
                counts[padded[i - length:i]][padded[i]] += 1
    return {"order": order,
            "counts": {history: {"n": sum(c.values()), "c": dict(c)} for history, c in counts.items()}}


class CharModel:
    def __init__(self, data):
        self.order, self.counts = data["order"], data["counts"]
        self.alphabet = len(self.counts.get("", {}).get("c", {})) + 1

    def probability(self, history, char):
        lower = 1.0 / self.alphabet if not history else self.probability(history[1:], char)
        entry = self.counts.get(history)
        if not entry:
            return lower
        types = len(entry["c"])
        return (entry["c"].get(char, 0) + types * lower) / (entry["n"] + types)

    def familiarity(self, text):
        """Mean log-probability per character."""
        padded = START * (self.order - 1) + text + END
        return statistics.fmean(math.log(self.probability(padded[i - self.order + 1:i], padded[i]))
                                for i in range(self.order - 1, len(padded)))
