"""Server API v2 submission: text-only signals, combined as set in weights/rule.json.

Signals, each scored so that higher means more likely correct:

- `centrality`: mean word-sequence similarity to the other candidates (team/inference/submission_central.py).
- `unfamiliar_words`: minus the mean bigram log-probability per word under the training
  descriptions. Publicly the most familiar wording scored far below chance.
- `unfamiliar_chars`: the same with a character model that keeps case and punctuation, so writing
  habits that words alone miss also count.
- `short`: fewer words.

Combinations (`method`):

- `single`: rank by `signals[0]`.
- `interleave`: the first remaining choice of each signal in turn, in the order given.
- `bayes`: rank by the product over signals of how likely the correct text is to have this
  candidate's rank under that signal, relative to a wrong one. `rank_shares[signal]` holds the share
  of public episodes with the correct text at ranks 1-5.

With `"gate": true`, centrality goes first whenever some candidate has no word missing from every
other candidate (the others look like edits of it). Bigram and character counts come from the
training descriptions only. Each episode is scored on its own candidate texts, with no state kept.
"""

import difflib
import json
import math
import re
import statistics
from pathlib import Path

from submission_central import similarity_matrix

WORD = re.compile(r"[a-z0-9']+")


def words(text):
    return WORD.findall(text.lower())


class CharModel:
    """Witten-Bell interpolated character n-gram model over raw text."""

    def __init__(self, counts, order):
        self.counts, self.order = counts, order
        self.alphabet = max(len(counts.get("", {}).get("c", {})), 1) + 1

    def probability(self, history, char):
        if history is None:
            return 1.0 / self.alphabet
        lower = self.probability(history[1:] if history else None, char)
        entry = self.counts.get(history)
        if not entry:
            return lower
        total, types = entry["n"], len(entry["c"])
        return (entry["c"].get(char, 0) + types * lower) / (total + types)

    def mean_log_prob(self, text):
        padded = "\x02" * (self.order - 1) + text + "\x03"
        values = [math.log(self.probability(padded[i - self.order + 1:i], padded[i]))
                  for i in range(self.order - 1, len(padded))]
        return statistics.fmean(values)


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir) / "weights"
        self.rule = json.loads((root / "rule.json").read_text(encoding="utf-8"))
        signals = set(self.rule["signals"])
        if "unfamiliar_words" in signals:
            counts = json.loads((root / "bigram_counts.json").read_text(encoding="utf-8"))
            self.unigrams, self.bigrams = counts["unigrams"], counts["bigrams"]
            self.total = sum(self.unigrams.values())
            self.vocabulary = len(self.unigrams) + 1
            self.prior = counts["prior"]
        if "unfamiliar_chars" in signals:
            counts = json.loads((root / "char_counts.json").read_text(encoding="utf-8"))
            self.chars = CharModel(counts["counts"], counts["order"])

    def word_familiarity(self, text):
        tokens = words(text)
        previous = ["<s>", *tokens[:-1]]
        return statistics.fmean(
            math.log((self.bigrams.get(f"{a} {b}", 0)
                      + self.prior * (self.unigrams.get(b, 0) + 1) / (self.total + self.vocabulary))
                     / (self.unigrams.get(a, 0) + self.prior))
            for a, b in zip(previous, tokens)) if tokens else 0.0

    def signal(self, name, texts, central):
        if name == "centrality":
            return central
        if name == "unfamiliar_words":
            return [-self.word_familiarity(text) for text in texts]
        if name == "unfamiliar_chars":
            return [-self.chars.mean_log_prob(text) for text in texts]
        if name == "short":
            return [-len(words(text)) for text in texts]
        raise ValueError(f"Unknown signal: {name}")

    @staticmethod
    def unique_words(texts):
        tokens = [words(text) for text in texts]
        counts = []
        for i, mine in enumerate(tokens):
            matched = [False] * len(mine)
            for j, theirs in enumerate(tokens):
                if j != i:
                    for block in difflib.SequenceMatcher(a=mine, b=theirs,
                                                         autojunk=False).get_matching_blocks():
                        matched[block.a:block.a + block.size] = [True] * block.size
            counts.append(matched.count(False))
        return counts

    def ranking(self, texts):
        count = len(texts)
        similarity = similarity_matrix(texts)
        central = [statistics.fmean(row[j] for j in range(count) if j != i)
                   for i, row in enumerate(similarity)]
        names = list(self.rule["signals"])
        if self.rule.get("gate") and 0 in self.unique_words(texts):
            names = ["centrality", *[n for n in names if n != "centrality"]]
        # Ties are broken by centrality, so every order is total and deterministic.
        orders = {name: sorted(range(count), key=lambda i, s=self.signal(name, texts, central):
                               (-s[i], -central[i])) for name in names}
        method = self.rule["method"]
        if method == "single":
            return orders[names[0]]
        if method == "interleave":
            ranking = []
            while len(ranking) < count:
                for name in names:
                    choice = next(i for i in orders[name] if i not in ranking)
                    ranking.append(choice)
                    if len(ranking) == count:
                        break
            return ranking
        if method == "bayes":
            evidence = [0.0] * count
            for name in names:
                shares = self.rule["rank_shares"][name]
                for rank, index in enumerate(orders[name][:5]):
                    correct = max(shares[rank], 1e-3)
                    wrong = max((1 - shares[rank]) / 4, 1e-3)
                    evidence[index] += math.log(correct / wrong)
            return sorted(range(count), key=lambda i: (-evidence[i], -central[i]))
        raise ValueError(f"Unknown method: {method}")

    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty.")
        if len(choices) == 1:
            return [1.0]
        ranking = self.ranking([choice["text"] for choice in choices])
        return [1.0 - ranking.index(i) / (len(choices) - 1) for i in range(len(choices))]

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
