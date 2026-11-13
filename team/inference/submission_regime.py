"""Server API v2 submission: decide from the candidates' structure whether to trust centrality.

If some candidate has no word that is missing from every other candidate (in a word-level
alignment), the others look like edits of it, and centrality almost always picks the correct text
on such sets. The ranking then takes centrality's first choice, `unfamiliar`'s first choice, and so
on in turn. Otherwise centrality is close to chance, and `unfamiliar` goes first. `unfamiliar`
prefers wording that the training descriptions make less likely (bigram counts stored in
weights/bigram_counts.json), because publicly the most familiar wording scored far below chance.
With `"gate": false` in weights/bigram_counts.json, centrality always goes first (plain
interleaving). Reads only the episode's candidate texts, with no state kept between episodes.
"""

import difflib
import json
import math
import re
import statistics
from pathlib import Path

from submission_central import similarity_matrix


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        counts = json.loads((Path(artifact_dir) / "weights/bigram_counts.json")
                            .read_text(encoding="utf-8"))
        self.unigrams = counts["unigrams"]
        self.bigrams = counts["bigrams"]
        self.total = sum(self.unigrams.values())
        self.vocabulary = len(self.unigrams) + 1
        self.prior = counts["prior"]
        self.gate = counts.get("gate", True)

    def familiarity(self, text):
        """Mean log p(word | previous word) under the training descriptions."""
        tokens = words(text)
        previous = ["<s>", *tokens[:-1]]
        return statistics.fmean(
            math.log((self.bigrams.get(f"{a} {b}", 0)
                      + self.prior * (self.unigrams.get(b, 0) + 1) / (self.total + self.vocabulary))
                     / (self.unigrams.get(a, 0) + self.prior))
            for a, b in zip(previous, tokens)) if tokens else 0.0

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
        unfamiliar = [-self.familiarity(text) for text in texts]
        by_central = sorted(range(count), key=lambda i: -central[i])
        by_unfamiliar = sorted(range(count), key=lambda i: (-unfamiliar[i], -central[i]))
        trust_centrality = not self.gate or 0 in self.unique_words(texts)
        first, second = ((by_central, by_unfamiliar) if trust_centrality
                         else (by_unfamiliar, by_central))
        ranking = []
        for pair in zip(first, second):
            for index in pair:
                if index not in ranking:
                    ranking.append(index)
        return ranking

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
