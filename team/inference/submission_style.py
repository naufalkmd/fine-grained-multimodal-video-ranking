"""Server API v2: favour the candidate that reads most like the organizers' training descriptions.

A word bigram model, counted from the training descriptions by build_counts(), scores each candidate
by its mean log probability per word. Unusual word combinations lower the score, which edits often
create. `style_sign` in inference_config.json set to -1 favours the least familiar wording instead:
this rule scored 0.2411 publicly, well below chance. `central_rank_weight` adds the rank from
candidate similarity. `merge` set to "interleave" instead takes the most central candidate, then the
top of the style ranking, and so on in turn. No images are used, and no state is kept between episodes.
"""

import difflib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

START, END = "<s>", "</s>"
UNIGRAM_SHARE = 0.2


def words(text):
    return [START, *re.findall(r"[a-z0-9']+", text.lower()), END]


def build_counts(descriptions):
    """Word and word-pair counts, as a JSON-ready dictionary."""
    unigrams, bigrams = Counter(), Counter()
    for text in descriptions:
        tokens = words(text)
        unigrams.update(tokens)
        bigrams.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return {"unigrams": dict(unigrams), "bigrams": dict(bigrams)}


def style_scores(texts, counts):
    """Mean log probability per word, mixing bigram and add-one unigram estimates."""
    unigrams, bigrams = counts["unigrams"], counts["bigrams"]
    total = sum(unigrams.values())
    vocabulary = len(unigrams) + 1
    scores = []
    for text in texts:
        tokens = words(text)
        values = []
        for first, second in zip(tokens, tokens[1:]):
            unigram = (unigrams.get(second, 0) + 1) / (total + vocabulary)
            seen = unigrams.get(first, 0)
            bigram = bigrams.get(f"{first} {second}", 0) / seen if seen else 0.0
            values.append(math.log((1 - UNIGRAM_SHARE) * bigram + UNIGRAM_SHARE * unigram))
        scores.append(statistics.fmean(values))
    return scores


def similarity_scores(texts):
    tokens = [words(text)[1:-1] for text in texts]
    if len(tokens) < 2:
        return [1.0 for _ in tokens]
    return [statistics.fmean(difflib.SequenceMatcher(a=tokens[i], b=tokens[j], autojunk=False).ratio()
                             for j in range(len(tokens)) if j != i) for i in range(len(tokens))]


def ranks(scores):
    """0 for the lowest score, up to len(scores) - 1 for the highest; ties share their mean rank."""
    return [(sum(other < score for other in scores) + sum(other <= score for other in scores) - 1) / 2
            for score in scores]


def combined_scores(texts, counts, central_weight, style_sign=1):
    style = [style_sign * value for value in style_scores(texts, counts)]
    if not central_weight:
        return style
    return [value + central_weight * central
            for value, central in zip(ranks(style), ranks(similarity_scores(texts)))]


def descending(values):
    return sorted(range(len(values)), key=lambda index: -values[index])


def interleaved(first, second):
    """Take from each ordering in turn, skipping what is already placed."""
    order = []
    for pair in zip(first, second):
        for index in pair:
            if index not in order:
                order.append(index)
    return order


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        self.central_weight = config.get("central_rank_weight", 0)
        self.style_sign = config.get("style_sign", 1)
        self.merge = config.get("merge", "sum")
        self.counts = json.loads((root / "models/style/counts.json").read_text(encoding="utf-8"))

    def _order(self, texts):
        if self.merge == "interleave":
            style = [self.style_sign * value for value in style_scores(texts, self.counts)]
            return interleaved(descending(similarity_scores(texts)), descending(style))
        return descending(combined_scores(texts, self.counts, self.central_weight, self.style_sign))

    def score(self, images, choices):
        """Place in the ranking mapped to [0, 1]."""
        if not choices:
            raise ValueError("choices must not be empty.")
        top = max(len(choices) - 1, 1)
        scores = [0.0] * len(choices)
        for place, index in enumerate(self._order([c["text"] for c in choices])):
            scores[index] = (top - place) / top
        return scores

    def predict(self, images, choices):
        return [choices[index]["id"] for index in self._order([c["text"] for c in choices])]
