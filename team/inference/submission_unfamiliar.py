"""Server API v2: favour the candidate whose differing words SmolVLM finds least expected, without frames.

Counting word pairs, the most familiar candidate scored 0.2411 publicly, far below chance, so the least
familiar one is favoured here. Each token gets log p(token | no frames). For every pair of candidates
the shared tokens are dropped and the rest are averaged, and a candidate's score is minus its mean
margin over the others. `merge` in inference_config.json set to "interleave" takes the most central candidate,
then the top of this ranking, and so on in turn. No images are used, and no state is kept between episodes.
"""

import difflib
import json
import statistics
from pathlib import Path

from submission_fill import TokenScorer, central_scores


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


def pair_margin(first, second):
    """First candidate's mean log-probability minus the second's, on the tokens they don't share.

    A mean, so that a candidate with extra words doesn't look less likely only for being longer.
    """
    first_ids, first_scores = first
    second_ids, second_scores = second
    own, other = [], []
    matcher = difflib.SequenceMatcher(a=first_ids, b=second_ids, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            own.extend(first_scores[i1:i2])
            other.extend(second_scores[j1:j2])
    if not own or not other:
        return 0.0
    return statistics.fmean(own) - statistics.fmean(other)


def unfamiliar_scores(token_rows):
    """Minus each candidate's mean pair margin over the other candidates."""
    count = len(token_rows)
    totals = [0.0] * count
    for i in range(count):
        for j in range(i + 1, count):
            margin = pair_margin(token_rows[i], token_rows[j])
            totals[i] -= margin
            totals[j] += margin
    return [value / max(count - 1, 1) for value in totals]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))["unfamiliar"]
        self.merge = config.get("merge", "none")
        self.scorer = TokenScorer(root / "models/smolvlm", device, 1, config.get("image_size", 512), 1.0)

    def _order(self, texts):
        order = descending(unfamiliar_scores(self.scorer.prior_token_scores(texts)))
        if self.merge == "interleave":
            return interleaved(descending(central_scores(texts)), order)
        return order

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
