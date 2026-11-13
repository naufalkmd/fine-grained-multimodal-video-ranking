"""Server API v2: compare candidates, then let SigLIP judge only the words that differ.

For each pair of candidates, difflib finds the words that differ. Each difference, with a few words of
context on either side, is matched against the frames, and the candidate whose wording fits the video
better gains the difference between the two matches. Shared words therefore can't dilute a one-word
change. The final ranking averages this rank with the rank from candidate centrality. There is no
training, and no state is kept between episodes.
"""

import difflib
import re

import torch
from submission_central import Submission as CentralSubmission
from submission_siglip import Submission as SiglipSubmission

CONTEXT_WORDS = 2
WORD = re.compile(r"\S+")


def ranks(scores):
    """0 for the lowest score, up to len(scores) - 1 for the highest; ties share their mean rank."""
    return [(sum(other < score for other in scores) + sum(other <= score for other in scores) - 1) / 2
            for score in scores]


def normalized(word):
    return word.lower().strip(".,;:!?\"'()")


def difference_phrases(first, second, context=CONTEXT_WORDS):
    """Phrases around each difference between two word lists, as (first's phrases, second's phrases)."""
    matcher = difflib.SequenceMatcher(a=[normalized(w) for w in first], b=[normalized(w) for w in second],
                                      autojunk=False)
    first_phrases, second_phrases = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            first_phrases.append(" ".join(first[max(i1 - context, 0):i2 + context]))
            second_phrases.append(" ".join(second[max(j1 - context, 0):j2 + context]))
    return first_phrases, second_phrases


def pairwise_margins(texts, match):
    """Each candidate's mean advantage over the others. `match` maps a list of phrases to scores."""
    words = [WORD.findall(text) for text in texts]
    pairs = {}
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs[i, j] = difference_phrases(words[i], words[j])
    phrases = list(dict.fromkeys(p for pair in pairs.values() for side in pair for p in side if p))
    lookup = dict(zip(phrases, match(phrases))) if phrases else {}

    def mean_match(side):
        values = [lookup[p] for p in side if p]
        return sum(values) / len(values) if values else 0.0

    margins = [0.0] * len(texts)
    for (i, j), (first, second) in pairs.items():
        margin = mean_match(first) - mean_match(second)
        margins[i] += margin
        margins[j] -= margin
    return [margin / max(len(texts) - 1, 1) for margin in margins]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        self.siglip = SiglipSubmission(artifact_dir, device)
        self.central = CentralSubmission(artifact_dir, device)

    @torch.inference_mode()
    def margins(self, images, choices):
        if not images:
            raise ValueError("An episode must contain at least one image.")
        if not choices:
            raise ValueError("choices must not be empty.")
        frames = self.siglip._images(images)

        def match(phrases):
            return (self.siglip._texts(phrases) @ frames.T).amax(dim=1).tolist()

        return pairwise_margins([choice["text"] for choice in choices], match)

    def _combined(self, images, choices):
        first = ranks(self.margins(images, choices))
        second = ranks(self.central.score(images, choices))
        return [a + b for a, b in zip(first, second)]

    def score(self, images, choices):
        """Combined rank mapped to [0, 1]."""
        top = max(2 * (len(choices) - 1), 1)
        return [value / top for value in self._combined(images, choices)]

    def predict(self, images, choices):
        combined = self._combined(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -combined[index])
        return [choices[index]["id"] for index in order]
