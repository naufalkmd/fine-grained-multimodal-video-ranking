"""Server API v2 submission that ranks candidates by how central each is among the others.

A diagnostic: it reads only the episode's own candidate texts and no video, so its public score
shows whether test negatives are edits of the correct description. When every negative changes
one part of the correct text, the correct text is the one most similar to all the others. Uses
nothing but the standard library, and keeps no state between episodes.
"""

import difflib
import re
import statistics


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        self.artifact_dir = artifact_dir

    def score(self, images, choices):
        """Each candidate's mean word-level similarity to the other candidates."""
        if not choices:
            raise ValueError("choices must not be empty.")
        texts = [words(choice["text"]) for choice in choices]
        if len(texts) == 1:
            return [1.0]
        return [statistics.fmean(
            difflib.SequenceMatcher(a=texts[i], b=texts[j], autojunk=False).ratio()
            for j in range(len(texts)) if j != i) for i in range(len(texts))]

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
