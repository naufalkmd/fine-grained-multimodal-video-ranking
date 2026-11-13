"""Server API v2 submission that ranks candidates by how central each is among the others.

A diagnostic: it reads only the episode's own candidate texts and no video, so its public score
shows whether test negatives are edits of the correct description. When every negative changes
one part of the correct text, the correct text is the one most similar to all the others. Uses
nothing but the standard library, and keeps no state between episodes.

Standard names: medoid / minimum Bayes risk (MBR) selection with word-sequence (Ratcliff/Obershelp)
similarity. Our names are mapped in docs/glossary.md.
"""

import difflib
import re
import statistics


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def similarity_matrix(texts):
    """Word-level similarity of every pair of texts; 1.0 on the diagonal."""
    tokens = [words(text) for text in texts]
    return [[1.0 if i == j else difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()
             for j, b in enumerate(tokens)] for i, a in enumerate(tokens)]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        self.artifact_dir = artifact_dir

    def score(self, images, choices):
        """Each candidate's mean word-level similarity to the other candidates."""
        if not choices:
            raise ValueError("choices must not be empty.")
        if len(choices) == 1:
            return [1.0]
        similarity = similarity_matrix([choice["text"] for choice in choices])
        return [statistics.fmean(row[j] for j in range(len(row)) if j != i)
                for i, row in enumerate(similarity)]

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
