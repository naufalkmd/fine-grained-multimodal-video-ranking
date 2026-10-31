"""Server API v2: text structure makes a shortlist, and image evidence orders it.

Centrality picks the `shortlist` candidates most similar to the others. They rank above the rest, in
order of image evidence, so the video decides first place. The rest keep their centrality order. Both
use only this episode's input, and no state is kept between episodes.
"""

import json
from pathlib import Path

from submission_central import Submission as CentralSubmission
from submission_evidence import Submission as EvidenceSubmission


def shortlist_order(central, evidence, size):
    """Candidate indices, best first: the shortlist by evidence, then the others by centrality."""
    by_central = sorted(range(len(central)), key=lambda index: -central[index])
    picked = sorted(by_central[:size], key=lambda index: -evidence[index])
    return picked + by_central[size:]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        self.evidence = EvidenceSubmission(artifact_dir, device)
        self.central = CentralSubmission(artifact_dir, device)
        config = json.loads((Path(artifact_dir) / "inference_config.json").read_text(encoding="utf-8"))
        self.size = config.get("shortlist_size", 3)

    def _order(self, images, choices):
        evidence = self.evidence._logits(images, choices).cpu().tolist()
        return shortlist_order(self.central.score(images, choices), evidence, self.size)

    def score(self, images, choices):
        """Position in the ranking mapped to [0, 1], highest for first place."""
        order = self._order(images, choices)
        top = max(len(choices) - 1, 1)
        values = [0.0] * len(choices)
        for position, index in enumerate(order):
            values[index] = (len(choices) - 1 - position) / top
        return values

    def predict(self, images, choices):
        return [choices[index]["id"] for index in self._order(images, choices)]
