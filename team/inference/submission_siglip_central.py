"""Server API v2: average each candidate's rank from SigLIP matching and from candidate centrality.

SigLIP compares each candidate with the video, and centrality compares it with the other candidates.
Ranks rather than raw scores, so that neither scale dominates. `central_rank_weight` in
inference_config.json sets how much centrality counts. Both use only this episode's input, and no
state is kept between episodes.
"""

import json
from pathlib import Path

from submission_central import Submission as CentralSubmission
from submission_siglip import Submission as SiglipSubmission


def ranks(scores):
    """0 for the lowest score, up to len(scores) - 1 for the highest; ties share their mean rank."""
    return [(sum(other < score for other in scores) + sum(other <= score for other in scores) - 1) / 2
            for score in scores]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        self.siglip = SiglipSubmission(artifact_dir, device)
        self.central = CentralSubmission(artifact_dir, device)
        config = json.loads((Path(artifact_dir) / "inference_config.json").read_text(encoding="utf-8"))
        self.central_weight = config.get("central_rank_weight", 1)

    def _combined(self, images, choices):
        first = ranks(self.siglip._matches(images, choices))
        second = ranks(self.central.score(images, choices))
        return [a + self.central_weight * b for a, b in zip(first, second)]

    def score(self, images, choices):
        """Combined rank mapped to [0, 1]."""
        top = max((1 + self.central_weight) * (len(choices) - 1), 1)
        return [value / top for value in self._combined(images, choices)]

    def predict(self, images, choices):
        combined = self._combined(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -combined[index])
        return [choices[index]["id"] for index in order]
