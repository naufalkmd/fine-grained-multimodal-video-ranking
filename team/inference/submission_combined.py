"""Server API v2: average each candidate's rank from a trained ranker and from SigLIP matching.

Ranks rather than raw scores, because the two models' scores have different scales, and there is
nothing to tune. Both use the video. No state is kept between episodes.
"""

from submission_ranker import Submission as RankerSubmission
from submission_siglip import Submission as SiglipSubmission


def ranks(scores):
    """0 for the lowest score, up to len(scores) - 1 for the highest; ties share their mean rank."""
    return [(sum(other < score for other in scores) + sum(other <= score for other in scores) - 1) / 2
            for score in scores]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        self.ranker = RankerSubmission(artifact_dir, device)
        self.siglip = SiglipSubmission(artifact_dir, device)

    def _combined(self, images, choices):
        first = ranks(self.ranker._logits(images, choices).cpu().tolist())
        second = ranks(self.siglip._matches(images, choices))
        return [a + b for a, b in zip(first, second)]

    def score(self, images, choices):
        """Combined rank mapped to [0, 1]."""
        top = max(2 * (len(choices) - 1), 1)
        return [value / top for value in self._combined(images, choices)]

    def predict(self, images, choices):
        combined = self._combined(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -combined[index])
        return [choices[index]["id"] for index in order]
