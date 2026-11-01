"""Server API v2: text rules propose, and the video decides between their top picks.

The ranking takes turns between `blend` and the least familiar wording (team/submission_text_rules.py), which puts
blend's top pick first. Here, when the two top picks differ, SmolVLM compares the frames' support for them on the
tokens they don't share, and the second pick goes first only if its margin exceeds `threshold` in
inference_config.json. Locally, a threshold of 3 gains on near-identical candidates with the right video and about
nothing with the wrong one. The rest of the ranking is unchanged. Each episode is scored on its own input only,
with no state kept between episodes.
"""

import json
from pathlib import Path

from submission_fill import TokenScorer, pair_margin
from submission_style import descending, style_scores
from submission_text_rules import Submission as TextRules


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))["arbiter"]
        self.threshold = config["threshold"]
        self.rules = TextRules(artifact_dir, device)
        self.scorer = TokenScorer(root / "models/smolvlm", device, config["max_frames"],
                                  config["image_size"], config["prior_weight"])

    def _order(self, images, texts):
        order = self.rules._order(texts)
        unfamiliar = descending([-value for value in style_scores(texts, self.rules.counts)])[0]
        if order[0] != unfamiliar:
            rows = self.scorer.token_scores(images, [texts[order[0]], texts[unfamiliar]])
            if pair_margin(rows[1], rows[0]) > self.threshold:
                order[0], order[1] = order[1], order[0]
        return order

    def score(self, images, choices):
        """Place in the ranking mapped to [0, 1]."""
        if not choices:
            raise ValueError("choices must not be empty.")
        top = max(len(choices) - 1, 1)
        scores = [0.0] * len(choices)
        for place, index in enumerate(self._order(images, [c["text"] for c in choices])):
            scores[index] = (top - place) / top
        return scores

    def predict(self, images, choices):
        return [choices[index]["id"] for index in self._order(images, [c["text"] for c in choices])]
