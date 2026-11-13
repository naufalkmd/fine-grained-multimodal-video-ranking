"""Server API v2 submission: `text_blend_interleave` with the frames' event order added.

The text rule is team/inference/submission_text_rules.py (`text_blend`'s top, then the least familiar wording's
top, in turn; public 0.5958). The order check is team/inference/submission_order.py (SigLIP, no training).
`order_merge` in inference_config.json picks how they combine:

- `third`: the text rule's top two, then turns among `text_blend`, unfamiliar wording and the order
  check for the remaining places.
- `arbiter`: the text rule's ranking, except that the order check decides which of its top two goes
  first.

No state is kept between episodes.
"""

import json
from pathlib import Path

from submission_order import Submission as OrderCheck
from submission_style import descending, interleaved, style_scores
from submission_text_rules import Submission as TextRules


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        config = json.loads((Path(artifact_dir) / "inference_config.json").read_text(encoding="utf-8"))
        self.merge = config["order_merge"]
        if self.merge not in ("third", "arbiter"):
            raise ValueError(f"Unknown order_merge: {self.merge}")
        self.text = TextRules(artifact_dir, device)
        self.order = OrderCheck(artifact_dir, device)

    def ranking(self, images, texts):
        rule = descending(self.text._values(texts))
        unfamiliar = descending([-value for value in style_scores(texts, self.text.counts)])
        fits = self.order.order_fit(images, texts)
        if self.merge == "third":
            by_order = descending(fits)
            merged = interleaved(rule, unfamiliar)[:2]
            for group in zip(rule, unfamiliar, by_order):
                for index in group:
                    if index not in merged:
                        merged.append(index)
            return merged
        merged = interleaved(rule, unfamiliar)
        if len(merged) > 1 and fits[merged[1]] > fits[merged[0]]:
            merged[0], merged[1] = merged[1], merged[0]
        return merged

    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty.")
        if len(choices) == 1:
            return [1.0]
        ranking = self.ranking(images, [choice["text"] for choice in choices])
        return [1.0 - ranking.index(i) / (len(choices) - 1) for i in range(len(choices))]

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
