"""Server API v2 submission: centrality decides between near-identical candidates, SmolVLM between
candidates that describe different scenes.

Every pair of candidates is compared once. If their word-level similarity is below `threshold`,
the one the fine-tuned SmolVLM scores higher wins (team/smolvlm_scorer.py); otherwise the more
central one wins (team/submission_central.py). Candidates are ranked by wins, ties by centrality.
Locally, SmolVLM's image use was large on pairs below 0.6 similarity and near zero on one-word
edits, where centrality is strong. With `video_margin` set, SmolVLM also decides any pair whose
scores differ by at least that much, so it overrides centrality only where it is confident; it then
runs on every episode. Otherwise it runs only when some pair is below the threshold. Each episode is
scored on its own input only, with no state kept between episodes.
"""

import json
import statistics
from pathlib import Path

from smolvlm_scorer import SmolVLMScorer
from submission_central import similarity_matrix


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        gate = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))["smolvlm_gate"]
        self.threshold = gate["threshold"]
        self.video_margin = gate.get("video_margin")
        self.scorer = SmolVLMScorer(root / "models/smolvlm", device, gate["max_frames"],
                                    gate["image_size"], gate["prior_weight"])

    def score(self, images, choices):
        """Wins scaled to [0, 1]; centrality, scaled below half a win, breaks ties."""
        if not choices:
            raise ValueError("choices must not be empty.")
        count = len(choices)
        if count == 1:
            return [1.0]
        similarity = similarity_matrix([choice["text"] for choice in choices])
        central = [statistics.fmean(row[j] for j in range(count) if j != i)
                   for i, row in enumerate(similarity)]
        video = None
        if self.video_margin is not None or any(
                similarity[i][j] < self.threshold for i in range(count) for j in range(i + 1, count)):
            video = self.scorer.score(images, [choice["text"] for choice in choices])
        wins = [0.0] * count
        for i in range(count):
            for j in range(i + 1, count):
                use_video = similarity[i][j] < self.threshold or (
                    self.video_margin is not None and abs(video[i] - video[j]) >= self.video_margin)
                scores = video if use_video else central
                if scores[i] == scores[j]:
                    wins[i] += 0.5
                    wins[j] += 0.5
                else:
                    wins[i if scores[i] > scores[j] else j] += 1.0
        return [(wins[i] + 0.49 * central[i]) / (count - 1 + 0.49) for i in range(count)]

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
