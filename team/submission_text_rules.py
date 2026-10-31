"""Server API v2: rank candidates by how they relate to the episode's other candidates, without images.

`rule` in inference_config.json picks the method:

- `consensus`: the share of the other candidates that contain each of a candidate's words, averaged.
  Words added or changed in only one candidate pull it down.
- `meaning`: mean MPNet cosine similarity to the other candidates.
- `blend`: the sum of the ranks from `consensus`, `meaning` and word-sequence similarity.

`merge` set to "interleave" takes the rule's top candidate, then the candidate with the least familiar wording
(team/submission_style.py), and so on in turn. With `gate` true as well, the least familiar wording goes first
whenever no candidate has all its words found in the others (no candidate looks like the one the others were edited
from). No state is kept between episodes.
"""

import difflib
import json
import re
import statistics
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def ranks(scores):
    """0 for the lowest score, up to len(scores) - 1 for the highest; ties share their mean rank."""
    return [(sum(other < score for other in scores) + sum(other <= score for other in scores) - 1) / 2
            for score in scores]


def sequence_scores(texts):
    tokens = [words(text) for text in texts]
    return [statistics.fmean(difflib.SequenceMatcher(a=tokens[i], b=tokens[j], autojunk=False).ratio()
                             for j in range(len(tokens)) if j != i) for i in range(len(tokens))]


def consensus_scores(texts):
    sets = [set(words(text)) for text in texts]
    others = max(len(sets) - 1, 1)
    scores = []
    for index, own in enumerate(sets):
        shares = [sum(word in other for j, other in enumerate(sets) if j != index) / others for word in own]
        scores.append(statistics.fmean(shares) if shares else 0.0)
    return scores


def unique_words(texts):
    """For each text, how many of its words match no word of any other text in a word-level alignment."""
    tokens = [words(text) for text in texts]
    counts = []
    for i, mine in enumerate(tokens):
        matched = [False] * len(mine)
        for j, theirs in enumerate(tokens):
            if j != i:
                for block in difflib.SequenceMatcher(a=mine, b=theirs, autojunk=False).get_matching_blocks():
                    matched[block.a:block.a + block.size] = [True] * block.size
        counts.append(matched.count(False))
    return counts


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        self.rule = config["rule"]
        self.merge = config.get("merge", "none")
        self.gate = config.get("gate", False)
        if self.merge == "interleave":
            self.counts = json.loads((root / "models/style/counts.json").read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self.tokenizer = self.model = None
        if self.rule in ("meaning", "blend"):
            self.tokenizer = AutoTokenizer.from_pretrained(root / "models/mpnet", local_files_only=True)
            self.model = AutoModel.from_pretrained(root / "models/mpnet", local_files_only=True, dtype=torch.float32)
            self.model.requires_grad_(False).to(self.device).eval()

    @torch.inference_mode()
    def _meaning(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=384, return_tensors="pt").to(self.device)
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vectors = torch.nn.functional.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1), dim=-1)
        similarity = vectors @ vectors.T
        count = max(len(texts) - 1, 1)
        return ((similarity.sum(dim=1) - similarity.diagonal()) / count).tolist()

    def _values(self, texts):
        if self.rule == "consensus":
            return consensus_scores(texts)
        if self.rule == "meaning":
            return self._meaning(texts)
        if self.rule == "blend":
            parts = [ranks(consensus_scores(texts)), ranks(self._meaning(texts)), ranks(sequence_scores(texts))]
            return [sum(values) for values in zip(*parts)]
        raise ValueError(f"Unknown rule: {self.rule}")

    def _order(self, texts):
        values = self._values(texts)
        order = sorted(range(len(texts)), key=lambda index: -values[index])
        if self.merge == "interleave":
            from submission_style import descending, interleaved, style_scores
            unfamiliar = descending([-value for value in style_scores(texts, self.counts)])
            if getattr(self, "gate", False) and 0 not in unique_words(texts):
                return interleaved(unfamiliar, order)
            return interleaved(order, unfamiliar)
        return order

    def score(self, images, choices):
        """Rank mapped to [0, 1]; with a merge, place in the ranking."""
        if not choices:
            raise ValueError("choices must not be empty.")
        texts = [choice["text"] for choice in choices]
        top = max(len(choices) - 1, 1)
        if self.merge == "none":
            return [position / top for position in ranks(self._values(texts))]
        scores = [0.0] * len(choices)
        for place, index in enumerate(self._order(texts)):
            scores[index] = (top - place) / top
        return scores

    def predict(self, images, choices):
        order = self._order([choice["text"] for choice in choices])
        return [choices[index]["id"] for index in order]
