"""Server API v2 submission: fill in the blank with SmolVLM, on only the words that differ.

Each token gets its support from the frames: log p(token | frames) - log p(token | no frames). For
every pair of candidates, the tokens they share are dropped and the rest are summed, so a one-word
edit is decided by that word and not by the sixty words around it. A candidate's score is its mean
margin over the others. `central_rank_weight` in inference_config.json adds the rank from candidate
similarity (team/inference/submission_central.py). Each episode is scored on its own input only, with no
state kept between episodes.
"""

import difflib
import json
import statistics
from pathlib import Path

import torch

from smolvlm_scorer import SmolVLMScorer, sample_frames
from submission_central import similarity_matrix


def pair_margin(first, second):
    """First candidate's support minus the second's, on the tokens they don't share."""
    first_ids, first_scores = first
    second_ids, second_scores = second
    margin = 0.0
    matcher = difflib.SequenceMatcher(a=first_ids, b=second_ids, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            margin += sum(first_scores[i1:i2]) - sum(second_scores[j1:j2])
    return margin


def margins(token_rows):
    """Each candidate's mean pair margin over the other candidates."""
    count = len(token_rows)
    totals = [0.0] * count
    for i in range(count):
        for j in range(i + 1, count):
            margin = pair_margin(token_rows[i], token_rows[j])
            totals[i] += margin
            totals[j] -= margin
    return [value / max(count - 1, 1) for value in totals]


def ranks(scores):
    """0 for the lowest score, up to len(scores) - 1 for the highest; ties share their mean rank."""
    return [(sum(other < score for other in scores) + sum(other <= score for other in scores) - 1) / 2
            for score in scores]


def central_scores(texts):
    similarity = similarity_matrix(texts)
    return [statistics.fmean(row[j] for j in range(len(row)) if j != i) for i, row in enumerate(similarity)]


class TokenScorer(SmolVLMScorer):
    def _padded(self, rows):
        width = max(len(row) for row in rows)
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.full((len(rows), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
        for row, values in enumerate(rows):
            ids[row, :len(values)] = torch.tensor(values, device=self.device)
            mask[row, :len(values)] = 1
        return ids, mask

    @torch.inference_mode()
    def token_scores(self, images, texts):
        """(token ids, per-token support from the frames) for each text."""
        frames = [image.convert("RGB") for image in sample_frames(images, self.max_frames)]
        candidate_ids = self._candidate_ids(texts)
        count = len(candidate_ids)
        inputs = self.processor(text=self._prompt(len(frames)), images=[frames],
                                return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        prefix = self.model(**inputs, use_cache=True)
        cache = prefix.past_key_values
        cache.batch_repeat_interleave(count)
        ids, mask = self._padded(candidate_ids)
        prefix_mask = torch.ones((count, inputs["input_ids"].shape[1]), dtype=torch.long, device=self.device)
        logits = self.model(input_ids=ids, attention_mask=torch.cat([prefix_mask, mask], dim=1),
                            past_key_values=cache, use_cache=False).logits
        first = prefix.logits[:, -1:].expand(count, 1, -1)
        logits = torch.cat([first, logits[:, :-1]], dim=1).float().log_softmax(dim=-1)
        conditional = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1).cpu().tolist()

        prior_prefix = self.tokenizer(self._prompt(0))["input_ids"]
        ids, mask = self._padded([prior_prefix + values for values in candidate_ids])
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
        prior = logits.float().log_softmax(dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu().tolist()
        start = len(prior_prefix) - 1
        return [(values, [a - self.prior_weight * b
                          for a, b in zip(conditional[row], prior[row][start:start + len(values)])])
                for row, values in enumerate(candidate_ids)]


    @torch.inference_mode()
    def prior_token_scores(self, texts):
        """(token ids, per-token log p(token | no frames)) for each text."""
        candidate_ids = self._candidate_ids(texts)
        prefix = self.tokenizer(self._prompt(0))["input_ids"]
        ids, mask = self._padded([prefix + values for values in candidate_ids])
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
        prior = logits.float().log_softmax(dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu().tolist()
        start = len(prefix) - 1
        return [(values, prior[row][start:start + len(values)]) for row, values in enumerate(candidate_ids)]


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        fill = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))["fill"]
        self.central_weight = fill["central_rank_weight"]
        self.scorer = TokenScorer(root / "models/smolvlm", device, fill["max_frames"],
                                  fill["image_size"], fill["prior_weight"])

    def _values(self, images, texts):
        video = margins(self.scorer.token_scores(images, texts))
        if not self.central_weight:
            return video
        return [a + self.central_weight * b for a, b in zip(ranks(video), ranks(central_scores(texts)))]

    def score(self, images, choices):
        """Rank mapped to [0, 1]."""
        if not choices:
            raise ValueError("choices must not be empty.")
        positions = ranks(self._values(images, [choice["text"] for choice in choices]))
        top = max(len(choices) - 1, 1)
        return [position / top for position in positions]

    def predict(self, images, choices):
        values = self._values(images, [choice["text"] for choice in choices])
        order = sorted(range(len(choices)), key=lambda index: -values[index])
        return [choices[index]["id"] for index in order]
