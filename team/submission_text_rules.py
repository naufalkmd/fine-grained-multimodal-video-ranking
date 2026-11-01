"""Server API v2: rank candidates by how they relate to the episode's other candidates, without images.

`rule` in inference_config.json picks the method:

- `consensus`: the share of the other candidates that contain each of a candidate's words, averaged.
  Words added or changed in only one candidate pull it down.
- `meaning`: mean MPNet cosine similarity to the other candidates.
- `blend`: the sum of the ranks from `consensus`, `meaning` and word-sequence similarity.

`merge` set to "interleave" takes the rule's top candidate, then the candidate with the least familiar wording
(team/submission_style.py), and so on in turn. With `gate` true, the least familiar wording goes first when no
candidate looks like the one the others were edited from: no candidate has all its words in the others, or, with
`gate_excess` set, even the best candidate's word edit distances to the others don't add up (`lowest_excess`). With
`chars` true, the least familiar wording averages, within the episode, standardized word-pair familiarity and
character familiarity (team/submission_chars.py, case and punctuation kept). No state is kept between episodes.
"""

import difflib
import json
import re
import statistics
from pathlib import Path

import torch
from submission_style import descending, interleaved, style_scores
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


def standardized(values):
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    mean = statistics.fmean(values)
    return [(value - mean) / spread if spread else 0.0 for value in values]


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


def word_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def lowest_excess(texts):
    """How far the best candidate is from being the text the others were edited from: 0 when it is.

    If c is the base and edits are independent, d(a, b) = d(a, c) + d(c, b) for the others a and b. A candidate's
    excess is the sum over pairs of the others of d(c, a) + d(c, b) - d(a, b), divided by the sum of d(c, a) + d(c, b).
    """
    tokens = [words(text) for text in texts]
    size = len(tokens)
    distance = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            distance[i][j] = distance[j][i] = word_distance(tokens[i], tokens[j])
    values = []
    for c in range(size):
        others = [i for i in range(size) if i != c]
        pairs = [(a, b) for index, a in enumerate(others) for b in others[index + 1:]]
        bottom = sum(distance[c][a] + distance[c][b] for a, b in pairs)
        top = sum(distance[c][a] + distance[c][b] - distance[a][b] for a, b in pairs)
        values.append(top / bottom if bottom else 0.0)
    return min(values)


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        self.rule = config["rule"]
        self.merge = config.get("merge", "none")
        self.gate = config.get("gate", False)
        self.gate_excess = config.get("gate_excess")
        if self.merge == "interleave":
            self.counts = json.loads((root / "models/style/counts.json").read_text(encoding="utf-8"))
        self.chars = None
        if config.get("chars"):
            from submission_chars import CharModel
            self.chars = CharModel(json.loads((root / "models/style/char_counts.json").read_text(encoding="utf-8")))
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

    def _no_base_candidate(self, texts):
        """True when the gate finds no candidate that looks like the one the others were edited from."""
        if not getattr(self, "gate", False):
            return False
        if 0 not in unique_words(texts):
            return True
        threshold = getattr(self, "gate_excess", None)
        return threshold is not None and lowest_excess(texts) >= threshold

    def _unfamiliar(self, texts):
        """Higher for less familiar wording."""
        word = [-value for value in style_scores(texts, self.counts)]
        if getattr(self, "chars", None) is None:
            return word
        char = [-self.chars.familiarity(text) for text in texts]
        return [a + b for a, b in zip(standardized(word), standardized(char))]

    def _order(self, texts):
        values = self._values(texts)
        order = sorted(range(len(texts)), key=lambda index: -values[index])
        if self.merge == "interleave":
            unfamiliar = descending(self._unfamiliar(texts))
            if self._no_base_candidate(texts):
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
