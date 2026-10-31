"""Generate negative descriptions from the organizers' data, using code and a fixed seed.

No word lists or hand-written text: every negative is built by rearranging or combining
original descriptions. Similar episodes are found with the organizers' action labels or TF-IDF
word overlap (see lexical.py), not with any model's scores. Pass only one split's episodes so negatives never cross train and validation.
"""

import json
import random
import re
from collections import Counter

import numpy as np

from team.lexical import lexical_similarity

NEGATIVE_TYPES = ("reorder", "added_action", "swapped_action", "similar_episode", "lexical",
                  "random")
MIXED_TYPES = ("reorder", "added_action", "swapped_action", "similar_episode")


def split_sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def load_action_labels(episode):
    path = episode.frames[0].parent.parent / "episode.json"
    actions = json.loads(path.read_text(encoding="utf-8"))["actions"]
    return [action["label"] for action in actions]


def reorder_clauses(sentence, rng):
    """Shuffle comma-separated clauses, keeping capitalization at the sentence start only."""
    body = sentence.rstrip(".!?")
    ending = sentence[len(body):]
    clauses = [clause for clause in re.split(r",\s+", body) if clause]
    if len(clauses) < 2:
        return sentence
    clauses[0] = clauses[0][0].lower() + clauses[0][1:]
    rng.shuffle(clauses)
    clauses[0] = clauses[0][0].upper() + clauses[0][1:]
    return ", ".join(clauses) + ending


def reversed_description(text):
    """Keep the opening and reverse the rest; unchanged if there are fewer than three sentences."""
    opening, *rest = split_sentences(text)
    return " ".join([opening, *reversed(rest)])


class NegativeGenerator:
    def __init__(self, descriptions, action_labels):
        if len(descriptions) != len(action_labels):
            raise ValueError("Each description needs its action labels.")
        if len(set(descriptions)) < 2:
            raise ValueError("At least two distinct descriptions are needed.")
        self.descriptions = list(descriptions)
        self.sentences = [split_sentences(text) for text in self.descriptions]
        vocabulary = {label: i for i, label in enumerate(sorted({
            label for labels in action_labels for label in labels}))}
        self.action_counts = np.zeros((len(descriptions), len(vocabulary)), dtype=np.int32)
        for index, labels in enumerate(action_labels):
            for label, count in Counter(labels).items():
                self.action_counts[index, vocabulary[label]] = count
        self._word_similarity = None

    def generate(self, index, negative_type, count, rng):
        """Up to `count` distinct negatives; fewer if this description can't produce enough."""
        make = {"reorder": self._reorder, "added_action": self._added_action,
                "swapped_action": self._swapped_action, "similar_episode": self._similar_episode,
                "lexical": self._lexical, "random": self._random}[negative_type]
        original = self.descriptions[index]
        results = {}
        for text in make(index, count, rng):
            if text != original:
                results[text] = None
            if len(results) == count:
                break
        return list(results)

    def _reorder(self, index, count, rng):
        # The opening stays first: moving it gives negatives away without looking at the video.
        opening, *rest = self.sentences[index]
        if not rest:
            return
        yield reversed_description(self.descriptions[index])
        for _ in range(count * 50):
            parts = list(rest)
            rng.shuffle(parts)
            if rng.random() < 0.5:
                position = rng.randrange(len(parts))
                parts[position] = reorder_clauses(parts[position], rng)
            yield " ".join([opening, *parts])

    def _donor_sentence(self, index, rng):
        """A later sentence from a different description that this description doesn't contain.

        Opening sentences are skipped because one in mid-text is easy to spot from text alone.
        """
        for _ in range(100):
            donor = rng.randrange(len(self.descriptions))
            if self.descriptions[donor] == self.descriptions[index] or len(self.sentences[donor]) < 2:
                continue
            sentence = rng.choice(self.sentences[donor][1:])
            if sentence not in self.sentences[index]:
                return sentence
        return None

    def _added_action(self, index, count, rng):
        sentences = self.sentences[index]
        for _ in range(count * 50):
            donor = self._donor_sentence(index, rng)
            if donor is not None:
                position = rng.randint(1, len(sentences))
                yield " ".join(sentences[:position] + [donor] + sentences[position:])

    def _swapped_action(self, index, count, rng):
        sentences = self.sentences[index]
        if len(sentences) < 2:
            return
        for _ in range(count * 50):
            donor = self._donor_sentence(index, rng)
            if donor is not None:
                position = rng.randrange(len(sentences))
                yield " ".join(sentences[:position] + [donor] + sentences[position + 1:])

    def _similar_episode(self, index, count, rng):
        """Random picks from the episodes whose action labels overlap most with this one."""
        own = self.action_counts[index]
        overlap = np.minimum(own, self.action_counts).sum(axis=1)
        union = np.maximum(own, self.action_counts).sum(axis=1).clip(min=1)
        similarity = overlap / union
        similarity[index] = -1
        tie_break = np.random.default_rng(rng.getrandbits(32)).random(len(similarity))
        nearest = np.lexsort((tie_break, -similarity))[:max(count * 5, 20)].tolist()
        rng.shuffle(nearest)
        for other in nearest:
            yield self.descriptions[other]

    def _lexical(self, index, count, rng):
        """Random picks from the descriptions with the most word overlap, after the hub correction."""
        if self._word_similarity is None:
            self._word_similarity = lexical_similarity(self.descriptions)
        similarity = self._word_similarity[index]
        nearest = np.lexsort((np.arange(len(similarity)), -similarity))
        nearest = [other for other in nearest.tolist() if other != index][:max(count * 5, 20)]
        rng.shuffle(nearest)
        for other in nearest:
            yield self.descriptions[other]

    def _random(self, index, count, rng):
        for _ in range(count * 50):
            yield self.descriptions[rng.randrange(len(self.descriptions))]


def candidate_set(description, negatives, rng):
    """Shuffle so that scores which tie never favour the correct position."""
    texts = [description, *negatives]
    rng.shuffle(texts)
    choices = [{"id": f"c{i}", "text": text} for i, text in enumerate(texts)]
    return {"correct": choices[texts.index(description)]["id"], "choices": choices}


def build_validation_sets(episode_ids, descriptions, action_labels, seed=2026, count=4):
    """Fixed candidates per episode: one set for each negative type, plus one of mixed types."""
    generator = NegativeGenerator(descriptions, action_labels)
    episodes = []
    for index, episode_id in enumerate(episode_ids):
        sets = {}
        for negative_type in NEGATIVE_TYPES:
            rng = random.Random(f"{seed}:{episode_id}:{negative_type}")
            negatives = generator.generate(index, negative_type, count, rng)
            if len(negatives) == count:
                sets[negative_type] = candidate_set(descriptions[index], negatives, rng)
        rng = random.Random(f"{seed}:{episode_id}:mixed")
        mixed = [text for negative_type in MIXED_TYPES
                 for text in generator.generate(index, negative_type, 1, rng)]
        if len(set(mixed)) == len(MIXED_TYPES):
            sets["mixed"] = candidate_set(descriptions[index], mixed, rng)
        episodes.append({"episode_id": episode_id, "sets": sets})
    return episodes
