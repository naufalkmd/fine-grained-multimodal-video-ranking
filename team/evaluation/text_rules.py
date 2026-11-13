"""Text-only ranking rules, and a check of them on the LLM-edited validation sets.

The rules read only the episode's candidate texts, plus word counts taken from training
descriptions:

- `centrality`: mean word-sequence similarity to the other candidates (team/inference/submission_central.py).
- `style`: mean bigram log-probability per word under the training descriptions; the most familiar
  wording first. Publicly it scored far below chance, so `unfamiliar` reverses it.
- `rarity`: every pair of candidates is compared only on the words where they differ, and the one
  whose differing words are less familiar wins; ranked by wins, ties by centrality.
- `interleave`: centrality's first choice, then `unfamiliar`'s, alternating.
- `rank_sum`: centrality's rank plus `unfamiliar`'s rank.
- `content_centrality`: centrality on content words only, leaving out the words found in at least
  `COMMON_SHARE` of the training descriptions. If the negatives are reworded copies of the correct
  text with one change each, rewording mostly touches common words, so the correct text is central
  again on the rest.
- `style_outlier`: the candidate whose use of common words and punctuation differs most from the
  others. If one writer reworded the negatives, the correct text is the odd one out in style,
  whoever that writer was.

`main` scores them on team/candidates/qwen_long_negatives.py output, to see whether those sets reproduce the
public scores of the rules that were uploaded.

    python -m team.evaluation.text_rules --generated cuda/outputs/qwen/long_negatives.jsonl \
        --splits cuda/outputs/baseline/splits.json

Standard names: medoid (MBR) selection, n-gram language-model likelihood, round-robin rank fusion and a rule-
based switch, checked offline. Our names are mapped in docs/glossary.md.
"""

import argparse
import difflib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from common import ROOT, load_config
from dataset import load_episodes
from team.inference.submission_central import similarity_matrix, words

RANK_SCORES = (1.0, 0.5, 0.25, 0.125, 0.0)
COMMON_SHARE = 0.2
PUNCTUATION = ",.;:"
PUBLIC = {"centrality": 0.5515, "style": 0.2411, "unfamiliar": 0.5327, "rank_sum": 0.5443}


class BigramModel:
    """Bigram log-probabilities smoothed towards add-one unigram probabilities, and the words that
    appear in at least COMMON_SHARE of the descriptions."""

    def __init__(self, descriptions, prior=1.0):
        self.unigrams, self.bigrams, documents = Counter(), Counter(), Counter()
        for text in descriptions:
            tokens = ["<s>", *words(text), "</s>"]
            self.unigrams.update(tokens)
            self.bigrams.update(zip(tokens, tokens[1:]))
            documents.update(set(words(text)))
        self.common = sorted(word for word, count in documents.items()
                             if count >= COMMON_SHARE * len(descriptions))
        self.total = sum(self.unigrams.values())
        self.vocabulary = len(self.unigrams) + 1
        self.prior = prior

    def unigram(self, word):
        return (self.unigrams[word] + 1) / (self.total + self.vocabulary)

    def token_log_probs(self, tokens):
        """log p(token | previous token) for each token, with <s> before the first."""
        previous = ["<s>", *tokens[:-1]]
        return [math.log((self.bigrams[(a, b)] + self.prior * self.unigram(b))
                         / (self.unigrams[a] + self.prior)) for a, b in zip(previous, tokens)]


def centrality(texts, model=None):
    similarity = similarity_matrix(texts)
    return [statistics.fmean(row[j] for j in range(len(row)) if j != i)
            for i, row in enumerate(similarity)]


def style(texts, model):
    return [statistics.fmean(model.token_log_probs(words(text))) for text in texts]


def unfamiliar(texts, model):
    return [-value for value in style(texts, model)]


def rarity(texts, model):
    tokens = [words(text) for text in texts]
    log_probs = [model.token_log_probs(t) for t in tokens]
    central = centrality(texts)
    wins = [0.0] * len(texts)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            mine, theirs = [], []
            matcher = difflib.SequenceMatcher(a=tokens[i], b=tokens[j], autojunk=False)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != "equal":
                    mine += log_probs[i][i1:i2]
                    theirs += log_probs[j][j1:j2]
            if mine and theirs and statistics.fmean(mine) != statistics.fmean(theirs):
                wins[i if statistics.fmean(mine) < statistics.fmean(theirs) else j] += 1.0
            else:
                wins[i] += 0.5
                wins[j] += 0.5
    return [wins[k] + 0.49 * central[k] for k in range(len(texts))]


def content_centrality(texts, model):
    common = set(model.common)
    tokens = [[w for w in words(text) if w not in common] for text in texts]
    return [statistics.fmean(
        difflib.SequenceMatcher(a=tokens[i], b=tokens[j], autojunk=False).ratio()
        for j in range(len(texts)) if j != i) for i in range(len(texts))]


def style_profile(text, common):
    tokens = words(text)
    counts = Counter(w for w in tokens if w in common)
    size = max(len(tokens), 1)
    return ([counts[w] / size for w in common]
            + [text.count(mark) / size for mark in PUNCTUATION])


def style_outlier(texts, model):
    profiles = [style_profile(text, model.common) for text in texts]
    central = centrality(texts)
    return [statistics.fmean(sum(abs(a - b) for a, b in zip(profiles[i], profiles[j]))
                             for j in range(len(texts)) if j != i) + 1e-6 * central[i]
            for i in range(len(texts))]


def unique_tokens(texts):
    """For each text, how many of its words match no word of any other text in a word-level
    alignment. A text whose words all survive in some other candidate has none."""
    tokens = [words(text) for text in texts]
    counts = []
    for i, mine in enumerate(tokens):
        matched = [False] * len(mine)
        for j, theirs in enumerate(tokens):
            if j == i:
                continue
            matcher = difflib.SequenceMatcher(a=mine, b=theirs, autojunk=False)
            for block in matcher.get_matching_blocks():
                for k in range(block.a, block.a + block.size):
                    matched[k] = True
        counts.append(matched.count(False))
    return counts


def regime_gate(texts, model):
    """Centrality's order when some candidate has no unique words (the negatives look like edits
    of it); otherwise `unfamiliar` first, then centrality, alternating."""
    unique = unique_tokens(texts)
    if 0 in unique:
        return interleave(texts, model)
    first, second = order(unfamiliar(texts, model)), order(centrality(texts))
    ranking = []
    for pair in zip(first, second):
        for index in pair:
            if index not in ranking:
                ranking.append(index)
    return [len(texts) - ranking.index(k) for k in range(len(texts))]


def order(scores):
    return sorted(range(len(scores)), key=lambda index: -scores[index])


def interleave(texts, model):
    first, second = order(centrality(texts)), order(unfamiliar(texts, model))
    ranking = []
    for pair in zip(first, second):
        for index in pair:
            if index not in ranking:
                ranking.append(index)
    return [len(texts) - ranking.index(k) for k in range(len(texts))]


def rank_sum(texts, model):
    ranks = [order(centrality(texts)), order(unfamiliar(texts, model))]
    central = centrality(texts)
    return [-sum(r.index(k) for r in ranks) + 0.01 * central[k] for k in range(len(texts))]


def content_plus_outlier(texts, model):
    ranks = [order(content_centrality(texts, model)), order(style_outlier(texts, model))]
    central = content_centrality(texts, model)
    return [-sum(r.index(k) for r in ranks) + 0.01 * central[k] for k in range(len(texts))]


RULES = {"centrality": centrality, "style": style, "unfamiliar": unfamiliar, "rarity": rarity,
         "interleave": interleave, "rank_sum": rank_sum, "content_centrality": content_centrality,
         "style_outlier": style_outlier, "content_plus_outlier": content_plus_outlier,
         "regime_gate": regime_gate}


def candidate_sets(path, seed):
    rng = random.Random(seed)
    sets = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if len(record["negatives"]) != 4:
            continue
        texts = [record["description"], *record["negatives"].values()]
        rng.shuffle(texts)
        sets.append({"episode_id": record["episode_id"], "texts": texts,
                     "correct": texts.index(record["description"])})
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "cuda/config.json")
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    episodes = {e.episode_id: e for e in load_episodes(load_config(args.config)["data_dir"])}
    train = json.loads(args.splits.read_text(encoding="utf-8"))["train"]
    model = BigramModel([episodes[i].description for i in train])
    sets = candidate_sets(args.generated, args.seed)
    print(f"Episodes with four checked negatives: {len(sets)}")
    per_episode = {}
    for name, rule in RULES.items():
        ranks = [order(rule(s["texts"], model)).index(s["correct"]) for s in sets]
        per_episode[name] = [RANK_SCORES[r] for r in ranks]
        share = Counter(ranks)
        print(f"{name:11s} {statistics.fmean(per_episode[name]):.4f}  public "
              f"{PUBLIC.get(name, float('nan')):.4f}  ranks % "
              + " / ".join(f"{100 * share[r] / len(ranks):.1f}" for r in range(5)))
    a, b = per_episode["centrality"], per_episode["unfamiliar"]
    if len(set(a)) > 1 and len(set(b)) > 1:
        print(f"Correlation of episode scores, centrality vs unfamiliar: "
              f"{statistics.correlation(a, b):+.3f} (public -0.008)")
    print(f"Common words ({len(model.common)}): {' '.join(model.common)}")


if __name__ == "__main__":
    main()
