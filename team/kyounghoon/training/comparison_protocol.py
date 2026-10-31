"""Pure evaluation helpers for the frozen-SigLIP temporal comparison.

Hard choices contain ONLY unedited original validation descriptions. TF-IDF is
fit on training descriptions, then used to choose two similar validation
descriptions plus two random validation descriptions for each validation video.
Similarity does not establish that another description is semantically false;
this diagnostic is not a verified temporal-reasoning test or leaderboard score.
No files are read or written and no model is downloaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Mapping


REWARDS = (1.0, 0.5, 0.25, 0.125, 0.0)
HARD_POLICY = "train_tfidf_two_similar_two_random_original_validation_v1"
HARD_WARNING = (
    "Other original validation captions are treated as negatives, but semantic "
    "false negatives have not been verified or eliminated. This is a fixed "
    "similar-caption diagnostic, not a leaderboard estimate or a verified "
    "test of temporal understanding."
)


def _hash_json(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normal_text(text):
    return " ".join(text.split()).casefold()


def _ids(values, label):
    values = list(values)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain nonempty string IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate IDs in {label}")
    return values


def _candidate_inputs(manifest, split, random_candidates, token_keys):
    episodes = manifest["episodes"]
    ids = _ids((row["episode_id"] for row in episodes), "manifest")
    by_id = dict(zip(ids, episodes))
    for episode in episodes:
        text = episode["description"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Manifest descriptions must be nonempty original strings")
    train_ids = _ids(split["train_ids"], "train_ids")
    val_ids = _ids(split["val_ids"], "val_ids")
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation IDs overlap")
    if not (set(train_ids) | set(val_ids)) <= set(by_id):
        raise ValueError("Split contains IDs absent from the manifest")
    if len(val_ids) < 5:
        raise ValueError("Five-choice evaluation needs at least five validation episodes")
    source = manifest["source"]["fingerprint"]
    if (split.get("source_fingerprint") != source
            or random_candidates.get("source_fingerprint") != source):
        raise ValueError("Source fingerprints do not agree")
    random_records = random_candidates["episodes"]
    random_ids = _ids((row["episode_id"] for row in random_records), "random candidates")
    if random_ids != val_ids:
        raise ValueError("Random candidates must cover validation IDs in split order")
    val_set = set(val_ids)
    for record in random_records:
        choices = record["choices"]
        choice_ids = _ids((choice["id"] for choice in choices), "choice IDs")
        if len(choices) != 5 or record["correct_choice_id"] not in choice_ids:
            raise ValueError("Existing random records must have five choices and one correct ID")
        positives = []
        for choice in choices:
            origin = choice["source_episode_id"]
            if origin not in val_set or choice["text"] != by_id[origin]["description"]:
                raise ValueError("Existing random choices must be original validation captions")
            if origin == record["episode_id"]:
                positives.append(choice["id"])
        if positives != [record["correct_choice_id"]]:
            raise ValueError("Existing random positive does not match its episode")
    keys = {}
    for episode_id in val_ids:
        key = token_keys.get(episode_id)
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-fA-F]{64}", key) is None:
            raise ValueError(f"Missing or invalid actual-token-input SHA256 for {episode_id}")
        keys[episode_id] = key.lower()
    return by_id, train_ids, val_ids, keys


def build_hard_candidates(manifest, split, random_candidates, token_keys, seed=2028):
    """Build fixed 2-similar + 2-random original-caption choices.

    ``token_keys`` maps validation episode IDs to SHA256 hashes of the actual
    fixed-length truncate64 token inputs (including special tokens and padding).
    All five choices have different normalized texts AND different token hashes.
    Training captions fit TF-IDF only; candidate captions come only from validation.
    The original random contract is checked and hashed, never mutated or replaced.

    Mining metadata is evaluator-only. Pass only {id, text} choices to models.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    by_id, train_ids, val_ids, keys = _candidate_inputs(
        manifest, split, random_candidates, token_keys)
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b", min_df=1,
        sublinear_tf=True, norm="l2", dtype=np.float64,
    )
    try:
        vectorizer.fit([by_id[episode_id]["description"] for episode_id in train_ids])
    except ValueError as error:
        raise ValueError(f"Cannot fit TF-IDF on training captions: {error}") from error
    val_features = vectorizer.transform([by_id[episode_id]["description"]
                                         for episode_id in val_ids])
    normalized = {episode_id: _normal_text(by_id[episode_id]["description"])
                  for episode_id in val_ids}
    rng = random.Random(seed)
    output, hard_similarities, random_similarities = [], [], []
    zero_queries = 0
    for index, episode_id in enumerate(val_ids):
        similarity = (val_features.getrow(index) @ val_features.T).toarray().ravel()
        if similarity.shape != (len(val_ids),) or not np.isfinite(similarity).all():
            raise ValueError("Invalid TF-IDF cosine similarities")
        if val_features.getrow(index).nnz == 0:
            zero_queries += 1
        similarity_by_id = dict(zip(val_ids, map(float, similarity)))
        used_texts = {normalized[episode_id]}
        used_keys = {keys[episode_id]}

        def eligible(origin):
            return (origin != episode_id and normalized[origin] not in used_texts
                    and keys[origin] not in used_keys)

        def select(origins, count):
            selected = []
            for origin in origins:
                if eligible(origin):
                    selected.append(origin)
                    used_texts.add(normalized[origin])
                    used_keys.add(keys[origin])
                    if len(selected) == count:
                        return selected
            raise ValueError(
                f"Cannot obtain four distinct noncolliding negatives for {episode_id}; "
                "do not relax caption or token-input collision checks")

        ranked = sorted(val_ids, key=lambda origin: (-similarity_by_id[origin], origin))
        hard_ids = select(ranked, 2)
        remaining = [origin for origin in val_ids if eligible(origin)]
        rng.shuffle(remaining)
        random_ids = select(remaining, 2)
        origins = [episode_id] + hard_ids + random_ids
        rng.shuffle(origins)
        choices = [{"id": f"choice_{number}", "text": by_id[origin]["description"],
                    "source_episode_id": origin}
                   for number, origin in enumerate(origins)]
        correct_id = next(choice["id"] for choice in choices
                          if choice["source_episode_id"] == episode_id)
        rng.shuffle(choices)
        hard_diagnostics = [{"source_episode_id": origin,
                             "cosine_similarity": similarity_by_id[origin]}
                            for origin in hard_ids]
        random_diagnostics = [{"source_episode_id": origin,
                               "cosine_similarity": similarity_by_id[origin]}
                              for origin in random_ids]
        hard_similarities.extend(row["cosine_similarity"] for row in hard_diagnostics)
        random_similarities.extend(row["cosine_similarity"] for row in random_diagnostics)
        output.append({"episode_id": episode_id, "choices": choices,
                       "correct_choice_id": correct_id,
                       "mining": {"hard_negatives": hard_diagnostics,
                                  "random_negatives": random_diagnostics,
                                  "query_has_tfidf_features": bool(val_features.getrow(index).nnz)}})

    def summary(values):
        return {"count": len(values), "min": min(values), "max": max(values),
                "mean": math.fsum(values) / len(values)}

    return {
        "schema_version": 1, "source_fingerprint": manifest["source"]["fingerprint"],
        "split_seed": split["seed"], "candidate_seed": seed,
        "negative_policy": HARD_POLICY, "warning": HARD_WARNING,
        "reference_random_candidates_sha256": _hash_json(random_candidates),
        "validation_token_keys_sha256": _hash_json(keys),
        "tfidf": {"fit_source": "training_descriptions_only", "fit_episodes": len(train_ids),
                  "ngram_range": [1, 2], "lowercase": True, "sublinear_tf": True,
                  "norm": "l2", "min_df": 1, "token_pattern": r"(?u)\b\w\w+\b",
                  "vocabulary_size": len(vectorizer.vocabulary_),
                  "vocabulary_sha256": _hash_json({term: int(index)
                                                   for term, index in vectorizer.vocabulary_.items()}),
                  "query": "original_positive_validation_caption",
                  "candidate_pool": "original_validation_captions_only",
                  "ties": "lexicographic_source_episode_id"},
        "mining_diagnostics": {"hard_selected_cosine": summary(hard_similarities),
                               "random_selected_cosine": summary(random_similarities),
                               "zero_feature_queries": zero_queries,
                               "semantic_false_negatives_verified": False,
                               "positive_token_collisions": 0,
                               "any_choice_token_collisions": 0},
        "episodes": output,
    }


def _rank_values(ranking, correct_id):
    ranking = _ids(ranking, "ranking")
    if len(ranking) != 5 or correct_id not in ranking:
        raise ValueError("Ranking must contain all five unique choices and the correct ID")
    rank = ranking.index(correct_id) + 1
    return {"correct_rank": rank, "rank_reward": REWARDS[rank - 1], "top1": rank == 1}


def rank_prediction(epid, choices, scores, correct_id):
    """Rank five scalar scores; ties use choice IDs, not input display position.

    ``scores`` may be an ID-to-number mapping or a sequence aligned with choices.
    Returned records intentionally omit choice text and any action metadata.
    """
    _ids([epid], "episode ID")
    ids = _ids((choice["id"] for choice in choices), "choice IDs")
    if len(ids) != 5 or correct_id not in ids:
        raise ValueError("Exactly five choices including the correct ID are required")
    if isinstance(scores, Mapping):
        if set(scores) != set(ids):
            raise ValueError("Score keys must match all choice IDs exactly")
        values = [scores[choice_id] for choice_id in ids]
    else:
        values = list(scores)
        if len(values) != len(ids):
            raise ValueError("Expected one scalar score for every choice")
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Scores must all be finite")
    score_map = dict(zip(ids, values))
    ranking = sorted(ids, key=lambda choice_id: (-score_map[choice_id], choice_id))
    return {"episode_id": epid, "ranking": ranking, "correct_choice_id": correct_id,
            "scores": score_map, **_rank_values(ranking, correct_id)}


def per_episode_rewards(records):
    """Recompute rewards from rankings, rejecting duplicate or malformed episodes."""
    records = list(records)
    _ids((record["episode_id"] for record in records), "prediction episode IDs")
    return {record["episode_id"]: _rank_values(record["ranking"], record["correct_choice_id"])
            ["rank_reward"] for record in records}


def metric_rows(records):
    """Aggregate five-choice rank rewards; never use a binary score threshold."""
    records = list(records)
    _ids((record["episode_id"] for record in records), "prediction episode IDs")
    metrics = [_rank_values(record["ranking"], record["correct_choice_id"])
               for record in records]
    counts = Counter(row["correct_rank"] for row in metrics)
    return {"episodes": len(metrics),
            "mean_rank_reward": math.fsum(row["rank_reward"] for row in metrics) / len(metrics),
            "top1_accuracy": sum(row["top1"] for row in metrics) / len(metrics),
            "correct_rank_counts": {str(rank): counts[rank] for rank in range(1, 6)}}


def paired_reward_interval(base_by_episode, cand_by_episode, bootstrap_samples=2000, seed=2029):
    """Paired episode bootstrap of candidate-minus-baseline mean reward.

    Inputs map the same episode IDs to their rewards averaged across all training
    seeds. Resample EPISODES jointly, not predictions/seeds as independent data.
    The interval is conditional on the fitted models; report training-seed
    variability separately. It does not correct model-selection bias or multiple
    comparisons and must not be presented as leaderboard-score uncertainty.
    """
    import numpy as np

    ids = sorted(_ids(base_by_episode, "baseline episode IDs"))
    if set(ids) != set(cand_by_episode):
        raise ValueError("Paired comparison requires identical episode coverage")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    base = np.asarray([base_by_episode[episode_id] for episode_id in ids], dtype=np.float64)
    cand = np.asarray([cand_by_episode[episode_id] for episode_id in ids], dtype=np.float64)
    if (base.shape != (len(ids),) or cand.shape != (len(ids),)
            or not np.isfinite(base).all() or not np.isfinite(cand).all()
            or (base < 0).any() or (base > 1).any() or (cand < 0).any() or (cand > 1).any()):
        raise ValueError("Episode mean rewards must be finite scalars between zero and one")
    delta = cand - base
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 256):
        stop = min(start + 256, bootstrap_samples)
        selected = rng.integers(0, len(ids), size=(stop - start, len(ids)))
        samples[start:stop] = delta[selected].mean(axis=1)
    tolerance = 1e-12
    return {
        "episodes": len(ids), "baseline_mean_reward": float(base.mean()),
        "candidate_mean_reward": float(cand.mean()), "mean_delta": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "win_tie_loss": {"win": int((delta > tolerance).sum()),
                         "tie": int((np.abs(delta) <= tolerance).sum()),
                         "loss": int((delta < -tolerance).sum())},
        "tie_tolerance": tolerance,
        "method": "paired_episode_percentile_bootstrap_of_seed_mean_rewards",
        "limitations": "Conditional on fitted models; does not include training-seed "
                       "uncertainty, model-selection bias, multiple-comparison correction, "
                       "or hidden-test distribution shift. Not a leaderboard confidence interval.",
    }
