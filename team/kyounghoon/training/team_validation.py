"""Adapt the team's unmodified TrainingNegatives sampler to fixed VAL-only choices.

The sampler receives validation originals and validation action labels only.
Its output is retained verbatim: token-input collisions are audited rather than
removed or resampled, preserving the team's selected distribution.
This module does not write files, encode images, or generate descriptions itself.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from team.train_negatives import TrainingNegatives, check_mix, parse_mix

UPSTREAM_REPOSITORY = "https://github.com/naufalkmd/team07-starterkit"
UPSTREAM_BRANCH = "feat/lexical-negatives"
UPSTREAM_COMMIT = "00afd4ccad21fa8a4f62108583de1bd700cb43aa"
VENDOR_FILES = ("train_negatives.py", "negatives.py", "lexical.py")
POLICY = "team_training_negatives_fixed_validation_epoch_v1"


def _hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_source_id(text):
    """Stable feature-bank key for a rule-generated description, not an episode."""
    return "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized(text):
    return " ".join(text.split()).casefold()


def _unique_ids(values, label):
    values = list(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} requires nonempty string IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate IDs in {label}")
    return values


def _token_key(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"Missing or invalid token-input SHA256 for {label}")
    return value.lower()


def build_team_candidates(manifest, split, token_keys,
                          mix_text="lexical=2,similar_episode=2",
                          fill="random", seed=2026, epoch=0, *,
                          action_labels=None, token_key_fn=None):
    """Return fixed five-choice candidates sampled by the original team class.

    action_labels is required and maps IDs to the full repeated action-label
    sequence. The manifest's deduplicated label_ids are not used.
    token_keys maps original validation IDs to actual truncate64 input hashes.
    A mix that generates new strings also requires token_key_fn(text).

    All sampler outputs are retained verbatim. Distinct text can become identical
    after truncation; such collisions are diagnostics, never hidden changes to
    the team's sampling or to the existing random candidate set.
    """
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("seed and epoch must be nonnegative integers")
    mix = parse_mix(mix_text)
    check_mix(mix, 4, fill)
    rows = manifest["episodes"]
    all_ids = _unique_ids((row["episode_id"] for row in rows), "manifest")
    by_id = dict(zip(all_ids, rows))
    val_ids = _unique_ids(split["val_ids"], "validation")
    train_ids = _unique_ids(split["train_ids"], "training")
    if len(val_ids) < 5:
        raise ValueError("At least five validation episodes are required")
    if set(val_ids) & set(train_ids):
        raise ValueError("Training and validation IDs overlap")
    if not (set(val_ids) | set(train_ids)) <= set(by_id):
        raise ValueError("Split refers to an episode absent from the manifest")
    fingerprint = manifest["source"]["fingerprint"]
    if split["source_fingerprint"] != fingerprint:
        raise ValueError("Manifest and split fingerprints differ")
    if action_labels is None:
        raise ValueError("Full repeated action_labels are required; unique label_ids are insufficient")
    descriptions, labels, keys = [], [], {}
    for episode_id in val_ids:
        text = by_id[episode_id]["description"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Validation descriptions must be nonempty original strings")
        descriptions.append(text)
        values = action_labels.get(episode_id)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Missing repeated action-label list for {episode_id}")
        if any(not isinstance(label, (str, int)) for label in values):
            raise ValueError("Action labels must be strings or integers")
        labels.append(list(values))
        keys[episode_id] = _token_key(token_keys.get(episode_id), episode_id)
    if len(set(descriptions)) != len(descriptions):
        raise ValueError("Duplicate original validation descriptions make their source ambiguous")
    source_by_text = dict(zip(descriptions, val_ids))
    sampler = TrainingNegatives(val_ids, descriptions, labels, mix, 4, fill=fill, seed=seed)
    sampled = sampler.epoch(epoch)
    if len(sampled) != len(val_ids):
        raise ValueError("Team sampler returned the wrong validation coverage")
    records, counts, collision_records = [], Counter(), []
    generated_texts = set()
    actual_counts_by_episode = {}
    for episode_id, positive, negatives in zip(val_ids, descriptions, sampled):
        if len(negatives) != 4:
            raise ValueError(f"Team sampler did not return four negatives for {episode_id}")
        texts = [positive, *(text for _, text in negatives)]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError(f"Team sampler returned an empty or invalid description for {episode_id}")
        if len({_normalized(text) for text in texts}) != 5:
            raise ValueError(f"Team candidates have normalized duplicate text for {episode_id}")
        typed = [("positive", positive), *negatives]
        rng = random.Random(f"{seed}:{episode_id}:fixed-validation:{epoch}")
        rng.shuffle(typed)
        choices, choice_types, input_keys = [], {}, {}
        correct_id = None
        for position, (negative_type, text) in enumerate(typed):
            choice_id = f"choice_{position}"
            origin = source_by_text.get(text)
            if origin is None:
                if token_key_fn is None:
                    raise ValueError("Rule-generated descriptions require token_key_fn for exact-input audit")
                origin = text_source_id(text)
                input_key = _token_key(token_key_fn(text), origin)
                generated_texts.add(text)
            else:
                input_key = keys[origin]
            choices.append({"id": choice_id, "text": text, "source_episode_id": origin})
            choice_types[choice_id] = negative_type
            input_keys[choice_id] = input_key
            if negative_type == "positive":
                if text != positive:
                    raise ValueError("Positive caption changed")
                correct_id = choice_id
        rng.shuffle(choices)
        grouped_keys = {}
        for choice_id, key in input_keys.items():
            grouped_keys.setdefault(key, []).append(choice_id)
        collision_groups = [ids for ids in grouped_keys.values() if len(ids) > 1]
        positive_collision = any(correct_id in ids for ids in collision_groups)
        if collision_groups:
            collision_records.append({"episode_id": episode_id,
                                      "choice_collision_groups": collision_groups,
                                      "positive_negative_collision": positive_collision})
        used = Counter(negative_type for negative_type, _ in negatives)
        counts.update(used)
        actual_counts_by_episode[episode_id] = dict(used)
        records.append({"episode_id": episode_id, "choices": choices,
                        "correct_choice_id": correct_id,
                        "mining": {"choice_types": choice_types,
                                   "requested_mix": dict(mix), "actual_negative_types": dict(used),
                                   "token_collision_groups": collision_groups,
                                   "positive_negative_token_collision": positive_collision}})
    vendor_root = Path(__file__).resolve().parent / "team"
    hashes = {name: hashlib.sha256((vendor_root / name).read_text(encoding="utf-8").encode()).hexdigest()
              for name in VENDOR_FILES}
    return {
        "schema_version": 1, "source_fingerprint": fingerprint,
        "split_seed": split["seed"], "candidate_seed": seed, "candidate_epoch": epoch,
        "negative_policy": POLICY,
        "warning": ("Fixed validation diagnostic produced by the team's original sampler. "
                    "Other descriptions or rule changes are not verified to be semantically false. "
                    "Exact token collisions are retained and audited; this is not a leaderboard score."),
        "upstream": {"repository": UPSTREAM_REPOSITORY, "branch": UPSTREAM_BRANCH,
                     "commit": UPSTREAM_COMMIT, "vendor_sha256": hashes,
                     "source_newlines": "universal_newlines_to_LF",
                     "class": "team.train_negatives.TrainingNegatives",
                     "sampler_rng_namespace": "unchanged upstream ':train:' namespace"},
        "sampling": {"pool": "validation_only", "k": 4, "mix": dict(mix),
                     "fill": fill, "seed": seed, "epoch": epoch,
                     "action_label_multiplicity_preserved": True,
                     "validation_action_labels_sha256": _hash(dict(zip(val_ids, labels))),
                     "validation_token_keys_sha256": _hash(keys),
                     "requested_mix": {name: value for name, value in mix.items() if value},
                     "types_used": dict(counts), "types_used_by_episode": actual_counts_by_episode,
                     "generated_unique_texts": len(generated_texts),
                     "token_collision_policy": "retain upstream outputs; audit without resampling"},
        "mining_diagnostics": {
            "episodes": len(records), "semantic_false_negatives_verified": False,
            "any_choice_token_collision_episodes": len(collision_records),
            "positive_negative_token_collision_episodes":
                sum(row["positive_negative_collision"] for row in collision_records),
            "token_collisions": collision_records},
        "episodes": records,
    }
