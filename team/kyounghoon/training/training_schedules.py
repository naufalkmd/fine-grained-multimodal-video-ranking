"""Shared, reproducible training-only negative schedules for the two-way ablation.

The team's sampler is used unchanged. Only original-caption types are supported
in this experiment, so every recorded negative resolves to an original TRAIN
episode and its frozen text features. Every epoch is saved once and reused by
all A-D heads. Validation descriptions and labels are never passed to a sampler.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import uuid
import zlib

from experiment_io import atomic_json, digest, file_record, immutable_json, read_json, verified_file
from team.train_negatives import TrainingNegatives, check_mix, parse_mix

ALLOWED_TYPES = {"lexical", "similar_episode", "random"}
UPSTREAM_COMMIT = "00afd4ccad21fa8a4f62108583de1bd700cb43aa"
VERSION = "training-only-original-caption-schedules-v1"


def _normal(text):
    return " ".join(text.split()).casefold()


def _source_hashes():
    root = Path(__file__).resolve().parent
    paths = ("training_schedules.py", "team/train_negatives.py", "team/negatives.py", "team/lexical.py")
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in paths}


def _callable_record(function):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        raise ValueError("random_schedule_fn must have inspectable source for reproducibility")
    return {"module": function.__module__, "qualname": function.__qualname__,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def _default_random_schedule(train_ids, token_keys, seed, epoch):
    """Match the previous comparison's random-negative schedule exactly."""
    pool = list(train_ids)
    rng = random.Random(seed * 1_000_003 + epoch)
    order = list(pool)
    rng.shuffle(order)
    rows = []
    if len({token_keys[x] for x in pool}) < 5:
        raise ValueError("Need at least five distinct training text inputs")
    for positive in order:
        choices, used = [positive], {token_keys[positive]}
        attempts = 0
        while len(choices) < 5:
            candidate = rng.choice(pool)
            attempts += 1
            if token_keys[candidate] not in used:
                choices.append(candidate)
                used.add(token_keys[candidate])
            if attempts > 1000 and len(choices) < 5:
                remaining = [x for x in pool if token_keys[x] not in used]
                if not remaining:
                    raise ValueError("Insufficient noncolliding training negatives")
                candidate = rng.choice(remaining)
                choices.append(candidate)
                used.add(token_keys[candidate])
        rows.append((positive, choices))
    return rows


class TrainingSchedules:
    def __init__(self, train_ids, descriptions, action_labels, token_keys, output_dir, *,
                 mix_text="lexical=2,similar_episode=2", fill="random", random_schedule_fn=None):
        self.train_ids = list(train_ids)
        if (len(self.train_ids) < 5 or len(set(self.train_ids)) != len(self.train_ids) or
                any(not isinstance(ep, str) or not ep for ep in self.train_ids)):
            raise ValueError("Need at least five unique nonempty training episode IDs")
        expected = set(self.train_ids)
        if set(descriptions) != expected or set(action_labels) != expected:
            raise ValueError("Descriptions and action labels must contain exactly the TRAIN IDs")
        self.descriptions = {ep: descriptions[ep] for ep in self.train_ids}
        if any(not isinstance(text, str) or not text.strip() for text in self.descriptions.values()):
            raise ValueError("Training descriptions must be nonempty original strings")
        if len({_normal(text) for text in self.descriptions.values()}) != len(self.train_ids):
            raise ValueError("Training captions must be distinct after normalization")
        self.action_labels = {}
        self.token_keys = {}
        for ep in self.train_ids:
            labels = action_labels[ep]
            if (not isinstance(labels, (list, tuple)) or
                    any(type(label) not in (str, int) for label in labels)):
                raise ValueError("Action labels must be full repeated lists of strings or integers")
            self.action_labels[ep] = list(labels)
            key = token_keys.get(ep)
            if not isinstance(key, str) or re.fullmatch(r"[0-9a-fA-F]{64}", key) is None:
                raise ValueError(f"Missing actual token-input SHA256 for training episode {ep}")
            self.token_keys[ep] = key.lower()
        self.mix = parse_mix(mix_text)
        check_mix(self.mix, 4, fill)
        if not set(self.mix) <= ALLOWED_TYPES or fill not in ALLOWED_TYPES:
            raise ValueError("This training ablation supports lexical, similar_episode, and random original captions only")
        self.fill = fill
        self.random_schedule_fn = random_schedule_fn or _default_random_schedule
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "schema": 1, "version": VERSION, "pool": "training_only", "episodes": len(self.train_ids),
            "k": 4, "team_mix": dict(self.mix),
            "team_mix_order": [[name, count] for name, count in self.mix.items()], "team_fill": fill,
            "episode_order": "Random(seed*1000003+epoch).shuffle(original_train_ids), shared across modes",
            "team_sampler": "unchanged team.train_negatives.TrainingNegatives.epoch(epoch)",
            "team_sampler_rng": "unchanged ':train:' namespace", "upstream_commit": UPSTREAM_COMMIT,
            "inputs_sha256": {"train_ids": digest(self.train_ids), "descriptions": digest(self.descriptions),
                              "repeated_action_labels": digest(self.action_labels), "token_keys": digest(self.token_keys)},
            "source_sha256": _source_hashes(),
            "random_schedule": _callable_record(self.random_schedule_fn),
            "token_collisions": "team outputs retained and audited; random inputs must be distinct",
        }
        self.fingerprint = digest(self.metadata)
        immutable_json(self.output_dir / "request.json", {**self.metadata, "fingerprint": self.fingerprint})
        self._sampler = None
        self._source_by_text = {text: ep for ep, text in self.descriptions.items()}

    def _order(self, seed, epoch):
        order = list(self.train_ids)
        random.Random(seed * 1_000_003 + epoch).shuffle(order)
        return order

    def _team_records(self, seed, epoch):
        if self._sampler is None:
            self._sampler = TrainingNegatives(
                self.train_ids, [self.descriptions[ep] for ep in self.train_ids],
                [self.action_labels[ep] for ep in self.train_ids],
                self.mix, 4, fill=self.fill, seed=seed)
        # Seed only controls fresh per-episode local Random instances. The
        # generator's immutable TRAIN-only action counts and lexical matrix
        # can safely be shared across seeds and epochs.
        self._sampler.seed = seed
        sampled = self._sampler.epoch(epoch)
        if len(sampled) != len(self.train_ids):
            raise ValueError("Team sampler returned the wrong training coverage")
        by_id = {}
        for ep, negatives in zip(self.train_ids, sampled):
            sources = []
            for negative_type, text in negatives:
                if text not in self._source_by_text:
                    raise ValueError("Training negative is not an original TRAIN caption")
                sources.append({"type": negative_type, "source": self._source_by_text[text]})
            by_id[ep] = {"episode_id": ep, "negative_sources": sources}
        return [by_id[ep] for ep in self._order(seed, epoch)]

    def _random_records(self, seed, epoch):
        schedule = self.random_schedule_fn(self.train_ids, self.token_keys, seed, epoch)
        records = []
        for ep, choices in schedule:
            if len(choices) != 5 or choices[0] != ep:
                raise ValueError("Random callback must put its positive first and return four negatives")
            records.append({"episode_id": ep, "negative_sources": [{"type": "random", "source": source}
                                                                   for source in choices[1:]]})
        return records

    def _checked(self, records, mode, seed, epoch):
        if [record["episode_id"] for record in records] != self._order(seed, epoch):
            raise ValueError("Training records must have the shared shuffled TRAIN order")
        expected = set(self.train_ids)
        schedule, counts, positive_collisions, any_collisions = [], Counter(), [], []
        for record in records:
            ep = record["episode_id"]
            negatives = record["negative_sources"]
            if len(negatives) != 4:
                raise ValueError("Every training episode requires four negatives")
            sources = [ep]
            for negative in negatives:
                if (negative["type"] not in ALLOWED_TYPES or negative["source"] not in expected or
                        negative["source"] == ep):
                    raise ValueError("Training negatives must be other original TRAIN episodes")
                if mode == "random" and negative["type"] != "random":
                    raise ValueError("Random schedules must contain only random negative types")
                sources.append(negative["source"])
                counts[negative["type"]] += 1
            if len({_normal(self.descriptions[source]) for source in sources}) != 5:
                raise ValueError("Training choices contain duplicate normalized text")
            keys = [self.token_keys[source] for source in sources]
            if keys[0] in keys[1:]:
                positive_collisions.append(ep)
            if len(set(keys)) != 5:
                any_collisions.append(ep)
            if mode == "random" and len(set(keys)) != 5:
                raise ValueError("Random callback returned colliding model inputs")
            schedule.append((ep, sources))
        audit = {
            "mode": mode, "seed": seed, "epoch": epoch, "episodes": len(records),
            "schedule_hash": digest(schedule), "types_used": dict(counts),
            "positive_negative_token_collision_episodes": len(positive_collisions),
            "any_choice_token_collision_episodes": len(any_collisions),
            "positive_negative_token_collision_ids": positive_collisions,
            "any_choice_token_collision_ids": any_collisions,
            "pool": "training_only", "config_fingerprint": self.fingerprint,
        }
        return schedule, audit

    def _read(self, path, marker, mode, seed, epoch):
        if marker.exists():
            metadata = read_json(marker)
            verified_file(path, metadata["file"])
        try:
            data = json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError, EOFError, zlib.error) as error:
            raise ValueError(f"Corrupt negative schedule: {path}") from error
        payload = {k: v for k, v in data.items() if k != "payload_sha256"}
        if data.get("payload_sha256") != digest(payload):
            raise ValueError("Negative-schedule payload checksum differs")
        if (data.get("config_fingerprint") != self.fingerprint or data.get("mode") != mode or
                data.get("seed") != seed or data.get("epoch") != epoch):
            raise ValueError("Negative schedule belongs to another configuration")
        schedule, audit = self._checked(data["records"], mode, seed, epoch)
        if data["audit"] != audit:
            raise ValueError("Negative-schedule audit does not agree with actual choices")
        return schedule, audit

    def epoch(self, mode, seed, epoch):
        if mode not in {"random", "team"}:
            raise ValueError("training mode must be random or team")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1 or type(epoch) is not int or epoch < 0:
            raise ValueError("seed must be a uint32 integer and epoch a nonnegative integer")
        path = self.output_dir / mode / f"seed{seed}" / f"epoch-{epoch:03d}.json.gz"
        marker = path.with_name(path.name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists() and not path.exists():
            raise ValueError("A committed negative schedule is missing")
        if path.exists():
            schedule, audit = self._read(path, marker, mode, seed, epoch)
        else:
            records = self._random_records(seed, epoch) if mode == "random" else self._team_records(seed, epoch)
            schedule, audit = self._checked(records, mode, seed, epoch)
            payload = {"schema": 1, "config_fingerprint": self.fingerprint, "mode": mode,
                       "seed": seed, "epoch": epoch, "records": records, "audit": audit}
            stored = {**payload, "payload_sha256": digest(payload)}
            encoded = json.dumps(stored, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode()
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(gzip.compress(encoded, mtime=0))
            os.replace(temporary, path)
        # A payload committed just before interruption can recreate its marker
        # after validating its self-checksum and all choice/audit invariants.
        file = {"path": path.relative_to(self.output_dir).as_posix(), **file_record(path)}
        immutable_json(marker, {"config_fingerprint": self.fingerprint, "file": file})
        return schedule, {**audit, "file": file}
