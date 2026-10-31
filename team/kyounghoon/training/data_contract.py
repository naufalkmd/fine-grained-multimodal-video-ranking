"""Immutable, data-only contracts for an original-caption ranking baseline.

No model, training, augmentation, or competition submission is run here. The
five-choice validation set uses other original validation captions as negatives;
it is a random-negative proxy, not the hidden competition test distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
CONTRACT_NAMES = ("manifest.json", "split.json", "validation_candidates.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
EPISODE_MEMBER = re.compile(r"^(Train/episode_\d+)/(.+)$")
PROXY_WARNING = (
    "Random original-caption negatives from the validation split. Identical "
    "normalized captions are excluded, but semantic false negatives can remain. "
    "This validation score is not the hidden-test leaderboard score."
)


class ContractError(ValueError):
    """A source archive or saved immutable contract is inconsistent."""


def natural_key(name: str) -> list[Any]:
    """Use the starter kit's natural ordering, including 2 before 10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", PurePosixPath(name).name)]


def normalize_description(text: str) -> str:
    return " ".join(text.split()).casefold()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_manifest(train_zip: str | Path) -> dict[str, Any]:
    """Read small annotations and the ZIP directory; do not extract images.

    Image identity uses member names, sizes, and CRCs, without hashing all image
    payloads. CRC is an accidental-change check, not a cryptographic guarantee.
    The source fingerprint also includes decoded descriptions and SHA256 hashes
    of the metadata bytes. The archive path itself is deliberately not stored.
    """
    train_zip = Path(train_zip)
    grouped: dict[str, dict[str, zipfile.ZipInfo]] = {}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(train_zip) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if (info.filename in seen or member.is_absolute()
                    or ".." in member.parts or "\\" in info.filename):
                raise ContractError(f"Duplicate or unsafe ZIP member: {info.filename}")
            seen.add(info.filename)
            records.append({"member": info.filename, "size": info.file_size,
                            "crc32": f"{info.CRC:08x}"})
            match = EPISODE_MEMBER.fullmatch(info.filename)
            if match and not info.is_dir():
                grouped.setdefault(match[1], {})[match[2]] = info
        if not grouped:
            raise ContractError("No Train/episode_*/ entries found in source ZIP.")
        episodes = []
        for folder in sorted(grouped, key=natural_key):
            members = grouped[folder]
            if "description.txt" not in members or "episode.json" not in members:
                raise ContractError(f"Missing description.txt or episode.json: {folder}")
            description = archive.read(members["description.txt"]).decode("utf-8-sig").strip()
            if not description:
                raise ContractError(f"Empty original description: {folder}")
            metadata_bytes = archive.read(members["episode.json"])
            try:
                metadata = json.loads(metadata_bytes)
                actions = metadata["actions"]
                if not isinstance(actions, list):
                    raise TypeError("actions must be a list")
                labels = [action["label_id"] for action in actions]
                if any(type(label) is not int for label in labels):
                    raise TypeError("label_id must be an integer")
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ContractError(f"Invalid action metadata: {folder}: {error}") from error
            images = [info.filename for relative, info in members.items()
                      if relative.startswith("images/")
                      and PurePosixPath(relative).suffix.lower() in IMAGE_SUFFIXES]
            images.sort(key=natural_key)
            if not images:
                raise ContractError(f"No supported original images: {folder}")
            episodes.append({
                "episode_id": PurePosixPath(folder).name,
                "description": description,
                "label_ids": sorted(set(labels)),
                "image_members": images,
                "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            })
    records.sort(key=lambda item: item["member"])
    zip_manifest_sha256 = sha256_json(records)
    fingerprint = sha256_json({"zip_manifest_sha256": zip_manifest_sha256,
                               "episodes": episodes})
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"zip_name": train_zip.name, "zip_size": train_zip.stat().st_size,
                   "zip_member_count": len(records), "fingerprint": fingerprint,
                   "zip_manifest_sha256": zip_manifest_sha256,
                   "fingerprint_method": "member_name_size_crc32+description+metadata_sha256-v1"},
        "episode_count": len(episodes),
        "episodes": episodes,
    }


def make_split(manifest: dict[str, Any], seed: int = 2026,
               fraction: float = 0.2) -> dict[str, Any]:
    """Match starterkit.dataset.split_indices for the same ordered episodes."""
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ContractError("validation fraction must be strictly between 0 and 1.")
    episodes = manifest["episodes"]
    if len(episodes) < 4:
        raise ContractError("At least four episodes are needed for the baseline split.")
    indices = list(range(len(episodes)))
    random.Random(seed).shuffle(indices)
    size = min(len(episodes) - 2, max(2, round(len(episodes) * fraction)))
    train_indices, val_indices = sorted(indices[size:]), sorted(indices[:size])
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": manifest["source"]["fingerprint"],
        "seed": seed,
        "fraction": fraction,
        "split_method": "starterkit-natural-episode-order-python-random-shuffle-v1",
        "train_ids": [episodes[index]["episode_id"] for index in train_indices],
        "val_ids": [episodes[index]["episode_id"] for index in val_indices],
    }


def make_validation_candidates(manifest: dict[str, Any], split: dict[str, Any],
                               candidate_seed: int | None = None) -> dict[str, Any]:
    """One positive and four distinct original captions, only inside validation.

    The origins are shuffled before candidate IDs are assigned, and the display
    order is independently shuffled afterwards. Neither position nor ID encodes
    the positive. Correct IDs and source IDs are evaluation metadata only.
    """
    seed = split["seed"] + 1 if candidate_seed is None else candidate_seed
    rng = random.Random(seed)
    by_id = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    validation = [by_id[episode_id] for episode_id in split["val_ids"]]
    by_text: dict[str, list[dict[str, Any]]] = {}
    for episode in validation:
        by_text.setdefault(normalize_description(episode["description"]), []).append(episode)
    if len(by_text) < 5:
        raise ContractError("Five-choice validation needs at least five distinct normalized validation captions.")
    groups = list(by_text)
    candidate_episodes = []
    for episode in validation:
        correct_normalized = normalize_description(episode["description"])
        eligible = [text for text in groups if text != correct_normalized]
        negative_texts = rng.sample(eligible, 4)
        origins = [episode] + [rng.choice(by_text[text]) for text in negative_texts]
        rng.shuffle(origins)
        choices = []
        correct_choice_id = None
        for index, origin in enumerate(origins):
            choice_id = f"choice_{index}"
            choices.append({"id": choice_id, "text": origin["description"],
                            "source_episode_id": origin["episode_id"]})
            if origin["episode_id"] == episode["episode_id"]:
                correct_choice_id = choice_id
        rng.shuffle(choices)
        candidate_episodes.append({"episode_id": episode["episode_id"],
                                   "choices": choices,
                                   "correct_choice_id": correct_choice_id})
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": manifest["source"]["fingerprint"],
        "split_seed": split["seed"],
        "candidate_seed": seed,
        "negative_policy": "random_original_within_validation_proxy",
        "warning": PROXY_WARNING,
        "episodes": candidate_episodes,
    }


def inference_choices(record: dict[str, Any]) -> list[dict[str, str]]:
    """Whitelist only the server-compatible candidate fields for a model."""
    return [{"id": choice["id"], "text": choice["text"]}
            for choice in record["choices"]]


def validate_contracts(manifest: dict[str, Any], split: dict[str, Any],
                       candidates: dict[str, Any]) -> None:
    for document in (manifest, split, candidates):
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ContractError("Unsupported data contract schema version.")
    source = manifest["source"]["fingerprint"]
    if any(document.get("source_fingerprint") != source for document in (split, candidates)):
        raise ContractError("Contract source fingerprints disagree.")
    episodes = manifest["episodes"]
    by_id = {episode["episode_id"]: episode for episode in episodes}
    if len(by_id) != len(episodes) or manifest["episode_count"] != len(episodes):
        raise ContractError("Duplicate episode IDs or incorrect episode_count.")
    content_fingerprint = sha256_json({
        "zip_manifest_sha256": manifest["source"]["zip_manifest_sha256"],
        "episodes": episodes,
    })
    if content_fingerprint != source:
        raise ContractError("Manifest content differs from its saved source fingerprint.")
    expected_split = make_split(manifest, split["seed"], split["fraction"])
    if split != expected_split:
        raise ContractError("Saved split differs from the deterministic baseline contract.")
    train_ids, val_ids = set(split["train_ids"]), set(split["val_ids"])
    if train_ids & val_ids or train_ids | val_ids != set(by_id):
        raise ContractError("Split is overlapping or does not cover all episodes.")
    records = candidates["episodes"]
    if [record["episode_id"] for record in records] != split["val_ids"]:
        raise ContractError("Candidate episodes do not match the validation split order.")
    for record in records:
        own = by_id[record["episode_id"]]
        choices = record["choices"]
        choice_ids = [choice["id"] for choice in choices]
        if len(choices) != 5 or len(set(choice_ids)) != 5:
            raise ContractError("Each validation episode needs five unique candidate IDs.")
        if len({normalize_description(choice["text"]) for choice in choices}) != 5:
            raise ContractError("Validation candidates contain identical normalized text.")
        if record["correct_choice_id"] not in choice_ids:
            raise ContractError("Validation correct_choice_id is missing.")
        positives = 0
        for choice in choices:
            origin_id = choice["source_episode_id"]
            if origin_id not in val_ids:
                raise ContractError("Validation negative originated outside the validation split.")
            if choice["text"] != by_id[origin_id]["description"]:
                raise ContractError("A validation caption differs from its original description.")
            is_own = origin_id == own["episode_id"]
            positives += is_own
            if (choice["id"] == record["correct_choice_id"]) != is_own:
                raise ContractError("Validation positive metadata is inconsistent.")
        if positives != 1:
            raise ContractError("Validation episode must contain exactly one original positive.")
    expected_candidates = make_validation_candidates(manifest, split, candidates["candidate_seed"])
    if candidates != expected_candidates:
        raise ContractError("Saved candidate set differs from the deterministic contract.")


def load_contracts(output_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read saved contracts and validate split/candidate consistency.

    This does not reopen the source ZIP. Rerun the CLI to verify source identity.
    """
    root = Path(output_dir)
    documents = tuple(json.loads((root / name).read_text(encoding="utf-8"))
                      for name in CONTRACT_NAMES)
    validate_contracts(*documents)
    return documents


def build_contracts(train_zip: str | Path, output_dir: str | Path, seed: int = 2026,
                    fraction: float = 0.2, candidate_seed: int | None = None
                    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = read_manifest(train_zip)
    split = make_split(manifest, seed, fraction)
    candidates = make_validation_candidates(manifest, split, candidate_seed)
    documents = (manifest, split, candidates)
    validate_contracts(*documents)
    root = Path(output_dir)
    # Check every pre-existing file before writing any missing file. A partial
    # interrupted first run can be resumed only when the existing files match.
    for name, document in zip(CONTRACT_NAMES, documents):
        path = root / name
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ContractError(f"Unreadable existing immutable contract: {path}") from error
            if existing != document:
                raise ContractError(f"Existing immutable contract differs; choose a new output directory: {path}")
    root.mkdir(parents=True, exist_ok=True)
    for name, document in zip(CONTRACT_NAMES, documents):
        path = root / name
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-seed", type=int)
    args = parser.parse_args()
    manifest, split, candidates = build_contracts(
        args.train_zip, args.output_dir, args.seed, args.validation_fraction, args.candidate_seed,
    )
    print(json.dumps({"episodes": manifest["episode_count"],
                      "frames": sum(len(episode["image_members"]) for episode in manifest["episodes"]),
                      "train": len(split["train_ids"]), "validation": len(split["val_ids"]),
                      "validation_choices_per_episode": 5,
                      "source_fingerprint": manifest["source"]["fingerprint"],
                      "negative_policy": candidates["negative_policy"],
                      "output_dir": str(args.output_dir.resolve())}, indent=2))
    print(PROXY_WARNING)


if __name__ == "__main__":
    main()
