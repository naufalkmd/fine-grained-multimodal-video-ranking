"""Atomic, versioned experiment records. Only local research artifacts are written."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing artifact has different content; use a new RUN_NAME: {path}")
    else:
        atomic_json(path, value)


def sources_hash():
    root = Path(__file__).resolve().parent
    return {p.name: file_sha(p) for p in sorted(root.glob("*.py"))}


def verified_file(path, metadata):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != metadata["bytes"] or file_sha(path) != metadata["sha256"]:
        raise ValueError(f"Missing or corrupt artifact: {path}")
    return path


def file_record(path):
    return {"bytes": Path(path).stat().st_size, "sha256": file_sha(path)}
