"""Benchmark-neutral integrity helpers for dataset materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class MaterializationError(RuntimeError):
    """Raised when a source or materialized dataset violates its contract."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"failed to read JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"JSON metadata must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MaterializationError(f"failed to read JSONL metadata: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MaterializationError(f"{path}:{line_number}: invalid JSONL row") from exc
        if not isinstance(value, dict):
            raise MaterializationError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "MaterializationError",
    "canonical_json",
    "file_sha256",
    "json_sha256",
    "read_json",
    "read_jsonl",
]
