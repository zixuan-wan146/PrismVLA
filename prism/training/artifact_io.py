"""Content manifests and crash-safe filesystem operations for training artifacts."""

from __future__ import annotations

from collections.abc import Collection, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from prism.data.normalization import canonical_json_bytes


MANIFEST_FORMAT = "prism-checkpoint-manifest-v1"
MANIFEST_FILENAME = "prism_manifest.json"
_HASH_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def build_manifest(directory: Path) -> dict[str, Any]:
    """Build and fsync a complete manifest for ``directory``.

    The manifest itself is excluded so it can be atomically written after all
    payload files have reached durable storage.
    """

    rows: list[dict[str, Any]] = []
    for file_path in _artifact_files(directory, exclude_manifest=True):
        _fsync_file(file_path)
        relative = file_path.relative_to(directory).as_posix()
        rows.append(
            {
                "path": relative,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    if not rows:
        raise RuntimeError("cannot create a checkpoint manifest without files")
    return {"format": MANIFEST_FORMAT, "files": rows}


def verify_manifest(
    directory: Path,
    *,
    required_paths: Collection[str] = (),
) -> None:
    """Verify the declared hashes and exact file set for an artifact directory."""

    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = _strict_mapping(read_json(manifest_path, label="checkpoint manifest"), "manifest")
    _expect_keys(manifest, {"format", "files"}, "checkpoint manifest")
    if manifest["format"] != MANIFEST_FORMAT:
        raise ValueError(f"unsupported checkpoint manifest format {manifest['format']!r}")
    rows = manifest["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("checkpoint manifest files must be a non-empty list")

    declared: set[str] = set()
    for index, value in enumerate(rows):
        row = _strict_mapping(value, f"checkpoint manifest row {index}")
        _expect_keys(row, {"path", "size_bytes", "sha256"}, f"checkpoint manifest row {index}")
        relative = _safe_relative_path(row["path"], f"checkpoint manifest row {index} path")
        if relative in declared:
            raise ValueError(f"checkpoint manifest contains duplicate path {relative!r}")
        declared.add(relative)
        size = _non_negative_int(row["size_bytes"], f"checkpoint manifest row {index} size")
        _stored_sha(row["sha256"], f"checkpoint manifest row {index} SHA256")
        file_path = directory / relative
        if not file_path.is_file() or file_path.is_symlink():
            raise FileNotFoundError(f"checkpoint manifest file is missing or unsafe: {file_path}")
        actual_size = file_path.stat().st_size
        if actual_size != size:
            raise ValueError(
                f"checkpoint file size mismatch for {relative}: stored {size}, computed {actual_size}"
            )
        actual_hash = sha256_file(file_path)
        if actual_hash != row["sha256"]:
            raise ValueError(
                f"checkpoint file hash mismatch for {relative}: stored {row['sha256']}, computed {actual_hash}"
            )

    actual = {
        file_path.relative_to(directory).as_posix()
        for file_path in _artifact_files(directory, exclude_manifest=True)
    }
    if actual != declared:
        raise ValueError(
            "checkpoint manifest file set mismatch: "
            f"missing={sorted(declared - actual)}, unexpected={sorted(actual - declared)}"
        )
    missing_required = sorted(set(required_paths) - declared)
    if missing_required:
        raise ValueError(f"checkpoint manifest is missing required files: {missing_required}")


def write_json_atomic(path: Path, value: Any) -> None:
    """Write canonical JSON through an fsynced sibling and atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary JSON file already exists: {temporary}")
    payload = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        if os.path.lexists(temporary):
            temporary.unlink()
        raise


def read_json(path: Path, *, label: str) -> Any:
    """Read JSON with a diagnostic that identifies the artifact role."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {label} from {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """Durably publish directory-entry changes on POSIX filesystems."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_files(directory: Path, *, exclude_manifest: bool) -> list[Path]:
    output: list[Path] = []
    for entry in directory.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"checkpoint may not contain symbolic links: {entry}")
        if entry.is_file():
            if exclude_manifest and entry == directory / MANIFEST_FILENAME:
                continue
            output.append(entry)
        elif not entry.is_dir():
            raise ValueError(f"checkpoint contains an unsupported filesystem entry: {entry}")
    return sorted(output, key=lambda value: value.relative_to(directory).as_posix())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return dict(value)


def _expect_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise ValueError(f"{label} keys mismatch: missing={missing}, unknown={unknown}")


def _stored_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA256")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative POSIX path, got {value!r}")
    return value


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT",
    "build_manifest",
    "fsync_directory",
    "read_json",
    "sha256_file",
    "verify_manifest",
    "write_json_atomic",
]
