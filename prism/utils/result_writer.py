"""Crash-safe JSON result publication shared by benchmark entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json_result_atomic(path: str | Path, payload: Any) -> Path:
    """Publish pretty JSON through an fsynced sibling and atomic replacement."""

    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary result file already exists: {temporary}")
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, result_path)
        _fsync_directory(result_path.parent)
    except BaseException:
        if os.path.lexists(temporary):
            temporary.unlink()
        raise
    return result_path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["write_json_result_atomic"]
