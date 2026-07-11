from __future__ import annotations

# --- migrated from src/prism/dataset/cache_utils.py ---
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from prism.utils.paths import display_project_path, find_repo_root


CACHE_FORMAT_VERSION = 3
DEFAULT_CACHE_RELATIVE_PATH = Path("run_outputs") / "training_data_cache"


def default_dataset_cache_dir(repo_root: str | Path | None = None) -> Path:
    base_dir = find_repo_root(__file__) if repo_root is None else Path(repo_root).expanduser()
    return base_dir / DEFAULT_CACHE_RELATIVE_PATH


def dataset_cache_namespace(
    dataset_config: Mapping[str, Any],
    dataset_path: str | Path,
    *,
    action_horizon: int,
    max_samples_per_file: int | None,
    action_segment_config: Mapping[str, Any] | None = None,
) -> str:
    repo_root = find_repo_root(__file__)
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "dataset_path": display_project_path(dataset_path, repo_root),
        "dataset_config": _jsonable(dataset_config),
        "action_horizon": int(action_horizon),
        "max_samples_per_file": max_samples_per_file,
    }
    if action_segment_config:
        payload["action_segment_config"] = _jsonable(action_segment_config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"v{CACHE_FORMAT_VERSION}_{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value

