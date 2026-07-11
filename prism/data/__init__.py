from __future__ import annotations

# --- migrated from src/prism/image_preprocessing.py ---
from typing import Any

import numpy as np
from PIL import Image


def rgb_array_to_pil(image: Any, image_size: int) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"RGB image must contain numeric pixel values, got dtype={array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("RGB image must contain only finite pixel values")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("RGB image pixel values must be in the 0..255 range")
    pil_image = Image.fromarray(array.astype(np.uint8, copy=False))
    return pil_image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)

# --- migrated from src/prism/dataset/config_utils.py ---
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


REQUIRED_POSITIVE_INT_KEYS = (
    "max_action_dim",
    "max_state_dim",
    "max_views",
)

DATASET_PATH_KEYS = ("path", "boundary_path")


def resolve_dataset_path(raw_path: str | Path, base_dir: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        raise ValueError(f"dataset path must be project-relative: {raw_path}")
    if path.parts and path.parts[0] == "..":
        raise ValueError(f"dataset path must stay inside the project: {raw_path}")
    return (Path(base_dir).expanduser() / path).resolve()


def iter_dataset_entries(config: Mapping[str, Any]) -> Iterator[tuple[Any, Any, Mapping[str, Any]]]:
    data_groups = config.get("data_groups")
    if not isinstance(data_groups, Mapping) or not data_groups:
        raise ValueError("data_groups must be a non-empty mapping")

    for group_name, group_config in data_groups.items():
        if not isinstance(group_config, Mapping) or not group_config:
            raise ValueError(f"data group {group_name!r} must contain datasets")
        for dataset_name, dataset_config in group_config.items():
            if not isinstance(dataset_config, Mapping):
                raise ValueError(f"dataset {group_name}/{dataset_name} must be a mapping")
            yield group_name, dataset_name, dataset_config


def validate_dataset_config_structure(config: Mapping[str, Any]) -> int:
    if not isinstance(config, Mapping):
        raise TypeError("dataset config must be a mapping")

    for key in REQUIRED_POSITIVE_INT_KEYS:
        value = config.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")

    dataset_count = 0
    for group_name, dataset_name, dataset_config in iter_dataset_entries(config):
        dataset_count += 1
        raw_path = dataset_config.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"dataset {group_name}/{dataset_name} has no path")
        for path_key in DATASET_PATH_KEYS:
            raw_value = dataset_config.get(path_key)
            if raw_value in (None, ""):
                continue
            if not isinstance(raw_value, str):
                raise ValueError(f"dataset {group_name}/{dataset_name} {path_key} must be a string")
            normalized = Path(raw_value).expanduser()
            if normalized.is_absolute():
                raise ValueError(f"dataset {group_name}/{dataset_name} {path_key} must be project-relative")
            if normalized.parts and normalized.parts[0] == "..":
                raise ValueError(f"dataset {group_name}/{dataset_name} {path_key} must stay inside the project")

    return dataset_count


def resolve_dataset_config_paths(config: Mapping[str, Any], base_dir: str | Path) -> dict[str, Any]:
    validate_dataset_config_structure(config)

    resolved_config = deepcopy(dict(config))
    for group_name, dataset_name, _dataset_config in iter_dataset_entries(resolved_config):
        dataset_config = resolved_config["data_groups"][group_name][dataset_name]
        for path_key in DATASET_PATH_KEYS:
            raw_value = dataset_config.get(path_key)
            if raw_value not in (None, ""):
                dataset_config[path_key] = str(resolve_dataset_path(raw_value, base_dir))

    return resolved_config

# --- migrated from src/prism/dataset/validation.py ---
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping



DEFAULT_VIEW_MAP = {
    "image_1": "observation.images.image_1",
    "image_2": "observation.images.image_2",
    "image_3": "observation.images.image_3",
}


@dataclass(frozen=True)
class DatasetValidationIssue:
    level: str
    path: str
    message: str


def validate_configured_datasets(
    config: Mapping[str, Any],
    base_dir: str | Path,
    *,
    require_videos: bool = True,
) -> list[DatasetValidationIssue]:
    validate_dataset_config_structure(config)
    resolved_config = resolve_dataset_config_paths(config, base_dir)
    issues: list[DatasetValidationIssue] = []

    max_state_dim = int(resolved_config["max_state_dim"])
    max_action_dim = int(resolved_config["max_action_dim"])
    for group_name, dataset_name, dataset_config in iter_dataset_entries(resolved_config):
        dataset_label = f"{group_name}/{dataset_name}"
        dataset_path = Path(str(dataset_config["path"]))
        view_map = dataset_config.get("view_map") or DEFAULT_VIEW_MAP
        if not isinstance(view_map, Mapping) or not view_map:
            issues.append(DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} view_map must be a mapping"))
            continue
        issues.extend(
            validate_dataset_path(
                dataset_path,
                dataset_label=dataset_label,
                view_map=view_map,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                require_videos=require_videos,
                state_stat_keys=dataset_config.get("state_stat_keys", ("observation.state", "state")),
                action_stat_keys=dataset_config.get("action_stat_keys", ("action", "actions")),
            )
        )
    return issues


def validate_dataset_path(
    dataset_path: Path,
    *,
    dataset_label: str,
    view_map: Mapping[str, Any],
    max_state_dim: int,
    max_action_dim: int,
    require_videos: bool,
    state_stat_keys: Any = ("observation.state",),
    action_stat_keys: Any = ("action",),
) -> list[DatasetValidationIssue]:
    issues: list[DatasetValidationIssue] = []
    if not dataset_path.exists():
        return [DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} path does not exist")]
    if not dataset_path.is_dir():
        return [DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} path is not a directory")]

    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    stats_json_path = dataset_path / "meta" / "stats.json"
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))

    issues.extend(validate_tasks_jsonl(tasks_path, dataset_label))
    issues.extend(validate_jsonl_objects(episodes_path, dataset_label, "episodes"))
    issues.extend(
        validate_stats_files(
            stats_json_path,
            episodes_stats_path,
            dataset_label,
            max_state_dim,
            max_action_dim,
            state_stat_keys=state_stat_keys,
            action_stat_keys=action_stat_keys,
        )
    )

    if not parquet_files:
        issues.append(DatasetValidationIssue("FAIL", str(dataset_path / "data/*/*.parquet"), f"{dataset_label} has no parquet files"))
    elif require_videos:
        issues.extend(validate_video_paths(dataset_path, parquet_files, view_map, dataset_label))

    return issues


def validate_tasks_jsonl(path: Path, dataset_label: str) -> list[DatasetValidationIssue]:
    issues, rows = read_jsonl_objects(path, dataset_label, "tasks")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("task_index"), int) or isinstance(row.get("task_index"), bool):
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{dataset_label} tasks line {index} has invalid task_index"))
        if not isinstance(row.get("task"), str) or not row.get("task"):
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{dataset_label} tasks line {index} has invalid task"))
    return issues


def validate_jsonl_objects(path: Path, dataset_label: str, label: str) -> list[DatasetValidationIssue]:
    issues, _rows = read_jsonl_objects(path, dataset_label, label)
    return issues


def read_jsonl_objects(path: Path, dataset_label: str, label: str) -> tuple[list[DatasetValidationIssue], list[dict[str, Any]]]:
    issues: list[DatasetValidationIssue] = []
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return [DatasetValidationIssue("FAIL", str(path), f"{dataset_label} missing {label} file")], rows

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{label} line {line_number} is invalid JSON: {exc}"))
            continue
        if not isinstance(row, dict):
            issues.append(DatasetValidationIssue("FAIL", str(path), f"{label} line {line_number} must be an object"))
            continue
        rows.append(row)

    if not rows:
        issues.append(DatasetValidationIssue("FAIL", str(path), f"{dataset_label} {label} file has no records"))
    return issues, rows


def validate_stats_files(
    stats_json_path: Path,
    episodes_stats_path: Path,
    dataset_label: str,
    max_state_dim: int,
    max_action_dim: int,
    state_stat_keys: Any = ("observation.state",),
    action_stat_keys: Any = ("action",),
) -> list[DatasetValidationIssue]:
    if stats_json_path.exists():
        try:
            payload = json.loads(stats_json_path.read_text())
        except json.JSONDecodeError as exc:
            return [DatasetValidationIssue("FAIL", str(stats_json_path), f"stats.json is invalid JSON: {exc}")]
        return _validate_stats_payload(
            payload,
            str(stats_json_path),
            dataset_label,
            max_state_dim,
            max_action_dim,
            state_stat_keys=state_stat_keys,
            action_stat_keys=action_stat_keys,
        )

    if not episodes_stats_path.exists():
        return [
            DatasetValidationIssue(
                "FAIL",
                str(episodes_stats_path),
                f"{dataset_label} missing stats.json or episodes_stats.jsonl",
            )
        ]

    issues, rows = read_jsonl_objects(episodes_stats_path, dataset_label, "episodes_stats")
    for index, row in enumerate(rows, start=1):
        stats = row.get("stats")
        if not isinstance(stats, dict):
            issues.append(DatasetValidationIssue("FAIL", str(episodes_stats_path), f"episodes_stats line {index} missing stats object"))
            continue
        issues.extend(
            _validate_stats_payload(
                stats,
                str(episodes_stats_path),
                f"{dataset_label} line {index}",
                max_state_dim,
                max_action_dim,
                state_stat_keys=state_stat_keys,
                action_stat_keys=action_stat_keys,
            )
        )
    return issues


def _validate_stats_payload(
    stats: Any,
    path_label: str,
    dataset_label: str,
    max_state_dim: int,
    max_action_dim: int,
    state_stat_keys: Any = ("observation.state",),
    action_stat_keys: Any = ("action",),
) -> list[DatasetValidationIssue]:
    if not isinstance(stats, dict):
        return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} stats must be an object")]

    issues: list[DatasetValidationIssue] = []
    issues.extend(validate_minmax_stat(stats, state_stat_keys, max_state_dim, path_label, dataset_label))
    issues.extend(validate_minmax_stat(stats, action_stat_keys, max_action_dim, path_label, dataset_label))
    return issues


def validate_minmax_stat(
    stats: Mapping[str, Any],
    stat_name: str | list[str] | tuple[str, ...],
    max_dim: int,
    path_label: str,
    dataset_label: str,
) -> list[DatasetValidationIssue]:
    stat_names = [stat_name] if isinstance(stat_name, str) else [str(name) for name in stat_name]
    stat = None
    selected_stat_name = stat_names[0]
    for candidate in stat_names:
        candidate_stat = stats.get(candidate)
        if isinstance(candidate_stat, Mapping):
            stat = candidate_stat
            selected_stat_name = candidate
            break
    if not isinstance(stat, Mapping):
        return [
            DatasetValidationIssue(
                "FAIL",
                path_label,
                f"{dataset_label} stats missing one of {', '.join(stat_names)} min/max object",
            )
        ]

    mins = stat.get("min")
    maxs = stat.get("max")
    issues: list[DatasetValidationIssue] = []
    issues.extend(validate_numeric_vector(mins, f"{selected_stat_name}.min", max_dim, path_label, dataset_label))
    issues.extend(validate_numeric_vector(maxs, f"{selected_stat_name}.max", max_dim, path_label, dataset_label))
    if issues:
        return issues

    if len(mins) != len(maxs):
        return [
            DatasetValidationIssue(
                "FAIL",
                path_label,
                f"{dataset_label} {selected_stat_name}.min and max must have the same length",
            )
        ]
    for index, (min_value, max_value) in enumerate(zip(mins, maxs)):
        if float(min_value) > float(max_value):
            issues.append(
                DatasetValidationIssue(
                    "FAIL",
                    path_label,
                    f"{dataset_label} {selected_stat_name}.min[{index}] must be <= max[{index}]",
                )
            )
    return issues


def validate_numeric_vector(
    value: Any,
    label: str,
    max_dim: int,
    path_label: str,
    dataset_label: str,
) -> list[DatasetValidationIssue]:
    if not isinstance(value, list) or not value:
        return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} {label} must be a non-empty list")]
    if len(value) > max_dim:
        return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} {label} length {len(value)} exceeds max_dim {max_dim}")]
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
            return [DatasetValidationIssue("FAIL", path_label, f"{dataset_label} {label}[{index}] must be a finite number")]
    return []


def validate_video_paths(
    dataset_path: Path,
    parquet_files: list[Path],
    view_map: Mapping[str, Any],
    dataset_label: str,
) -> list[DatasetValidationIssue]:
    issues: list[DatasetValidationIssue] = []
    for view_key, view_value in view_map.items():
        view_folders = _normalize_view_candidates(view_value)
        if not view_folders:
            issues.append(DatasetValidationIssue("FAIL", str(dataset_path), f"{dataset_label} view {view_key!r} has invalid folder"))
            continue
        for parquet_path in parquet_files:
            candidates = [
                dataset_path / "videos" / parquet_path.parent.name / view_folder / f"{parquet_path.stem}.mp4"
                for view_folder in view_folders
            ]
            if not any(video_path.exists() for video_path in candidates):
                joined = ", ".join(str(video_path) for video_path in candidates)
                issues.append(DatasetValidationIssue("FAIL", joined, f"{dataset_label} missing video for {parquet_path}"))
    return issues


def _normalize_view_candidates(view_value: Any) -> list[str]:
    if isinstance(view_value, str):
        return [view_value] if view_value else []
    if isinstance(view_value, list):
        return [str(item) for item in view_value if str(item)]
    return []


# Public re-exports from the flattened data layer.
try:
    from prism.data.segments import *  # noqa: F403
    from prism.data.libero import *  # noqa: F403
    from prism.data.calvin import *  # noqa: F403
    from prism.data.cache import *  # noqa: F403
except ImportError:
    # Keep lightweight helpers importable when optional dataset dependencies are absent.
    pass
