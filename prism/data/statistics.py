"""Explicit, strict statistics scans over complete LeRobot v2.1 datasets."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prism.data.benchmark_contracts import validate_benchmark_data_contract
from prism.data.lerobot import LeRobotDataset
from prism.data.normalization import (
    NormalizationStatistics,
    canonical_sha256,
    canonicalize_assembled_features,
    compute_statistics,
    save_statistics,
)
from prism.data.schema import DataSpec


@dataclass(frozen=True)
class StatisticsDatasetSource:
    """One named physical training dataset and its declared training splits."""

    name: str
    path: Path
    splits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("statistics dataset name must be a non-empty string")
        path = Path(self.path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"statistics dataset root is not a directory: {path}")
        _validate_ordered_names(self.splits, f"dataset {self.name!r} splits")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class StatisticsPlan:
    """Complete explicit source and split contract for one statistics group."""

    data_spec: DataSpec
    group: str
    datasets: tuple[StatisticsDatasetSource, ...]
    train_splits: tuple[str, ...] = ()
    eval_splits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.data_spec, DataSpec):
            raise TypeError(f"data_spec must be DataSpec, got {type(self.data_spec).__name__}")
        self.data_spec.validate()
        if not isinstance(self.group, str) or not self.group.strip():
            raise ValueError("statistics group must be a non-empty string")
        if not isinstance(self.datasets, tuple) or not self.datasets:
            raise ValueError("statistics datasets must be a non-empty explicit tuple")
        if any(not isinstance(item, StatisticsDatasetSource) for item in self.datasets):
            raise TypeError("statistics datasets must contain StatisticsDatasetSource")
        names = tuple(dataset.name for dataset in self.datasets)
        _validate_ordered_names(names, "statistics dataset names")
        _validate_ordered_names(self.train_splits, "train_splits")
        _validate_ordered_names(self.eval_splits, "eval_splits")
        validate_benchmark_data_contract(
            benchmark=self.data_spec.benchmark,
            group=self.group,
            dataset_names=names,
            dataset_path_names=tuple(dataset.path.name for dataset in self.datasets),
            dataset_splits=tuple(dataset.splits for dataset in self.datasets),
            train_splits=self.train_splits or None,
            eval_splits=self.eval_splits or None,
        )

    @property
    def provenance(self) -> dict[str, list[str]]:
        if not self.train_splits and not self.eval_splits:
            return {}
        return {
            "train_splits": list(self.train_splits),
            "eval_splits": list(self.eval_splits),
        }


@dataclass(frozen=True)
class StatisticsProgress:
    dataset_name: str
    completed_episodes: int
    total_episodes: int
    completed_frames: int
    total_frames: int


def compute_lerobot_statistics(
    plan: StatisticsPlan,
    *,
    progress: Callable[[StatisticsProgress], None] | None = None,
) -> NormalizationStatistics:
    """Scan complete physical roots, canonicalize rows, and compute one group."""

    if not isinstance(plan, StatisticsPlan):
        raise TypeError(f"plan must be StatisticsPlan, got {type(plan).__name__}")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    with ExitStack() as stack:
        opened = [
            (
                source,
                stack.enter_context(LeRobotDataset(source.path, plan.data_spec, verify_files=True)),
            )
            for source in plan.datasets
        ]
        total_frames = sum(
            dataset.episode_length(episode_id) for _, dataset in opened for episode_id in dataset.episode_ids()
        )
        if total_frames <= 0:
            raise ValueError("statistics sources contain no frames")
        states = np.empty((total_frames, plan.data_spec.state_dim), dtype=np.float32)
        actions = np.empty((total_frames, plan.data_spec.action_dim), dtype=np.float32)

        cursor = 0
        for source, dataset in opened:
            episode_ids = dataset.episode_ids()
            dataset_frames = sum(dataset.episode_length(item) for item in episode_ids)
            completed_frames = 0
            for completed_episodes, episode_id in enumerate(episode_ids, start=1):
                episode = dataset.read_numeric_episode(episode_id)
                length = episode.states.shape[0]
                if episode.actions.shape[0] != length:
                    raise ValueError(
                        f"dataset {source.name!r} episode {episode_id} has different state/action row counts"
                    )
                end = cursor + length
                states[cursor:end] = canonicalize_assembled_features(
                    episode.states,
                    plan.data_spec.state,
                )
                actions[cursor:end] = canonicalize_assembled_features(
                    episode.actions,
                    plan.data_spec.action,
                )
                cursor = end
                completed_frames += length
                if progress is not None:
                    progress(
                        StatisticsProgress(
                            dataset_name=source.name,
                            completed_episodes=completed_episodes,
                            total_episodes=len(episode_ids),
                            completed_frames=completed_frames,
                            total_frames=dataset_frames,
                        )
                    )
        if cursor != total_frames:
            raise RuntimeError(f"statistics scan filled {cursor} rows but allocated {total_frames}")

    return compute_statistics(
        states,
        actions,
        group=plan.group,
        robot_key=plan.data_spec.robot_key,
        datasets=tuple(source.name for source in plan.datasets),
        schema_hash=canonical_sha256(plan.data_spec),
        provenance=plan.provenance,
        state_continuous_indices=_continuous_indices(plan.data_spec.state),
    )


def write_lerobot_statistics(
    plan: StatisticsPlan,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[StatisticsProgress], None] | None = None,
) -> tuple[Path, NormalizationStatistics]:
    """Compute and atomically write statistics, refusing replacement by default."""

    target = Path(output_path).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(f"statistics output already exists: {target}; pass overwrite=True explicitly")
    artifact = compute_lerobot_statistics(plan, progress=progress)
    return save_statistics(artifact, target), artifact


def _continuous_indices(features: tuple) -> tuple[int, ...]:
    indices: list[int] = []
    cursor = 0
    for feature in features:
        end = cursor + feature.width
        if feature.normalization == "q01_q99":
            indices.extend(range(cursor, end))
        cursor = end
    return tuple(indices)


def _validate_ordered_names(values: tuple[str, ...], label: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be an explicit tuple")
    invalid = [value for value in values if not isinstance(value, str) or not value]
    if invalid:
        raise ValueError(f"{label} must contain non-empty strings, got {invalid!r}")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"{label} must be unique, duplicates: {duplicates}")


__all__ = [
    "StatisticsDatasetSource",
    "StatisticsPlan",
    "StatisticsProgress",
    "compute_lerobot_statistics",
    "write_lerobot_statistics",
]
