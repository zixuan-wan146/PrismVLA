"""Accepted benchmark dataset, split, and normalization contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


CALVIN_TRAIN_SPLITS = ("A", "B", "C")
CALVIN_EVAL_SPLITS = ("D",)
CALVIN_STATISTICS_GROUP = "calvin_abc"
CALVIN_EVAL_ROOT_NAME = "task_D_D"

LIBERO_STATISTICS_GROUP = "libero"
LIBERO_DATASET_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def validate_benchmark_data_contract(
    *,
    benchmark: str,
    group: str,
    dataset_names: Sequence[str],
    dataset_path_names: Sequence[str],
    dataset_splits: Sequence[Sequence[str] | None],
    train_splits: Sequence[str] | None,
    eval_splits: Sequence[str] | None,
) -> dict[str, Any] | None:
    """Validate the one accepted LIBERO or CALVIN train/statistics contract."""

    names = tuple(dataset_names)
    path_names = tuple(dataset_path_names)
    split_rows = tuple(None if splits is None else tuple(splits) for splits in dataset_splits)
    if not names or len(path_names) != len(names) or len(split_rows) != len(names):
        raise ValueError("benchmark contract requires equally sized, non-empty dataset metadata")
    train = None if train_splits is None else tuple(train_splits)
    evaluation = None if eval_splits is None else tuple(eval_splits)

    if benchmark == "calvin":
        if group != CALVIN_STATISTICS_GROUP:
            raise ValueError(f"CALVIN data.normalization.group must be {CALVIN_STATISTICS_GROUP!r}, got {group!r}")
        if train != CALVIN_TRAIN_SPLITS:
            raise ValueError(
                "CALVIN data.train_splits must be exactly "
                f"{list(CALVIN_TRAIN_SPLITS)!r}, got "
                f"{None if train is None else list(train)!r}"
            )
        if evaluation != CALVIN_EVAL_SPLITS:
            raise ValueError(
                "CALVIN data.eval_splits must be exactly "
                f"{list(CALVIN_EVAL_SPLITS)!r}, got "
                f"{None if evaluation is None else list(evaluation)!r}"
            )
        split_union: set[str] = set()
        for name, path_name, splits in zip(names, path_names, split_rows, strict=True):
            if not splits:
                raise ValueError(f"CALVIN training dataset {name!r} must explicitly declare splits")
            forbidden = sorted(set(splits) & set(CALVIN_EVAL_SPLITS))
            if forbidden:
                raise ValueError(f"CALVIN training dataset {name!r} leaks eval split D: {forbidden}")
            unknown = sorted(set(splits) - set(CALVIN_TRAIN_SPLITS))
            if unknown:
                raise ValueError(f"CALVIN training dataset {name!r} has unsupported training splits: {unknown}")
            if path_name == CALVIN_EVAL_ROOT_NAME:
                raise ValueError(
                    f"CALVIN training dataset {name!r} points to forbidden scene-D "
                    f"training root {CALVIN_EVAL_ROOT_NAME}"
                )
            split_union.update(splits)
        if split_union != set(CALVIN_TRAIN_SPLITS):
            raise ValueError(f"CALVIN training dataset split union must be exactly A/B/C, got {sorted(split_union)}")
        return {
            "train_splits": list(CALVIN_TRAIN_SPLITS),
            "eval_splits": list(CALVIN_EVAL_SPLITS),
        }

    if benchmark == "libero":
        if group != LIBERO_STATISTICS_GROUP:
            raise ValueError(f"LIBERO data.normalization.group must be {LIBERO_STATISTICS_GROUP!r}, got {group!r}")
        if train is not None or evaluation is not None:
            raise ValueError("LIBERO data must not declare CALVIN train_splits or eval_splits")
        unknown_names = sorted(set(names) - set(LIBERO_DATASET_NAMES))
        if unknown_names:
            raise ValueError(
                "LIBERO data.datasets contains unknown suites: "
                f"{unknown_names}; expected a non-empty subset of "
                f"{list(LIBERO_DATASET_NAMES)}"
            )
        with_splits = [name for name, splits in zip(names, split_rows, strict=True) if splits]
        if with_splits:
            raise ValueError(f"LIBERO dataset entries must not declare CALVIN splits: {with_splits}")
        return None

    raise ValueError(f"training and statistics support only CALVIN and LIBERO DataSpecs, got benchmark={benchmark!r}")


__all__ = [
    "CALVIN_EVAL_ROOT_NAME",
    "CALVIN_EVAL_SPLITS",
    "CALVIN_STATISTICS_GROUP",
    "CALVIN_TRAIN_SPLITS",
    "LIBERO_DATASET_NAMES",
    "LIBERO_STATISTICS_GROUP",
    "validate_benchmark_data_contract",
]
