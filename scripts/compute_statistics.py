"""Compute an explicit versioned normalization artifact from LeRobot v2.1 roots."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.data.schema import DataSpec
from prism.data.statistics import (
    StatisticsDatasetSource,
    StatisticsPlan,
    StatisticsProgress,
    write_lerobot_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output statistics JSON")
    parser.add_argument(
        "--data-spec",
        required=True,
        metavar="MODULE:OBJECT",
        help="Trusted project DataSpec reference under experiments.*",
    )
    parser.add_argument("--group", required=True, help="Statistics group name")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named LeRobot v2.1 root; repeat in the exact training order",
    )
    parser.add_argument(
        "--dataset-split",
        action="append",
        default=[],
        metavar="NAME=SPLIT",
        help="Training split represented by a dataset; repeat for each split",
    )
    parser.add_argument(
        "--train-split",
        action="append",
        default=[],
        metavar="SPLIT",
        help="Declared training split order; repeat as needed",
    )
    parser.add_argument(
        "--eval-split",
        action="append",
        default=[],
        metavar="SPLIT",
        help="Excluded evaluation split; repeat as needed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing statistics artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_spec = _load_data_spec(args.data_spec)
    dataset_rows = [_assignment(value, "--dataset") for value in args.dataset]
    split_rows = [_assignment(value, "--dataset-split") for value in args.dataset_split]
    split_map: dict[str, list[str]] = {name: [] for name, _ in dataset_rows}
    for name, split in split_rows:
        if name not in split_map:
            raise ValueError(f"--dataset-split references unknown dataset {name!r}")
        split_map[name].append(split)
    plan = StatisticsPlan(
        data_spec=data_spec,
        group=args.group,
        datasets=tuple(
            StatisticsDatasetSource(
                name=name,
                path=Path(path),
                splits=tuple(split_map[name]),
            )
            for name, path in dataset_rows
        ),
        train_splits=tuple(args.train_split),
        eval_splits=tuple(args.eval_split),
    )
    reporter = _ProgressReporter()
    output, artifact = write_lerobot_statistics(
        plan,
        args.output,
        overwrite=args.overwrite,
        progress=reporter,
    )
    group = artifact["groups"][plan.group]
    print(
        json.dumps(
            {
                "output": str(output),
                "content_sha256": artifact["content_sha256"],
                "datasets": group["datasets"],
                "state_count": group["state"]["count"],
                "action_count": group["action"]["count"],
                "state_clip_rate_low": group["state"]["clip_rate_low"],
                "state_clip_rate_high": group["state"]["clip_rate_high"],
                "action_clip_rate_low": group["action"]["clip_rate_low"],
                "action_clip_rate_high": group["action"]["clip_rate_high"],
            },
            sort_keys=True,
        )
    )


class _ProgressReporter:
    def __call__(self, progress: StatisticsProgress) -> None:
        if progress.completed_episodes % 100 == 0 or progress.completed_episodes == progress.total_episodes:
            print(
                f"{progress.dataset_name}: episodes "
                f"{progress.completed_episodes}/{progress.total_episodes}, "
                f"frames {progress.completed_frames}/{progress.total_frames}"
            )


def _assignment(value: str, flag: str) -> tuple[str, str]:
    name, separator, item = value.partition("=")
    if not separator or not name or not item:
        raise ValueError(f"{flag} must use non-empty NAME=VALUE syntax, got {value!r}")
    return name, item


def _load_data_spec(reference: str) -> DataSpec:
    module_name, separator, object_name = reference.partition(":")
    if not separator or not module_name.startswith("experiments.") or not object_name.isidentifier():
        raise ValueError("--data-spec must be a trusted experiments.* module:object reference")
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, object_name)
    except AttributeError as exc:
        raise AttributeError(f"DataSpec object not found: {reference}") from exc
    if not isinstance(value, DataSpec):
        raise TypeError(f"{reference} must resolve to DataSpec, got {type(value).__name__}")
    return value


if __name__ == "__main__":
    main()
