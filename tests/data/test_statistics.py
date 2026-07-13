from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.calvin.data import CALVIN_DATA_SPEC
from prism.data.statistics import (
    StatisticsDatasetSource,
    StatisticsPlan,
    compute_lerobot_statistics,
    write_lerobot_statistics,
)


def test_scans_complete_roots_canonicalizes_and_records_split_provenance(
    tmp_path: Path,
) -> None:
    root = _make_dataset(tmp_path / "calvin_abc")
    plan = _calvin_plan(root)
    progress = []

    artifact = compute_lerobot_statistics(plan, progress=progress.append)

    group = artifact["groups"]["calvin_abc"]
    assert group["datasets"] == ["calvin_abc"]
    assert group["provenance"] == {
        "train_splits": ["A", "B", "C"],
        "eval_splits": ["D"],
    }
    assert group["state"]["count"] == 4
    assert group["action"]["count"] == 4
    assert group["state"]["continuous_indices"] == [0, 1, 2, 3, 4, 5, 7]
    assert group["state"]["identity_indices"] == [6]
    assert group["action"]["gripper_semantic"] == "open_01"
    assert group["action"]["q01"][0] == pytest.approx(0.03)
    assert group["action"]["q99"][0] == pytest.approx(2.97)
    assert progress[-1].completed_episodes == 1
    assert progress[-1].completed_frames == 4


def test_split_contract_rejects_eval_leak_and_incomplete_training_union(
    tmp_path: Path,
) -> None:
    root = _make_dataset(tmp_path / "calvin_abc")

    with pytest.raises(ValueError, match="leaks eval split D"):
        StatisticsPlan(
            data_spec=CALVIN_DATA_SPEC,
            group="calvin_abc",
            datasets=(StatisticsDatasetSource("calvin_abc", root, splits=("A", "B", "C", "D")),),
            train_splits=("A", "B", "C"),
            eval_splits=("D",),
        )

    with pytest.raises(ValueError, match="split union"):
        StatisticsPlan(
            data_spec=CALVIN_DATA_SPEC,
            group="calvin_abc",
            datasets=(StatisticsDatasetSource("calvin_ab", root, splits=("A", "B")),),
            train_splits=("A", "B", "C"),
            eval_splits=("D",),
        )


def test_calvin_contract_rejects_scene_d_root_even_if_mislabeled(tmp_path: Path) -> None:
    scene_d = _make_dataset(tmp_path / "task_D_D")

    with pytest.raises(ValueError, match="forbidden scene-D training root"):
        _calvin_plan(scene_d)


def test_scan_refuses_physically_incomplete_dataset(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "calvin_abc")
    (root / "videos/chunk-000/wrist_image/episode_000000.mp4").unlink()

    with pytest.raises(FileNotFoundError, match="is incomplete"):
        compute_lerobot_statistics(_calvin_plan(root))


def test_atomic_write_refuses_implicit_overwrite(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "calvin_abc")
    output = tmp_path / "artifacts" / "statistics.json"

    saved, artifact = write_lerobot_statistics(_calvin_plan(root), output)

    assert saved == output
    assert json.loads(output.read_text(encoding="utf-8"))["content_sha256"] == artifact["content_sha256"]
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_lerobot_statistics(_calvin_plan(root), output)


def test_direct_statistics_cli_loads_project_dataspec_from_any_working_directory(
    tmp_path: Path,
) -> None:
    root = _make_dataset(tmp_path / "calvin_abc")
    output = tmp_path / "statistics.json"
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/compute_statistics.py"),
            str(output),
            "--data-spec",
            "experiments.calvin.data:CALVIN_DATA_SPEC",
            "--group",
            "calvin_abc",
            "--dataset",
            f"calvin_abc={root}",
            "--dataset-split",
            "calvin_abc=A",
            "--dataset-split",
            "calvin_abc=B",
            "--dataset-split",
            "calvin_abc=C",
            "--train-split",
            "A",
            "--train-split",
            "B",
            "--train-split",
            "C",
            "--eval-split",
            "D",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["format"] == "prism-normalization-v1"


def _calvin_plan(root: Path) -> StatisticsPlan:
    return StatisticsPlan(
        data_spec=CALVIN_DATA_SPEC,
        group="calvin_abc",
        datasets=(
            StatisticsDatasetSource(
                name="calvin_abc",
                path=root,
                splits=("A", "B", "C"),
            ),
        ),
        train_splits=("A", "B", "C"),
        eval_splits=("D",),
    )


def _make_dataset(root: Path) -> Path:
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": 1,
        "total_frames": 4,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "state": {"dtype": "float32", "shape": [8]},
            "actions": {"dtype": "float32", "shape": [7]},
            "image": {"dtype": "video", "shape": [16, 16, 3]},
            "wrist_image": {"dtype": "video", "shape": [16, 16, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    _write_json(root / "meta/info.json", info)
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": 0, "tasks": ["open the drawer"], "length": 4}],
    )
    _write_jsonl(
        root / "meta/tasks.jsonl",
        [{"task_index": 0, "task": "open the drawer"}],
    )
    states = np.arange(32, dtype=np.float32).reshape(4, 8)
    states[:, 6] = 0.0
    actions = np.arange(28, dtype=np.float32).reshape(4, 7)
    actions[:, :6] = np.arange(24, dtype=np.float32).reshape(4, 6)
    actions[:, 0] = np.arange(4, dtype=np.float32)
    actions[:, 6] = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    table = pd.DataFrame(
        {
            "state": list(states),
            "actions": list(actions),
            "timestamp": np.arange(4, dtype=np.float32) / 10,
            "frame_index": np.arange(4, dtype=np.int64),
            "episode_index": np.zeros(4, dtype=np.int64),
            "index": np.arange(4, dtype=np.int64),
            "task_index": np.zeros(4, dtype=np.int64),
        }
    )
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(parquet, index=False)
    for key in ("image", "wrist_image"):
        video = root / f"videos/chunk-000/{key}/episode_000000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"existence is sufficient for numeric statistics")
    return root


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
