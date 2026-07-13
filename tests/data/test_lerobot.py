from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest

from experiments.calvin.data import CALVIN_DATA_SPEC
from prism.data.lerobot import LeRobotDataset


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_video(path: Path, values: list[int], *, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "0", "preset": "ultrafast"}
        for value in values:
            array = np.full((16, 16, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_dataset(root: Path) -> Path:
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": 1,
        "total_frames": 4,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
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
    _write_json(root / "meta" / "info.json", info)
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["open the drawer"], "length": 4}],
    )
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": "open the drawer"}],
    )

    states = [np.arange(8, dtype=np.float32) + frame for frame in range(4)]
    actions = [np.arange(7, dtype=np.float32) + frame * 10 for frame in range(4)]
    table = pd.DataFrame(
        {
            "state": states,
            "actions": actions,
            "timestamp": np.arange(4, dtype=np.float32) / 10,
            "frame_index": np.arange(4, dtype=np.int64),
            "episode_index": np.zeros(4, dtype=np.int64),
            "index": np.arange(4, dtype=np.int64),
            "task_index": np.zeros(4, dtype=np.int64),
        }
    )
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(parquet, index=False)
    _write_video(
        root / "videos" / "chunk-000" / "image" / "episode_000000.mp4",
        [10, 50, 90, 130],
        fps=10,
    )
    _write_video(
        root / "videos" / "chunk-000" / "wrist_image" / "episode_000000.mp4",
        [20, 60, 100, 140],
        fps=10,
    )
    return root


def test_reads_numeric_language_and_ordered_video_frames(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "dataset")

    with LeRobotDataset(root, CALVIN_DATA_SPEC) as dataset:
        assert dataset.episode_ids() == (0,)
        assert dataset.episode_length(0) == 4
        assert dataset.fps == 10

        frame = dataset.read_numeric_frame(0, 2)
        np.testing.assert_array_equal(frame.state, np.arange(8, dtype=np.float32) + 2)
        np.testing.assert_array_equal(frame.action, np.arange(7, dtype=np.float32) + 20)
        assert frame.timestamp == pytest.approx(0.2)
        assert dataset.read_instruction(0, 2) == "open the drawer"

        episode = dataset.read_numeric_episode(0)
        assert episode.episode_index == 0
        assert episode.states.shape == (4, 8)
        assert episode.actions.shape == (4, 7)
        np.testing.assert_array_equal(episode.states[2], frame.state)
        np.testing.assert_array_equal(episode.actions[2], frame.action)
        np.testing.assert_allclose(episode.timestamps, np.arange(4) / 10)
        np.testing.assert_array_equal(episode.task_indices, np.zeros(4, dtype=np.int64))

        actions = dataset.read_actions(0, 1, 4)
        assert actions.shape == (3, 7)
        np.testing.assert_array_equal(actions[0], np.arange(7, dtype=np.float32) + 10)
        np.testing.assert_array_equal(actions[-1], np.arange(7, dtype=np.float32) + 30)

        images = dataset.read_images(0, [3, 1, 3])
        assert tuple(images) == ("primary", "wrist")
        assert images["primary"].shape == (3, 16, 16, 3)
        assert images["primary"].dtype == np.uint8
        assert images["primary"][:, 8, 8, 0].tolist() == pytest.approx([130, 50, 130], abs=3)
        assert images["wrist"][:, 8, 8, 0].tolist() == pytest.approx([140, 60, 140], abs=3)


def test_missing_episode_artifact_fails_during_open(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "dataset")
    (root / "videos" / "chunk-000" / "wrist_image" / "episode_000000.mp4").unlink()

    with pytest.raises(FileNotFoundError, match="is incomplete"):
        LeRobotDataset(root, CALVIN_DATA_SPEC)


def test_wrong_storage_version_is_rejected(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = "v3.0"
    _write_json(info_path, info)

    with pytest.raises(ValueError, match="requires LeRobot v2.1"):
        LeRobotDataset(root, CALVIN_DATA_SPEC)


def test_corrupt_parquet_index_fails_when_episode_is_read(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "dataset")
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    table = pd.read_parquet(parquet)
    table.loc[2, "frame_index"] = 99
    table.to_parquet(parquet, index=False)

    with LeRobotDataset(root, CALVIN_DATA_SPEC) as dataset:
        with pytest.raises(ValueError, match="frame_index is not contiguous"):
            dataset.read_numeric_frame(0, 0)


def test_frame_and_action_ranges_are_checked(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "dataset")

    with LeRobotDataset(root, CALVIN_DATA_SPEC) as dataset:
        with pytest.raises(IndexError, match="outside"):
            dataset.read_numeric_frame(0, 4)
        with pytest.raises(IndexError, match="outside"):
            dataset.read_actions(0, -1, 2)
        with pytest.raises(IndexError, match="outside"):
            dataset.read_images(0, [4])
