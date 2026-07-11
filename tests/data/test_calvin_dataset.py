from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prism.data.calvin import CalvinEpisodeReader
from prism.data.calvin import calvin_prompt_for_episode
from prism.data.calvin import iter_calvin_episode_files
from prism.data.calvin import read_calvin_tasks


pytest.importorskip("pyarrow")


def test_calvin_episode_reader_reads_lerobot_parquet_images_state_actions(tmp_path: Path):
    root = _write_calvin_lerobot_root(tmp_path, lengths=(4,))

    reader = CalvinEpisodeReader(root, episode_index=0)
    frame = reader.read_frame(2)

    assert len(reader) == 4
    assert reader.state_dim == 8
    assert reader.action_dim == 7
    assert sorted(frame.images_by_view) == ["image", "wrist_image"]
    assert frame.images_by_view["image"].size == (3, 2)
    assert frame.state_vector.tolist() == pytest.approx([20.0 + i for i in range(8)])
    assert frame.action.tolist() == pytest.approx([200.0 + i for i in range(7)])
    assert reader.read_future_actions(1, 3).shape == (2, 7)


def test_iter_calvin_episode_files_and_prompt(tmp_path: Path):
    root = _write_calvin_lerobot_root(tmp_path, lengths=(4, 5))

    episodes = iter_calvin_episode_files(root)
    tasks = read_calvin_tasks(root)
    prompt = calvin_prompt_for_episode({"episode_index": 1, "task_index": 1, "tasks": []}, tasks)

    assert [episode.episode_index for episode in episodes] == [0, 1]
    assert [episode.length for episode in episodes] == [4, 5]
    assert prompt == "slide the door to the left"


def _write_calvin_lerobot_root(tmp_path: Path, *, lengths: tuple[int, ...]) -> Path:
    root = tmp_path / "calvin" / "lerobot" / "task_ABC_D"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "chunks_size": 1000,
                "fps": 10,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"task_index": 0, "task": "open the drawer"}),
                json.dumps({"task_index": 1, "task": "slide the door to the left"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    episode_rows = []
    for episode_index, length in enumerate(lengths):
        episode_rows.append(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "tasks": ["open the drawer"] if episode_index == 0 else [],
                    "length": length,
                    "task_index": episode_index % 2,
                }
            )
        )
        _write_episode_parquet(root, episode_index=episode_index, length=length)
    (root / "meta" / "episodes.jsonl").write_text("\n".join(episode_rows) + "\n", encoding="utf-8")
    return root


def _write_episode_parquet(root: Path, *, episode_index: int, length: int) -> None:
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    wrist = np.zeros((2, 2, 3), dtype=np.uint8)
    rows = []
    for frame_index in range(length):
        image[:, :, 0] = frame_index
        wrist[:, :, 1] = frame_index + 1
        rows.append(
            {
                "state": (np.arange(8, dtype=np.float32) + frame_index * 10).tolist(),
                "actions": (np.arange(7, dtype=np.float32) + frame_index * 100).tolist(),
                "image": image.copy().tolist(),
                "wrist_image": wrist.copy().tolist(),
                "timestamp": frame_index / 10.0,
                "frame_index": frame_index,
                "episode_index": episode_index,
                "index": episode_index * 1000 + frame_index,
                "task_index": episode_index % 2,
            }
        )
    pd.DataFrame(rows).to_parquet(chunk / f"episode_{episode_index:06d}.parquet")
