from __future__ import annotations

import numpy as np
import pytest

from prism.data.cache import write_memory_replay_jsonl
from prism.data.cache import MemoryReplayFrameDataset
from prism.data.cache import collate_memory_replay_frames
from prism.data.cache import MemoryReplayFrameReader


h5py = pytest.importorskip("h5py")


def test_memory_replay_frame_reader_reads_libero_current_short_and_future_actions(tmp_path):
    libero_root = tmp_path / "libero" / "datasets"
    hdf5_path = libero_root / "libero_spatial" / "pick_demo.hdf5"
    _write_libero_episode(hdf5_path)
    row = {
        "benchmark": "LIBERO",
        "episode_id": "libero_spatial:pick_demo:demo_0",
        "episode_key": "demo_0",
        "source_path": "libero_spatial/pick_demo.hdf5",
        "current_step": 4,
        "episode_length": 6,
        "action_start": 4,
        "action_end": 6,
        "action_valid_count": 2,
        "short_steps": [0, 2],
        "short_mask": [True, True],
    }

    sample = MemoryReplayFrameReader(benchmark="LIBERO", data_root=libero_root).read(row)

    assert sample.current.tau == 4
    assert [frame.tau for frame in sample.short_frames if frame is not None] == [0, 2]
    assert sample.short_mask == (True, True)
    assert sample.future_actions.shape == (2, 7)
    assert sample.future_actions[0, 0] == pytest.approx(28.0)
    assert sample.current.state_vector.shape == (8,)
    assert sample.current.images_by_view["agentview_rgb"].size == (3, 2)
    assert sample.current.images_by_view["eye_in_hand_rgb"].size == (3, 2)


def test_memory_replay_frame_dataset_and_collate_return_training_ready_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    libero_root = tmp_path / "libero" / "datasets"
    hdf5_path = libero_root / "libero_spatial" / "pick_demo.hdf5"
    _write_libero_episode(hdf5_path)
    rows = [
        {
            "benchmark": "LIBERO",
            "episode_id": "libero_spatial:pick_demo:demo_0",
            "episode_key": "demo_0",
            "source_path": "libero_spatial/pick_demo.hdf5",
            "current_step": step,
            "episode_length": 6,
            "action_start": step,
            "action_end": step + 2,
            "action_valid_count": 2,
            "short_steps": [None, step - 1 if step > 0 else None],
            "short_mask": [False, step > 0],
        }
        for step in (0, 2)
    ]
    index_path = write_memory_replay_jsonl(tmp_path / "index.jsonl", rows)

    dataset = MemoryReplayFrameDataset(
        benchmark="LIBERO",
        data_root=libero_root,
        index_path=index_path,
        image_transform=lambda image: np.asarray(image).shape,
    )
    batch = collate_memory_replay_frames([dataset[0], dataset[1]])

    assert len(dataset) == 2
    assert batch["current_step"].tolist() == [0, 2]
    assert tuple(batch["current_state"].shape) == (2, 8)
    assert tuple(batch["future_actions"].shape) == (2, 2, 7)
    assert batch["short_mask"].dtype == torch.bool
    assert batch["short_mask"].tolist() == [[False, False], [False, True]]
    assert batch["short_steps"].tolist() == [[-1, -1], [-1, 1]]
    assert batch["current_images"][0]["agentview_rgb"] == (2, 3, 3)
    assert batch["short_images"][1][1]["eye_in_hand_rgb"] == (2, 3, 3)


def _write_libero_episode(path):
    path.parent.mkdir(parents=True)
    images = np.zeros((6, 2, 3, 3), dtype=np.uint8)
    images[:, :, :, 0] = np.arange(6, dtype=np.uint8).reshape(6, 1, 1)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.arange(42, dtype=np.float32).reshape(6, 7))
        demo.create_dataset("obs/agentview_rgb", data=images)
        demo.create_dataset("obs/eye_in_hand_rgb", data=images + 1)
        demo.create_dataset("obs/ee_states", data=np.ones((6, 7), dtype=np.float32))
        demo.create_dataset("obs/gripper_states", data=np.full((6, 1), 0.5, dtype=np.float32))

