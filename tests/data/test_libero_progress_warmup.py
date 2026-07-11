import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from prism.data.libero import ImageStatsVLSummaryEncoder
from prism.data.libero import LiberoProgressWarmupDataset
from prism.data.libero import TemperatureSuiteSampler
from prism.data.libero import build_libero_progress_vl_embedding_cache
from prism.data.libero import build_libero_progress_windows
from prism.data.libero import collate_libero_progress_warmup_windows


def test_build_windows_allows_short_burnin():
    steps = [
        {
            "step_index": index,
            "episode_id": "libero_spatial:task:demo_0",
            "suite": "libero_spatial",
            "task_name": "task",
            "replan_index": index,
        }
        for index in range(5)
    ]
    windows = build_libero_progress_windows(
        steps,
        burnin_replan_steps=3,
        loss_replan_steps=2,
        allow_short_burnin=True,
    )

    assert len(windows) == 4
    assert windows[0]["burnin_step_indices"] == []
    assert windows[0]["loss_step_indices"] == [0, 1]
    assert windows[2]["burnin_step_indices"] == [0, 1]
    assert windows[2]["loss_step_indices"] == [2, 3]


def test_libero_progress_vl_embedding_cache_dataset_and_sampler(tmp_path: Path):
    data_root, index_path = _write_fake_libero_replay(tmp_path)
    result = build_libero_progress_vl_embedding_cache(
        data_root=data_root,
        index_path=index_path,
        output_root=tmp_path / "warmup",
        vl_encoder=ImageStatsVLSummaryEncoder(hidden_dim=4),
        action_horizon=4,
        replan_stride=2,
        burnin_replan_steps=2,
        loss_replan_steps=2,
        allow_short_burnin=True,
    )
    dataset = LiberoProgressWarmupDataset(result.manifest_path)
    first = dataset[0]
    batch = collate_libero_progress_warmup_windows([first, dataset[1]])
    sampler = TemperatureSuiteSampler(dataset, samples_per_epoch=5, alpha=0.5, seed=123)

    assert result.step_count == 5
    assert result.window_count == 4
    assert first["burnin"] == []
    assert len(first["loss"]) == 2
    assert dataset.steps[0]["prompt"] == "pick up the test cup"
    assert tuple(batch["loss"]["vl_summary"].shape) == (2, 2, 4)
    assert tuple(batch["loss"]["executed_actions"].shape) == (2, 2, 2, 2)
    assert batch["burnin_mask"].shape[0] == 2
    assert len(list(iter(sampler))) == 5


def _write_fake_libero_replay(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "libero" / "datasets"
    task_dir = data_root / "libero_spatial"
    task_dir.mkdir(parents=True)
    hdf5_path = task_dir / "fake_task_demo.hdf5"
    length = 12
    actions = np.stack([np.array([float(step), float(step) + 0.5], dtype=np.float32) for step in range(length)])
    images = np.zeros((length, 4, 4, 3), dtype=np.uint8)
    for step in range(length):
        images[step, :, :, :] = step
    with h5py.File(hdf5_path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "pick up the test cup"})
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=actions)
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=images)
        obs.create_dataset("eye_in_hand_rgb", data=255 - images)
        obs.create_dataset("ee_states", data=np.ones((length, 3), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.ones((length, 2), dtype=np.float32))

    index_path = tmp_path / "libero_replay.jsonl"
    rows = []
    for step in range(0, length - 4 + 1):
        rows.append(
            {
                "action_end": step + 4,
                "action_horizon": 4,
                "action_start": step,
                "action_valid_count": 4,
                "benchmark": "LIBERO",
                "current_step": step,
                "episode_id": "libero_spatial:fake_task_demo:demo_0",
                "episode_key": "demo_0",
                "episode_length": length,
                "short_mask": [False, False],
                "short_steps": [None, None],
                "source_path": "libero_spatial/fake_task_demo.hdf5",
                "task_name": "fake_task",
            }
        )
    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return data_root, index_path
