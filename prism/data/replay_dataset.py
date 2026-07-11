from __future__ import annotations

# --- migrated from src/prism/dataset/memory_replay_dataset.py ---
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prism.data.memory_replay import read_memory_replay_jsonl
from prism.data.replay_frames import MemoryReplayFrameReader
from prism.data.replay_frames import MemoryReplayFrameSample


ImageTransform = Callable[[Any], Any]


@dataclass(frozen=True)
class MemoryReplayDatasetConfig:
    benchmark: str
    data_root: str | Path
    index_path: str | Path
    view_names: tuple[str, ...] | None = None


class MemoryReplayFrameDataset:
    """PyTorch-compatible dataset over deterministic memory replay index rows."""

    def __init__(
        self,
        *,
        benchmark: str,
        data_root: str | Path,
        index_path: str | Path,
        view_names: Sequence[str] | None = None,
        image_transform: ImageTransform | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.config = MemoryReplayDatasetConfig(
            benchmark=str(benchmark),
            data_root=Path(data_root).expanduser(),
            index_path=Path(index_path).expanduser(),
            view_names=None if view_names is None else tuple(str(view_name) for view_name in view_names),
        )
        self.rows = read_memory_replay_jsonl(self.config.index_path)
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive when provided")
            self.rows = self.rows[: int(max_samples)]
        if not self.rows:
            raise ValueError(f"memory replay index has no rows: {self.config.index_path}")
        self.reader = MemoryReplayFrameReader(
            benchmark=self.config.benchmark,
            data_root=self.config.data_root,
            view_names=self.config.view_names,
        )
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.reader.read(self.rows[int(index)])
        return memory_replay_sample_to_item(sample, image_transform=self.image_transform)


def memory_replay_sample_to_item(
    sample: MemoryReplayFrameSample,
    *,
    image_transform: ImageTransform | None = None,
) -> dict[str, Any]:
    current_images = _transform_images(sample.current.images_by_view, image_transform)
    short_images = tuple(
        None if frame is None else _transform_images(frame.images_by_view, image_transform)
        for frame in sample.short_frames
    )
    short_steps = tuple(-1 if frame is None else int(frame.tau) for frame in sample.short_frames)
    return {
        "benchmark": sample.benchmark,
        "episode_id": sample.episode_id,
        "prompt": sample.prompt,
        "current_step": sample.current_step,
        "current_images": current_images,
        "current_state": np.asarray(sample.current.state_vector, dtype=np.float32),
        "short_images": short_images,
        "short_steps": np.asarray(short_steps, dtype=np.int64),
        "short_mask": np.asarray(sample.short_mask, dtype=bool),
        "executed_actions": np.asarray(sample.executed_actions, dtype=np.float32),
        "executed_action_mask": np.asarray(sample.executed_action_mask, dtype=bool),
        "future_actions": np.asarray(sample.future_actions, dtype=np.float32),
        "action_valid_count": int(sample.action_valid_count),
    }


def collate_memory_replay_frames(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must contain at least one item")
    torch = _require_torch()
    return {
        "benchmark": [str(item["benchmark"]) for item in batch],
        "episode_id": [str(item["episode_id"]) for item in batch],
        "prompt": [str(item.get("prompt", "")) for item in batch],
        "current_step": torch.tensor([int(item["current_step"]) for item in batch], dtype=torch.long),
        "current_images": [item["current_images"] for item in batch],
        "current_state": torch.tensor(np.stack([item["current_state"] for item in batch]), dtype=torch.float32),
        "short_images": [item["short_images"] for item in batch],
        "short_steps": torch.tensor(np.stack([item["short_steps"] for item in batch]), dtype=torch.long),
        "short_mask": torch.tensor(np.stack([item["short_mask"] for item in batch]), dtype=torch.bool),
        "executed_actions": torch.tensor(np.stack([item["executed_actions"] for item in batch]), dtype=torch.float32),
        "executed_action_mask": torch.tensor(
            np.stack([item["executed_action_mask"] for item in batch]),
            dtype=torch.bool,
        ),
        "future_actions": torch.tensor(np.stack([item["future_actions"] for item in batch]), dtype=torch.float32),
        "action_valid_count": torch.tensor([int(item["action_valid_count"]) for item in batch], dtype=torch.long),
    }


def _transform_images(images_by_view: Mapping[str, Any], image_transform: ImageTransform | None) -> dict[str, Any]:
    if image_transform is None:
        return dict(images_by_view)
    return {view_name: image_transform(image) for view_name, image in images_by_view.items()}


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("collate_memory_replay_frames requires torch") from exc
    return torch

