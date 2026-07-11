from __future__ import annotations

# --- migrated from src/prism/training/stage2/common/dataset.py ---
import json
import logging
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prism.data.memory_replay import DEFAULT_EXECUTED_ACTION_STRIDE
from prism.data.replay_frames import MemoryReplayFrameReader, ReplayFrame
from prism.data.token_cache_dataset import _apply_token_cache_normalization, _parse_token_cache_normalization
from prism.utils.paths import display_project_path, project_path
from prism.utils.seeding import build_torch_generator, seed_data_worker


def prepare_stage2_dataset(config: dict[str, Any], *, repo_root: str | Path) -> "RawEpisodeSequenceDataset":
    index_path = project_path(config.get("dataset_config_path"), repo_root, label="--dataset_config_path")
    normalization_path = config.get("normalization_source_path")
    normalization = None
    if normalization_path:
        normalization = load_stage2_normalization(
            project_path(normalization_path, repo_root, label="--normalization_source_path")
        )
    dataset = RawEpisodeSequenceDataset(
        index_path,
        sequence_len=int(config.get("sequence_len", 16)),
        action_horizon=int(config.get("horizon", 32)),
        action_dim=int(config.get("per_action_dim", 7)),
        state_dim=int(config.get("state_dim", 8)),
        normalization=normalization,
        sample_valid_future_horizon_only=bool(config.get("sample_valid_future_horizon_only", True)),
        max_episodes=config.get("max_samples_per_file"),
    )
    logging.info(
        "Loaded Stage2 raw %s replay index: episodes=%s sequence_len=%s manifest=%s",
        dataset.benchmark,
        len(dataset),
        int(config.get("sequence_len", 16)),
        display_project_path(index_path, repo_root),
    )
    return dataset


def prepare_stage2_dataloader(dataset: "RawEpisodeSequenceDataset", config: dict[str, Any]):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for Stage2 dataloading") from exc

    batch_size = int(config.get("batch_size", 1))
    num_workers = int(config.get("num_workers", 0))
    seed = int(config.get("seed", 42))
    shuffle = bool(config.get("shuffle_episodes", True))
    if len(dataset) == 0:
        raise ValueError("Stage2 dataset is empty")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=collate_raw_episode_sequences,
        worker_init_fn=seed_data_worker,
        generator=build_torch_generator(seed),
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Stage2 dataloader has no episode groups. Dataset size={len(dataset)}, batch_size={batch_size}."
        )
    logging.info(
        "Initialized Stage2 dataloader: episode_batch_size=%s sequence_len=%s num_workers=%s shuffle=%s",
        batch_size,
        dataset.sequence_len,
        num_workers,
        shuffle,
    )
    return dataloader


class RawEpisodeSequenceDataset:
    """MemoryVLA-style cheap group sampler over raw benchmark episodes."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        sequence_len: int = 16,
        action_horizon: int = 32,
        action_dim: int = 7,
        state_dim: int = 8,
        short_offsets: Sequence[int] | None = None,
        executed_action_stride: int | None = None,
        data_root: str | Path | None = None,
        normalization: Mapping[str, Any] | None = None,
        sample_valid_future_horizon_only: bool = True,
        max_episodes: int | None = None,
    ) -> None:
        self.index_path = Path(index_path).expanduser()
        with self.index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        benchmark = str(payload.get("benchmark", "LIBERO")).upper()
        expected_format = _episode_replay_index_format(benchmark)
        if payload.get("format") != expected_format:
            raise ValueError(f"expected {expected_format}, got {payload.get('format')!r}")
        if benchmark not in {"LIBERO", "CALVIN"}:
            raise ValueError(f"RawEpisodeSequenceDataset requires LIBERO or CALVIN, got {benchmark!r}")

        self.benchmark = benchmark
        self.sequence_len = int(sequence_len)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.sample_valid_future_horizon_only = bool(sample_valid_future_horizon_only)
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        self.short_offsets = tuple(int(offset) for offset in (short_offsets or payload.get("short_offsets") or (16, 8)))
        self.executed_action_stride = int(
            executed_action_stride or payload.get("executed_action_stride") or DEFAULT_EXECUTED_ACTION_STRIDE
        )
        self.data_root = Path(data_root or _data_root_from_index(payload, benchmark) or self.index_path.parent).expanduser()
        self.normalization = normalization
        self.arm2stats_dict = None if normalization is None else dict(normalization["stats"])
        self.reader = MemoryReplayFrameReader(benchmark=self.benchmark, data_root=self.data_root)

        episodes = list(payload.get("episodes") or [])
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]
        self.episodes = [episode for episode in episodes if self._episode_has_valid_steps(episode)]
        if not self.episodes:
            raise ValueError("Stage2 replay index has no valid episodes after horizon filtering")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[int(index)]
        steps = self._sample_steps(episode)
        return {
            "episode_id": str(episode["episode_id"]),
            "prompt": str(episode.get("prompt") or ""),
            "sampled_steps": steps,
            "steps": [self._materialize_step(episode, current_step) for current_step in steps],
        }

    def _episode_has_valid_steps(self, episode: Mapping[str, Any]) -> bool:
        episode_length = int(episode.get("episode_length", 0))
        if self.sample_valid_future_horizon_only:
            return episode_length >= self.action_horizon
        return episode_length > 0

    def _sample_steps(self, episode: Mapping[str, Any]) -> list[int]:
        episode_length = int(episode["episode_length"])
        last_step = episode_length - self.action_horizon if self.sample_valid_future_horizon_only else episode_length - 1
        if last_step < 0:
            raise ValueError(f"episode {episode.get('episode_id')} is shorter than horizon={self.action_horizon}")
        candidates = list(range(last_step + 1))
        if len(candidates) >= self.sequence_len:
            selected = random.sample(candidates, self.sequence_len)
        else:
            selected = candidates + [candidates[-1]] * (self.sequence_len - len(candidates))
        return sorted(int(step) for step in selected)

    def _materialize_step(self, episode: Mapping[str, Any], current_step: int) -> dict[str, Any]:
        row = self._row_for_step(episode, current_step)
        sample = self.reader.read(row)
        prompt = str(episode.get("prompt") or sample.prompt)
        tensors = self._normalized_tensors(sample, prompt=prompt)
        short_images, short_image_masks = _short_images_and_masks(sample.short_frames, sample.short_mask)
        return {
            "episode_id": sample.episode_id,
            "prompt": prompt,
            "current_step": int(sample.current_step),
            "images": _images_from_frame(sample.current),
            "image_mask": _ones_image_mask(sample.current),
            "states": tensors["states"],
            "actions": tensors["actions"],
            "action_mask": tensors["action_mask"],
            "short_images": short_images,
            "short_image_masks": short_image_masks,
            "executed_actions": tensors["executed_actions"],
            "executed_action_mask": tensors["executed_action_mask"],
        }

    def _row_for_step(self, episode: Mapping[str, Any], current_step: int) -> dict[str, Any]:
        short_steps = [current_step - offset if current_step - offset >= 0 else None for offset in self.short_offsets]
        short_mask = [step is not None for step in short_steps]
        executed_start = max(0, current_step - self.executed_action_stride)
        return {
            "benchmark": self.benchmark,
            "episode_id": str(episode["episode_id"]),
            "episode_key": str(episode.get("episode_key", episode.get("episode_index", ""))),
            "episode_index": episode.get("episode_index"),
            "source_path": str(episode["source_path"]),
            "prompt": str(episode.get("prompt") or ""),
            "task_name": str(episode.get("task_name") or ""),
            "current_step": int(current_step),
            "short_steps": short_steps,
            "short_mask": short_mask,
            "executed_action_start": int(executed_start),
            "executed_action_end": int(current_step),
            "executed_action_stride": int(self.executed_action_stride),
            "action_start": int(current_step),
            "action_end": int(current_step + self.action_horizon),
            "action_valid_count": int(self.action_horizon),
        }

    def _normalized_tensors(self, sample: Any, *, prompt: str) -> dict[str, Any]:
        torch = _require_torch()
        raw = {
            "episode_id": sample.episode_id,
            "prompt": prompt,
            "current_state": sample.current.state_vector,
            "future_actions": sample.future_actions,
            "executed_actions": sample.executed_actions,
            "executed_action_mask": sample.executed_action_mask,
        }
        normalized = _apply_token_cache_normalization(raw, self.normalization)
        states = torch.as_tensor(normalized["current_state"], dtype=torch.float32).cpu()
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"state dim {states.shape[-1]} != configured state_dim={self.state_dim}")
        actions, action_mask = _pad_action_chunk(
            normalized["future_actions"],
            horizon=self.action_horizon,
            action_dim=self.action_dim,
        )
        executed_actions, executed_action_mask = _pad_executed_actions(
            normalized["executed_actions"],
            normalized["executed_action_mask"],
            stride=self.executed_action_stride,
            action_dim=self.action_dim,
        )
        return {
            "states": states,
            "actions": actions,
            "action_mask": action_mask,
            "executed_actions": executed_actions,
            "executed_action_mask": executed_action_mask,
        }


def collate_raw_episode_sequences(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = _require_torch()
    if not batch:
        raise ValueError("Stage2 collate received an empty batch")
    max_steps = max(len(item["steps"]) for item in batch)
    trajectory_steps = []
    for step_index in range(max_steps):
        active = [(batch_index, item["steps"][step_index]) for batch_index, item in enumerate(batch)]
        batch_indices = torch.tensor([batch_index for batch_index, _step in active], dtype=torch.long)
        steps = [step for _batch_index, step in active]
        trajectory_steps.append(
            {
                "batch_indices": batch_indices,
                "loss_mask": torch.ones(len(steps), dtype=torch.bool),
                "images": [step["images"] for step in steps],
                "image_mask": torch.stack([step["image_mask"] for step in steps]),
                "prompts": [step["prompt"] for step in steps],
                "states": torch.stack([step["states"] for step in steps]),
                "actions": torch.stack([step["actions"] for step in steps]),
                "action_mask": torch.stack([step["action_mask"] for step in steps]),
                "short_images": [step["short_images"] for step in steps],
                "short_image_masks": [step["short_image_masks"] for step in steps],
                "executed_actions": torch.stack([step["executed_actions"] for step in steps]),
                "executed_action_mask": torch.stack([step["executed_action_mask"] for step in steps]),
                "current_steps": torch.tensor([int(step["current_step"]) for step in steps], dtype=torch.long),
            }
        )
    return {
        "batch_size": len(batch),
        "episode_ids": [str(item["episode_id"]) for item in batch],
        "sampled_steps": [list(item["sampled_steps"]) for item in batch],
        "trajectory_steps": trajectory_steps,
    }


class LiberoRawEpisodeSequenceDataset(RawEpisodeSequenceDataset):
    """Backward-compatible alias for the original LIBERO Stage2 dataset name."""


def collate_libero_raw_episode_sequences(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return collate_raw_episode_sequences(batch)


def _episode_replay_index_format(benchmark: str) -> str:
    if benchmark == "LIBERO":
        return "libero_episode_replay_index"
    if benchmark == "CALVIN":
        return "calvin_episode_replay_index"
    return f"{benchmark.lower()}_episode_replay_index"


def _data_root_from_index(payload: Mapping[str, Any], benchmark: str) -> str | None:
    if benchmark == "LIBERO":
        value = payload.get("libero_root")
    elif benchmark == "CALVIN":
        value = payload.get("calvin_dataset_root") or payload.get("calvin_root")
    else:
        value = None
    return None if value in (None, "") else str(value)


def load_stage2_normalization(path: str | Path) -> dict[str, Any] | None:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload.get("normalization"), Mapping):
        return _parse_token_cache_normalization(payload)
    if isinstance(payload.get("stats"), Mapping):
        stats = payload["stats"]
        robot_key = str(payload.get("robot_key") or next(iter(stats)))
        return {
            "type": str(payload.get("type") or "train_split_minmax_to_minus_one_one"),
            "robot_key": robot_key,
            "stats": stats,
            "clip_after_normalization": bool(payload.get("clip_after_normalization", True)),
        }
    if isinstance(payload, Mapping) and payload:
        robot_key = str(next(iter(payload)))
        if isinstance(payload.get(robot_key), Mapping):
            return {
                "type": "train_split_minmax_to_minus_one_one",
                "robot_key": robot_key,
                "stats": dict(payload),
                "clip_after_normalization": True,
            }
    return None


def _images_from_frame(frame: ReplayFrame) -> list[Any]:
    return [image for _view, image in sorted(frame.images_by_view.items())]


def _ones_image_mask(frame: ReplayFrame):
    torch = _require_torch()
    return torch.ones(len(frame.images_by_view), dtype=torch.bool)


def _short_images_and_masks(
    short_frames: Sequence[ReplayFrame | None], short_mask: Sequence[bool]
) -> tuple[tuple[list[Any] | None, ...], tuple[Any | None, ...]]:
    images = []
    masks = []
    for frame, enabled in zip(short_frames, short_mask):
        if frame is None or not enabled:
            images.append(None)
            masks.append(None)
            continue
        frame_images = _images_from_frame(frame)
        images.append(frame_images)
        masks.append(_ones_image_mask(frame))
    return tuple(images), tuple(masks)


def _pad_action_chunk(value: Any, *, horizon: int, action_dim: int):
    torch = _require_torch()
    tensor = torch.as_tensor(value, dtype=torch.float32).cpu()
    if tensor.ndim != 2 or tensor.shape[-1] != action_dim:
        raise ValueError(f"future actions must have shape [T, {action_dim}], got {tuple(tensor.shape)}")
    output = torch.zeros(horizon, action_dim, dtype=torch.float32)
    mask = torch.zeros(horizon, action_dim, dtype=torch.bool)
    valid = min(int(tensor.shape[0]), int(horizon))
    if valid > 0:
        output[:valid] = tensor[:valid]
        mask[:valid] = True
    return output, mask


def _pad_executed_actions(value: Any, mask_value: Any, *, stride: int, action_dim: int):
    torch = _require_torch()
    tensor = torch.as_tensor(value, dtype=torch.float32).cpu()
    mask = torch.as_tensor(mask_value, dtype=torch.bool).cpu()
    if tensor.ndim != 2 or tensor.shape[-1] != action_dim:
        raise ValueError(f"executed actions must have shape [T, {action_dim}], got {tuple(tensor.shape)}")
    if mask.ndim != 1 or mask.shape[0] != tensor.shape[0]:
        raise ValueError(f"executed mask shape {tuple(mask.shape)} does not match actions {tuple(tensor.shape)}")
    output = torch.zeros(stride, action_dim, dtype=torch.float32)
    output_mask = torch.zeros(stride, dtype=torch.bool)
    valid = min(int(tensor.shape[0]), int(stride))
    if valid > 0:
        output[-valid:] = tensor[-valid:]
        output_mask[-valid:] = mask[-valid:]
    output = output * output_mask.unsqueeze(-1).to(dtype=output.dtype)
    return output, output_mask


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for Stage2 raw episode training") from exc
    return torch
