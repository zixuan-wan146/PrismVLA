from __future__ import annotations

from prism.data.memory_replay import DEFAULT_EXECUTED_ACTION_STRIDE

# --- migrated from src/prism/dataset/memory_replay_frames.py ---
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image



@dataclass(frozen=True)
class ReplayFrame:
    tau: int
    images_by_view: Mapping[str, Image.Image]
    state_vector: np.ndarray


@dataclass(frozen=True)
class MemoryReplayFrameSample:
    benchmark: str
    episode_id: str
    prompt: str
    current_step: int
    current: ReplayFrame
    short_frames: tuple[ReplayFrame | None, ...]
    short_mask: tuple[bool, ...]
    executed_actions: np.ndarray
    executed_action_mask: np.ndarray
    future_actions: np.ndarray
    action_valid_count: int


class MemoryReplayFrameReader:
    """Resolve replay-index rows into current/history frames and future actions."""

    def __init__(
        self,
        *,
        benchmark: str,
        data_root: str | Path,
        view_names: Sequence[str] | None = None,
    ) -> None:
        self.benchmark = str(benchmark).upper()
        self.data_root = Path(data_root).expanduser()
        self.view_names = None if view_names is None else tuple(str(name) for name in view_names)
        if self.benchmark not in {"LIBERO", "CALVIN"}:
            raise ValueError(f"unsupported replay benchmark: {benchmark!r}")

    def read(self, row: Mapping[str, Any]) -> MemoryReplayFrameSample:
        benchmark = str(row.get("benchmark") or self.benchmark).upper()
        if benchmark != self.benchmark:
            raise ValueError(f"row benchmark {benchmark!r} does not match reader benchmark {self.benchmark!r}")
        if self.benchmark == "LIBERO":
            return self._read_libero(row)
        if self.benchmark == "CALVIN":
            return self._read_calvin(row)
        raise ValueError(f"unsupported replay benchmark: {self.benchmark!r}")

    def _read_libero(self, row: Mapping[str, Any]) -> MemoryReplayFrameSample:
        from prism.data.libero import DEFAULT_LIBERO_VIEW_NAMES, LiberoEpisodeReader

        source_path = _required(row, "source_path")
        demo_key = str(row.get("episode_key") or _demo_key_from_episode_id(_required(row, "episode_id")))
        reader = LiberoEpisodeReader(
            self.data_root / source_path,
            demo_key=demo_key,
            view_names=self.view_names or DEFAULT_LIBERO_VIEW_NAMES,
        )
        current_step = int(row["current_step"])
        current = _libero_frame_to_replay(reader.read_frame(current_step))
        prompt = _libero_prompt_for_row(row, reader)
        short_frames = tuple(
            _libero_frame_to_replay(reader.read_frame(int(step))) if step is not None else None
            for step in row.get("short_steps", [])
        )
        executed_actions, executed_action_mask = _read_executed_action_segment(reader, row, current_step=current_step)
        future_actions = reader.read_future_actions(int(row["action_start"]), int(row["action_end"]))
        return _build_sample(
            row,
            prompt=prompt,
            current=current,
            short_frames=short_frames,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
            future_actions=future_actions,
        )

    def _read_calvin(self, row: Mapping[str, Any]) -> MemoryReplayFrameSample:
        from prism.data.calvin import DEFAULT_CALVIN_VIEW_NAMES, CalvinEpisodeReader

        episode_index = _calvin_episode_index_from_row(row)
        reader = CalvinEpisodeReader(
            self.data_root,
            episode_index=episode_index,
            view_names=self.view_names or DEFAULT_CALVIN_VIEW_NAMES,
        )
        current_step = int(row["current_step"])
        current = _calvin_frame_to_replay(reader.read_frame(current_step))
        prompt = str(row.get("prompt") or row.get("task_name") or "").strip()
        short_frames = tuple(
            _calvin_frame_to_replay(reader.read_frame(int(step))) if step is not None else None
            for step in row.get("short_steps", [])
        )
        executed_actions, executed_action_mask = _read_executed_action_segment(reader, row, current_step=current_step)
        future_actions = reader.read_future_actions(int(row["action_start"]), int(row["action_end"]))
        return _build_sample(
            row,
            prompt=prompt,
            current=current,
            short_frames=short_frames,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
            future_actions=future_actions,
        )


def _build_sample(
    row: Mapping[str, Any],
    *,
    prompt: str,
    current: ReplayFrame,
    short_frames: tuple[ReplayFrame | None, ...],
    executed_actions: np.ndarray,
    executed_action_mask: np.ndarray,
    future_actions: np.ndarray,
) -> MemoryReplayFrameSample:
    short_mask = tuple(bool(value) for value in row.get("short_mask", [frame is not None for frame in short_frames]))
    if len(short_mask) != len(short_frames):
        raise ValueError("short_mask length does not match short_frames length")
    return MemoryReplayFrameSample(
        benchmark=str(row.get("benchmark", "")),
        episode_id=str(row["episode_id"]),
        prompt=str(prompt),
        current_step=int(row["current_step"]),
        current=current,
        short_frames=short_frames,
        short_mask=short_mask,
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        executed_action_mask=np.asarray(executed_action_mask, dtype=bool),
        future_actions=np.asarray(future_actions, dtype=np.float32),
        action_valid_count=int(row["action_valid_count"]),
    )


def _libero_prompt_for_row(row: Mapping[str, Any], reader: Any) -> str:
    from prism.data.libero import _require_h5py as _require_libero_h5py

    prompt = ""
    try:
        h5py = _require_libero_h5py()
        with h5py.File(reader.hdf5_path, "r") as handle:
            problem_info = handle["data"].attrs.get("problem_info")
        if isinstance(problem_info, bytes):
            problem_info = problem_info.decode("utf-8")
        if problem_info:
            payload = json.loads(str(problem_info))
            prompt = str(payload.get("language_instruction") or "")
    except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError):
        prompt = ""
    if not prompt:
        prompt = str(row.get("task_name") or "").replace("_", " ").strip()
    return prompt


def _read_executed_action_segment(reader: Any, row: Mapping[str, Any], *, current_step: int) -> tuple[np.ndarray, np.ndarray]:
    stride = int(row.get("executed_action_stride", DEFAULT_EXECUTED_ACTION_STRIDE))
    if stride <= 0:
        raise ValueError(f"executed_action_stride must be positive, got {stride}")
    end = int(row.get("executed_action_end", current_step))
    start = int(row.get("executed_action_start", max(0, end - stride)))
    if end != current_step:
        raise ValueError(f"executed_action_end must equal current_step {current_step}, got {end}")
    if start < 0 or start > end:
        raise ValueError(f"invalid executed action segment [{start}, {end})")

    action_dim = int(getattr(reader, "action_dim"))
    output = np.zeros((stride, action_dim), dtype=np.float32)
    mask = np.zeros((stride,), dtype=bool)
    if end <= start:
        return output, mask

    actions = np.asarray(reader.read_future_actions(start, end), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != action_dim:
        raise ValueError(f"executed actions must have shape [T, {action_dim}], got {actions.shape}")
    valid_count = min(int(actions.shape[0]), stride)
    output[-valid_count:] = actions[-valid_count:]
    mask[-valid_count:] = True
    return output, mask


def _libero_frame_to_replay(frame) -> ReplayFrame:
    return ReplayFrame(
        tau=int(frame.tau),
        images_by_view=frame.images_by_view,
        state_vector=np.asarray(frame.state_vector, dtype=np.float32),
    )


def _calvin_frame_to_replay(frame) -> ReplayFrame:
    return ReplayFrame(
        tau=int(frame.tau),
        images_by_view=frame.images_by_view,
        state_vector=np.asarray(frame.state_vector, dtype=np.float32),
    )


def _required(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise KeyError(f"replay row is missing required key: {key}")
    return str(value)


def _demo_key_from_episode_id(episode_id: str) -> str:
    parts = str(episode_id).split(":")
    if not parts:
        raise ValueError(f"cannot infer LIBERO demo key from episode_id={episode_id!r}")
    return parts[-1]


def _calvin_episode_index_from_row(row: Mapping[str, Any]) -> int:
    for key in ("episode_index", "episode_key"):
        value = row.get(key)
        if value not in (None, ""):
            return int(value)
    suffix = _required(row, "episode_id").split(":")[-1]
    if suffix.startswith("episode_"):
        suffix = suffix[len("episode_") :]
    return int(suffix)

