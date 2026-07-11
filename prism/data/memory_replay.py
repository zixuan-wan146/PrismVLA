from __future__ import annotations

# --- migrated from src/prism/dataset/memory_replay.py ---
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_SHORT_OFFSETS = (16, 8)
DEFAULT_MEMORY_LONG_CAPACITY = 0
DEFAULT_MEMORY_ACTION_HORIZON = 32
DEFAULT_EXECUTED_ACTION_STRIDE = 16


@dataclass(frozen=True)
class MemoryReplaySample:
    episode_id: str
    current_step: int
    episode_length: int
    action_horizon: int
    action_start: int
    action_valid_count: int
    executed_action_stride: int
    executed_action_start: int
    executed_action_valid_count: int
    short_steps: tuple[int | None, ...]
    short_mask: tuple[bool, ...]
    long_steps: tuple[int, ...]
    benchmark: str | None = None
    task_name: str | None = None
    source_path: str | None = None
    instruction_path: str | None = None
    episode_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "episode_id": self.episode_id,
            "current_step": self.current_step,
            "episode_length": self.episode_length,
            "action_horizon": self.action_horizon,
            "action_start": self.action_start,
            "action_end": self.action_start + self.action_valid_count,
            "action_valid_count": self.action_valid_count,
            "executed_action_stride": self.executed_action_stride,
            "executed_action_start": self.executed_action_start,
            "executed_action_end": self.current_step,
            "executed_action_valid_count": self.executed_action_valid_count,
            "short_steps": list(self.short_steps),
            "short_mask": list(self.short_mask),
            "long_steps": list(self.long_steps),
        }
        for key in ("benchmark", "task_name", "source_path", "instruction_path", "episode_key"):
            value = getattr(self, key)
            if value not in (None, ""):
                payload[key] = value
        return payload


def build_memory_replay_samples(
    *,
    episode_id: str,
    episode_length: int,
    action_horizon: int = DEFAULT_MEMORY_ACTION_HORIZON,
    stride: int = 1,
    short_offsets: Sequence[int] = DEFAULT_MEMORY_SHORT_OFFSETS,
    executed_action_stride: int = DEFAULT_EXECUTED_ACTION_STRIDE,
    action_start_offset: int = 0,
    long_candidate_steps: Iterable[int] | None = None,
    long_capacity: int = DEFAULT_MEMORY_LONG_CAPACITY,
    include_tail: bool = False,
    benchmark: str | None = None,
    task_name: str | None = None,
    source_path: str | None = None,
    instruction_path: str | None = None,
    episode_key: str | None = None,
) -> list[MemoryReplaySample]:
    episode_length = int(episode_length)
    action_horizon = int(action_horizon)
    stride = int(stride)
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    executed_action_stride = int(executed_action_stride)
    if executed_action_stride <= 0:
        raise ValueError(f"executed_action_stride must be positive, got {executed_action_stride}")
    action_start_offset = int(action_start_offset)
    if action_start_offset < 0:
        raise ValueError(f"action_start_offset must be non-negative, got {action_start_offset}")
    offsets = _normalize_short_offsets(short_offsets)
    if int(long_capacity) != 0:
        raise ValueError("long_capacity must be 0; long memory is produced by the progress-state planner")
    if long_candidate_steps:
        raise ValueError("long_candidate_steps is deprecated; replay indexes only carry fixed short visual memory")

    samples: list[MemoryReplaySample] = []
    for current_step in range(0, episode_length, stride):
        action_start = current_step + action_start_offset
        action_valid_count = min(action_horizon, max(0, episode_length - action_start))
        if action_valid_count < action_horizon and not include_tail:
            continue
        short_steps = tuple((current_step - offset) if current_step - offset >= 0 else None for offset in offsets)
        short_mask = tuple(step is not None for step in short_steps)
        executed_start = max(0, current_step - executed_action_stride)
        executed_valid_count = current_step - executed_start
        samples.append(
            MemoryReplaySample(
                episode_id=str(episode_id),
                current_step=current_step,
                episode_length=episode_length,
                action_horizon=action_horizon,
                action_start=action_start,
                action_valid_count=action_valid_count,
                executed_action_stride=executed_action_stride,
                executed_action_start=executed_start,
                executed_action_valid_count=executed_valid_count,
                short_steps=short_steps,
                short_mask=short_mask,
                long_steps=(),
                benchmark=benchmark,
                task_name=task_name,
                source_path=source_path,
                instruction_path=instruction_path,
                episode_key=episode_key,
            )
        )
    return samples


def build_memory_replay_manifest(
    *,
    benchmark: str,
    action_horizon: int,
    stride: int,
    short_offsets: Sequence[int],
    executed_action_stride: int = DEFAULT_EXECUTED_ACTION_STRIDE,
    action_start_offset: int = 0,
    long_capacity: int = DEFAULT_MEMORY_LONG_CAPACITY,
    include_tail: bool = False,
    sample_count: int = 0,
    episode_count: int = 0,
    task_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if int(long_capacity) != 0:
        raise ValueError("long_capacity must be 0; long memory is produced by the progress-state planner")
    if int(action_start_offset) < 0:
        raise ValueError(f"action_start_offset must be non-negative, got {action_start_offset}")
    return {
        "format": "memory_replay_index",
        "version": 1,
        "benchmark": benchmark,
        "action_horizon": int(action_horizon),
        "stride": int(stride),
        "short_offsets": list(_normalize_short_offsets(short_offsets)),
        "executed_action_stride": int(executed_action_stride),
        "action_start_offset": int(action_start_offset),
        "long_capacity": int(long_capacity),
        "include_tail": bool(include_tail),
        "sample_count": int(sample_count),
        "episode_count": int(episode_count),
        "task_counts": dict(sorted((task_counts or {}).items())),
    }


def write_memory_replay_jsonl(path: str | Path, samples: Sequence[MemoryReplaySample | Mapping[str, Any]]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = sample.to_dict() if isinstance(sample, MemoryReplaySample) else dict(sample)
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return output_path


def read_memory_replay_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path).expanduser()
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_short_offsets(short_offsets: Sequence[int]) -> tuple[int, ...]:
    offsets = tuple(sorted({int(offset) for offset in short_offsets}, reverse=True))
    if not offsets:
        raise ValueError("short_offsets must contain at least one offset")
    if any(offset <= 0 for offset in offsets):
        raise ValueError(f"short_offsets must be positive, got {short_offsets}")
    return offsets

