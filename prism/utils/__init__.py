from __future__ import annotations

# --- migrated from src/prism/utils/normalization.py ---
import json
from pathlib import Path

import torch

from prism.config import TARGET_STATE_DIM


def pad_vector(values, target_dim: int = TARGET_STATE_DIM) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.shape[0] > target_dim:
        raise ValueError(f"Input length {tensor.shape[0]} exceeds expected {target_dim}")
    if tensor.shape[0] < target_dim:
        pad = torch.zeros(target_dim - tensor.shape[0], dtype=torch.float32)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor


def minmax_normalize(value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    normalized = 2 * (value - min_value) / (max_value - min_value + 1e-8) - 1
    return torch.clamp(normalized, -1.0, 1.0)


def minmax_denormalize(value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + 1.0) * (max_value - min_value + 1e-8) + min_value


class NormalizationStats:
    def __init__(self, stats_or_path, target_dim: int = TARGET_STATE_DIM, robot_key: str | None = None):
        self.target_dim = int(target_dim)
        if isinstance(stats_or_path, (str, Path)):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        if not isinstance(stats, dict) or not stats:
            raise ValueError("norm_stats.json must contain at least one robot key")

        self.robot_keys = tuple(str(key) for key in stats)
        if robot_key is None:
            robot_key = next(iter(stats)) if len(stats) == 1 else None
        if robot_key is not None and robot_key not in stats:
            raise KeyError(f"robot_key {robot_key!r} not found in norm_stats.json; available keys: {list(stats.keys())}")
        self.robot_key = None if robot_key is None else str(robot_key)
        self._stats = stats
        self._prepared: dict[str, dict[str, torch.Tensor]] = {}
        if self.robot_key is not None:
            self._prepare_robot(self.robot_key)

    def normalize_state(self, state: torch.Tensor, robot_key: str | None = None) -> torch.Tensor:
        prepared = self._prepare_robot(self._resolve_robot_key(robot_key))
        state_dim = state.shape[-1]
        state_min_full = prepared["state_min"]
        state_max_full = prepared["state_max"]
        if state_dim > state_min_full.shape[0]:
            raise ValueError(f"State dimension {state_dim} exceeds normalizer dimension {state_min_full.shape[0]}")
        state_min = state_min_full[:state_dim].to(state.device, dtype=state.dtype)
        state_max = state_max_full[:state_dim].to(state.device, dtype=state.dtype)
        return minmax_normalize(state, state_min, state_max)

    def denormalize_action(self, action: torch.Tensor, robot_key: str | None = None) -> torch.Tensor:
        prepared = self._prepare_robot(self._resolve_robot_key(robot_key))
        if action.ndim == 1:
            action = action.view(1, -1)
        action_dim = action.shape[-1]
        action_min_full = prepared["action_min"]
        action_max_full = prepared["action_max"]
        if action_dim > action_min_full.shape[0]:
            raise ValueError(f"Action dimension {action_dim} exceeds normalizer dimension {action_min_full.shape[0]}")
        action_min = action_min_full[:action_dim].to(action.device, dtype=action.dtype)
        action_max = action_max_full[:action_dim].to(action.device, dtype=action.dtype)
        return minmax_denormalize(action, action_min, action_max)

    def normalize_action(self, action: torch.Tensor, robot_key: str | None = None) -> torch.Tensor:
        prepared = self._prepare_robot(self._resolve_robot_key(robot_key))
        action_dim = action.shape[-1]
        action_min_full = prepared["action_min"]
        action_max_full = prepared["action_max"]
        if action_dim > action_min_full.shape[0]:
            raise ValueError(f"Action dimension {action_dim} exceeds normalizer dimension {action_min_full.shape[0]}")
        action_min = action_min_full[:action_dim].to(action.device, dtype=action.dtype)
        action_max = action_max_full[:action_dim].to(action.device, dtype=action.dtype)
        return minmax_normalize(action, action_min, action_max)

    def _prepare_robot(self, robot_key: str) -> dict[str, torch.Tensor]:
        if robot_key in self._prepared:
            return self._prepared[robot_key]
        if robot_key not in self._stats:
            raise KeyError(f"robot_key {robot_key!r} not found in norm_stats.json; available keys: {list(self._stats.keys())}")
        robot_stats = self._stats[robot_key]
        prepared = {
            "state_min": pad_vector(robot_stats["observation.state"]["min"], self.target_dim),
            "state_max": pad_vector(robot_stats["observation.state"]["max"], self.target_dim),
            "action_min": pad_vector(robot_stats["action"]["min"], self.target_dim),
            "action_max": pad_vector(robot_stats["action"]["max"], self.target_dim),
        }
        self._prepared[robot_key] = prepared
        return prepared

    def _resolve_robot_key(self, robot_key: str | None) -> str:
        selected = robot_key or self.robot_key
        if selected is None:
            raise ValueError(f"robot_key is required when norm_stats.json has multiple keys: {list(self._stats.keys())}")
        return selected

# --- migrated from src/prism/utils/cuda_memory.py ---
from dataclasses import dataclass, field
import logging
from typing import Any


BYTES_PER_GIB = 1024**3


@dataclass
class CudaMemoryFloor:
    """Keep CUDA memory usage above a requested floor for long-running jobs."""

    torch: Any
    target_gb: float
    device: Any = "cuda"
    chunk_mb: int = 256
    _chunks: list[Any] = field(default_factory=list, init=False)

    def start(self) -> "CudaMemoryFloor":
        if self.target_gb <= 0:
            raise ValueError("target_gb must be positive")
        if self.chunk_mb <= 0:
            raise ValueError("chunk_mb must be positive")
        stats = self.refill_to_target()
        logging.info(
            "CUDA memory floor active: target=%.2f GiB used=%.2f GiB reserved_by_floor=%.2f GiB",
            self.target_gb,
            stats["used_gb"],
            self.reserved_gb,
        )
        return self

    def refill_to_target(self) -> dict[str, float | int | str]:
        device = _normalize_cuda_device(self.torch, self.device)
        if device.type != "cuda":
            raise ValueError(f"CUDA memory floor requires a CUDA device, got {device}")
        self.torch.cuda.set_device(device)

        target_bytes = int(float(self.target_gb) * BYTES_PER_GIB)
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(device)
        if target_bytes >= total_bytes:
            raise ValueError(
                f"CUDA memory floor {self.target_gb:.2f} GiB is not below total GPU memory "
                f"{total_bytes / BYTES_PER_GIB:.2f} GiB"
            )

        chunk_bytes = int(self.chunk_mb * 1024 * 1024)
        while True:
            stats = cuda_memory_stats(self.torch, device)
            used_bytes = int(stats["used_bytes"])
            if used_bytes >= target_bytes:
                break
            bytes_to_allocate = min(chunk_bytes, target_bytes - used_bytes)
            self._chunks.append(_allocate_cuda_chunk(self.torch, device, bytes_to_allocate))

        stats = cuda_memory_stats(self.torch, device)
        if int(stats["used_bytes"]) < target_bytes:
            raise RuntimeError(
                f"CUDA memory floor target was not reached: target={self.target_gb:.2f} GiB, "
                f"used={stats['used_gb']:.2f} GiB"
            )
        return stats

    @property
    def reserved_gb(self) -> float:
        return sum(int(chunk.numel() * chunk.element_size()) for chunk in self._chunks) / BYTES_PER_GIB

    def trim_to_target(self) -> dict[str, float | int | str]:
        device = _normalize_cuda_device(self.torch, self.device)
        target_bytes = int(float(self.target_gb) * BYTES_PER_GIB)
        while self._chunks:
            stats = cuda_memory_stats(self.torch, device)
            used_bytes = int(stats["used_bytes"])
            chunk = self._chunks[-1]
            chunk_bytes = int(chunk.numel() * chunk.element_size())
            if used_bytes - chunk_bytes < target_bytes:
                break
            self._chunks.pop()
            del chunk
            self.torch.cuda.empty_cache()
        return cuda_memory_stats(self.torch, device)

    def release(self) -> None:
        self._chunks.clear()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def close(self) -> None:
        self.release()


def reserve_cuda_memory_floor(
    torch: Any,
    *,
    target_gb: float | None,
    device: Any = "cuda",
    chunk_mb: int = 256,
) -> CudaMemoryFloor | None:
    if target_gb is None:
        return None
    return CudaMemoryFloor(torch=torch, target_gb=float(target_gb), device=device, chunk_mb=int(chunk_mb)).start()


def cuda_memory_stats(torch: Any, device: Any = "cuda") -> dict[str, float | int | str]:
    device = _normalize_cuda_device(torch, device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "device": str(device),
            "total_bytes": 0,
            "free_bytes": 0,
            "used_bytes": 0,
            "total_gb": 0.0,
            "free_gb": 0.0,
            "used_gb": 0.0,
        }
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = int(total_bytes - free_bytes)
    return {
        "device": str(device),
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "used_bytes": used_bytes,
        "total_gb": float(total_bytes / BYTES_PER_GIB),
        "free_gb": float(free_bytes / BYTES_PER_GIB),
        "used_gb": float(used_bytes / BYTES_PER_GIB),
    }


def _normalize_cuda_device(torch: Any, device: Any) -> Any:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _allocate_cuda_chunk(torch: Any, device: Any, bytes_to_allocate: int) -> Any:
    bytes_to_allocate = max(1, int(bytes_to_allocate))
    chunk_bytes = bytes_to_allocate
    while chunk_bytes > 0:
        try:
            return torch.empty((chunk_bytes,), dtype=torch.uint8, device=device)
        except RuntimeError:
            if chunk_bytes <= 1024 * 1024:
                raise
            chunk_bytes //= 2
            torch.cuda.empty_cache()
    raise RuntimeError("failed to allocate CUDA memory floor chunk")

