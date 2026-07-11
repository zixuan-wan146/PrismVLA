from __future__ import annotations

# --- migrated from src/prism/runtime_config.py ---
import os
from typing import Iterable

import numpy as np


IMAGE_SIZE = int(os.getenv("PRISM_IMAGE_SIZE", "448"))
TARGET_STATE_DIM = int(os.getenv("PRISM_STATE_DIM", "24"))

DEFAULT_SERVER_HOST = os.getenv("PRISM_SERVER_HOST", "0.0.0.0")
DEFAULT_SERVER_PORT = int(os.getenv("PRISM_SERVER_PORT", "9000"))
DEFAULT_SERVER_URI = os.getenv("PRISM_SERVER_URI", f"ws://127.0.0.1:{DEFAULT_SERVER_PORT}")
DEFAULT_MAX_MESSAGE_SIZE = int(os.getenv("PRISM_MAX_MESSAGE_SIZE", "100000000"))


def pad_1d(values: Iterable[float], target_dim: int = TARGET_STATE_DIM, fill_value: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size > target_dim:
        raise ValueError(f"Input length {array.size} exceeds target dimension {target_dim}")
    if array.size == target_dim:
        return array

    padded = np.full(target_dim, fill_value, dtype=np.float32)
    padded[: array.size] = array
    return padded


def build_action_mask(valid_dim: int, target_dim: int = TARGET_STATE_DIM) -> list[int]:
    if valid_dim > target_dim:
        raise ValueError(f"valid_dim {valid_dim} exceeds target dimension {target_dim}")
    return [1] * valid_dim + [0] * (target_dim - valid_dim)


def normalize_mask(mask: Iterable[int], target_dim: int) -> list[int]:
    flat_mask = np.asarray(mask, dtype=np.int32).reshape(-1)
    if flat_mask.size > target_dim:
        raise ValueError(f"Mask length {flat_mask.size} exceeds target dimension {target_dim}")
    if flat_mask.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.int32)
        padded[: flat_mask.size] = flat_mask
        flat_mask = padded
    return flat_mask.astype(int).tolist()
