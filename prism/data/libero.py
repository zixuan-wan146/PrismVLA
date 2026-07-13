from __future__ import annotations

# --- migrated from src/prism/dataset/libero.py ---
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_LIBERO_VIEW_NAMES = ("agentview_rgb", "eye_in_hand_rgb")


@dataclass(frozen=True)
class LiberoFrame:
    tau: int
    images_by_view: Mapping[str, Image.Image]
    action: np.ndarray
    state_vector: np.ndarray


class LiberoEpisodeReader:
    """Read one LIBERO demonstration from a local HDF5 file without importing the simulator."""

    def __init__(
        self,
        hdf5_path: str | Path,
        *,
        demo_key: str,
        view_names: Sequence[str] = DEFAULT_LIBERO_VIEW_NAMES,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser()
        if not self.hdf5_path.exists():
            raise FileNotFoundError(self.hdf5_path)
        self.demo_key = str(demo_key)
        self.view_names = tuple(str(name) for name in view_names)
        if not self.view_names:
            raise ValueError("view_names must contain at least one view")

        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            self.demo_path = f"data/{self.demo_key}"
            if self.demo_path not in handle:
                raise KeyError(f"demo {self.demo_key!r} is missing from {self.hdf5_path}")
            demo = handle[self.demo_path]
            self.length = _dataset_length(demo, "actions")
            self.action_dim = int(np.asarray(demo["actions"].shape[-1]).item())
            self._validate_view_lengths(demo)

    def __len__(self) -> int:
        return self.length

    def read_frame(self, index: int) -> LiberoFrame:
        index = int(index)
        if index < 0 or index >= self.length:
            raise IndexError(f"frame index {index} out of range for episode length {self.length}")

        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            demo = handle[self.demo_path]
            images_by_view = {
                view_name: Image.fromarray(np.asarray(demo[f"obs/{view_name}"][index], dtype=np.uint8)).convert("RGB")
                for view_name in self.view_names
            }
            action = np.asarray(demo["actions"][index], dtype=np.float32).reshape(-1)
            state_vector = read_libero_state_vector(demo, index)
        return LiberoFrame(
            tau=index,
            images_by_view=images_by_view,
            action=action,
            state_vector=state_vector,
        )

    def read_future_actions(self, start: int, end: int) -> np.ndarray:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.length:
            raise IndexError(f"invalid action slice [{start}, {end}) for episode length {self.length}")
        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            return np.asarray(handle[self.demo_path]["actions"][start:end], dtype=np.float32)

    def _validate_view_lengths(self, demo: Any) -> None:
        for view_name in self.view_names:
            key = f"obs/{view_name}"
            if key not in demo:
                raise KeyError(f"LIBERO demo {self.demo_key!r} is missing image view: {key}")
            view_length = int(demo[key].shape[0])
            if view_length != self.length:
                raise ValueError(f"view {view_name!r} length {view_length} does not match action length {self.length}")


def read_libero_state_vector(demo: Any, index: int) -> np.ndarray:
    parts = []
    for key in ("obs/ee_states", "obs/gripper_states"):
        if key in demo:
            parts.append(np.asarray(demo[key][index], dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def _dataset_length(handle: Any, key: str) -> int:
    if key not in handle:
        raise KeyError(f"dataset key {key!r} is missing")
    shape = handle[key].shape
    if not shape:
        raise ValueError(f"dataset key {key!r} must have a time dimension")
    return int(shape[0])


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LiberoEpisodeReader requires h5py to read HDF5 demonstrations") from exc
    return h5py

