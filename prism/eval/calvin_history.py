from __future__ import annotations

from prism.eval.calvin_observation import build_calvin_images_by_view

# --- migrated from src/prism/benchmarks/calvin/history.py ---
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np



class CalvinObservationHistory:
    def __init__(self, *, max_offset: int) -> None:
        self.max_offset = int(max_offset)
        if self.max_offset <= 0:
            raise ValueError(f"max_offset must be positive, got {self.max_offset}")
        self._images_by_step: dict[int, dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        self._images_by_step.clear()

    def record(self, step_index: int, obs: Mapping[str, Any]) -> None:
        step_index = int(step_index)
        self._images_by_step[step_index] = {
            view_name: np.asarray(image, dtype=np.uint8).copy()
            for view_name, image in build_calvin_images_by_view(obs).items()
        }
        self._prune(current_step=step_index)

    def images_by_offset(self, *, current_step: int, offsets: Iterable[int]) -> dict[int, dict[str, np.ndarray]]:
        output = {}
        for offset in offsets:
            offset = int(offset)
            if offset <= 0:
                raise ValueError(f"short-memory offset must be positive, got {offset}")
            step_index = int(current_step) - offset
            if step_index in self._images_by_step:
                output[offset] = {
                    view_name: image.copy()
                    for view_name, image in self._images_by_step[step_index].items()
                }
        return output

    def _prune(self, *, current_step: int) -> None:
        min_step = int(current_step) - self.max_offset
        stale_steps = [step for step in self._images_by_step if step < min_step]
        for step in stale_steps:
            del self._images_by_step[step]

