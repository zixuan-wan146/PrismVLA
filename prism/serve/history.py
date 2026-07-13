from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseHistoryPayload:
    images_by_view: Mapping[str, np.ndarray]
    step_ages: np.ndarray
    valid_mask: np.ndarray


class SparseHistoryBuffer:
    """Capture only the two within-chunk observations required by the policy."""

    def __init__(
        self,
        view_names: Sequence[str],
        *,
        capture_offsets: Sequence[int] = (2, 5),
        replan_stride: int = 8,
    ) -> None:
        self.view_names = tuple(str(name) for name in view_names)
        self.capture_offsets = tuple(int(offset) for offset in capture_offsets)
        self.replan_stride = int(replan_stride)
        if not self.view_names or len(set(self.view_names)) != len(self.view_names):
            raise ValueError("view_names must be non-empty and unique")
        if self.capture_offsets != (2, 5) or self.replan_stride != 8:
            raise ValueError("The accepted history schedule is capture_offsets=(2, 5), replan_stride=8")
        self._frames: dict[int, dict[str, np.ndarray]] = {}

    @property
    def step_ages(self) -> tuple[int, ...]:
        return tuple(self.replan_stride - offset for offset in self.capture_offsets)

    @property
    def captured_offsets(self) -> tuple[int, ...]:
        return tuple(offset for offset in self.capture_offsets if offset in self._frames)

    def reset(self) -> None:
        self._frames.clear()

    def capture(self, local_step: int, images_by_view: Mapping[str, np.ndarray]) -> bool:
        local_step = int(local_step)
        if local_step not in self.capture_offsets:
            return False
        if local_step in self._frames:
            raise ValueError(f"History offset {local_step} was captured more than once")
        normalized = _normalize_current_images(images_by_view, self.view_names)
        self._frames[local_step] = {name: image.copy() for name, image in normalized.items()}
        return True

    def consume(self, current_images_by_view: Mapping[str, np.ndarray]) -> SparseHistoryPayload:
        current = _normalize_current_images(current_images_by_view, self.view_names)
        valid_mask = np.array([offset in self._frames for offset in self.capture_offsets], dtype=np.bool_)
        history_by_view: dict[str, np.ndarray] = {}
        for view_name in self.view_names:
            slots: list[np.ndarray] = []
            for offset in self.capture_offsets:
                if offset in self._frames:
                    image = self._frames[offset][view_name]
                    if image.shape != current[view_name].shape:
                        raise ValueError(
                            f"Historical {view_name!r} shape {image.shape} does not match current {current[view_name].shape}"
                        )
                    slots.append(image)
                else:
                    slots.append(np.zeros_like(current[view_name], dtype=np.uint8))
            history_by_view[view_name] = np.ascontiguousarray(np.stack(slots, axis=0), dtype=np.uint8)
        payload = SparseHistoryPayload(
            images_by_view=history_by_view,
            step_ages=np.asarray(self.step_ages, dtype=np.int32),
            valid_mask=valid_mask,
        )
        self.reset()
        return payload


def empty_history_payload(
    current_images_by_view: Mapping[str, np.ndarray],
    *,
    view_names: Sequence[str] | None = None,
) -> SparseHistoryPayload:
    ordered_names = tuple(current_images_by_view) if view_names is None else tuple(view_names)
    return SparseHistoryBuffer(ordered_names).consume(current_images_by_view)


def _normalize_current_images(
    images_by_view: Mapping[str, np.ndarray],
    view_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if tuple(images_by_view) != view_names:
        raise ValueError(f"Expected ordered views {view_names}, got {tuple(images_by_view)}")
    normalized: dict[str, np.ndarray] = {}
    for view_name in view_names:
        image = np.asarray(images_by_view[view_name])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{view_name!r} must be an HxWx3 image")
        if image.dtype != np.uint8:
            raise ValueError(f"{view_name!r} must have dtype uint8")
        normalized[view_name] = np.ascontiguousarray(image)
    return normalized
