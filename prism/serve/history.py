from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


VisualObservationT = TypeVar("VisualObservationT")
MemoryT = TypeVar("MemoryT")


@dataclass(frozen=True)
class HistoryCaptureTarget:
    """Server-side memory destination for one within-chunk observation."""

    target_generation: int
    slot: int


class HistoryPrecomputeSchedule:
    """Track generations and the fixed O2/O5 capture schedule without storing images."""

    def __init__(
        self,
        *,
        capture_offsets: Sequence[int] = (2, 5),
        replan_stride: int = 8,
    ) -> None:
        self.capture_offsets = tuple(capture_offsets)
        self.replan_stride = replan_stride
        if (
            any(isinstance(offset, bool) or not isinstance(offset, int) for offset in self.capture_offsets)
            or isinstance(self.replan_stride, bool)
            or not isinstance(self.replan_stride, int)
            or self.capture_offsets != (2, 5)
            or self.replan_stride != 8
        ):
            raise ValueError("The accepted history schedule is capture_offsets=(2, 5), replan_stride=8")
        self.reset()

    @property
    def current_generation(self) -> int:
        return self._current_generation

    @property
    def scheduled_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._scheduled_slots))

    def reset(self) -> None:
        self._current_generation = 0
        self._scheduled_slots: set[int] = set()

    def target_for_step(self, local_step: int) -> HistoryCaptureTarget | None:
        if isinstance(local_step, bool) or not isinstance(local_step, int):
            raise ValueError(f"local_step must be an integer, got {local_step!r}")
        if local_step <= 0 or local_step > self.replan_stride:
            raise ValueError(f"local_step must be in [1, {self.replan_stride}], got {local_step}")
        try:
            slot = self.capture_offsets.index(local_step)
        except ValueError:
            return None
        if slot in self._scheduled_slots:
            raise ValueError(f"History slot {slot} at offset {local_step} was scheduled more than once")
        self._scheduled_slots.add(slot)
        return HistoryCaptureTarget(target_generation=self._current_generation + 1, slot=slot)

    def advance_generation(self) -> int:
        expected_slots = set(range(len(self.capture_offsets)))
        if self._scheduled_slots != expected_slots:
            raise RuntimeError(
                "Cannot advance history generation before both capture slots were scheduled: "
                f"got {sorted(self._scheduled_slots)}"
            )
        self._current_generation += 1
        self._scheduled_slots.clear()
        return self._current_generation


class ConnectionHistoryState(Generic[VisualObservationT, MemoryT]):
    """Bounded token-only history state owned by one WebSocket connection."""

    def __init__(self) -> None:
        self._stream_id: str | None = None
        self._last_inferred_generation = -1
        self._building_generation: int | None = None
        self._visual_slots: dict[int, VisualObservationT] = {}
        self._ready_generation: int | None = None
        self._ready_memory: MemoryT | None = None

    @property
    def stream_id(self) -> str | None:
        return self._stream_id

    @property
    def last_inferred_generation(self) -> int:
        return self._last_inferred_generation

    @property
    def cached_visual_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._visual_slots))

    @property
    def ready_generation(self) -> int | None:
        return self._ready_generation

    def reset(self, stream_id: str) -> None:
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        self.clear()
        self._stream_id = stream_id

    def clear(self) -> None:
        self._stream_id = None
        self._last_inferred_generation = -1
        self._building_generation = None
        self._visual_slots.clear()
        self._ready_generation = None
        self._ready_memory = None

    def add_observation(
        self,
        *,
        stream_id: str,
        target_generation: int,
        slot: int,
        observation: VisualObservationT,
        build_memory: Callable[[tuple[VisualObservationT, VisualObservationT]], MemoryT],
    ) -> bool:
        self._require_stream(stream_id)
        self._require_exact_int(target_generation, "target_generation")
        self._require_exact_int(slot, "slot")
        if slot not in (0, 1):
            raise ValueError(f"slot must be 0 or 1, got {slot}")
        expected_generation = self._last_inferred_generation + 1
        if target_generation <= 0 or target_generation != expected_generation:
            raise ValueError(
                f"Expected history for generation {expected_generation}, got {target_generation}"
            )
        if self._ready_generation is not None:
            raise RuntimeError(f"Memory for generation {self._ready_generation} is already ready")
        if self._building_generation is None:
            self._building_generation = target_generation
        elif self._building_generation != target_generation:
            raise ValueError(
                f"Already building generation {self._building_generation}, got {target_generation}"
            )
        if slot in self._visual_slots:
            raise ValueError(f"History slot {slot} for generation {target_generation} was pushed more than once")
        self._visual_slots[slot] = observation
        if len(self._visual_slots) < 2:
            return False

        ordered_observations = (self._visual_slots[0], self._visual_slots[1])
        self._visual_slots.clear()
        self._building_generation = None
        try:
            memory = build_memory(ordered_observations)
        except Exception:
            self._ready_generation = None
            self._ready_memory = None
            raise
        self._ready_generation = target_generation
        self._ready_memory = memory
        return True

    def memory_for_inference(
        self,
        *,
        stream_id: str,
        generation: int,
        empty_memory: Callable[[], MemoryT],
    ) -> MemoryT:
        self._require_stream(stream_id)
        self._require_exact_int(generation, "generation")
        expected_generation = self._last_inferred_generation + 1
        if generation != expected_generation:
            raise ValueError(f"Expected inference generation {expected_generation}, got {generation}")
        if generation == 0:
            return empty_memory()
        if self._ready_generation != generation or self._ready_memory is None:
            if self._building_generation == generation:
                missing = sorted({0, 1} - set(self._visual_slots))
                raise RuntimeError(
                    f"Memory for generation {generation} is not ready; missing history slots {missing}"
                )
            raise RuntimeError(f"Memory for generation {generation} is not ready")
        return self._ready_memory

    def mark_inference_complete(self, *, stream_id: str, generation: int) -> None:
        self._require_stream(stream_id)
        self._require_exact_int(generation, "generation")
        expected_generation = self._last_inferred_generation + 1
        if generation != expected_generation:
            raise ValueError(f"Expected completed generation {expected_generation}, got {generation}")
        if generation > 0 and (self._ready_generation != generation or self._ready_memory is None):
            raise RuntimeError(f"Memory for generation {generation} was not ready")
        self._last_inferred_generation = generation
        self._ready_generation = None
        self._ready_memory = None

    def _require_stream(self, stream_id: str) -> None:
        if self._stream_id is None:
            raise RuntimeError("History stream is not initialized; send reset_history first")
        if stream_id != self._stream_id:
            raise ValueError(f"Active stream is {self._stream_id!r}, got {stream_id!r}")

    @staticmethod
    def _require_exact_int(value: int, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer, got {value!r}")
