"""One-batch CPU preprocessing pipeline for model-owned policy collation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from prism.models.batch import PolicyBatch


def iter_preprocessed_batches(
    raw_batches: Iterable[Any],
    collator: Any,
    *,
    preprocessing_workers: int,
    pin_memory: bool,
) -> Iterator[tuple[Any, Any]]:
    """Collate one batch ahead so CPU processing can overlap GPU execution."""

    if type(preprocessing_workers) is not int or preprocessing_workers not in {0, 1}:
        raise ValueError("preprocessing_workers must be 0 or 1")
    if type(pin_memory) is not bool:
        raise TypeError("pin_memory must be a boolean")

    iterator = iter(raw_batches)
    if preprocessing_workers == 0:
        for raw_batch in iterator:
            yield raw_batch, _prepare_batch(raw_batch, collator, pin_memory=pin_memory)
        return

    try:
        current_raw = next(iterator)
    except StopIteration:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="prism-preprocess") as executor:
        current_future = executor.submit(_prepare_batch, current_raw, collator, pin_memory=pin_memory)
        for next_raw in iterator:
            current_batch = current_future.result()
            next_future = executor.submit(_prepare_batch, next_raw, collator, pin_memory=pin_memory)
            yield current_raw, current_batch
            current_raw = next_raw
            current_future = next_future
        yield current_raw, current_future.result()


def _prepare_batch(raw_batch: Any, collator: Any, *, pin_memory: bool) -> Any:
    batch = collator(raw_batch)
    if not pin_memory:
        return batch
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory()
    if not isinstance(batch, PolicyBatch):
        raise TypeError(f"model-owned collator must return PolicyBatch, got {type(batch).__name__}")
    return PolicyBatch(
        current_inputs=_pin_mapping(batch.current_inputs),
        history_inputs=_pin_mapping(batch.history_inputs),
        history_step_ages=batch.history_step_ages.pin_memory(),
        history_valid_mask=batch.history_valid_mask.pin_memory(),
        state=batch.state.pin_memory(),
        executed_actions=batch.executed_actions.pin_memory(),
        executed_action_valid_mask=batch.executed_action_valid_mask.pin_memory(),
        target_actions=batch.target_actions.pin_memory(),
        action_valid_mask=batch.action_valid_mask.pin_memory(),
        action_dim_mask=None if batch.action_dim_mask is None else batch.action_dim_mask.pin_memory(),
    )


def _pin_mapping(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.pin_memory() for name, value in values.items()}


__all__ = ["iter_preprocessed_batches"]
