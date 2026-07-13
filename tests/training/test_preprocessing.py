from __future__ import annotations

import threading

import pytest
import torch

from prism.training.preprocessing import iter_preprocessed_batches


@pytest.mark.parametrize("workers", [0, 1])
def test_preprocessed_batches_preserve_order(workers: int) -> None:
    thread_names: list[str] = []

    def collator(value: int) -> torch.Tensor:
        thread_names.append(threading.current_thread().name)
        return torch.tensor([value], dtype=torch.float32)

    batches = list(
        iter_preprocessed_batches(
            [1, 2, 3],
            collator,
            preprocessing_workers=workers,
            pin_memory=False,
        )
    )

    assert [raw for raw, _ in batches] == [1, 2, 3]
    assert [batch.item() for _, batch in batches] == [1, 2, 3]
    if workers == 1:
        assert all(name.startswith("prism-preprocess") for name in thread_names)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires the remote CUDA runtime")
def test_preprocessed_tensor_is_pinned_before_device_transfer() -> None:
    [(_, batch)] = list(
        iter_preprocessed_batches(
            [1],
            lambda value: torch.tensor([value], dtype=torch.float32),
            preprocessing_workers=1,
            pin_memory=True,
        )
    )

    assert batch.is_pinned()
