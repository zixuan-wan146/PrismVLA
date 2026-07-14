"""Two-process worker proving checkpoint restore failures are collective."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Callable

from accelerate import Accelerator
import torch.distributed as dist

import prism.training.checkpoint as checkpoint
from tests.training.test_checkpoint import _progress, _snapshot


class _LoadFailureAccelerator:
    """Delegate to Accelerate while injecting one rank-local load failure."""

    def __init__(self, accelerator: Accelerator, *, fail_on_rank: int) -> None:
        self._accelerator = accelerator
        self.process_index = accelerator.process_index
        self.num_processes = accelerator.num_processes
        self.is_main_process = accelerator.is_main_process
        self._fail_on_rank = fail_on_rank

    def save_state(self, output_dir: str) -> None:
        self._accelerator.save_state(output_dir)

    def load_state(self, input_dir: str) -> None:
        if self.process_index == self._fail_on_rank:
            raise OSError("injected rank-zero Accelerate load failure")
        self._accelerator.load_state(input_dir)

    def wait_for_everyone(self) -> None:
        self._accelerator.wait_for_everyone()


def _collect_outcome(
    accelerator: Accelerator,
    operation: Callable[[], object],
) -> list[dict | None]:
    try:
        operation()
    except Exception as exc:
        local_result = {"type": type(exc).__name__, "message": str(exc)}
    else:
        local_result = {"type": None, "message": "checkpoint unexpectedly loaded"}
    results: list[dict | None] = [None] * accelerator.num_processes
    dist.all_gather_object(results, local_result)
    return results


def main(output_path: str) -> None:
    accelerator = Accelerator(cpu=True)
    destination = checkpoint.save_checkpoint(
        Path(output_path).parent / "checkpoint",
        accelerator=accelerator,
        config=_snapshot(),
        progress=_progress(0),
    )

    load_results = _collect_outcome(
        accelerator,
        lambda: checkpoint.load_checkpoint(
            destination,
            accelerator=_LoadFailureAccelerator(accelerator, fail_on_rank=0),
            expected_config=_snapshot(),
        ),
    )
    accelerator.wait_for_everyone()

    original_restore_rng_state = checkpoint.restore_rng_state
    if accelerator.process_index == 1:

        def fail_restore_rng_state(_payload: dict) -> None:
            raise RuntimeError("injected rank-one RNG restore failure")

        checkpoint.restore_rng_state = fail_restore_rng_state
    try:
        rng_results = _collect_outcome(
            accelerator,
            lambda: checkpoint.load_checkpoint(
                destination,
                accelerator=accelerator,
                expected_config=_snapshot(),
            ),
        )
    finally:
        checkpoint.restore_rng_state = original_restore_rng_state

    if accelerator.is_main_process:
        Path(output_path).write_text(
            json.dumps(
                {
                    "load_state": load_results,
                    "restore_rng_state": rng_results,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: distributed_checkpoint_load_failure_worker.py OUTPUT_JSON"
        )
    main(sys.argv[1])
