"""Two-process worker proving rank-zero checkpoint failures are collective."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from accelerate import Accelerator
import torch.distributed as dist

import prism.training.checkpoint as checkpoint
from tests.training.test_checkpoint import _progress, _snapshot


def main(output_path: str) -> None:
    accelerator = Accelerator(cpu=True)
    if accelerator.is_main_process:

        def fail_git_metadata(_repository_root: Path) -> dict:
            raise OSError("injected rank-zero provenance I/O failure")

        checkpoint.collect_git_metadata = fail_git_metadata

    try:
        checkpoint.save_checkpoint(
            Path(output_path).parent / "checkpoint",
            accelerator=accelerator,
            config=_snapshot(),
            progress=_progress(0),
        )
    except Exception as exc:
        local_result = {"type": type(exc).__name__, "message": str(exc)}
    else:
        local_result = {"type": None, "message": "checkpoint unexpectedly succeeded"}

    results: list[dict | None] = [None] * accelerator.num_processes
    dist.all_gather_object(results, local_result)
    if accelerator.is_main_process:
        Path(output_path).write_text(json.dumps(results, sort_keys=True), encoding="utf-8")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_checkpoint_failure_worker.py OUTPUT_JSON")
    main(sys.argv[1])
