from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_rank_zero_checkpoint_io_failure_reaches_every_rank(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "tests" / "training" / "distributed_checkpoint_failure_worker.py"
    output = tmp_path / "collective-error.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
            str(output),
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    outcomes = json.loads(output.read_text(encoding="utf-8"))
    assert outcomes == [outcomes[0], outcomes[0]]
    assert outcomes[0]["type"] == "CheckpointError"
    assert "prepare checkpoint staging" in outcomes[0]["message"]
    assert "injected rank-zero provenance I/O failure" in outcomes[0]["message"]
    assert not (tmp_path / "checkpoint").exists()
