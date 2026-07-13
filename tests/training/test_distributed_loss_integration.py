from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_real_two_process_ddp_uses_global_masked_denominator(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker = project_root / "tests" / "training" / "distributed_loss_worker.py"
    output = tmp_path / "result.json"
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
        timeout=120,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["weights"] == pytest.approx([0.96, 0.96])
    assert payload["metrics"]["total_l1"] == pytest.approx(7.0 / 175.0)
    assert payload["metrics"]["gripper_transition_recall"] == pytest.approx(1.0)
