"""Opt-in real-model training smoke for the remote data-disk environment."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from prism.training.checkpoint import read_checkpoint_metadata
from prism.training.config import load_train_config
from prism.training.runner import run_resolved_training


def test_real_single_gpu_training_step_and_checkpoint() -> None:
    if os.environ.get("PRISM_RUN_TRAIN_SMOKE") != "1":
        pytest.skip("set PRISM_RUN_TRAIN_SMOKE=1 in the remote model environment")
    output_value = os.environ.get("PRISM_TRAIN_SMOKE_OUTPUT_DIR")
    if not output_value:
        pytest.skip("set PRISM_TRAIN_SMOKE_OUTPUT_DIR to a new data-disk directory")

    project_root = Path(__file__).resolve().parents[2]
    config_path = os.environ.get(
        "PRISM_TRAIN_SMOKE_CONFIG",
        "configs/train/calvin_smoke.yaml",
    )
    output_dir = Path(output_value).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"smoke output must be a new immutable artifact directory: {output_dir}"
        )
    config = load_train_config(config_path, project_root=project_root)
    config = replace(
        config,
        experiment=replace(config.experiment, output_dir=output_dir),
    )

    progress = run_resolved_training(config)

    checkpoint = output_dir / "checkpoints" / "step-00000001"
    metadata = read_checkpoint_metadata(checkpoint)
    assert progress.completed_optimizer_steps == 1
    assert metadata.progress == progress
    assert metadata.architecture_sha256 == config.model.architecture_sha256
    assert metadata.statistics_sha256 == config.data.normalization.content_sha256
