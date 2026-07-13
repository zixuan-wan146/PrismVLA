from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch

from experiments.calvin.data import CALVIN_DATA_SPEC
from prism.data.normalization import canonical_sha256
from prism.data.normalization import compute_statistics
from prism.training.checkpoint import MANIFEST_FILENAME
from prism.training.checkpoint import METADATA_FILENAME
from prism.training.checkpoint import CheckpointMetadata
from prism.training.checkpoint import TrainingProgress
from prism.training.checkpoint import load_checkpoint
from prism.training.checkpoint import read_checkpoint_metadata
from prism.training.checkpoint import save_checkpoint
from prism.training.config import TRAIN_CONFIG_SNAPSHOT_FORMAT


class _FakeAccelerator:
    process_index = 0
    num_processes = 1
    is_main_process = True

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        fail_on_save: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fail_on_save = fail_on_save
        self.save_calls: list[Path] = []
        self.load_calls: list[Path] = []
        self.wait_calls = 0

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def save_state(self, output_dir: str) -> None:
        directory = Path(output_dir)
        self.save_calls.append(directory)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            directory / "accelerator_state.pt",
        )
        if self.fail_on_save:
            raise RuntimeError("injected save failure")

    def load_state(self, input_dir: str) -> None:
        directory = Path(input_dir)
        self.load_calls.append(directory)
        state = torch.load(
            directory / "accelerator_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


def _objects() -> tuple[
    torch.nn.Linear,
    torch.optim.SGD,
    torch.optim.lr_scheduler.StepLR,
]:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    return model, optimizer, scheduler


def _snapshot(
    *,
    experiment_name: str = "checkpoint-fixture",
    gradient_accumulation_steps: int = 1,
    global_samples_per_epoch: int = 16,
    batch_size_per_rank: int = 1,
    drop_last: bool = True,
) -> dict[str, Any]:
    data_spec = asdict(CALVIN_DATA_SPEC)
    schema_hash = canonical_sha256(data_spec)
    state_values = np.asarray(
        [
            [0.0, 1.0, 2.0, 0.1, 0.2, 0.3, 0.0, 0.02],
            [1.0, 2.0, 3.0, 0.2, 0.3, 0.4, 0.0, 0.03],
            [2.0, 3.0, 4.0, 0.3, 0.4, 0.5, 0.0, 0.04],
            [3.0, 4.0, 5.0, 0.4, 0.5, 0.6, 0.0, 0.05],
        ],
        dtype=np.float32,
    )
    action_values = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    statistics = compute_statistics(
        state_values,
        action_values,
        group="calvin_abc",
        robot_key="calvin",
        datasets=("task_ABC_D",),
        schema_hash=schema_hash,
        provenance={"train_splits": ["A", "B", "C"], "eval_splits": ["D"]},
        state_continuous_indices=(0, 1, 2, 3, 4, 5, 7),
    )
    architecture = {
        "backbone": {"model_name": "fixture"},
        "temporal": {
            "action_horizon": 8,
            "replan_stride": 8,
            "history_capture_offsets": [2, 5],
        },
        "action_head": {
            "objective": "direct_masked_l1",
            "action_dim": 7,
            "action_hidden_size": 32,
            "num_attention_heads": 4,
            "ffn_ratio": 2.0,
        },
    }
    return {
        "format": TRAIN_CONFIG_SNAPSHOT_FORMAT,
        "source_config": "configs/train/fixture.yaml",
        "experiment": {
            "name": experiment_name,
            "output_dir": "outputs/checkpoint-fixture",
            "seed": 17,
        },
        "model": {
            "architecture_config": "configs/model/fixture.yaml",
            "architecture_sha256": canonical_sha256(architecture),
            "architecture": architecture,
        },
        "data": {
            "spec": "experiments.calvin.data:CALVIN_DATA_SPEC",
            "data_spec_sha256": schema_hash,
            "data_spec": data_spec,
            "root": "data/calvin",
            "anchor_stride": 1,
            "include_tail": True,
            "datasets": [
                {
                    "name": "task_ABC_D",
                    "path": "data/calvin/task_ABC_D",
                    "weight": 1.0,
                    "splits": ["A", "B", "C"],
                }
            ],
            "normalization": {
                "group": "calvin_abc",
                "statistics_path": "data/calvin/statistics.json",
                "content_sha256": statistics["content_sha256"],
                "statistics": statistics,
            },
            "loader": {
                "global_samples_per_epoch": global_samples_per_epoch,
                "batch_size_per_rank": batch_size_per_rank,
                "num_workers": 0,
                "preprocessing_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": drop_last,
            },
            "train_splits": ["A", "B", "C"],
            "eval_splits": ["D"],
        },
        "optimization": {
            "optimizer": "adamw",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1.0e-8,
            "no_decay_rule": "bias_and_low_dimensional",
            "language_model": {
                "trainable": False,
                "learning_rate": None,
                "weight_decay": None,
            },
            "vision_encoder": {
                "trainable": False,
                "learning_rate": None,
                "weight_decay": None,
            },
            "action_queries": {
                "trainable": True,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
            },
            "history_qformer": {
                "trainable": True,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.01,
            },
            "action_head": {
                "trainable": True,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.01,
            },
        },
        "trainer": {
            "max_steps": 100,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision": "no",
            "scheduler": "linear_warmup_decay",
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "log_interval": 1,
            "save_interval": 1,
        },
        "derived": {
            "temporal_contract": {
                "action_horizon": 8,
                "replan_stride": 8,
                "history_capture_offsets": [2, 5],
                "history_step_ages": [6, 3],
                "num_history_frames": 2,
                "num_ordered_views": 2,
            },
            "source": "model.architecture.temporal",
        },
    }


def _progress(step: int) -> TrainingProgress:
    return TrainingProgress(
        completed_optimizer_steps=step,
        gradient_accumulation_micro_step=0,
        epoch=0,
        virtual_sample_cursor=step,
        virtual_batch_cursor=step,
    )


def test_checkpoint_is_atomic_immutable_and_embeds_complete_metadata(tmp_path: Path):
    torch.manual_seed(3)
    model, optimizer, scheduler = _objects()
    accelerator = _FakeAccelerator(model, optimizer, scheduler)
    snapshot = _snapshot()
    destination = tmp_path / "step-00000002"

    returned = save_checkpoint(
        destination,
        accelerator=accelerator,
        config=snapshot,
        progress=_progress(2),
    )

    assert returned == destination.resolve()
    assert destination.is_dir()
    assert not (tmp_path / ".step-00000002.incomplete").exists()
    assert accelerator.save_calls == [tmp_path / ".step-00000002.incomplete"]
    assert (destination / MANIFEST_FILENAME).is_file()
    assert (destination / METADATA_FILENAME).is_file()
    assert (destination / "prism_rng" / "rank-00000.json").is_file()

    metadata = read_checkpoint_metadata(destination)
    assert isinstance(metadata, CheckpointMetadata)
    assert metadata.progress == _progress(2)
    assert metadata.world_size == 1
    assert metadata.resolved_train_snapshot_sha256 == canonical_sha256(snapshot)
    assert metadata.architecture_sha256 == snapshot["model"]["architecture_sha256"]
    assert metadata.data_spec_sha256 == snapshot["data"]["data_spec_sha256"]
    assert metadata.statistics_sha256 == snapshot["data"]["normalization"]["content_sha256"]
    assert metadata.normalization_statistics["content_sha256"] == metadata.statistics_sha256
    assert len(metadata.git["commit"]) in {40, 64}
    assert isinstance(metadata.git["dirty"], bool)
    assert metadata.environment["accelerate"]
    assert metadata.environment["torch"]

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        save_checkpoint(
            destination,
            accelerator=accelerator,
            config=snapshot,
            progress=_progress(2),
        )


def test_interrupted_save_never_creates_a_complete_checkpoint(tmp_path: Path):
    model, optimizer, scheduler = _objects()
    accelerator = _FakeAccelerator(
        model,
        optimizer,
        scheduler,
        fail_on_save=True,
    )
    destination = tmp_path / "step-00000001"

    with pytest.raises(RuntimeError, match="injected save failure"):
        save_checkpoint(
            destination,
            accelerator=accelerator,
            config=_snapshot(),
            progress=_progress(1),
        )

    assert not destination.exists()
    assert (tmp_path / ".step-00000001.incomplete").is_dir()
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        read_checkpoint_metadata(tmp_path / ".step-00000001.incomplete")


def test_load_rejects_config_mismatch_and_corruption_before_accelerator_state(
    tmp_path: Path,
):
    model, optimizer, scheduler = _objects()
    accelerator = _FakeAccelerator(model, optimizer, scheduler)
    snapshot = _snapshot()
    destination = save_checkpoint(
        tmp_path / "step-00000001",
        accelerator=accelerator,
        config=snapshot,
        progress=_progress(1),
    )

    mismatched = deepcopy(snapshot)
    mismatched["experiment"]["name"] = "different-run"
    with pytest.raises(ValueError, match="resolved train config hash mismatch"):
        load_checkpoint(
            destination,
            accelerator=accelerator,
            expected_config=mismatched,
        )
    assert accelerator.load_calls == []

    with (destination / "accelerator_state.pt").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="file size mismatch"):
        load_checkpoint(
            destination,
            accelerator=accelerator,
            expected_config=snapshot,
        )
    assert accelerator.load_calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("architecture", "architecture objective"),
        ("schema", "DataSpec hash mismatch"),
        ("statistics", "statistics content hash mismatch"),
    ],
)
def test_embedded_architecture_schema_and_statistics_are_independently_verified(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    model, optimizer, scheduler = _objects()
    accelerator = _FakeAccelerator(model, optimizer, scheduler)
    destination = save_checkpoint(
        tmp_path / mutation,
        accelerator=accelerator,
        config=_snapshot(),
        progress=_progress(1),
    )
    metadata_path = destination / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    snapshot = metadata["resolved_train_snapshot"]
    if mutation == "architecture":
        snapshot["model"]["architecture"]["action_head"]["objective"] = "flow_matching"
    elif mutation == "schema":
        snapshot["data"]["data_spec"]["name"] = "tampered"
    else:
        snapshot["data"]["normalization"]["statistics"]["groups"]["calvin_abc"]["state"]["count"] += 1
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    _refresh_manifest_entry(destination, METADATA_FILENAME)

    with pytest.raises(ValueError, match=message):
        read_checkpoint_metadata(destination)


def test_nonzero_accumulation_micro_step_is_rejected_without_false_exact_resume_claim(
    tmp_path: Path,
):
    model, optimizer, scheduler = _objects()
    accelerator = _FakeAccelerator(model, optimizer, scheduler)
    progress = TrainingProgress(
        completed_optimizer_steps=1,
        gradient_accumulation_micro_step=1,
        epoch=0,
        virtual_sample_cursor=3,
        virtual_batch_cursor=3,
    )

    with pytest.raises(ValueError, match="optimizer synchronization boundary"):
        save_checkpoint(
            tmp_path / "mid-accumulation",
            accelerator=accelerator,
            config=_snapshot(gradient_accumulation_steps=4),
            progress=progress,
        )
    assert accelerator.save_calls == []


def test_global_sample_and_per_rank_batch_cursor_must_agree(tmp_path: Path):
    model, optimizer, scheduler = _objects()
    accelerator = _FakeAccelerator(model, optimizer, scheduler)
    inconsistent = TrainingProgress(
        completed_optimizer_steps=1,
        gradient_accumulation_micro_step=0,
        epoch=0,
        virtual_sample_cursor=5,
        virtual_batch_cursor=2,
    )

    with pytest.raises(ValueError, match="sample/batch cursor mismatch"):
        save_checkpoint(
            tmp_path / "bad-cursor",
            accelerator=accelerator,
            config=_snapshot(batch_size_per_rank=2),
            progress=inconsistent,
        )


def test_interrupt_resume_preserves_sample_identity_rng_and_loss_continuity(
    tmp_path: Path,
):
    baseline_model, baseline_optimizer, baseline_scheduler, baseline_accelerator = _seeded_run()
    baseline = _run_steps(
        baseline_model,
        baseline_optimizer,
        baseline_scheduler,
        start=0,
        stop=7,
    )

    model, optimizer, scheduler, accelerator = _seeded_run()
    prefix = _run_steps(model, optimizer, scheduler, start=0, stop=3)
    assert prefix == baseline[:3]
    destination = save_checkpoint(
        tmp_path / "step-00000003",
        accelerator=accelerator,
        config=_snapshot(),
        progress=_progress(3),
    )

    with torch.no_grad():
        model.weight.fill_(123.0)
    random.random()
    np.random.random()
    torch.rand(())

    restored = load_checkpoint(
        destination,
        accelerator=accelerator,
        expected_config=_snapshot(),
    )
    resumed = _run_steps(
        model,
        optimizer,
        scheduler,
        start=restored.virtual_batch_cursor,
        stop=7,
    )

    assert [row[0] for row in resumed] == [row[0] for row in baseline[3:]]
    assert resumed == baseline[3:]
    assert torch.equal(model.weight, baseline_model.weight)
    assert optimizer.state_dict() == baseline_optimizer.state_dict()
    assert scheduler.state_dict() == baseline_scheduler.state_dict()
    assert accelerator.load_calls == [destination.resolve()]
    assert baseline_accelerator.load_calls == []


def _seeded_run() -> tuple[
    torch.nn.Linear,
    torch.optim.SGD,
    torch.optim.lr_scheduler.StepLR,
    _FakeAccelerator,
]:
    random.seed(811)
    np.random.seed(812)
    torch.manual_seed(813)
    model, optimizer, scheduler = _objects()
    return model, optimizer, scheduler, _FakeAccelerator(model, optimizer, scheduler)


def _run_steps(
    model: torch.nn.Linear,
    optimizer: torch.optim.SGD,
    scheduler: torch.optim.lr_scheduler.StepLR,
    *,
    start: int,
    stop: int,
) -> list[tuple[str, float, float, float, float]]:
    records: list[tuple[str, float, float, float, float]] = []
    for virtual_batch_cursor in range(start, stop):
        identity = hashlib.sha256(f"epoch=0:virtual={virtual_batch_cursor}".encode("ascii")).hexdigest()
        python_draw = random.random()
        numpy_draw = float(np.random.random())
        torch_draw = float(torch.rand(()))
        identity_value = int(identity[:8], 16) / float(1 << 32)
        input_value = torch.tensor(
            [[identity_value + python_draw + numpy_draw + torch_draw]],
            dtype=torch.float32,
        )
        target = torch.tensor([[identity_value * 0.5 - 0.25]], dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(input_value), target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        records.append(
            (
                identity,
                python_draw,
                numpy_draw,
                torch_draw,
                float(loss.detach()),
            )
        )
    return records


def _refresh_manifest_entry(checkpoint: Path, relative_path: str) -> None:
    manifest_path = checkpoint / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = checkpoint / relative_path
    payload = target.read_bytes()
    for row in manifest["files"]:
        if row["path"] == relative_path:
            row["size_bytes"] = len(payload)
            row["sha256"] = hashlib.sha256(payload).hexdigest()
            break
    else:
        raise AssertionError(f"manifest row not found: {relative_path}")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
