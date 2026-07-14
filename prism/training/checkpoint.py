"""Atomic, content-verified lifecycle for Accelerate training checkpoints.

Accelerate owns model, optimizer, scheduler, scaler, and distributed state.
This module coordinates publication and restoration while delegating artifact
I/O and serialized Prism metadata to their focused modules.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any, Callable

import torch.distributed as dist

from prism.data.normalization import canonical_json_bytes
from prism.training.artifact_io import MANIFEST_FILENAME
from prism.training.artifact_io import MANIFEST_FORMAT
from prism.training.artifact_io import build_manifest
from prism.training.artifact_io import fsync_directory
from prism.training.artifact_io import read_json
from prism.training.artifact_io import sha256_file
from prism.training.artifact_io import verify_manifest
from prism.training.artifact_io import write_json_atomic
from prism.training.checkpoint_metadata import CHECKPOINT_FORMAT
from prism.training.checkpoint_metadata import METADATA_FILENAME
from prism.training.checkpoint_metadata import RNG_DIRECTORY
from prism.training.checkpoint_metadata import CheckpointMetadata
from prism.training.checkpoint_metadata import TrainingProgress
from prism.training.checkpoint_metadata import build_metadata_payload
from prism.training.checkpoint_metadata import capture_rng_state
from prism.training.checkpoint_metadata import collect_environment_versions
from prism.training.checkpoint_metadata import collect_git_metadata
from prism.training.checkpoint_metadata import parse_metadata
from prism.training.checkpoint_metadata import resolve_snapshot
from prism.training.checkpoint_metadata import restore_rng_state
from prism.training.checkpoint_metadata import validate_progress
from prism.training.checkpoint_metadata import validate_rng_payload
from prism.training.config import ResolvedTrainConfig


class CheckpointError(RuntimeError):
    """A checkpoint phase failed and the same error was propagated to every rank."""


def save_checkpoint(
    path: str | Path,
    *,
    accelerator: Any,
    config: ResolvedTrainConfig | Mapping[str, Any],
    progress: TrainingProgress,
) -> Path:
    """Save registered Accelerate state plus complete Prism metadata atomically.

    The model, optimizer, scheduler, and scaler must already be prepared or
    registered with ``accelerator``. The final path must not exist: checkpoints
    are immutable and are never silently overwritten.
    """

    context = _accelerator_context(accelerator)
    snapshot, snapshot_hashes = resolve_snapshot(config)
    if not isinstance(progress, TrainingProgress):
        raise TypeError(f"progress must be TrainingProgress, got {type(progress).__name__}")
    validate_progress(progress, snapshot, world_size=context["world_size"])

    target = _checkpoint_path(path)
    staging = target.with_name(f".{target.name}.incomplete")
    if os.path.lexists(target):
        raise FileExistsError(f"checkpoint already exists and will not be overwritten: {target}")
    if os.path.lexists(staging):
        raise FileExistsError(
            f"incomplete checkpoint staging directory already exists: {staging}; "
            "inspect or remove it explicitly before retrying"
        )

    git_metadata: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None

    def prepare_staging() -> None:
        nonlocal git_metadata, environment
        repository_root = config.project_root if isinstance(config, ResolvedTrainConfig) else Path.cwd()
        git_metadata = collect_git_metadata(repository_root)
        environment = collect_environment_versions()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()

    _run_collective_checkpoint_phase(
        context=context,
        phase="prepare checkpoint staging",
        operation=prepare_staging,
        main_process_only=True,
    )

    rng_payload: dict[str, Any] | None = None

    def capture_rank_rng() -> None:
        nonlocal rng_payload
        rng_payload = capture_rng_state(rank=context["rank"])

    _run_collective_checkpoint_phase(
        context=context,
        phase="capture rank RNG state",
        operation=capture_rank_rng,
    )
    assert rng_payload is not None

    _run_collective_checkpoint_phase(
        context=context,
        phase="save Accelerate state",
        operation=lambda: accelerator.save_state(str(staging)),
    )

    def validate_accelerator_state() -> None:
        if not any(entry.is_file() for entry in staging.rglob("*")):
            raise RuntimeError(
                "Accelerator.save_state produced no files; prepare/register model, optimizer, "
                "and scheduler before checkpointing"
            )

    _run_collective_checkpoint_phase(
        context=context,
        phase="validate Accelerate state",
        operation=validate_accelerator_state,
        main_process_only=True,
    )

    rng_relative = f"{RNG_DIRECTORY}/rank-{context['rank']:05d}.json"

    _run_collective_checkpoint_phase(
        context=context,
        phase="write rank RNG state",
        operation=lambda: write_json_atomic(staging / rng_relative, rng_payload),
    )

    def publish_checkpoint() -> None:
        assert git_metadata is not None
        assert environment is not None
        rng_rows: list[dict[str, Any]] = []
        for rank in range(context["world_size"]):
            relative = f"{RNG_DIRECTORY}/rank-{rank:05d}.json"
            rng_path = staging / relative
            if not rng_path.is_file():
                raise RuntimeError(f"missing explicit RNG state for rank {rank}: {rng_path}")
            payload = read_json(rng_path, label=f"rank {rank} RNG state")
            validate_rng_payload(payload, expected_rank=rank)
            rng_rows.append(
                {
                    "rank": rank,
                    "path": relative,
                    "sha256": sha256_file(rng_path),
                }
            )

        metadata_payload = build_metadata_payload(
            world_size=context["world_size"],
            progress=progress,
            snapshot=snapshot,
            snapshot_hashes=snapshot_hashes,
            git_metadata=git_metadata,
            environment=environment,
            rng_rows=rng_rows,
        )
        write_json_atomic(staging / METADATA_FILENAME, metadata_payload)
        manifest = build_manifest(staging)
        write_json_atomic(staging / MANIFEST_FILENAME, manifest)
        fsync_directory(staging)
        os.replace(staging, target)
        fsync_directory(target.parent)

    _run_collective_checkpoint_phase(
        context=context,
        phase="publish checkpoint",
        operation=publish_checkpoint,
        main_process_only=True,
    )
    return target


def load_checkpoint(
    path: str | Path,
    *,
    accelerator: Any,
    expected_config: ResolvedTrainConfig | Mapping[str, Any],
) -> TrainingProgress:
    """Validate and restore one complete checkpoint.

    This restores registered Accelerate state and the current rank's explicit
    Python/NumPy/Torch RNG. It does not construct or seek a DataLoader. The
    caller must rebuild the deterministic epoch iterator and skip exactly
    ``virtual_batch_cursor`` local batches before consuming the next batch.
    """

    context = _accelerator_context(accelerator)
    checkpoint = _checkpoint_path(path)
    metadata = read_checkpoint_metadata(checkpoint)
    expected_snapshot, expected_hashes = resolve_snapshot(expected_config)

    if metadata.world_size != context["world_size"]:
        raise ValueError(
            f"checkpoint world size mismatch: stored {metadata.world_size}, current {context['world_size']}"
        )
    comparisons = (
        ("architecture", metadata.architecture_sha256, expected_hashes.architecture),
        ("DataSpec schema", metadata.data_spec_sha256, expected_hashes.data_spec),
        ("normalization statistics", metadata.statistics_sha256, expected_hashes.statistics),
        (
            "resolved train config",
            metadata.resolved_train_snapshot_sha256,
            expected_hashes.config,
        ),
    )
    for label, stored, expected in comparisons:
        if stored != expected:
            raise ValueError(f"checkpoint {label} hash mismatch: stored {stored}, expected {expected}")
    if canonical_json_bytes(metadata.resolved_train_snapshot) != canonical_json_bytes(expected_snapshot):
        raise ValueError("checkpoint resolved train snapshot differs from expected config")

    validate_progress(metadata.progress, expected_snapshot, world_size=context["world_size"])
    rng_relative = metadata.rng_rank_files[context["rank"]]
    rng_payload = read_json(checkpoint / rng_relative, label=f"rank {context['rank']} RNG state")
    validate_rng_payload(rng_payload, expected_rank=context["rank"])

    accelerator.wait_for_everyone()
    accelerator.load_state(str(checkpoint))
    accelerator.wait_for_everyone()
    restore_rng_state(rng_payload)
    return metadata.progress


def read_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    """Read hash-verified metadata without loading model or optimizer state."""

    checkpoint = _checkpoint_path(path)
    if checkpoint.name.startswith(".") and checkpoint.name.endswith(".incomplete"):
        raise ValueError(f"refusing to read an incomplete checkpoint staging directory: {checkpoint}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    verify_manifest(checkpoint, required_paths=(METADATA_FILENAME,))
    payload = read_json(checkpoint / METADATA_FILENAME, label="checkpoint metadata")
    return parse_metadata(payload, checkpoint=checkpoint)


def _accelerator_context(accelerator: Any) -> dict[str, Any]:
    for method in ("save_state", "load_state", "wait_for_everyone"):
        if not callable(getattr(accelerator, method, None)):
            raise TypeError(f"accelerator must provide callable {method}()")
    rank = _non_negative_int(getattr(accelerator, "process_index", None), "accelerator process_index")
    world_size = _positive_int(getattr(accelerator, "num_processes", None), "accelerator num_processes")
    is_main = getattr(accelerator, "is_main_process", None)
    if type(is_main) is not bool:
        raise TypeError("accelerator is_main_process must be a boolean")
    if rank >= world_size:
        raise ValueError(f"accelerator process_index={rank} is outside world_size={world_size}")
    if is_main != (rank == 0):
        raise ValueError("accelerator is_main_process must be true exactly on process_index zero")
    return {"rank": rank, "world_size": world_size, "is_main_process": is_main}


def _run_collective_checkpoint_phase(
    *,
    context: Mapping[str, Any],
    phase: str,
    operation: Callable[[], None],
    main_process_only: bool = False,
) -> None:
    """Run one phase and all-gather a serializable error before any later phase."""

    rank = int(context["rank"])
    world_size = int(context["world_size"])
    local_error: dict[str, Any] | None = None
    if not main_process_only or bool(context["is_main_process"]):
        try:
            operation()
        except Exception as exc:
            local_error = {
                "rank": rank,
                "exception": type(exc).__name__,
                "message": str(exc),
            }

    errors: list[dict[str, Any] | None]
    if world_size == 1:
        errors = [local_error]
    else:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("multi-process checkpointing requires an initialized torch.distributed group")
        errors = [None] * world_size
        dist.all_gather_object(errors, local_error)

    failures = [error for error in errors if error is not None]
    if failures:
        failure = min(failures, key=lambda value: int(value["rank"]))
        raise CheckpointError(
            f"checkpoint phase {phase!r} failed on rank {failure['rank']}: "
            f"{failure['exception']}: {failure['message']}"
        ) from None


def _checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.name:
        raise ValueError(f"checkpoint path must name a directory, got {path!r}")
    return candidate.resolve(strict=False)


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


__all__ = [
    "CHECKPOINT_FORMAT",
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT",
    "METADATA_FILENAME",
    "CheckpointMetadata",
    "CheckpointError",
    "TrainingProgress",
    "load_checkpoint",
    "read_checkpoint_metadata",
    "save_checkpoint",
]
