"""Validated checkpoint metadata, snapshots, provenance, and RNG state.

This module owns the serialized PrismVLA contract layered on top of
Accelerate state: resolved configuration, schema/statistics hashes,
deterministic data cursors, repository/environment provenance, and explicit
per-rank RNG state.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from prism.data.normalization import canonical_json_bytes
from prism.data.normalization import canonical_sha256
from prism.data.normalization import statistics_content_sha256
from prism.data.normalization import validate_statistics
from prism.data.schema import data_spec_from_mapping
from prism.training.artifact_io import read_json
from prism.training.artifact_io import sha256_file
from prism.training.config import OPTIMIZATION_GROUP_NAMES
from prism.training.config import ResolvedTrainConfig
from prism.training.config import TRAIN_CONFIG_SNAPSHOT_FORMAT
from prism.training.config import build_checkpoint_snapshot


CHECKPOINT_FORMAT = "prism-checkpoint-v2"
GIT_PROVENANCE_FORMAT = "prism-git-provenance-v1"
RNG_FORMAT = "prism-rng-v1"
METADATA_FILENAME = "prism_metadata.json"
RNG_DIRECTORY = "prism_rng"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RESUME_SEMANTICS = {
    "checkpoint_boundary": "optimizer_sync_boundary",
    "dataloader_positioning": "caller_rebuilds_epoch_iterator_and_seeks_virtual_batch_cursor",
    "virtual_sample_cursor": "global_virtual_samples_consumed_in_current_epoch",
    "virtual_batch_cursor": "per_rank_batches_consumed_in_current_epoch",
    "mid_epoch_exact_resume": (
        "conditional_on_stateless_sample_mapping_deterministic_rank_sharding_"
        "deterministic_transforms_and_tested_runner_cursor_seek"
    ),
}


@dataclass(frozen=True)
class TrainingProgress:
    """Position immediately before the next training micro-batch is consumed.

    ``virtual_sample_cursor`` is the number of global virtual samples consumed
    in the current epoch (summed across ranks). ``virtual_batch_cursor`` is the
    synchronized number of local batches consumed by each rank. Checkpoints are
    intentionally restricted to optimizer synchronization boundaries, so the
    recorded accumulation micro-step is currently always zero.
    """

    completed_optimizer_steps: int
    gradient_accumulation_micro_step: int
    epoch: int
    virtual_sample_cursor: int
    virtual_batch_cursor: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

    @property
    def completed_steps(self) -> int:
        """Concise alias used by loggers and command-line status output."""

        return self.completed_optimizer_steps

    @property
    def micro_step(self) -> int:
        return self.gradient_accumulation_micro_step

    @property
    def batch_cursor(self) -> int:
        return self.virtual_batch_cursor


@dataclass(frozen=True)
class CheckpointMetadata:
    """Validated Prism metadata embedded alongside Accelerate state."""

    created_at_utc: str
    world_size: int
    progress: TrainingProgress
    resolved_train_snapshot: Mapping[str, Any]
    resolved_train_snapshot_sha256: str
    architecture_sha256: str
    data_spec_sha256: str
    statistics_sha256: str
    git: Mapping[str, Any]
    environment: Mapping[str, Any]
    rng_rank_files: tuple[str, ...]

    @property
    def normalization_statistics(self) -> Mapping[str, Any]:
        """Return the hash-verified, embedded statistics used by inference."""

        data = self.resolved_train_snapshot["data"]
        assert isinstance(data, Mapping)
        normalization = data["normalization"]
        assert isinstance(normalization, Mapping)
        statistics = normalization["statistics"]
        assert isinstance(statistics, Mapping)
        return statistics


@dataclass(frozen=True)
class _SnapshotHashes:
    config: str
    architecture: str
    data_spec: str
    statistics: str


def build_metadata_payload(
    *,
    world_size: int,
    progress: TrainingProgress,
    snapshot: Mapping[str, Any],
    snapshot_hashes: _SnapshotHashes,
    git_metadata: Mapping[str, Any],
    environment: Mapping[str, Any],
    rng_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the canonical metadata document written beside Accelerate state."""

    return {
        "format": CHECKPOINT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_size": world_size,
        "progress": asdict(progress),
        "resolved_train_snapshot": snapshot,
        "hashes": {
            "resolved_train_snapshot_sha256": snapshot_hashes.config,
            "architecture_sha256": snapshot_hashes.architecture,
            "data_spec_sha256": snapshot_hashes.data_spec,
            "statistics_sha256": snapshot_hashes.statistics,
        },
        "git": dict(git_metadata),
        "environment": dict(environment),
        "rng": {"format": RNG_FORMAT, "ranks": rng_rows},
        "accelerator_state": {"backend": "Accelerate.save_state/load_state"},
        "resume_semantics": _RESUME_SEMANTICS,
    }


def resolve_snapshot(
    config: ResolvedTrainConfig | Mapping[str, Any],
) -> tuple[dict[str, Any], _SnapshotHashes]:
    if isinstance(config, ResolvedTrainConfig):
        value: Any = build_checkpoint_snapshot(config)
    elif isinstance(config, Mapping):
        value = config
    else:
        raise TypeError(
            f"config must be ResolvedTrainConfig or a resolved snapshot mapping, got {type(config).__name__}"
        )
    snapshot = json.loads(canonical_json_bytes(value).decode("utf-8"))
    if not isinstance(snapshot, dict):
        raise TypeError("resolved train snapshot must be a mapping")
    hashes = _validate_snapshot(snapshot)
    return snapshot, hashes


def _validate_snapshot(snapshot: Mapping[str, Any]) -> _SnapshotHashes:
    _expect_keys(
        snapshot,
        {
            "format",
            "source_config",
            "experiment",
            "model",
            "data",
            "optimization",
            "trainer",
            "derived",
        },
        "resolved train snapshot",
    )
    if snapshot["format"] != TRAIN_CONFIG_SNAPSHOT_FORMAT:
        raise ValueError(
            f"unsupported resolved train snapshot format {snapshot['format']!r}; "
            f"expected {TRAIN_CONFIG_SNAPSHOT_FORMAT!r}"
        )
    _non_empty_text(snapshot["source_config"], "resolved train snapshot source_config")

    experiment = _strict_mapping(snapshot["experiment"], "snapshot experiment")
    _expect_keys(experiment, {"name", "output_dir", "seed"}, "snapshot experiment")
    _non_empty_text(experiment["name"], "snapshot experiment name")
    _non_empty_text(experiment["output_dir"], "snapshot experiment output_dir")
    _non_negative_int(experiment["seed"], "snapshot experiment seed")

    model = _strict_mapping(snapshot["model"], "snapshot model")
    _expect_keys(
        model,
        {"architecture_config", "architecture_sha256", "architecture"},
        "snapshot model",
    )
    _non_empty_text(model["architecture_config"], "snapshot architecture_config")
    architecture = _strict_mapping(model["architecture"], "snapshot architecture")
    action_head = _strict_mapping(architecture.get("action_head"), "snapshot action_head")
    if action_head.get("objective") != "direct_masked_l1":
        raise ValueError("checkpoint architecture objective must be 'direct_masked_l1'")
    architecture_hash = canonical_sha256(architecture)
    _stored_sha(model["architecture_sha256"], "snapshot architecture_sha256")
    if model["architecture_sha256"] != architecture_hash:
        raise ValueError(
            f"snapshot architecture hash mismatch: stored {model['architecture_sha256']}, computed {architecture_hash}"
        )

    data = _strict_mapping(snapshot["data"], "snapshot data")
    _expect_keys(
        data,
        {
            "spec",
            "data_spec_sha256",
            "data_spec",
            "root",
            "anchor_stride",
            "include_tail",
            "datasets",
            "normalization",
            "loader",
            "train_splits",
            "eval_splits",
        },
        "snapshot data",
    )
    _non_empty_text(data["spec"], "snapshot data spec reference")
    data_spec = _strict_mapping(data["data_spec"], "snapshot DataSpec")
    resolved_data_spec = data_spec_from_mapping(data_spec, label="snapshot DataSpec")
    data_spec_hash = canonical_sha256(data_spec)
    _stored_sha(data["data_spec_sha256"], "snapshot data_spec_sha256")
    if data["data_spec_sha256"] != data_spec_hash:
        raise ValueError(
            f"snapshot DataSpec hash mismatch: stored {data['data_spec_sha256']}, computed {data_spec_hash}"
        )
    robot_key = resolved_data_spec.robot_key
    anchor_stride = _positive_int(data["anchor_stride"], "snapshot anchor_stride")

    normalization = _strict_mapping(data["normalization"], "snapshot normalization")
    _expect_keys(
        normalization,
        {"group", "statistics_path", "content_sha256", "statistics"},
        "snapshot normalization",
    )
    group = _non_empty_text(normalization["group"], "snapshot normalization group")
    _non_empty_text(normalization["statistics_path"], "snapshot statistics_path")
    statistics = _strict_mapping(normalization["statistics"], "snapshot statistics")
    validate_statistics(
        statistics,
        group=group,
        expected_schema_hash=data_spec_hash,
        expected_robot_key=robot_key,
    )
    statistics_hash = statistics_content_sha256(statistics)
    _stored_sha(normalization["content_sha256"], "snapshot statistics content_sha256")
    if normalization["content_sha256"] != statistics_hash:
        raise ValueError(
            f"snapshot statistics hash mismatch: stored {normalization['content_sha256']}, computed {statistics_hash}"
        )

    loader = _strict_mapping(data["loader"], "snapshot loader")
    _expect_keys(
        loader,
        {
            "global_samples_per_epoch",
            "batch_size_per_rank",
            "num_workers",
            "preprocessing_workers",
            "pin_memory",
            "persistent_workers",
            "drop_last",
        },
        "snapshot loader",
    )
    _positive_int(loader["global_samples_per_epoch"], "snapshot global_samples_per_epoch")
    _positive_int(loader["batch_size_per_rank"], "snapshot batch_size_per_rank")

    optimization = _strict_mapping(snapshot["optimization"], "snapshot optimization")
    _expect_keys(
        optimization,
        {
            "optimizer",
            "beta1",
            "beta2",
            "epsilon",
            "no_decay_rule",
            *OPTIMIZATION_GROUP_NAMES,
        },
        "snapshot optimization",
    )
    if optimization["optimizer"] != "adamw":
        raise ValueError("snapshot optimizer must be 'adamw'")
    if optimization["no_decay_rule"] != "bias_and_low_dimensional":
        raise ValueError("snapshot no_decay_rule must be 'bias_and_low_dimensional'")
    for group_name in OPTIMIZATION_GROUP_NAMES:
        group = _strict_mapping(optimization[group_name], f"snapshot optimization {group_name}")
        _expect_keys(group, {"trainable", "learning_rate", "weight_decay"}, f"snapshot optimization {group_name}")
        if type(group["trainable"]) is not bool:
            raise TypeError(f"snapshot optimization {group_name} trainable must be a boolean")
        if group["trainable"]:
            if group["learning_rate"] is None or group["weight_decay"] is None:
                raise ValueError(f"snapshot trainable optimization group {group_name} has unresolved values")
        elif group["learning_rate"] is not None or group["weight_decay"] is not None:
            raise ValueError(f"snapshot frozen optimization group {group_name} must use null optimizer values")

    trainer = _strict_mapping(snapshot["trainer"], "snapshot trainer")
    _expect_keys(
        trainer,
        {
            "max_steps",
            "gradient_accumulation_steps",
            "mixed_precision",
            "scheduler",
            "warmup_steps",
            "max_grad_norm",
            "log_interval",
            "save_interval",
        },
        "snapshot trainer",
    )
    _positive_int(trainer["max_steps"], "snapshot max_steps")
    _positive_int(
        trainer["gradient_accumulation_steps"],
        "snapshot gradient_accumulation_steps",
    )

    derived = _strict_mapping(snapshot["derived"], "snapshot derived")
    _expect_keys(derived, {"temporal_contract", "source"}, "snapshot derived")
    if derived["source"] != "model.architecture.temporal":
        raise ValueError("snapshot derived temporal contract has an unsupported source")
    temporal_contract = _strict_mapping(
        derived["temporal_contract"],
        "snapshot temporal contract",
    )
    replan_stride = _positive_int(
        temporal_contract.get("replan_stride"),
        "snapshot temporal replan_stride",
    )
    if anchor_stride != replan_stride:
        raise ValueError(
            "snapshot anchor_stride must equal temporal replan_stride: "
            f"{anchor_stride} != {replan_stride}"
        )

    return _SnapshotHashes(
        config=canonical_sha256(snapshot),
        architecture=architecture_hash,
        data_spec=data_spec_hash,
        statistics=statistics_hash,
    )


def validate_progress(
    progress: TrainingProgress,
    snapshot: Mapping[str, Any],
    *,
    world_size: int,
) -> None:
    trainer = _strict_mapping(snapshot["trainer"], "snapshot trainer")
    data = _strict_mapping(snapshot["data"], "snapshot data")
    loader = _strict_mapping(data["loader"], "snapshot loader")
    max_steps = _positive_int(trainer["max_steps"], "snapshot max_steps")
    accumulation = _positive_int(
        trainer["gradient_accumulation_steps"],
        "snapshot gradient_accumulation_steps",
    )
    if progress.completed_optimizer_steps > max_steps:
        raise ValueError(
            f"completed_optimizer_steps exceeds trainer.max_steps: {progress.completed_optimizer_steps} > {max_steps}"
        )
    if progress.gradient_accumulation_micro_step >= accumulation:
        raise ValueError(
            f"gradient_accumulation_micro_step must be smaller than gradient_accumulation_steps={accumulation}"
        )
    if progress.gradient_accumulation_micro_step != 0:
        raise ValueError(
            "Prism checkpoints may only be saved/restored at an optimizer synchronization "
            "boundary (gradient_accumulation_micro_step must be zero); pending gradients "
            "are not part of Accelerator.save_state"
        )

    samples_per_epoch = _positive_int(
        loader["global_samples_per_epoch"],
        "snapshot global_samples_per_epoch",
    )
    batch_size = _positive_int(loader["batch_size_per_rank"], "snapshot batch_size_per_rank")
    if samples_per_epoch % world_size:
        raise ValueError(
            f"global_samples_per_epoch={samples_per_epoch} must be divisible by world_size={world_size} "
            "for duplicate-free deterministic rank sharding"
        )
    local_samples = samples_per_epoch // world_size
    drop_last = loader["drop_last"]
    if type(drop_last) is not bool:
        raise TypeError("snapshot loader drop_last must be a boolean")
    if drop_last:
        max_batches = local_samples // batch_size
        expected_samples = progress.virtual_batch_cursor * batch_size * world_size
    else:
        max_batches = math.ceil(local_samples / batch_size)
        expected_samples = min(
            progress.virtual_batch_cursor * batch_size * world_size,
            samples_per_epoch,
        )
    if progress.virtual_batch_cursor > max_batches:
        raise ValueError(
            f"virtual_batch_cursor={progress.virtual_batch_cursor} exceeds {max_batches} "
            "batches in the configured virtual epoch"
        )
    if progress.virtual_sample_cursor != expected_samples:
        raise ValueError(
            "virtual sample/batch cursor mismatch: expected "
            f"{expected_samples} global samples after {progress.virtual_batch_cursor} "
            f"local batches, got {progress.virtual_sample_cursor}"
        )


def parse_metadata(payload: Any, *, checkpoint: Path) -> CheckpointMetadata:
    root = _strict_mapping(payload, "checkpoint metadata")
    _expect_keys(
        root,
        {
            "format",
            "created_at_utc",
            "world_size",
            "progress",
            "resolved_train_snapshot",
            "hashes",
            "git",
            "environment",
            "rng",
            "accelerator_state",
            "resume_semantics",
        },
        "checkpoint metadata",
    )
    if root["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {root['format']!r}; expected {CHECKPOINT_FORMAT!r}")
    created = _non_empty_text(root["created_at_utc"], "checkpoint created_at_utc")
    try:
        parsed_created = datetime.fromisoformat(created)
    except ValueError as exc:
        raise ValueError("checkpoint created_at_utc is not ISO-8601") from exc
    if parsed_created.tzinfo is None:
        raise ValueError("checkpoint created_at_utc must include a timezone")
    world_size = _positive_int(root["world_size"], "checkpoint world_size")

    progress_payload = _strict_mapping(root["progress"], "checkpoint progress")
    _expect_keys(
        progress_payload,
        {
            "completed_optimizer_steps",
            "gradient_accumulation_micro_step",
            "epoch",
            "virtual_sample_cursor",
            "virtual_batch_cursor",
        },
        "checkpoint progress",
    )
    progress = TrainingProgress(**progress_payload)

    snapshot = _strict_mapping(root["resolved_train_snapshot"], "resolved train snapshot")
    computed_hashes = _validate_snapshot(snapshot)
    hashes = _strict_mapping(root["hashes"], "checkpoint hashes")
    _expect_keys(
        hashes,
        {
            "resolved_train_snapshot_sha256",
            "architecture_sha256",
            "data_spec_sha256",
            "statistics_sha256",
        },
        "checkpoint hashes",
    )
    expected_hashes = {
        "resolved_train_snapshot_sha256": computed_hashes.config,
        "architecture_sha256": computed_hashes.architecture,
        "data_spec_sha256": computed_hashes.data_spec,
        "statistics_sha256": computed_hashes.statistics,
    }
    for key, expected in expected_hashes.items():
        _stored_sha(hashes[key], f"checkpoint {key}")
        if hashes[key] != expected:
            raise ValueError(f"checkpoint {key} mismatch: stored {hashes[key]}, computed {expected}")
    validate_progress(progress, snapshot, world_size=world_size)

    git = _strict_mapping(root["git"], "checkpoint git metadata")
    _expect_keys(
        git,
        {
            "format",
            "commit",
            "dirty",
            "tracked_diff",
            "tracked_diff_sha256",
            "untracked_files",
        },
        "checkpoint git metadata",
    )
    if git["format"] != GIT_PROVENANCE_FORMAT:
        raise ValueError(f"unsupported checkpoint git provenance format {git['format']!r}")
    if not isinstance(git["commit"], str) or _GIT_COMMIT_RE.fullmatch(git["commit"]) is None:
        raise ValueError("checkpoint git commit must be a lowercase Git object id")
    if type(git["dirty"]) is not bool:
        raise TypeError("checkpoint git dirty must be a boolean")
    tracked_diff = git["tracked_diff"]
    if not isinstance(tracked_diff, str):
        raise TypeError("checkpoint tracked Git diff must be text")
    _stored_sha(git["tracked_diff_sha256"], "checkpoint tracked Git diff SHA256")
    computed_diff_sha = hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest()
    if git["tracked_diff_sha256"] != computed_diff_sha:
        raise ValueError(
            "checkpoint tracked Git diff hash mismatch: "
            f"stored {git['tracked_diff_sha256']}, computed {computed_diff_sha}"
        )
    untracked_files = git["untracked_files"]
    if not isinstance(untracked_files, list):
        raise TypeError("checkpoint untracked Git inventory must be a list")
    untracked_paths: list[str] = []
    for index, value in enumerate(untracked_files):
        row = _strict_mapping(value, f"checkpoint untracked Git row {index}")
        _expect_keys(
            row,
            {"path", "kind", "size_bytes", "sha256"},
            f"checkpoint untracked Git row {index}",
        )
        untracked_paths.append(
            _safe_relative_path(row["path"], f"checkpoint untracked Git row {index} path")
        )
        if row["kind"] not in {"file", "symlink"}:
            raise ValueError(f"checkpoint untracked Git row {index} has unsupported kind {row['kind']!r}")
        _non_negative_int(row["size_bytes"], f"checkpoint untracked Git row {index} size")
        _stored_sha(row["sha256"], f"checkpoint untracked Git row {index} SHA256")
    if untracked_paths != sorted(set(untracked_paths)):
        raise ValueError("checkpoint untracked Git inventory must have unique paths in sorted order")
    if git["dirty"] != bool(tracked_diff or untracked_files):
        raise ValueError("checkpoint git dirty flag disagrees with its tracked diff and untracked inventory")

    environment = _strict_mapping(root["environment"], "checkpoint environment")
    _expect_keys(
        environment,
        {
            "python",
            "platform",
            "prismvla",
            "accelerate",
            "torch",
            "torch_cuda",
            "cudnn",
            "numpy",
            "transformers",
        },
        "checkpoint environment",
    )
    for key, value in environment.items():
        if key in {"torch_cuda", "cudnn"} and value is None:
            continue
        _non_empty_text(value, f"checkpoint environment {key}")

    rng = _strict_mapping(root["rng"], "checkpoint RNG metadata")
    _expect_keys(rng, {"format", "ranks"}, "checkpoint RNG metadata")
    if rng["format"] != RNG_FORMAT:
        raise ValueError(f"unsupported checkpoint RNG format {rng['format']!r}")
    rows = rng["ranks"]
    if not isinstance(rows, list) or len(rows) != world_size:
        raise ValueError(f"checkpoint RNG metadata must contain exactly {world_size} rank rows")
    rng_paths: list[str] = []
    for expected_rank, value in enumerate(rows):
        row = _strict_mapping(value, f"checkpoint RNG row {expected_rank}")
        _expect_keys(row, {"rank", "path", "sha256"}, f"checkpoint RNG row {expected_rank}")
        if row["rank"] != expected_rank:
            raise ValueError(
                f"checkpoint RNG rows must be ordered by rank; expected {expected_rank}, got {row['rank']}"
            )
        relative = _safe_relative_path(row["path"], f"checkpoint RNG rank {expected_rank} path")
        _stored_sha(row["sha256"], f"checkpoint RNG rank {expected_rank} SHA256")
        rng_file = checkpoint / relative
        actual = sha256_file(rng_file)
        if actual != row["sha256"]:
            raise ValueError(
                f"checkpoint RNG rank {expected_rank} hash mismatch: stored {row['sha256']}, computed {actual}"
            )
        rng_payload = read_json(rng_file, label=f"rank {expected_rank} RNG state")
        validate_rng_payload(rng_payload, expected_rank=expected_rank)
        rng_paths.append(relative)

    accelerator_state = _strict_mapping(root["accelerator_state"], "accelerator state metadata")
    _expect_keys(accelerator_state, {"backend"}, "accelerator state metadata")
    if accelerator_state["backend"] != "Accelerate.save_state/load_state":
        raise ValueError("checkpoint accelerator state backend is unsupported")
    if root["resume_semantics"] != _RESUME_SEMANTICS:
        raise ValueError("checkpoint resume semantics are missing or unsupported")

    return CheckpointMetadata(
        created_at_utc=created,
        world_size=world_size,
        progress=progress,
        resolved_train_snapshot=_freeze_json(snapshot),
        resolved_train_snapshot_sha256=computed_hashes.config,
        architecture_sha256=computed_hashes.architecture,
        data_spec_sha256=computed_hashes.data_spec,
        statistics_sha256=computed_hashes.statistics,
        git=_freeze_json(git),
        environment=_freeze_json(environment),
        rng_rank_files=tuple(rng_paths),
    )


def capture_rng_state(*, rank: int) -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_states: list[str] = []
    if torch.cuda.is_available():
        cuda_states = [_encode_torch_rng(state) for state in torch.cuda.get_rng_state_all()]
    payload = {
        "format": RNG_FORMAT,
        "rank": rank,
        "python": {
            "version": python_state[0],
            "state": list(python_state[1]),
            "gauss_next": python_state[2],
        },
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": {
            "cpu": _encode_torch_rng(torch.get_rng_state()),
            "cuda_device_count": len(cuda_states),
            "cuda": cuda_states,
        },
    }
    validate_rng_payload(payload, expected_rank=rank)
    return payload


def validate_rng_payload(payload: Any, *, expected_rank: int) -> None:
    root = _strict_mapping(payload, f"rank {expected_rank} RNG state")
    _expect_keys(root, {"format", "rank", "python", "numpy", "torch"}, "RNG state")
    if root["format"] != RNG_FORMAT:
        raise ValueError(f"unsupported RNG state format {root['format']!r}")
    if root["rank"] != expected_rank:
        raise ValueError(f"RNG state rank mismatch: expected {expected_rank}, got {root['rank']}")

    python_state = _strict_mapping(root["python"], "Python RNG state")
    _expect_keys(python_state, {"version", "state", "gauss_next"}, "Python RNG state")
    version = _non_negative_int(python_state["version"], "Python RNG version")
    state_values = python_state["state"]
    if not isinstance(state_values, list) or not state_values:
        raise ValueError("Python RNG state must be a non-empty integer list")
    if any(type(value) is not int for value in state_values):
        raise TypeError("Python RNG state must contain only integers")
    gauss_next = python_state["gauss_next"]
    if gauss_next is not None and (
        isinstance(gauss_next, bool) or not isinstance(gauss_next, (int, float)) or not math.isfinite(float(gauss_next))
    ):
        raise ValueError("Python RNG gauss_next must be a finite number or null")
    try:
        random.Random().setstate((version, tuple(state_values), gauss_next))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Python RNG state") from exc

    numpy_state = _strict_mapping(root["numpy"], "NumPy RNG state")
    _expect_keys(
        numpy_state,
        {"bit_generator", "state", "position", "has_gauss", "cached_gaussian"},
        "NumPy RNG state",
    )
    bit_generator = _non_empty_text(numpy_state["bit_generator"], "NumPy bit generator")
    values = numpy_state["state"]
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise ValueError("NumPy RNG state must be a non-empty integer list")
    array = np.asarray(values, dtype=np.uint32)
    position = _non_negative_int(numpy_state["position"], "NumPy RNG position")
    has_gauss = _non_negative_int(numpy_state["has_gauss"], "NumPy RNG has_gauss")
    if has_gauss not in {0, 1}:
        raise ValueError("NumPy RNG has_gauss must be zero or one")
    cached = numpy_state["cached_gaussian"]
    if isinstance(cached, bool) or not isinstance(cached, (int, float)) or not math.isfinite(float(cached)):
        raise ValueError("NumPy cached_gaussian must be finite")
    try:
        np.random.RandomState().set_state((bit_generator, array, position, has_gauss, float(cached)))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid NumPy RNG state") from exc

    torch_state = _strict_mapping(root["torch"], "Torch RNG state")
    _expect_keys(torch_state, {"cpu", "cuda_device_count", "cuda"}, "Torch RNG state")
    cpu = _decode_torch_rng(torch_state["cpu"], "Torch CPU RNG state")
    try:
        torch.Generator(device="cpu").set_state(cpu)
    except RuntimeError as exc:
        raise ValueError("invalid Torch CPU RNG state") from exc
    cuda_count = _non_negative_int(torch_state["cuda_device_count"], "Torch CUDA device count")
    cuda = torch_state["cuda"]
    if not isinstance(cuda, list) or len(cuda) != cuda_count:
        raise ValueError("Torch CUDA RNG state count does not match cuda_device_count")
    for index, encoded in enumerate(cuda):
        _decode_torch_rng(encoded, f"Torch CUDA RNG state {index}")


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    python_state = _strict_mapping(payload["python"], "Python RNG state")
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["state"]),
            python_state["gauss_next"],
        )
    )
    numpy_state = _strict_mapping(payload["numpy"], "NumPy RNG state")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch_state = _strict_mapping(payload["torch"], "Torch RNG state")
    torch.set_rng_state(_decode_torch_rng(torch_state["cpu"], "Torch CPU RNG state"))
    stored_cuda_count = int(torch_state["cuda_device_count"])
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if stored_cuda_count != current_cuda_count:
        raise ValueError(
            f"Torch CUDA RNG device-count mismatch: stored {stored_cuda_count}, current {current_cuda_count}"
        )
    if stored_cuda_count:
        torch.cuda.set_rng_state_all(
            [
                _decode_torch_rng(value, f"Torch CUDA RNG state {index}")
                for index, value in enumerate(torch_state["cuda"])
            ]
        )


def _encode_torch_rng(state: torch.Tensor) -> str:
    array = state.detach().cpu().to(dtype=torch.uint8).contiguous().numpy()
    return base64.b64encode(array.tobytes()).decode("ascii")


def _decode_torch_rng(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty base64 text")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid base64") from exc
    if not decoded:
        raise ValueError(f"{label} decodes to an empty state")
    return torch.from_numpy(np.frombuffer(decoded, dtype=np.uint8).copy())


def collect_git_metadata(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    commit_result = _run_git(root, "rev-parse", "--verify", "HEAD")
    commit = commit_result.stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError(f"git returned an invalid commit id for {root}: {commit!r}")
    tracked_diff = _run_git(
        root,
        "-c",
        "core.quotepath=true",
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
    ).stdout
    untracked_output = _run_git_bytes(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout
    untracked_paths = [os.fsdecode(value) for value in untracked_output.split(b"\0") if value]
    untracked_files = [_untracked_file_metadata(root, relative) for relative in sorted(untracked_paths)]
    return {
        "format": GIT_PROVENANCE_FORMAT,
        "commit": commit,
        "dirty": bool(tracked_diff or untracked_files),
        "tracked_diff": tracked_diff,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest(),
        "untracked_files": untracked_files,
    }


def _untracked_file_metadata(repository_root: Path, relative: str) -> dict[str, Any]:
    normalized = _safe_relative_path(relative, "untracked Git path")
    path = repository_root.joinpath(*PurePosixPath(normalized).parts)
    file_stat = path.lstat()
    if stat.S_ISREG(file_stat.st_mode):
        kind = "file"
        size_bytes = file_stat.st_size
        content_sha256 = sha256_file(path)
    elif stat.S_ISLNK(file_stat.st_mode):
        kind = "symlink"
        target = os.fsencode(os.readlink(path))
        size_bytes = len(target)
        content_sha256 = hashlib.sha256(target).hexdigest()
    else:
        raise ValueError(f"untracked Git path has unsupported filesystem type: {normalized}")
    return {
        "path": normalized,
        "kind": kind,
        "size_bytes": size_bytes,
        "sha256": content_sha256,
    }


def _run_git(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        raise RuntimeError(f"failed to collect required git metadata from {repository_root}: {detail}") from exc


def _run_git_bytes(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = os.fsdecode(exc.stderr).strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise RuntimeError(f"failed to collect required git metadata from {repository_root}: {detail}") from exc


def collect_environment_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prismvla": _distribution_version("prismvla"),
        "accelerate": _distribution_version("accelerate"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": None if torch.backends.cudnn.version() is None else str(torch.backends.cudnn.version()),
        "numpy": np.__version__,
        "transformers": _distribution_version("transformers"),
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"required environment distribution {name!r} is not installed; checkpoint provenance would be incomplete"
        ) from exc


def _strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return dict(value)


def _expect_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise ValueError(f"{label} keys mismatch: missing={missing}, unknown={unknown}")


def _stored_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA256")
    return value


def _non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    text = _non_empty_text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative POSIX path, got {text!r}")
    return text


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


__all__ = [
    "CHECKPOINT_FORMAT",
    "GIT_PROVENANCE_FORMAT",
    "METADATA_FILENAME",
    "RNG_DIRECTORY",
    "CheckpointMetadata",
    "TrainingProgress",
    "build_metadata_payload",
    "capture_rng_state",
    "collect_environment_versions",
    "collect_git_metadata",
    "parse_metadata",
    "resolve_snapshot",
    "restore_rng_state",
    "validate_progress",
    "validate_rng_payload",
]
