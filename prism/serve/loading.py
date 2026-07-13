"""Verified checkpoint-to-policy reconstruction for inference serving."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from prism.data.normalization import canonical_sha256
from prism.data.schema import DataSpec, data_spec_from_mapping
from prism.models.config import architecture_config_from_mapping
from prism.models.factory import build_prism_policy
from prism.models.policy import PrismPolicy
from prism.training.checkpoint import CheckpointMetadata, read_checkpoint_metadata


@dataclass(frozen=True)
class LoadedPolicyCheckpoint:
    """Policy and contracts reconstructed from one verified checkpoint directory."""

    policy: PrismPolicy
    data_spec: DataSpec
    statistics_group: str
    metadata: CheckpointMetadata
    checkpoint_path: Path


def load_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    local_files_only: bool | None = None,
) -> LoadedPolicyCheckpoint:
    """Rebuild the exact policy graph and strictly restore its saved weights."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    metadata = read_checkpoint_metadata(checkpoint)
    snapshot = metadata.resolved_train_snapshot
    model_snapshot = _mapping(snapshot.get("model"), "checkpoint snapshot model")
    architecture = architecture_config_from_mapping(
        _mapping(model_snapshot.get("architecture"), "checkpoint snapshot architecture"),
        label="checkpoint snapshot architecture",
    )
    if canonical_sha256(architecture) != metadata.architecture_sha256:
        raise ValueError("reconstructed policy architecture does not match checkpoint metadata")

    data_snapshot = _mapping(snapshot.get("data"), "checkpoint snapshot data")
    data_spec = data_spec_from_mapping(
        _mapping(data_snapshot.get("data_spec"), "checkpoint snapshot DataSpec"),
        label="checkpoint snapshot DataSpec",
    )
    if canonical_sha256(data_spec) != metadata.data_spec_sha256:
        raise ValueError("reconstructed DataSpec does not match checkpoint metadata")
    normalization = _mapping(
        data_snapshot.get("normalization"),
        "checkpoint snapshot normalization",
    )
    statistics_group = normalization.get("group")
    if not isinstance(statistics_group, str) or not statistics_group:
        raise ValueError("checkpoint normalization group must be non-empty text")

    if local_files_only is not None and type(local_files_only) is not bool:
        raise TypeError("local_files_only must be a boolean or null")
    policy = build_prism_policy(
        architecture,
        state_dim=data_spec.state_dim,
        local_files_only=local_files_only,
    )
    target_device = resolve_inference_device(device)
    policy.to(device=target_device)
    load_accelerate_model_weights(policy, checkpoint)
    policy.requires_grad_(False)
    policy.eval()
    if _parameter_device(policy) != target_device:
        raise RuntimeError(
            f"loaded policy device mismatch: expected {target_device}, got {_parameter_device(policy)}"
        )
    return LoadedPolicyCheckpoint(
        policy=policy,
        data_spec=data_spec,
        statistics_group=statistics_group,
        metadata=metadata,
        checkpoint_path=checkpoint,
    )


def load_accelerate_model_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Strictly load the model artifact emitted by ``Accelerator.save_state``."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
    try:
        from accelerate.utils import load_checkpoint_in_model
    except ImportError as exc:  # pragma: no cover - environment contract guard
        raise RuntimeError("checkpoint inference requires the Accelerate package") from exc
    load_checkpoint_in_model(
        model,
        str(Path(checkpoint).expanduser().resolve()),
        strict=True,
    )


def resolve_inference_device(value: str | torch.device | None) -> torch.device:
    """Resolve an explicit device or select CUDA when available."""

    if value is None or value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid inference device {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"inference device must be CPU or CUDA, got {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA inference was requested but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is outside {torch.cuda.device_count()} visible devices"
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _parameter_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("loaded policy must contain at least one parameter")
    return parameter.device


__all__ = [
    "LoadedPolicyCheckpoint",
    "load_accelerate_model_weights",
    "load_policy_checkpoint",
    "resolve_inference_device",
]
