from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch

from prism.data.normalization import (
    DataSpecNormalizer,
    canonical_json_bytes,
    canonical_sha256,
    denormalize_action,
    statistics_content_sha256,
    validate_statistics,
)
from prism.data.schema import DataSpec
from prism.models.batch import PolicyInferenceBatch
from prism.models.config import PrismArchitectureConfig
from prism.models.policy import PrismPolicy
from prism.serve.protocol import PolicyRequest
from prism.schema import PolicyInput
from prism.training.checkpoint import CheckpointMetadata, read_checkpoint_metadata


class PolicyBackend(Protocol):
    """Model-agnostic inference boundary used by the benchmark server."""

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def infer(self, request: PolicyRequest) -> Mapping[str, Any] | Any: ...


class CheckpointPolicyBackend:
    """Verified checkpoint policy with normalization and inference contracts."""

    def __init__(
        self,
        *,
        loaded_policy: PrismPolicy,
        data_spec: DataSpec,
        statistics: Mapping[str, Any],
        statistics_group: str,
        checkpoint_metadata: CheckpointMetadata,
        checkpoint_path: str | Path,
        weights_state: str = "injected_loaded_policy",
    ) -> None:
        if not isinstance(data_spec, DataSpec):
            raise TypeError(f"data_spec must be DataSpec, got {type(data_spec).__name__}")
        data_spec.validate()
        if not isinstance(statistics_group, str) or not statistics_group:
            raise ValueError("statistics_group must be a non-empty string")
        if not isinstance(checkpoint_metadata, CheckpointMetadata):
            raise TypeError("checkpoint_metadata must be hash-verified CheckpointMetadata")
        if not isinstance(weights_state, str) or not weights_state:
            raise ValueError("weights_state must be non-empty text")

        architecture = getattr(loaded_policy, "architecture", None)
        if not isinstance(architecture, PrismArchitectureConfig):
            raise TypeError("loaded_policy must expose a resolved PrismArchitectureConfig")
        architecture.validate_for_policy()
        if canonical_sha256(architecture) != checkpoint_metadata.architecture_sha256:
            raise ValueError("loaded policy architecture does not match checkpoint metadata")
        if getattr(loaded_policy, "state_dim", None) != data_spec.state_dim:
            raise ValueError(
                "loaded policy state dimension does not match DataSpec: "
                f"{getattr(loaded_policy, 'state_dim', None)} != {data_spec.state_dim}"
            )
        if architecture.action_head.action_dim != data_spec.action_dim:
            raise ValueError(
                "loaded policy action dimension does not match DataSpec: "
                f"{architecture.action_head.action_dim} != {data_spec.action_dim}"
            )

        normalizer = DataSpecNormalizer(
            data_spec=data_spec,
            statistics=dict(statistics),
            statistics_group=statistics_group,
        )
        if statistics_content_sha256(normalizer.statistics) != checkpoint_metadata.statistics_sha256:
            raise ValueError("checkpoint statistics hash does not match embedded statistics")
        collator = loaded_policy.make_collator()
        if not callable(getattr(collator, "collate_inference", None)):
            raise TypeError("loaded policy collator must implement collate_inference(inputs)")
        if not callable(getattr(loaded_policy, "predict", None)):
            raise TypeError("loaded policy must implement predict(inference_batch)")
        policy_device = _policy_parameter_device(loaded_policy)

        loaded_policy.eval()
        self._policy = loaded_policy
        self._collator = collator
        self._data_spec = data_spec
        self._normalizer = normalizer
        self._statistics_group = statistics_group
        self._group_statistics = normalizer.statistics["groups"][statistics_group]
        self._checkpoint_metadata = checkpoint_metadata
        self._policy_device = policy_device
        self._metadata = MappingProxyType(
            {
                "backend": "checkpoint-policy",
                "benchmark": data_spec.benchmark,
                "robot_key": data_spec.robot_key,
                "ordered_views": data_spec.view_names,
                "state_dim": data_spec.state_dim,
                "action_dim": data_spec.action_dim,
                "action_horizon": architecture.temporal.action_horizon,
                "statistics_group": statistics_group,
                "statistics_sha256": checkpoint_metadata.statistics_sha256,
                "data_spec_sha256": checkpoint_metadata.data_spec_sha256,
                "architecture_sha256": checkpoint_metadata.architecture_sha256,
                "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
                "device": str(policy_device),
                "weights_state": weights_state,
            }
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        local_files_only: bool | None = None,
    ) -> "CheckpointPolicyBackend":
        """Reconstruct the policy and strictly restore weights from one checkpoint."""

        from prism.serve.loading import load_policy_checkpoint

        loaded = load_policy_checkpoint(
            checkpoint_path,
            device=device,
            local_files_only=local_files_only,
        )
        return cls._from_verified_components(
            checkpoint_path=loaded.checkpoint_path,
            loaded_policy=loaded.policy,
            data_spec=loaded.data_spec,
            statistics_group=loaded.statistics_group,
            metadata=loaded.metadata,
            weights_state="verified_checkpoint_manifest",
        )

    @classmethod
    def from_loaded_policy(
        cls,
        checkpoint_path: str | Path,
        *,
        loaded_policy: PrismPolicy,
        data_spec: DataSpec,
        statistics_group: str,
    ) -> "CheckpointPolicyBackend":
        """Test/advanced path for an externally restored policy instance."""

        metadata = read_checkpoint_metadata(checkpoint_path)
        return cls._from_verified_components(
            checkpoint_path=checkpoint_path,
            loaded_policy=loaded_policy,
            data_spec=data_spec,
            statistics_group=statistics_group,
            metadata=metadata,
            weights_state="injected_loaded_policy",
        )

    @classmethod
    def _from_verified_components(
        cls,
        *,
        checkpoint_path: str | Path,
        loaded_policy: PrismPolicy,
        data_spec: DataSpec,
        statistics_group: str,
        metadata: CheckpointMetadata,
        weights_state: str,
    ) -> "CheckpointPolicyBackend":
        snapshot_data = _mapping(
            metadata.resolved_train_snapshot.get("data"),
            "checkpoint snapshot data",
        )
        embedded_spec = _mapping(
            snapshot_data.get("data_spec"),
            "checkpoint snapshot DataSpec",
        )
        expected_spec_hash = canonical_sha256(data_spec)
        if expected_spec_hash != metadata.data_spec_sha256:
            raise ValueError(
                "DataSpec hash does not match checkpoint metadata: "
                f"expected {expected_spec_hash}, stored {metadata.data_spec_sha256}"
            )
        if canonical_json_bytes(asdict(data_spec)) != canonical_json_bytes(embedded_spec):
            raise ValueError("DataSpec differs from the checkpoint-embedded schema")

        normalization = _mapping(
            snapshot_data.get("normalization"),
            "checkpoint snapshot normalization",
        )
        embedded_group = normalization.get("group")
        if embedded_group != statistics_group:
            raise ValueError(
                "statistics group does not match checkpoint metadata: "
                f"expected {statistics_group!r}, stored {embedded_group!r}"
            )
        statistics = metadata.normalization_statistics
        if statistics.get("content_sha256") != metadata.statistics_sha256:
            raise ValueError("checkpoint statistics content_sha256 does not match metadata hash")
        dataset_names = _checkpoint_dataset_names(snapshot_data.get("datasets"))
        validate_statistics(
            statistics,
            group=statistics_group,
            expected_schema_hash=expected_spec_hash,
            expected_robot_key=data_spec.robot_key,
            expected_datasets=dataset_names,
        )
        return cls(
            loaded_policy=loaded_policy,
            data_spec=data_spec,
            statistics=statistics,
            statistics_group=statistics_group,
            checkpoint_metadata=metadata,
            checkpoint_path=checkpoint_path,
            weights_state=weights_state,
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def infer(self, request: PolicyRequest) -> Mapping[str, Any]:
        """Run raw canonical state -> normalized policy -> raw canonical actions."""

        self._validate_request(request)
        normalized_state = self._normalizer.normalize_state(np.asarray(request.state, dtype=np.float32))
        normalized_request = replace(request, state=normalized_state)
        batch = self._collator.collate_inference((normalized_request,))
        batch = _move_inference_batch_to_device(batch, self._policy_device)
        with torch.inference_mode():
            normalized_actions = self._policy.predict(batch)
        if not isinstance(normalized_actions, torch.Tensor):
            raise TypeError("loaded policy predict() must return a torch.Tensor")
        expected_shape = (
            1,
            self._policy.architecture.temporal.action_horizon,
            self._data_spec.action_dim,
        )
        if normalized_actions.shape != expected_shape:
            raise ValueError(
                f"loaded policy returned shape {tuple(normalized_actions.shape)}, expected {expected_shape}"
            )
        if not torch.is_floating_point(normalized_actions) or not torch.isfinite(normalized_actions).all():
            raise ValueError("loaded policy actions must be finite floating values")
        normalized_numpy = normalized_actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = denormalize_action(
            normalized_numpy[0],
            self._group_statistics,
        )
        response: dict[str, Any] = {"actions": actions}
        if getattr(request, "return_debug", False):
            response["debug"] = {
                "statistics_group": self._statistics_group,
                "statistics_sha256": self._checkpoint_metadata.statistics_sha256,
            }
        return response

    def _validate_request(self, request: PolicyRequest) -> None:
        if not isinstance(request, PolicyInput):
            raise TypeError(f"request must be PolicyInput/PolicyRequest, got {type(request).__name__}")
        if request.benchmark != self._data_spec.benchmark:
            raise ValueError(f"request benchmark must be {self._data_spec.benchmark!r}, got {request.benchmark!r}")
        if request.robot_key != self._data_spec.robot_key:
            raise ValueError(f"request robot_key must be {self._data_spec.robot_key!r}, got {request.robot_key!r}")
        expected_views = self._data_spec.view_names
        if tuple(request.images_by_view) != expected_views:
            raise ValueError(f"request ordered views must be {expected_views!r}, got {tuple(request.images_by_view)!r}")
        if tuple(request.history_images_by_view) != expected_views:
            raise ValueError(
                f"request history ordered views must be {expected_views!r}, "
                f"got {tuple(request.history_images_by_view)!r}"
            )
        self._validate_view_arrays(request)
        state = np.asarray(request.state)
        if (
            state.shape != (self._data_spec.state_dim,)
            or not np.issubdtype(state.dtype, np.floating)
            or not np.isfinite(state).all()
        ):
            raise ValueError(
                f"request state must be finite floating canonical raw state with shape ({self._data_spec.state_dim},)"
            )
        if request.action_dim != self._data_spec.action_dim:
            raise ValueError(f"request action_dim must be {self._data_spec.action_dim}, got {request.action_dim}")
        expected_ages = tuple(self._policy.architecture.temporal.history_step_ages)
        ages = np.asarray(request.history_step_ages)
        if (
            ages.shape != (len(expected_ages),)
            or not np.issubdtype(ages.dtype, np.integer)
            or tuple(int(value) for value in ages) != expected_ages
        ):
            raise ValueError(f"request history_step_ages must equal {expected_ages!r}")
        history_mask = np.asarray(request.history_valid_mask)
        if history_mask.dtype != np.bool_ or history_mask.shape != ages.shape:
            raise ValueError("request history_valid_mask must be boolean and match history_step_ages")

    def _validate_view_arrays(self, request: PolicyRequest) -> None:
        history_count = self._policy.architecture.history.num_history_frames
        for view_name in self._data_spec.view_names:
            current = np.asarray(request.images_by_view[view_name])
            if (
                current.ndim != 3
                or current.shape[-1] != 3
                or current.shape[0] <= 0
                or current.shape[1] <= 0
                or current.dtype != np.uint8
            ):
                raise ValueError(f"request view {view_name!r} must be non-empty uint8 HxWx3")
            history = np.asarray(request.history_images_by_view[view_name])
            if history.shape != (history_count, *current.shape) or history.dtype != np.uint8:
                raise ValueError(
                    f"request history view {view_name!r} must be uint8 with shape {(history_count, *current.shape)}"
                )


def _checkpoint_dataset_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("checkpoint snapshot datasets must be a non-empty sequence")
    names: list[str] = []
    for index, item in enumerate(value):
        row = _mapping(item, f"checkpoint snapshot datasets[{index}]")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"checkpoint snapshot datasets[{index}].name must be non-empty")
        names.append(name)
    return tuple(names)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _policy_parameter_device(policy: PrismPolicy) -> torch.device:
    parameters = getattr(policy, "parameters", None)
    if not callable(parameters):
        raise TypeError("loaded policy must expose parameters()")
    first_parameter = next(iter(parameters()), None)
    if first_parameter is None:
        raise ValueError("loaded policy must contain at least one parameter to determine its device")
    return first_parameter.device


def _move_inference_batch_to_device(
    batch: PolicyInferenceBatch,
    device: torch.device,
) -> PolicyInferenceBatch:
    if not isinstance(batch, PolicyInferenceBatch):
        raise TypeError(f"inference collator must return PolicyInferenceBatch, got {type(batch).__name__}")
    return PolicyInferenceBatch(
        current_inputs={key: value.to(device=device, non_blocking=True) for key, value in batch.current_inputs.items()},
        history_inputs={key: value.to(device=device, non_blocking=True) for key, value in batch.history_inputs.items()},
        history_step_ages=batch.history_step_ages.to(
            device=device,
            non_blocking=True,
        ),
        history_valid_mask=batch.history_valid_mask.to(
            device=device,
            non_blocking=True,
        ),
        state=batch.state.to(device=device, non_blocking=True),
    )


__all__ = ["CheckpointPolicyBackend", "PolicyBackend"]
