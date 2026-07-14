from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
    normalize_action,
    statistics_content_sha256,
    validate_statistics,
)
from prism.data.schema import DataSpec
from prism.models.batch import PolicyCurrentBatch
from prism.models.config import PrismArchitectureConfig
from prism.models.history_qformer import HistoryMemoryOutput
from prism.models.policy import PrismPolicy
from prism.models.policy import PolicyRuntimeOutput
from prism.models.task_state_planner import TaskStatePlannerRuntimeState
from prism.models.vlm import EncodedHistoryObservation
from prism.schema import CurrentPolicyInput
from prism.serve.protocol import HistoryObservationRequest, PolicyRequest
from prism.training.checkpoint import CheckpointMetadata, read_checkpoint_metadata


class PolicyBackend(Protocol):
    """Model-agnostic inference boundary used by the benchmark server."""

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def encode_history_observation(
        self,
        request: HistoryObservationRequest,
    ) -> EncodedHistoryObservation: ...

    def build_history_memory(
        self,
        observations: tuple[EncodedHistoryObservation, EncodedHistoryObservation],
    ) -> HistoryMemoryOutput: ...

    def empty_history_memory(self) -> HistoryMemoryOutput: ...

    def infer(
        self,
        request: PolicyRequest,
        memory: HistoryMemoryOutput,
    ) -> Mapping[str, Any] | Any: ...

    def infer_cycle(
        self,
        request: PolicyRequest,
        memory: HistoryMemoryOutput,
        *,
        planning_state: TaskStatePlannerRuntimeState | None,
    ) -> "PolicyBackendInference": ...


@dataclass(frozen=True)
class PolicyBackendInference:
    """One response plus the session-local planning state to commit on success."""

    response: Mapping[str, Any] | Any
    planning_state: TaskStatePlannerRuntimeState | None


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
        if not callable(getattr(collator, "collate_current_inference", None)):
            raise TypeError("loaded policy collator must implement collate_current_inference(inputs)")
        if not callable(getattr(loaded_policy, "predict_with_memory_and_plan", None)):
            raise TypeError(
                "loaded policy must implement "
                "predict_with_memory_and_plan(current_batch, memory, planning_state)"
            )
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
                "task_state_tokens": architecture.task_state_planner.num_state_tokens,
                "plan_tokens": architecture.task_state_planner.num_plan_tokens,
                "plan_horizon_actions": architecture.task_state_planner.plan_horizon_actions,
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

    def encode_history_observation(
        self,
        request: HistoryObservationRequest,
    ) -> EncodedHistoryObservation:
        """Immediately replace one transient two-camera image payload with visual tokens."""

        self._validate_history_request(request)
        images = tuple(request.images_by_view[name] for name in self._data_spec.view_names)
        with torch.inference_mode():
            return self._policy.query_memory_encoder.encode_history_observation(images)

    def build_history_memory(
        self,
        observations: tuple[EncodedHistoryObservation, EncodedHistoryObservation],
    ) -> HistoryMemoryOutput:
        """Compress the two visual-token slots into fixed-size memory tokens."""

        with torch.inference_mode():
            return self._policy.query_memory_encoder.build_history_memory(
                observations,
                self._policy.architecture.temporal.history_step_ages,
            )

    def empty_history_memory(self) -> HistoryMemoryOutput:
        """Create initial invalid memory without encoding placeholder images."""

        with torch.inference_mode():
            return self._policy.query_memory_encoder.empty_history_memory(batch_size=1)

    def infer(
        self,
        request: PolicyRequest,
        memory: HistoryMemoryOutput,
    ) -> Mapping[str, Any]:
        """Stateless compatibility entry point; serving uses ``infer_cycle``."""

        return self.infer_cycle(request, memory, planning_state=None).response

    def infer_cycle(
        self,
        request: PolicyRequest,
        memory: HistoryMemoryOutput,
        *,
        planning_state: TaskStatePlannerRuntimeState | None,
    ) -> PolicyBackendInference:
        """Normalize inputs, advance the planning state, and denormalize actions."""

        self._validate_request(request)
        normalized_state = self._normalizer.normalize_state(np.asarray(request.state, dtype=np.float32))
        if request.executed_actions is None and request.executed_action_valid_mask is None:
            raw_executed_actions = np.zeros(
                (
                    self._policy.architecture.task_state_planner.action_horizon,
                    self._data_spec.action_dim,
                ),
                dtype=np.float32,
            )
            executed_action_valid_mask = np.zeros(
                (self._policy.architecture.task_state_planner.action_horizon,),
                dtype=np.bool_,
            )
        elif request.executed_actions is None or request.executed_action_valid_mask is None:
            raise ValueError(
                "request executed_actions and executed_action_valid_mask must be provided together"
            )
        else:
            raw_executed_actions = np.asarray(request.executed_actions, dtype=np.float32)
            executed_action_valid_mask = np.asarray(
                request.executed_action_valid_mask,
                dtype=np.bool_,
            )
        normalized_executed_actions = normalize_action(
            raw_executed_actions,
            self._group_statistics,
            valid_mask=executed_action_valid_mask,
        )
        normalized_request = CurrentPolicyInput(
            benchmark=request.benchmark,
            prompt=request.prompt,
            images_by_view=request.images_by_view,
            state=normalized_state,
            action_dim=request.action_dim,
            robot_key=request.robot_key,
            executed_actions=normalized_executed_actions,
            executed_action_valid_mask=executed_action_valid_mask,
        )
        batch = self._collator.collate_current_inference((normalized_request,))
        batch = _move_current_batch_to_device(batch, self._policy_device)
        with torch.inference_mode():
            runtime_output = self._policy.predict_with_memory_and_plan(
                batch,
                memory,
                planning_state=planning_state,
            )
        if not isinstance(runtime_output, PolicyRuntimeOutput):
            raise TypeError(
                "loaded policy predict_with_memory_and_plan() must return PolicyRuntimeOutput"
            )
        normalized_actions = runtime_output.predicted_actions
        if not isinstance(normalized_actions, torch.Tensor):
            raise TypeError("loaded policy predict_with_memory() must return a torch.Tensor")
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
            response["debug"].update(
                {
                    "task_state": runtime_output.task_state.detach()
                    .to(dtype=torch.float32, device="cpu")
                    .numpy()[0],
                    "plan_tokens": runtime_output.plan_tokens.detach()
                    .to(dtype=torch.float32, device="cpu")
                    .numpy()[0],
                }
            )
        return PolicyBackendInference(
            response=response,
            planning_state=runtime_output.planning_state.detached(),
        )

    def _validate_request(self, request: PolicyRequest) -> None:
        if not isinstance(request, PolicyRequest):
            raise TypeError(f"request must be PolicyRequest, got {type(request).__name__}")
        self._validate_identity_and_views(request)
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

    def _validate_history_request(self, request: HistoryObservationRequest) -> None:
        if not isinstance(request, HistoryObservationRequest):
            raise TypeError(
                f"request must be HistoryObservationRequest, got {type(request).__name__}"
            )
        self._validate_identity_and_views(request)

    def _validate_identity_and_views(
        self,
        request: PolicyRequest | HistoryObservationRequest,
    ) -> None:
        if request.benchmark != self._data_spec.benchmark:
            raise ValueError(f"request benchmark must be {self._data_spec.benchmark!r}, got {request.benchmark!r}")
        if request.robot_key != self._data_spec.robot_key:
            raise ValueError(f"request robot_key must be {self._data_spec.robot_key!r}, got {request.robot_key!r}")
        expected_views = self._data_spec.view_names
        if tuple(request.images_by_view) != expected_views:
            raise ValueError(f"request ordered views must be {expected_views!r}, got {tuple(request.images_by_view)!r}")
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


def _move_current_batch_to_device(
    batch: PolicyCurrentBatch,
    device: torch.device,
) -> PolicyCurrentBatch:
    if not isinstance(batch, PolicyCurrentBatch):
        raise TypeError(f"inference collator must return PolicyCurrentBatch, got {type(batch).__name__}")
    return PolicyCurrentBatch(
        current_inputs={key: value.to(device=device, non_blocking=True) for key, value in batch.current_inputs.items()},
        state=batch.state.to(device=device, non_blocking=True),
        executed_actions=batch.executed_actions.to(device=device, non_blocking=True),
        executed_action_valid_mask=batch.executed_action_valid_mask.to(
            device=device,
            non_blocking=True,
        ),
    )


__all__ = ["CheckpointPolicyBackend", "PolicyBackend", "PolicyBackendInference"]
