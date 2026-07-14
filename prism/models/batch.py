from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from prism.data.schema import VLASample
from prism.models.config import PrismArchitectureConfig
from prism.models.vlm import Qwen35QueryMemoryEncoder
from prism.schema import CurrentPolicyInput, PolicyInput


@dataclass(frozen=True)
class PolicyCurrentBatch:
    """Prepared current-observation inputs paired with normalized robot state."""

    current_inputs: Mapping[str, torch.Tensor]
    state: torch.Tensor
    executed_actions: torch.Tensor
    executed_action_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensor_mapping(self.current_inputs, "current_inputs")
        missing_current = sorted(
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"} - set(self.current_inputs)
        )
        if missing_current:
            raise ValueError(f"current_inputs is missing required tensors: {missing_current}")
        if self.state.ndim != 2 or not torch.is_floating_point(self.state):
            raise ValueError("state must be a floating tensor with shape [B, state_dim]")
        if self.state.device.type == "cpu" and not torch.isfinite(self.state).all():
            raise ValueError("state must contain only finite values")
        if self.state.shape[0] <= 0:
            raise ValueError("PolicyCurrentBatch must contain at least one sample")
        if self.executed_actions.ndim != 3 or not torch.is_floating_point(self.executed_actions):
            raise ValueError("executed_actions must be a floating tensor with shape [B, horizon, action_dim]")
        if self.executed_actions.shape[0] != self.state.shape[0]:
            raise ValueError("executed_actions batch size must match state")
        if self.executed_actions.device.type == "cpu" and not torch.isfinite(self.executed_actions).all():
            raise ValueError("executed_actions must contain only finite values")
        if (
            self.executed_action_valid_mask.dtype != torch.bool
            or self.executed_action_valid_mask.shape != self.executed_actions.shape[:2]
        ):
            raise ValueError(
                "executed_action_valid_mask must be boolean with shape [B, horizon]"
            )
        invalid_actions = ~self.executed_action_valid_mask.unsqueeze(-1)
        if self.executed_actions.device.type == "cpu" and torch.any(
            self.executed_actions.masked_select(invalid_actions) != 0
        ):
            raise ValueError("executed_actions must be zero at invalid positions")
        attention_mask = self.current_inputs["attention_mask"]
        if attention_mask.ndim != 2 or attention_mask.shape[0] != self.state.shape[0]:
            raise ValueError("current attention_mask batch size must match state")

    @property
    def batch_size(self) -> int:
        return self.state.shape[0]

    def validate_against(self, architecture: PrismArchitectureConfig, *, state_dim: int) -> None:
        architecture.validate_for_policy()
        if self.state.shape[1] != state_dim:
            raise ValueError(f"state width must be {state_dim}, got {self.state.shape[1]}")
        expected_actions = (
            self.batch_size,
            architecture.task_state_planner.action_horizon,
            architecture.task_state_planner.action_dim,
        )
        if self.executed_actions.shape != expected_actions:
            raise ValueError(
                f"executed_actions must have shape {expected_actions}, got {tuple(self.executed_actions.shape)}"
            )


@dataclass(frozen=True)
class PolicyInferenceBatch:
    """Prepared query-memory inputs for target-free policy inference."""

    current_inputs: Mapping[str, torch.Tensor]
    history_inputs: Mapping[str, torch.Tensor]
    history_step_ages: torch.Tensor
    history_valid_mask: torch.Tensor
    state: torch.Tensor
    executed_actions: torch.Tensor
    executed_action_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensor_mapping(self.current_inputs, "current_inputs")
        _validate_tensor_mapping(self.history_inputs, "history_inputs")
        missing_current = sorted(
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"} - set(self.current_inputs)
        )
        if missing_current:
            raise ValueError(f"current_inputs is missing required tensors: {missing_current}")
        missing_history = sorted({"pixel_values", "image_grid_thw"} - set(self.history_inputs))
        if missing_history:
            raise ValueError(f"history_inputs is missing required tensors: {missing_history}")

        if self.state.ndim != 2 or not torch.is_floating_point(self.state):
            raise ValueError("state must be a floating tensor with shape [B, state_dim]")
        if self.state.device.type == "cpu" and not torch.isfinite(self.state).all():
            raise ValueError("state must contain only finite values")
        batch_size = self.state.shape[0]
        if batch_size <= 0:
            raise ValueError("PolicyInferenceBatch must contain at least one sample")

        if self.history_step_ages.ndim != 2 or self.history_step_ages.shape[0] != batch_size:
            raise ValueError("history_step_ages must have shape [B, history_frames]")
        if self.history_step_ages.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("history_step_ages must have an integer dtype")
        if self.history_valid_mask.dtype != torch.bool or self.history_valid_mask.shape != self.history_step_ages.shape:
            raise ValueError("history_valid_mask must be boolean and match history_step_ages")
        attention_mask = self.current_inputs["attention_mask"]
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
            raise ValueError("current attention_mask batch size must match state")
        _validate_executed_action_tensors(
            self.executed_actions,
            self.executed_action_valid_mask,
            batch_size=batch_size,
        )

    @property
    def batch_size(self) -> int:
        return self.state.shape[0]

    def validate_against(
        self,
        architecture: PrismArchitectureConfig,
        *,
        state_dim: int,
    ) -> None:
        """Validate model-dependent dimensions without introducing defaults."""

        architecture.validate_for_policy()
        if self.state.shape[1] != state_dim:
            raise ValueError(f"state width must be {state_dim}, got {self.state.shape[1]}")
        expected_history_shape = (
            self.batch_size,
            architecture.history.num_history_frames,
        )
        if self.history_step_ages.shape != expected_history_shape:
            raise ValueError(
                f"history tensors must have shape {expected_history_shape}, got {tuple(self.history_step_ages.shape)}"
            )
        expected_actions = (
            self.batch_size,
            architecture.task_state_planner.action_horizon,
            architecture.task_state_planner.action_dim,
        )
        if self.executed_actions.shape != expected_actions:
            raise ValueError(
                f"executed_actions must have shape {expected_actions}, got {tuple(self.executed_actions.shape)}"
            )


@dataclass(frozen=True)
class PolicyBatch(PolicyInferenceBatch):
    """Prepared query-memory inputs and normalized direct-action targets."""

    target_actions: torch.Tensor
    action_valid_mask: torch.Tensor
    action_dim_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.target_actions.ndim != 3 or not torch.is_floating_point(self.target_actions):
            raise ValueError("target_actions must be a floating tensor with shape [B, horizon, action_dim]")
        if self.target_actions.shape[0] != self.batch_size:
            raise ValueError("state and target_actions batch sizes must match")
        if self.target_actions.device.type == "cpu" and not torch.isfinite(self.target_actions).all():
            raise ValueError("target_actions must contain only finite values")
        if self.action_valid_mask.dtype != torch.bool or self.action_valid_mask.shape != self.target_actions.shape[:2]:
            raise ValueError("action_valid_mask must be boolean with shape [B, horizon]")
        if self.action_dim_mask is not None and (
            self.action_dim_mask.dtype != torch.bool
            or self.action_dim_mask.shape != (self.batch_size, self.target_actions.shape[-1])
        ):
            raise ValueError("action_dim_mask must be boolean with shape [B, action_dim]")

    def validate_against(
        self,
        architecture: PrismArchitectureConfig,
        *,
        state_dim: int,
    ) -> None:
        super().validate_against(architecture, state_dim=state_dim)
        expected_target_shape = (
            self.batch_size,
            architecture.temporal.action_horizon,
            architecture.action_head.action_dim,
        )
        if self.target_actions.shape != expected_target_shape:
            raise ValueError(
                f"target_actions must have shape {expected_target_shape}, got {tuple(self.target_actions.shape)}"
            )


class PolicyBatchCollator:
    """Perform model-owned CPU preprocessing before ``PrismPolicy.forward``."""

    def __init__(
        self,
        architecture: PrismArchitectureConfig,
        query_memory_encoder: Qwen35QueryMemoryEncoder,
        *,
        state_dim: int,
    ) -> None:
        architecture.validate_for_policy()
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        self.architecture = architecture
        self.query_memory_encoder = query_memory_encoder
        self.state_dim = state_dim

    def __call__(self, samples: Sequence[VLASample]) -> PolicyBatch:
        if not samples:
            raise ValueError("samples must contain at least one VLASample")
        for index, sample in enumerate(samples):
            if not isinstance(sample, VLASample):
                raise TypeError(f"samples[{index}] must be VLASample, got {type(sample).__name__}")
            sample.validate()
            self._validate_sample_dimensions(sample, index=index)

        inference = self.collate_inference([sample.policy_input for sample in samples])
        target_actions = torch.from_numpy(
            np.stack([np.asarray(sample.target_actions, dtype=np.float32) for sample in samples])
        )
        action_valid_mask = torch.from_numpy(np.stack([sample.action_valid_mask for sample in samples])).to(
            dtype=torch.bool
        )
        action_dim_mask = torch.ones(
            len(samples),
            self.architecture.action_head.action_dim,
            dtype=torch.bool,
        )
        return PolicyBatch(
            current_inputs=inference.current_inputs,
            history_inputs=inference.history_inputs,
            history_step_ages=inference.history_step_ages,
            history_valid_mask=inference.history_valid_mask,
            state=inference.state,
            executed_actions=inference.executed_actions,
            executed_action_valid_mask=inference.executed_action_valid_mask,
            target_actions=target_actions,
            action_valid_mask=action_valid_mask,
            action_dim_mask=action_dim_mask,
        )

    def collate_inference(
        self,
        inputs: Sequence[PolicyInput],
    ) -> PolicyInferenceBatch:
        """Prepare model inputs without manufacturing action targets or masks."""

        if not inputs:
            raise ValueError("inputs must contain at least one PolicyInput")
        for index, policy_input in enumerate(inputs):
            if not isinstance(policy_input, PolicyInput):
                raise TypeError(f"inputs[{index}] must be PolicyInput, got {type(policy_input).__name__}")
            self._validate_policy_input_dimensions(policy_input, index=index)

        prepared = self.query_memory_encoder.prepare_requests(inputs)
        state = torch.from_numpy(
            np.stack([np.asarray(policy_input.state, dtype=np.float32) for policy_input in inputs])
        )
        executed_actions, executed_action_valid_mask = self._collate_executed_actions(inputs)
        return PolicyInferenceBatch(
            current_inputs=prepared.current_inputs,
            history_inputs=prepared.history_inputs,
            history_step_ages=prepared.history_step_ages,
            history_valid_mask=prepared.history_valid_mask,
            state=state,
            executed_actions=executed_actions,
            executed_action_valid_mask=executed_action_valid_mask,
        )

    def collate_current_inference(
        self,
        inputs: Sequence[CurrentPolicyInput],
    ) -> PolicyCurrentBatch:
        """Prepare a runtime batch without materializing or processing history images."""

        if not inputs:
            raise ValueError("inputs must contain at least one CurrentPolicyInput")
        for index, policy_input in enumerate(inputs):
            if not isinstance(policy_input, CurrentPolicyInput):
                raise TypeError(
                    f"inputs[{index}] must be CurrentPolicyInput, got {type(policy_input).__name__}"
                )
            self._validate_current_input_dimensions(policy_input, index=index)

        current_inputs = self.query_memory_encoder.prepare_current_requests(inputs)
        state = torch.from_numpy(
            np.stack([np.asarray(policy_input.state, dtype=np.float32) for policy_input in inputs])
        )
        executed_actions, executed_action_valid_mask = self._collate_executed_actions(inputs)
        return PolicyCurrentBatch(
            current_inputs=current_inputs,
            state=state,
            executed_actions=executed_actions,
            executed_action_valid_mask=executed_action_valid_mask,
        )

    def _validate_sample_dimensions(self, sample: VLASample, *, index: int) -> None:
        self._validate_policy_input_dimensions(sample.policy_input, index=index)
        expected_action_shape = (
            self.architecture.temporal.action_horizon,
            self.architecture.action_head.action_dim,
        )
        if sample.target_actions.shape != expected_action_shape:
            raise ValueError(
                f"samples[{index}].target_actions must have shape "
                f"{expected_action_shape}, got {sample.target_actions.shape}"
            )

    def _validate_policy_input_dimensions(
        self,
        policy_input: PolicyInput,
        *,
        index: int,
    ) -> None:
        self._validate_current_input_dimensions(policy_input, index=index)

    def _validate_current_input_dimensions(
        self,
        policy_input: CurrentPolicyInput | PolicyInput,
        *,
        index: int,
    ) -> None:
        state = policy_input.state
        if (
            not isinstance(state, np.ndarray)
            or state.shape != (self.state_dim,)
            or not np.issubdtype(state.dtype, np.floating)
            or not np.isfinite(state).all()
        ):
            raise ValueError(f"inputs[{index}].state must be a finite floating array with shape ({self.state_dim},)")
        if policy_input.action_dim != self.architecture.action_head.action_dim:
            raise ValueError(f"inputs[{index}].action_dim must be {self.architecture.action_head.action_dim}")
        _normalized_executed_actions(
            policy_input,
            horizon=self.architecture.task_state_planner.action_horizon,
            action_dim=self.architecture.task_state_planner.action_dim,
            label=f"inputs[{index}]",
        )

    def _collate_executed_actions(
        self,
        inputs: Sequence[CurrentPolicyInput | PolicyInput],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [
            _normalized_executed_actions(
                policy_input,
                horizon=self.architecture.task_state_planner.action_horizon,
                action_dim=self.architecture.task_state_planner.action_dim,
                label=f"inputs[{index}]",
            )
            for index, policy_input in enumerate(inputs)
        ]
        return (
            torch.from_numpy(np.stack([row[0] for row in rows])).to(dtype=torch.float32),
            torch.from_numpy(np.stack([row[1] for row in rows])).to(dtype=torch.bool),
        )


def _validate_tensor_mapping(
    value: Mapping[str, torch.Tensor],
    name: str,
) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty tensor mapping")
    non_tensors = sorted(key for key, item in value.items() if not isinstance(item, torch.Tensor))
    if non_tensors:
        raise TypeError(f"{name} contains non-tensor values at keys: {non_tensors}")


def _normalized_executed_actions(
    policy_input: CurrentPolicyInput | PolicyInput,
    *,
    horizon: int,
    action_dim: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    actions_value = policy_input.executed_actions
    mask_value = policy_input.executed_action_valid_mask
    if actions_value is None and mask_value is None:
        return (
            np.zeros((horizon, action_dim), dtype=np.float32),
            np.zeros((horizon,), dtype=np.bool_),
        )
    if actions_value is None or mask_value is None:
        raise ValueError(
            f"{label}.executed_actions and executed_action_valid_mask must be provided together"
        )
    actions = np.asarray(actions_value)
    mask = np.asarray(mask_value)
    if actions.shape != (horizon, action_dim) or not np.issubdtype(actions.dtype, np.floating):
        raise ValueError(
            f"{label}.executed_actions must be floating with shape ({horizon}, {action_dim})"
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"{label}.executed_actions must contain only finite values")
    if mask.dtype != np.bool_ or mask.shape != (horizon,):
        raise ValueError(
            f"{label}.executed_action_valid_mask must be boolean with shape ({horizon},)"
        )
    if np.any(actions[~mask] != 0):
        raise ValueError(f"{label}.executed_actions must be zero at invalid positions")
    return np.ascontiguousarray(actions, dtype=np.float32), np.ascontiguousarray(mask)


def _validate_executed_action_tensors(
    actions: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    batch_size: int,
) -> None:
    if actions.ndim != 3 or actions.shape[0] != batch_size or not torch.is_floating_point(actions):
        raise ValueError("executed_actions must be floating with shape [B, horizon, action_dim]")
    if actions.device.type == "cpu" and not torch.isfinite(actions).all():
        raise ValueError("executed_actions must contain only finite values")
    if valid_mask.dtype != torch.bool or valid_mask.shape != actions.shape[:2]:
        raise ValueError("executed_action_valid_mask must be boolean with shape [B, horizon]")
    if actions.device.type == "cpu" and torch.any(
        actions.masked_select(~valid_mask.unsqueeze(-1)) != 0
    ):
        raise ValueError("executed_actions must be zero at invalid positions")


__all__ = ["PolicyBatch", "PolicyBatchCollator", "PolicyCurrentBatch", "PolicyInferenceBatch"]
