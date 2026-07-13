from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from prism.models.action_head import DirectActionHead, decode_gripper_open
from prism.models.batch import PolicyBatch, PolicyBatchCollator, PolicyInferenceBatch
from prism.models.config import PrismArchitectureConfig
from prism.models.vlm import PreparedQueryMemoryBatch, Qwen35QueryMemoryEncoder


@dataclass(frozen=True)
class PolicyOutput:
    """Direct-action prediction, differentiable loss, and detached diagnostics."""

    predicted_actions: torch.Tensor
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


class PrismPolicy(nn.Module):
    """End-to-end query-memory encoder and direct masked-L1 policy."""

    def __init__(
        self,
        architecture: PrismArchitectureConfig,
        query_memory_encoder: Qwen35QueryMemoryEncoder,
        *,
        state_dim: int,
    ) -> None:
        super().__init__()
        architecture.validate_for_policy()
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        self.architecture = architecture
        self.query_memory_encoder = query_memory_encoder
        self.state_dim = state_dim
        self.action_head = DirectActionHead(architecture, state_dim=state_dim)

    def make_collator(self) -> PolicyBatchCollator:
        """Create the CPU-side collator paired with this model's encoder."""

        return PolicyBatchCollator(
            self.architecture,
            self.query_memory_encoder,
            state_dim=self.state_dim,
        )

    def forward(self, batch: PolicyBatch) -> PolicyOutput:
        if not isinstance(batch, PolicyBatch):
            raise TypeError(f"PrismPolicy.forward expects PolicyBatch, got {type(batch).__name__}")
        batch.validate_against(self.architecture, state_dim=self.state_dim)
        predicted_actions = self._predict_actions(batch)
        action_config = self.architecture.action_head
        loss, metrics = masked_action_l1(
            predicted_actions,
            batch.target_actions,
            batch.action_valid_mask,
            batch.action_dim_mask,
            gripper_index=action_config.gripper_index,
            gripper_threshold=action_config.gripper_threshold,
        )
        return PolicyOutput(
            predicted_actions=predicted_actions,
            loss=loss,
            metrics=metrics,
        )

    def predict(self, batch: PolicyInferenceBatch) -> torch.Tensor:
        """Predict normalized actions without creating targets or a fake loss."""

        if not isinstance(batch, PolicyInferenceBatch):
            raise TypeError(f"PrismPolicy.predict expects PolicyInferenceBatch, got {type(batch).__name__}")
        batch.validate_against(self.architecture, state_dim=self.state_dim)
        return self._predict_actions(batch)

    def _predict_actions(self, batch: PolicyInferenceBatch) -> torch.Tensor:
        """Shared encoder/action-head path used by training and inference."""

        encoder_output = self.query_memory_encoder.forward_prepared(
            PreparedQueryMemoryBatch(
                current_inputs=batch.current_inputs,
                history_inputs=batch.history_inputs,
                history_step_ages=batch.history_step_ages,
                history_valid_mask=batch.history_valid_mask,
            )
        )
        return self.action_head(encoder_output, batch.state)


def masked_action_l1(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_valid_mask: torch.Tensor,
    action_dim_mask: torch.Tensor | None,
    *,
    gripper_index: int,
    gripper_threshold: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the exact element-masked direct-action L1 contract."""

    if predicted_actions.ndim != 3 or not torch.is_floating_point(predicted_actions):
        raise ValueError("predicted_actions must be a floating tensor with shape [B, horizon, action_dim]")
    if target_actions.shape != predicted_actions.shape or not torch.is_floating_point(target_actions):
        raise ValueError("target_actions must be floating and match predicted_actions shape")
    if not torch.isfinite(predicted_actions).all() or not torch.isfinite(target_actions).all():
        raise ValueError("predicted_actions and target_actions must be finite")
    if action_valid_mask.dtype != torch.bool or action_valid_mask.shape != predicted_actions.shape[:2]:
        raise ValueError("action_valid_mask must be boolean with shape [B, horizon]")
    batch_size, _, action_dim = predicted_actions.shape
    if gripper_index < 0 or gripper_index >= action_dim:
        raise ValueError("gripper_index is outside the action dimension")
    if not 0.0 <= gripper_threshold <= 1.0:
        raise ValueError("gripper_threshold must be in [0, 1]")

    if action_dim_mask is None:
        action_dim_mask = torch.ones(
            batch_size,
            action_dim,
            dtype=torch.bool,
            device=predicted_actions.device,
        )
    elif action_dim_mask.dtype != torch.bool or action_dim_mask.shape != (batch_size, action_dim):
        raise ValueError("action_dim_mask must be boolean with shape [B, action_dim]")

    target_actions = target_actions.to(
        device=predicted_actions.device,
        dtype=predicted_actions.dtype,
    )
    action_valid_mask = action_valid_mask.to(device=predicted_actions.device)
    action_dim_mask = action_dim_mask.to(device=predicted_actions.device)
    element_mask = action_valid_mask.unsqueeze(-1) & action_dim_mask.unsqueeze(1)
    if not element_mask.any():
        raise ValueError("masked action L1 requires at least one valid element")

    absolute_error = torch.abs(predicted_actions - target_actions)
    loss = _masked_mean(absolute_error, element_mask)

    motion_dimension_mask = action_dim_mask.clone()
    motion_dimension_mask[:, gripper_index] = False
    motion_element_mask = action_valid_mask.unsqueeze(-1) & motion_dimension_mask.unsqueeze(1)
    gripper_element_mask = action_valid_mask & action_dim_mask[:, gripper_index].unsqueeze(1)
    predicted_open = decode_gripper_open(
        predicted_actions,
        gripper_index=gripper_index,
        threshold=gripper_threshold,
    )
    target_open = decode_gripper_open(
        target_actions,
        gripper_index=gripper_index,
        threshold=gripper_threshold,
    )

    gripper_error = absolute_error[..., gripper_index]
    metrics = {
        "total_l1": loss.detach(),
        "motion_l1": _masked_mean_or_zero(absolute_error, motion_element_mask).detach(),
        "gripper_l1": _masked_mean_or_zero(gripper_error, gripper_element_mask).detach(),
        "gripper_accuracy": _masked_mean_or_zero(
            (predicted_open == target_open).to(dtype=predicted_actions.dtype),
            gripper_element_mask,
        ).detach(),
        "predicted_open_ratio": _masked_mean_or_zero(
            predicted_open.to(dtype=predicted_actions.dtype),
            gripper_element_mask,
        ).detach(),
        "target_open_ratio": _masked_mean_or_zero(
            target_open.to(dtype=predicted_actions.dtype),
            gripper_element_mask,
        ).detach(),
        "gripper_transition_recall": _gripper_transition_recall(
            predicted_open,
            target_open,
            action_valid_mask,
            action_dim_mask[:, gripper_index],
            dtype=predicted_actions.dtype,
        ).detach(),
    }
    return loss, metrics


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum()


def _masked_mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return values.new_zeros(())
    return _masked_mean(values, mask)


def _gripper_transition_recall(
    predicted_open: torch.Tensor,
    target_open: torch.Tensor,
    action_valid_mask: torch.Tensor,
    gripper_dim_valid: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    adjacent_valid = action_valid_mask[:, :-1] & action_valid_mask[:, 1:] & gripper_dim_valid.unsqueeze(1)
    target_transitions = (target_open[:, :-1] != target_open[:, 1:]) & adjacent_valid
    if not target_transitions.any():
        return torch.zeros((), dtype=dtype, device=predicted_open.device)
    predicted_transitions = predicted_open[:, :-1] != predicted_open[:, 1:]
    true_positive = (predicted_transitions & target_transitions).sum()
    return true_positive.to(dtype=dtype) / target_transitions.sum().to(dtype=dtype)


__all__ = ["PolicyOutput", "PrismPolicy", "masked_action_l1"]
