from __future__ import annotations

from prism.serve.protocol import PolicyRequest

# --- migrated from src/prism/runtime/feature_extractor.py ---
from collections.abc import Mapping

import numpy as np
import torch
from torchvision import transforms

from prism.data import rgb_array_to_pil
from prism.config import IMAGE_SIZE


def decode_images_by_view(images_by_view: Mapping[str, np.ndarray], device: torch.device) -> list[torch.Tensor]:
    images = []
    for view_name, image in images_by_view.items():
        tensor = decode_image_array(image, device)
        expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{view_name} image_size must be {expected_shape}, got {tuple(tensor.shape)}")
        images.append(tensor)
    return images


def decode_image_array(image: np.ndarray, device: torch.device) -> torch.Tensor:
    pil = rgb_array_to_pil(image, IMAGE_SIZE)
    return transforms.ToTensor()(pil).to(device)


# --- migrated from src/prism/runtime/inference_engine.py ---
from contextlib import nullcontext

import torch

from prism.models.policy import PrismPolicy
from prism.utils import NormalizationStats


class PolicyInferenceEngine:
    def __init__(
        self,
        model: PrismPolicy,
        normalizer: NormalizationStats,
        *,
        state_dim: int,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.state_dim = int(state_dim)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def infer(self, request: PolicyRequest, runtime_state: RuntimePolicyState | None = None):
        device = self.device
        model_action_dim = int(
            getattr(self.model, "per_action_dim", self.model.config.get("per_action_dim", request.action_dim))
        )
        if int(request.action_dim) != model_action_dim:
            raise ValueError(
                f"request action_dim={request.action_dim} does not match model action_dim={model_action_dim}"
            )

        if runtime_state is not None and request.reset_memory:
            runtime_state.reset(self.model)

        images = decode_images_by_view(request.images_by_view, device)
        state = torch.as_tensor(request.state, dtype=torch.float32, device=device)
        norm_state = self.normalizer.normalize_state(
            pad_state_tensor(state, target_dim=self.state_dim),
            robot_key=request.robot_key,
        ).to(dtype=torch.float32)
        image_mask = torch.ones(len(images), dtype=torch.int32, device=device)
        action_mask = torch.ones(1, model_action_dim, dtype=torch.int32, device=device)

        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            embedding_output = self.model.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=request.prompt,
                return_hidden_states=self.model.use_bridge,
            )
            if hasattr(embedding_output, "fused_tokens"):
                hidden_states = embedding_output.hidden_states
                visual_tokens = getattr(embedding_output, "visual_tokens", None)
                fused_tokens = visual_tokens if visual_tokens is not None else embedding_output.fused_tokens
                planner_vl_summary = getattr(embedding_output, "planner_vl_summary", None)
            else:
                fused_tokens = embedding_output
                hidden_states = None
                visual_tokens = None
                planner_vl_summary = None

            executed_actions, executed_action_mask = (None, None)
            memory_context, memory_context_mask, short_memory_time_ids = (None, None, None)
            request_progress_inputs = self._executed_actions_from_request(
                request,
                device=device,
                dtype=fused_tokens.dtype,
                robot_key=request.robot_key,
            )
            if request_progress_inputs is not None:
                executed_actions, executed_action_mask = request_progress_inputs
            if runtime_state is not None:
                if request_progress_inputs is not None:
                    runtime_state.store_executed_action_inputs(executed_actions, executed_action_mask)
                else:
                    executed_actions, executed_action_mask = runtime_state.progress_inputs(
                        self.model,
                        device=device,
                        dtype=fused_tokens.dtype,
                    )
            if request.short_memory_images_by_offset is not None:
                memory_context, memory_context_mask, short_memory_time_ids = self._short_memory_from_request(
                    request,
                    device=device,
                    dtype=fused_tokens.dtype,
                )
            progress_state = None
            if runtime_state is not None:
                progress_state = runtime_state.progress_state_input(
                    self.model,
                    batch_size=1,
                    device=device,
                    dtype=fused_tokens.dtype,
                )

            action = self.model.predict_action(
                fused_tokens,
                norm_state,
                action_mask=action_mask,
                hidden_states=hidden_states,
                memory_context=memory_context,
                memory_context_mask=memory_context_mask,
                short_memory_time_ids=short_memory_time_ids,
                executed_actions=executed_actions,
                executed_action_mask=executed_action_mask,
                progress_state=progress_state,
                planner_vl_summary=planner_vl_summary,
            )
            if action.numel() % model_action_dim != 0:
                raise ValueError(
                    f"Model returned {action.numel()} action values, not divisible by per_action_dim={model_action_dim}"
                )
            normalized_action = action.reshape(1, -1, model_action_dim)
            if runtime_state is not None:
                runtime_state.store_progress_state(self.model)
            if runtime_state is not None and request_progress_inputs is None:
                runtime_state.store_executed_actions(self.model, normalized_action)
            denormalized_action = self.normalizer.denormalize_action(normalized_action[0], robot_key=request.robot_key)
            denormalized_action = denormalized_action.to(dtype=torch.float32)
            actions = denormalized_action.cpu().numpy().tolist()
            if not request.return_debug:
                return actions
            return {"actions": actions}

    def _executed_actions_from_request(
        self,
        request: PolicyRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
        robot_key: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if request.executed_actions is None:
            return None
        planner = self.model.progress_state_planner
        if planner is None:
            return None
        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        actions = torch.as_tensor(request.executed_actions, dtype=torch.float32, device=device)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"executed_actions must have shape [R, A] or [B, R, A], got {tuple(actions.shape)}")
        if int(actions.shape[0]) != 1:
            raise ValueError(f"runtime executed_actions supports batch size 1, got {tuple(actions.shape)}")
        if int(actions.shape[-1]) != action_dim:
            raise ValueError(f"executed_actions action dim {actions.shape[-1]} != planner action_dim {action_dim}")
        actions = self.normalizer.normalize_action(actions, robot_key=robot_key).to(device=device, dtype=dtype)

        if request.executed_action_mask is None:
            mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(request.executed_action_mask, dtype=torch.bool, device=device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.ndim != 2 or tuple(mask.shape) != tuple(actions.shape[:2]):
                raise ValueError(
                    "executed_action_mask must have shape [R] or [B, R] matching executed_actions, "
                    f"got {tuple(mask.shape)} for actions {tuple(actions.shape)}"
                )

        if int(actions.shape[1]) > stride:
            actions = actions[:, :stride, :]
            mask = mask[:, :stride]
        elif int(actions.shape[1]) < stride:
            pad_steps = stride - int(actions.shape[1])
            action_pad = torch.zeros(actions.shape[0], pad_steps, action_dim, device=device, dtype=dtype)
            mask_pad = torch.zeros(actions.shape[0], pad_steps, device=device, dtype=torch.bool)
            actions = torch.cat([actions, action_pad], dim=1)
            mask = torch.cat([mask, mask_pad], dim=1)
        return actions, mask

    def _short_memory_from_request(
        self,
        request: PolicyRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        visual_tokens_by_offset = {}
        for offset, images_by_view in (request.short_memory_images_by_offset or {}).items():
            images = decode_images_by_view(images_by_view, device)
            image_mask = torch.ones(len(images), dtype=torch.int32, device=device)
            embedding_output = self.model.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=request.prompt,
                return_hidden_states=True,
            )
            visual_tokens = getattr(embedding_output, "visual_tokens", None)
            if visual_tokens is None:
                visual_tokens = embedding_output
            visual_tokens_by_offset[int(offset)] = visual_tokens
        return build_short_memory_inputs_from_visual_tokens(
            self.model,
            visual_tokens_by_offset,
            device=device,
            dtype=dtype,
        )


def pad_state_tensor(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] > target_dim:
        raise ValueError(f"State dimension {state.shape[1]} exceeds expected {target_dim}")
    if state.shape[1] < target_dim:
        padding = torch.zeros((state.shape[0], target_dim - state.shape[1]), device=state.device)
        state = torch.cat([state, padding], dim=1)
    return state


# --- migrated from src/prism/runtime/memory_builder.py ---
from collections.abc import Mapping

import torch

from prism.models.policy import PrismPolicy
from prism.models.planner import ProgressState


class RuntimePolicyState:
    def __init__(self) -> None:
        self.executed_actions: torch.Tensor | None = None
        self.executed_action_mask: torch.Tensor | None = None
        self.progress_state: ProgressState | None = None

    def reset(self, model: PrismPolicy) -> None:
        _ = model
        self.executed_actions = None
        self.executed_action_mask = None
        self.progress_state = None

    def progress_inputs(
        self,
        model: PrismPolicy,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        planner = model.progress_state_planner
        if planner is None:
            return None, None

        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        if self.executed_actions is None:
            actions = torch.zeros(1, stride, action_dim, device=device, dtype=dtype)
            mask = torch.zeros(1, stride, device=device, dtype=torch.bool)
            return actions, mask

        return (
            self.executed_actions.to(device=device, dtype=dtype),
            self.executed_action_mask.to(device=device) if self.executed_action_mask is not None else None,
        )

    def progress_state_input(
        self,
        model: PrismPolicy,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ProgressState | None:
        planner = model.progress_state_planner
        if planner is None:
            return None
        if self.progress_state is None:
            return planner.initial_state(batch_size, device=device, dtype=dtype)
        return ProgressState(
            completed_events=self.progress_state.completed_events.to(device=device, dtype=dtype),
            current_stage=self.progress_state.current_stage.to(device=device, dtype=dtype),
        )

    def store_progress_state(self, model: PrismPolicy) -> None:
        output = getattr(model, "last_progress_planner_output", None)
        if output is None:
            return
        self.progress_state = ProgressState(
            completed_events=output.progress_state.completed_events.detach().cpu(),
            current_stage=output.progress_state.current_stage.detach().cpu(),
        )

    def store_executed_actions(self, model: PrismPolicy, normalized_action: torch.Tensor) -> None:
        planner = model.progress_state_planner
        if planner is None:
            return

        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        action = normalized_action[:, :stride, :action_dim].detach()
        if action.shape[1] != stride:
            pad = torch.zeros(
                action.shape[0],
                stride - action.shape[1],
                action_dim,
                device=action.device,
                dtype=action.dtype,
            )
            action = torch.cat([action, pad], dim=1)
        self.executed_actions = action.cpu()
        self.executed_action_mask = torch.ones(action.shape[:2], dtype=torch.bool)

    def store_executed_action_inputs(self, actions: torch.Tensor, mask: torch.Tensor | None) -> None:
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B, R, A], got {tuple(actions.shape)}")
        self.executed_actions = actions.detach().cpu()
        if mask is None:
            self.executed_action_mask = torch.ones(actions.shape[:2], dtype=torch.bool)
        else:
            if mask.ndim != 2 or tuple(mask.shape) != tuple(actions.shape[:2]):
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match actions {tuple(actions.shape[:2])}")
            self.executed_action_mask = mask.detach().cpu().bool()


def pack_runtime_visual_tokens(tokens: torch.Tensor, *, target_tokens: int) -> torch.Tensor:
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"tokens must have shape [1, N, D] or [N, D], got {tuple(tokens.shape)}")
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    token_count = int(tokens.shape[1])
    if token_count <= 0:
        raise ValueError("visual token sequence must be non-empty")
    if token_count == target_tokens:
        return tokens.contiguous()
    if token_count < target_tokens:
        output = torch.zeros(1, target_tokens, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
        output[:, :token_count, :] = tokens
        return output

    boundaries = torch.linspace(0, token_count, steps=target_tokens + 1, device=tokens.device).round().long()
    output = torch.zeros(1, target_tokens, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
    for index in range(target_tokens):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            end = min(token_count, start + 1)
        output[:, index, :] = tokens[:, start:end, :].mean(dim=1)
    return output


def build_short_memory_inputs_from_visual_tokens(
    model: PrismPolicy,
    visual_tokens_by_offset: Mapping[int, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    memory, mask, time_ids = empty_short_memory_inputs(model, device=device, dtype=dtype)
    if memory is None or mask is None or time_ids is None:
        return None, None, None

    offsets = short_memory_offsets(model)
    entry_tokens = int(model.config.get("memory_entry_tokens", 16))
    for entry_index, offset in enumerate(offsets):
        tokens = visual_tokens_by_offset.get(int(offset))
        if tokens is None:
            continue
        packed = pack_runtime_visual_tokens(tokens.to(device=device, dtype=dtype), target_tokens=entry_tokens)
        start = entry_index * entry_tokens
        end = start + entry_tokens
        memory[:, start:end, :] = packed
        mask[:, start:end] = True
    return memory, mask, time_ids


def empty_short_memory_inputs(
    model: PrismPolicy,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not bool(getattr(model, "use_direct_bridge", False)):
        return None, None, None
    capacity = int(model.config.get("memory_short_capacity", 2))
    entry_tokens = int(model.config.get("memory_entry_tokens", 16))
    hidden_dim = int(model.config.get("embed_dim", 896))
    if capacity <= 0 or entry_tokens <= 0:
        return None, None, None

    memory = torch.zeros(1, capacity * entry_tokens, hidden_dim, device=device, dtype=dtype)
    mask = torch.zeros(1, capacity * entry_tokens, device=device, dtype=torch.bool)
    time_ids = torch.arange(capacity, device=device, dtype=torch.long).repeat_interleave(entry_tokens)
    return memory, mask, time_ids.unsqueeze(0)


def short_memory_offsets(model: PrismPolicy) -> tuple[int, ...]:
    capacity = int(model.config.get("memory_short_capacity", 2))
    raw_offsets = model.config.get("memory_short_offsets")
    if raw_offsets is None:
        return tuple(range(capacity, 0, -1))
    offsets = tuple(int(offset) for offset in raw_offsets)
    if len(offsets) < capacity:
        offsets = offsets + tuple(range(capacity - len(offsets), 0, -1))
    return offsets[:capacity]
