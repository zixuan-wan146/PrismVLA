from __future__ import annotations

# --- migrated from src/prism/model/prism_policy.py ---
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union

from PIL import Image
import torch
import torch.nn as nn

from prism.config import resolve_experiment_config
from prism.models.action_head import BridgeAdapter, BridgeAdapterConfig, BridgeAdapterOutput, FlowmatchingActionHead
from prism.models.planner import ProgressPlannerOutput, ProgressState, ProgressStateConfig, ProgressStatePlanner
from prism.models.vlm import InternVL3Embedder, InternVL3EmbeddingOutput


class PrismPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        config = resolve_experiment_config(config)
        self.config = config
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)

        self.load_vlm = bool(config.get("load_vlm", True))
        self.embedder = None
        if self.load_vlm:
            vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
            self.embedder = InternVL3Embedder(
                model_name=vlm_name,
                device=self._device,
                allow_image_token_truncation=bool(config.get("allow_image_token_truncation", False)),
                local_files_only=bool(config.get("vlm_local_files_only", False)),
            )

        action_head_type = config.get("action_head", "flowmatching").lower()
        if action_head_type != "flowmatching":
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

        horizon = config.get("action_horizon", config.get("horizon", 32))
        per_action_dim = config.get("per_action_dim", 7)
        action_dim = horizon * per_action_dim

        config["horizon"] = horizon
        config["per_action_dim"] = per_action_dim
        config["action_dim"] = action_dim

        if action_dim != horizon * per_action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) must equal horizon ({horizon}) * "
                f"per_action_dim ({per_action_dim})"
            )

        self.horizon = horizon
        self.per_action_dim = per_action_dim

        action_head_config = SimpleNamespace(
            embed_dim=config.get("embed_dim", 896),
            hidden_dim=config.get("hidden_dim", 1024),
            ffn_dim=config.get("action_head_ffn_dim", config.get("embed_dim", 896) * 4),
            action_dim=action_dim,
            horizon=horizon,
            per_action_dim=per_action_dim,
            state_dim=config.get("state_dim", 7),
            state_hidden_dim=config.get("state_hidden_dim", 1024),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 8),
            dropout=config.get("dropout", 0.0),
            num_inference_timesteps=config.get("num_inference_timesteps", 15),
            inference_tau_schedule=config.get("inference_tau_schedule", "midpoint"),
            avoid_endpoint_tau=config.get("avoid_endpoint_tau", True),
            num_categories=config.get("num_categories", 1),
            num_plan_slots=config.get("num_plan_slots", 8),
            visual_gate_lambda=config.get("visual_gate_lambda", 0.5),
            plan_gate_lambda=config.get("plan_gate_lambda", 0.25),
            short_memory_time_bins=config.get("short_memory_time_bins", 2),
            max_vlm_tokens=config.get("max_vlm_tokens", None),
        )
        self.action_head = FlowmatchingActionHead(config=action_head_config).to(self._device)
        self.use_bridge = bool(config.get("use_bridge", False))
        self.bridge_variant = str(config.get("bridge_variant", "crosskv"))
        self.use_direct_bridge = self.use_bridge and self.bridge_variant == "direct"
        self.bridge_context_mode = str(
            config.get("bridge_context_mode", "bridge_residual" if self.use_bridge else "fused_only")
        )
        self.memory_placement = str(config.get("memory_placement", "crosskv"))
        self.bridge_adapter = None
        self.progress_state_planner = None
        self.runtime_progress_state: ProgressState | None = None
        self.skill_tokens = None
        self.fused_residual_gate = None
        self.last_bridge_output: BridgeAdapterOutput | None = None
        self.last_progress_planner_output: ProgressPlannerOutput | None = None

        if self.use_bridge and not self.use_direct_bridge:
            if self.bridge_context_mode not in {
                "fused_only",
                "bridge_clean",
                "bridge_residual",
                "bridge_gated_residual",
            }:
                raise ValueError(f"Unknown bridge_context_mode: {self.bridge_context_mode}")
            bridge_config = BridgeAdapterConfig(
                embed_dim=config.get("bridge_hidden_dim", config.get("embed_dim", 896)),
                raw_dim=config.get("bridge_raw_dim", config.get("embed_dim", 896)),
                state_dim=config.get("state_dim", 7),
                num_layers=config.get("bridge_num_layers", 2),
                num_heads=config.get("bridge_num_heads", 8),
                num_bridge_tokens=config.get("bridge_num_tokens", 16),
                num_action_queries=config.get("bridge_num_action_queries", 64),
                dropout=config.get("bridge_dropout", config.get("dropout", 0.0)),
                raw_gate_init=config.get("bridge_raw_gate_init", 0.0),
                ffn_mult=config.get("bridge_ffn_mult", 4),
            )
            self.bridge_adapter = BridgeAdapter(bridge_config).to(self._device)
            if self.bridge_context_mode == "bridge_gated_residual":
                gate_init = float(config.get("bridge_fused_gate_init", 0.0))
                self.fused_residual_gate = nn.Parameter(torch.tensor(gate_init))

            if bool(config.get("skill_tokens_enabled", False)):
                skill_count = int(config.get("skill_num_tokens", 4))
                if skill_count <= 0:
                    raise ValueError(f"skill_num_tokens must be positive, got {skill_count}")
                self.skill_tokens = nn.Parameter(torch.empty(skill_count, bridge_config.embed_dim))
                nn.init.normal_(self.skill_tokens, mean=0.0, std=0.02)

        if bool(config.get("progress_planner_enabled", False)) or config.get("progress_planner_checkpoint"):
            if not self.use_direct_bridge:
                raise ValueError("progress planner integration requires bridge.variant=direct")
            if (
                bool(config.get("progress_planner_enabled", False))
                and not config.get("progress_planner_checkpoint")
                and not bool(config.get("finetune_progress_planner", False))
            ):
                raise ValueError(
                    "progress_planner_enabled=true with no checkpoint and finetune_progress_planner=false "
                    "would use a random frozen progress planner"
                )
            self.progress_state_planner = self._build_progress_state_planner(config).to(self._device)

    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,
        prompt: str = "",
        return_cls_only: Union[bool, None] = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | InternVL3EmbeddingOutput:
        if self.embedder is None:
            raise RuntimeError("VLM embedder is not loaded; set load_vlm=true for image-based training or inference")
        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image tensor.")

        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
            return_hidden_states=return_hidden_states,
            selected_layers=self.config.get("bridge_raw_layers", None),
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError(f"Unsupported state input type: {type(state_input)!r}")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
        hidden_states: list[torch.Tensor] | None = None,
        memory_context: torch.Tensor | None = None,
        memory_context_mask: torch.Tensor | None = None,
        short_memory_time_ids: torch.Tensor | None = None,
        executed_actions: torch.Tensor | None = None,
        executed_action_mask: torch.Tensor | None = None,
        progress_state: ProgressState | torch.Tensor | None = None,
        planner_fused_tokens: torch.Tensor | None = None,
        planner_vl_summary: torch.Tensor | None = None,
        planner_state: torch.Tensor | None = None,
        plan_token_mask: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_direct_bridge:
            plan_tokens = None
            if self.progress_state_planner is not None:
                plan_tokens = self._get_or_update_progress_plan_tokens(
                    fused_tokens,
                    state,
                    executed_actions=executed_actions,
                    executed_action_mask=executed_action_mask,
                    progress_state=progress_state,
                    planner_fused_tokens=planner_fused_tokens,
                    planner_vl_summary=planner_vl_summary,
                    planner_state=planner_state,
                )
            else:
                self.last_progress_planner_output = None
            self.last_bridge_output = None
            short_memory_time_ids = self._resolve_short_memory_time_ids(memory_context, short_memory_time_ids)

            if actions_gt is None:
                return self.action_head.get_action(
                    fused_tokens,
                    state=state,
                    action_mask=action_mask,
                    embodiment_id=embodiment_ids,
                    vlm_hidden_states=hidden_states,
                    short_memory_tokens=memory_context,
                    short_memory_time_ids=short_memory_time_ids,
                    short_memory_mask=memory_context_mask,
                    plan_tokens=plan_tokens,
                    plan_token_mask=plan_token_mask,
                )

            return self.action_head(
                fused_tokens,
                state=state,
                actions_gt=actions_gt,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
                vlm_hidden_states=hidden_states,
                short_memory_tokens=memory_context,
                short_memory_time_ids=short_memory_time_ids,
                short_memory_mask=memory_context_mask,
                plan_tokens=plan_tokens,
                plan_token_mask=plan_token_mask,
            )

        fused_tokens = self._augment_context_with_bridge(
            fused_tokens,
            state=state,
            hidden_states=hidden_states,
            memory_context=memory_context,
            memory_context_mask=memory_context_mask,
            plan_token_mask=plan_token_mask,
        )
        if actions_gt is None:
            return self.action_head.get_action(
                fused_tokens,
                state=state,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
            )

        return self.action_head(
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_ids,
        )

    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        embedding_output = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only,
            return_hidden_states=self.use_bridge,
        )
        if isinstance(embedding_output, InternVL3EmbeddingOutput):
            hidden_states = embedding_output.hidden_states
            fused_tokens = (
                embedding_output.visual_tokens
                if embedding_output.visual_tokens is not None
                else embedding_output.fused_tokens
            )
        else:
            fused_tokens = embedding_output
            hidden_states = None
        state_tensor = self.prepare_state(state_input)
        action = self.predict_action(
            fused_tokens,
            state_tensor,
            action_mask=action_mask,
            hidden_states=hidden_states,
            memory_context=None,
        )
        return action

    def forward(
        self,
        fused_tokens,
        state=None,
        actions_gt=None,
        action_mask=None,
        embodiment_ids=None,
        hidden_states=None,
        memory_context=None,
        memory_context_mask=None,
        short_memory_time_ids=None,
        planner_fused_tokens=None,
        planner_vl_summary=None,
        planner_state=None,
        executed_actions=None,
        executed_action_mask=None,
        progress_state=None,
        plan_token_mask=None,
    ):
        return self.predict_action(
            fused_tokens,
            state,
            actions_gt,
            action_mask,
            embodiment_ids,
            hidden_states=hidden_states,
            memory_context=memory_context,
            memory_context_mask=memory_context_mask,
            short_memory_time_ids=short_memory_time_ids,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
            progress_state=progress_state,
            planner_fused_tokens=planner_fused_tokens,
            planner_vl_summary=planner_vl_summary,
            planner_state=planner_state,
            plan_token_mask=plan_token_mask,
        )

    def _augment_context_with_bridge(
        self,
        fused_tokens: torch.Tensor,
        *,
        state: torch.Tensor | None,
        hidden_states: list[torch.Tensor] | None,
        memory_context: torch.Tensor | None,
        memory_context_mask: torch.Tensor | None,
        plan_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.bridge_adapter is None:
            self.last_bridge_output = None
            return fused_tokens

        self.last_progress_planner_output = None

        bridge_output = self.bridge_adapter(
            fused_tokens,
            hidden_states=hidden_states,
            state=state,
            plan_tokens=None,
            plan_token_mask=plan_token_mask,
            memory_context=memory_context if self.memory_placement == "crosskv" else None,
            memory_context_mask=memory_context_mask if self.memory_placement == "crosskv" else None,
        )
        self.last_bridge_output = bridge_output
        return self._build_action_context(fused_tokens, bridge_output.bridge_tokens, memory_context)

    def _build_action_context(
        self,
        fused_tokens: torch.Tensor,
        bridge_tokens: torch.Tensor,
        memory_context: torch.Tensor | None,
    ) -> torch.Tensor:
        fused_tokens = _ensure_rank3(fused_tokens, "fused_tokens")
        if self.bridge_context_mode == "fused_only":
            context_tokens = fused_tokens
        elif self.bridge_context_mode == "bridge_clean":
            context_tokens = bridge_tokens
        elif self.bridge_context_mode == "bridge_residual":
            context_tokens = torch.cat([fused_tokens, bridge_tokens], dim=1)
        elif self.bridge_context_mode == "bridge_gated_residual":
            if self.fused_residual_gate is None:
                raise RuntimeError("fused_residual_gate was not initialized")
            gate = torch.tanh(self.fused_residual_gate).to(device=fused_tokens.device, dtype=fused_tokens.dtype)
            context_tokens = torch.cat([gate * fused_tokens, bridge_tokens], dim=1)
        else:
            raise ValueError(f"Unknown bridge_context_mode: {self.bridge_context_mode}")

        if self.memory_placement == "mixed_latent" and memory_context is not None:
            memory_context = _ensure_rank3(memory_context, "memory_context").to(
                device=context_tokens.device,
                dtype=context_tokens.dtype,
            )
            if memory_context.shape[1] > 0:
                context_tokens = torch.cat([context_tokens, memory_context], dim=1)

        if self.skill_tokens is not None:
            skill_tokens = self.skill_tokens.to(device=context_tokens.device, dtype=context_tokens.dtype)
            skill_tokens = skill_tokens.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
            context_tokens = torch.cat([context_tokens, skill_tokens], dim=1)

        return context_tokens

    def _get_or_update_progress_plan_tokens(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor | None,
        *,
        executed_actions: torch.Tensor | None,
        executed_action_mask: torch.Tensor | None,
        progress_state: ProgressState | torch.Tensor | None = None,
        planner_fused_tokens: torch.Tensor | None = None,
        planner_vl_summary: torch.Tensor | None = None,
        planner_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.progress_state_planner is None:
            raise RuntimeError("progress state planner is not initialized")
        if state is None:
            raise ValueError("state is required when progress planner is enabled")
        if executed_actions is None:
            raise ValueError("executed_actions is required when progress planner is enabled")

        planner_state = state if planner_state is None else planner_state
        vl_summary = self._resolve_progress_vl_summary(
            fused_tokens,
            planner_fused_tokens=planner_fused_tokens,
            planner_vl_summary=planner_vl_summary,
        )
        batch_size = int(vl_summary.shape[0])
        previous_state = self._resolve_progress_state(progress_state, batch_size, vl_summary.device, vl_summary.dtype)
        planner_context = (
            nullcontext()
            if bool(self.config.get("finetune_progress_planner", False))
            else torch.no_grad()
        )
        with planner_context:
            output = self.progress_state_planner.forward_step(
                previous_state,
                vl_summary,
                planner_state,
                executed_actions,
                executed_action_mask,
            )
        self.last_progress_planner_output = output
        if not self.training:
            self.runtime_progress_state = ProgressState(
                completed_events=output.progress_state.completed_events.detach(),
                current_stage=output.progress_state.current_stage.detach(),
            )
        return output.planner_token

    def _resolve_progress_vl_summary(
        self,
        fused_tokens: torch.Tensor,
        *,
        planner_fused_tokens: torch.Tensor | None = None,
        planner_vl_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.progress_state_planner is None:
            raise RuntimeError("progress state planner is not initialized")
        hidden_dim = int(self.progress_state_planner.config.hidden_dim)
        if planner_vl_summary is not None:
            if planner_vl_summary.ndim == 3 and planner_vl_summary.shape[1] == 1:
                planner_vl_summary = planner_vl_summary.squeeze(1)
            if planner_vl_summary.ndim != 2 or planner_vl_summary.shape[-1] != hidden_dim:
                raise ValueError(
                    "planner_vl_summary must have shape "
                    f"[B, {hidden_dim}] or [B, 1, {hidden_dim}], got {tuple(planner_vl_summary.shape)}"
                )
            return planner_vl_summary.to(device=fused_tokens.device, dtype=fused_tokens.dtype)

        planner_tokens = fused_tokens if planner_fused_tokens is None else planner_fused_tokens
        planner_tokens = _ensure_rank3(planner_tokens, "planner_fused_tokens").to(
            device=fused_tokens.device,
            dtype=fused_tokens.dtype,
        )
        if planner_tokens.shape[-1] != hidden_dim:
            raise ValueError(f"planner_fused_tokens last dim {planner_tokens.shape[-1]} != {hidden_dim}")
        return planner_tokens.mean(dim=1)

    def _resolve_progress_state(
        self,
        progress_state: ProgressState | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ProgressState | torch.Tensor:
        if progress_state is not None:
            return progress_state
        if not self.training and self.runtime_progress_state is not None:
            if self.runtime_progress_state.completed_events.shape[0] == batch_size:
                return self.runtime_progress_state
        if self.progress_state_planner is None:
            raise RuntimeError("progress state planner is not initialized")
        return self.progress_state_planner.initial_state(batch_size, device=device, dtype=dtype)

    def reset_progress_state(self) -> None:
        self.runtime_progress_state = None
        self.last_progress_planner_output = None

    def _build_progress_state_planner(self, config: dict) -> ProgressStatePlanner:
        checkpoint = None
        checkpoint_path = config.get("progress_planner_checkpoint")
        if checkpoint_path:
            checkpoint_file = Path(str(checkpoint_path)).expanduser()
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            if checkpoint.get("format") != "progress_state_planner_warmup":
                raise ValueError(f"invalid progress planner checkpoint format: {checkpoint.get('format')!r}")
        if checkpoint is not None:
            model_config = dict(checkpoint.get("model_config") or {})
            if not model_config:
                raise KeyError(f"progress planner checkpoint lacks model_config: {checkpoint_path}")
        else:
            model_config = {
                key.removeprefix("progress_planner_"): value
                for key, value in config.items()
                if key.startswith("progress_planner_")
                and key not in {"progress_planner_enabled", "progress_planner_checkpoint"}
            }
            model_config.setdefault("hidden_dim", config.get("embed_dim", 896))
            model_config.setdefault("state_dim", config.get("state_dim", 7))
            model_config.setdefault("action_dim", config.get("per_action_dim", 7))
            model_config.setdefault("replan_stride", config.get("progress_planner_replan_stride", 16))
        planner = ProgressStatePlanner(ProgressStateConfig(**model_config))
        if checkpoint is not None:
            planner.load_state_dict(checkpoint["model_state_dict"])
        return planner

    def _resolve_short_memory_time_ids(
        self,
        memory_context: torch.Tensor | None,
        short_memory_time_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if short_memory_time_ids is not None or memory_context is None:
            return short_memory_time_ids
        memory_context = _ensure_rank3(memory_context, "memory_context")
        entry_tokens = int(self.config.get("memory_entry_tokens", 16))
        capacity = int(self.config.get("memory_short_capacity", 2))
        if entry_tokens <= 0 or capacity <= 0:
            return None
        expected_tokens = entry_tokens * capacity
        if memory_context.shape[1] != expected_tokens:
            return None
        time_ids = torch.arange(capacity, device=memory_context.device, dtype=torch.long)
        time_ids = time_ids.repeat_interleave(entry_tokens)
        return time_ids.unsqueeze(0).expand(memory_context.shape[0], -1)

    def _freeze_module(self, module: nn.Module, name: str):
        logging.info(f"Freezing {name} parameters...")
        for param in module.parameters():
            param.requires_grad = False

    def set_finetune_flags(self):
        if self.embedder is None:
            if self.config.get("finetune_vlm", False):
                raise ValueError("finetune_vlm=true requires load_vlm=true")
            logging.info("Skipping VLM freeze because load_vlm=false")
        elif not self.config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
        else:
            logging.info("Finetuning VLM (InternVL3)...")

        if not self.config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            logging.info("Finetuning Action Head...")

        if self.progress_state_planner is not None and not self.config.get("finetune_progress_planner", False):
            self._freeze_module(self.progress_state_planner, "Progress-State Planner")
            self.progress_state_planner.eval()


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor



# Backward-compatible local alias for migrated checkpoints/tests.
PrismBridgeVLA = PrismPolicy
