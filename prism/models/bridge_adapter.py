from __future__ import annotations

from prism.models.bridge_attention import BridgeAttentionBlock

# --- migrated from src/prism/model/bridge/adapter.py ---
from dataclasses import dataclass

import torch
import torch.nn as nn



@dataclass(frozen=True)
class BridgeAdapterConfig:
    embed_dim: int = 896
    raw_dim: int | None = None
    state_dim: int = 7
    num_layers: int = 2
    num_heads: int = 8
    num_bridge_tokens: int = 16
    num_action_queries: int = 64
    dropout: float = 0.0
    raw_gate_init: float = 0.0
    ffn_mult: int = 4


@dataclass(frozen=True)
class BridgeAdapterOutput:
    bridge_tokens: torch.Tensor
    boundary_logits: torch.Tensor
    progress_logits: torch.Tensor
    raw_gate_values: torch.Tensor


class BridgeAdapter(nn.Module):
    """Context adapter inserted before the existing flow-matching action head."""

    def __init__(self, config: BridgeAdapterConfig) -> None:
        super().__init__()
        if config.num_layers <= 0:
            raise ValueError("BridgeAdapter requires at least one layer")
        if config.num_bridge_tokens <= 0:
            raise ValueError("num_bridge_tokens must be positive")
        if config.num_action_queries <= 0:
            raise ValueError("num_action_queries must be positive")

        self.config = config
        self.action_tokens = nn.Parameter(torch.empty(config.num_bridge_tokens, config.embed_dim))
        self.action_queries = nn.Parameter(torch.empty(config.num_action_queries, config.embed_dim))
        nn.init.normal_(self.action_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)

        self.state_proj = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        self.memory_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.blocks = nn.ModuleList(
            [
                BridgeAttentionBlock(
                    hidden_dim=config.embed_dim,
                    raw_dim=config.raw_dim or config.embed_dim,
                    query_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    raw_gate_init=config.raw_gate_init,
                    ffn_mult=config.ffn_mult,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 1),
        )
        self.progress_head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 1),
        )

    def forward(
        self,
        fused_tokens: torch.Tensor,
        *,
        hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        state: torch.Tensor | None = None,
        plan_tokens: torch.Tensor | None = None,
        plan_token_mask: torch.Tensor | None = None,
        memory_context: torch.Tensor | None = None,
        memory_context_mask: torch.Tensor | None = None,
    ) -> BridgeAdapterOutput:
        fused_tokens = _ensure_rank3(fused_tokens, "fused_tokens")
        batch_size = fused_tokens.shape[0]
        device = fused_tokens.device
        dtype = fused_tokens.dtype

        action_tokens = self.action_tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        action_queries = self.action_queries.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        proprio_embedding = self._project_state(state, batch_size, device, dtype)
        plan_tokens = self._prepare_plan_tokens(plan_tokens, device, dtype)
        query_key_padding_mask = self._build_query_key_padding_mask(
            batch_size=batch_size,
            action_query_count=action_queries.shape[1],
            proprio_count=proprio_embedding.shape[1],
            plan_tokens=plan_tokens,
            plan_token_mask=plan_token_mask,
            memory_context=memory_context,
            memory_context_mask=memory_context_mask,
            device=device,
        )
        memory_context = self._project_memory(memory_context, device, dtype)

        raw_layers = _normalize_hidden_states(hidden_states, fused_tokens)
        for index, block in enumerate(self.blocks):
            raw_features = raw_layers[min(index, len(raw_layers) - 1)].to(device=device, dtype=dtype)
            action_tokens = block(
                action_tokens=action_tokens,
                raw_features=raw_features,
                action_query_features=action_queries,
                proprio_embedding=proprio_embedding,
                plan_tokens=plan_tokens,
                memory_context=memory_context,
                query_key_padding_mask=query_key_padding_mask,
            )

        bridge_tokens = self.output_norm(action_tokens)
        pooled = bridge_tokens.mean(dim=1)
        boundary_logits = self.boundary_head(pooled)
        progress_logits = self.progress_head(pooled)
        raw_gate_values = torch.stack([block.raw_gate_value.to(device=device, dtype=dtype) for block in self.blocks])
        return BridgeAdapterOutput(
            bridge_tokens=bridge_tokens,
            boundary_logits=boundary_logits,
            progress_logits=progress_logits,
            raw_gate_values=raw_gate_values,
        )

    def _project_state(
        self,
        state: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state is None:
            return torch.zeros(batch_size, 1, self.config.embed_dim, device=device, dtype=dtype)
        if state.ndim == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        if state.ndim != 2:
            raise ValueError(f"state must have shape [B, state_dim] or [B, 1, state_dim], got {tuple(state.shape)}")
        if state.shape[-1] != self.config.state_dim:
            raise ValueError(f"state last dimension {state.shape[-1]} != state_dim {self.config.state_dim}")
        projected = self.state_proj.to(device=device, dtype=dtype)(state.to(device=device, dtype=dtype))
        return projected.unsqueeze(1)

    def _project_memory(
        self,
        memory_context: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if memory_context is None:
            return None
        memory_context = _ensure_rank3(memory_context, "memory_context").to(device=device, dtype=dtype)
        if memory_context.shape[1] == 0:
            return memory_context
        if memory_context.shape[-1] != self.config.embed_dim:
            raise ValueError(
                f"memory_context last dimension {memory_context.shape[-1]} != embed_dim {self.config.embed_dim}"
            )
        return self.memory_proj.to(device=device, dtype=dtype)(memory_context)

    def _prepare_plan_tokens(
        self,
        plan_tokens: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if plan_tokens is None:
            return None
        plan_tokens = _ensure_rank3(plan_tokens, "plan_tokens").to(device=device, dtype=dtype)
        if plan_tokens.shape[1] == 0:
            return plan_tokens
        if plan_tokens.shape[-1] != self.config.embed_dim:
            raise ValueError(f"plan_tokens last dimension {plan_tokens.shape[-1]} != embed_dim {self.config.embed_dim}")
        return plan_tokens

    def _build_query_key_padding_mask(
        self,
        *,
        batch_size: int,
        action_query_count: int,
        proprio_count: int,
        plan_tokens: torch.Tensor | None,
        plan_token_mask: torch.Tensor | None,
        memory_context: torch.Tensor | None,
        memory_context_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        masks = [
            torch.zeros(batch_size, action_query_count, dtype=torch.bool, device=device),
            torch.zeros(batch_size, proprio_count, dtype=torch.bool, device=device),
        ]
        if plan_tokens is not None and plan_tokens.shape[1] > 0:
            if plan_token_mask is None:
                masks.append(torch.zeros(batch_size, plan_tokens.shape[1], dtype=torch.bool, device=device))
            else:
                plan_token_mask = plan_token_mask.to(device=device).bool()
                if plan_token_mask.shape != plan_tokens.shape[:2]:
                    raise ValueError(
                        f"plan_token_mask shape {tuple(plan_token_mask.shape)} must match "
                        f"plan token prefix {tuple(plan_tokens.shape[:2])}"
                    )
                masks.append(~plan_token_mask)
        if memory_context is not None:
            memory_context = _ensure_rank3(memory_context, "memory_context")
            if memory_context.shape[1] > 0:
                if memory_context_mask is None:
                    masks.append(torch.zeros(batch_size, memory_context.shape[1], dtype=torch.bool, device=device))
                else:
                    memory_context_mask = memory_context_mask.to(device=device).bool()
                    if memory_context_mask.shape != memory_context.shape[:2]:
                        raise ValueError(
                            f"memory_context_mask shape {tuple(memory_context_mask.shape)} must match "
                            f"memory context prefix {tuple(memory_context.shape[:2])}"
                        )
                    masks.append(~memory_context_mask)
        if not masks:
            return None
        merged = torch.cat(masks, dim=1)
        return merged if merged.any() else None


def _normalize_hidden_states(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    fused_tokens: torch.Tensor,
) -> list[torch.Tensor]:
    if hidden_states is None:
        return [fused_tokens]
    if len(hidden_states) == 0:
        raise ValueError("hidden_states must not be empty when provided")
    return [_ensure_rank3(hidden_state, "hidden_state") for hidden_state in hidden_states]


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor

