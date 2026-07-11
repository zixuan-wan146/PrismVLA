from __future__ import annotations

# --- migrated from src/prism/model/bridge/bridge_attention.py ---
import math

import torch
import torch.nn as nn


class BridgeAttentionBlock(nn.Module):
    """VLA-Adapter style bridge block for action-latent conditioning.

    The block keeps separate paths for action-token self-attention, raw VLM
    feature cross-attention, and action-query/proprio/plan/memory cross-attention.
    Raw VLM injection is gated with a zero-initialized tanh gate by default.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        raw_dim: int | None = None,
        query_dim: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        raw_gate_init: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.hidden_dim = hidden_dim
        raw_dim = hidden_dim if raw_dim is None else raw_dim
        query_dim = hidden_dim if query_dim is None else query_dim

        self.raw_proj = nn.Linear(raw_dim, hidden_dim) if raw_dim != hidden_dim else nn.Identity()
        self.query_proj = nn.Linear(query_dim, hidden_dim) if query_dim != hidden_dim else nn.Identity()

        self.action_norm = nn.LayerNorm(hidden_dim)
        self.raw_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.raw_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.raw_gate = nn.Parameter(torch.tensor(float(raw_gate_init)))
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    @property
    def raw_gate_value(self) -> torch.Tensor:
        return torch.tanh(self.raw_gate)

    def forward(
        self,
        action_tokens: torch.Tensor,
        raw_features: torch.Tensor,
        action_query_features: torch.Tensor,
        proprio_embedding: torch.Tensor | None = None,
        plan_tokens: torch.Tensor | None = None,
        memory_context: torch.Tensor | None = None,
        raw_key_padding_mask: torch.Tensor | None = None,
        query_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_tokens = _ensure_rank3(action_tokens, "action_tokens")
        raw_features = _ensure_rank3(raw_features, "raw_features")
        action_query_features = _ensure_rank3(action_query_features, "action_query_features")

        if action_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"action_tokens last dimension {action_tokens.shape[-1]} != hidden_dim {self.hidden_dim}"
            )

        raw_context = self.raw_norm(self.raw_proj(raw_features))
        query_context = self.query_norm(self.query_proj(action_query_features))
        conditions = [query_context]

        if proprio_embedding is not None:
            conditions.append(_ensure_rank3(proprio_embedding, "proprio_embedding"))
        if plan_tokens is not None:
            plan_tokens = _ensure_rank3(plan_tokens, "plan_tokens")
            if plan_tokens.shape[1] > 0:
                conditions.append(plan_tokens)
        if memory_context is not None:
            memory_context = _ensure_rank3(memory_context, "memory_context")
            if memory_context.shape[1] > 0:
                conditions.append(memory_context)

        condition_context = torch.cat(conditions, dim=1)
        if condition_context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"condition context last dimension {condition_context.shape[-1]} != hidden_dim {self.hidden_dim}"
            )

        query_tokens = self.action_norm(action_tokens)
        self_out, _ = self.self_attn(query_tokens, query_tokens, query_tokens, need_weights=False)
        raw_out, _ = self.raw_attn(
            query_tokens,
            raw_context,
            raw_context,
            key_padding_mask=raw_key_padding_mask,
            need_weights=False,
        )
        query_out, _ = self.query_attn(
            query_tokens,
            condition_context,
            condition_context,
            key_padding_mask=query_key_padding_mask,
            need_weights=False,
        )

        fused = action_tokens
        fused = fused + self.dropout(self_out)
        fused = fused + self.dropout(self.raw_gate_value * raw_out)
        fused = fused + self.dropout(query_out)
        return fused + self.ffn(fused)


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor


def inverse_tanh(value: float) -> float:
    if not -1.0 < value < 1.0:
        raise ValueError(f"value must be in (-1, 1), got {value}")
    return 0.5 * math.log((1.0 + value) / (1.0 - value))

