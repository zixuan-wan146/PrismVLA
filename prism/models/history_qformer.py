from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from prism.models.config import HistoryQFormerConfig


@dataclass(frozen=True)
class HistoryMemoryOutput:
    tokens: torch.Tensor
    valid_mask: torch.Tensor


class HistoryQFormerBlock(nn.Module):
    def __init__(self, config: HistoryQFormerConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.query_self_norm = nn.LayerNorm(hidden_size)
        self.query_cross_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        mlp_size = hidden_size * config.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_queries = self.query_self_norm(queries)
        self_attention, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + self_attention
        normalized_context = self.context_norm(context)
        cross_attention, _ = self.cross_attention(
            self.query_cross_norm(queries),
            normalized_context,
            normalized_context,
            key_padding_mask=~context_valid_mask,
            need_weights=False,
        )
        queries = queries + cross_attention
        return queries + self.mlp(self.mlp_norm(queries))


class HistoryQFormer(nn.Module):
    """Compress two sparse, two-camera visual histories into fixed memory tokens."""

    def __init__(self, config: HistoryQFormerConfig | None = None) -> None:
        super().__init__()
        self.config = HistoryQFormerConfig() if config is None else config
        self.config.validate()
        self.input_projection = nn.Linear(self.config.input_dim, self.config.hidden_size)
        self.relative_age_embedding = nn.Embedding(self.config.max_relative_age + 1, self.config.hidden_size)
        self.memory_queries = nn.Parameter(torch.empty(self.config.num_memory_tokens, self.config.hidden_size))
        self.blocks = nn.ModuleList(HistoryQFormerBlock(self.config) for _ in range(self.config.num_layers))
        self.output_norm = nn.LayerNorm(self.config.hidden_size)
        nn.init.normal_(self.memory_queries, mean=0.0, std=0.02)

    def forward(
        self,
        history_visual_tokens: torch.Tensor,
        history_step_ages: torch.Tensor,
        history_valid_mask: torch.Tensor,
        history_token_mask: torch.Tensor | None = None,
    ) -> HistoryMemoryOutput:
        if history_visual_tokens.ndim != 4:
            raise ValueError("history_visual_tokens must have shape [B, history, tokens, input_dim]")
        batch_size, history_count, tokens_per_history, input_dim = history_visual_tokens.shape
        expected = (batch_size, self.config.num_history_frames)
        if history_count != self.config.num_history_frames or input_dim != self.config.input_dim:
            raise ValueError(
                f"Expected history shape [B, {self.config.num_history_frames}, L, {self.config.input_dim}], "
                f"got {tuple(history_visual_tokens.shape)}"
            )
        if history_step_ages.shape != expected or history_valid_mask.shape != expected:
            raise ValueError(f"history_step_ages and history_valid_mask must have shape {expected}")
        if history_step_ages.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError("history_step_ages must contain integers")
        if history_valid_mask.dtype != torch.bool:
            raise ValueError("history_valid_mask must be boolean")
        if history_step_ages.device.type == "cpu" and (
            torch.any(history_step_ages < 0) or torch.any(history_step_ages > self.config.max_relative_age)
        ):
            raise ValueError(f"history_step_ages must be in [0, {self.config.max_relative_age}]")
        if history_token_mask is None:
            history_token_mask = torch.ones(
                batch_size,
                history_count,
                tokens_per_history,
                dtype=torch.bool,
                device=history_visual_tokens.device,
            )
        if history_token_mask.shape != (batch_size, history_count, tokens_per_history):
            raise ValueError("history_token_mask shape does not match history_visual_tokens")
        if history_token_mask.dtype != torch.bool:
            raise ValueError("history_token_mask must be boolean")

        projection_weight = self.input_projection.weight
        history_visual_tokens = history_visual_tokens.to(
            device=projection_weight.device,
            dtype=projection_weight.dtype,
        )
        history_step_ages = history_step_ages.to(device=projection_weight.device, dtype=torch.long)
        history_valid_mask = history_valid_mask.to(device=projection_weight.device)
        history_token_mask = history_token_mask.to(device=projection_weight.device)

        context = self.input_projection(history_visual_tokens)
        age_embeddings = self.relative_age_embedding(history_step_ages).unsqueeze(2)
        context = context + age_embeddings
        context_valid_mask = history_token_mask & history_valid_mask.unsqueeze(-1)
        context = context.reshape(batch_size, history_count * tokens_per_history, self.config.hidden_size)
        context_valid_mask = context_valid_mask.reshape(batch_size, history_count * tokens_per_history)

        sample_valid_mask = context_valid_mask.any(dim=1)
        safe_context_valid_mask = context_valid_mask.clone()
        safe_context_valid_mask[:, 0] |= ~sample_valid_mask
        queries = self.memory_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, context, safe_context_valid_mask)
        output_tokens = self.output_norm(queries)
        output_tokens = output_tokens * sample_valid_mask[:, None, None].to(dtype=output_tokens.dtype)

        memory_valid_mask = sample_valid_mask.unsqueeze(1).expand(-1, self.config.num_memory_tokens).clone()
        return HistoryMemoryOutput(tokens=output_tokens, valid_mask=memory_valid_mask)
