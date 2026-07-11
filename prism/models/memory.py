from __future__ import annotations

# --- migrated from src/prism/model/himem/shortmemory.py ---
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PerceptualMemoryEntry:
    tokens: torch.Tensor
    timestep: int

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if self.tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [N, D], got {tuple(self.tokens.shape)}")
        timestep = int(self.timestep)
        if timestep < 0:
            raise ValueError(f"timestep must be non-negative, got {self.timestep}")
        object.__setattr__(self, "timestep", timestep)


@dataclass(frozen=True)
class PerceptualMemoryOutput:
    current_tokens: torch.Tensor
    retrieved_tokens: torch.Tensor
    tokens: torch.Tensor
    mask: torch.Tensor
    gate: torch.Tensor
    episode_ids: tuple[str, ...]
    timesteps: tuple[int, ...]

    @property
    def memory_context(self) -> torch.Tensor:
        return self.tokens

    @property
    def memory_context_mask(self) -> torch.Tensor:
        return self.mask

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "memory_context": self.memory_context,
            "memory_context_mask": self.memory_context_mask,
        }


@dataclass(frozen=True)
class RecentVisualMemoryOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    offsets: tuple[int, ...]

    @property
    def memory_context(self) -> torch.Tensor:
        return self.tokens

    @property
    def memory_context_mask(self) -> torch.Tensor:
        return self.mask

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "memory_context": self.memory_context,
            "memory_context_mask": self.memory_context_mask,
        }


@dataclass(frozen=True)
class PerceptualVisualMemoryConfig:
    hidden_dim: int = 896
    memory_tokens: int = 16
    capacity: int = 4
    num_heads: int = 8
    retrieval_layers: int = 2
    dropout: float = 0.0
    ffn_mult: int = 4
    consolidation: str = "merge"
    store_fused: bool = True
    detach_on_update: bool = True


@dataclass(frozen=True)
class FixedRecentVisualMemoryConfig:
    hidden_dim: int = 896
    tokens_per_observation: int = 16
    offsets: tuple[int, ...] = (8, 16)
    compressor: str = "bottleneck_se"
    num_heads: int = 8
    bottleneck_ratio: int = 4
    dropout: float = 0.0


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        timesteps = timesteps.to(dtype=torch.float32)
        half = self.hidden_dim // 2
        if half == 0:
            embedding = timesteps.unsqueeze(-1)
        else:
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=timesteps.device, dtype=torch.float32)
                / max(half - 1, 1)
            )
            args = timesteps.unsqueeze(-1) * frequencies
            embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.hidden_dim % 2 == 1:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        param_dtype = self.proj[0].weight.dtype
        embedding = self.proj(embedding.to(dtype=param_dtype))
        return embedding if dtype is None else embedding.to(dtype=dtype)


class MemoryVLACrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * int(ffn_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * int(ffn_mult), hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_tokens = _ensure_rank3(query_tokens, "query_tokens")
        memory_tokens = _ensure_rank3(memory_tokens, "memory_tokens")
        if memory_tokens.shape[1] == 0:
            return query_tokens
        key_padding_mask = None
        if memory_mask is not None:
            memory_mask = memory_mask.to(device=memory_tokens.device).bool()
            if memory_mask.shape != memory_tokens.shape[:2]:
                raise ValueError(
                    f"memory_mask shape {tuple(memory_mask.shape)} must match memory prefix "
                    f"{tuple(memory_tokens.shape[:2])}"
                )
            key_padding_mask = ~memory_mask
        attended, _ = self.attn(
            self.query_norm(query_tokens),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = query_tokens + self.dropout(attended)
        return output + self.ffn(output)


class MemoryVLAGateFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.bias)

    def forward(self, current_tokens: torch.Tensor, retrieved_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_tokens = _ensure_rank3(current_tokens, "current_tokens")
        retrieved_tokens = _ensure_rank3(retrieved_tokens, "retrieved_tokens")
        if current_tokens.shape != retrieved_tokens.shape:
            raise ValueError(
                f"current_tokens shape {tuple(current_tokens.shape)} must match retrieved_tokens "
                f"{tuple(retrieved_tokens.shape)}"
            )
        gate = torch.sigmoid(self.gate(torch.cat([current_tokens, retrieved_tokens], dim=-1)))
        fused = gate * current_tokens + (1.0 - gate) * retrieved_tokens
        return fused, gate


class PerceptualTokenCompressor(nn.Module):
    """Compress arbitrary visual token sequences into a fixed short-memory budget."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        memory_tokens: int = 16,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if memory_tokens <= 0:
            raise ValueError(f"memory_tokens must be positive, got {memory_tokens}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.memory_tokens = int(memory_tokens)
        self.queries = nn.Parameter(torch.empty(self.memory_tokens, self.hidden_dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.source_norm = nn.LayerNorm(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.attn = nn.MultiheadAttention(self.hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, visual_tokens: torch.Tensor, visual_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        visual_tokens = _ensure_rank3(visual_tokens, "visual_tokens")
        if visual_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"visual_tokens dim {visual_tokens.shape[-1]} != hidden_dim {self.hidden_dim}")
        batch_size = visual_tokens.shape[0]
        key_padding_mask = None
        if visual_token_mask is not None:
            visual_token_mask = visual_token_mask.to(device=visual_tokens.device).bool()
            if visual_token_mask.shape != visual_tokens.shape[:2]:
                raise ValueError(
                    f"visual_token_mask shape {tuple(visual_token_mask.shape)} must match visual token prefix "
                    f"{tuple(visual_tokens.shape[:2])}"
                )
            if (~visual_token_mask).all(dim=1).any():
                raise ValueError("each sample must have at least one valid visual token")
            key_padding_mask = ~visual_token_mask
        queries = self.queries.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        source = self.source_norm(visual_tokens)
        attended, _ = self.attn(
            self.query_norm(queries),
            source,
            source,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.output_norm(queries + attended)


class BottleneckSETokenCompressor(nn.Module):
    """MemoryVLA-style bottleneck-SE compressor for square visual-token grids."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        memory_tokens: int = 16,
        bottleneck_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if memory_tokens <= 0:
            raise ValueError(f"memory_tokens must be positive, got {memory_tokens}")
        output_grid = int(math.isqrt(int(memory_tokens)))
        if output_grid * output_grid != int(memory_tokens):
            raise ValueError("BottleneckSETokenCompressor requires memory_tokens to be a square number")
        if int(bottleneck_ratio) <= 0:
            raise ValueError(f"bottleneck_ratio must be positive, got {bottleneck_ratio}")

        self.hidden_dim = int(hidden_dim)
        self.memory_tokens = int(memory_tokens)
        self.output_grid = output_grid
        bottleneck_dim = max(1, self.hidden_dim // int(bottleneck_ratio))
        se_dim = max(1, bottleneck_dim // 4)

        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.reduce = nn.Conv2d(self.hidden_dim, bottleneck_dim, kernel_size=1)
        self.se_reduce = nn.Conv2d(bottleneck_dim, se_dim, kernel_size=1)
        self.se_expand = nn.Conv2d(se_dim, bottleneck_dim, kernel_size=1)
        self.expand = nn.Conv2d(bottleneck_dim, self.hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, visual_tokens: torch.Tensor, visual_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        visual_tokens = _ensure_rank3(visual_tokens, "visual_tokens")
        if visual_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"visual_tokens dim {visual_tokens.shape[-1]} != hidden_dim {self.hidden_dim}")
        batch_size, token_count, _ = visual_tokens.shape
        input_grid = int(math.isqrt(int(token_count)))
        if input_grid * input_grid != int(token_count):
            raise ValueError(
                "BottleneckSETokenCompressor requires square visual-token grids; "
                f"got token_count={token_count}"
            )

        if visual_token_mask is not None:
            visual_token_mask = visual_token_mask.to(device=visual_tokens.device).bool()
            if visual_token_mask.shape != visual_tokens.shape[:2]:
                raise ValueError(
                    f"visual_token_mask shape {tuple(visual_token_mask.shape)} must match visual token prefix "
                    f"{tuple(visual_tokens.shape[:2])}"
                )
            if (~visual_token_mask).all(dim=1).any():
                raise ValueError("each sample must have at least one valid visual token")
            visual_tokens = visual_tokens.masked_fill(~visual_token_mask[:, :, None], 0.0)

        original_dtype = visual_tokens.dtype
        compute_dtype = self.reduce.weight.dtype
        normalized = self.input_norm(visual_tokens.to(dtype=compute_dtype))
        grid = normalized.transpose(1, 2).reshape(batch_size, self.hidden_dim, input_grid, input_grid)
        hidden = F.gelu(self.reduce(grid))
        se = F.adaptive_avg_pool2d(hidden, output_size=1)
        se = F.gelu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        hidden = hidden * se
        restored = self.expand(self.dropout(hidden)) + grid
        pooled = F.adaptive_avg_pool2d(restored, output_size=(self.output_grid, self.output_grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.output_norm(tokens).to(dtype=original_dtype)


class FixedRecentVisualMemory(nn.Module):
    """Compress fixed recent visual observations into Bridge-Attn memory context."""

    def __init__(self, config: FixedRecentVisualMemoryConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword arguments, not both")
        self.config = config or FixedRecentVisualMemoryConfig(**kwargs)
        self._validate_config(self.config)
        if self.config.compressor == "bottleneck_se":
            self.compressor = BottleneckSETokenCompressor(
                hidden_dim=self.config.hidden_dim,
                memory_tokens=self.config.tokens_per_observation,
                bottleneck_ratio=self.config.bottleneck_ratio,
                dropout=self.config.dropout,
            )
        elif self.config.compressor == "query_cross_attention":
            self.compressor = PerceptualTokenCompressor(
                hidden_dim=self.config.hidden_dim,
                memory_tokens=self.config.tokens_per_observation,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
        else:
            raise ValueError("compressor must be one of 'bottleneck_se' or 'query_cross_attention'")

    def forward(
        self,
        visual_tokens_by_offset: Sequence[torch.Tensor | None] | dict[int, torch.Tensor | None],
        *,
        visual_token_masks_by_offset: Sequence[torch.Tensor | None] | dict[int, torch.Tensor | None] | None = None,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecentVisualMemoryOutput:
        batch_size, device, dtype = self._infer_batch_device_dtype(
            visual_tokens_by_offset,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        parts: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for offset_index, offset in enumerate(self.config.offsets):
            visual_tokens = _get_offset_value(visual_tokens_by_offset, offset=offset, index=offset_index)
            visual_mask = _get_offset_value(
                visual_token_masks_by_offset,
                offset=offset,
                index=offset_index,
            )
            compressed, token_mask = self._compress_one_offset(
                visual_tokens,
                visual_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            parts.append(compressed)
            masks.append(token_mask)
        return RecentVisualMemoryOutput(
            tokens=torch.cat(parts, dim=1),
            mask=torch.cat(masks, dim=1),
            offsets=self.config.offsets,
        )

    def _compress_one_offset(
        self,
        visual_tokens: torch.Tensor | None,
        visual_token_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_count = self.config.tokens_per_observation
        if visual_tokens is None:
            empty_tokens = torch.zeros(batch_size, token_count, self.config.hidden_dim, device=device, dtype=dtype)
            empty_mask = torch.zeros(batch_size, token_count, device=device, dtype=torch.bool)
            return empty_tokens, empty_mask

        visual_tokens = _ensure_rank3(visual_tokens, "visual_tokens").to(device=device, dtype=dtype)
        if visual_tokens.shape[0] != batch_size:
            raise ValueError(f"visual_tokens batch size {visual_tokens.shape[0]} != expected {batch_size}")
        if visual_tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"visual_tokens dim {visual_tokens.shape[-1]} != hidden_dim {self.config.hidden_dim}")

        if visual_token_mask is None:
            row_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            valid_token_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=device)
        else:
            valid_token_mask = visual_token_mask.to(device=device).bool()
            if valid_token_mask.shape != visual_tokens.shape[:2]:
                raise ValueError(
                    f"visual_token_mask shape {tuple(valid_token_mask.shape)} must match "
                    f"visual_tokens prefix {tuple(visual_tokens.shape[:2])}"
                )
            row_mask = valid_token_mask.any(dim=1)

        output = torch.zeros(batch_size, token_count, self.config.hidden_dim, device=device, dtype=dtype)
        if row_mask.any():
            compressed = self.compressor(
                visual_tokens[row_mask],
                visual_token_mask=valid_token_mask[row_mask],
            )
            output[row_mask] = compressed
        output_mask = row_mask[:, None].expand(batch_size, token_count).clone()
        return output, output_mask

    def _infer_batch_device_dtype(
        self,
        visual_tokens_by_offset: Sequence[torch.Tensor | None] | dict[int, torch.Tensor | None],
        *,
        batch_size: int | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> tuple[int, torch.device, torch.dtype]:
        for offset_index, offset in enumerate(self.config.offsets):
            visual_tokens = _get_offset_value(visual_tokens_by_offset, offset=offset, index=offset_index)
            if visual_tokens is None:
                continue
            visual_tokens = _ensure_rank3(visual_tokens, "visual_tokens")
            inferred_batch = int(visual_tokens.shape[0])
            if batch_size is not None and int(batch_size) != inferred_batch:
                raise ValueError(f"batch_size {batch_size} != inferred batch size {inferred_batch}")
            resolved_device = torch.device(device) if device is not None else visual_tokens.device
            resolved_dtype = dtype or visual_tokens.dtype
            return inferred_batch, resolved_device, resolved_dtype
        if batch_size is None:
            raise ValueError("batch_size is required when all recent visual observations are missing")
        resolved_device = torch.device(device) if device is not None else self.compressor.queries.device
        resolved_dtype = dtype or self.compressor.queries.dtype
        return int(batch_size), resolved_device, resolved_dtype

    @staticmethod
    def _validate_config(config: FixedRecentVisualMemoryConfig) -> None:
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}")
        if config.tokens_per_observation <= 0:
            raise ValueError(f"tokens_per_observation must be positive, got {config.tokens_per_observation}")
        if not config.offsets:
            raise ValueError("offsets must contain at least one value")
        if any(int(offset) <= 0 for offset in config.offsets):
            raise ValueError(f"offsets must be positive, got {config.offsets}")
        if config.compressor not in {"bottleneck_se", "query_cross_attention"}:
            raise ValueError("compressor must be one of 'bottleneck_se' or 'query_cross_attention'")
        if config.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {config.num_heads}")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(f"hidden_dim {config.hidden_dim} must be divisible by num_heads {config.num_heads}")
        if config.bottleneck_ratio <= 0:
            raise ValueError(f"bottleneck_ratio must be positive, got {config.bottleneck_ratio}")


class PerceptualVisualMemoryBank(nn.Module):
    """Episode-level perceptual memory with retrieval, gate fusion, and compact consolidation."""

    def __init__(self, config: PerceptualVisualMemoryConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword arguments, not both")
        self.config = config or PerceptualVisualMemoryConfig(**kwargs)
        self._validate_config(self.config)

        self.compressor = PerceptualTokenCompressor(
            hidden_dim=self.config.hidden_dim,
            memory_tokens=self.config.memory_tokens,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.timestep_embedding = SinusoidalTimestepEmbedding(self.config.hidden_dim)
        self.retrieval_layers = nn.ModuleList(
            [
                MemoryVLACrossAttentionBlock(
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                    ffn_mult=self.config.ffn_mult,
                )
                for _ in range(self.config.retrieval_layers)
            ]
        )
        self.fusion = MemoryVLAGateFusion(self.config.hidden_dim)
        self._bank: dict[str, list[PerceptualMemoryEntry]] = {}

    def reset(self) -> None:
        self._bank.clear()

    def clear_episode(self, episode_id: str | int) -> None:
        self._bank.pop(str(episode_id), None)

    def entries(self, episode_id: str | int) -> tuple[PerceptualMemoryEntry, ...]:
        return tuple(self._bank.get(str(episode_id), ()))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        *,
        visual_token_mask: torch.Tensor | None = None,
        episode_ids: Sequence[str | int] | None = None,
        timesteps: Sequence[int] | torch.Tensor | None = None,
        update: bool = True,
    ) -> PerceptualMemoryOutput:
        current_tokens = self.compressor(visual_tokens, visual_token_mask=visual_token_mask)
        return self.process_compressed(
            current_tokens,
            episode_ids=episode_ids,
            timesteps=timesteps,
            update=update,
        )

    def process_compressed(
        self,
        current_tokens: torch.Tensor,
        *,
        episode_ids: Sequence[str | int] | None = None,
        timesteps: Sequence[int] | torch.Tensor | None = None,
        update: bool = True,
    ) -> PerceptualMemoryOutput:
        current_tokens = _ensure_rank3(current_tokens, "current_tokens")
        if current_tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"current_tokens dim {current_tokens.shape[-1]} != hidden_dim {self.config.hidden_dim}")
        batch_size, token_count, _ = current_tokens.shape
        if token_count != self.config.memory_tokens:
            raise ValueError(
                f"current_tokens length {token_count} != configured memory_tokens {self.config.memory_tokens}"
            )
        normalized_ids = _normalize_episode_ids(episode_ids, batch_size)
        normalized_timesteps = self._normalize_timesteps(timesteps, normalized_ids)

        fused_parts: list[torch.Tensor] = []
        retrieved_parts: list[torch.Tensor] = []
        gate_parts: list[torch.Tensor] = []
        for index, (episode_id, timestep) in enumerate(zip(normalized_ids, normalized_timesteps, strict=True)):
            current = current_tokens[index : index + 1]
            retrieved = self._retrieve_one(current, episode_id)
            if retrieved is None:
                fused = current
                retrieved = torch.zeros_like(current)
                gate = torch.ones_like(current)
            else:
                fused, gate = self.fusion(current, retrieved)
            fused_parts.append(fused)
            retrieved_parts.append(retrieved)
            gate_parts.append(gate)
            if update:
                self._append_entry(episode_id, fused if self.config.store_fused else current, timestep)

        tokens = torch.cat(fused_parts, dim=0)
        retrieved_tokens = torch.cat(retrieved_parts, dim=0)
        gate = torch.cat(gate_parts, dim=0)
        mask = torch.ones(batch_size, token_count, dtype=torch.bool, device=current_tokens.device)
        return PerceptualMemoryOutput(
            current_tokens=current_tokens,
            retrieved_tokens=retrieved_tokens,
            tokens=tokens,
            mask=mask,
            gate=gate,
            episode_ids=normalized_ids,
            timesteps=normalized_timesteps,
        )

    def _retrieve_one(self, current: torch.Tensor, episode_id: str) -> torch.Tensor | None:
        entries = self._bank.get(episode_id, [])
        if not entries:
            return None
        memory_tokens = torch.cat([entry.tokens for entry in entries], dim=0).to(
            device=current.device,
            dtype=current.dtype,
        )
        timestep_values = [
            entry.timestep
            for entry in entries
            for _ in range(entry.tokens.shape[0])
        ]
        timestep_tensor = torch.tensor(timestep_values, dtype=torch.long, device=current.device).unsqueeze(0)
        memory = memory_tokens.unsqueeze(0)
        memory = memory + self.timestep_embedding(timestep_tensor, dtype=current.dtype)
        memory_mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=current.device)
        retrieved = current
        for layer in self.retrieval_layers:
            retrieved = layer(retrieved, memory, memory_mask)
        return retrieved

    def _append_entry(self, episode_id: str, tokens: torch.Tensor, timestep: int) -> None:
        tokens = tokens.squeeze(0)
        if self.config.detach_on_update:
            tokens = tokens.detach()
        tokens = tokens.clone()
        self._bank.setdefault(episode_id, []).append(PerceptualMemoryEntry(tokens=tokens, timestep=timestep))
        self._consolidate_episode(episode_id)

    def _consolidate_episode(self, episode_id: str) -> None:
        entries = self._bank.get(episode_id, [])
        capacity = self.config.capacity
        if capacity == 0:
            self._bank[episode_id] = []
            return
        while len(entries) > capacity:
            if self.config.consolidation == "fifo":
                entries.pop(0)
            else:
                merge_index = _most_similar_adjacent_pair(entries)
                first = entries[merge_index]
                second = entries[merge_index + 1]
                merged = PerceptualMemoryEntry(
                    tokens=0.5 * (first.tokens + second.tokens),
                    timestep=second.timestep,
                )
                entries[merge_index : merge_index + 2] = [merged]
        self._bank[episode_id] = entries

    def _normalize_timesteps(
        self,
        timesteps: Sequence[int] | torch.Tensor | None,
        episode_ids: tuple[str, ...],
    ) -> tuple[int, ...]:
        if timesteps is None:
            values = []
            for episode_id in episode_ids:
                entries = self._bank.get(episode_id, [])
                values.append(0 if not entries else entries[-1].timestep + 1)
            return tuple(values)
        if isinstance(timesteps, torch.Tensor):
            values = [int(value) for value in timesteps.detach().cpu().reshape(-1).tolist()]
        else:
            values = [int(value) for value in timesteps]
        if len(values) != len(episode_ids):
            raise ValueError(f"timesteps has {len(values)} values for batch size {len(episode_ids)}")
        if any(value < 0 for value in values):
            raise ValueError(f"timesteps must be non-negative, got {values}")
        return tuple(values)

    @staticmethod
    def _validate_config(config: PerceptualVisualMemoryConfig) -> None:
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}")
        if config.memory_tokens <= 0:
            raise ValueError(f"memory_tokens must be positive, got {config.memory_tokens}")
        if config.capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {config.capacity}")
        if config.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {config.num_heads}")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(f"hidden_dim {config.hidden_dim} must be divisible by num_heads {config.num_heads}")
        if config.retrieval_layers < 0:
            raise ValueError(f"retrieval_layers must be non-negative, got {config.retrieval_layers}")
        if config.ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be positive, got {config.ffn_mult}")
        if config.consolidation not in {"fifo", "merge", "tome"}:
            raise ValueError("consolidation must be one of 'fifo', 'merge', or 'tome'")


def _normalize_episode_ids(episode_ids: Sequence[str | int] | None, batch_size: int) -> tuple[str, ...]:
    if episode_ids is None:
        return tuple(f"default:{index}" for index in range(batch_size))
    values = tuple(str(value) for value in episode_ids)
    if len(values) != batch_size:
        raise ValueError(f"episode_ids has {len(values)} values for batch size {batch_size}")
    return values


def _get_offset_value(
    values: Sequence[torch.Tensor | None] | dict[int, torch.Tensor | None] | None,
    *,
    offset: int,
    index: int,
) -> torch.Tensor | None:
    if values is None:
        return None
    if isinstance(values, dict):
        return values.get(int(offset))
    if index >= len(values):
        raise ValueError(f"expected value for offset index {index}, got only {len(values)} values")
    return values[index]


def _most_similar_adjacent_pair(entries: Sequence[PerceptualMemoryEntry]) -> int:
    if len(entries) < 2:
        raise ValueError("at least two entries are required")
    pooled = torch.stack([entry.tokens.float().mean(dim=0) for entry in entries], dim=0)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    similarities = (pooled[:-1] * pooled[1:]).sum(dim=-1)
    return int(torch.argmax(similarities).item())


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor

