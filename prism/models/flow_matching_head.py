from __future__ import annotations

# --- migrated from src/prism/model/action_head/flow_matching.py ---
import math
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len: int):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        device = self.pe.device
        dtype = self.pe.dtype
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=dtype, device=device) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim, dtype=dtype, device=device)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        self.pe = torch.cat([self.pe, extra_pe.unsqueeze(0)], dim=1)


class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.randn(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.randn(num_categories, out_dim))

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        if self.num_categories <= 1:
            return self.linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        category_id = _expand_category_ids(category_id, x_flat.shape[0], x.device)
        weight_selected = self.weight[category_id]
        bias_selected = self.bias[category_id]
        out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)


class CategorySpecificMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        return self.fc2(out, category_id)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):
        batch_size, horizon, action_dim = action_seq.shape
        if horizon != self.horizon:
            raise ValueError(f"Action sequence length {horizon} must match horizon {self.horizon}")

        x = action_seq.reshape(batch_size * horizon, action_dim)
        cat_ids = _repeat_batch_categories(category_id, batch_size, horizon, action_seq.device)
        out = self.activation(self.W1(x, cat_ids))
        pos_enc = self.pos_encoding(horizon).to(device=out.device, dtype=out.dtype)
        pos_enc = pos_enc.repeat(batch_size, 1, 1).reshape(batch_size * horizon, -1)
        out = out + pos_enc
        out = self.activation(self.W2(out, cat_ids))
        out = self.W3(out, cat_ids)
        return out.view(batch_size, horizon, self.embed_dim)


class MultiEmbodimentActionDecoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, action_dim: int, num_categories: int = 1):
        super().__init__()
        self.W1 = CategorySpecificLinear(embed_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, action_dim, num_categories)
        self.activation = nn.GELU()

    def forward(self, action_tokens: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.W1(action_tokens, category_id))
        out = self.activation(self.W2(out, category_id))
        return self.W3(out, category_id)


class DirectBridgeActionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
        visual_gate_lambda: float = 0.5,
    ):
        super().__init__()
        self.visual_gate_lambda = float(visual_gate_lambda)
        self.visual_gate_logit = nn.Parameter(torch.zeros(()))

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.visual_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.action_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_self = nn.LayerNorm(embed_dim)
        self.norm_visual = nn.LayerNorm(embed_dim)
        self.norm_action = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        visual_context: torch.Tensor,
        action_context: torch.Tensor,
        *,
        visual_key_padding_mask: torch.Tensor | None = None,
        action_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_self = self.norm_self(action_tokens)
        self_out, _ = self.self_attn(x_self, x_self, x_self, need_weights=False)
        x = action_tokens + self_out

        x_visual = self.norm_visual(x)
        visual_out, _ = self.visual_attn(
            x_visual,
            visual_context,
            visual_context,
            key_padding_mask=visual_key_padding_mask,
            need_weights=False,
        )

        x_action = self.norm_action(x)
        action_out, _ = self.action_attn(
            x_action,
            action_context,
            action_context,
            key_padding_mask=action_key_padding_mask,
            need_weights=False,
        )

        visual_scale = 1.0 + self.visual_gate_lambda * torch.tanh(self.visual_gate_logit)
        x = x + visual_scale.to(dtype=x.dtype, device=x.device) * visual_out + action_out
        return x + self.ffn(self.norm_ffn(x))


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        config=None,
        embed_dim: int = 896,
        hidden_dim: int = 1024,
        action_dim: int = 32 * 7,
        horizon: int = 32,
        per_action_dim: int = 7,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.0,
        num_inference_timesteps: int = 15,
        num_categories: int = 1,
        short_memory_time_bins: int = 2,
    ):
        super().__init__()

        if config is not None:
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            per_action_dim = getattr(config, "per_action_dim", per_action_dim)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            short_memory_time_bins = getattr(config, "short_memory_time_bins", short_memory_time_bins)
            self.config = config
        else:
            self.config = SimpleNamespace(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                num_inference_timesteps=num_inference_timesteps,
                inference_tau_schedule="midpoint",
                avoid_endpoint_tau=True,
                num_categories=num_categories,
                short_memory_time_bins=short_memory_time_bins,
            )

        if action_dim != horizon * per_action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) must equal horizon ({horizon}) * per_action_dim ({per_action_dim})"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.num_plan_slots = int(getattr(self.config, "num_plan_slots", 8))
        self.plan_gate_lambda = float(getattr(self.config, "plan_gate_lambda", 0.25))
        self.inference_tau_schedule = str(getattr(self.config, "inference_tau_schedule", "midpoint")).lower()
        self.avoid_endpoint_tau = bool(getattr(self.config, "avoid_endpoint_tau", True))
        self.max_vlm_tokens = getattr(self.config, "max_vlm_tokens", None)
        if self.inference_tau_schedule != "midpoint":
            raise ValueError("FlowmatchingActionHead currently supports only midpoint inference_tau_schedule")
        if not self.avoid_endpoint_tau:
            raise ValueError("FlowmatchingActionHead requires avoid_endpoint_tau=True for midpoint inference")

        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=per_action_dim,
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            horizon=horizon,
            num_categories=num_categories,
        )
        self.action_decoder = MultiEmbodimentActionDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            action_dim=per_action_dim,
            num_categories=num_categories,
        )

        self.vlm_norm = nn.LayerNorm(embed_dim)
        self.short_memory_norm = nn.LayerNorm(embed_dim)
        self.short_memory_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.short_memory_adapter_gamma = nn.Parameter(torch.tensor(0.1))
        time_bins = int(getattr(self.config, "short_memory_time_bins", 2))
        if time_bins <= 0:
            raise ValueError("short_memory_time_bins must be positive")
        self.short_memory_time_embedding = nn.Embedding(time_bins, embed_dim)

        self.state_encoder = None
        self.null_state_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
            self.state_encoder = CategorySpecificMLP(
                input_dim=self.config.state_dim,
                hidden_dim=state_hidden,
                output_dim=embed_dim,
                num_categories=num_categories,
            )

        self.plan_adapter = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.plan_slot_embeddings = nn.Parameter(torch.empty(self.num_plan_slots, embed_dim))
        nn.init.normal_(self.plan_slot_embeddings, mean=0.0, std=0.02)
        self.plan_gate_logits = nn.Parameter(torch.zeros(num_layers))

        self.vlm_src_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mem_src_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.plan_src_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.state_src_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))

        ffn_dim = int(getattr(self.config, "ffn_dim", embed_dim * 4))
        visual_gate_lambda = float(getattr(self.config, "visual_gate_lambda", 0.5))
        self.transformer_blocks = nn.ModuleList(
            [
                DirectBridgeActionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=ffn_dim,
                    dropout=dropout,
                    visual_gate_lambda=visual_gate_lambda,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        actions_gt: torch.Tensor = None,
        embodiment_id: torch.LongTensor = None,
        action_mask: torch.Tensor = None,
        vlm_hidden_states: Sequence[torch.Tensor] | None = None,
        short_memory_tokens: torch.Tensor | None = None,
        short_memory_time_ids: torch.Tensor | None = None,
        short_memory_mask: torch.Tensor | None = None,
        plan_tokens: torch.Tensor | None = None,
        plan_token_mask: torch.Tensor | None = None,
    ):
        if actions_gt is None:
            return self.get_action(
                fused_tokens,
                state=state,
                embodiment_id=embodiment_id,
                action_mask=action_mask,
                vlm_hidden_states=vlm_hidden_states,
                short_memory_tokens=short_memory_tokens,
                short_memory_time_ids=short_memory_time_ids,
                short_memory_mask=short_memory_mask,
                plan_tokens=plan_tokens,
                plan_token_mask=plan_token_mask,
            )

        fused_tokens = _ensure_rank3(fused_tokens, "fused_tokens")
        batch_size = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        t = torch.distributions.Beta(2, 2).sample((batch_size,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)
        time_emb = self._time_embedding(t, device=device, dtype=fused_tokens.dtype)
        noise = torch.rand_like(actions_gt) * 2 - 1

        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            if action_mask.shape != noise.shape:
                raise ValueError(f"action_mask shape {action_mask.shape} != noise shape {noise.shape}")
            noise = noise * action_mask

        noise_seq = noise.view(batch_size, self.horizon, self.per_action_dim)
        actions_gt_seq = actions_gt.view(batch_size, self.horizon, self.per_action_dim)
        action_intermediate_seq = (1 - t.view(batch_size, 1, 1)) * noise_seq + t.view(batch_size, 1, 1) * actions_gt_seq

        pred_velocity = self._predict_velocity(
            action_intermediate_seq,
            fused_tokens=fused_tokens,
            state=state,
            embodiment_id=embodiment_id,
            time_emb=time_emb,
            vlm_hidden_states=vlm_hidden_states,
            short_memory_tokens=short_memory_tokens,
            short_memory_time_ids=short_memory_time_ids,
            short_memory_mask=short_memory_mask,
            plan_tokens=plan_tokens,
            plan_token_mask=plan_token_mask,
        )
        return pred_velocity, noise

    def get_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_id: torch.LongTensor = None,
        action_mask: torch.Tensor = None,
        vlm_hidden_states: Sequence[torch.Tensor] | None = None,
        short_memory_tokens: torch.Tensor | None = None,
        short_memory_time_ids: torch.Tensor | None = None,
        short_memory_mask: torch.Tensor | None = None,
        plan_tokens: torch.Tensor | None = None,
        plan_token_mask: torch.Tensor | None = None,
    ):
        fused_tokens = _ensure_rank3(fused_tokens, "fused_tokens")
        batch_size = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        if action_mask is None:
            raise ValueError("action_mask must be provided for inference with flow matching.")

        action = torch.rand(batch_size, self.action_dim, device=device, dtype=fused_tokens.dtype) * 2 - 1
        action_seq = action.view(batch_size, self.horizon, self.per_action_dim)
        action_mask = action_mask.to(dtype=action_seq.dtype, device=action_seq.device)
        if action_mask.shape == (batch_size, self.per_action_dim):
            action_mask = action_mask.view(batch_size, 1, self.per_action_dim).repeat(1, self.horizon, 1)
        elif action_mask.shape != action_seq.shape:
            raise ValueError(f"action_mask shape {action_mask.shape} != action sequence shape {action_seq.shape}")
        action_seq = action_seq * action_mask

        num_steps = int(getattr(self.config, "num_inference_timesteps", 15))
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), (i + 0.5) / num_steps, device=device, dtype=fused_tokens.dtype)
            time_emb = self._time_embedding(t, device=device, dtype=fused_tokens.dtype)
            pred = self._predict_velocity(
                action_seq,
                fused_tokens=fused_tokens,
                state=state,
                embodiment_id=embodiment_id,
                time_emb=time_emb,
                vlm_hidden_states=vlm_hidden_states,
                short_memory_tokens=short_memory_tokens,
                short_memory_time_ids=short_memory_time_ids,
                short_memory_mask=short_memory_mask,
                plan_tokens=plan_tokens,
                plan_token_mask=plan_token_mask,
            )
            action = action_seq.reshape(batch_size, self.action_dim) + dt * pred
            action_seq = action.view(batch_size, self.horizon, self.per_action_dim) * action_mask

        return action_seq.reshape(batch_size, self.action_dim)

    def _predict_velocity(
        self,
        action_seq: torch.Tensor,
        *,
        fused_tokens: torch.Tensor,
        state: torch.Tensor | None,
        embodiment_id: torch.LongTensor,
        time_emb: torch.Tensor,
        vlm_hidden_states: Sequence[torch.Tensor] | None,
        short_memory_tokens: torch.Tensor | None,
        short_memory_time_ids: torch.Tensor | None,
        short_memory_mask: torch.Tensor | None,
        plan_tokens: torch.Tensor | None,
        plan_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        action_tokens = self.action_encoder(action_seq, embodiment_id)
        x = action_tokens + time_emb.unsqueeze(1)

        for layer_index, block in enumerate(self.transformer_blocks):
            visual_context, visual_mask = self._build_visual_context(
                fused_tokens=fused_tokens,
                vlm_hidden_states=vlm_hidden_states,
                layer_index=layer_index,
                short_memory_tokens=short_memory_tokens,
                short_memory_time_ids=short_memory_time_ids,
                short_memory_mask=short_memory_mask,
            )
            action_context, action_context_mask = self._build_action_context(
                state=state,
                embodiment_id=embodiment_id,
                batch_size=action_seq.shape[0],
                device=action_seq.device,
                dtype=action_seq.dtype,
                plan_tokens=plan_tokens,
                plan_token_mask=plan_token_mask,
                layer_index=layer_index,
            )
            x = block(
                x,
                visual_context,
                action_context,
                visual_key_padding_mask=visual_mask,
                action_key_padding_mask=action_context_mask,
            )

        decoded = self.action_decoder(self.norm_out(x), embodiment_id)
        return decoded.reshape(action_seq.shape[0], self.action_dim)

    def _build_visual_context(
        self,
        *,
        fused_tokens: torch.Tensor,
        vlm_hidden_states: Sequence[torch.Tensor] | None,
        layer_index: int,
        short_memory_tokens: torch.Tensor | None,
        short_memory_time_ids: torch.Tensor | None,
        short_memory_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        vlm_tokens = self._select_vlm_tokens(fused_tokens, vlm_hidden_states, layer_index)
        vlm_tokens = self.vlm_norm(vlm_tokens) + self.vlm_src_emb.to(device=vlm_tokens.device, dtype=vlm_tokens.dtype)
        masks = [torch.zeros(vlm_tokens.shape[:2], dtype=torch.bool, device=vlm_tokens.device)]
        contexts = [vlm_tokens]

        if short_memory_tokens is not None:
            memory_tokens = _ensure_rank3(short_memory_tokens, "short_memory_tokens").to(
                device=vlm_tokens.device,
                dtype=vlm_tokens.dtype,
            )
            if memory_tokens.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"short_memory_tokens last dimension {memory_tokens.shape[-1]} != embed_dim {self.embed_dim}"
                )
            adapted = memory_tokens + self.short_memory_adapter_gamma.to(memory_tokens.dtype) * self.short_memory_adapter(
                self.short_memory_norm(memory_tokens)
            )
            time_ids = self._prepare_short_memory_time_ids(
                short_memory_time_ids,
                batch_size=memory_tokens.shape[0],
                token_count=memory_tokens.shape[1],
                device=memory_tokens.device,
            )
            adapted = (
                adapted
                + self.mem_src_emb.to(device=memory_tokens.device, dtype=memory_tokens.dtype)
                + self.short_memory_time_embedding(time_ids).to(dtype=memory_tokens.dtype)
            )
            contexts.append(adapted)
            masks.append(_valid_mask_to_padding_mask(short_memory_mask, memory_tokens.shape[:2], memory_tokens.device))

        visual_context = torch.cat(contexts, dim=1)
        visual_mask = torch.cat(masks, dim=1)
        return visual_context, visual_mask if visual_mask.any() else None

    def _build_action_context(
        self,
        *,
        state: torch.Tensor | None,
        embodiment_id: torch.LongTensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        plan_tokens: torch.Tensor | None,
        plan_token_mask: torch.Tensor | None,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state_token = self._encode_state_token(state, embodiment_id, batch_size, device, dtype)
        state_token = state_token + self.state_src_emb.to(device=device, dtype=dtype)

        contexts = []
        masks = []
        if plan_tokens is not None:
            plan_slots = self._expand_plan_slots(plan_tokens, plan_token_mask, device, dtype, layer_index)
            contexts.append(plan_slots)
            masks.append(torch.zeros(plan_slots.shape[:2], dtype=torch.bool, device=device))

        contexts.append(state_token)
        masks.append(torch.zeros(state_token.shape[:2], dtype=torch.bool, device=device))
        action_context = torch.cat(contexts, dim=1)
        action_mask = torch.cat(masks, dim=1)
        return action_context, action_mask if action_mask.any() else None

    def _select_vlm_tokens(
        self,
        fused_tokens: torch.Tensor,
        vlm_hidden_states: Sequence[torch.Tensor] | None,
        layer_index: int,
    ) -> torch.Tensor:
        if vlm_hidden_states is None or len(vlm_hidden_states) == 0:
            tokens = _ensure_rank3(fused_tokens, "fused_tokens")
        else:
            selected_index = min(
                len(vlm_hidden_states) - 1,
                (layer_index * len(vlm_hidden_states)) // max(1, self.num_layers),
            )
            tokens = _ensure_rank3(vlm_hidden_states[selected_index], "vlm_hidden_state")
        tokens = tokens.to(device=fused_tokens.device, dtype=fused_tokens.dtype)
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"VLM token dim {tokens.shape[-1]} != embed_dim {self.embed_dim}")
        if self.max_vlm_tokens is not None and tokens.shape[1] > int(self.max_vlm_tokens):
            tokens = tokens[:, : int(self.max_vlm_tokens), :]
        return tokens

    def _encode_state_token(
        self,
        state: torch.Tensor | None,
        embodiment_id: torch.LongTensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state is None or self.state_encoder is None:
            return self.null_state_token.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        if state.ndim == 3 and state.shape[1] == 1:
            state = state.squeeze(1)
        if state.ndim != 2:
            raise ValueError(f"state must have shape [B, state_dim] or [B, 1, state_dim], got {tuple(state.shape)}")
        encoded = self.state_encoder(state.to(device=device, dtype=dtype), embodiment_id)
        return encoded.unsqueeze(1)

    def _expand_plan_slots(
        self,
        plan_tokens: torch.Tensor,
        plan_token_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        layer_index: int,
    ) -> torch.Tensor:
        plan_tokens = _ensure_rank3(plan_tokens, "plan_tokens").to(device=device, dtype=dtype)
        if plan_tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"plan_tokens last dimension {plan_tokens.shape[-1]} != embed_dim {self.embed_dim}")
        adapted = self.plan_adapter(plan_tokens)
        if plan_token_mask is None:
            pooled = adapted.mean(dim=1, keepdim=True)
        else:
            valid = plan_token_mask.to(device=device).bool()
            if valid.shape != adapted.shape[:2]:
                raise ValueError(
                    f"plan_token_mask shape {tuple(valid.shape)} must match plan token prefix {tuple(adapted.shape[:2])}"
                )
            weights = valid.to(dtype=dtype).unsqueeze(-1)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (adapted * weights).sum(dim=1, keepdim=True) / denom
        slots = pooled + self.plan_slot_embeddings.to(device=device, dtype=dtype).unsqueeze(0)
        scale = 1.0 + self.plan_gate_lambda * torch.tanh(self.plan_gate_logits[layer_index])
        return scale.to(device=device, dtype=dtype) * slots + self.plan_src_emb.to(device=device, dtype=dtype)

    def _prepare_short_memory_time_ids(
        self,
        short_memory_time_ids: torch.Tensor | None,
        *,
        batch_size: int,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if short_memory_time_ids is None:
            return torch.zeros(batch_size, token_count, dtype=torch.long, device=device)
        time_ids = short_memory_time_ids.to(device=device).long()
        if time_ids.shape != (batch_size, token_count):
            raise ValueError(
                f"short_memory_time_ids shape {tuple(time_ids.shape)} must equal {(batch_size, token_count)}"
            )
        min_id = int(time_ids.min().item()) if time_ids.numel() else 0
        max_id = int(time_ids.max().item()) if time_ids.numel() else 0
        time_bins = int(self.short_memory_time_embedding.num_embeddings)
        if min_id < 0 or max_id >= time_bins:
            raise ValueError(
                f"short_memory_time_ids must be in [0, {time_bins - 1}], got min={min_id}, max={max_id}. "
                "Increase short_memory_time_bins or fix the short-memory packing config."
            )
        return time_ids

    def _time_embedding(self, t: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        time_index = (t * 1000).long().clamp(0, 999)
        time_table = self.time_pos_enc(1000).to(device=device, dtype=dtype)
        return time_table[0, time_index, :]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor


def _valid_mask_to_padding_mask(
    valid_mask: torch.Tensor | None,
    expected_shape: torch.Size | tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.zeros(expected_shape, dtype=torch.bool, device=device)
    valid_mask = valid_mask.to(device=device).bool()
    if tuple(valid_mask.shape) != tuple(expected_shape):
        raise ValueError(f"mask shape {tuple(valid_mask.shape)} must match {tuple(expected_shape)}")
    return ~valid_mask


def _expand_category_ids(category_id: torch.LongTensor, target_count: int, device: torch.device) -> torch.LongTensor:
    if category_id.dim() == 0:
        return category_id.to(device=device).repeat(target_count)
    category_id = category_id.to(device=device).view(-1)
    if category_id.numel() == target_count:
        return category_id
    if target_count % category_id.numel() != 0:
        raise ValueError(f"Cannot expand {category_id.numel()} category ids to {target_count} rows")
    repeat_count = target_count // category_id.numel()
    return category_id.unsqueeze(1).expand(-1, repeat_count).reshape(-1)


def _repeat_batch_categories(
    category_id: torch.LongTensor,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> torch.LongTensor:
    if category_id.dim() == 0:
        return category_id.to(device=device).repeat(batch_size * horizon)
    category_id = category_id.to(device=device).view(-1)
    if category_id.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} category ids, got {category_id.numel()}")
    return category_id.unsqueeze(1).expand(-1, horizon).reshape(-1)

