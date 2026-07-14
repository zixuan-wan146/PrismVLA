from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.models.config import TaskStatePlannerConfig


@dataclass(frozen=True)
class MambaStreamingCache:
    """Causal convolution and selective-SSM state for one planning cycle."""

    conv_state: torch.Tensor
    ssm_state: torch.Tensor

    def detached(self) -> "MambaStreamingCache":
        return MambaStreamingCache(
            conv_state=self.conv_state.detach(),
            ssm_state=self.ssm_state.detach(),
        )


@dataclass(frozen=True)
class TaskStatePlannerRuntimeState:
    """Session state persisted between planning cycles and cleared on reset."""

    task_state: torch.Tensor
    mamba_cache: MambaStreamingCache

    def detached(self) -> "TaskStatePlannerRuntimeState":
        return TaskStatePlannerRuntimeState(
            task_state=self.task_state.detach(),
            mamba_cache=self.mamba_cache.detached(),
        )


@dataclass(frozen=True)
class TaskStatePlanOutput:
    task_state: torch.Tensor
    plan_tokens: torch.Tensor
    runtime_state: TaskStatePlannerRuntimeState


class StreamingMambaStep(nn.Module):
    """A single canonical Mamba-1 selective state-space step.

    The caller flattens state slots into the batch dimension, so every slot owns
    independent convolution and SSM caches while all slots share these weights.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.inner_size = hidden_size * expand
        self.dt_rank = math.ceil(hidden_size / 16)

        self.in_projection = nn.Linear(hidden_size, 2 * self.inner_size)
        self.conv1d = nn.Conv1d(
            self.inner_size,
            self.inner_size,
            kernel_size=d_conv,
            groups=self.inner_size,
            padding=0,
        )
        self.parameter_projection = nn.Linear(
            self.inner_size,
            self.dt_rank + 2 * d_state,
            bias=False,
        )
        self.dt_projection = nn.Linear(self.dt_rank, self.inner_size, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(self.inner_size, -1)
            .contiguous()
        )
        self.D = nn.Parameter(torch.ones(self.inner_size, dtype=torch.float32))
        self.out_projection = nn.Linear(self.inner_size, hidden_size)
        self._reset_mamba_parameters()

    def _reset_mamba_parameters(self) -> None:
        dt_scale = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_projection.weight, -dt_scale, dt_scale)
        dt = torch.exp(
            torch.rand(self.inner_size)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projection.bias.copy_(inverse_softplus)

    def initial_cache(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MambaStreamingCache:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return MambaStreamingCache(
            conv_state=torch.zeros(
                batch_size,
                self.inner_size,
                self.d_conv,
                device=device,
                dtype=dtype,
            ),
            ssm_state=torch.zeros(
                batch_size,
                self.inner_size,
                self.d_state,
                device=device,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: MambaStreamingCache | None,
    ) -> tuple[torch.Tensor, MambaStreamingCache]:
        if hidden_states.ndim != 3 or tuple(hidden_states.shape[1:]) != (1, self.hidden_size):
            raise ValueError(
                "Mamba input must have one explicit planning-cycle step with shape "
                f"[N, 1, {self.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        if not torch.is_floating_point(hidden_states):
            raise TypeError("Mamba input must be floating point")

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states[:, 0, :]
        parameter_dtype = self.in_projection.weight.dtype
        hidden_states = hidden_states.to(
            device=self.in_projection.weight.device,
            dtype=parameter_dtype,
        )
        if cache is None:
            cache = self.initial_cache(
                batch_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        self._validate_cache(cache, batch_size=batch_size, device=hidden_states.device)

        projected, gate = self.in_projection(hidden_states).chunk(2, dim=-1)
        conv_state = cache.conv_state.to(device=hidden_states.device, dtype=projected.dtype)
        next_conv_state = torch.cat((conv_state[..., 1:], projected.unsqueeze(-1)), dim=-1)
        conv_weights = self.conv1d.weight[:, 0, :]
        convolved = torch.sum(next_conv_state * conv_weights.unsqueeze(0), dim=-1)
        if self.conv1d.bias is not None:
            convolved = convolved + self.conv1d.bias
        convolved = F.silu(convolved)

        parameters = self.parameter_projection(convolved)
        dt_low_rank, B, C = torch.split(
            parameters,
            (self.dt_rank, self.d_state, self.d_state),
            dim=-1,
        )
        dt = F.softplus(self.dt_projection(dt_low_rank).float())
        A = -torch.exp(self.A_log.float())
        previous_ssm_state = cache.ssm_state.to(device=hidden_states.device, dtype=torch.float32)
        discrete_A = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        discrete_B_input = (
            dt.unsqueeze(-1)
            * B.float().unsqueeze(1)
            * convolved.float().unsqueeze(-1)
        )
        next_ssm_state = previous_ssm_state * discrete_A + discrete_B_input
        scanned = torch.sum(next_ssm_state * C.float().unsqueeze(1), dim=-1)
        scanned = scanned + self.D.float().unsqueeze(0) * convolved.float()
        scanned = scanned.to(dtype=gate.dtype) * F.silu(gate)
        output = self.out_projection(scanned)
        return output.unsqueeze(1), MambaStreamingCache(
            conv_state=next_conv_state,
            ssm_state=next_ssm_state,
        )

    def _validate_cache(
        self,
        cache: MambaStreamingCache,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if not isinstance(cache, MambaStreamingCache):
            raise TypeError(f"cache must be MambaStreamingCache, got {type(cache).__name__}")
        expected_conv = (batch_size, self.inner_size, self.d_conv)
        expected_ssm = (batch_size, self.inner_size, self.d_state)
        if tuple(cache.conv_state.shape) != expected_conv:
            raise ValueError(
                f"Mamba convolution cache must have shape {expected_conv}, "
                f"got {tuple(cache.conv_state.shape)}"
            )
        if tuple(cache.ssm_state.shape) != expected_ssm:
            raise ValueError(
                f"Mamba SSM cache must have shape {expected_ssm}, got {tuple(cache.ssm_state.shape)}"
            )
        if not torch.is_floating_point(cache.conv_state) or not torch.is_floating_point(cache.ssm_state):
            raise TypeError("Mamba cache tensors must be floating point")
        if cache.conv_state.device != device or cache.ssm_state.device != device:
            raise ValueError("Mamba cache tensors must be on the same device as the current state")


class ResidualMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.layers(self.norm(hidden_states))


class TaskStatePlanPipeline(nn.Module):
    """Update eight persistent task-state tokens and read sixteen plan tokens."""

    def __init__(self, config: TaskStatePlannerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        # This is deliberately the only Q12 projection used by both branches.
        self.shared_query_projection = nn.Sequential(
            nn.Linear(config.query_input_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )
        self.executed_action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.action_mlp_hidden_size),
            nn.GELU(),
            nn.Linear(config.action_mlp_hidden_size, config.hidden_size),
        )
        self.execution_position_embeddings = nn.Parameter(
            torch.empty(config.action_horizon, config.hidden_size)
        )
        self.executed_action_norm = nn.LayerNorm(config.hidden_size)
        self.initial_state_tokens = nn.Parameter(
            torch.empty(config.num_state_tokens, config.hidden_size)
        )

        self.state_query_norm = nn.LayerNorm(config.hidden_size)
        self.state_cross_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.state_self_norm = nn.LayerNorm(config.hidden_size)
        self.state_self_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.state_mamba_norm = nn.LayerNorm(config.hidden_size)
        self.temporal_mamba = StreamingMambaStep(
            hidden_size=config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )

        self.plan_queries = nn.Parameter(
            torch.empty(config.num_plan_tokens, config.hidden_size)
        )
        self.planner_state_norm = nn.LayerNorm(config.hidden_size)
        self.plan_reader_norm = nn.LayerNorm(config.hidden_size)
        self.plan_reader_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.plan_reader_mlp = ResidualMLP(
            config.hidden_size,
            config.mlp_hidden_size,
            config.mlp_dropout,
        )
        self.plan_mixer_norm = nn.LayerNorm(config.hidden_size)
        self.plan_mixer_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.plan_mixer_mlp = ResidualMLP(
            config.hidden_size,
            config.mlp_hidden_size,
            config.mlp_dropout,
        )
        self.plan_output_norm = nn.LayerNorm(config.hidden_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.execution_position_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.initial_state_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.plan_queries, mean=0.0, std=0.02)

    def forward(
        self,
        query_layer12: torch.Tensor,
        executed_actions: torch.Tensor,
        executed_action_valid_mask: torch.Tensor,
        *,
        query_valid_mask: torch.Tensor | None = None,
        previous_state: TaskStatePlannerRuntimeState | None = None,
    ) -> TaskStatePlanOutput:
        batch_size = self._validate_inputs(
            query_layer12,
            executed_actions,
            executed_action_valid_mask,
            query_valid_mask=query_valid_mask,
            previous_state=previous_state,
        )
        device = self.shared_query_projection[0].weight.device
        dtype = self.shared_query_projection[0].weight.dtype
        query_layer12 = query_layer12.to(device=device, dtype=dtype)
        executed_actions = executed_actions.to(device=device, dtype=dtype)
        executed_action_valid_mask = executed_action_valid_mask.to(device=device)
        if query_valid_mask is None:
            query_valid_mask = torch.ones(
                batch_size,
                self.config.num_query_tokens,
                dtype=torch.bool,
                device=device,
            )
        else:
            query_valid_mask = query_valid_mask.to(device=device)

        projected_queries = self.shared_query_projection(query_layer12)
        action_features = self.executed_action_encoder(executed_actions)
        action_features = self.executed_action_norm(
            action_features + self.execution_position_embeddings.unsqueeze(0)
        )
        update_context = torch.cat((projected_queries, action_features), dim=1)
        context_valid_mask = torch.cat(
            (query_valid_mask, executed_action_valid_mask),
            dim=1,
        )

        if previous_state is None:
            state_tokens = self.initial_state_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            mamba_cache = None
        else:
            state_tokens = previous_state.task_state.to(device=device, dtype=dtype)
            mamba_cache = self._flatten_cache(previous_state.mamba_cache)

        normalized_state = self.state_query_norm(state_tokens)
        state_update, _ = self.state_cross_attention(
            normalized_state,
            update_context,
            update_context,
            key_padding_mask=~context_valid_mask,
            need_weights=False,
        )
        state_tokens = state_tokens + state_update
        normalized_state = self.state_self_norm(state_tokens)
        state_update, _ = self.state_self_attention(
            normalized_state,
            normalized_state,
            normalized_state,
            need_weights=False,
        )
        state_tokens = state_tokens + state_update

        mamba_input = self.state_mamba_norm(state_tokens).reshape(
            batch_size * self.config.num_state_tokens,
            1,
            self.config.hidden_size,
        )
        mamba_output, next_flat_cache = self.temporal_mamba(mamba_input, mamba_cache)
        state_tokens = state_tokens + mamba_output.reshape(
            batch_size,
            self.config.num_state_tokens,
            self.config.hidden_size,
        )

        planner_context = torch.cat(
            (self.planner_state_norm(state_tokens), projected_queries),
            dim=1,
        )
        planner_context_valid = torch.cat(
            (
                torch.ones(
                    batch_size,
                    self.config.num_state_tokens,
                    dtype=torch.bool,
                    device=device,
                ),
                query_valid_mask,
            ),
            dim=1,
        )
        plan_tokens = self.plan_queries.unsqueeze(0).expand(batch_size, -1, -1)
        plan_update, _ = self.plan_reader_attention(
            self.plan_reader_norm(plan_tokens),
            planner_context,
            planner_context,
            key_padding_mask=~planner_context_valid,
            need_weights=False,
        )
        plan_tokens = self.plan_reader_mlp(plan_tokens + plan_update)
        normalized_plan = self.plan_mixer_norm(plan_tokens)
        plan_update, _ = self.plan_mixer_attention(
            normalized_plan,
            normalized_plan,
            normalized_plan,
            need_weights=False,
        )
        plan_tokens = self.plan_mixer_mlp(plan_tokens + plan_update)
        plan_tokens = self.plan_output_norm(plan_tokens)

        next_cache = self._unflatten_cache(next_flat_cache, batch_size=batch_size)
        runtime_state = TaskStatePlannerRuntimeState(
            task_state=state_tokens,
            mamba_cache=next_cache,
        )
        return TaskStatePlanOutput(
            task_state=state_tokens,
            plan_tokens=plan_tokens,
            runtime_state=runtime_state,
        )

    def _validate_inputs(
        self,
        query_layer12: torch.Tensor,
        executed_actions: torch.Tensor,
        executed_action_valid_mask: torch.Tensor,
        *,
        query_valid_mask: torch.Tensor | None,
        previous_state: TaskStatePlannerRuntimeState | None,
    ) -> int:
        expected_query_tail = (self.config.num_query_tokens, self.config.query_input_dim)
        if query_layer12.ndim != 3 or tuple(query_layer12.shape[1:]) != expected_query_tail:
            raise ValueError(
                f"Q12 must have shape [B, {expected_query_tail[0]}, {expected_query_tail[1]}], "
                f"got {tuple(query_layer12.shape)}"
            )
        if not torch.is_floating_point(query_layer12):
            raise TypeError("Q12 must be floating point")
        batch_size = query_layer12.shape[0]
        expected_actions = (batch_size, self.config.action_horizon, self.config.action_dim)
        if tuple(executed_actions.shape) != expected_actions or not torch.is_floating_point(executed_actions):
            raise ValueError(
                f"executed_actions must be floating with shape {expected_actions}, "
                f"got {tuple(executed_actions.shape)}"
            )
        expected_action_mask = (batch_size, self.config.action_horizon)
        if (
            executed_action_valid_mask.dtype != torch.bool
            or tuple(executed_action_valid_mask.shape) != expected_action_mask
        ):
            raise ValueError(
                f"executed_action_valid_mask must be boolean with shape {expected_action_mask}"
            )
        if query_valid_mask is not None and (
            query_valid_mask.dtype != torch.bool
            or tuple(query_valid_mask.shape) != (batch_size, self.config.num_query_tokens)
        ):
            raise ValueError(
                "query_valid_mask must be boolean with shape "
                f"[{batch_size}, {self.config.num_query_tokens}]"
            )
        if previous_state is not None:
            if not isinstance(previous_state, TaskStatePlannerRuntimeState):
                raise TypeError(
                    "previous_state must be TaskStatePlannerRuntimeState or None, "
                    f"got {type(previous_state).__name__}"
                )
            expected_state = (
                batch_size,
                self.config.num_state_tokens,
                self.config.hidden_size,
            )
            if tuple(previous_state.task_state.shape) != expected_state:
                raise ValueError(
                    f"previous task state must have shape {expected_state}, "
                    f"got {tuple(previous_state.task_state.shape)}"
                )
            if not torch.is_floating_point(previous_state.task_state):
                raise TypeError("previous task state must be floating point")
            self._validate_structured_cache(previous_state.mamba_cache, batch_size=batch_size)
        return batch_size

    def _validate_structured_cache(
        self,
        cache: MambaStreamingCache,
        *,
        batch_size: int,
    ) -> None:
        expected_conv = (
            batch_size,
            self.config.num_state_tokens,
            self.temporal_mamba.inner_size,
            self.config.mamba_d_conv,
        )
        expected_ssm = (
            batch_size,
            self.config.num_state_tokens,
            self.temporal_mamba.inner_size,
            self.config.mamba_d_state,
        )
        if not isinstance(cache, MambaStreamingCache):
            raise TypeError(f"mamba_cache must be MambaStreamingCache, got {type(cache).__name__}")
        if tuple(cache.conv_state.shape) != expected_conv:
            raise ValueError(
                f"structured Mamba convolution cache must have shape {expected_conv}, "
                f"got {tuple(cache.conv_state.shape)}"
            )
        if tuple(cache.ssm_state.shape) != expected_ssm:
            raise ValueError(
                f"structured Mamba SSM cache must have shape {expected_ssm}, "
                f"got {tuple(cache.ssm_state.shape)}"
            )

    def _flatten_cache(self, cache: MambaStreamingCache) -> MambaStreamingCache:
        return MambaStreamingCache(
            conv_state=cache.conv_state.reshape(
                -1,
                self.temporal_mamba.inner_size,
                self.config.mamba_d_conv,
            ),
            ssm_state=cache.ssm_state.reshape(
                -1,
                self.temporal_mamba.inner_size,
                self.config.mamba_d_state,
            ),
        )

    def _unflatten_cache(
        self,
        cache: MambaStreamingCache,
        *,
        batch_size: int,
    ) -> MambaStreamingCache:
        return MambaStreamingCache(
            conv_state=cache.conv_state.reshape(
                batch_size,
                self.config.num_state_tokens,
                self.temporal_mamba.inner_size,
                self.config.mamba_d_conv,
            ),
            ssm_state=cache.ssm_state.reshape(
                batch_size,
                self.config.num_state_tokens,
                self.temporal_mamba.inner_size,
                self.config.mamba_d_state,
            ),
        )
