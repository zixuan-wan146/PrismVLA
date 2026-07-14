from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch
import torch.nn as nn

from prism.models.config import TaskStatePlannerConfig
from prism.models.task_state_planner import StreamingMambaStep, TaskStatePlanPipeline


def test_task_state_plan_pipeline_matches_the_accepted_topology_and_capacity():
    config = TaskStatePlannerConfig()
    pipeline = TaskStatePlanPipeline(config)

    assert sum(parameter.numel() for parameter in pipeline.parameters()) == 8_686_080
    assert isinstance(pipeline.shared_query_projection[0], nn.Linear)
    assert pipeline.shared_query_projection[0].in_features == 1024
    assert pipeline.shared_query_projection[0].out_features == 512
    assert isinstance(pipeline.shared_query_projection[1], nn.LayerNorm)
    assert pipeline.execution_position_embeddings.shape == (8, 512)
    assert pipeline.initial_state_tokens.shape == (8, 512)
    assert pipeline.plan_queries.shape == (16, 512)
    assert pipeline.state_cross_attention.num_heads == 8
    assert pipeline.state_self_attention.num_heads == 8
    assert pipeline.plan_reader_attention.num_heads == 8
    assert pipeline.plan_mixer_attention.num_heads == 8
    assert pipeline.state_cross_attention.dropout == 0.0
    assert "robot_state" not in inspect.signature(pipeline.forward).parameters
    assert "proprio" not in inspect.signature(pipeline.forward).parameters


def test_forward_uses_q12_projection_once_and_has_exact_state_plan_cache_shapes():
    torch.manual_seed(5)
    pipeline = TaskStatePlanPipeline(TaskStatePlannerConfig()).eval()
    query_layer12 = torch.randn(2, 32, 1024)
    executed_actions = torch.zeros(2, 8, 7)
    initial_mask = torch.zeros(2, 8, dtype=torch.bool)
    projection_inputs: list[torch.Tensor] = []

    handle = pipeline.shared_query_projection.register_forward_pre_hook(
        lambda _module, args: projection_inputs.append(args[0].detach().clone())
    )
    try:
        with torch.inference_mode():
            output = pipeline(query_layer12, executed_actions, initial_mask)
    finally:
        handle.remove()

    assert len(projection_inputs) == 1
    torch.testing.assert_close(projection_inputs[0], query_layer12)
    assert output.task_state.shape == (2, 8, 512)
    assert output.plan_tokens.shape == (2, 16, 512)
    assert output.runtime_state.task_state is output.task_state
    assert output.runtime_state.mamba_cache.conv_state.shape == (2, 8, 1024, 4)
    assert output.runtime_state.mamba_cache.ssm_state.shape == (2, 8, 1024, 16)
    assert torch.isfinite(output.task_state).all()
    assert torch.isfinite(output.plan_tokens).all()


def test_reset_is_deterministic_while_the_next_cycle_advances_temporal_state():
    torch.manual_seed(11)
    pipeline = TaskStatePlanPipeline(TaskStatePlannerConfig()).eval()
    query_layer12 = torch.randn(1, 32, 1024)
    actions = torch.randn(1, 8, 7)
    valid = torch.ones(1, 8, dtype=torch.bool)

    with torch.inference_mode():
        first = pipeline(query_layer12, actions, valid)
        reset = pipeline(query_layer12, actions, valid)
        next_cycle = pipeline(
            query_layer12,
            actions,
            valid,
            previous_state=first.runtime_state,
        )

    torch.testing.assert_close(first.task_state, reset.task_state)
    torch.testing.assert_close(first.plan_tokens, reset.plan_tokens)
    assert not torch.allclose(first.task_state, next_cycle.task_state)
    assert not torch.allclose(
        first.runtime_state.mamba_cache.ssm_state,
        next_cycle.runtime_state.mamba_cache.ssm_state,
    )


def test_initial_action_positions_are_masked_instead_of_consuming_dummy_actions():
    torch.manual_seed(17)
    pipeline = TaskStatePlanPipeline(TaskStatePlannerConfig()).eval()
    query_layer12 = torch.randn(1, 32, 1024)
    invalid = torch.zeros(1, 8, dtype=torch.bool)

    with torch.inference_mode():
        zero_actions = pipeline(
            query_layer12,
            torch.zeros(1, 8, 7),
            invalid,
        )
        arbitrary_masked_actions = pipeline(
            query_layer12,
            torch.randn(1, 8, 7) * 100.0,
            invalid,
        )

    torch.testing.assert_close(zero_actions.task_state, arbitrary_masked_actions.task_state)
    torch.testing.assert_close(zero_actions.plan_tokens, arbitrary_masked_actions.plan_tokens)


def test_mamba_slots_have_independent_caches_and_causal_history():
    torch.manual_seed(23)
    mamba = StreamingMambaStep(
        hidden_size=512,
        d_state=16,
        d_conv=4,
        expand=2,
    ).eval()
    first_input = torch.randn(3, 1, 512)
    changed_first_slot = first_input.clone()
    changed_first_slot[0] += 3.0

    with torch.inference_mode():
        first_output, first_cache = mamba(first_input, None)
        changed_output, changed_cache = mamba(changed_first_slot, None)
        continued_output, _ = mamba(torch.zeros_like(first_input), first_cache)
        reset_output, _ = mamba(torch.zeros_like(first_input), None)

    assert not torch.allclose(first_output[0], changed_output[0])
    torch.testing.assert_close(first_output[1:], changed_output[1:])
    torch.testing.assert_close(first_cache.conv_state[1:], changed_cache.conv_state[1:])
    torch.testing.assert_close(first_cache.ssm_state[1:], changed_cache.ssm_state[1:])
    assert not torch.allclose(continued_output, reset_output)


def test_valid_executed_actions_and_q12_receive_gradients_but_masked_actions_do_not():
    torch.manual_seed(29)
    pipeline = TaskStatePlanPipeline(TaskStatePlannerConfig())
    query_layer12 = torch.randn(1, 32, 1024, requires_grad=True)
    actions = torch.randn(1, 8, 7, requires_grad=True)
    valid = torch.tensor([[True, True, True, True, False, False, False, False]])

    output = pipeline(query_layer12, actions, valid)
    (output.task_state.square().mean() + output.plan_tokens.square().mean()).backward()

    assert query_layer12.grad is not None
    assert torch.count_nonzero(query_layer12.grad) > 0
    assert actions.grad is not None
    assert torch.count_nonzero(actions.grad[:, :4]) > 0
    assert torch.count_nonzero(actions.grad[:, 4:]) == 0
    assert pipeline.shared_query_projection[0].weight.grad is not None


@pytest.mark.parametrize(
    "change",
    [
        {"query_layer": 11},
        {"num_state_tokens": 7},
        {"num_plan_tokens": 15},
        {"attention_dropout": 0.1},
        {"mamba_d_state": 8},
    ],
)
def test_task_state_planner_rejects_pipeline_contract_drift(change):
    with pytest.raises(ValueError, match="accepted pipeline|zero attention"):
        replace(TaskStatePlannerConfig(), **change).validate()
