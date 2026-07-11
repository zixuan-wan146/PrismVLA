import pytest

torch = pytest.importorskip("torch")

from prism.models.planner import ProgressPretrainHeads
from prism.models.planner import ProgressStateConfig
from prism.models.planner import ProgressStatePlanner
from prism.models.planner import progress_diagnostics
from prism.models.planner import progress_warmup_loss
from prism.models.policy import PrismPolicy


def test_progress_state_planner_shapes_and_loss():
    config = ProgressStateConfig(
        hidden_dim=16,
        state_dim=5,
        action_dim=3,
        replan_stride=4,
        latent_dim=6,
        action_summary_hidden_dim=8,
        state_hidden_dim=8,
        updater_hidden_dim=32,
        planner_ffn_dim=32,
        planner_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    model = ProgressStatePlanner(config)
    heads = ProgressPretrainHeads(config)
    batch_size = 2
    state = model.initial_state(batch_size)

    output = model.forward_step(
        state,
        vl_summary=torch.randn(batch_size, 16),
        robot_state=torch.randn(batch_size, 5),
        executed_actions=torch.randn(batch_size, 4, 3),
        executed_mask=torch.ones(batch_size, 4, dtype=torch.bool),
    )
    head_output = heads(output.planner_token, output.progress_state)
    target = torch.randn(batch_size, 6)
    loss, metrics = progress_warmup_loss(head_output, target, use_order_loss=False)
    diagnostics = progress_diagnostics(
        output.planner_token,
        output.progress_state.current_stage,
        head_output.planner_intent,
        head_output.stage_intent,
    )

    assert tuple(output.progress_state.tokens.shape) == (batch_size, 2, 16)
    assert tuple(output.planner_token.shape) == (batch_size, 1, 16)
    assert tuple(head_output.planner_intent.shape) == (batch_size, 6)
    assert loss.item() >= 0.0
    assert set(metrics) == {"plan_loss", "stage_loss", "mem_pool_loss", "order_loss"}
    assert "cos_g_p" in diagnostics


def test_action_summary_zero_mask_outputs_zero():
    config = ProgressStateConfig(
        hidden_dim=8,
        state_dim=4,
        action_dim=2,
        replan_stride=3,
        latent_dim=5,
        action_summary_hidden_dim=8,
        state_hidden_dim=8,
        updater_hidden_dim=16,
        planner_ffn_dim=16,
        planner_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = ProgressStatePlanner(config)
    summary = model.action_summary(
        torch.randn(2, 3, 2),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    assert torch.allclose(summary, torch.zeros_like(summary))


def test_prism_progress_planner_helper_outputs_plan_token():
    config = ProgressStateConfig(
        hidden_dim=16,
        state_dim=5,
        action_dim=3,
        replan_stride=4,
        latent_dim=6,
        action_summary_hidden_dim=8,
        state_hidden_dim=8,
        updater_hidden_dim=32,
        planner_ffn_dim=32,
        planner_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    policy = PrismPolicy.__new__(PrismPolicy)
    torch.nn.Module.__init__(policy)
    policy.progress_state_planner = ProgressStatePlanner(config)
    policy.config = {"finetune_progress_planner": False}
    policy.runtime_progress_state = None
    policy.last_progress_planner_output = None
    policy.train()

    plan = policy._get_or_update_progress_plan_tokens(
        torch.randn(2, 5, 16),
        torch.randn(2, 5),
        executed_actions=torch.randn(2, 4, 3),
        executed_action_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    assert tuple(plan.shape) == (2, 1, 16)
    assert policy.last_progress_planner_output is not None
    assert tuple(policy.last_progress_planner_output.progress_state.tokens.shape) == (2, 2, 16)


def test_prism_progress_planner_accepts_explicit_vl_summary():
    config = ProgressStateConfig(
        hidden_dim=16,
        state_dim=5,
        action_dim=3,
        replan_stride=4,
        latent_dim=6,
        action_summary_hidden_dim=8,
        state_hidden_dim=8,
        updater_hidden_dim=32,
        planner_ffn_dim=32,
        planner_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    policy = PrismPolicy.__new__(PrismPolicy)
    torch.nn.Module.__init__(policy)
    policy.progress_state_planner = ProgressStatePlanner(config)
    policy.config = {"finetune_progress_planner": False}
    policy.runtime_progress_state = None
    policy.last_progress_planner_output = None
    policy.train()

    plan = policy._get_or_update_progress_plan_tokens(
        torch.randn(2, 5, 16),
        torch.randn(2, 5),
        executed_actions=torch.randn(2, 4, 3),
        executed_action_mask=torch.ones(2, 4, dtype=torch.bool),
        planner_fused_tokens=torch.randn(2, 5, 8),
        planner_vl_summary=torch.randn(2, 16),
    )

    assert tuple(plan.shape) == (2, 1, 16)
