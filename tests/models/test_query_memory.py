from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from prism.models.config import (
    DirectActionHeadConfig,
    HistoryQFormerConfig,
    PrismArchitectureConfig,
    architecture_config_from_mapping,
    load_architecture_config,
)
from prism.models.history_qformer import HistoryQFormer
from prism.models.query_features import gather_layerwise_action_queries
from prism.models.query_memory_bridge import LayerwiseQueryMemoryBridge
from prism.models.vlm import (
    EncodedHistoryObservation,
    pack_encoded_history_observations,
    pack_two_camera_history_features,
)


def _resolved_architecture() -> PrismArchitectureConfig:
    return PrismArchitectureConfig(
        action_head=DirectActionHeadConfig(
            action_hidden_size=32,
            num_attention_heads=4,
            ffn_ratio=2,
        )
    )


def test_accepted_architecture_config_loads_from_yaml():
    config = load_architecture_config("configs/model/qwen35_query_memory.yaml")

    assert config.backbone.num_hidden_layers == 16
    assert config.backbone.num_action_queries == 32
    assert config.backbone.image_size == 256
    assert config.history.num_layers == 2
    assert config.history.num_heads == 4
    assert config.history.num_memory_tokens == 16
    assert config.task_state_planner.query_layer == 12
    assert config.task_state_planner.num_state_tokens == 8
    assert config.task_state_planner.num_plan_tokens == 16
    assert config.temporal.history_step_ages == (6, 3)
    assert config.action_head.objective == "direct_masked_l1"
    assert config.action_head.action_dim == 7
    assert config.action_head.gripper_index == 6
    assert config.action_head.gripper_threshold == pytest.approx(0.5)


def test_accepted_action_policy_dimensions_are_resolved():
    config = load_architecture_config("configs/model/qwen35_query_memory.yaml")

    config.validate_for_policy()
    assert config.action_head.action_hidden_size == 512
    assert config.action_head.num_attention_heads == 8
    assert config.action_head.ffn_ratio == 4


def test_checkpoint_canonical_architecture_round_trips_without_yaml():
    expected = load_architecture_config("configs/model/qwen35_query_memory.yaml")

    reconstructed = architecture_config_from_mapping(asdict(expected))

    assert reconstructed == expected


def test_architecture_yaml_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "backbone:\n  image_size: 256\n  image_size: 320\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate YAML mapping key 'image_size'"):
        load_architecture_config(path)


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("history", "num_layers"), True),
        (("action_head", "ffn_ratio"), 4.5),
        (("num_bridge_layers",), 16.0),
    ],
)
def test_architecture_integer_fields_reject_booleans_and_fractional_values(
    field_path: tuple[str, ...],
    value: object,
):
    raw = asdict(load_architecture_config("configs/model/qwen35_query_memory.yaml"))
    target = raw
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value

    with pytest.raises(TypeError, match="must be an integer"):
        architecture_config_from_mapping(raw)


def test_architecture_boolean_and_float_fields_are_exact_and_finite():
    raw = asdict(load_architecture_config("configs/model/qwen35_query_memory.yaml"))
    raw["backbone"]["local_files_only"] = 0
    with pytest.raises(TypeError, match="must be a boolean"):
        architecture_config_from_mapping(raw)

    raw = asdict(load_architecture_config("configs/model/qwen35_query_memory.yaml"))
    raw["history"]["dropout"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        architecture_config_from_mapping(raw)


def test_gather_layerwise_queries_excludes_h0_and_preserves_all_16_levels():
    batch_size, sequence_length, hidden_size = 2, 36, 1024
    mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    mask[:, -32:] = True
    hidden_states = tuple(torch.full((batch_size, sequence_length, hidden_size), float(level)) for level in range(17))

    queries = gather_layerwise_action_queries(hidden_states, mask)

    assert len(queries) == 16
    assert all(query.shape == (batch_size, 32, hidden_size) for query in queries)
    assert torch.all(queries[0] == 1)
    assert torch.all(queries[-1] == 16)


def test_history_qformer_returns_16_tokens_and_zeroes_missing_history():
    config = HistoryQFormerConfig()
    qformer = HistoryQFormer(config).eval()
    visual_tokens = torch.randn(2, 2, 12, 1024, requires_grad=True)
    ages = torch.tensor([[6, 3], [6, 3]])
    valid = torch.tensor([[True, True], [False, False]])

    output = qformer(visual_tokens, ages, valid)

    assert output.tokens.shape == (2, 16, 512)
    assert output.valid_mask.shape == (2, 16)
    assert output.valid_mask[0].all()
    assert not output.valid_mask[1].any()
    assert torch.count_nonzero(output.tokens[1]) == 0
    assert torch.isfinite(output.tokens).all()
    output.tokens[0].square().mean().backward()
    assert visual_tokens.grad is not None
    assert torch.count_nonzero(visual_tokens.grad[0]) > 0
    assert torch.count_nonzero(visual_tokens.grad[1]) == 0
    assert qformer.memory_queries.grad is not None


def test_history_qformer_empty_memory_skips_placeholder_attention_inputs():
    qformer = HistoryQFormer().eval()

    output = qformer.empty_memory(batch_size=2)

    assert output.tokens.shape == (2, 16, 512)
    assert output.tokens.dtype == qformer.memory_queries.dtype
    assert torch.count_nonzero(output.tokens) == 0
    assert not output.valid_mask.any()


def test_encoded_history_observation_packing_pads_tokens_without_reordering_slots():
    first = EncodedHistoryObservation(
        tokens=torch.full((5, 1024), 1.0, dtype=torch.bfloat16),
        valid_mask=torch.tensor([True, True, True, True, False]),
    )
    second = EncodedHistoryObservation(
        tokens=torch.full((3, 1024), 2.0, dtype=torch.bfloat16),
        valid_mask=torch.ones(3, dtype=torch.bool),
    )

    packed, mask = pack_encoded_history_observations((first, second))

    assert packed.shape == (1, 2, 5, 1024)
    assert mask.tolist() == [[[True, True, True, True, False], [True, True, True, False, False]]]
    assert torch.all(packed[0, 0] == 1)
    assert torch.all(packed[0, 1, :3] == 2)
    assert torch.count_nonzero(packed[0, 1, 3:]) == 0


def test_history_qformer_applies_relative_ages_to_the_matching_token_spans():
    qformer = HistoryQFormer().eval()
    captured = {}
    with torch.no_grad():
        qformer.input_projection.weight.zero_()
        qformer.input_projection.bias.zero_()
        for age in range(qformer.relative_age_embedding.num_embeddings):
            qformer.relative_age_embedding.weight[age].fill_(float(age))

    def capture_context(module, args):
        del module
        captured["context"] = args[1].detach().clone()

    handle = qformer.blocks[0].register_forward_pre_hook(capture_context)
    try:
        qformer(
            torch.zeros(1, 2, 3, 1024),
            torch.tensor([[6, 3]]),
            torch.tensor([[True, True]]),
        )
    finally:
        handle.remove()

    assert torch.all(captured["context"][:, :3] == 6)
    assert torch.all(captured["context"][:, 3:] == 3)


def test_history_qformer_accepts_backbone_dtype_with_fp32_parameters():
    qformer = HistoryQFormer().eval()

    output = qformer(
        torch.randn(1, 2, 3, 1024, dtype=torch.bfloat16),
        torch.tensor([[6, 3]]),
        torch.tensor([[True, True]]),
    )

    assert output.tokens.dtype == torch.float32
    assert torch.isfinite(output.tokens).all()


def test_history_qformer_rejects_fractional_relative_ages():
    with pytest.raises(ValueError, match="must contain integers"):
        HistoryQFormer()(
            torch.randn(1, 2, 3, 1024),
            torch.tensor([[6.5, 3.5]]),
            torch.tensor([[True, True]]),
        )


def test_two_camera_history_feature_packing_preserves_time_and_view_order():
    features = tuple(torch.full((tokens, 1024), float(index)) for index, tokens in enumerate([3, 2, 4, 1, 2, 2, 1, 3]))

    packed, mask = pack_two_camera_history_features(features, batch_size=2)

    assert packed.shape == (2, 2, 5, 1024)
    assert mask.sum(dim=-1).tolist() == [[5, 5], [4, 4]]
    assert torch.all(packed[0, 0, :3] == 0)
    assert torch.all(packed[0, 0, 3:5] == 1)
    assert torch.all(packed[1, 1, :1] == 6)
    assert torch.all(packed[1, 1, 1:4] == 7)


def test_layerwise_bridge_uses_all_levels_and_memory_gate_starts_at_point_one():
    bridge = LayerwiseQueryMemoryBridge(_resolved_architecture())
    action_states = torch.randn(2, 8, 32)
    current = tuple(torch.randn(2, 32, 1024) for _ in range(16))
    current_mask = torch.ones(2, 32, dtype=torch.bool)
    memory = torch.randn(2, 16, 512)
    memory_mask = torch.tensor([[True] * 16, [False] * 16])

    output = bridge(action_states, current, current_mask, memory, memory_mask)

    assert output.shape == action_states.shape
    assert torch.isfinite(output).all()
    assert all(block.memory_gate.item() == pytest.approx(0.1) for block in bridge.blocks)


def test_layerwise_bridge_rejects_a_sample_without_current_queries():
    bridge = LayerwiseQueryMemoryBridge(_resolved_architecture())
    current_mask = torch.tensor([[True] * 32, [False] * 32])

    with pytest.raises(ValueError, match="at least one valid current-query token"):
        bridge(
            torch.randn(2, 8, 32),
            tuple(torch.randn(2, 32, 1024) for _ in range(16)),
            current_mask,
            torch.randn(2, 16, 512),
            torch.zeros(2, 16, dtype=torch.bool),
        )


def test_layerwise_bridge_consumes_each_aligned_current_level_once():
    bridge = LayerwiseQueryMemoryBridge(_resolved_architecture())
    observed_levels = []
    handles = []
    for block in bridge.blocks:
        handles.append(
            block.register_forward_pre_hook(lambda module, args: observed_levels.append(float(args[1].mean())))
        )
    try:
        bridge(
            torch.randn(1, 8, 32),
            tuple(torch.full((1, 32, 1024), float(level)) for level in range(1, 17)),
            torch.ones(1, 32, dtype=torch.bool),
            torch.zeros(1, 16, 512),
            torch.zeros(1, 16, dtype=torch.bool),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert observed_levels == [float(level) for level in range(1, 17)]


def test_bridge_casts_bfloat16_conditioning_and_preserves_memory_gradients():
    bridge = LayerwiseQueryMemoryBridge(_resolved_architecture())
    block = bridge.blocks[0]
    action_states = torch.randn(1, 2, 32, dtype=torch.bfloat16, requires_grad=True)
    current = torch.randn(1, 3, 1024, dtype=torch.bfloat16, requires_grad=True)
    memory = torch.randn(1, 2, 512, dtype=torch.bfloat16, requires_grad=True)

    output = block(
        action_states,
        current,
        torch.ones(1, 3, dtype=torch.bool),
        memory,
        torch.ones(1, 2, dtype=torch.bool),
    )
    output.square().mean().backward()

    assert output.dtype == torch.float32
    assert action_states.grad is not None
    assert current.grad is not None
    assert memory.grad is not None
    assert block.memory_gate.grad is not None


def test_layerwise_bridge_requires_resolved_action_dimensions():
    with pytest.raises(ValueError, match="not resolved"):
        LayerwiseQueryMemoryBridge(PrismArchitectureConfig())
