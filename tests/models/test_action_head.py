from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn as nn

from prism.data.schema import VLASample
from prism.models.action_head import DirectActionHead, decode_gripper_open
from prism.models.batch import PolicyBatch, PolicyBatchCollator, PolicyCurrentBatch, PolicyInferenceBatch
from prism.models.config import DirectActionHeadConfig, PrismArchitectureConfig
from prism.models.history_qformer import HistoryMemoryOutput
from prism.models.policy import PrismPolicy, masked_action_l1
from prism.models.vlm import PreparedQueryMemoryBatch, QueryMemoryEncoderOutput
from prism.schema import CurrentPolicyInput, PolicyInput


STATE_DIM = 8


def _resolved_architecture() -> PrismArchitectureConfig:
    return PrismArchitectureConfig(
        action_head=DirectActionHeadConfig(
            action_hidden_size=32,
            num_attention_heads=4,
            ffn_ratio=2,
        )
    )


def _encoder_output(
    batch_size: int,
    *,
    requires_grad: bool = False,
    memory_valid: bool = True,
) -> QueryMemoryEncoderOutput:
    architecture = _resolved_architecture()
    current = tuple(
        torch.randn(
            batch_size,
            architecture.backbone.num_action_queries,
            architecture.backbone.hidden_size,
            requires_grad=requires_grad,
        )
        for _ in range(architecture.num_bridge_layers)
    )
    memory = torch.randn(
        batch_size,
        architecture.history.num_memory_tokens,
        architecture.history.hidden_size,
        requires_grad=requires_grad,
    )
    return QueryMemoryEncoderOutput(
        layerwise_query_features=current,
        query_valid_mask=torch.ones(
            batch_size,
            architecture.backbone.num_action_queries,
            dtype=torch.bool,
        ),
        memory=HistoryMemoryOutput(
            tokens=memory,
            valid_mask=torch.full(
                (batch_size, architecture.history.num_memory_tokens),
                memory_valid,
                dtype=torch.bool,
            ),
        ),
    )


def _prepared_batch_tensors(batch_size: int) -> PreparedQueryMemoryBatch:
    return PreparedQueryMemoryBatch(
        current_inputs={
            "input_ids": torch.zeros(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
            "pixel_values": torch.zeros(batch_size * 2, 3, 4, 4),
            "image_grid_thw": torch.ones(batch_size * 2, 3, dtype=torch.long),
        },
        history_inputs={
            "pixel_values": torch.zeros(batch_size * 4, 3, 4, 4),
            "image_grid_thw": torch.ones(batch_size * 4, 3, dtype=torch.long),
        },
        history_step_ages=torch.tensor([[6, 3]]).expand(batch_size, -1).clone(),
        history_valid_mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )


def _policy_batch(batch_size: int = 2) -> PolicyBatch:
    prepared = _prepared_batch_tensors(batch_size)
    return PolicyBatch(
        current_inputs=prepared.current_inputs,
        history_inputs=prepared.history_inputs,
        history_step_ages=prepared.history_step_ages,
        history_valid_mask=prepared.history_valid_mask,
        state=torch.randn(batch_size, STATE_DIM),
        executed_actions=torch.zeros(batch_size, 8, 7),
        executed_action_valid_mask=torch.zeros(batch_size, 8, dtype=torch.bool),
        target_actions=torch.zeros(batch_size, 8, 7),
        action_valid_mask=torch.ones(batch_size, 8, dtype=torch.bool),
        action_dim_mask=torch.ones(batch_size, 7, dtype=torch.bool),
    )


def _vla_sample(*, state_value: float = 0.0) -> VLASample:
    current = {
        "primary": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    history = {
        "primary": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((2, 4, 4, 3), dtype=np.uint8),
    }
    policy_input = PolicyInput(
        benchmark="libero",
        prompt="pick up the object",
        images_by_view=current,
        history_images_by_view=history,
        history_step_ages=np.asarray([6, 3], dtype=np.int64),
        history_valid_mask=np.asarray([True, True], dtype=np.bool_),
        state=np.full((STATE_DIM,), state_value, dtype=np.float32),
        action_dim=7,
        robot_key="libero",
    )
    return VLASample(
        policy_input=policy_input,
        dataset_name="libero_spatial",
        statistics_group="libero",
        episode_index=0,
        frame_index=0,
        target_actions=np.zeros((8, 7), dtype=np.float32),
        action_valid_mask=np.ones((8,), dtype=np.bool_),
    )


class _RecordingPrepareEncoder:
    def __init__(self) -> None:
        self.requests: list[PolicyInput] = []

    def prepare_requests(
        self,
        requests: list[PolicyInput],
    ) -> PreparedQueryMemoryBatch:
        self.requests.extend(requests)
        return _prepared_batch_tensors(len(requests))

    def prepare_current_requests(self, requests: list[CurrentPolicyInput]):
        self.requests.extend(requests)
        return _prepared_batch_tensors(len(requests)).current_inputs


class _DifferentiablePreparedEncoder(nn.Module):
    def __init__(self, architecture: PrismArchitectureConfig) -> None:
        super().__init__()
        self.architecture = architecture
        self.current_features = nn.Parameter(torch.randn(architecture.backbone.hidden_size))
        self.memory_features = nn.Parameter(torch.randn(architecture.history.hidden_size))
        self.forward_calls = 0

    def prepare_requests(self, requests: list[PolicyInput]) -> PreparedQueryMemoryBatch:
        del requests
        raise AssertionError("processor work must not run inside PrismPolicy.forward")

    def forward_prepared(
        self,
        batch: PreparedQueryMemoryBatch,
    ) -> QueryMemoryEncoderOutput:
        self.forward_calls += 1
        batch_size = batch.current_inputs["attention_mask"].shape[0]
        current = self.current_features.view(1, 1, -1).expand(
            batch_size,
            self.architecture.backbone.num_action_queries,
            -1,
        )
        memory = self.memory_features.view(1, 1, -1).expand(
            batch_size,
            self.architecture.history.num_memory_tokens,
            -1,
        )
        return QueryMemoryEncoderOutput(
            layerwise_query_features=tuple(current for _ in range(self.architecture.num_bridge_layers)),
            query_valid_mask=torch.ones(
                batch_size,
                self.architecture.backbone.num_action_queries,
                dtype=torch.bool,
            ),
            memory=HistoryMemoryOutput(
                tokens=memory,
                valid_mask=torch.ones(
                    batch_size,
                    self.architecture.history.num_memory_tokens,
                    dtype=torch.bool,
                ),
            ),
        )

    def forward_current_with_memory(
        self,
        current_inputs,
        memory: HistoryMemoryOutput,
    ) -> QueryMemoryEncoderOutput:
        self.forward_calls += 1
        batch_size = current_inputs["attention_mask"].shape[0]
        current = self.current_features.view(1, 1, -1).expand(
            batch_size,
            self.architecture.backbone.num_action_queries,
            -1,
        )
        return QueryMemoryEncoderOutput(
            layerwise_query_features=tuple(current for _ in range(self.architecture.num_bridge_layers)),
            query_valid_mask=torch.ones(
                batch_size,
                self.architecture.backbone.num_action_queries,
                dtype=torch.bool,
            ),
            memory=memory,
        )


def test_direct_action_head_requires_a_resolved_architecture_config():
    with pytest.raises(ValueError, match="not resolved"):
        DirectActionHead(PrismArchitectureConfig(), state_dim=STATE_DIM)

    with pytest.raises(ValueError, match="not resolved"):
        PrismPolicy(
            PrismArchitectureConfig(),
            _DifferentiablePreparedEncoder(_resolved_architecture()),
            state_dim=STATE_DIM,
        )


def test_direct_action_head_uses_eight_tokens_and_returns_unbounded_actions():
    architecture = _resolved_architecture()
    head = DirectActionHead(architecture, state_dim=STATE_DIM).eval()
    with torch.no_grad():
        head.output_projection.weight.zero_()
        head.output_projection.bias.fill_(2.0)

    actions = head(_encoder_output(2, memory_valid=False), torch.zeros(2, STATE_DIM))

    assert head.action_step_queries.shape == (8, 32)
    assert head.temporal_position_embeddings.shape == (8, 32)
    assert architecture.backbone.num_action_queries == 32
    assert actions.shape == (2, 8, 7)
    assert torch.all(actions == 2.0)
    assert len(head.bridge.blocks) == 16
    assert not any(isinstance(module, (nn.Sigmoid, nn.Tanh)) for module in head.modules())


def test_action_state_current_and_history_paths_all_receive_gradients():
    head = DirectActionHead(_resolved_architecture(), state_dim=STATE_DIM)
    encoder_output = _encoder_output(1, requires_grad=True)
    state = torch.randn(1, STATE_DIM, requires_grad=True)

    head(encoder_output, state).square().mean().backward()

    first_block = head.bridge.blocks[0]
    assert state.grad is not None and torch.count_nonzero(state.grad) > 0
    assert head.action_step_queries.grad is not None
    assert torch.count_nonzero(head.action_step_queries.grad) > 0
    assert head.temporal_position_embeddings.grad is not None
    assert torch.count_nonzero(head.temporal_position_embeddings.grad) > 0
    assert head.state_projection.weight.grad is not None
    assert first_block.action_self_attention.in_proj_weight.grad is not None
    assert first_block.current_attention.in_proj_weight.grad is not None
    assert first_block.memory_attention.in_proj_weight.grad is not None
    assert first_block.ffn[0].weight.grad is not None
    assert encoder_output.layerwise_query_features[0].grad is not None
    assert torch.count_nonzero(encoder_output.layerwise_query_features[0].grad) > 0
    assert encoder_output.memory.tokens.grad is not None
    assert torch.count_nonzero(encoder_output.memory.tokens.grad) > 0


def test_action_self_attention_is_noncausal():
    block = DirectActionHead(_resolved_architecture(), state_dim=STATE_DIM).bridge.blocks[0]
    action_states = torch.randn(1, 8, 32, requires_grad=True)

    output = block(
        action_states,
        torch.randn(1, 32, 1024),
        torch.ones(1, 32, dtype=torch.bool),
        torch.zeros(1, 16, 512),
        torch.zeros(1, 16, dtype=torch.bool),
    )
    output[:, 0].sum().backward()

    assert action_states.grad is not None
    assert torch.count_nonzero(action_states.grad[:, -1]) > 0


def test_masked_action_l1_uses_the_exact_element_denominator():
    predicted = torch.zeros(2, 3, 7)
    target = torch.arange(1, 43, dtype=torch.float32).reshape(2, 3, 7)
    time_mask = torch.tensor([[True, True, False], [False, True, False]])
    dim_mask = torch.tensor(
        [
            [True, False, True, False, False, False, True],
            [False, True, False, True, False, True, False],
        ]
    )
    element_mask = time_mask.unsqueeze(-1) & dim_mask.unsqueeze(1)
    expected = (target * element_mask).sum() / element_mask.sum()

    loss, metrics = masked_action_l1(
        predicted,
        target,
        time_mask,
        dim_mask,
        gripper_index=6,
        gripper_threshold=0.5,
    )

    assert loss == pytest.approx(expected.item())
    assert metrics["total_l1"] == pytest.approx(expected.item())


def test_masked_action_l1_ignores_tail_and_invalid_dimensions():
    predicted = torch.zeros(1, 3, 7)
    target = torch.zeros_like(predicted)
    target[:, 1:, :] = 1000.0
    target[:, 0, 1] = 500.0
    time_mask = torch.tensor([[True, False, False]])
    dim_mask = torch.tensor([[True, False, True, True, True, True, True]])

    loss, _ = masked_action_l1(
        predicted,
        target,
        time_mask,
        dim_mask,
        gripper_index=6,
        gripper_threshold=0.5,
    )

    assert loss.item() == 0.0


def test_masked_action_l1_rejects_an_all_masked_batch():
    with pytest.raises(ValueError, match="at least one valid element"):
        masked_action_l1(
            torch.zeros(1, 8, 7),
            torch.zeros(1, 8, 7),
            torch.zeros(1, 8, dtype=torch.bool),
            torch.ones(1, 7, dtype=torch.bool),
            gripper_index=6,
            gripper_threshold=0.5,
        )


def test_gripper_metrics_use_strict_threshold_and_transition_recall():
    predicted = torch.zeros(1, 4, 7)
    target = torch.zeros_like(predicted)
    predicted[0, :, 6] = torch.tensor([0.5, 0.5001, 0.6, 0.6])
    target[0, :, 6] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    gripper_only = torch.tensor([[False, False, False, False, False, False, True]])

    _, metrics = masked_action_l1(
        predicted,
        target,
        torch.ones(1, 4, dtype=torch.bool),
        gripper_only,
        gripper_index=6,
        gripper_threshold=0.5,
    )

    assert not decode_gripper_open(
        predicted,
        gripper_index=6,
        threshold=0.5,
    )[0, 0]
    assert metrics["gripper_accuracy"] == pytest.approx(0.5)
    assert metrics["predicted_open_ratio"] == pytest.approx(0.75)
    assert metrics["target_open_ratio"] == pytest.approx(0.25)
    assert metrics["gripper_transition_recall"] == pytest.approx(0.5)
    assert metrics["motion_l1"].item() == 0.0


def test_policy_batch_collator_prepares_requests_on_cpu_before_forward():
    encoder = _RecordingPrepareEncoder()
    collator = PolicyBatchCollator(
        _resolved_architecture(),
        encoder,
        state_dim=STATE_DIM,
    )

    batch = collator([_vla_sample(), _vla_sample(state_value=1.0)])

    assert len(encoder.requests) == 2
    assert batch.state.shape == (2, STATE_DIM)
    assert batch.target_actions.shape == (2, 8, 7)
    assert batch.action_valid_mask.shape == (2, 8)
    assert batch.action_dim_mask is not None and batch.action_dim_mask.all()
    assert batch.state.device.type == "cpu"
    assert batch.current_inputs["input_ids"].device.type == "cpu"


def test_policy_batch_collator_has_target_free_inference_path():
    encoder = _RecordingPrepareEncoder()
    collator = PolicyBatchCollator(
        _resolved_architecture(),
        encoder,
        state_dim=STATE_DIM,
    )
    policy_input = _vla_sample().policy_input

    batch = collator.collate_inference([policy_input])

    assert isinstance(batch, PolicyInferenceBatch)
    assert not isinstance(batch, PolicyBatch)
    assert not hasattr(batch, "target_actions")
    assert not hasattr(batch, "action_valid_mask")
    assert batch.state.shape == (1, STATE_DIM)
    assert len(encoder.requests) == 1
    assert encoder.requests[0] is policy_input


def test_policy_batch_collator_prepares_current_only_runtime_path():
    encoder = _RecordingPrepareEncoder()
    collator = PolicyBatchCollator(
        _resolved_architecture(),
        encoder,
        state_dim=STATE_DIM,
    )
    policy_input = _vla_sample().policy_input
    current_input = CurrentPolicyInput(
        benchmark=policy_input.benchmark,
        prompt=policy_input.prompt,
        images_by_view=policy_input.images_by_view,
        state=policy_input.state,
        action_dim=policy_input.action_dim,
        robot_key=policy_input.robot_key,
    )

    batch = collator.collate_current_inference([current_input])

    assert isinstance(batch, PolicyCurrentBatch)
    assert not hasattr(batch, "history_inputs")
    assert batch.state.shape == (1, STATE_DIM)
    assert encoder.requests == [current_input]


def test_collator_rejects_a_state_width_mismatch_before_processor_work():
    encoder = _RecordingPrepareEncoder()
    collator = PolicyBatchCollator(
        _resolved_architecture(),
        encoder,
        state_dim=STATE_DIM,
    )
    sample = _vla_sample()
    invalid_input = replace(sample.policy_input, state=np.zeros((STATE_DIM - 1,), dtype=np.float32))
    invalid_sample = replace(sample, policy_input=invalid_input)

    with pytest.raises(ValueError, match="state"):
        collator([invalid_sample])

    assert not encoder.requests


def test_prism_policy_reuses_forward_prepared_and_backpropagates_to_encoder():
    architecture = _resolved_architecture()
    encoder = _DifferentiablePreparedEncoder(architecture)
    policy = PrismPolicy(architecture, encoder, state_dim=STATE_DIM)

    output = policy(_policy_batch())
    output.loss.backward()

    assert output.predicted_actions.shape == (2, 8, 7)
    assert encoder.forward_calls == 1
    assert encoder.current_features.grad is not None
    assert torch.count_nonzero(encoder.current_features.grad) > 0
    assert encoder.memory_features.grad is not None
    assert torch.count_nonzero(encoder.memory_features.grad) > 0
    assert set(output.metrics) >= {
        "total_l1",
        "motion_l1",
        "gripper_l1",
        "gripper_accuracy",
        "predicted_open_ratio",
        "target_open_ratio",
        "gripper_transition_recall",
    }


def test_prism_policy_predict_and_forward_share_prediction_path_without_fake_loss():
    architecture = _resolved_architecture()
    encoder = _DifferentiablePreparedEncoder(architecture)
    policy = PrismPolicy(architecture, encoder, state_dim=STATE_DIM).eval()
    training_batch = _policy_batch(batch_size=1)
    inference_batch = PolicyInferenceBatch(
        current_inputs=training_batch.current_inputs,
        history_inputs=training_batch.history_inputs,
        history_step_ages=training_batch.history_step_ages,
        history_valid_mask=training_batch.history_valid_mask,
        state=training_batch.state,
        executed_actions=training_batch.executed_actions,
        executed_action_valid_mask=training_batch.executed_action_valid_mask,
    )

    with torch.no_grad():
        training_prediction = policy(training_batch).predicted_actions
        inference_prediction = policy.predict(inference_batch)

    torch.testing.assert_close(inference_prediction, training_prediction)
    assert encoder.forward_calls == 2
    assert isinstance(inference_prediction, torch.Tensor)
    assert not hasattr(inference_prediction, "loss")


def test_prism_policy_predicts_with_precomputed_memory_without_history_inputs():
    architecture = _resolved_architecture()
    encoder = _DifferentiablePreparedEncoder(architecture)
    policy = PrismPolicy(architecture, encoder, state_dim=STATE_DIM).eval()
    prepared = _prepared_batch_tensors(1)
    batch = PolicyCurrentBatch(
        current_inputs=prepared.current_inputs,
        state=torch.zeros(1, STATE_DIM),
        executed_actions=torch.zeros(1, 8, 7),
        executed_action_valid_mask=torch.zeros(1, 8, dtype=torch.bool),
    )
    memory = HistoryMemoryOutput(
        tokens=torch.randn(1, architecture.history.num_memory_tokens, architecture.history.hidden_size),
        valid_mask=torch.ones(1, architecture.history.num_memory_tokens, dtype=torch.bool),
    )

    with torch.no_grad():
        prediction = policy.predict_with_memory(batch, memory)

    assert prediction.shape == (1, 8, 7)
    assert encoder.forward_calls == 1


def test_runtime_cycle_selects_qwen_layer12_and_returns_state_and_plan_tokens():
    architecture = _resolved_architecture()

    class LayerNumberEncoder(_DifferentiablePreparedEncoder):
        def forward_current_with_memory(self, current_inputs, memory):
            output = super().forward_current_with_memory(current_inputs, memory)
            return QueryMemoryEncoderOutput(
                layerwise_query_features=tuple(
                    torch.full_like(output.layerwise_query_features[0], float(layer))
                    for layer in range(1, 17)
                ),
                query_valid_mask=output.query_valid_mask,
                memory=output.memory,
            )

    encoder = LayerNumberEncoder(architecture)
    policy = PrismPolicy(architecture, encoder, state_dim=STATE_DIM).eval()
    prepared = _prepared_batch_tensors(1)
    batch = PolicyCurrentBatch(
        current_inputs=prepared.current_inputs,
        state=torch.zeros(1, STATE_DIM),
        executed_actions=torch.zeros(1, 8, 7),
        executed_action_valid_mask=torch.zeros(1, 8, dtype=torch.bool),
    )
    memory = HistoryMemoryOutput(
        tokens=torch.zeros(1, 16, 512),
        valid_mask=torch.zeros(1, 16, dtype=torch.bool),
    )
    observed = []
    handle = policy.task_state_planner.shared_query_projection.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    try:
        with torch.inference_mode():
            output = policy.predict_with_memory_and_plan(batch, memory)
    finally:
        handle.remove()

    assert encoder.forward_calls == 1
    assert len(observed) == 1
    assert torch.all(observed[0] == 12.0)
    assert output.predicted_actions.shape == (1, 8, 7)
    assert output.task_state.shape == (1, 8, 512)
    assert output.plan_tokens.shape == (1, 16, 512)
