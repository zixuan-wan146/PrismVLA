from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from prism.serve.engine import PolicyRequest
from prism.serve.engine import PolicyInferenceEngine
from prism.serve.engine import RuntimePolicyState


@dataclass
class FakeEmbeddingOutput:
    fused_tokens: torch.Tensor
    hidden_states: list[torch.Tensor]
    visual_tokens: torch.Tensor
    planner_vl_summary: torch.Tensor


class FakeNormalizer:
    def normalize_state(self, state, robot_key=None):
        return state

    def denormalize_action(self, action, robot_key=None):
        return action

    def normalize_action(self, action, robot_key=None):
        return action + 10.0


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = {
            "per_action_dim": 2,
            "state_dim": 2,
            "memory_short_capacity": 2,
            "memory_entry_tokens": 2,
            "memory_short_offsets": [16, 8],
            "embed_dim": 3,
        }
        self.per_action_dim = 2
        self.use_bridge = True
        self.use_direct_bridge = True
        self.progress_state_planner = None
        self.embedding_call_values = [1.0, 16.0, 8.0]
        self.predict_kwargs = None

    def reset_progress_state(self):
        pass

    def get_vl_embeddings(self, **kwargs):
        value = self.embedding_call_values.pop(0)
        tokens = torch.full((1, 4, 3), value, dtype=torch.float32)
        return FakeEmbeddingOutput(
            fused_tokens=torch.full((1, 5, 3), value + 100.0),
            hidden_states=[torch.full((1, 5, 3), value + 200.0)],
            visual_tokens=tokens,
            planner_vl_summary=torch.full((1, 3), value + 300.0),
        )

    def predict_action(self, fused_tokens, state, **kwargs):
        self.predict_kwargs = kwargs
        return torch.zeros(1, 1, 2, dtype=torch.float32)


class FakeProgressPlanner:
    config = SimpleNamespace(replan_stride=2, action_dim=2)

    def initial_state(self, batch_size, *, device, dtype):
        return SimpleNamespace(
            completed_events=torch.zeros(batch_size, 3, device=device, dtype=dtype),
            current_stage=torch.ones(batch_size, 3, device=device, dtype=dtype),
        )


def test_inference_engine_builds_short_memory_from_request_offsets():
    model = FakeModel()
    engine = PolicyInferenceEngine(model, FakeNormalizer(), state_dim=2)
    request = PolicyRequest(
        benchmark="libero",
        prompt="pick",
        images_by_view={"agentview_rgb": _image(1)},
        state=np.array([0.1, 0.2], dtype=np.float32),
        action_dim=2,
        short_memory_images_by_offset={
            16: {"agentview_rgb": _image(16)},
            8: {"agentview_rgb": _image(8)},
        },
    )

    engine.infer(request, RuntimePolicyState())

    memory_context = model.predict_kwargs["memory_context"]
    memory_context_mask = model.predict_kwargs["memory_context_mask"]
    short_memory_time_ids = model.predict_kwargs["short_memory_time_ids"]
    assert memory_context.tolist() == [
        [
            [16.0, 16.0, 16.0],
            [16.0, 16.0, 16.0],
            [8.0, 8.0, 8.0],
            [8.0, 8.0, 8.0],
        ]
    ]
    assert memory_context_mask.tolist() == [[True, True, True, True]]
    assert short_memory_time_ids.tolist() == [[0, 0, 1, 1]]
    assert model.predict_kwargs["planner_vl_summary"].tolist() == [[301.0, 301.0, 301.0]]


def test_inference_engine_does_not_use_previous_visual_tokens_as_short_memory():
    model = FakeModel()
    model.embedding_call_values = [1.0]
    engine = PolicyInferenceEngine(model, FakeNormalizer(), state_dim=2)
    request = PolicyRequest(
        benchmark="libero",
        prompt="pick",
        images_by_view={"agentview_rgb": _image(1)},
        state=np.array([0.1, 0.2], dtype=np.float32),
        action_dim=2,
    )
    runtime_state = RuntimePolicyState()

    engine.infer(request, runtime_state)

    assert model.predict_kwargs["memory_context"] is None
    assert model.predict_kwargs["memory_context_mask"] is None
    assert model.predict_kwargs["short_memory_time_ids"] is None


def test_inference_engine_prefers_request_executed_actions_over_model_output_cache():
    model = FakeModel()
    model.progress_state_planner = FakeProgressPlanner()
    engine = PolicyInferenceEngine(model, FakeNormalizer(), state_dim=2)
    request = PolicyRequest(
        benchmark="libero",
        prompt="pick",
        images_by_view={"agentview_rgb": _image(1)},
        state=np.array([0.1, 0.2], dtype=np.float32),
        action_dim=2,
        executed_actions=np.array([[1.0, 2.0]], dtype=np.float32),
        executed_action_mask=np.array([True]),
    )
    runtime_state = RuntimePolicyState()

    engine.infer(request, runtime_state)

    assert model.predict_kwargs["executed_actions"].tolist() == [[[11.0, 12.0], [0.0, 0.0]]]
    assert model.predict_kwargs["executed_action_mask"].tolist() == [[True, False]]
    assert model.predict_kwargs["progress_state"].current_stage.tolist() == [[1.0, 1.0, 1.0]]
    assert runtime_state.executed_actions.tolist() == [[[11.0, 12.0], [0.0, 0.0]]]


def _image(value: int) -> np.ndarray:
    return np.full((448, 448, 3), value, dtype=np.uint8)
