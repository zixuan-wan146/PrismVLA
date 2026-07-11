from __future__ import annotations

import torch

from prism.models.planner import ProgressState
from prism.serve.engine import build_short_memory_inputs_from_visual_tokens
from prism.serve.engine import RuntimePolicyState
from prism.serve.engine import short_memory_offsets


class FakeModel:
    use_direct_bridge = True
    config = {
        "memory_short_capacity": 2,
        "memory_entry_tokens": 2,
        "memory_short_offsets": [16, 8],
        "embed_dim": 3,
    }


class FakePlanner:
    config = type("Config", (), {"replan_stride": 2, "action_dim": 2})()

    def initial_state(self, batch_size, *, device, dtype):
        return ProgressState(
            completed_events=torch.zeros(batch_size, 3, device=device, dtype=dtype),
            current_stage=torch.ones(batch_size, 3, device=device, dtype=dtype),
        )


class FakePlannerModel(FakeModel):
    progress_state_planner = FakePlanner()

    def __init__(self):
        self.last_progress_planner_output = None
        self.reset_count = 0

    def reset_progress_state(self):
        self.reset_count += 1


class FakePlannerOutput:
    def __init__(self, state):
        self.progress_state = state


def test_short_memory_inputs_follow_configured_offset_order():
    model = FakeModel()
    memory, mask, time_ids = build_short_memory_inputs_from_visual_tokens(
        model,
        {
            8: torch.full((1, 4, 3), 8.0),
            16: torch.full((1, 4, 3), 16.0),
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert memory is not None
    assert mask is not None
    assert time_ids is not None
    assert memory.shape == (1, 4, 3)
    assert mask.tolist() == [[True, True, True, True]]
    assert time_ids.tolist() == [[0, 0, 1, 1]]
    assert memory[0, :2].tolist() == [[16.0, 16.0, 16.0], [16.0, 16.0, 16.0]]
    assert memory[0, 2:].tolist() == [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]


def test_short_memory_inputs_leave_missing_offsets_masked_out():
    model = FakeModel()
    memory, mask, time_ids = build_short_memory_inputs_from_visual_tokens(
        model,
        {8: torch.full((1, 4, 3), 8.0)},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert memory is not None
    assert mask is not None
    assert time_ids is not None
    assert mask.tolist() == [[False, False, True, True]]
    assert memory[0, :2].abs().sum().item() == 0.0
    assert memory[0, 2:].tolist() == [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]


def test_short_memory_offsets_use_checkpoint_order():
    assert short_memory_offsets(FakeModel()) == (16, 8)


def test_runtime_policy_state_keeps_progress_state_per_connection():
    model = FakePlannerModel()
    runtime_state = RuntimePolicyState()

    initial = runtime_state.progress_state_input(
        model,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert initial is not None
    assert initial.current_stage.tolist() == [[1.0, 1.0, 1.0]]

    updated = ProgressState(
        completed_events=torch.full((1, 3), 2.0),
        current_stage=torch.full((1, 3), 3.0),
    )
    model.last_progress_planner_output = FakePlannerOutput(updated)
    runtime_state.store_progress_state(model)

    restored = runtime_state.progress_state_input(
        model,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert restored is not None
    assert restored.completed_events.tolist() == [[2.0, 2.0, 2.0]]
    assert restored.current_stage.tolist() == [[3.0, 3.0, 3.0]]

    runtime_state.reset(model)
    assert runtime_state.progress_state is None
    assert model.reset_count == 0
