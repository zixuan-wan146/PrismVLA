from __future__ import annotations

import asyncio
from dataclasses import replace
import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("PRISM_RUN_LIBERO_INTEGRATION") != "1",
    reason="run this test in the dedicated LIBERO simulator environment",
)


def test_real_libero_rollout_preserves_control_budget_and_precomputes_history(monkeypatch):
    from experiments.libero.config import LiberoClientConfig
    from experiments.libero.eval import run
    from prism.serve.client import InProcessPolicyClient
    from prism.serve.protocol import (
        history_observation_from_mapping,
        history_observation_to_mapping,
        policy_request_from_mapping,
        policy_request_to_mapping,
    )
    from prism.serve.wire import pack_message, unpack_message

    requests = []
    history_observations = []
    resets = []

    def infer(request):
        roundtrip = policy_request_from_mapping(unpack_message(pack_message(policy_request_to_mapping(request))))
        requests.append(roundtrip)
        return {"actions": np.zeros((8, 7), dtype=np.float32)}

    def push_history(request):
        roundtrip = history_observation_from_mapping(
            unpack_message(pack_message(history_observation_to_mapping(request)))
        )
        history_observations.append(roundtrip)

    config = replace(
        LiberoClientConfig.from_env(),
        horizon=8,
        num_episodes=1,
        task_limit=1,
        task_offset=0,
        episode_offset=0,
    )
    monkeypatch.setattr("experiments.libero.eval.save_video", lambda *args, **kwargs: "")
    results = asyncio.run(
        run(
            config.server_url,
            config=config,
            max_steps=9,
            num_episodes=1,
            horizon=8,
            task_suite_name="libero_spatial",
            policy_client=InProcessPolicyClient(
                infer,
                reset_history=resets.append,
                push_history_observation=push_history,
            ),
        )
    )

    assert len(results) == 1
    assert results[0].decision_steps == 2
    assert results[0].control_steps == 9
    assert [request.memory_generation for request in requests] == [0, 1]
    assert all(not hasattr(request, "history_images_by_view") for request in requests)
    assert [(request.target_generation, request.slot) for request in history_observations] == [(1, 0), (1, 1)]
    assert resets == ["libero:libero_spatial:0:0"]
    assert tuple(requests[0].images_by_view) == ("primary", "wrist")
    assert requests[0].state.shape == (8,)
    assert not requests[0].executed_action_valid_mask.any()
    assert requests[1].executed_action_valid_mask.all()
    assert np.all(np.isin(requests[1].executed_actions[:, 6], [0.0, 1.0]))
