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


def test_real_libero_rollout_preserves_control_budget_and_sparse_history(monkeypatch):
    from experiments.libero.config import LiberoClientConfig
    from experiments.libero.eval import run
    from prism.serve.client import InProcessPolicyClient
    from prism.serve.protocol import policy_request_from_mapping, policy_request_to_mapping
    from prism.serve.wire import pack_message, unpack_message

    requests = []

    def infer(request):
        roundtrip = policy_request_from_mapping(unpack_message(pack_message(policy_request_to_mapping(request))))
        requests.append(roundtrip)
        return {"actions": np.zeros((8, 7), dtype=np.float32)}

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
            policy_client=InProcessPolicyClient(infer),
        )
    )

    assert len(results) == 1
    assert results[0].decision_steps == 2
    assert results[0].control_steps == 9
    assert [request.history_valid_mask.tolist() for request in requests] == [
        [False, False],
        [True, True],
    ]
    assert all(request.history_step_ages.tolist() == [6, 3] for request in requests)
    assert tuple(requests[0].images_by_view) == ("agentview_rgb", "eye_in_hand_rgb")
    assert requests[0].state.shape == (8,)
