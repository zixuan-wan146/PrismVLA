from __future__ import annotations

import asyncio
from dataclasses import replace
import logging
import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("PRISM_RUN_CALVIN_INTEGRATION") != "1",
    reason="run this test in the dedicated CALVIN simulator environment",
)


class _NeverSuccessfulTaskOracle:
    def get_task_info_for_set(self, start_info, current_info, tasks):
        del start_info, current_info, tasks
        return set()


def test_real_calvin_rollout_preserves_control_budget_and_precomputes_history():
    from experiments.calvin.config import CalvinClientConfig, configure_calvin_environment
    from experiments.calvin.eval import make_env, rollout_subtask
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
        CalvinClientConfig.from_env(),
        horizon=8,
        max_steps_per_subtask=9,
        save_video=False,
    )
    configure_calvin_environment(config)
    env = make_env(config)
    try:
        env.reset()
        result = asyncio.run(
            rollout_subtask(
                policy_client=InProcessPolicyClient(
                    infer,
                    reset_history=resets.append,
                    push_history_observation=push_history,
                ),
                env=env,
                task_oracle=_NeverSuccessfulTaskOracle(),
                subtask="open_drawer",
                prompt="open the drawer",
                config=config,
                sequence_id=0,
                subtask_index=0,
                log=logging.getLogger(__name__),
            )
        )
    finally:
        env.close()

    assert result["decision_steps"] == 2
    assert result["control_steps"] == 9
    assert result["failure_reason"] == "max_steps_exhausted"
    assert [request.memory_generation for request in requests] == [0, 1]
    assert all(not hasattr(request, "history_images_by_view") for request in requests)
    assert [(request.target_generation, request.slot) for request in history_observations] == [(1, 0), (1, 1)]
    assert resets == ["calvin:0:0:open_drawer"]
    assert tuple(requests[0].images_by_view) == ("primary", "wrist")
    assert requests[0].state.shape == (8,)
    assert not requests[0].executed_action_valid_mask.any()
    assert requests[1].executed_action_valid_mask.all()
    assert np.all(np.isin(requests[1].executed_actions[:, 6], [0.0, 1.0]))
