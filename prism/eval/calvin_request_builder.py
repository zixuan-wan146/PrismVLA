from __future__ import annotations

from prism.eval.calvin_history import CalvinObservationHistory
from prism.eval.calvin_observation import build_calvin_images_by_view, build_calvin_state
from prism.eval.calvin_spec import CALVIN_SPEC

# --- migrated from src/prism/benchmarks/calvin/request_builder.py ---
from collections.abc import Mapping
from typing import Any

import numpy as np

from prism.serve.engine import PolicyRequest



def build_request_from_observation(
    obs: Mapping[str, Any],
    prompt: str,
    *,
    history: CalvinObservationHistory | None = None,
    current_step: int | None = None,
    reset_memory: bool = False,
    executed_actions: Any | None = None,
    executed_action_mask: Any | None = None,
    robot_key: str | None = CALVIN_SPEC.name,
) -> dict[str, Any]:
    short_memory_images_by_offset = None
    if history is not None:
        if current_step is None:
            raise ValueError("current_step is required when history is provided")
        short_memory_images_by_offset = history.images_by_offset(
            current_step=int(current_step),
            offsets=CALVIN_SPEC.short_memory_offsets,
        )
    request = PolicyRequest(
        benchmark=CALVIN_SPEC.name,
        prompt=str(prompt or ""),
        images_by_view=build_calvin_images_by_view(obs),
        state=build_calvin_state(obs),
        action_dim=CALVIN_SPEC.action_dim,
        robot_key=robot_key,
        reset_memory=bool(reset_memory),
        short_memory_images_by_offset=short_memory_images_by_offset,
        executed_actions=None if executed_actions is None else np.asarray(executed_actions, dtype=np.float32),
        executed_action_mask=None if executed_action_mask is None else np.asarray(executed_action_mask, dtype=bool),
    )
    return policy_request_to_json(request)


def policy_request_to_json(request: PolicyRequest) -> dict[str, Any]:
    payload = {
        "benchmark": request.benchmark,
        "prompt": request.prompt,
        "images_by_view": {
            view_name: image.astype("uint8").tolist()
            for view_name, image in request.images_by_view.items()
        },
        "state": request.state.astype("float32").tolist(),
        "action_dim": int(request.action_dim),
        "robot_key": request.robot_key,
        "reset_memory": bool(request.reset_memory),
    }
    if request.short_memory_images_by_offset is not None:
        payload["short_memory_images_by_offset"] = {
            str(offset): {
                view_name: image.astype("uint8").tolist()
                for view_name, image in images_by_view.items()
            }
            for offset, images_by_view in request.short_memory_images_by_offset.items()
        }
    if request.executed_actions is not None:
        payload["executed_actions"] = request.executed_actions.astype("float32").tolist()
    if request.executed_action_mask is not None:
        payload["executed_action_mask"] = request.executed_action_mask.astype(bool).astype(int).tolist()
    return payload

