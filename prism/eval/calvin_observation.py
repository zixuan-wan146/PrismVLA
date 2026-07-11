from __future__ import annotations

# --- migrated from src/prism/benchmarks/calvin/observation.py ---
from collections.abc import Mapping
from typing import Any

import numpy as np


CALVIN_ENV_VIEW_TO_CACHE_VIEW = {
    "rgb_static": "image",
    "rgb_gripper": "wrist_image",
}


def build_calvin_images_by_view(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        cache_view: np.ascontiguousarray(_extract_rgb(obs, env_key), dtype=np.uint8)
        for env_key, cache_view in CALVIN_ENV_VIEW_TO_CACHE_VIEW.items()
    }


def build_calvin_state(obs: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(_extract_robot_obs(obs), dtype=np.float32).reshape(-1)


def _extract_rgb(obs: Mapping[str, Any], key: str) -> np.ndarray:
    rgb_obs = obs.get("rgb_obs")
    if isinstance(rgb_obs, Mapping) and key in rgb_obs:
        return np.asarray(rgb_obs[key], dtype=np.uint8)
    if key in obs:
        return np.asarray(obs[key], dtype=np.uint8)
    raise KeyError(f"CALVIN observation has no RGB image {key!r}")


def _extract_robot_obs(obs: Mapping[str, Any]) -> np.ndarray:
    if "robot_obs" in obs:
        return np.asarray(obs["robot_obs"], dtype=np.float32)
    state_obs = obs.get("state_obs")
    if isinstance(state_obs, Mapping) and "robot_obs" in state_obs:
        return np.asarray(state_obs["robot_obs"], dtype=np.float32)
    if "state" in obs:
        return np.asarray(obs["state"], dtype=np.float32)
    raise KeyError("CALVIN observation has no robot_obs")

