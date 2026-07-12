from __future__ import annotations

# --- migrated from src/prism/runtime/contract.py ---
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from prism.config import TARGET_STATE_DIM


@dataclass(frozen=True)
class PolicyRequest:
    benchmark: str
    prompt: str
    images_by_view: Mapping[str, np.ndarray]
    state: np.ndarray
    action_dim: int
    robot_key: str | None = None
    reset_memory: bool = False
    short_memory_images_by_offset: Mapping[int, Mapping[str, np.ndarray]] | None = None
    executed_actions: np.ndarray | None = None
    executed_action_mask: np.ndarray | None = None
    return_debug: bool = False


def policy_request_from_mapping(data: Mapping[str, Any]) -> PolicyRequest:
    if not isinstance(data, Mapping):
        raise TypeError(f"Policy request must be a mapping, got {type(data).__name__}")

    missing = [field for field in ("benchmark", "prompt", "images_by_view", "state", "action_dim") if field not in data]
    if missing:
        raise ValueError(f"Missing required policy request fields: {missing}")

    benchmark = str(data["benchmark"])
    prompt = "" if data["prompt"] is None else str(data["prompt"])
    images_by_view = _validate_images_by_view(data["images_by_view"])
    state = _validate_vector(data["state"], "state")
    action_dim = _positive_int(data["action_dim"], "action_dim")
    robot_key = data.get("robot_key")
    if robot_key is not None:
        robot_key = str(robot_key)

    return PolicyRequest(
        benchmark=benchmark,
        prompt=prompt,
        images_by_view=images_by_view,
        state=state,
        action_dim=action_dim,
        robot_key=robot_key,
        reset_memory=bool(data.get("reset_memory", False)),
        short_memory_images_by_offset=_optional_short_memory_images(data.get("short_memory_images_by_offset")),
        executed_actions=_optional_array(data.get("executed_actions"), "executed_actions"),
        executed_action_mask=_optional_bool_vector(data.get("executed_action_mask"), "executed_action_mask"),
        return_debug=bool(data.get("return_debug", False)),
    )


def policy_request_to_mapping(request: PolicyRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark": request.benchmark,
        "prompt": request.prompt,
        "images_by_view": {
            view_name: np.ascontiguousarray(image, dtype=np.uint8)
            for view_name, image in request.images_by_view.items()
        },
        "state": np.asarray(request.state, dtype=np.float32),
        "action_dim": int(request.action_dim),
        "robot_key": request.robot_key,
        "reset_memory": bool(request.reset_memory),
    }
    if request.short_memory_images_by_offset is not None:
        payload["short_memory_images_by_offset"] = {
            int(offset): {
                view_name: np.ascontiguousarray(image, dtype=np.uint8) for view_name, image in images_by_view.items()
            }
            for offset, images_by_view in request.short_memory_images_by_offset.items()
        }
    if request.executed_actions is not None:
        payload["executed_actions"] = np.asarray(request.executed_actions, dtype=np.float32)
    if request.executed_action_mask is not None:
        payload["executed_action_mask"] = np.asarray(request.executed_action_mask, dtype=bool)
    if request.return_debug:
        payload["return_debug"] = True
    return payload


def checkpoint_normalizer_dim(config: Mapping[str, Any], default_dim: int = TARGET_STATE_DIM) -> int:
    return max(
        _positive_int_or_default(config.get("state_dim"), default_dim),
        _positive_int_or_default(config.get("per_action_dim"), default_dim),
    )


def _validate_images_by_view(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("images_by_view must be an object mapping view name to image")
    if not value:
        raise ValueError("images_by_view must contain at least one image")
    images = {}
    for view_name, image in value.items():
        name = str(view_name)
        if not name:
            raise ValueError("view names must be non-empty")
        images[name] = _validate_image_array(image, f"images_by_view[{name!r}]")
    return images


def _validate_image_array(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{field_name} must have shape HxWx3, got ndim={array.ndim}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{field_name} must have non-empty height and width, got shape={array.shape}")
    if array.shape[2] != 3:
        raise ValueError(f"{field_name} must have 3 channels, got shape={array.shape}")
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{field_name} must contain numeric pixel values, got dtype={array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite pixel values")
    if array.min() < 0 or array.max() > 255:
        raise ValueError(f"{field_name} pixel values must be in the 0..255 range")
    return np.asarray(array, dtype=np.uint8)


def _validate_vector(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _validate_binary_vector(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    invalid_values = sorted({int(item) for item in array.tolist()} - {0, 1})
    if invalid_values:
        raise ValueError(f"{field_name} must contain only 0/1 values, got {invalid_values}")
    return array


def _optional_short_memory_images(value: Any) -> dict[int, dict[str, np.ndarray]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("short_memory_images_by_offset must be an object mapping offset to images_by_view")
    output = {}
    for raw_offset, images_by_view in value.items():
        offset = _positive_int(raw_offset, "short memory offset")
        output[offset] = _validate_images_by_view(images_by_view)
    return output


def _optional_array(value: Any, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty when provided")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _optional_bool_vector(value: Any, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _validate_binary_vector(value, field_name).astype(bool)


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive, got {parsed}")
    return parsed


def _positive_int_or_default(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
