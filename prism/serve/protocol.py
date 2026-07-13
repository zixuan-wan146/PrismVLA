from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np

from prism.schema import PolicyInput


@dataclass(frozen=True)
class PolicyRequest(PolicyInput):
    """Validated wire request compatible with the model-facing PolicyInput."""

    return_debug: bool = False


def policy_request_from_mapping(data: Mapping[str, Any]) -> PolicyRequest:
    if not isinstance(data, Mapping):
        raise TypeError(f"Policy request must be a mapping, got {type(data).__name__}")

    required_fields = (
        "benchmark",
        "prompt",
        "images_by_view",
        "history_images_by_view",
        "history_step_ages",
        "history_valid_mask",
        "state",
        "action_dim",
    )
    allowed_fields = {*required_fields, "robot_key", "return_debug"}
    unknown = sorted(str(field) for field in data if field not in allowed_fields)
    if unknown:
        raise ValueError(f"Unsupported policy request fields: {unknown}")
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required policy request fields: {missing}")

    benchmark = str(data["benchmark"])
    prompt = "" if data["prompt"] is None else str(data["prompt"])
    images_by_view = _validate_images_by_view(data["images_by_view"])
    history_step_ages = _validate_history_step_ages(data["history_step_ages"])
    history_valid_mask = _validate_history_valid_mask(data["history_valid_mask"], history_step_ages.shape[0])
    history_images_by_view = _validate_history_images_by_view(
        data["history_images_by_view"],
        images_by_view,
        history_step_ages.shape[0],
    )
    state = _validate_vector(data["state"], "state")
    action_dim = _positive_int(data["action_dim"], "action_dim")
    robot_key = data.get("robot_key")
    if robot_key is not None:
        robot_key = str(robot_key)

    return PolicyRequest(
        benchmark=benchmark,
        prompt=prompt,
        images_by_view=images_by_view,
        history_images_by_view=history_images_by_view,
        history_step_ages=history_step_ages,
        history_valid_mask=history_valid_mask,
        state=state,
        action_dim=action_dim,
        robot_key=robot_key,
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
        "history_images_by_view": {
            view_name: np.ascontiguousarray(images, dtype=np.uint8)
            for view_name, images in request.history_images_by_view.items()
        },
        "history_step_ages": np.asarray(request.history_step_ages, dtype=np.int32),
        "history_valid_mask": np.asarray(request.history_valid_mask, dtype=np.bool_),
        "state": np.asarray(request.state, dtype=np.float32),
        "action_dim": int(request.action_dim),
        "robot_key": request.robot_key,
    }
    if request.return_debug:
        payload["return_debug"] = True
    return payload


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


def _validate_history_step_ages(value: Any) -> np.ndarray:
    ages = np.asarray(value)
    if ages.ndim != 1 or ages.shape[0] != 2:
        raise ValueError(f"history_step_ages must have shape [2], got {ages.shape}")
    if not np.issubdtype(ages.dtype, np.integer):
        raise ValueError("history_step_ages must contain integers")
    ages = ages.astype(np.int32, copy=False)
    if tuple(ages.tolist()) != (6, 3):
        raise ValueError(f"history_step_ages must equal the accepted [6, 3] schedule, got {ages.tolist()}")
    return np.ascontiguousarray(ages)


def _validate_history_valid_mask(value: Any, history_count: int) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != (history_count,):
        raise ValueError(f"history_valid_mask must have shape [{history_count}], got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("history_valid_mask must be boolean")
    return np.ascontiguousarray(mask, dtype=np.bool_)


def _validate_history_images_by_view(
    value: Any,
    current_images_by_view: Mapping[str, np.ndarray],
    history_count: int,
) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("history_images_by_view must be an object mapping view name to image sequence")
    if tuple(value) != tuple(current_images_by_view):
        raise ValueError(
            "history_images_by_view must have the same ordered view names as images_by_view: "
            f"expected {tuple(current_images_by_view)}, got {tuple(value)}"
        )
    history: dict[str, np.ndarray] = {}
    for view_name, current_image in current_images_by_view.items():
        images = np.asarray(value[view_name])
        expected_shape = (history_count, *current_image.shape)
        if images.shape != expected_shape:
            raise ValueError(
                f"history_images_by_view[{view_name!r}] must have shape {expected_shape}, got {images.shape}"
            )
        if not np.issubdtype(images.dtype, np.number) and not np.issubdtype(images.dtype, np.bool_):
            raise ValueError(f"history_images_by_view[{view_name!r}] must contain numeric pixels")
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
            raise ValueError(f"history_images_by_view[{view_name!r}] pixels must be finite and in 0..255")
        history[view_name] = np.ascontiguousarray(images, dtype=np.uint8)
    return history


def _validate_vector(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive, got {parsed}")
    return parsed


def parse_action_response(
    message: Any,
    *,
    horizon: int,
    min_action_dim: int,
) -> list[list[float]]:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if min_action_dim <= 0:
        raise ValueError(f"min_action_dim must be positive, got {min_action_dim}")

    payload = _decode_legacy_json(message)
    if isinstance(payload, Mapping):
        if "error" in payload:
            raise RuntimeError(f"Prism server returned error: {payload['error']}")
        if "actions" not in payload:
            raise ValueError(f"Action response object must contain 'actions', got keys: {sorted(payload.keys())}")
        payload = payload["actions"]

    if isinstance(payload, np.ndarray):
        payload = payload.tolist()
    if not isinstance(payload, list):
        raise ValueError(f"Action response must be a list or array, got {type(payload).__name__}")
    if len(payload) < horizon:
        raise ValueError(f"Action response has {len(payload)} step(s), expected at least horizon {horizon}")

    actions: list[list[float]] = []
    for step, row in enumerate(payload[:horizon]):
        if isinstance(row, np.ndarray):
            row = row.tolist()
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"Action at step {step} must be a sequence, got {type(row).__name__}")
        if len(row) < min_action_dim:
            raise ValueError(f"Action at step {step} has dimension {len(row)}, expected at least {min_action_dim}")
        actions.append([_to_finite_float(value, step, dim) for dim, value in enumerate(row)])
    return actions


def _decode_legacy_json(message: Any) -> Any:
    if not isinstance(message, str):
        return message
    try:
        return json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Action response is not valid JSON: {exc}") from exc


def _to_finite_float(value: Any, step: int, dim: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action value at step {step}, dim {dim} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Action value at step {step}, dim {dim} must be finite, got {parsed}")
    return parsed
