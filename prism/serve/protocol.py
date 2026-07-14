from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np


MAX_STREAM_ID_LENGTH = 256
EXECUTED_ACTION_HORIZON = 8


@dataclass(frozen=True)
class PolicyRequest:
    """Current-observation inference request referencing server-side memory."""

    benchmark: str
    prompt: str
    images_by_view: Mapping[str, np.ndarray]
    state: np.ndarray
    action_dim: int
    stream_id: str
    memory_generation: int
    robot_key: str | None = None
    return_debug: bool = False
    executed_actions: np.ndarray | None = None
    executed_action_valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class HistoryObservationRequest:
    """Transient two-camera observation encoded immediately by the model server."""

    benchmark: str
    images_by_view: Mapping[str, np.ndarray]
    stream_id: str
    target_generation: int
    slot: int
    robot_key: str | None = None


@dataclass(frozen=True)
class HistoryResetRequest:
    """Explicit episode/subtask boundary for one connection-local stream."""

    stream_id: str


def policy_request_from_mapping(data: Mapping[str, Any]) -> PolicyRequest:
    if not isinstance(data, Mapping):
        raise TypeError(f"Policy request must be a mapping, got {type(data).__name__}")

    required_fields = (
        "benchmark",
        "prompt",
        "images_by_view",
        "state",
        "action_dim",
        "stream_id",
        "memory_generation",
    )
    allowed_fields = {
        *required_fields,
        "robot_key",
        "return_debug",
        "executed_actions",
        "executed_action_valid_mask",
    }
    unknown = sorted(str(field) for field in data if field not in allowed_fields)
    if unknown:
        raise ValueError(f"Unsupported policy request fields: {unknown}")
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required policy request fields: {missing}")

    benchmark = str(data["benchmark"])
    prompt = "" if data["prompt"] is None else str(data["prompt"])
    images_by_view = _validate_images_by_view(data["images_by_view"])
    state = _validate_vector(data["state"], "state")
    action_dim = _positive_int(data["action_dim"], "action_dim")
    stream_id = _validate_stream_id(data["stream_id"])
    memory_generation = _nonnegative_int(data["memory_generation"], "memory_generation")
    robot_key = data.get("robot_key")
    if robot_key is not None:
        robot_key = str(robot_key)
    return_debug = data.get("return_debug", False)
    if not isinstance(return_debug, (bool, np.bool_)):
        raise ValueError(f"return_debug must be boolean, got {return_debug!r}")
    executed_actions, executed_action_valid_mask = _validate_executed_action_history(
        data.get("executed_actions"),
        data.get("executed_action_valid_mask"),
        action_dim=action_dim,
    )

    return PolicyRequest(
        benchmark=benchmark,
        prompt=prompt,
        images_by_view=images_by_view,
        state=state,
        action_dim=action_dim,
        stream_id=stream_id,
        memory_generation=memory_generation,
        robot_key=robot_key,
        return_debug=bool(return_debug),
        executed_actions=executed_actions,
        executed_action_valid_mask=executed_action_valid_mask,
    )


def policy_request_to_mapping(request: PolicyRequest) -> dict[str, Any]:
    if not isinstance(request, PolicyRequest):
        raise TypeError(f"request must be PolicyRequest, got {type(request).__name__}")
    request = policy_request_from_mapping(
        {
            "benchmark": request.benchmark,
            "prompt": request.prompt,
            "images_by_view": request.images_by_view,
            "state": request.state,
            "action_dim": request.action_dim,
            "stream_id": request.stream_id,
            "memory_generation": request.memory_generation,
            "robot_key": request.robot_key,
            "return_debug": request.return_debug,
            "executed_actions": request.executed_actions,
            "executed_action_valid_mask": request.executed_action_valid_mask,
        }
    )
    payload: dict[str, Any] = {
        "benchmark": request.benchmark,
        "prompt": request.prompt,
        "images_by_view": {
            view_name: np.ascontiguousarray(image, dtype=np.uint8)
            for view_name, image in request.images_by_view.items()
        },
        "state": np.asarray(request.state, dtype=np.float32),
        "action_dim": int(request.action_dim),
        "stream_id": request.stream_id,
        "memory_generation": int(request.memory_generation),
        "robot_key": request.robot_key,
        "executed_actions": np.asarray(request.executed_actions, dtype=np.float32),
        "executed_action_valid_mask": np.asarray(
            request.executed_action_valid_mask,
            dtype=np.bool_,
        ),
    }
    if request.return_debug:
        payload["return_debug"] = True
    return payload


def history_observation_from_mapping(data: Mapping[str, Any]) -> HistoryObservationRequest:
    if not isinstance(data, Mapping):
        raise TypeError(f"History observation must be a mapping, got {type(data).__name__}")
    required_fields = (
        "benchmark",
        "images_by_view",
        "stream_id",
        "target_generation",
        "slot",
    )
    allowed_fields = {*required_fields, "robot_key"}
    unknown = sorted(str(field) for field in data if field not in allowed_fields)
    if unknown:
        raise ValueError(f"Unsupported history observation fields: {unknown}")
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required history observation fields: {missing}")
    robot_key = data.get("robot_key")
    return HistoryObservationRequest(
        benchmark=str(data["benchmark"]),
        images_by_view=_validate_images_by_view(data["images_by_view"]),
        stream_id=_validate_stream_id(data["stream_id"]),
        target_generation=_positive_int(data["target_generation"], "target_generation"),
        slot=_history_slot(data["slot"]),
        robot_key=None if robot_key is None else str(robot_key),
    )


def history_observation_to_mapping(request: HistoryObservationRequest) -> dict[str, Any]:
    if not isinstance(request, HistoryObservationRequest):
        raise TypeError(
            f"request must be HistoryObservationRequest, got {type(request).__name__}"
        )
    request = history_observation_from_mapping(
        {
            "benchmark": request.benchmark,
            "images_by_view": request.images_by_view,
            "stream_id": request.stream_id,
            "target_generation": request.target_generation,
            "slot": request.slot,
            "robot_key": request.robot_key,
        }
    )
    return {
        "benchmark": request.benchmark,
        "images_by_view": {
            view_name: np.ascontiguousarray(image, dtype=np.uint8)
            for view_name, image in request.images_by_view.items()
        },
        "stream_id": request.stream_id,
        "target_generation": int(request.target_generation),
        "slot": int(request.slot),
        "robot_key": request.robot_key,
    }


def history_reset_from_mapping(data: Mapping[str, Any]) -> HistoryResetRequest:
    if not isinstance(data, Mapping):
        raise TypeError(f"History reset must be a mapping, got {type(data).__name__}")
    unknown = sorted(str(field) for field in data if field != "stream_id")
    if unknown:
        raise ValueError(f"Unsupported history reset fields: {unknown}")
    if "stream_id" not in data:
        raise ValueError("Missing required history reset field: stream_id")
    return HistoryResetRequest(stream_id=_validate_stream_id(data["stream_id"]))


def history_reset_to_mapping(request: HistoryResetRequest) -> dict[str, Any]:
    if not isinstance(request, HistoryResetRequest):
        raise TypeError(f"request must be HistoryResetRequest, got {type(request).__name__}")
    return {"stream_id": _validate_stream_id(request.stream_id)}


def _validate_images_by_view(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("images_by_view must be an object mapping view name to image")
    if not value:
        raise ValueError("images_by_view must contain at least one image")
    images = {}
    for view_name, image in value.items():
        if not isinstance(view_name, str):
            raise ValueError(f"view names must be strings, got {type(view_name).__name__}")
        name = view_name
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
    if array.dtype != np.dtype(np.uint8):
        raise ValueError(f"{field_name} must have dtype uint8, got dtype={array.dtype}")
    return np.ascontiguousarray(array)


def _validate_vector(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def _validate_executed_action_history(
    actions_value: Any,
    mask_value: Any,
    *,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if actions_value is None and mask_value is None:
        return (
            np.zeros((EXECUTED_ACTION_HORIZON, action_dim), dtype=np.float32),
            np.zeros((EXECUTED_ACTION_HORIZON,), dtype=np.bool_),
        )
    if actions_value is None or mask_value is None:
        raise ValueError(
            "executed_actions and executed_action_valid_mask must be provided together"
        )
    actions = np.asarray(actions_value)
    expected_actions = (EXECUTED_ACTION_HORIZON, action_dim)
    if actions.shape != expected_actions or not np.issubdtype(actions.dtype, np.floating):
        raise ValueError(
            f"executed_actions must be floating with shape {expected_actions}, got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("executed_actions must contain only finite values")
    mask = np.asarray(mask_value)
    if mask.dtype != np.bool_ or mask.shape != (EXECUTED_ACTION_HORIZON,):
        raise ValueError(
            "executed_action_valid_mask must be boolean with shape "
            f"({EXECUTED_ACTION_HORIZON},)"
        )
    if np.any(actions[~mask] != 0):
        raise ValueError("executed_actions must be zero at invalid positions")
    return np.ascontiguousarray(actions, dtype=np.float32), np.ascontiguousarray(mask)


def _validate_stream_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"stream_id must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError("stream_id must not be empty")
    if len(value) > MAX_STREAM_ID_LENGTH:
        raise ValueError(
            f"stream_id must be at most {MAX_STREAM_ID_LENGTH} characters, got {len(value)}"
        )
    return value


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return int(value)


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _integer(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive, got {parsed}")
    return parsed


def _nonnegative_int(value: Any, field_name: str) -> int:
    parsed = _integer(value, field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative, got {parsed}")
    return parsed


def _history_slot(value: Any) -> int:
    slot = _nonnegative_int(value, "slot")
    if slot not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")
    return slot


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
