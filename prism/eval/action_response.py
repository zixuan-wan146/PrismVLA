from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np


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
