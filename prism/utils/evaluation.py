"""Shared, validated settings for benchmark evaluation clients."""

from __future__ import annotations

import math
from typing import Any, Mapping


DEFAULT_POLICY_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_POLICY_INFERENCE_TIMEOUT_SECONDS = 120.0


def env_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    return finite_positive_seconds(parsed, name)


def finite_positive_seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return parsed


__all__ = [
    "DEFAULT_POLICY_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_POLICY_INFERENCE_TIMEOUT_SECONDS",
    "env_float",
    "finite_positive_seconds",
]
