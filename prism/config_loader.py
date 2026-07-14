from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import yaml

from prism.utils.paths import sanitize_project_paths


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42


@dataclass(frozen=True)
class PrismDataConfig:
    benchmark: Literal["libero", "calvin"]


@dataclass(frozen=True)
class PrismConfig:
    data: PrismDataConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> PrismConfig:
    """Load a benchmark evaluation profile.

    Model and experiment parameters intentionally do not live here while the new
    policy architecture is being designed.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config root must be a mapping: {config_path}")
    raw = dict(loaded)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested_override(raw, key.strip(), _parse_override_value(value))

    benchmark = str(raw.get("benchmark", "")).lower()
    if benchmark not in {"libero", "calvin"}:
        raise ValueError(f"Unsupported benchmark {benchmark!r}; expected 'libero' or 'calvin'")
    return PrismConfig(
        data=PrismDataConfig(benchmark=benchmark),  # type: ignore[arg-type]
        runtime=RuntimeConfig(seed=int(raw.get("seed", 42))),
        raw=raw,
    )


def _parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_nested_override(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not dotted_key or any(not part for part in parts):
        raise ValueError(f"override key must be non-empty dotted text, got {dotted_key!r}")
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            child: dict[str, Any] = {}
            current[part] = child
        elif isinstance(existing, Mapping):
            child = dict(existing)
            current[part] = child
        else:
            raise ValueError(
                f"override {dotted_key!r} cannot descend through non-mapping key {part!r}"
            )
        current = child
    current[parts[-1]] = value


def parse_profile_env(profile_env: Any) -> dict[str, str]:
    if profile_env in (None, ""):
        return {}
    if isinstance(profile_env, Mapping):
        return {str(key): str(value) for key, value in profile_env.items()}
    parsed: dict[str, str] = {}
    for raw_line in str(profile_env).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"profile_env line must be KEY=VALUE, got {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"profile_env line has an empty key: {raw_line!r}")
        parsed[key] = value.strip()
    return parsed


def merge_profile_environment(
    profile_env: Any,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply profile values as defaults while preserving the ambient environment."""

    merged = parse_profile_env(profile_env)
    ambient = os.environ if environ is None else environ
    merged.update({str(key): str(value) for key, value in ambient.items()})
    return merged


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def run_with_environment(environ: Mapping[str, str], fn: Callable[[], Any]) -> Any:
    previous = dict(os.environ)
    os.environ.update(environ)
    try:
        return fn()
    finally:
        os.environ.clear()
        os.environ.update(previous)


def print_dry_run(benchmark: str, config: Any) -> int:
    payload = asdict(config) if is_dataclass(config) else dict(config)
    safe_payload = sanitize_project_paths(payload, Path.cwd())
    print(f"{benchmark} eval dry-run ok")
    for key in sorted(safe_payload):
        print(f"{key}: {safe_payload[key]}")
    return 0
