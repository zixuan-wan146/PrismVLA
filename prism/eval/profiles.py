from __future__ import annotations

from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from prism.utils.paths import sanitize_project_paths


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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def run_with_environment(environ: Mapping[str, str], fn: Callable[[], Any]) -> Any:
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(previous)
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
