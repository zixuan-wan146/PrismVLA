from __future__ import annotations

# --- migrated from src/prism/benchmarks/base.py ---
from dataclasses import dataclass
import os
from typing import Protocol

from prism.eval.metadata import build_run_metadata
from prism.eval.profiles import as_bool, parse_profile_env, print_dry_run, run_with_environment
from prism.serve.protocol import PolicyRequest

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkRunner",
    "BenchmarkSpec",
    "build_run_metadata",
    "parse_profile_env",
]



@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    view_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    short_memory_offsets: tuple[int, ...]
    replan_stride: int

class BenchmarkAdapter(Protocol):
    spec: BenchmarkSpec

    def build_request(
        self,
        obs,
        prompt: str,
        history,
        *,
        reset_memory: bool,
    ) -> PolicyRequest:
        ...

    def parse_model_action(self, action_values):
        ...




class BenchmarkRunner:
    """Config-dispatched benchmark facade used by scripts/eval.py."""

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def run(self):
        benchmark = str(getattr(getattr(self.cfg, "data", None), "benchmark", "")).lower()
        raw = dict(getattr(self.cfg, "raw", {}) or {})
        environ = dict(os.environ)
        environ.update(parse_profile_env(raw.get("profile_env", "")))

        if benchmark == "libero":
            from prism.eval.libero import LiberoClientConfig, configure_mujoco_environment, main as libero_main

            config = LiberoClientConfig.from_env(environ)
            if as_bool(raw.get("dry_run", False)):
                configure_mujoco_environment(config, environ)
                return print_dry_run("libero", config)
            return run_with_environment(environ, libero_main)

        if benchmark == "calvin":
            from prism.eval.calvin import CalvinClientConfig, configure_calvin_environment, main as calvin_main

            config = CalvinClientConfig.from_env(environ)
            if as_bool(raw.get("dry_run", False)):
                configure_calvin_environment(config, environ)
                return print_dry_run("calvin", config)
            return run_with_environment(environ, calvin_main)

        raise ValueError(f"Unsupported evaluation benchmark {benchmark!r}; expected 'libero' or 'calvin'")
