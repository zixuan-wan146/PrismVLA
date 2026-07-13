from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, MutableMapping


DEFAULT_TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DEFAULT_MAX_STEPS_BY_TASK_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
DEFAULT_MAX_STEPS = [DEFAULT_MAX_STEPS_BY_TASK_SUITE[name] for name in DEFAULT_TASK_SUITES]


def _env_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value


def env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = _env_value(environ, name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_list(environ: Mapping[str, str], name: str, default: list[str]) -> list[str]:
    value = _env_value(environ, name)
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one non-empty item")
    return items


def env_int_list(environ: Mapping[str, str], name: str, default: list[int]) -> list[int]:
    value = _env_value(environ, name)
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one integer")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers, got {value!r}") from exc


def align_max_steps(max_steps: list[int], task_suites: list[str]) -> list[int]:
    if len(max_steps) == 1 and len(task_suites) > 1:
        return max_steps * len(task_suites)
    if len(max_steps) != len(task_suites):
        raise ValueError(
            "PRISM_LIBERO_MAX_STEPS must provide one integer per task suite: "
            f"got {len(max_steps)} values for {len(task_suites)} suites"
        )
    return max_steps


def default_max_steps(task_suites: list[str]) -> list[int]:
    missing = [name for name in task_suites if name not in DEFAULT_MAX_STEPS_BY_TASK_SUITE]
    if missing:
        raise ValueError(
            f"PRISM_LIBERO_MAX_STEPS is required for task suites without a default control-step budget: {missing}"
        )
    return [DEFAULT_MAX_STEPS_BY_TASK_SUITE[name] for name in task_suites]


@dataclass(frozen=True)
class LiberoClientConfig:
    horizon: int
    max_steps: list[int]
    server_url: str
    ckpt_name: str
    task_suites: list[str]
    log_dir: str
    video_dir: str
    log_file: str
    result_file: str
    num_episodes: int
    task_limit: int
    task_offset: int
    episode_offset: int
    seed: int
    mujoco_gl: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LiberoClientConfig":
        environ = os.environ if environ is None else environ
        ckpt_name = environ.get("PRISM_LIBERO_CKPT_NAME", "Prism_libero_all")
        log_dir = environ.get("PRISM_LIBERO_LOG_DIR", "./log_file")
        video_dir = environ.get("PRISM_LIBERO_VIDEO_DIR", f"./video_log_file/{ckpt_name}")
        log_file = environ.get("PRISM_LIBERO_LOG_FILE", os.path.join(log_dir, f"{ckpt_name}.txt"))
        result_file = environ.get("PRISM_LIBERO_RESULT_FILE", os.path.join(log_dir, f"{ckpt_name}_results.json"))
        task_suites = env_list(environ, "PRISM_LIBERO_TASK_SUITES", DEFAULT_TASK_SUITES)
        if _env_value(environ, "PRISM_LIBERO_MAX_STEPS") is None:
            max_steps = default_max_steps(task_suites)
        else:
            max_steps = align_max_steps(
                env_int_list(environ, "PRISM_LIBERO_MAX_STEPS", []),
                task_suites,
            )

        config = cls(
            horizon=env_int(environ, "PRISM_LIBERO_HORIZON", 8),
            max_steps=max_steps,
            server_url=environ.get("PRISM_SERVER_URI", environ.get("PRISM_LIBERO_SERVER_URL", "ws://127.0.0.1:9000")),
            ckpt_name=ckpt_name,
            task_suites=task_suites,
            log_dir=log_dir,
            video_dir=video_dir,
            log_file=log_file,
            result_file=result_file,
            num_episodes=env_int(environ, "PRISM_LIBERO_EPISODES", 10),
            task_limit=env_int(environ, "PRISM_LIBERO_TASK_LIMIT", 0),
            task_offset=env_int(environ, "PRISM_LIBERO_TASK_OFFSET", 0),
            episode_offset=env_int(environ, "PRISM_LIBERO_EPISODE_OFFSET", 0),
            seed=env_int(environ, "PRISM_LIBERO_SEED", 42),
            mujoco_gl=environ.get("PRISM_MUJOCO_GL", "osmesa"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.horizon != 8:
            raise ValueError(f"PRISM_LIBERO_HORIZON must equal the architecture horizon 8, got {self.horizon}")
        if self.num_episodes <= 0:
            raise ValueError(f"PRISM_LIBERO_EPISODES must be positive, got {self.num_episodes}")
        if self.task_limit < 0:
            raise ValueError(f"PRISM_LIBERO_TASK_LIMIT must be non-negative, got {self.task_limit}")
        if self.task_offset < 0:
            raise ValueError(f"PRISM_LIBERO_TASK_OFFSET must be non-negative, got {self.task_offset}")
        if self.episode_offset < 0:
            raise ValueError(f"PRISM_LIBERO_EPISODE_OFFSET must be non-negative, got {self.episode_offset}")
        invalid_max_steps = [value for value in self.max_steps if value <= 0]
        if invalid_max_steps:
            raise ValueError(f"PRISM_LIBERO_MAX_STEPS values must be positive, got {invalid_max_steps}")
        if self.mujoco_gl not in {"osmesa", "egl", "glfw"}:
            raise ValueError(f"PRISM_MUJOCO_GL must be one of osmesa, egl, glfw; got {self.mujoco_gl!r}")


def configure_mujoco_environment(
    config: LiberoClientConfig,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    environ = os.environ if environ is None else environ
    environ.setdefault("MUJOCO_GL", config.mujoco_gl)
    if config.mujoco_gl == "egl":
        environ.setdefault("PYOPENGL_PLATFORM", "egl")


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_STEPS_BY_TASK_SUITE",
    "DEFAULT_TASK_SUITES",
    "LiberoClientConfig",
    "align_max_steps",
    "configure_mujoco_environment",
    "default_max_steps",
    "env_int",
    "env_int_list",
    "env_list",
]
