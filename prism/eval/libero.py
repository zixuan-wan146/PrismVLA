from __future__ import annotations

# --- migrated from src/prism/benchmarks/libero/spec.py ---
from prism.eval.runner import BenchmarkSpec


LIBERO_SPEC = BenchmarkSpec(
    name="libero",
    view_names=("agentview_rgb", "eye_in_hand_rgb"),
    state_dim=8,
    action_dim=7,
    short_memory_offsets=(16, 8),
    replan_stride=16,
)

# --- migrated from src/prism/benchmarks/libero/protocol.py ---


LIBERO_BENCHMARK = LIBERO_SPEC.name
LIBERO_VIEW_ORDER = LIBERO_SPEC.view_names
LIBERO_STATE_DIM = LIBERO_SPEC.state_dim
LIBERO_ACTION_DIM = LIBERO_SPEC.action_dim
LIBERO_SHORT_MEMORY_OFFSETS = LIBERO_SPEC.short_memory_offsets
LIBERO_REPLAN_STRIDE = LIBERO_SPEC.replan_stride


__all__ = [
    "LIBERO_ACTION_DIM",
    "LIBERO_BENCHMARK",
    "LIBERO_REPLAN_STRIDE",
    "LIBERO_SHORT_MEMORY_OFFSETS",
    "LIBERO_STATE_DIM",
    "LIBERO_VIEW_ORDER",
]

# --- migrated from src/prism/benchmarks/libero/action.py ---

__all__ = ["LIBERO_CONTROL_DIM", "parse_action_response", "to_libero_action"]

# --- migrated from src/prism/benchmarks/libero/action_protocol.py ---
from collections.abc import Mapping, Sequence
import json
from typing import Any


LIBERO_CONTROL_DIM = 7


def parse_action_response(
    message: str,
    horizon: int,
    min_action_dim: int = LIBERO_CONTROL_DIM,
) -> list[list[float]]:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Action response is not valid JSON: {exc}") from exc

    if isinstance(payload, Mapping):
        if "error" in payload:
            raise RuntimeError(f"Prism server returned error: {payload['error']}")
        if "actions" not in payload:
            raise ValueError(f"Action response object must contain 'actions', got keys: {sorted(payload.keys())}")
        payload = payload["actions"]

    if not isinstance(payload, list):
        raise ValueError(f"Action response must be a list, got {type(payload).__name__}")

    if len(payload) < horizon:
        raise ValueError(f"Action response has {len(payload)} step(s), expected at least horizon {horizon}")

    actions: list[list[float]] = []
    for step, row in enumerate(payload[:horizon]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"Action at step {step} must be a sequence, got {type(row).__name__}")
        if len(row) < min_action_dim:
            raise ValueError(
                f"Action at step {step} has dimension {len(row)}, expected at least {min_action_dim}"
            )
        actions.append([_to_float(value, step, dim) for dim, value in enumerate(row)])
    return actions


def to_libero_action(action: Sequence[float], control_dim: int = LIBERO_CONTROL_DIM) -> list[float]:
    if len(action) < control_dim:
        raise ValueError(f"Action dimension {len(action)} is smaller than LIBERO control dim {control_dim}")
    libero_action = [float(value) for value in action[:control_dim]]
    # Stage1 is trained on raw LIBERO HDF5 actions, where the environment
    # gripper command is already encoded as -1/+1. Preserve that sign instead
    # of applying the OpenVLA/RLDS gripper inversion rule.
    libero_action[6] = 1.0 if libero_action[6] >= 0.0 else -1.0
    return libero_action


def _to_float(value: Any, step: int, dim: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action value at step {step}, dim {dim} is not numeric: {value!r}") from exc

# --- migrated from src/prism/benchmarks/libero/data_protocol.py ---

LIBERO_ENV_VIEW_TO_CACHE_VIEW = {
    "agentview_image": "agentview_rgb",
    "robot0_eye_in_hand_image": "eye_in_hand_rgb",
}
LIBERO_VIEW_KEYS = tuple(LIBERO_ENV_VIEW_TO_CACHE_VIEW.values())
LIBERO_ACTION_MASK = [1] * 7 + [0] * 17
LIBERO_ROBOT_KEY = "libero"

__all__ = [
    "LIBERO_ACTION_MASK",
    "LIBERO_ENV_VIEW_TO_CACHE_VIEW",
    "LIBERO_ROBOT_KEY",
    "LIBERO_VIEW_KEYS",
    "build_libero_images_by_view",
    "build_libero_state",
    "build_request_from_observation",
    "quat2axisangle",
]

# --- migrated from src/prism/benchmarks/libero/observation.py ---
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


LIBERO_ENV_VIEW_TO_CACHE_VIEW = {
    "agentview_image": "agentview_rgb",
    "robot0_eye_in_hand_image": "eye_in_hand_rgb",
}


def build_libero_images_by_view(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        cache_view: np.ascontiguousarray(obs[env_key])
        for env_key, cache_view in LIBERO_ENV_VIEW_TO_CACHE_VIEW.items()
    }


def build_libero_state(obs: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)


def quat2axisangle(quat: Sequence[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[0] < 4:
        raise ValueError(f"quat must contain at least 4 values, got shape {quat.shape}")
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den

# --- migrated from src/prism/benchmarks/libero/request_builder.py ---
from collections.abc import Mapping
from typing import Any

from prism.serve.engine import PolicyRequest



def build_request_from_observation(
    obs: Mapping[str, Any],
    prompt: str,
    *,
    history: LiberoObservationHistory | None = None,
    current_step: int | None = None,
    reset_memory: bool = False,
    executed_actions: Any | None = None,
    executed_action_mask: Any | None = None,
    robot_key: str | None = LIBERO_SPEC.name,
) -> dict[str, Any]:
    short_memory_images_by_offset = None
    if history is not None:
        if current_step is None:
            raise ValueError("current_step is required when history is provided")
        short_memory_images_by_offset = history.images_by_offset(
            current_step=int(current_step),
            offsets=LIBERO_SPEC.short_memory_offsets,
        )
    request = PolicyRequest(
        benchmark=LIBERO_SPEC.name,
        prompt=str(prompt or ""),
        images_by_view=build_libero_images_by_view(obs),
        state=build_libero_state(obs),
        action_dim=LIBERO_SPEC.action_dim,
        robot_key=robot_key,
        reset_memory=bool(reset_memory),
        short_memory_images_by_offset=short_memory_images_by_offset,
        executed_actions=None if executed_actions is None else _float_array(executed_actions),
        executed_action_mask=None if executed_action_mask is None else _bool_array(executed_action_mask),
    )
    return policy_request_to_json(request)


def policy_request_to_json(request: PolicyRequest) -> dict[str, Any]:
    payload = {
        "benchmark": request.benchmark,
        "prompt": request.prompt,
        "images_by_view": {
            view_name: image.astype("uint8").tolist()
            for view_name, image in request.images_by_view.items()
        },
        "state": request.state.astype("float32").tolist(),
        "action_dim": int(request.action_dim),
        "robot_key": request.robot_key,
        "reset_memory": bool(request.reset_memory),
    }
    if request.short_memory_images_by_offset is not None:
        payload["short_memory_images_by_offset"] = {
            str(offset): {
                view_name: image.astype("uint8").tolist()
                for view_name, image in images_by_view.items()
            }
            for offset, images_by_view in request.short_memory_images_by_offset.items()
        }
    if request.executed_actions is not None:
        payload["executed_actions"] = request.executed_actions.astype("float32").tolist()
    if request.executed_action_mask is not None:
        payload["executed_action_mask"] = request.executed_action_mask.astype(bool).astype(int).tolist()
    return payload


def _float_array(value: Any):
    import numpy as np

    return np.asarray(value, dtype=np.float32)


def _bool_array(value: Any):
    import numpy as np

    return np.asarray(value, dtype=bool)

# --- migrated from src/prism/benchmarks/libero/history.py ---
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np



class LiberoObservationHistory:
    def __init__(self, *, max_offset: int) -> None:
        self.max_offset = int(max_offset)
        if self.max_offset <= 0:
            raise ValueError(f"max_offset must be positive, got {self.max_offset}")
        self._images_by_step: dict[int, dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        self._images_by_step.clear()

    def record(self, step_index: int, obs: Mapping[str, Any]) -> None:
        step_index = int(step_index)
        self._images_by_step[step_index] = {
            view_name: np.asarray(image, dtype=np.uint8).copy()
            for view_name, image in build_libero_images_by_view(obs).items()
        }
        self._prune(current_step=step_index)

    def images_by_offset(
        self,
        *,
        current_step: int,
        offsets: Iterable[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        output = {}
        for offset in offsets:
            offset = int(offset)
            if offset <= 0:
                raise ValueError(f"short-memory offset must be positive, got {offset}")
            step_index = int(current_step) - offset
            if step_index in self._images_by_step:
                output[offset] = {
                    view_name: image.copy()
                    for view_name, image in self._images_by_step[step_index].items()
                }
        return output

    def _prune(self, *, current_step: int) -> None:
        min_step = int(current_step) - self.max_offset
        stale_steps = [step for step in self._images_by_step if step < min_step]
        for step in stale_steps:
            del self._images_by_step[step]

# --- migrated from src/prism/benchmarks/libero/config.py ---
from dataclasses import dataclass
import os
from typing import Mapping, MutableMapping


DEFAULT_TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DEFAULT_MAX_STEPS = [25, 25, 25, 95]


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
        max_steps = align_max_steps(env_int_list(environ, "PRISM_LIBERO_MAX_STEPS", DEFAULT_MAX_STEPS), task_suites)

        config = cls(
            horizon=env_int(environ, "PRISM_LIBERO_HORIZON", 32),
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

    @property
    def SERVER_URL(self) -> str:
        return self.server_url

    @property
    def SEED(self) -> int:
        return self.seed

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"PRISM_LIBERO_HORIZON must be positive, got {self.horizon}")
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

# --- migrated from src/prism/benchmarks/libero/eval_summary.py ---
from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism.eval.runner import build_run_metadata


@dataclass(frozen=True)
class EpisodeResult:
    task_suite: str
    task_id: int
    episode_id: int
    task_description: str
    success: bool
    decision_steps: int
    control_steps: int
    failure_reason: str = ""
    video_path: str = ""


def summarize_episode_results(results: Sequence[EpisodeResult | Mapping[str, Any]]) -> dict[str, Any]:
    episodes = [_episode_to_dict(result) for result in results]
    successful = [episode for episode in episodes if episode["success"]]

    suite_names = sorted({episode["task_suite"] for episode in episodes})
    suite_summaries = {
        suite_name: _summarize_subset(
            [episode for episode in episodes if episode["task_suite"] == suite_name]
        )
        for suite_name in suite_names
    }

    summary = _summarize_subset(episodes)
    summary["suites"] = suite_summaries
    summary["successful_episode_ids"] = [
        {
            "task_suite": episode["task_suite"],
            "task_id": episode["task_id"],
            "episode_id": episode["episode_id"],
        }
        for episode in successful
    ]
    return summary


def write_result_summary(
    path: str | Path,
    *,
    config: Any,
    results: Sequence[EpisodeResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [_episode_to_dict(result) for result in results]
    payload = {
        "config": _serialize_config(config),
        "metadata": dict(metadata) if metadata is not None else build_run_metadata(),
        "summary": summarize_episode_results(episodes),
        "episodes": episodes,
    }
    with result_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return result_path


def _summarize_subset(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_episodes = len(episodes)
    successful_episodes = sum(1 for episode in episodes if episode["success"])
    success_decision_steps = [
        int(episode["decision_steps"]) for episode in episodes if episode["success"]
    ]
    all_decision_steps = [int(episode["decision_steps"]) for episode in episodes]
    all_control_steps = [int(episode["control_steps"]) for episode in episodes]

    return {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "failed_episodes": total_episodes - successful_episodes,
        "success_rate": successful_episodes / total_episodes if total_episodes else 0.0,
        "average_decision_steps": _mean(all_decision_steps),
        "average_control_steps": _mean(all_control_steps),
        "average_success_decision_steps": _mean(success_decision_steps),
    }


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _episode_to_dict(result: EpisodeResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, EpisodeResult):
        payload = asdict(result)
    else:
        payload = dict(result)
    payload["success"] = bool(payload["success"])
    payload["task_id"] = int(payload["task_id"])
    payload["episode_id"] = int(payload["episode_id"])
    payload["decision_steps"] = int(payload["decision_steps"])
    payload["control_steps"] = int(payload["control_steps"])
    payload["failure_reason"] = str(payload.get("failure_reason") or "")
    payload["video_path"] = str(payload.get("video_path") or "")
    return payload


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    return {"repr": repr(config)}

# --- migrated from src/prism/benchmarks/libero/runner.py ---
import asyncio
import json
import logging
import os
import pathlib
import random
from typing import Any

import numpy as np
import websockets



LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]
LOG = logging.getLogger(__name__)


def obs_to_json_dict(
    obs: Any,
    prompt: str,
    resize_size: int = 448,
    history: LiberoObservationHistory | None = None,
    current_step: int | None = None,
    reset_memory: bool = False,
    executed_actions: list[list[float]] | None = None,
    executed_action_mask: list[bool] | None = None,
) -> dict[str, Any]:
    _ = resize_size
    return build_request_from_observation(
        obs,
        prompt,
        history=history,
        current_step=current_step,
        reset_memory=reset_memory,
        executed_actions=executed_actions,
        executed_action_mask=executed_action_mask,
    )


def configure_logging(config: LiberoClientConfig) -> None:
    os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config.log_file, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def get_libero_env(task: Any, config: LiberoClientConfig, resolution: int = 448, seed: int | None = None):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    seed = config.seed if seed is None else seed
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def save_video(frames: list[np.ndarray], filename: str, fps: int = 20, save_dir: str = "videos_2") -> str:
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if not frames:
        LOG.warning("No frames to save. File not created: %s", filepath)
        return ""

    import imageio

    imageio.mimsave(filepath, frames, fps=fps)
    LOG.info("Video saved: %s (%s frames)", filepath, len(frames))
    return filepath


async def run(
    server_url: str,
    *,
    config: LiberoClientConfig,
    max_steps: int,
    num_episodes: int | None = None,
    horizon: int | None = None,
    task_suite_name: str,
) -> list[EpisodeResult]:
    from libero.libero import benchmark

    horizon = config.horizon if horizon is None else horizon
    num_episodes = config.num_episodes if num_episodes is None else num_episodes
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_start = min(config.task_offset, num_tasks_in_suite)
    task_stop = num_tasks_in_suite
    if config.task_limit > 0:
        task_stop = min(task_start + config.task_limit, num_tasks_in_suite)
    task_ids = range(task_start, task_stop)

    LOG.info("Number of tasks: %s", num_tasks_in_suite)

    total_success = 0
    total_episodes = 0
    total_decision_steps = 0
    total_success_decision_steps = 0
    suite_results: list[EpisodeResult] = []

    async with websockets.connect(server_url, ping_interval=None, ping_timeout=None) as ws:
        LOG.info("===========================Start task suite %s========================", task_suite_name)

        for task_id in task_ids:
            LOG.info("task_id=%s", task_id)

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env = None
            try:
                env, task_description = get_libero_env(task, config, resolution=448, seed=config.seed)

                LOG.info("\n========= Start task%s: %s =========", task_id + 1, task_description)

                task_success = 0
                episode_start = min(config.episode_offset, len(initial_states))
                episode_stop = min(episode_start + num_episodes, len(initial_states))
                episode_indices = range(episode_start, episode_stop)
                task_episodes = len(episode_indices)

                for ep in episode_indices:
                    LOG.info("===== Task %s | Episode %s =====", task_id, ep + 1)

                    env.reset()

                    obs = env.set_init_state(initial_states[ep])
                    for _ in range(10):
                        obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)

                    prompt = str(task_description)
                    LOG.info(prompt)
                    episode_done = False
                    episode_failed = False
                    failure_reason = ""
                    decision_steps = 0
                    control_steps = 0
                    frames: list[np.ndarray] = []
                    history = LiberoObservationHistory(max_offset=max(LIBERO_SPEC.short_memory_offsets))
                    history.record(control_steps, obs)
                    last_executed_actions: list[list[float]] = []
                    last_executed_action_mask: list[bool] = []
                    episode_gripper_values: list[float] = []

                    for step in range(max_steps):
                        decision_steps += 1

                        send_data = obs_to_json_dict(
                            obs,
                            prompt,
                            history=history,
                            current_step=control_steps,
                            reset_memory=(step == 0),
                            executed_actions=last_executed_actions or None,
                            executed_action_mask=last_executed_action_mask or None,
                        )
                        await ws.send(json.dumps(send_data))
                        LOG.debug("[Step %s] Send observation", step)

                        result = await ws.recv()
                        try:
                            actions = parse_action_response(result, horizon=horizon)
                            LOG.debug("[Step %s] received actions (gripper=%s)", step, actions[0][6])
                        except Exception as exc:
                            failure_reason = f"action_parse_error: {exc}"
                            LOG.error("Action parsing failed: %s, content: %s", exc, result)
                            break

                        current_executed_actions: list[list[float]] = []
                        current_executed_action_mask: list[bool] = []
                        for action_values in actions:
                            action = to_libero_action(action_values)
                            episode_gripper_values.append(float(action[6]))
                            LOG.debug(action[:7])
                            LOG.debug("gripper action %s", action[6])
                            try:
                                obs, reward, done, info = env.step(action)
                                control_steps += 1
                                history.record(control_steps, obs)
                                current_executed_actions.append(action)
                                current_executed_action_mask.append(True)
                            except ValueError as exc:
                                failure_reason = f"invalid_action: {exc}"
                                LOG.error("Action is not valid: %s", exc)
                                episode_failed = True
                                break

                            frame = np.hstack(
                                [
                                    np.rot90(obs["agentview_image"], 2),
                                    np.rot90(obs["robot0_eye_in_hand_image"], 2),
                                ]
                            )
                            frames.append(frame)

                            LOG.debug("[Step %s] reward=%.2f, done=%s, info=%s", step, reward, done, info)
                            if done:
                                LOG.info("Task completed")
                                episode_done = True
                                task_success += 1
                                total_success += 1
                                total_success_decision_steps += decision_steps
                                break
                        last_executed_actions = current_executed_actions
                        last_executed_action_mask = current_executed_action_mask
                        if episode_done or episode_failed:
                            break

                    if not episode_done and not failure_reason:
                        failure_reason = "max_steps_exhausted"
                    if episode_gripper_values:
                        positive = sum(1 for value in episode_gripper_values if value >= 0.0)
                        negative = len(episode_gripper_values) - positive
                        LOG.info(
                            "Episode gripper sign distribution: close_ratio(raw>=0,+1)=%.4f negative_ratio=%.4f count=%s",
                            positive / len(episode_gripper_values),
                            negative / len(episode_gripper_values),
                            len(episode_gripper_values),
                        )

                    video_path = save_video(
                        frames,
                        f"task{task_id + 1}_episode{ep + 1}.mp4",
                        fps=30,
                        save_dir=os.path.join(config.video_dir, task_suite_name),
                    )

                    total_decision_steps += decision_steps
                    suite_results.append(
                        EpisodeResult(
                            task_suite=task_suite_name,
                            task_id=task_id,
                            episode_id=ep,
                            task_description=prompt,
                            success=episode_done,
                            decision_steps=decision_steps,
                            control_steps=control_steps,
                            failure_reason="" if episode_done else failure_reason,
                            video_path=video_path,
                        )
                    )

                    if episode_done:
                        LOG.info("Task %s | Episode %s: Success", task_id, ep + 1)
                    else:
                        LOG.info("Task %s | Episode %s: Fail (%s)", task_id, ep + 1, failure_reason)

                LOG.info("========= Task %s Summary: %s/%s Successful =========", task_id + 1, task_success, task_episodes)
                total_episodes += task_episodes
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        LOG.warning("Failed to close LIBERO env for task %s: %s", task_id, exc)

        LOG.info("\n========= Overall Task Summary =========")
        LOG.info("Total Successful Episodes: %s/%s", total_success, total_episodes)
        if total_episodes > 0:
            LOG.info("Success Rate: %.4f", total_success / total_episodes)
            LOG.info("Average Decision Steps: %.2f", total_decision_steps / total_episodes)
        if total_success > 0:
            LOG.info("Average Successful Decision Steps: %.2f", total_success_decision_steps / total_success)

    return suite_results


def main() -> int:
    config = LiberoClientConfig.from_env()
    configure_mujoco_environment(config)
    configure_logging(config)
    np.random.seed(config.seed)
    random.seed(config.seed)

    all_results: list[EpisodeResult] = []
    for name, max_steps in zip(config.task_suites, config.max_steps):
        suite_results = asyncio.run(
            run(
                config.server_url,
                config=config,
                max_steps=max_steps,
                num_episodes=config.num_episodes,
                horizon=config.horizon,
                task_suite_name=name,
            )
        )
        all_results.extend(suite_results)
        result_path = write_result_summary(config.result_file, config=config, results=all_results)
        LOG.info("LIBERO result summary saved: %s", result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# Compatibility aliases for the flat PrismVLA eval module.
import sys as _sys
for _name in ('spec', 'protocol', 'action', 'action_protocol', 'data_protocol', 'observation', 'request_builder', 'history', 'config', 'eval_summary', 'runner'):
    _sys.modules[f"{__name__}.{_name}"] = _sys.modules[__name__]
