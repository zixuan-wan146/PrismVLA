from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import logging
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from experiments.libero.config import LiberoClientConfig, configure_mujoco_environment
from experiments.libero.data import LIBERO_IMAGE_TRANSFORM
from prism.config import as_bool, load_config, parse_profile_env, print_dry_run, run_with_environment
from prism.data.normalization import decode_gripper_for_environment, decode_gripper_open
from prism.serve.client import PolicyClient, WebSocketPolicyClient
from prism.serve.history import HistoryPrecomputeSchedule
from prism.serve.protocol import (
    HistoryObservationRequest,
    PolicyRequest,
    parse_action_response as parse_policy_action_response,
)
from prism.utils.result_writer import write_json_result_atomic
from prism.utils.run_metadata import build_run_metadata
from prism.utils.seeding import set_global_seed


LIBERO_BENCHMARK = "libero"
LIBERO_VIEW_ORDER = ("primary", "wrist")
LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7
LIBERO_CONTROL_DIM = 7
LIBERO_MOTION_LOW = np.full(6, -1.0, dtype=np.float32)
LIBERO_MOTION_HIGH = np.full(6, 1.0, dtype=np.float32)


def parse_action_response(
    message: Any,
    horizon: int,
    min_action_dim: int = LIBERO_CONTROL_DIM,
) -> list[list[float]]:
    return parse_policy_action_response(
        message,
        horizon=horizon,
        min_action_dim=min_action_dim,
    )


def to_libero_action(action: Sequence[float], control_dim: int = LIBERO_CONTROL_DIM) -> list[float]:
    if len(action) < control_dim:
        raise ValueError(f"Action dimension {len(action)} is smaller than LIBERO control dim {control_dim}")
    if control_dim != LIBERO_CONTROL_DIM:
        raise ValueError(f"LIBERO control dim must be {LIBERO_CONTROL_DIM}, got {control_dim}")
    values = np.array(action[:control_dim], dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("LIBERO action must contain only finite values")
    values[:6] = np.clip(values[:6], LIBERO_MOTION_LOW, LIBERO_MOTION_HIGH)
    libero_action = values.astype(np.float64).tolist()
    libero_action[6] = float(
        decode_gripper_for_environment(
            np.asarray(libero_action[6], dtype=np.float32),
            LIBERO_BENCHMARK,
        )
    )
    return libero_action


LIBERO_ENV_VIEW_TO_CACHE_VIEW = {
    "agentview_image": "primary",
    "robot0_eye_in_hand_image": "wrist",
}


def build_libero_images_by_view(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        cache_view: _canonicalize_libero_image(obs[env_key])
        for env_key, cache_view in LIBERO_ENV_VIEW_TO_CACHE_VIEW.items()
    }


def _canonicalize_libero_image(image: Any) -> np.ndarray:
    if LIBERO_IMAGE_TRANSFORM != "rotate_180":
        raise ValueError(f"unsupported LIBERO image transform {LIBERO_IMAGE_TRANSFORM!r}")
    return np.ascontiguousarray(
        np.rot90(np.asarray(image, dtype=np.uint8), 2),
        dtype=np.uint8,
    )


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
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator


def build_request_from_observation(
    obs: Mapping[str, Any],
    prompt: str,
    *,
    stream_id: str,
    memory_generation: int,
    robot_key: str | None = LIBERO_BENCHMARK,
    executed_actions: np.ndarray | None = None,
    executed_action_valid_mask: np.ndarray | None = None,
) -> PolicyRequest:
    images_by_view = build_libero_images_by_view(obs)
    return PolicyRequest(
        benchmark=LIBERO_BENCHMARK,
        prompt=str(prompt or ""),
        images_by_view=images_by_view,
        state=build_libero_state(obs),
        action_dim=LIBERO_ACTION_DIM,
        stream_id=stream_id,
        memory_generation=memory_generation,
        robot_key=robot_key,
        executed_actions=executed_actions,
        executed_action_valid_mask=executed_action_valid_mask,
    )


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
        suite_name: _summarize_subset([episode for episode in episodes if episode["task_suite"] == suite_name])
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
    episodes = [_episode_to_dict(result) for result in results]
    payload = {
        "config": _serialize_config(config),
        "metadata": dict(metadata) if metadata is not None else build_run_metadata(),
        "summary": summarize_episode_results(episodes),
        "episodes": episodes,
    }
    return write_json_result_atomic(result_path, payload)


def _summarize_subset(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_episodes = len(episodes)
    successful_episodes = sum(1 for episode in episodes if episode["success"])
    success_decision_steps = [int(episode["decision_steps"]) for episode in episodes if episode["success"]]
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
    payload = asdict(result) if isinstance(result, EpisodeResult) else dict(result)
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
        return {key: value for key, value in vars(config).items() if not key.startswith("_")}
    return {"repr": repr(config)}


LIBERO_DUMMY_ACTION = [0.0] * 7


LOG = logging.getLogger(__name__)


def build_policy_request(
    obs: Any,
    prompt: str,
    *,
    stream_id: str,
    memory_generation: int,
    executed_actions: np.ndarray | None = None,
    executed_action_valid_mask: np.ndarray | None = None,
) -> PolicyRequest:
    return build_request_from_observation(
        obs,
        prompt,
        stream_id=stream_id,
        memory_generation=memory_generation,
        executed_actions=executed_actions,
        executed_action_valid_mask=executed_action_valid_mask,
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


def get_libero_env(task: Any, config: LiberoClientConfig, seed: int | None = None):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    seed = config.seed if seed is None else seed
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=config.camera_resolution,
        camera_widths=config.camera_resolution,
    )
    env.seed(seed)
    return env, task.language


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
    policy_client: PolicyClient | None = None,
) -> list[EpisodeResult]:
    from libero.libero import benchmark

    horizon = config.horizon if horizon is None else horizon
    num_episodes = config.num_episodes if num_episodes is None else num_episodes
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_start = min(config.task_offset, num_tasks_in_suite)
    task_stop = num_tasks_in_suite
    if config.task_limit > 0:
        task_stop = min(task_start + config.task_limit, num_tasks_in_suite)

    LOG.info("Number of tasks: %s", num_tasks_in_suite)
    total_success = 0
    total_episodes = 0
    total_decision_steps = 0
    total_success_decision_steps = 0
    suite_results: list[EpisodeResult] = []

    client = policy_client or WebSocketPolicyClient(
        server_url,
        connect_timeout_seconds=config.connect_timeout_seconds,
        inference_timeout_seconds=config.inference_timeout_seconds,
    )
    async with client:
        LOG.info("===========================Start task suite %s========================", task_suite_name)
        for task_id in range(task_start, task_stop):
            LOG.info("task_id=%s", task_id)
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env = None
            try:
                env, task_description = get_libero_env(task, config, seed=config.seed)
                LOG.info("\n========= Start task%s: %s =========", task_id + 1, task_description)
                task_success = 0
                episode_start = min(config.episode_offset, len(initial_states))
                episode_stop = min(episode_start + num_episodes, len(initial_states))
                episode_indices = range(episode_start, episode_stop)
                task_episodes = len(episode_indices)

                for episode_id in episode_indices:
                    LOG.info("===== Task %s | Episode %s =====", task_id, episode_id + 1)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_id])
                    for _ in range(10):
                        obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)

                    episode_result = await _rollout_episode(
                        client=client,
                        env=env,
                        initial_obs=obs,
                        prompt=str(task_description),
                        stream_id=f"libero:{task_suite_name}:{task_id}:{episode_id}",
                        horizon=horizon,
                        max_steps=max_steps,
                    )
                    decision_steps = int(episode_result["decision_steps"])
                    control_steps = int(episode_result["control_steps"])
                    episode_done = bool(episode_result["success"])
                    failure_reason = str(episode_result["failure_reason"])
                    frames = episode_result["frames"]
                    if episode_done:
                        task_success += 1
                        total_success += 1
                        total_success_decision_steps += decision_steps

                    video_path = save_video(
                        frames,
                        f"task{task_id + 1}_episode{episode_id + 1}.mp4",
                        fps=config.video_fps,
                        save_dir=os.path.join(config.video_dir, task_suite_name),
                    )
                    total_decision_steps += decision_steps
                    suite_results.append(
                        EpisodeResult(
                            task_suite=task_suite_name,
                            task_id=task_id,
                            episode_id=episode_id,
                            task_description=str(task_description),
                            success=episode_done,
                            decision_steps=decision_steps,
                            control_steps=control_steps,
                            failure_reason="" if episode_done else failure_reason,
                            video_path=video_path,
                        )
                    )
                    if episode_done:
                        LOG.info("Task %s | Episode %s: Success", task_id, episode_id + 1)
                    else:
                        LOG.info("Task %s | Episode %s: Fail (%s)", task_id, episode_id + 1, failure_reason)

                LOG.info(
                    "========= Task %s Summary: %s/%s Successful =========",
                    task_id + 1,
                    task_success,
                    task_episodes,
                )
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


async def _rollout_episode(
    *,
    client: PolicyClient,
    env: Any,
    initial_obs: Any,
    prompt: str,
    stream_id: str,
    horizon: int,
    max_steps: int,
) -> dict[str, Any]:
    history_schedule = HistoryPrecomputeSchedule()
    if horizon != history_schedule.replan_stride:
        raise ValueError(
            f"History precompute requires horizon={history_schedule.replan_stride}, got {horizon}"
        )
    obs = initial_obs
    decision_steps = 0
    control_steps = 0
    frames: list[np.ndarray] = []
    gripper_values: list[float] = []
    failure_reason = ""
    previous_executed_actions = np.zeros(
        (horizon, LIBERO_ACTION_DIM),
        dtype=np.float32,
    )
    previous_executed_action_valid_mask = np.zeros((horizon,), dtype=np.bool_)
    await client.reset_history(stream_id)

    while control_steps < max_steps:
        step = decision_steps
        decision_steps += 1
        request = build_policy_request(
            obs,
            prompt,
            stream_id=stream_id,
            memory_generation=history_schedule.current_generation,
            executed_actions=previous_executed_actions,
            executed_action_valid_mask=previous_executed_action_valid_mask,
        )
        result = await client.infer(request)
        LOG.debug("[Step %s] Send observation", step)
        try:
            actions = parse_action_response(result, horizon=horizon)
            LOG.debug("[Step %s] received actions (gripper=%s)", step, actions[0][6])
        except Exception as exc:
            failure_reason = f"action_parse_error: {exc}"
            LOG.error("Action parsing failed: %s, content: %s", exc, result)
            break

        episode_done = False
        episode_failed = False
        current_executed_actions = np.zeros_like(previous_executed_actions)
        current_executed_action_valid_mask = np.zeros_like(
            previous_executed_action_valid_mask
        )
        for chunk_step, action_values in enumerate(actions, start=1):
            action = to_libero_action(action_values)
            gripper_values.append(float(action[6]))
            try:
                obs, reward, done, info = env.step(action)
                control_steps += 1
            except ValueError as exc:
                failure_reason = f"invalid_action: {exc}"
                LOG.error("Action is not valid: %s", exc)
                episode_failed = True
                break

            except Exception as exc:
                failure_reason = f"env_step_error: {exc}"
                LOG.error("LIBERO environment step failed: %s", exc)
                episode_failed = True
                break

            canonical_executed_action = np.asarray(action, dtype=np.float32)
            canonical_executed_action[6] = float(
                decode_gripper_open(np.asarray(action_values[6], dtype=np.float32))
            )
            current_executed_actions[chunk_step - 1] = canonical_executed_action
            current_executed_action_valid_mask[chunk_step - 1] = True

            post_step_images = build_libero_images_by_view(obs)
            capture_target = history_schedule.target_for_step(chunk_step)
            if capture_target is not None:
                await client.push_history_observation(
                    HistoryObservationRequest(
                        benchmark=LIBERO_BENCHMARK,
                        images_by_view=post_step_images,
                        stream_id=stream_id,
                        target_generation=capture_target.target_generation,
                        slot=capture_target.slot,
                        robot_key=LIBERO_BENCHMARK,
                    )
                )
            frames.append(np.hstack([post_step_images["primary"], post_step_images["wrist"]]))
            LOG.debug("[Step %s] reward=%.2f, done=%s, info=%s", step, reward, done, info)
            if done:
                LOG.info("Task completed")
                episode_done = True
                break
            if control_steps >= max_steps:
                break
        if episode_done:
            _log_gripper_distribution(gripper_values)
            return {
                "success": True,
                "decision_steps": decision_steps,
                "control_steps": control_steps,
                "failure_reason": "",
                "frames": frames,
            }
        if episode_failed:
            break
        if control_steps < max_steps:
            previous_executed_actions = current_executed_actions
            previous_executed_action_valid_mask = current_executed_action_valid_mask
            history_schedule.advance_generation()

    _log_gripper_distribution(gripper_values)
    return {
        "success": False,
        "decision_steps": decision_steps,
        "control_steps": control_steps,
        "failure_reason": failure_reason or "max_steps_exhausted",
        "frames": frames,
    }


def _log_gripper_distribution(values: list[float]) -> None:
    if not values:
        return
    positive = sum(1 for value in values if value >= 0.0)
    negative = len(values) - positive
    LOG.info(
        "Episode gripper sign distribution: close_ratio(raw>=0,+1)=%.4f negative_ratio=%.4f count=%s",
        positive / len(values),
        negative / len(values),
        len(values),
    )


def evaluate(config: LiberoClientConfig | None = None) -> int:
    config = LiberoClientConfig.from_env() if config is None else config
    configure_mujoco_environment(config)
    configure_logging(config)
    np.random.seed(config.seed)
    random.seed(config.seed)
    run_metadata = build_run_metadata()

    all_results: list[EpisodeResult] = []
    for name, max_steps in zip(config.task_suites, config.max_steps):
        all_results.extend(
            asyncio.run(
                run(
                    config.server_url,
                    config=config,
                    max_steps=max_steps,
                    num_episodes=config.num_episodes,
                    horizon=config.horizon,
                    task_suite_name=name,
                )
            )
        )
        result_path = write_result_summary(
            config.result_file,
            config=config,
            results=all_results,
            metadata=run_metadata,
        )
        LOG.info("LIBERO result summary saved: %s", result_path)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PrismVLA on LIBERO")
    parser.add_argument("--config", default="experiments/libero/configs/eval.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = load_config(args.config, overrides=args.overrides)
    if profile.data.benchmark != LIBERO_BENCHMARK:
        raise ValueError(f"Expected a LIBERO profile, got {profile.data.benchmark!r}")

    set_global_seed(profile.runtime.seed)
    environ = dict(os.environ)
    environ.update(parse_profile_env(profile.raw.get("profile_env", "")))
    config = LiberoClientConfig.from_env(environ)
    if as_bool(profile.raw.get("dry_run", False)):
        configure_mujoco_environment(config, environ)
        return print_dry_run(LIBERO_BENCHMARK, config)
    return int(run_with_environment(environ, lambda: evaluate(config)))


if __name__ == "__main__":
    raise SystemExit(main())
