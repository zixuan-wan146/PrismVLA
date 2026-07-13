from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, is_dataclass
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from experiments.calvin.config import (
    CalvinClientConfig,
    configure_calvin_environment,
)
from prism.config import as_bool, load_config, parse_profile_env, print_dry_run, run_with_environment
from prism.data.normalization import decode_gripper_for_environment
from prism.serve.client import PolicyClient, WebSocketPolicyClient
from prism.serve.history import SparseHistoryBuffer, SparseHistoryPayload, empty_history_payload
from prism.serve.protocol import PolicyRequest, parse_action_response as parse_policy_action_response
from prism.utils.run_metadata import build_run_metadata
from prism.utils.seeding import set_global_seed


CALVIN_BENCHMARK = "calvin"
CALVIN_VIEW_ORDER = ("primary", "wrist")
CALVIN_STATE_DIM = 8
CALVIN_ACTION_DIM = 7


CALVIN_CONTROL_DIM = 7
CALVIN_RELATIVE_MOTION_LOW = np.full(6, -1.0, dtype=np.float32)
CALVIN_RELATIVE_MOTION_HIGH = np.full(6, 1.0, dtype=np.float32)


def parse_action_response(
    message: Any,
    horizon: int,
    min_action_dim: int = CALVIN_CONTROL_DIM,
) -> list[list[float]]:
    return parse_policy_action_response(
        message,
        horizon=horizon,
        min_action_dim=min_action_dim,
    )


def to_calvin_action(
    action: Sequence[float],
    *,
    control_dim: int = CALVIN_CONTROL_DIM,
) -> list[float]:
    if len(action) < control_dim:
        raise ValueError(f"Action dimension {len(action)} is smaller than CALVIN control dim {control_dim}")
    if control_dim != CALVIN_CONTROL_DIM:
        raise ValueError(f"CALVIN control dim must be {CALVIN_CONTROL_DIM}, got {control_dim}")
    values = np.asarray(action[:control_dim], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("CALVIN action must contain only finite values")
    values[:6] = np.clip(values[:6], CALVIN_RELATIVE_MOTION_LOW, CALVIN_RELATIVE_MOTION_HIGH)
    calvin_action = values.astype(np.float64).tolist()
    calvin_action[6] = float(
        decode_gripper_for_environment(
            np.asarray(calvin_action[6], dtype=np.float32),
            CALVIN_BENCHMARK,
        )
    )
    return calvin_action


CALVIN_ENV_VIEW_TO_CACHE_VIEW = {
    "rgb_static": "primary",
    "rgb_gripper": "wrist",
}


def build_calvin_images_by_view(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        cache_view: np.ascontiguousarray(_extract_rgb(obs, env_key), dtype=np.uint8)
        for env_key, cache_view in CALVIN_ENV_VIEW_TO_CACHE_VIEW.items()
    }


def build_calvin_state(obs: Mapping[str, Any]) -> np.ndarray:
    robot_obs = np.asarray(_extract_robot_obs(obs), dtype=np.float32).reshape(-1)
    if robot_obs.size < 8:
        raise ValueError(f"CALVIN robot_obs must contain at least 8 values, got {robot_obs.size}")
    # The canonical training layout is TCP xyz/rpy, an explicit zero pad, and
    # gripper width. Raw CALVIN robot_obs stores width at index 6, followed by
    # seven arm joints and a signed gripper command.
    return np.concatenate((robot_obs[:6], np.zeros((1,), dtype=np.float32), robot_obs[6:7])).astype(
        np.float32, copy=False
    )


def _extract_rgb(obs: Mapping[str, Any], key: str) -> np.ndarray:
    rgb_obs = obs.get("rgb_obs")
    if isinstance(rgb_obs, Mapping) and key in rgb_obs:
        return np.asarray(rgb_obs[key], dtype=np.uint8)
    if key in obs:
        return np.asarray(obs[key], dtype=np.uint8)
    raise KeyError(f"CALVIN observation has no RGB image {key!r}")


def _extract_robot_obs(obs: Mapping[str, Any]) -> np.ndarray:
    if "robot_obs" in obs:
        return np.asarray(obs["robot_obs"], dtype=np.float32)
    state_obs = obs.get("state_obs")
    if isinstance(state_obs, Mapping) and "robot_obs" in state_obs:
        return np.asarray(state_obs["robot_obs"], dtype=np.float32)
    if "state" in obs:
        return np.asarray(obs["state"], dtype=np.float32)
    raise KeyError("CALVIN observation has no robot_obs")


def build_request_from_observation(
    obs: Mapping[str, Any],
    prompt: str,
    *,
    history: SparseHistoryPayload | None = None,
    robot_key: str | None = CALVIN_BENCHMARK,
) -> PolicyRequest:
    images_by_view = build_calvin_images_by_view(obs)
    history = empty_history_payload(images_by_view, view_names=CALVIN_VIEW_ORDER) if history is None else history
    return PolicyRequest(
        benchmark=CALVIN_BENCHMARK,
        prompt=str(prompt or ""),
        images_by_view=images_by_view,
        history_images_by_view=history.images_by_view,
        history_step_ages=history.step_ages,
        history_valid_mask=history.valid_mask,
        state=build_calvin_state(obs),
        action_dim=CALVIN_ACTION_DIM,
        robot_key=robot_key,
    )


@dataclass(frozen=True)
class SequenceResult:
    sequence_id: int
    initial_state: str
    subtasks: list[str]
    successful_subtasks: int
    success: bool
    decision_steps: int
    control_steps: int
    failed_subtask: str = ""
    failure_reason: str = ""
    video_paths: list[str] | None = None


def summarize_sequence_results(results: Sequence[SequenceResult | Mapping[str, Any]]) -> dict[str, Any]:
    sequences = [_sequence_to_dict(result) for result in results]
    total_sequences = len(sequences)
    successful_sequences = sum(1 for sequence in sequences if sequence["success"])
    successful_counts = [int(sequence["successful_subtasks"]) for sequence in sequences]
    total_subtasks = sum(len(sequence["subtasks"]) for sequence in sequences)
    successful_subtasks = sum(successful_counts)
    return {
        "total_sequences": total_sequences,
        "successful_sequences": successful_sequences,
        "failed_sequences": total_sequences - successful_sequences,
        "sequence_success_rate": successful_sequences / total_sequences if total_sequences else 0.0,
        "average_successful_subtasks": _mean(successful_counts),
        "total_subtasks": total_subtasks,
        "successful_subtasks": successful_subtasks,
        "subtask_success_rate": successful_subtasks / total_subtasks if total_subtasks else 0.0,
        "chain_success_rates": _chain_success_rates(
            successful_counts, max_chain_length=_max_sequence_length(sequences)
        ),
        "average_decision_steps": _mean([int(sequence["decision_steps"]) for sequence in sequences]),
        "average_control_steps": _mean([int(sequence["control_steps"]) for sequence in sequences]),
        "task_info": _task_info(sequences),
        "successful_sequence_ids": [sequence["sequence_id"] for sequence in sequences if sequence["success"]],
    }


def write_result_summary(
    path: str | Path,
    *,
    config: Any,
    results: Sequence[SequenceResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    result_path = Path(path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = [_sequence_to_dict(result) for result in results]
    payload = {
        "config": _serialize_config(config),
        "metadata": dict(metadata) if metadata is not None else build_run_metadata(),
        "summary": summarize_sequence_results(sequences),
        "sequences": sequences,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result_path


def _chain_success_rates(successful_counts: Sequence[int], *, max_chain_length: int) -> dict[str, float]:
    if not successful_counts:
        return {str(index): 0.0 for index in range(1, max_chain_length + 1)}
    return {
        str(index): sum(1 for count in successful_counts if count >= index) / len(successful_counts)
        for index in range(1, max_chain_length + 1)
    }


def _task_info(sequences: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int | float]]:
    task_success: dict[str, int] = {}
    task_total: dict[str, int] = {}
    for sequence in sequences:
        successful_subtasks = int(sequence["successful_subtasks"])
        subtasks = [str(task) for task in sequence["subtasks"]]
        for index, subtask in enumerate(subtasks):
            task_total[subtask] = task_total.get(subtask, 0) + 1
            if index < successful_subtasks:
                task_success[subtask] = task_success.get(subtask, 0) + 1
            else:
                task_success.setdefault(subtask, 0)
    return {
        task: {
            "success": task_success.get(task, 0),
            "total": total,
            "success_rate": task_success.get(task, 0) / total if total else 0.0,
        }
        for task, total in sorted(task_total.items())
    }


def _max_sequence_length(sequences: Sequence[Mapping[str, Any]]) -> int:
    if not sequences:
        return 5
    return max(len(sequence["subtasks"]) for sequence in sequences)


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sequence_to_dict(result: SequenceResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, SequenceResult):
        payload = asdict(result)
    else:
        payload = dict(result)
    subtasks = payload.get("subtasks") or []
    if not isinstance(subtasks, Sequence) or isinstance(subtasks, (str, bytes, bytearray)):
        raise ValueError("CALVIN sequence result subtasks must be a list")
    successful_subtasks = int(payload["successful_subtasks"])
    payload["sequence_id"] = int(payload["sequence_id"])
    payload["initial_state"] = str(payload.get("initial_state") or "")
    payload["subtasks"] = [str(task) for task in subtasks]
    payload["successful_subtasks"] = successful_subtasks
    payload["success"] = bool(payload.get("success", successful_subtasks >= len(payload["subtasks"])))
    payload["decision_steps"] = int(payload["decision_steps"])
    payload["control_steps"] = int(payload["control_steps"])
    payload["failed_subtask"] = str(payload.get("failed_subtask") or "")
    payload["failure_reason"] = str(payload.get("failure_reason") or "")
    payload["video_paths"] = [str(path) for path in (payload.get("video_paths") or [])]
    return payload


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return {key: value for key, value in vars(config).items() if not key.startswith("_")}
    return {"repr": repr(config)}


LOG = logging.getLogger(__name__)


class CalvinEnvWrapperRaw:
    def __init__(
        self,
        abs_datasets_dir: str | Path,
        observation_space: dict[str, list[str]],
        *,
        show_gui: bool = False,
        **kwargs: Any,
    ) -> None:
        from calvin_env.envs.play_table_env import get_env

        self.env = get_env(abs_datasets_dir, show_gui=show_gui, obs_space=observation_space, **kwargs)
        self.observation_space_keys = observation_space
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        self._closed = False

    def step(self, action_values: list[float]):
        action = np.asarray(action_values, dtype=np.float32).reshape(-1)
        if self.relative_actions:
            if action.size != 7:
                raise ValueError(f"CALVIN relative action must have 7 values, got {action.size}")
            env_action: Any = action.tolist()
        else:
            if action.size == 7:
                env_action = np.split(action, [3, 6])
            elif action.size == 8:
                env_action = np.split(action, [3, 7])
            else:
                raise ValueError(f"CALVIN absolute action must have 7 or 8 values, got {action.size}")
        return self.env.step(env_action)

    def reset(self, *, scene_obs: Any = None, robot_obs: Any = None):
        if scene_obs is not None or robot_obs is not None:
            return self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        return self.env.reset()

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        return self.env.get_obs()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.env, "close", None)
        if callable(close):
            try:
                close()
            finally:
                # CALVIN's env destructor calls close() again; make that second call a no-op.
                setattr(self.env, "close", lambda *args, **kwargs: None)


def default_observation_space() -> dict[str, list[str]]:
    return {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }


def configure_logging(config: CalvinClientConfig) -> None:
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


def make_env(config: CalvinClientConfig) -> CalvinEnvWrapperRaw:
    dataset_path = Path(config.dataset_path).expanduser()
    validation_path = dataset_path if dataset_path.name == "validation" else dataset_path / "validation"
    return CalvinEnvWrapperRaw(validation_path, default_observation_space(), show_gui=config.show_gui)


def load_task_oracle(config: CalvinClientConfig):
    import hydra
    from omegaconf import OmegaConf

    conf_dir = Path(config.calvin_root).expanduser() / "calvin_models" / "conf"
    task_cfg_path = conf_dir / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    if not task_cfg_path.exists():
        raise FileNotFoundError(f"CALVIN task oracle config not found: {task_cfg_path}")
    task_cfg = OmegaConf.load(task_cfg_path)
    return hydra.utils.instantiate(task_cfg)


def load_validation_annotations(config: CalvinClientConfig) -> dict[str, list[str]]:
    from omegaconf import OmegaConf

    if config.annotations_path:
        annotations_path = Path(config.annotations_path).expanduser()
    else:
        annotations_path = (
            Path(config.calvin_root).expanduser()
            / "calvin_models"
            / "conf"
            / "annotations"
            / "new_playtable_validation.yaml"
        )
    if not annotations_path.exists():
        raise FileNotFoundError(f"CALVIN validation annotations not found: {annotations_path}")
    raw_annotations = OmegaConf.load(annotations_path)
    annotations = OmegaConf.to_container(raw_annotations, resolve=True)
    if not isinstance(annotations, dict):
        raise ValueError(f"CALVIN annotations must be a mapping: {annotations_path}")
    normalized: dict[str, list[str]] = {}
    for key, value in annotations.items():
        if isinstance(value, list):
            normalized[str(key)] = [str(item) for item in value]
        else:
            normalized[str(key)] = [str(value)]
    return normalized


def load_eval_sequences(config: CalvinClientConfig) -> list[tuple[Any, list[str]]]:
    from calvin_agent.evaluation.multistep_sequences import get_sequences

    total_to_request = config.num_sequences + config.sequence_offset
    sequences = list(get_sequences(total_to_request))
    selected = sequences[config.sequence_offset : config.sequence_offset + config.num_sequences]
    if len(selected) != config.num_sequences:
        raise ValueError(
            f"CALVIN get_sequences returned {len(sequences)} sequence(s), cannot select "
            f"{config.num_sequences} sequence(s) from offset {config.sequence_offset}"
        )
    return [(initial_state, [str(task) for task in eval_sequence]) for initial_state, eval_sequence in selected]


def build_policy_request(
    obs: dict[str, Any],
    *,
    prompt: str,
    history: SparseHistoryPayload | None = None,
) -> PolicyRequest:
    return build_request_from_observation(obs, prompt, history=history)


async def run_calvin_eval(
    config: CalvinClientConfig | None = None,
    *,
    policy_client: PolicyClient | None = None,
) -> list[SequenceResult]:
    config = CalvinClientConfig.from_env() if config is None else config
    configure_calvin_environment(config)
    configure_logging(config)
    random.seed(config.seed)
    np.random.seed(config.seed)

    LOG.info("Loading CALVIN task oracle and validation annotations")
    task_oracle = load_task_oracle(config)
    annotations = load_validation_annotations(config)
    sequences = load_eval_sequences(config)

    LOG.info("Creating CALVIN environment from %s", config.dataset_path)
    env = make_env(config)
    results: list[SequenceResult] = []
    client = policy_client or WebSocketPolicyClient(config.server_url)
    try:
        async with client:
            LOG.info("Policy client ready for %s", config.server_url)
            for local_index, (initial_state, eval_sequence) in enumerate(sequences):
                sequence_id = config.sequence_offset + local_index
                LOG.info("Sequence %s: %s", sequence_id, " -> ".join(eval_sequence))
                result = await evaluate_sequence(
                    policy_client=client,
                    env=env,
                    task_oracle=task_oracle,
                    annotations=annotations,
                    config=config,
                    sequence_id=sequence_id,
                    initial_state=initial_state,
                    eval_sequence=eval_sequence,
                    log=LOG,
                )
                results.append(result)
                result_path = write_result_summary(config.result_file, config=config, results=results)
                LOG.info(
                    "Sequence %s complete: %s/%s subtasks, results saved to %s",
                    sequence_id,
                    result.successful_subtasks,
                    len(result.subtasks),
                    result_path,
                )
    finally:
        try:
            env.close()
        except Exception as exc:
            LOG.warning("Failed to close CALVIN env: %s", exc)
    return results


async def evaluate_sequence(
    *,
    policy_client: PolicyClient,
    env: CalvinEnvWrapperRaw,
    task_oracle: Any,
    annotations: dict[str, list[str]],
    config: CalvinClientConfig,
    sequence_id: int,
    initial_state: Any,
    eval_sequence: list[str],
    log: logging.Logger,
) -> SequenceResult:
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    successful_subtasks = 0
    total_decision_steps = 0
    total_control_steps = 0
    failure_reason = ""
    failed_subtask = ""
    video_paths: list[str] = []
    for subtask_index, subtask in enumerate(eval_sequence):
        prompt = _prompt_for_subtask(annotations, subtask)
        rollout = await rollout_subtask(
            policy_client=policy_client,
            env=env,
            task_oracle=task_oracle,
            subtask=subtask,
            prompt=prompt,
            config=config,
            sequence_id=sequence_id,
            subtask_index=subtask_index,
            log=log,
        )
        total_decision_steps += int(rollout["decision_steps"])
        total_control_steps += int(rollout["control_steps"])
        video_paths.extend(rollout["video_paths"])
        if rollout["success"]:
            successful_subtasks += 1
            continue
        failure_reason = str(rollout["failure_reason"])
        failed_subtask = subtask
        break

    return SequenceResult(
        sequence_id=sequence_id,
        initial_state=_initial_state_repr(initial_state),
        subtasks=list(eval_sequence),
        successful_subtasks=successful_subtasks,
        success=successful_subtasks == len(eval_sequence),
        decision_steps=total_decision_steps,
        control_steps=total_control_steps,
        failed_subtask=failed_subtask,
        failure_reason=failure_reason,
        video_paths=video_paths,
    )


async def rollout_subtask(
    *,
    policy_client: PolicyClient,
    env: CalvinEnvWrapperRaw,
    task_oracle: Any,
    subtask: str,
    prompt: str,
    config: CalvinClientConfig,
    sequence_id: int,
    subtask_index: int,
    log: logging.Logger,
) -> dict[str, Any]:
    obs = env.get_obs()
    start_info = env.get_info()
    frames: list[np.ndarray] = []
    decision_steps = 0
    control_steps = 0
    history_buffer = SparseHistoryBuffer(view_names=CALVIN_VIEW_ORDER)
    while control_steps < config.max_steps_per_subtask:
        decision_steps += 1
        current_images = build_calvin_images_by_view(obs)
        history = history_buffer.consume(current_images)
        request = build_policy_request(
            obs,
            prompt=prompt,
            history=history,
        )
        response = await policy_client.infer(request)
        try:
            action_chunk = parse_action_response(response, horizon=config.horizon)
        except Exception as exc:
            log.error("CALVIN action parsing failed for %s: %s", subtask, exc)
            video_paths = _maybe_save_video(frames, config, sequence_id, subtask_index, subtask, "parse_error")
            return _rollout_result(False, decision_steps, control_steps, f"action_parse_error: {exc}", video_paths)

        for chunk_step, action_values in enumerate(action_chunk, start=1):
            action = to_calvin_action(action_values)
            try:
                obs, _reward, _done, current_info = env.step(action)
            except Exception as exc:
                log.error("CALVIN env step failed for %s: %s", subtask, exc)
                video_paths = _maybe_save_video(frames, config, sequence_id, subtask_index, subtask, "env_error")
                return _rollout_result(
                    False,
                    decision_steps,
                    control_steps,
                    f"env_step_error: {exc}",
                    video_paths,
                )
            control_steps += 1
            history_buffer.capture(chunk_step, build_calvin_images_by_view(obs))
            if config.save_video:
                frames.append(_compose_video_frame(obs))

            if _check_success(task_oracle, start_info, current_info, subtask):
                video_paths = _maybe_save_video(frames, config, sequence_id, subtask_index, subtask, "success")
                return _rollout_result(
                    True,
                    decision_steps,
                    control_steps,
                    "",
                    video_paths,
                )
            if control_steps >= config.max_steps_per_subtask:
                break
    video_paths = _maybe_save_video(frames, config, sequence_id, subtask_index, subtask, "fail")
    return _rollout_result(
        False,
        decision_steps,
        control_steps,
        "max_steps_exhausted",
        video_paths,
    )


def _rollout_result(
    success: bool,
    decision_steps: int,
    control_steps: int,
    failure_reason: str,
    video_paths: list[str],
) -> dict[str, Any]:
    return {
        "success": bool(success),
        "decision_steps": int(decision_steps),
        "control_steps": int(control_steps),
        "failure_reason": str(failure_reason),
        "video_paths": list(video_paths),
    }


def _check_success(task_oracle: Any, start_info: Any, current_info: Any, subtask: str) -> bool:
    return len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0


def _prompt_for_subtask(annotations: dict[str, list[str]], subtask: str) -> str:
    prompts = annotations.get(subtask)
    if not prompts:
        return subtask.replace("_", " ")
    return str(prompts[0])


def _compose_video_frame(obs: dict[str, Any]) -> np.ndarray:
    images = build_calvin_images_by_view(obs)
    static_img = images["primary"]
    gripper_img = images["wrist"]
    min_height = min(static_img.shape[0], gripper_img.shape[0])
    if static_img.shape[0] != min_height:
        static_img = static_img[:min_height]
    if gripper_img.shape[0] != min_height:
        gripper_img = gripper_img[:min_height]
    return np.hstack([copy.deepcopy(static_img), copy.deepcopy(gripper_img)])


def _maybe_save_video(
    frames: list[np.ndarray],
    config: CalvinClientConfig,
    sequence_id: int,
    subtask_index: int,
    subtask: str,
    status: str,
) -> list[str]:
    if not config.save_video or not frames:
        return []
    import imageio.v2 as imageio

    video_dir = Path(config.video_dir).expanduser()
    video_dir.mkdir(parents=True, exist_ok=True)
    safe_subtask = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in subtask)
    path = video_dir / f"sequence{sequence_id:04d}_subtask{subtask_index + 1}_{safe_subtask}_{status}.mp4"
    imageio.mimsave(path, frames, fps=config.video_fps)
    return [str(path)]


def _initial_state_repr(initial_state: Any) -> str:
    try:
        return json.dumps(initial_state, default=str, sort_keys=True)
    except TypeError:
        return repr(initial_state)


def evaluate(config: CalvinClientConfig | None = None) -> int:
    asyncio.run(run_calvin_eval(config))
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PrismVLA on CALVIN")
    parser.add_argument("--config", default="experiments/calvin/configs/eval.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    profile = load_config(args.config, overrides=args.overrides)
    if profile.data.benchmark != CALVIN_BENCHMARK:
        raise ValueError(f"Expected a CALVIN profile, got {profile.data.benchmark!r}")

    set_global_seed(profile.runtime.seed)
    environ = dict(os.environ)
    environ.update(parse_profile_env(profile.raw.get("profile_env", "")))
    config = CalvinClientConfig.from_env(environ)
    if as_bool(profile.raw.get("dry_run", False)):
        configure_calvin_environment(config, environ)
        return print_dry_run(CALVIN_BENCHMARK, config)
    return int(run_with_environment(environ, lambda: evaluate(config)))


if __name__ == "__main__":
    raise SystemExit(main())
