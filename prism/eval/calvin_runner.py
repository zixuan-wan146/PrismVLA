from __future__ import annotations

from prism.eval.calvin_action_protocol import parse_action_response, to_calvin_action
from prism.eval.calvin_config import CalvinClientConfig, configure_calvin_environment
from prism.eval.calvin_eval_summary import SequenceResult, write_result_summary
from prism.eval.calvin_history import CalvinObservationHistory
from prism.eval.calvin_observation import build_calvin_images_by_view
from prism.eval.calvin_request_builder import build_request_from_observation
from prism.eval.calvin_spec import CALVIN_SPEC
from prism.eval.policy_client import PolicyClient, WebSocketPolicyClient
from prism.serve.protocol import PolicyRequest

# --- migrated from src/prism/benchmarks/calvin/runner.py ---
import asyncio
import copy
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

import numpy as np


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
    history: CalvinObservationHistory | None,
    current_step: int,
    reset_memory: bool,
    executed_actions: list[list[float]] | None,
    executed_action_mask: list[bool] | None,
) -> PolicyRequest:
    return build_request_from_observation(
        obs,
        prompt,
        history=history,
        current_step=current_step,
        reset_memory=reset_memory,
        executed_actions=executed_actions,
        executed_action_mask=executed_action_mask,
    )


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
    history = CalvinObservationHistory(max_offset=max(CALVIN_SPEC.short_memory_offsets))
    history.record(0, env.get_obs())
    last_executed_actions: list[list[float]] = []
    last_executed_action_mask: list[bool] = []

    for subtask_index, subtask in enumerate(eval_sequence):
        prompt = _prompt_for_subtask(annotations, subtask)
        reset_memory = config.reset_memory_scope == "subtask" or (
            config.reset_memory_scope == "sequence" and subtask_index == 0
        )
        rollout = await rollout_subtask(
            policy_client=policy_client,
            env=env,
            task_oracle=task_oracle,
            subtask=subtask,
            prompt=prompt,
            reset_memory_on_first_decision=reset_memory,
            config=config,
            sequence_id=sequence_id,
            subtask_index=subtask_index,
            history=history,
            starting_control_step=total_control_steps,
            last_executed_actions=last_executed_actions,
            last_executed_action_mask=last_executed_action_mask,
            log=log,
        )
        total_decision_steps += int(rollout["decision_steps"])
        total_control_steps += int(rollout["control_steps"])
        video_paths.extend(rollout["video_paths"])
        last_executed_actions = list(rollout["last_executed_actions"])
        last_executed_action_mask = list(rollout["last_executed_action_mask"])
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
    reset_memory_on_first_decision: bool,
    config: CalvinClientConfig,
    sequence_id: int,
    subtask_index: int,
    history: CalvinObservationHistory,
    starting_control_step: int,
    last_executed_actions: list[list[float]],
    last_executed_action_mask: list[bool],
    log: logging.Logger,
) -> dict[str, Any]:
    obs = env.get_obs()
    start_info = env.get_info()
    frames: list[np.ndarray] = []
    decision_steps = 0
    control_steps = 0
    reset_memory = reset_memory_on_first_decision
    current_last_actions = list(last_executed_actions)
    current_last_mask = list(last_executed_action_mask)

    while control_steps < config.max_steps_per_subtask:
        decision_steps += 1
        request = build_policy_request(
            obs,
            prompt=prompt,
            history=history,
            current_step=starting_control_step + control_steps,
            reset_memory=reset_memory,
            executed_actions=current_last_actions or None,
            executed_action_mask=current_last_mask or None,
        )
        reset_memory = False
        response = await policy_client.infer(request)
        try:
            action_chunk = parse_action_response(response, horizon=config.horizon)
        except Exception as exc:
            log.error("CALVIN action parsing failed for %s: %s", subtask, exc)
            video_paths = _maybe_save_video(frames, config, sequence_id, subtask_index, subtask, "parse_error")
            return _rollout_result(
                False, decision_steps, control_steps, f"action_parse_error: {exc}", video_paths, [], []
            )

        executed_this_decision: list[list[float]] = []
        executed_mask_this_decision: list[bool] = []
        for action_values in action_chunk:
            action = to_calvin_action(action_values, gripper_mode=config.gripper_mode)
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
                    executed_this_decision,
                    executed_mask_this_decision,
                )
            control_steps += 1
            global_control_step = starting_control_step + control_steps
            history.record(global_control_step, obs)
            executed_this_decision.append(action)
            executed_mask_this_decision.append(True)
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
                    executed_this_decision,
                    executed_mask_this_decision,
                )
            if control_steps >= config.max_steps_per_subtask:
                break
        current_last_actions = executed_this_decision
        current_last_mask = executed_mask_this_decision

    video_paths = _maybe_save_video(frames, config, sequence_id, subtask_index, subtask, "fail")
    return _rollout_result(
        False,
        decision_steps,
        control_steps,
        "max_steps_exhausted",
        video_paths,
        current_last_actions,
        current_last_mask,
    )


def _rollout_result(
    success: bool,
    decision_steps: int,
    control_steps: int,
    failure_reason: str,
    video_paths: list[str],
    last_executed_actions: list[list[float]],
    last_executed_action_mask: list[bool],
) -> dict[str, Any]:
    return {
        "success": bool(success),
        "decision_steps": int(decision_steps),
        "control_steps": int(control_steps),
        "failure_reason": str(failure_reason),
        "video_paths": list(video_paths),
        "last_executed_actions": list(last_executed_actions),
        "last_executed_action_mask": list(last_executed_action_mask),
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
    static_img = images["image"]
    gripper_img = images["wrist_image"]
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


def main() -> int:
    asyncio.run(run_calvin_eval())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
