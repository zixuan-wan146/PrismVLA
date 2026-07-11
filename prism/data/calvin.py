from __future__ import annotations

# --- migrated from src/prism/dataset/calvin.py ---
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_CALVIN_VIEW_NAMES = ("image", "wrist_image")
DEFAULT_CALVIN_ROBOT_KEY = "calvin"
DEFAULT_CALVIN_DATASET_NAME = "task_ABC_D"


@dataclass(frozen=True)
class CalvinEpisodeFile:
    dataset_root: Path
    episode_index: int
    parquet_path: Path
    tasks: tuple[str, ...]
    length: int


@dataclass(frozen=True)
class CalvinFrame:
    tau: int
    images_by_view: Mapping[str, Image.Image]
    action: np.ndarray
    state_vector: np.ndarray


class CalvinEpisodeReader:
    """Read one LeRobot-format CALVIN episode without importing the simulator."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        episode_index: int,
        view_names: Sequence[str] = DEFAULT_CALVIN_VIEW_NAMES,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        if not self.dataset_root.exists():
            raise FileNotFoundError(self.dataset_root)
        self.episode_index = int(episode_index)
        self.view_names = tuple(str(name) for name in view_names)
        if not self.view_names:
            raise ValueError("view_names must contain at least one view")

        self.info = read_calvin_info(self.dataset_root)
        self.parquet_path = calvin_episode_parquet_path(self.dataset_root, self.episode_index, info=self.info)
        if not self.parquet_path.exists():
            raise FileNotFoundError(self.parquet_path)
        self._dataframe = _read_parquet(self.parquet_path)
        self.length = int(len(self._dataframe))
        if self.length <= 0:
            raise ValueError(f"CALVIN episode {self.episode_index} is empty: {self.parquet_path}")
        self.action_dim = int(np.asarray(self._dataframe.iloc[0]["actions"], dtype=np.float32).reshape(-1).shape[0])
        self.state_dim = int(np.asarray(self._dataframe.iloc[0]["state"], dtype=np.float32).reshape(-1).shape[0])

    def __len__(self) -> int:
        return self.length

    def read_frame(self, index: int) -> CalvinFrame:
        index = int(index)
        if index < 0 or index >= self.length:
            raise IndexError(f"frame index {index} out of range for episode length {self.length}")
        row = self._dataframe.iloc[index]
        images_by_view = {
            view_name: self._read_image(view_name, index, row)
            for view_name in self.view_names
        }
        return CalvinFrame(
            tau=index,
            images_by_view=images_by_view,
            action=np.asarray(row["actions"], dtype=np.float32).reshape(-1),
            state_vector=np.asarray(row["state"], dtype=np.float32).reshape(-1),
        )

    def read_future_actions(self, start: int, end: int) -> np.ndarray:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.length:
            raise IndexError(f"invalid action slice [{start}, {end}) for episode length {self.length}")
        values = [
            np.asarray(action, dtype=np.float32).reshape(-1)
            for action in self._dataframe.iloc[start:end]["actions"].tolist()
        ]
        if not values:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        return np.stack(values, axis=0).astype(np.float32, copy=False)

    def _read_image(self, view_name: str, frame_index: int, row: Any) -> Image.Image:
        if view_name in row.index:
            value = row[view_name]
            if value is not None:
                return _value_to_image(value, label=view_name)

        video_path = calvin_episode_video_path(
            self.dataset_root,
            self.episode_index,
            view_name,
            info=self.info,
        )
        if video_path.exists():
            return _read_video_frame(video_path, frame_index)
        raise FileNotFoundError(
            f"CALVIN image source for view {view_name!r} is missing. "
            f"Expected parquet column {view_name!r} or video file {video_path}."
        )


def iter_calvin_episode_files(
    dataset_root: str | Path,
    *,
    max_episodes: int | None = None,
) -> tuple[CalvinEpisodeFile, ...]:
    root = Path(dataset_root).expanduser()
    info = read_calvin_info(root)
    episodes = read_calvin_episode_metadata(root)
    output: list[CalvinEpisodeFile] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        parquet_path = calvin_episode_parquet_path(root, episode_index, info=info)
        if not parquet_path.exists():
            continue
        tasks = tuple(str(task) for task in episode.get("tasks") or ())
        length = int(episode.get("length", 0))
        if length <= 0:
            length = int(len(_read_parquet(parquet_path)))
        output.append(
            CalvinEpisodeFile(
                dataset_root=root,
                episode_index=episode_index,
                parquet_path=parquet_path,
                tasks=tasks,
                length=length,
            )
        )
        if max_episodes is not None and len(output) >= int(max_episodes):
            break
    return tuple(output)


def read_calvin_info(dataset_root: str | Path) -> dict[str, Any]:
    path = Path(dataset_root).expanduser() / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_calvin_episode_metadata(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_root).expanduser() / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    episodes = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def read_calvin_tasks(dataset_root: str | Path) -> dict[int, str]:
    path = Path(dataset_root).expanduser() / "meta" / "tasks.jsonl"
    if not path.exists():
        return {}
    tasks = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            tasks[int(payload["task_index"])] = str(payload["task"])
    return tasks


def calvin_episode_parquet_path(dataset_root: str | Path, episode_index: int, *, info: Mapping[str, Any] | None = None) -> Path:
    root = Path(dataset_root).expanduser()
    info = read_calvin_info(root) if info is None else info
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunks_size
    template = str(info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return root / template.format(episode_chunk=episode_chunk, episode_index=int(episode_index))


def calvin_episode_video_path(
    dataset_root: str | Path,
    episode_index: int,
    view_name: str,
    *,
    info: Mapping[str, Any] | None = None,
) -> Path:
    root = Path(dataset_root).expanduser()
    info = read_calvin_info(root) if info is None else info
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = int(episode_index) // chunks_size
    template = str(info.get("video_path") or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return root / template.format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
        video_key=str(view_name),
    )


def calvin_prompt_for_episode(episode: Mapping[str, Any], tasks_by_index: Mapping[int, str] | None = None) -> str:
    tasks = [str(task).strip() for task in episode.get("tasks") or () if str(task).strip()]
    if tasks:
        return "; ".join(tasks)
    task_index = episode.get("task_index")
    if task_index is not None and tasks_by_index and int(task_index) in tasks_by_index:
        return str(tasks_by_index[int(task_index)])
    return f"calvin episode {int(episode.get('episode_index', 0))}"


def _read_parquet(path: Path):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required to read LeRobot CALVIN parquet episodes") from exc
    return pd.read_parquet(path)


def _value_to_image(value: Any, *, label: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == object and array.size == 1:
        array = np.asarray(array.item())
    if array.dtype == object:
        array = np.asarray(_to_nested_builtin(value), dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"CALVIN image {label!r} must have shape HxWx3, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(array)).convert("RGB")


def _to_nested_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_nested_builtin(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_nested_builtin(item) for item in value]
    return value


def _read_video_frame(path: Path, frame_index: int) -> Image.Image:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("av is required to decode LeRobot CALVIN video frames") from exc

    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == int(frame_index):
                return frame.to_image().convert("RGB")
    raise IndexError(f"video {path} does not contain frame {frame_index}")

# --- migrated from src/prism/dataset/calvin_progress_warmup.py ---
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from prism.data.calvin import DEFAULT_CALVIN_ROBOT_KEY
from prism.data.calvin import DEFAULT_CALVIN_VIEW_NAMES
from prism.data.calvin import CalvinEpisodeReader
from prism.data.libero import LIBERO_PROGRESS_WARMUP_FORMAT
from prism.data.libero import LIBERO_PROGRESS_WARMUP_VERSION
from prism.data.libero import ActionNormalizer
from prism.data.libero import VLSummaryEncoder
from prism.data.libero import _encode_target_intent
from prism.data.libero import _flush_vl_summary_batch
from prism.data.libero import action_normalizer_from_stats
from prism.data.libero import build_libero_progress_windows
from prism.data.libero import load_action_segment_autoencoder
from prism.data.libero import resolve_storage_dtype
from prism.models.planner import ActionSegmentAutoencoder


CALVIN_PROGRESS_WARMUP_FORMAT = LIBERO_PROGRESS_WARMUP_FORMAT
CALVIN_PROGRESS_WARMUP_VERSION = LIBERO_PROGRESS_WARMUP_VERSION


@dataclass(frozen=True)
class CalvinProgressWarmupBuildResult:
    output_root: Path
    manifest_path: Path
    step_count: int
    window_count: int


def build_calvin_progress_vl_embedding_cache(
    *,
    episode_index_path: str | Path,
    output_root: str | Path,
    vl_encoder: VLSummaryEncoder,
    calvin_dataset_root: str | Path | None = None,
    action_horizon: int = 32,
    replan_stride: int = 16,
    burnin_replan_steps: int = 8,
    loss_replan_steps: int = 8,
    allow_short_burnin: bool = True,
    intent_encoder: ActionSegmentAutoencoder | None = None,
    intent_encoder_checkpoint: str | Path | None = None,
    action_normalizer: ActionNormalizer | None = None,
    norm_stats_path: str | Path | None = None,
    robot_key: str | None = DEFAULT_CALVIN_ROBOT_KEY,
    storage_dtype: torch.dtype = torch.float32,
    view_names: Sequence[str] | None = DEFAULT_CALVIN_VIEW_NAMES,
    max_episodes: int | None = None,
    max_steps: int | None = None,
    progress_interval: int | None = 100,
    vl_batch_size: int = 1,
) -> CalvinProgressWarmupBuildResult:
    if action_horizon <= 0 or replan_stride <= 0 or burnin_replan_steps < 0 or loss_replan_steps <= 0:
        raise ValueError("invalid horizon/stride/window configuration")
    if int(vl_batch_size) <= 0:
        raise ValueError("vl_batch_size must be positive")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError("max_episodes must be positive when provided")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be positive when provided")

    index_path = Path(episode_index_path).expanduser()
    episode_index = read_calvin_episode_replay_index(index_path)
    dataset_root = Path(calvin_dataset_root or episode_index["calvin_dataset_root"]).expanduser()
    normalizer = action_normalizer or (lambda tensor: tensor)
    selected_view_names = tuple(DEFAULT_CALVIN_VIEW_NAMES if view_names is None else view_names)
    if not selected_view_names:
        raise ValueError("view_names must not be empty")

    steps: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for episode_payload in _iter_episode_payloads(episode_index["episodes"], max_episodes=max_episodes):
        reader = CalvinEpisodeReader(
            dataset_root,
            episode_index=int(episode_payload["episode_index"]),
            view_names=selected_view_names,
        )
        prompt = str(episode_payload.get("prompt") or episode_payload.get("task_name") or "")
        episode_id = str(episode_payload["episode_id"])
        task_name = str(episode_payload.get("task_name", ""))
        suite = str(episode_payload.get("suite", "calvin"))
        for node in sorted(episode_payload["nodes"], key=lambda item: int(item["current_step"])):
            if max_steps is not None and len(steps) + len(pending) >= int(max_steps):
                break
            current_step = int(node["current_step"])
            if current_step % int(replan_stride) != 0:
                continue
            if int(node["action_valid_count"]) < int(action_horizon):
                continue
            future_start, future_end = [int(value) for value in node["future_action_range"]]
            if future_end - future_start < int(action_horizon):
                continue
            target_actions = torch.as_tensor(
                reader.read_future_actions(future_start, future_start + int(action_horizon)),
                dtype=torch.float32,
            )
            if target_actions.shape[0] != int(action_horizon):
                continue
            target_actions = normalizer(target_actions).float()
            executed_actions, executed_mask = _read_executed_actions(
                reader,
                node,
                replan_stride=int(replan_stride),
                action_dim=int(target_actions.shape[-1]),
                action_normalizer=normalizer,
            )
            frame = reader.read_frame(current_step)
            pending.append(
                {
                    "images_by_view": frame.images_by_view,
                    "step": {
                        "step_index": -1,
                        "sample_index": len(steps) + len(pending),
                        "episode_id": episode_id,
                        "suite": suite,
                        "task_name": task_name,
                        "prompt": prompt,
                        "current_step": current_step,
                        "replan_index": current_step // int(replan_stride),
                        "state": torch.as_tensor(frame.state_vector, dtype=torch.float32).cpu(),
                        "executed_actions": executed_actions.cpu(),
                        "executed_action_mask": executed_mask.cpu(),
                        "target_intent": _encode_target_intent(intent_encoder, target_actions).cpu(),
                    },
                }
            )
            if len(pending) >= int(vl_batch_size):
                _flush_vl_summary_batch(
                    pending,
                    steps,
                    vl_encoder=vl_encoder,
                    storage_dtype=storage_dtype,
                    progress_interval=progress_interval,
                )
                pending = []
        if max_steps is not None and len(steps) + len(pending) >= int(max_steps):
            break

    if pending:
        _flush_vl_summary_batch(
            pending,
            steps,
            vl_encoder=vl_encoder,
            storage_dtype=storage_dtype,
            progress_interval=progress_interval,
        )

    windows = build_calvin_progress_windows(
        steps,
        burnin_replan_steps=burnin_replan_steps,
        loss_replan_steps=loss_replan_steps,
        allow_short_burnin=allow_short_burnin,
    )
    output_path = Path(output_root).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = output_path / "data.pt"
    torch.save(
        {
            "format": CALVIN_PROGRESS_WARMUP_FORMAT,
            "version": CALVIN_PROGRESS_WARMUP_VERSION,
            "steps": steps,
            "windows": windows,
        },
        data_path,
    )
    manifest = {
        "format": CALVIN_PROGRESS_WARMUP_FORMAT,
        "version": CALVIN_PROGRESS_WARMUP_VERSION,
        "benchmark": "CALVIN",
        "calvin_dataset_root": str(dataset_root),
        "episode_index_path": str(index_path),
        "episode_index_format": str(episode_index["format"]),
        "data_path": data_path.name,
        "embedding": "vl_summary",
        "encoder": str(getattr(vl_encoder, "name", vl_encoder.__class__.__name__)),
        "hidden_dim": int(getattr(vl_encoder, "hidden_dim", int(steps[0]["vl_summary"].shape[-1]) if steps else 0)),
        "view_names": [str(name) for name in selected_view_names],
        "action_horizon": int(action_horizon),
        "replan_stride": int(replan_stride),
        "burnin_replan_steps": int(burnin_replan_steps),
        "loss_replan_steps": int(loss_replan_steps),
        "allow_short_burnin": bool(allow_short_burnin),
        "vl_batch_size": int(vl_batch_size),
        "intent_encoder_checkpoint": None if intent_encoder_checkpoint is None else str(Path(intent_encoder_checkpoint).expanduser()),
        "norm_stats_path": None if norm_stats_path is None else str(Path(norm_stats_path).expanduser()),
        "robot_key": robot_key,
        "state_dim": int(steps[0]["state"].shape[-1]) if steps else None,
        "action_dim": int(steps[0]["executed_actions"].shape[-1]) if steps else None,
        "step_count": len(steps),
        "window_count": len(windows),
        "suite_window_counts": _window_suite_counts(windows),
        "sampler": {
            "default": "temperature_suite",
            "sampling_alpha": 0.5,
            "samples_per_epoch": 8192,
        },
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CalvinProgressWarmupBuildResult(
        output_root=output_path,
        manifest_path=manifest_path,
        step_count=len(steps),
        window_count=len(windows),
    )


def build_calvin_progress_warmup_cache(**kwargs: Any) -> CalvinProgressWarmupBuildResult:
    return build_calvin_progress_vl_embedding_cache(**kwargs)


def build_calvin_progress_windows(
    steps: Sequence[Mapping[str, Any]],
    *,
    burnin_replan_steps: int = 8,
    loss_replan_steps: int = 8,
    allow_short_burnin: bool = True,
) -> list[dict[str, Any]]:
    return build_libero_progress_windows(
        steps,
        burnin_replan_steps=burnin_replan_steps,
        loss_replan_steps=loss_replan_steps,
        allow_short_burnin=allow_short_burnin,
    )


def read_calvin_episode_replay_index(path: str | Path) -> dict[str, Any]:
    index_path = Path(path).expanduser()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("format") != "calvin_episode_replay_index":
        raise ValueError(f"{index_path} is not a CALVIN episode replay index")
    if str(payload.get("benchmark", "")).upper() != "CALVIN":
        raise ValueError(f"{index_path} benchmark must be CALVIN")
    if not isinstance(payload.get("episodes"), list) or not payload["episodes"]:
        raise ValueError(f"{index_path} contains no episodes")
    return payload


def count_calvin_planned_replan_steps(
    episode_index: Mapping[str, Any],
    *,
    horizon: int,
    replan_stride: int,
    max_episodes: int | None = None,
    max_steps: int | None = None,
) -> int:
    count = 0
    for episode in _iter_episode_payloads(episode_index["episodes"], max_episodes=max_episodes):
        for node in episode.get("nodes", []):
            if max_steps is not None and count >= int(max_steps):
                return count
            if int(node["current_step"]) % int(replan_stride) != 0:
                continue
            if int(node["action_valid_count"]) < int(horizon):
                continue
            count += 1
    return count


def calvin_action_normalizer_from_stats(
    norm_stats_path: str | Path | None,
    *,
    robot_key: str | None = DEFAULT_CALVIN_ROBOT_KEY,
) -> ActionNormalizer:
    return action_normalizer_from_stats(norm_stats_path, robot_key=robot_key)


def load_calvin_action_segment_autoencoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ActionSegmentAutoencoder:
    model = load_action_segment_autoencoder(checkpoint_path, device=device)
    if int(model.config.action_dim) != 7:
        raise ValueError(f"CALVIN action AE must use action_dim=7, got {model.config.action_dim}")
    return model


calvin_resolve_storage_dtype = resolve_storage_dtype


def _read_executed_actions(
    reader: CalvinEpisodeReader,
    node: Mapping[str, Any],
    *,
    replan_stride: int,
    action_dim: int,
    action_normalizer: ActionNormalizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    start, end = [int(value) for value in node["executed_action_range"]]
    valid_count = int(node.get("executed_action_valid_count", max(0, end - start)))
    if valid_count <= 0 or end <= start:
        return (
            torch.zeros(replan_stride, action_dim, dtype=torch.float32),
            torch.zeros(replan_stride, dtype=torch.bool),
        )
    executed_raw = reader.read_future_actions(start, end)
    executed = torch.as_tensor(executed_raw, dtype=torch.float32)
    if executed.shape[0] > replan_stride:
        executed = executed[-replan_stride:]
    if executed.shape[0] < replan_stride:
        pad = torch.zeros(replan_stride - executed.shape[0], action_dim, dtype=torch.float32)
        executed = torch.cat([pad, executed], dim=0)
    executed = action_normalizer(executed).float()
    mask = torch.zeros(replan_stride, dtype=torch.bool)
    mask[-min(valid_count, replan_stride) :] = True
    return executed, mask


def _iter_episode_payloads(
    episodes: Sequence[Mapping[str, Any]],
    *,
    max_episodes: int | None,
) -> Sequence[Mapping[str, Any]]:
    if max_episodes is None:
        return episodes
    return episodes[: int(max_episodes)]


def _window_suite_counts(windows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for window in windows:
        counts[str(window["suite"])] += 1
    return dict(sorted(counts.items()))

