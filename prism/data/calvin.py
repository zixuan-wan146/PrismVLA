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

