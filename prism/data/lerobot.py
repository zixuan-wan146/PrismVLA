"""Strict single-version LeRobot v2.1 storage access."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from prism.data.schema import DataSpec, FeatureSlice, LEROBOT_STORAGE_FORMAT, ViewSpec


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_index: int
    length: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class RawFrame:
    episode_index: int
    frame_index: int
    timestamp: float
    task_index: int
    state: np.ndarray
    action: np.ndarray


@dataclass(frozen=True)
class RawTrainingWindow:
    """Current state/instruction and its contiguous action target window."""

    state: np.ndarray
    actions: np.ndarray
    instruction: str


@dataclass(frozen=True)
class NumericEpisode:
    """All validated numeric rows for one episode, without decoding video."""

    episode_index: int
    timestamps: np.ndarray
    task_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray


class LeRobotDataset:
    """Read one complete LeRobot v2.1 root without model or benchmark logic."""

    def __init__(
        self,
        dataset_root: str | Path,
        data_spec: DataSpec,
        *,
        verify_files: bool = True,
        table_cache_size: int = 8,
        video_cache_size: int = 8,
    ) -> None:
        self.root = Path(dataset_root).expanduser()
        self.spec = data_spec
        self.spec.validate()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if table_cache_size <= 0 or video_cache_size <= 0:
            raise ValueError("table_cache_size and video_cache_size must be positive")
        self._table_cache_size = int(table_cache_size)
        self._table_cache: OrderedDict[int, Any] = OrderedDict()
        self._video_pool = _VideoPool(max_open=int(video_cache_size))

        self.info = _read_json(self.root / "meta" / "info.json")
        self._validate_info()
        self._episodes = self._read_episodes()
        self._tasks = self._read_tasks()
        self._episode_by_id = {episode.episode_index: episode for episode in self._episodes}
        self._global_offsets = self._build_global_offsets()
        self._validate_metadata_counts()
        if verify_files:
            self.validate_physical_files()

    def __enter__(self) -> "LeRobotDataset":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_table_cache"] = OrderedDict()
        state["_video_pool"] = _VideoPool(max_open=self._video_pool.max_open)
        return state

    def close(self) -> None:
        self._table_cache.clear()
        self._video_pool.close()

    def episode_ids(self) -> tuple[int, ...]:
        return tuple(episode.episode_index for episode in self._episodes)

    def episode_length(self, episode_id: int) -> int:
        return self._episode(episode_id).length

    def read_numeric_frame(self, episode_id: int, frame_index: int) -> RawFrame:
        episode = self._episode(episode_id)
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= episode.length:
            raise IndexError(
                f"dataset={self.spec.name} episode={episode_id} frame={frame_index} is outside [0, {episode.length})"
            )
        table = self._load_table(episode_id)
        row = table.iloc[frame_index]
        state = self._assemble_features(row, self.spec.state, label="state")
        action = self._assemble_features(row, self.spec.action, label="action")
        return RawFrame(
            episode_index=int(episode_id),
            frame_index=frame_index,
            timestamp=float(row["timestamp"]),
            task_index=int(row["task_index"]),
            state=state,
            action=action,
        )

    def read_numeric_episode(self, episode_id: int) -> NumericEpisode:
        """Read one complete validated numeric episode in storage order."""

        episode = self._episode(episode_id)
        table = self._load_table(episode_id)
        states = self._assemble_feature_table(table, self.spec.state, label="state")
        actions = self._assemble_feature_table(table, self.spec.action, label="action")
        return NumericEpisode(
            episode_index=episode.episode_index,
            timestamps=np.ascontiguousarray(table["timestamp"].to_numpy(dtype=np.float32)),
            task_indices=np.ascontiguousarray(table["task_index"].to_numpy(dtype=np.int64)),
            states=states,
            actions=actions,
        )

    def read_training_window(self, episode_id: int, start: int, end: int) -> RawTrainingWindow:
        """Read all numeric values needed by one training sample in one table pass."""

        episode = self._episode(episode_id)
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > episode.length:
            raise IndexError(
                f"dataset={self.spec.name} episode={episode_id} training range "
                f"[{start}, {end}) is outside [0, {episode.length})"
            )
        table = self._load_table(episode_id)
        row = table.iloc[start]
        state = self._assemble_features(row, self.spec.state, label="state")
        actions = self._assemble_feature_table(table.iloc[start:end], self.spec.action, label="action")
        instruction = self._instruction(int(row["task_index"]), episode_id=episode_id, frame_index=start)
        return RawTrainingWindow(
            state=state,
            actions=actions,
            instruction=instruction,
        )

    def read_actions(self, episode_id: int, start: int, end: int) -> np.ndarray:
        episode = self._episode(episode_id)
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > episode.length:
            raise IndexError(
                f"dataset={self.spec.name} episode={episode_id} action range "
                f"[{start}, {end}) is outside [0, {episode.length})"
            )
        if start == end:
            return np.zeros((0, self.spec.action_dim), dtype=np.float32)
        table = self._load_table(episode_id)
        return self._assemble_feature_table(table.iloc[start:end], self.spec.action, label="action")

    def read_images(
        self,
        episode_id: int,
        frame_indices: Sequence[int],
        views: Sequence[ViewSpec] | None = None,
    ) -> Mapping[str, np.ndarray]:
        episode = self._episode(episode_id)
        indices = tuple(int(index) for index in frame_indices)
        invalid = [index for index in indices if index < 0 or index >= episode.length]
        if invalid:
            raise IndexError(
                f"dataset={self.spec.name} episode={episode_id} image frames {invalid} "
                f"are outside [0, {episode.length})"
            )
        requested_views = self.spec.views if views is None else tuple(views)
        if not requested_views:
            raise ValueError("views must contain at least one ViewSpec")
        allowed = {view.name: view for view in self.spec.views}
        for view in requested_views:
            if not isinstance(view, ViewSpec) or allowed.get(view.name) != view:
                raise ValueError(f"view {view!r} is not declared by DataSpec {self.spec.name!r}")

        output: dict[str, np.ndarray] = {}
        for view in requested_views:
            expected_shape = self._view_shape(view.source_key)
            if not indices:
                output[view.name] = np.zeros((0, *expected_shape), dtype=np.uint8)
                continue
            path = self._video_path(episode_id, view.source_key)
            try:
                frames = self._video_pool.read(path, indices, fps=self.fps)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to decode dataset={self.spec.name} episode={episode_id} "
                    f"view={view.name} frames={indices} video={path}"
                ) from exc
            array = np.stack(frames, axis=0)
            if array.shape != (len(indices), *expected_shape) or array.dtype != np.uint8:
                raise ValueError(
                    f"dataset={self.spec.name} episode={episode_id} view={view.name} "
                    f"decoded shape/dtype {array.shape}/{array.dtype}, expected "
                    f"{(len(indices), *expected_shape)}/uint8"
                )
            output[view.name] = np.ascontiguousarray(array)
        return output

    def read_instruction(self, episode_id: int, frame_index: int) -> str:
        episode = self._episode(episode_id)
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= episode.length:
            raise IndexError(
                f"dataset={self.spec.name} episode={episode_id} frame={frame_index} is outside [0, {episode.length})"
            )
        row = self._load_table(episode_id).iloc[frame_index]
        return self._instruction(int(row["task_index"]), episode_id=episode_id, frame_index=frame_index)

    def _instruction(self, task_index: int, *, episode_id: int, frame_index: int) -> str:
        try:
            return self._tasks[task_index]
        except KeyError as exc:
            raise KeyError(
                f"dataset={self.spec.name} episode={episode_id} frame={frame_index} "
                f"references missing task_index={task_index}"
            ) from exc

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    def validate_physical_files(self) -> None:
        missing: list[str] = []
        missing_count = 0
        for episode in self._episodes:
            expected = [self._parquet_path(episode.episode_index)]
            expected.extend(self._video_path(episode.episode_index, view.source_key) for view in self.spec.views)
            for path in expected:
                if not path.is_file():
                    missing_count += 1
                    if len(missing) < 10:
                        missing.append(str(path.relative_to(self.root)))
        if missing_count:
            raise FileNotFoundError(
                f"dataset={self.spec.name} is incomplete: {missing_count} expected "
                f"episode artifact(s) are missing; first missing={missing}"
            )

    def _validate_info(self) -> None:
        if self.spec.storage_format != LEROBOT_STORAGE_FORMAT:
            raise ValueError(f"unsupported DataSpec storage format {self.spec.storage_format!r}")
        version = self.info.get("codebase_version")
        if version != "v2.1":
            raise ValueError(f"dataset={self.spec.name} requires LeRobot v2.1, got {version!r}")
        for key in (
            "total_episodes",
            "total_frames",
            "total_tasks",
            "chunks_size",
            "fps",
            "data_path",
            "video_path",
            "features",
        ):
            if key not in self.info:
                raise ValueError(f"dataset={self.spec.name} info.json is missing {key!r}")
        for key in ("total_episodes", "total_frames", "total_tasks", "chunks_size", "fps"):
            if type(self.info[key]) is not int or self.info[key] <= 0:
                raise ValueError(f"dataset={self.spec.name} info field {key!r} must be a positive integer")
        features = self.info["features"]
        if not isinstance(features, Mapping):
            raise ValueError(f"dataset={self.spec.name} info.features must be a mapping")
        for view in self.spec.views:
            feature = self._feature_info(view.source_key)
            if feature.get("dtype") != "video":
                raise ValueError(f"view source {view.source_key!r} must be declared as video")
            self._view_shape(view.source_key)
        for feature in (*self.spec.state, *self.spec.action):
            info = self._feature_info(feature.source_key)
            shape = info.get("shape")
            if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int:
                raise ValueError(f"numeric source {feature.source_key!r} must have one vector dimension")
            if feature.end > shape[0]:
                raise ValueError(
                    f"DataSpec feature {feature.name!r} slice [{feature.start}, {feature.end}) "
                    f"exceeds physical source {feature.source_key!r} width {shape[0]}"
                )

    def _read_episodes(self) -> tuple[EpisodeMetadata, ...]:
        records = _read_json_lines(self.root / "meta" / "episodes.jsonl")
        episodes = []
        for record in records:
            index = record.get("episode_index")
            length = record.get("length")
            tasks = record.get("tasks")
            if type(index) is not int or type(length) is not int or length <= 0:
                raise ValueError(f"dataset={self.spec.name} has invalid episode metadata {record!r}")
            if not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) for task in tasks):
                raise ValueError(f"dataset={self.spec.name} episode={index} has invalid tasks metadata")
            episodes.append(EpisodeMetadata(index, length, tuple(tasks)))
        ids = [episode.episode_index for episode in episodes]
        if ids != list(range(len(episodes))):
            raise ValueError(
                f"dataset={self.spec.name} episode indices must be contiguous 0..N-1, got first ids={ids[:10]}"
            )
        return tuple(episodes)

    def _read_tasks(self) -> dict[int, str]:
        records = _read_json_lines(self.root / "meta" / "tasks.jsonl")
        tasks: dict[int, str] = {}
        for record in records:
            index = record.get("task_index")
            text = record.get("task")
            if type(index) is not int or not isinstance(text, str) or not text.strip():
                raise ValueError(f"dataset={self.spec.name} has invalid task metadata {record!r}")
            if index in tasks:
                raise ValueError(f"dataset={self.spec.name} repeats task_index={index}")
            tasks[index] = text
        if sorted(tasks) != list(range(len(tasks))):
            raise ValueError(f"dataset={self.spec.name} task indices must be contiguous 0..N-1")
        return tasks

    def _build_global_offsets(self) -> dict[int, int]:
        offsets: dict[int, int] = {}
        offset = 0
        for episode in self._episodes:
            offsets[episode.episode_index] = offset
            offset += episode.length
        return offsets

    def _validate_metadata_counts(self) -> None:
        if len(self._episodes) != int(self.info["total_episodes"]):
            raise ValueError(
                f"dataset={self.spec.name} declares {self.info['total_episodes']} episodes "
                f"but episodes.jsonl contains {len(self._episodes)}"
            )
        frame_count = sum(episode.length for episode in self._episodes)
        if frame_count != int(self.info["total_frames"]):
            raise ValueError(
                f"dataset={self.spec.name} declares {self.info['total_frames']} frames "
                f"but episode lengths sum to {frame_count}"
            )
        if len(self._tasks) != int(self.info["total_tasks"]):
            raise ValueError(
                f"dataset={self.spec.name} declares {self.info['total_tasks']} tasks "
                f"but tasks.jsonl contains {len(self._tasks)}"
            )

    def _episode(self, episode_id: int) -> EpisodeMetadata:
        episode_id = int(episode_id)
        try:
            return self._episode_by_id[episode_id]
        except KeyError as exc:
            raise IndexError(f"dataset={self.spec.name} has no episode_index={episode_id}") from exc

    def _load_table(self, episode_id: int):
        if episode_id in self._table_cache:
            table = self._table_cache.pop(episode_id)
            self._table_cache[episode_id] = table
            return table
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("pandas is required to read LeRobot Parquet episodes") from exc
        path = self._parquet_path(episode_id)
        if not path.is_file():
            raise FileNotFoundError(f"dataset={self.spec.name} episode={episode_id} is missing parquet={path}")
        table = pd.read_parquet(path)
        self._validate_table(episode_id, table, path)
        self._table_cache[episode_id] = table
        if len(self._table_cache) > self._table_cache_size:
            self._table_cache.popitem(last=False)
        return table

    def _validate_table(self, episode_id: int, table: Any, path: Path) -> None:
        episode = self._episode(episode_id)
        required = {
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            *(feature.source_key for feature in self.spec.state),
            *(feature.source_key for feature in self.spec.action),
        }
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(
                f"dataset={self.spec.name} episode={episode_id} parquet={path} is missing columns {missing}"
            )
        if len(table) != episode.length:
            raise ValueError(
                f"dataset={self.spec.name} episode={episode_id} parquet={path} has "
                f"{len(table)} rows, expected {episode.length}"
            )
        expected_frames = np.arange(episode.length, dtype=np.int64)
        actual_frames = table["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_frames, expected_frames):
            raise ValueError(f"dataset={self.spec.name} episode={episode_id} frame_index is not contiguous")
        actual_episode = table["episode_index"].to_numpy(dtype=np.int64)
        if not np.all(actual_episode == episode_id):
            raise ValueError(f"dataset={self.spec.name} episode={episode_id} contains foreign episode_index values")
        offset = self._global_offsets[episode_id]
        expected_global = np.arange(offset, offset + episode.length, dtype=np.int64)
        if not np.array_equal(table["index"].to_numpy(dtype=np.int64), expected_global):
            raise ValueError(f"dataset={self.spec.name} episode={episode_id} global index is not contiguous")
        expected_time = expected_frames.astype(np.float64) / float(self.fps)
        actual_time = table["timestamp"].to_numpy(dtype=np.float64)
        if not np.allclose(actual_time, expected_time, rtol=0.0, atol=1.0e-5):
            raise ValueError(f"dataset={self.spec.name} episode={episode_id} timestamp does not match fps")
        task_indices = table["task_index"].to_numpy(dtype=np.int64)
        if np.any([int(index) not in self._tasks for index in task_indices]):
            raise ValueError(f"dataset={self.spec.name} episode={episode_id} contains unknown task_index")

    def _assemble_features(
        self,
        row: Any,
        features: tuple[FeatureSlice, ...],
        *,
        label: str,
    ) -> np.ndarray:
        pieces = []
        for feature in features:
            try:
                vector = np.asarray(row[feature.source_key], dtype=np.float32).reshape(-1)
            except KeyError as exc:
                raise KeyError(f"dataset={self.spec.name} {label} source {feature.source_key!r} is missing") from exc
            if feature.end > vector.shape[0]:
                raise ValueError(
                    f"dataset={self.spec.name} {label} feature={feature.name} slice "
                    f"[{feature.start}, {feature.end}) exceeds row width {vector.shape[0]}"
                )
            pieces.append(vector[feature.start : feature.end])
        output = np.concatenate(pieces).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise ValueError(f"dataset={self.spec.name} {label} contains non-finite values")
        return output

    def _assemble_feature_table(
        self,
        table: Any,
        features: tuple[FeatureSlice, ...],
        *,
        label: str,
    ) -> np.ndarray:
        sources: dict[str, np.ndarray] = {}
        pieces: list[np.ndarray] = []
        for feature in features:
            if feature.source_key not in sources:
                try:
                    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in table[feature.source_key]]
                except KeyError as exc:
                    raise KeyError(
                        f"dataset={self.spec.name} {label} source {feature.source_key!r} is missing"
                    ) from exc
                widths = {row.shape[0] for row in rows}
                if len(widths) != 1:
                    raise ValueError(
                        f"dataset={self.spec.name} {label} source {feature.source_key!r} has inconsistent row widths"
                    )
                sources[feature.source_key] = np.stack(rows, axis=0)
            source = sources[feature.source_key]
            if feature.end > source.shape[1]:
                raise ValueError(
                    f"dataset={self.spec.name} {label} feature={feature.name} slice "
                    f"[{feature.start}, {feature.end}) exceeds row width {source.shape[1]}"
                )
            pieces.append(source[:, feature.start : feature.end])
        output = np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise ValueError(f"dataset={self.spec.name} {label} contains non-finite values")
        return np.ascontiguousarray(output)

    def _feature_info(self, key: str) -> Mapping[str, Any]:
        features = self.info["features"]
        if key not in features or not isinstance(features[key], Mapping):
            raise ValueError(f"dataset={self.spec.name} info.features is missing source {key!r}")
        return features[key]

    def _view_shape(self, key: str) -> tuple[int, int, int]:
        shape = self._feature_info(key).get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[-1] != 3
        ):
            raise ValueError(f"dataset={self.spec.name} video source {key!r} has invalid RGB shape {shape!r}")
        return tuple(shape)

    def _parquet_path(self, episode_id: int) -> Path:
        chunk = int(episode_id) // int(self.info["chunks_size"])
        return self.root / str(self.info["data_path"]).format(
            episode_chunk=chunk,
            episode_index=int(episode_id),
        )

    def _video_path(self, episode_id: int, video_key: str) -> Path:
        chunk = int(episode_id) // int(self.info["chunks_size"])
        return self.root / str(self.info["video_path"]).format(
            episode_chunk=chunk,
            episode_index=int(episode_id),
            video_key=video_key,
        )


class _VideoPool:
    def __init__(self, *, max_open: int) -> None:
        self.max_open = int(max_open)
        self._containers: OrderedDict[Path, Any] = OrderedDict()

    def close(self) -> None:
        while self._containers:
            _, container = self._containers.popitem(last=False)
            container.close()

    def read(self, path: Path, frame_indices: tuple[int, ...], *, fps: int) -> list[np.ndarray]:
        container = self._container(path)
        stream = container.streams.video[0]
        targets = set(frame_indices)
        first = min(targets)
        last = max(targets)
        if stream.time_base is None:
            raise ValueError(f"video {path} has no time base")
        timestamp = int((first / float(fps)) / float(stream.time_base))
        container.seek(timestamp, stream=stream, backward=True, any_frame=False)
        decoded: dict[int, np.ndarray] = {}
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"video {path} yielded a frame without pts")
            index = int(round(float(frame.pts * stream.time_base) * float(fps)))
            if index in targets and index not in decoded:
                decoded[index] = np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            if index > last and targets.issubset(decoded):
                break
            if index > last + 2:
                break
        missing = sorted(targets - set(decoded))
        if missing:
            raise IndexError(f"video {path} did not decode requested frame indices {missing}")
        return [decoded[index] for index in frame_indices]

    def _container(self, path: Path):
        if path in self._containers:
            container = self._containers.pop(path)
            self._containers[path] = container
            return container
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            import av
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyAV is required to decode LeRobot videos") from exc
        container = av.open(str(path), mode="r")
        if not container.streams.video:
            container.close()
            raise ValueError(f"video {path} has no video stream")
        self._containers[path] = container
        if len(self._containers) > self.max_open:
            _, evicted = self._containers.popitem(last=False)
            evicted.close()
        return container


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected in {path}")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSON object expected in {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"metadata file is empty: {path}")
    return records


__all__ = ["EpisodeMetadata", "LeRobotDataset", "NumericEpisode", "RawFrame"]
