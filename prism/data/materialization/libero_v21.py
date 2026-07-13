"""Deterministic LIBERO HDF5 to LeRobot v2.1 materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from prism.data.benchmark_contracts import LIBERO_DATASET_NAMES
from prism.data.materialization.common import MaterializationError
from prism.data.materialization.common import canonical_json
from prism.data.materialization.common import file_sha256
from prism.data.materialization.common import json_sha256
from prism.data.materialization.common import read_json
from prism.data.materialization.common import read_jsonl

LIBERO_SUITES = LIBERO_DATASET_NAMES
LIBERO_FPS = 20
LEROBOT_VERSION = "v2.1"
CHUNKS_SIZE = 1000
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_TRANSFORMS = frozenset({"none", "rotate_180"})
IMAGE_FEATURES = {
    "observation.images.image": "obs/agentview_rgb",
    "observation.images.wrist_image": "obs/eye_in_hand_rgb",
}
PARQUET_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)

ImageTransform = Literal["none", "rotate_180"]


@dataclass(frozen=True)
class VideoEncodingConfig:
    """Explicit video encoding settings stored in materialization provenance."""

    codec: str = "libsvtav1"
    pixel_format: str = "yuv420p"
    crf: int = 30
    gop: int = 2
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.codec.strip():
            raise ValueError("video codec must be non-empty")
        if not self.pixel_format.strip():
            raise ValueError("video pixel_format must be non-empty")
        if type(self.crf) is not int or self.crf < 0:
            raise ValueError("video crf must be a non-negative integer")
        if type(self.gop) is not int or self.gop <= 0:
            raise ValueError("video gop must be a positive integer")
        keys = [str(key) for key, _ in self.options]
        if any(not key for key in keys):
            raise ValueError("video option names must be non-empty")
        if len(keys) != len(set(keys)):
            raise ValueError("video option names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "crf": self.crf,
            "gop": self.gop,
            "options": {str(key): str(value) for key, value in self.options},
        }


@dataclass(frozen=True)
class SourceFilePlan:
    relative_path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class EpisodePlan:
    source_path: Path
    source_relative_path: str
    demo_key: str
    episode_index: int
    task_index: int
    task: str
    length: int
    global_index_from: int
    global_index_to: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_relative_path,
            "demo_key": self.demo_key,
            "episode_index": self.episode_index,
            "task_index": self.task_index,
            "task": self.task,
            "length": self.length,
            "global_index_from": self.global_index_from,
            "global_index_to": self.global_index_to,
        }


@dataclass(frozen=True)
class MaterializationPlan:
    source_root: Path
    suite: str
    image_transform: ImageTransform
    source_files: tuple[SourceFilePlan, ...]
    tasks: tuple[str, ...]
    episodes: tuple[EpisodePlan, ...]
    total_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "prism-libero-plan-v1",
            "codebase_version": LEROBOT_VERSION,
            "suite": self.suite,
            "fps": LIBERO_FPS,
            "image_transform": self.image_transform,
            "source_files": [item.to_dict() for item in self.source_files],
            "tasks": [{"task_index": task_index, "task": task} for task_index, task in enumerate(self.tasks)],
            "episodes": [episode.to_dict() for episode in self.episodes],
            "total_episodes": len(self.episodes),
            "total_frames": self.total_frames,
        }

    @property
    def sha256(self) -> str:
        return json_sha256(self.to_dict())


def build_libero_v21_plan(
    source_root: str | Path,
    *,
    suite: str,
    image_transform: ImageTransform,
) -> MaterializationPlan:
    """Inspect and validate one raw LIBERO suite, returning a deterministic plan."""

    transform = _validate_image_transform(image_transform)
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    suite_dir = root / suite
    if not suite_dir.is_dir():
        if root.name == suite:
            suite_dir = root
        else:
            raise FileNotFoundError(f"LIBERO suite directory does not exist: {suite_dir}")

    source_paths = sorted(
        path for path in suite_dir.iterdir() if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not source_paths:
        raise MaterializationError(f"no HDF5 files found in LIBERO suite directory: {suite_dir}")

    h5py = _require_h5py()
    tasks: list[str] = []
    episodes: list[EpisodePlan] = []
    source_files: list[SourceFilePlan] = []
    global_index = 0

    for task_index, source_path in enumerate(source_paths):
        relative_path = _relative_source_path(source_path, root)
        source_files.append(
            SourceFilePlan(
                relative_path=relative_path,
                sha256=file_sha256(source_path),
                size_bytes=source_path.stat().st_size,
            )
        )
        with h5py.File(source_path, "r") as handle:
            if "data" not in handle:
                raise MaterializationError(f"{relative_path}: missing HDF5 group 'data'")
            data = handle["data"]
            env_args = _read_json_attribute(data.attrs, "env_args", source_label=relative_path)
            problem_info = _read_json_attribute(data.attrs, "problem_info", source_label=relative_path)
            _validate_environment_metadata(env_args, source_label=relative_path)
            task = problem_info.get("language_instruction")
            if not isinstance(task, str) or not task.strip():
                raise MaterializationError(
                    f"{relative_path}: problem_info.language_instruction must be a non-empty string"
                )
            task = task.strip()
            tasks.append(task)

            demo_keys = sorted(data.keys(), key=_demo_sort_key)
            if not demo_keys:
                raise MaterializationError(f"{relative_path}: HDF5 group 'data' contains no demos")
            _validate_optional_count(data.attrs, "num_demos", len(demo_keys), source_label=relative_path)

            source_total = 0
            for demo_key in demo_keys:
                demo = data[demo_key]
                length = _validate_demo(demo, source_label=f"{relative_path}:data/{demo_key}")
                episode_index = len(episodes)
                episodes.append(
                    EpisodePlan(
                        source_path=source_path,
                        source_relative_path=relative_path,
                        demo_key=demo_key,
                        episode_index=episode_index,
                        task_index=task_index,
                        task=task,
                        length=length,
                        global_index_from=global_index,
                        global_index_to=global_index + length,
                    )
                )
                source_total += length
                global_index += length
            _validate_optional_count(data.attrs, "total", source_total, source_label=relative_path)

    return MaterializationPlan(
        source_root=root,
        suite=suite,
        image_transform=transform,
        source_files=tuple(source_files),
        tasks=tuple(tasks),
        episodes=tuple(episodes),
        total_frames=global_index,
    )


def materialize_libero_v21(
    source_root: str | Path,
    output_root: str | Path,
    *,
    image_transform: ImageTransform,
    suites: Sequence[str] = LIBERO_SUITES,
    video_encoding: VideoEncodingConfig | None = None,
    resume: bool = True,
) -> tuple[Path, ...]:
    """Materialize each requested suite into its own LeRobot v2.1 dataset root."""

    selected = tuple(str(suite) for suite in suites)
    if not selected:
        raise ValueError("suites must contain at least one suite")
    if len(selected) != len(set(selected)):
        raise ValueError("suites must not contain duplicates")
    unknown = sorted(set(selected) - set(LIBERO_SUITES))
    if unknown:
        raise ValueError(f"unknown LIBERO suites: {unknown}")

    destination_root = Path(output_root).expanduser().resolve()
    outputs = []
    for suite in selected:
        plan = build_libero_v21_plan(
            source_root,
            suite=suite,
            image_transform=image_transform,
        )
        outputs.append(
            materialize_libero_v21_plan(
                plan,
                destination_root / suite,
                video_encoding=video_encoding,
                resume=resume,
            )
        )
    return tuple(outputs)


def materialize_libero_v21_plan(
    plan: MaterializationPlan,
    output_dir: str | Path,
    *,
    video_encoding: VideoEncodingConfig | None = None,
    resume: bool = True,
) -> Path:
    """Execute a validated plan with atomic artifacts and journal-based resume."""

    encoding = video_encoding or VideoEncodingConfig()
    output = Path(output_dir).expanduser().resolve()
    run_spec = _run_spec(plan, encoding)

    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
        _validate_completed_dataset(output, plan=plan, run_spec=run_spec)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (f".{output.name}.lerobot-v2.1.partial-{run_spec['run_sha256'][:12]}")
    other_staging = sorted(
        path for path in output.parent.glob(f".{output.name}.lerobot-v2.1.partial-*") if path != staging
    )
    if other_staging:
        raise MaterializationError(
            "a partial materialization with a different plan or encoding already exists: "
            + ", ".join(str(path) for path in other_staging)
        )
    if staging.exists() and not resume:
        raise FileExistsError(f"partial materialization exists and resume is disabled: {staging}")

    _initialize_or_validate_staging(staging, run_spec=run_spec)
    journal_dir = staging / ".materialization" / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)

    for episode in plan.episodes:
        journal_path = journal_dir / f"episode_{episode.episode_index:06d}.json"
        if journal_path.exists():
            journal = read_json(journal_path)
            _validate_episode_journal(
                staging,
                journal,
                episode=episode,
                run_sha256=run_spec["run_sha256"],
            )
            continue
        journal = _materialize_episode(
            staging,
            episode=episode,
            image_transform=plan.image_transform,
            video_encoding=encoding,
            run_sha256=run_spec["run_sha256"],
        )
        _atomic_write_json(journal_path, journal)

    _finalize_metadata(staging, plan=plan, run_spec=run_spec)
    _validate_completed_dataset(staging, plan=plan, run_spec=run_spec)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output created concurrently: {output}")
    os.replace(staging, output)
    return output


def _materialize_episode(
    staging_root: Path,
    *,
    episode: EpisodePlan,
    image_transform: ImageTransform,
    video_encoding: VideoEncodingConfig,
    run_sha256: str,
) -> dict[str, Any]:
    h5py = _require_h5py()
    with h5py.File(episode.source_path, "r") as handle:
        demo_path = f"data/{episode.demo_key}"
        if demo_path not in handle:
            raise MaterializationError(f"{episode.source_relative_path}: source demo disappeared: {demo_path}")
        demo = handle[demo_path]
        current_length = _validate_demo(
            demo,
            source_label=f"{episode.source_relative_path}:{demo_path}",
        )
        if current_length != episode.length:
            raise MaterializationError(
                f"{episode.source_relative_path}:{demo_path}: source length changed from "
                f"{episode.length} to {current_length}"
            )

        state = np.concatenate(
            (
                np.asarray(demo["obs/ee_states"], dtype=np.float32),
                np.asarray(demo["obs/gripper_states"], dtype=np.float32),
            ),
            axis=1,
        )
        raw_action = np.asarray(demo["actions"], dtype=np.float32)
        action = raw_action.copy()
        action[:, 6] = (1.0 - raw_action[:, 6]) / 2.0
        if not np.all(np.isin(action[:, 6], (0.0, 1.0))):
            raise MaterializationError(
                f"{episode.source_relative_path}:{demo_path}: canonical gripper action must be exactly 0 or 1"
            )

        frame_index = np.arange(episode.length, dtype=np.int64)
        episode_index = np.full(episode.length, episode.episode_index, dtype=np.int64)
        global_index = np.arange(
            episode.global_index_from,
            episode.global_index_to,
            dtype=np.int64,
        )
        task_index = np.full(episode.length, episode.task_index, dtype=np.int64)
        timestamp = frame_index.astype(np.float32) / np.float32(LIBERO_FPS)

        chunk = episode.episode_index // CHUNKS_SIZE
        parquet_path = staging_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode.episode_index:06d}.parquet"
        parquet_artifact = _write_parquet_atomic(
            parquet_path,
            state=state,
            action=action,
            timestamp=timestamp,
            frame_index=frame_index,
            episode_index=episode_index,
            global_index=global_index,
            task_index=task_index,
        )

        artifacts = [
            {
                **parquet_artifact,
                "path": parquet_path.relative_to(staging_root).as_posix(),
            }
        ]
        image_stats: dict[str, Any] = {}
        for feature_key, source_key in IMAGE_FEATURES.items():
            video_path = (
                staging_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / feature_key
                / f"episode_{episode.episode_index:06d}.mp4"
            )
            video_artifact = _write_video_atomic(
                video_path,
                frames=demo[source_key],
                image_transform=image_transform,
                video_encoding=video_encoding,
                expected_frames=episode.length,
            )
            artifacts.append(
                {
                    **video_artifact,
                    "feature_key": feature_key,
                    "path": video_path.relative_to(staging_root).as_posix(),
                }
            )
            image_stats[feature_key] = _image_stats(
                demo[source_key],
                image_transform=image_transform,
            )

    stats = {
        "observation.state": _array_stats(state),
        "action": _array_stats(action),
        **image_stats,
        "timestamp": _array_stats(timestamp[:, None]),
        "frame_index": _array_stats(frame_index[:, None]),
        "episode_index": _array_stats(episode_index[:, None]),
        "index": _array_stats(global_index[:, None]),
        "task_index": _array_stats(task_index[:, None]),
    }
    return {
        "schema_version": "prism-libero-episode-journal-v1",
        "run_sha256": run_sha256,
        "episode": {
            "episode_index": episode.episode_index,
            "tasks": [episode.task],
            "length": episode.length,
        },
        "stats": stats,
        "artifacts": artifacts,
    }


def _write_parquet_atomic(
    target: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    global_index: np.ndarray,
    task_index: np.ndarray,
) -> dict[str, Any]:
    pa, pq = _require_pyarrow()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(target)
    temp.unlink(missing_ok=True)

    state_values = pa.array(state.reshape(-1), type=pa.float32())
    action_values = pa.array(action.reshape(-1), type=pa.float32())
    table = pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(state_values, 8),
            "action": pa.FixedSizeListArray.from_arrays(action_values, 7),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
        }
    )
    huggingface_metadata = {
        "info": {
            "features": {
                "observation.state": _hf_sequence_feature("float32", 8),
                "action": _hf_sequence_feature("float32", 7),
                "timestamp": _hf_value_feature("float32"),
                "frame_index": _hf_value_feature("int64"),
                "episode_index": _hf_value_feature("int64"),
                "index": _hf_value_feature("int64"),
                "task_index": _hf_value_feature("int64"),
            }
        }
    }
    table = table.replace_schema_metadata(
        {
            b"huggingface": canonical_json(huggingface_metadata).encode("utf-8"),
        }
    )

    try:
        pq.write_table(table, temp, compression="snappy", use_dictionary=False)
        _validate_parquet(temp, expected_rows=len(state))
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return {
        "kind": "parquet",
        "sha256": file_sha256(target),
        "size_bytes": target.stat().st_size,
        "rows": len(state),
    }


def _write_video_atomic(
    target: Path,
    *,
    frames: Any,
    image_transform: ImageTransform,
    video_encoding: VideoEncodingConfig,
    expected_frames: int,
) -> dict[str, Any]:
    av = _require_av()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(target)
    temp.unlink(missing_ok=True)
    encoder_options = {
        "crf": str(video_encoding.crf),
        **{str(key): str(value) for key, value in video_encoding.options},
    }

    try:
        with av.open(str(temp), mode="w") as container:
            stream = container.add_stream(
                video_encoding.codec,
                rate=LIBERO_FPS,
                options=encoder_options,
            )
            stream.width = IMAGE_WIDTH
            stream.height = IMAGE_HEIGHT
            stream.pix_fmt = video_encoding.pixel_format
            stream.codec_context.gop_size = video_encoding.gop
            for frame_index in range(expected_frames):
                array = np.asarray(frames[frame_index])
                transformed = _apply_image_transform(array, image_transform)
                video_frame = av.VideoFrame.from_ndarray(transformed, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        video_info = _probe_video(
            temp,
            expected_frames=expected_frames,
            expected_width=IMAGE_WIDTH,
            expected_height=IMAGE_HEIGHT,
            expected_fps=LIBERO_FPS,
        )
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return {
        "kind": "video",
        "sha256": file_sha256(target),
        "size_bytes": target.stat().st_size,
        "frames": expected_frames,
        "video_info": video_info,
    }


def _finalize_metadata(
    staging_root: Path,
    *,
    plan: MaterializationPlan,
    run_spec: Mapping[str, Any],
) -> None:
    journal_dir = staging_root / ".materialization" / "journal"
    journals = []
    for episode in plan.episodes:
        journal_path = journal_dir / f"episode_{episode.episode_index:06d}.json"
        if not journal_path.exists():
            raise MaterializationError(f"missing episode journal: {journal_path}")
        journal = read_json(journal_path)
        _validate_episode_journal(
            staging_root,
            journal,
            episode=episode,
            run_sha256=str(run_spec["run_sha256"]),
        )
        journals.append(journal)

    first_video = next(artifact for artifact in journals[0]["artifacts"] if artifact["kind"] == "video")
    video_info = first_video["video_info"]
    info = _build_info(plan, video_info=video_info)
    tasks = [{"task_index": task_index, "task": task} for task_index, task in enumerate(plan.tasks)]
    episodes = [journal["episode"] for journal in journals]
    episode_stats = [
        {
            "episode_index": int(journal["episode"]["episode_index"]),
            "stats": journal["stats"],
        }
        for journal in journals
    ]

    meta_dir = staging_root / "meta"
    _atomic_write_json(meta_dir / "info.json", info)
    _atomic_write_jsonl(meta_dir / "tasks.jsonl", tasks)
    _atomic_write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _atomic_write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)

    provenance: dict[str, Any] = {
        "schema_version": "prism-libero-materialization-v1",
        **dict(run_spec),
        "source_files": [source.to_dict() for source in plan.source_files],
        "source_episodes": [episode.to_dict() for episode in plan.episodes],
        "artifacts": [artifact for journal in journals for artifact in journal["artifacts"]],
    }
    provenance["content_sha256"] = json_sha256(provenance)
    _atomic_write_json(meta_dir / "materialization.json", provenance)


def _build_info(
    plan: MaterializationPlan,
    *,
    video_info: Mapping[str, Any],
) -> dict[str, Any]:
    video_feature = {
        "dtype": "video",
        "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "names": ["height", "width", "channel"],
        "info": dict(video_info),
    }
    return {
        "codebase_version": LEROBOT_VERSION,
        "robot_type": "franka",
        "fps": LIBERO_FPS,
        "total_episodes": len(plan.episodes),
        "total_frames": plan.total_frames,
        "total_tasks": len(plan.tasks),
        "total_videos": len(plan.episodes) * len(IMAGE_FEATURES),
        "total_chunks": math.ceil(len(plan.episodes) / CHUNKS_SIZE),
        "chunks_size": CHUNKS_SIZE,
        "splits": {"train": f"0:{len(plan.episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": [
                    "ee_x",
                    "ee_y",
                    "ee_z",
                    "ee_roll",
                    "ee_pitch",
                    "ee_yaw",
                    "gripper_left",
                    "gripper_right",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_roll",
                    "delta_pitch",
                    "delta_yaw",
                    "gripper_open",
                ],
            },
            "observation.images.image": video_feature,
            "observation.images.wrist_image": json.loads(json.dumps(video_feature)),
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _validate_completed_dataset(
    root: Path,
    *,
    plan: MaterializationPlan,
    run_spec: Mapping[str, Any],
) -> None:
    required_meta = (
        "info.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "materialization.json",
    )
    for name in required_meta:
        path = root / "meta" / name
        if not path.is_file():
            raise MaterializationError(f"completed dataset is missing metadata: {path}")

    info = read_json(root / "meta" / "info.json")
    expected_info_fields = {
        "codebase_version": LEROBOT_VERSION,
        "fps": LIBERO_FPS,
        "total_episodes": len(plan.episodes),
        "total_frames": plan.total_frames,
        "total_tasks": len(plan.tasks),
        "total_videos": len(plan.episodes) * len(IMAGE_FEATURES),
    }
    for key, expected in expected_info_fields.items():
        if info.get(key) != expected:
            raise MaterializationError(f"metadata info field {key!r} is {info.get(key)!r}, expected {expected!r}")

    provenance = read_json(root / "meta" / "materialization.json")
    for key, expected in run_spec.items():
        if provenance.get(key) != expected:
            raise MaterializationError(f"materialization provenance field {key!r} does not match current run")
    recorded_content_hash = provenance.get("content_sha256")
    unhashed = dict(provenance)
    unhashed.pop("content_sha256", None)
    if recorded_content_hash != json_sha256(unhashed):
        raise MaterializationError("materialization provenance content_sha256 is invalid")

    expected_source_files = [source.to_dict() for source in plan.source_files]
    if provenance.get("source_files") != expected_source_files:
        raise MaterializationError("materialization provenance source_files do not match the plan")
    expected_source_episodes = [episode.to_dict() for episode in plan.episodes]
    if provenance.get("source_episodes") != expected_source_episodes:
        raise MaterializationError("materialization provenance source_episodes do not match the plan")

    tasks = read_jsonl(root / "meta" / "tasks.jsonl")
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    episode_stats = read_jsonl(root / "meta" / "episodes_stats.jsonl")
    if len(tasks) != len(plan.tasks):
        raise MaterializationError("tasks.jsonl row count does not match the plan")
    if len(episodes) != len(plan.episodes):
        raise MaterializationError("episodes.jsonl row count does not match the plan")
    if len(episode_stats) != len(plan.episodes):
        raise MaterializationError("episodes_stats.jsonl row count does not match the plan")

    for episode in plan.episodes:
        journal_path = root / ".materialization" / "journal" / f"episode_{episode.episode_index:06d}.json"
        if not journal_path.is_file():
            raise MaterializationError(f"completed dataset is missing journal: {journal_path}")
        _validate_episode_journal(
            root,
            read_json(journal_path),
            episode=episode,
            run_sha256=str(run_spec["run_sha256"]),
        )


def _validate_episode_journal(
    root: Path,
    journal: Mapping[str, Any],
    *,
    episode: EpisodePlan,
    run_sha256: str,
) -> None:
    if journal.get("schema_version") != "prism-libero-episode-journal-v1":
        raise MaterializationError(f"episode {episode.episode_index}: unsupported or missing journal schema")
    if journal.get("run_sha256") != run_sha256:
        raise MaterializationError(f"episode {episode.episode_index}: journal belongs to a different run")
    metadata = journal.get("episode")
    if not isinstance(metadata, Mapping):
        raise MaterializationError(f"episode {episode.episode_index}: journal metadata is invalid")
    expected_metadata = {
        "episode_index": episode.episode_index,
        "tasks": [episode.task],
        "length": episode.length,
    }
    if dict(metadata) != expected_metadata:
        raise MaterializationError(f"episode {episode.episode_index}: journal episode metadata does not match the plan")

    artifacts = journal.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1 + len(IMAGE_FEATURES):
        raise MaterializationError(f"episode {episode.episode_index}: journal artifact list is invalid")
    parquet_count = 0
    video_features = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise MaterializationError(f"episode {episode.episode_index}: invalid artifact record")
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            raise MaterializationError(f"episode {episode.episode_index}: artifact path must be relative")
        path = root / relative_path
        if not path.is_file():
            raise MaterializationError(f"episode {episode.episode_index}: missing artifact {relative_path}")
        if artifact.get("sha256") != file_sha256(path):
            raise MaterializationError(f"episode {episode.episode_index}: checksum mismatch for {relative_path}")
        if artifact.get("size_bytes") != path.stat().st_size:
            raise MaterializationError(f"episode {episode.episode_index}: size mismatch for {relative_path}")
        kind = artifact.get("kind")
        if kind == "parquet":
            parquet_count += 1
            _validate_parquet(path, expected_rows=episode.length)
        elif kind == "video":
            feature_key = artifact.get("feature_key")
            video_features.add(feature_key)
            _probe_video(
                path,
                expected_frames=episode.length,
                expected_width=IMAGE_WIDTH,
                expected_height=IMAGE_HEIGHT,
                expected_fps=LIBERO_FPS,
            )
        else:
            raise MaterializationError(f"episode {episode.episode_index}: unknown artifact kind {kind!r}")
    if parquet_count != 1 or video_features != set(IMAGE_FEATURES):
        raise MaterializationError(f"episode {episode.episode_index}: journal does not contain the required artifacts")
    if not isinstance(journal.get("stats"), Mapping):
        raise MaterializationError(f"episode {episode.episode_index}: journal stats are missing")


def _validate_parquet(path: Path, *, expected_rows: int) -> None:
    pa, pq = _require_pyarrow()
    table = pq.read_table(path)
    if table.num_rows != expected_rows:
        raise MaterializationError(f"{path}: parquet has {table.num_rows} rows, expected {expected_rows}")
    if tuple(table.column_names) != PARQUET_COLUMNS:
        raise MaterializationError(
            f"{path}: parquet columns {tuple(table.column_names)!r} do not match {PARQUET_COLUMNS!r}"
        )
    expected_types = {
        "observation.state": pa.list_(pa.float32(), 8),
        "action": pa.list_(pa.float32(), 7),
        "timestamp": pa.float32(),
        "frame_index": pa.int64(),
        "episode_index": pa.int64(),
        "index": pa.int64(),
        "task_index": pa.int64(),
    }
    for field_name, expected_type in expected_types.items():
        actual_type = table.schema.field(field_name).type
        if actual_type != expected_type:
            raise MaterializationError(
                f"{path}: parquet field {field_name!r} has type {actual_type}, expected {expected_type}"
            )
    metadata = table.schema.metadata or {}
    if b"huggingface" not in metadata:
        raise MaterializationError(f"{path}: parquet schema is missing Hugging Face metadata")


def _probe_video(
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: int,
) -> dict[str, Any]:
    av = _require_av()
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise MaterializationError(f"{path}: expected exactly one video stream")
        if container.streams.audio:
            raise MaterializationError(f"{path}: video must not contain audio")
        stream = container.streams.video[0]
        rate = stream.average_rate
        fps = float(rate) if rate is not None else 0.0
        if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-6):
            raise MaterializationError(f"{path}: video fps {fps} != {expected_fps}")
        decoded_frames = 0
        for frame in container.decode(video=0):
            if frame.width != expected_width or frame.height != expected_height:
                raise MaterializationError(
                    f"{path}: decoded frame is {frame.width}x{frame.height}, expected "
                    f"{expected_width}x{expected_height}"
                )
            decoded_frames += 1
        if decoded_frames != expected_frames:
            raise MaterializationError(f"{path}: decoded {decoded_frames} frames, expected {expected_frames}")
        codec = stream.codec_context.codec
        codec_name = getattr(codec, "canonical_name", None) or getattr(codec, "name", None) or stream.codec_context.name
        pixel_format = stream.codec_context.format.name if stream.codec_context.format is not None else None
    return {
        "video.height": expected_height,
        "video.width": expected_width,
        "video.channels": 3,
        "video.fps": float(expected_fps),
        "video.codec": str(codec_name),
        "video.pix_fmt": pixel_format,
        "video.is_depth_map": False,
        "has_audio": False,
    }


def _validate_demo(demo: Any, *, source_label: str) -> int:
    required = {
        "actions": (7,),
        "obs/ee_states": (6,),
        "obs/gripper_states": (2,),
        "obs/agentview_rgb": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "obs/eye_in_hand_rgb": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    }
    if "actions" not in demo:
        raise MaterializationError(f"{source_label}: missing dataset 'actions'")
    actions_dataset = demo["actions"]
    if len(actions_dataset.shape) != 2 or tuple(actions_dataset.shape[1:]) != (7,):
        raise MaterializationError(f"{source_label}: actions must have shape [T, 7], got {actions_dataset.shape}")
    length = int(actions_dataset.shape[0])
    if length <= 0:
        raise MaterializationError(f"{source_label}: demonstrations must not be empty")

    for key, trailing_shape in required.items():
        if key not in demo:
            raise MaterializationError(f"{source_label}: missing dataset {key!r}")
        dataset = demo[key]
        expected_shape = (length, *trailing_shape)
        if tuple(dataset.shape) != expected_shape:
            raise MaterializationError(f"{source_label}: {key} must have shape {expected_shape}, got {dataset.shape}")
        if key.startswith("obs/") and key.endswith("_rgb"):
            if dataset.dtype != np.dtype(np.uint8):
                raise MaterializationError(f"{source_label}: {key} must have dtype uint8, got {dataset.dtype}")
        elif not np.issubdtype(dataset.dtype, np.floating):
            raise MaterializationError(f"{source_label}: {key} must have a floating dtype, got {dataset.dtype}")

    _validate_optional_count(demo.attrs, "num_samples", length, source_label=source_label)
    actions = np.asarray(actions_dataset)
    ee_state = np.asarray(demo["obs/ee_states"])
    gripper_state = np.asarray(demo["obs/gripper_states"])
    for key, values in (
        ("actions", actions),
        ("obs/ee_states", ee_state),
        ("obs/gripper_states", gripper_state),
    ):
        if not np.isfinite(values).all():
            raise MaterializationError(f"{source_label}: {key} contains non-finite values")
    if not np.isin(actions[:, 6], (-1.0, 1.0)).all():
        unique = np.unique(actions[:, 6]).tolist()
        raise MaterializationError(f"{source_label}: raw gripper actions must be exactly -1 or +1, got {unique}")
    return length


def _validate_environment_metadata(
    env_args: Mapping[str, Any],
    *,
    source_label: str,
) -> None:
    env_kwargs = env_args.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping):
        raise MaterializationError(f"{source_label}: env_args.env_kwargs must be an object")
    if env_kwargs.get("control_freq") != LIBERO_FPS:
        raise MaterializationError(
            f"{source_label}: control_freq must be {LIBERO_FPS}, got {env_kwargs.get('control_freq')!r}"
        )
    controller = env_kwargs.get("controller_configs")
    if not isinstance(controller, Mapping):
        raise MaterializationError(f"{source_label}: env_args.env_kwargs.controller_configs must be an object")
    if controller.get("type") != "OSC_POSE":
        raise MaterializationError(
            f"{source_label}: controller type must be 'OSC_POSE', got {controller.get('type')!r}"
        )
    if controller.get("control_delta") is not True:
        raise MaterializationError(f"{source_label}: controller control_delta must be true")


def _read_json_attribute(
    attributes: Any,
    name: str,
    *,
    source_label: str,
) -> Mapping[str, Any]:
    if name not in attributes:
        raise MaterializationError(f"{source_label}: missing HDF5 attribute {name!r}")
    raw = attributes[name]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise MaterializationError(f"{source_label}: HDF5 attribute {name!r} must contain JSON text")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"{source_label}: HDF5 attribute {name!r} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise MaterializationError(f"{source_label}: HDF5 attribute {name!r} must decode to an object")
    return value


def _validate_optional_count(
    attributes: Any,
    name: str,
    expected: int,
    *,
    source_label: str,
) -> None:
    if name not in attributes:
        return
    try:
        actual = int(attributes[name])
    except (TypeError, ValueError) as exc:
        raise MaterializationError(f"{source_label}: HDF5 attribute {name!r} must be an integer") from exc
    if actual != expected:
        raise MaterializationError(f"{source_label}: HDF5 attribute {name!r} is {actual}, expected {expected}")


def _demo_sort_key(name: str) -> int:
    prefix = "demo_"
    if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
        raise MaterializationError(f"demo keys must be named 'demo_<non-negative integer>', got {name!r}")
    return int(name[len(prefix) :])


def _apply_image_transform(
    image: np.ndarray,
    image_transform: ImageTransform,
) -> np.ndarray:
    transform = _validate_image_transform(image_transform)
    array = np.asarray(image)
    if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise MaterializationError(f"image must have shape {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}, got {array.shape}")
    if array.dtype != np.uint8:
        raise MaterializationError(f"image must have dtype uint8, got {array.dtype}")
    if transform == "none":
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array[::-1, ::-1])


def _image_stats(
    frames: Any,
    *,
    image_transform: ImageTransform,
) -> dict[str, Any]:
    length = int(frames.shape[0])
    minimum_samples = min(100, length)
    sample_count = min(
        length,
        max(minimum_samples, min(int(length**0.75), 10_000)),
    )
    indices = np.linspace(0, length - 1, sample_count, dtype=np.int64)
    samples = np.stack(
        [_apply_image_transform(np.asarray(frames[int(index)]), image_transform) for index in indices],
        axis=0,
    )
    normalized = samples.astype(np.float32) / np.float32(255.0)
    channels_first = normalized.transpose(0, 3, 1, 2)
    return _array_stats(channels_first, axes=(0, 2, 3))


def _array_stats(
    values: np.ndarray,
    *,
    axes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    array = np.asarray(values)
    if array.size == 0:
        raise MaterializationError("cannot compute statistics for an empty array")
    target_shape = tuple(1 if axis in axes else array.shape[axis] for axis in range(1, array.ndim))
    return {
        "min": np.min(array, axis=axes).reshape(target_shape).tolist(),
        "max": np.max(array, axis=axes).reshape(target_shape).tolist(),
        "mean": np.mean(array, axis=axes, dtype=np.float64).reshape(target_shape).tolist(),
        "std": np.std(array, axis=axes, dtype=np.float64).reshape(target_shape).tolist(),
        "count": [int(array.shape[0])],
    }


def _initialize_or_validate_staging(
    staging: Path,
    *,
    run_spec: Mapping[str, Any],
) -> None:
    run_path = staging / ".materialization" / "run.json"
    if staging.exists():
        if not staging.is_dir():
            raise MaterializationError(f"partial materialization is not a directory: {staging}")
        if not run_path.is_file():
            raise MaterializationError(f"partial materialization has no run manifest: {run_path}")
        recorded = read_json(run_path)
        if recorded != dict(run_spec):
            raise MaterializationError(f"partial materialization belongs to a different run: {staging}")
        return
    run_path.parent.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(run_path, dict(run_spec))


def _run_spec(
    plan: MaterializationPlan,
    video_encoding: VideoEncodingConfig,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "plan_sha256": plan.sha256,
        "suite": plan.suite,
        "codebase_version": LEROBOT_VERSION,
        "fps": LIBERO_FPS,
        "image_transform": plan.image_transform,
        "video_encoding": video_encoding.to_dict(),
    }
    content["run_sha256"] = json_sha256(content)
    return content


def _validate_image_transform(value: str) -> ImageTransform:
    if value not in IMAGE_TRANSFORMS:
        raise ValueError(f"image_transform must be one of {sorted(IMAGE_TRANSFORMS)}, got {value!r}")
    return value  # type: ignore[return-value]


def _relative_source_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.tmp{target.suffix}")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(path)
    temp.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_path(path)
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _hf_sequence_feature(dtype: str, length: int) -> dict[str, Any]:
    return {
        "feature": _hf_value_feature(dtype),
        "length": length,
        "_type": "Sequence",
    }


def _hf_value_feature(dtype: str) -> dict[str, str]:
    return {"dtype": dtype, "_type": "Value"}


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LIBERO materialization requires h5py; install the data extra") from exc
    return h5py


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LIBERO materialization requires pyarrow; install the data extra") from exc
    return pa, pq


def _require_av():
    try:
        import av
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LIBERO materialization requires av; install the data extra") from exc
    return av


__all__ = [
    "IMAGE_TRANSFORMS",
    "LIBERO_SUITES",
    "EpisodePlan",
    "MaterializationError",
    "MaterializationPlan",
    "SourceFilePlan",
    "VideoEncodingConfig",
    "build_libero_v21_plan",
    "materialize_libero_v21",
    "materialize_libero_v21_plan",
]
