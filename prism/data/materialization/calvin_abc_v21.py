"""Fail-closed reconstruction of the complete Collision CALVIN ABC-D LeRobot v2.1 root."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
from functools import cached_property
import os
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np

from prism.data.materialization.libero_v21 import MaterializationError
from prism.data.materialization.libero_v21 import _canonical_json
from prism.data.materialization.libero_v21 import _file_sha256
from prism.data.materialization.libero_v21 import _json_sha256
from prism.data.materialization.libero_v21 import _read_json
from prism.data.materialization.libero_v21 import _read_jsonl

COLLISION_REPOSITORY = "CollisionCode/calvin_abc_d_lerobot_v2.1"
COLLISION_REVISION = "7e206b2aa210c5166276b8e9777955bfd1a1e8ac"
TRALY_REPOSITORY = "Traly/calvin_abc_d-lerobot"
TRALY_REVISION = "92bf05b93a4ba8a8825f2bffb1b78ff4cb4e6c63"
TRALY_EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
TRALY_DATA_PATHS = (
    "data/chunk-000/file-000.parquet",
    "data/chunk-000/file-001.parquet",
    "data/chunk-000/file-002.parquet",
)
COLLISION_COMPLETED_VIDEO_PATHS = (
    "videos/chunk-008/wrist_image/episode_008904.mp4",
    "videos/chunk-008/wrist_image/episode_008965.mp4",
    "videos/chunk-008/wrist_image/episode_008994.mp4",
)
TRALY_REQUIRED_PATHS = (
    "meta/info.json",
    "meta/tasks.parquet",
    TRALY_EPISODES_PATH,
    *TRALY_DATA_PATHS,
)
TARGET_FPS = 10
TARGET_VIEWS = ("image", "wrist_image")
PARQUET_COLUMNS = (
    "state",
    "actions",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
_DATA_PATTERN = re.compile(r"data/chunk-(\d{3})/episode_(\d{6})\.parquet")
_VIDEO_PATTERN = re.compile(r"videos/chunk-(\d{3})/(image|wrist_image)/episode_(\d{6})\.mp4")
_PROGRESS = Callable[[str], None]


@dataclass(frozen=True)
class CalvinABCContract:
    """Fixed physical and logical cardinalities for one reconstruction."""

    target_episodes: int = 17_870
    target_frames: int = 1_071_743
    target_tasks: int = 389
    target_videos: int = 35_740
    target_present_parquets: int = 7_870
    donor_episodes: int = 17_870
    donor_frames: int = 1_071_743
    donor_tasks: int = 389
    chunks_size: int = 1_000

    def to_dict(self) -> dict[str, int]:
        return {
            "target_episodes": self.target_episodes,
            "target_frames": self.target_frames,
            "target_tasks": self.target_tasks,
            "target_videos": self.target_videos,
            "target_present_parquets": self.target_present_parquets,
            "donor_episodes": self.donor_episodes,
            "donor_frames": self.donor_frames,
            "donor_tasks": self.donor_tasks,
            "chunks_size": self.chunks_size,
        }


CALVIN_ABC_CONTRACT = CalvinABCContract()


@dataclass(frozen=True)
class SourceArtifact:
    """One hash-verified input artifact, addressed relative to its source root."""

    source: str
    relative_path: str
    size_bytes: int
    sha256: str
    remote_identity_kind: str
    remote_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "remote_identity_kind": self.remote_identity_kind,
            "remote_identity": self.remote_identity,
        }


@dataclass(frozen=True)
class EpisodeMapping:
    """Exact target-to-Traly permutation and numeric slice."""

    target_episode_index: int
    donor_episode_index: int
    donor_from_index: int
    donor_to_index: int
    length: int
    task: str
    signature_sha256: str
    data_file_index: int
    target_task_index: int
    target_from_index: int
    target_to_index: int

    def mapping_dict(self) -> dict[str, Any]:
        """Canonical public mapping row; its JSONL hash is an audit invariant."""

        return {
            "target_episode_index": self.target_episode_index,
            "donor_episode_index": self.donor_episode_index,
            "donor_from_index": self.donor_from_index,
            "donor_to_index": self.donor_to_index,
            "length": self.length,
            "task": self.task,
            "signature_sha256": self.signature_sha256,
            "data_file_index": self.data_file_index,
        }

    def to_dict(self) -> dict[str, Any]:
        row = self.mapping_dict()
        row.update(
            {
                "target_task_index": self.target_task_index,
                "target_from_index": self.target_from_index,
                "target_to_index": self.target_to_index,
            }
        )
        return row


@dataclass(frozen=True)
class CalvinABCMaterializationPlan:
    """Validated reconstruction plan. Serialized paths are always relative."""

    collision_root: Path
    donor_root: Path
    contract: CalvinABCContract
    collision_tree_sha256: str
    collision_artifacts: tuple[SourceArtifact, ...]
    donor_artifacts: tuple[SourceArtifact, ...]
    mappings: tuple[EpisodeMapping, ...]
    present_episode_ids: tuple[int, ...]
    static_collision_paths: tuple[str, ...]

    @cached_property
    def mapping_sha256(self) -> str:
        return _jsonl_sha256(mapping.mapping_dict() for mapping in self.mappings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "prism-calvin-abc-v21-plan-v1",
            "target": {
                "repository": COLLISION_REPOSITORY,
                "revision": COLLISION_REVISION,
                "tree_manifest_sha256": self.collision_tree_sha256,
            },
            "numeric_donor": {
                "repository": TRALY_REPOSITORY,
                "revision": TRALY_REVISION,
            },
            "contract": self.contract.to_dict(),
            "collision_artifacts": [item.to_dict() for item in self.collision_artifacts],
            "donor_artifacts": [item.to_dict() for item in self.donor_artifacts],
            "mapping_sha256": self.mapping_sha256,
            "mappings": [item.to_dict() for item in self.mappings],
            "present_episode_ids": list(self.present_episode_ids),
            "static_collision_paths": list(self.static_collision_paths),
        }

    @cached_property
    def sha256(self) -> str:
        return _json_sha256(self.to_dict())


@dataclass(frozen=True)
class ExistingGateReport:
    checked_episodes: int
    checked_frames: int
    schema_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_episodes": self.checked_episodes,
            "checked_frames": self.checked_frames,
            "schema_fingerprint": self.schema_fingerprint,
            "state_mismatches": 0,
            "action_mismatches": 0,
            "first_action_mismatches": 0,
            "scalar_mismatches": 0,
        }


def build_calvin_abc_v21_plan(
    collision_root: str | Path,
    donor_root: str | Path,
    *,
    contract: CalvinABCContract = CALVIN_ABC_CONTRACT,
    hash_workers: int = 8,
    progress: _PROGRESS | None = None,
) -> CalvinABCMaterializationPlan:
    """Validate both pinned authorities and construct the exact episode bijection."""

    if hash_workers <= 0:
        raise ValueError("hash_workers must be positive")
    collision = Path(collision_root).expanduser().resolve()
    donor = Path(donor_root).expanduser().resolve()
    if not collision.is_dir():
        raise FileNotFoundError(collision)
    if not donor.is_dir():
        raise FileNotFoundError(donor)

    _emit(progress, "validating Collision metadata and pinned tree")
    target = _validate_target_metadata(collision, contract)
    collision_tree_sha256, collision_artifacts, present_ids, static_paths = _validate_collision_tree(
        collision, contract, hash_workers=hash_workers
    )

    _emit(progress, "validating Traly metadata, revision, and source hashes")
    donor_artifacts = _validate_donor_artifacts(donor)
    donor_rows = _validate_donor_metadata(donor, contract)

    _emit(progress, "building exact min/max/mean episode bijection")
    mappings = _build_episode_mappings(target, donor_rows, contract)
    return CalvinABCMaterializationPlan(
        collision_root=collision,
        donor_root=donor,
        contract=contract,
        collision_tree_sha256=collision_tree_sha256,
        collision_artifacts=collision_artifacts,
        donor_artifacts=donor_artifacts,
        mappings=mappings,
        present_episode_ids=present_ids,
        static_collision_paths=static_paths,
    )


def materialize_calvin_abc_v21(
    plan: CalvinABCMaterializationPlan,
    output_root: str | Path,
    *,
    resume: bool = True,
    decode_samples: bool = True,
    progress: _PROGRESS | None = None,
) -> Path:
    """Reconstruct a complete sibling root without modifying either input root."""

    output = Path(output_root).expanduser().resolve()
    _validate_output_location(plan, output)
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite completed dataset: {output}")
        _validate_completed_run(plan, output, decode_samples=decode_samples, progress=progress)
        return output

    _emit(progress, "loading and validating all Traly numeric rows")
    donor_store = _DonorStore(plan)
    _emit(progress, "running bit-exact gate over every existing Collision Parquet")
    schema, gate_report = _validate_existing_gate(plan, donor_store, progress=progress)

    staging = output.parent / f".{output.name}.calvin-v21.partial-{plan.sha256[:16]}"
    if staging.exists() and not resume:
        raise FileExistsError(f"refusing to resume partial dataset: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    lock_path = staging / ".materialization" / "lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MaterializationError(f"another materialization owns {staging}") from exc

        _cleanup_stale_temps(staging)
        run_spec = _run_spec(plan)
        _write_json_idempotent(staging / ".materialization" / "run.json", run_spec)
        _write_jsonl_idempotent(
            staging / ".materialization" / "sources" / "collision.jsonl",
            (item.to_dict() for item in plan.collision_artifacts),
        )
        _write_jsonl_idempotent(
            staging / ".materialization" / "sources" / "traly.jsonl",
            (item.to_dict() for item in plan.donor_artifacts),
        )
        _write_jsonl_idempotent(
            staging / ".materialization" / "episode_mapping.jsonl",
            (item.mapping_dict() for item in plan.mappings),
        )
        _write_json_idempotent(
            staging / ".materialization" / "existing_gate.json",
            gate_report.to_dict(),
        )

        _emit(progress, "hardlinking immutable Collision metadata")
        for relative_path in plan.static_collision_paths:
            _link_or_validate(
                plan.collision_root / relative_path,
                staging / relative_path,
            )

        artifact_by_path = {item.relative_path: item for item in plan.collision_artifacts}
        present = set(plan.present_episode_ids)
        _emit(progress, "materializing 17,870 episode transactions")
        for index, mapping in enumerate(plan.mappings):
            journal_path = staging / ".materialization" / "journal" / f"episode_{mapping.target_episode_index:06d}.json"
            if journal_path.is_file():
                _validate_episode_journal(
                    plan,
                    staging,
                    mapping,
                    journal_path,
                    artifact_by_path,
                    donor_store,
                    schema,
                )
                continue

            data_relative = _target_data_path(mapping.target_episode_index, plan.contract.chunks_size)
            data_target = staging / data_relative
            if mapping.target_episode_index in present:
                _link_or_validate(plan.collision_root / data_relative, data_target)
                data_mode = "hardlink"
                data_sha256 = artifact_by_path[data_relative].sha256
            else:
                _publish_generated_episode(
                    data_target,
                    mapping,
                    donor_store,
                    schema,
                )
                data_mode = "generated"
                data_sha256 = _file_sha256(data_target)

            videos: list[dict[str, Any]] = []
            for view in TARGET_VIEWS:
                relative = _target_video_path(
                    mapping.target_episode_index,
                    view,
                    plan.contract.chunks_size,
                )
                _link_or_validate(plan.collision_root / relative, staging / relative)
                videos.append(
                    {
                        "path": relative,
                        "sha256": artifact_by_path[relative].sha256,
                        "mode": "hardlink",
                    }
                )

            _write_json_idempotent(
                journal_path,
                {
                    "schema_version": "prism-calvin-abc-v21-episode-v1",
                    "plan_sha256": plan.sha256,
                    "mapping": mapping.to_dict(),
                    "data": {
                        "path": data_relative,
                        "mode": data_mode,
                        "sha256": data_sha256,
                        "size_bytes": data_target.stat().st_size,
                        "rows": mapping.length,
                    },
                    "videos": videos,
                },
            )
            if progress is not None and ((index + 1) % 1_000 == 0 or index + 1 == len(plan.mappings)):
                progress(f"episodes complete: {index + 1}/{len(plan.mappings)}")

        provenance = {
            "schema_version": "prism-calvin-abc-v21-materialization-v1",
            "plan_sha256": plan.sha256,
            "mapping_sha256": plan.mapping_sha256,
            "target_repository": COLLISION_REPOSITORY,
            "target_revision": COLLISION_REVISION,
            "numeric_donor_repository": TRALY_REPOSITORY,
            "numeric_donor_revision": TRALY_REVISION,
            "collision_source_completion": {
                "description": (
                    "These three files were absent after the initial missing-only "
                    "snapshot recovery and were downloaded from the pinned Collision "
                    "revision before plan construction."
                ),
                "repository": COLLISION_REPOSITORY,
                "revision": COLLISION_REVISION,
                "artifacts": [
                    artifact_by_path[path].to_dict()
                    for path in COLLISION_COMPLETED_VIDEO_PATHS
                    if path in artifact_by_path
                ],
            },
            "numeric_conversion": {
                "state": "float32([observation.state[:6], 0, observation.state[6]])",
                "actions": "float32(action.relative), no shift, including row zero",
                "timestamp": "float32(frame_index) / float32(10)",
                "indices": "regenerated from Collision target ordering",
            },
            "existing_gate": gate_report.to_dict(),
        }
        _write_json_idempotent(staging / "meta" / "materialization.json", provenance)

        _emit(progress, "running strict full-root validation")
        validation = _validate_full_root(
            plan,
            staging,
            donor_store,
            schema,
            decode_samples=decode_samples,
            progress=progress,
        )
        _write_json_idempotent(staging / ".materialization" / "validation.json", validation)
        _fsync_tree_metadata(staging)
        _rename_dir_noreplace(staging, output)
        _fsync_dir(output.parent)

    return output


def _validate_target_metadata(root: Path, contract: CalvinABCContract) -> dict[str, Any]:
    info = _read_json(root / "meta" / "info.json")
    expected = {
        "codebase_version": "v2.1",
        "total_episodes": contract.target_episodes,
        "total_frames": contract.target_frames,
        "total_tasks": contract.target_tasks,
        "total_videos": contract.target_videos,
        "chunks_size": contract.chunks_size,
        "fps": TARGET_FPS,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise MaterializationError(f"Collision info field {key!r} is {info.get(key)!r}, expected {value!r}")
    if info.get("data_path") != ("data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"):
        raise MaterializationError("unexpected Collision data_path")
    if info.get("video_path") != ("videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"):
        raise MaterializationError("unexpected Collision video_path")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise MaterializationError("Collision info.features must be an object")
    if features.get("state", {}).get("shape") != [8]:
        raise MaterializationError("Collision state feature must have shape [8]")
    if features.get("actions", {}).get("shape") != [7]:
        raise MaterializationError("Collision actions feature must have shape [7]")
    for view in TARGET_VIEWS:
        if features.get(view, {}).get("dtype") != "video":
            raise MaterializationError(f"Collision view {view!r} must be a video")

    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    stats = _read_jsonl(root / "meta" / "episodes_stats.jsonl")
    if len(tasks) != contract.target_tasks:
        raise MaterializationError("Collision task count mismatch")
    if len(episodes) != contract.target_episodes:
        raise MaterializationError("Collision episode count mismatch")
    if len(stats) != contract.target_episodes:
        raise MaterializationError("Collision episode stats count mismatch")
    _require_contiguous_ids(tasks, "task_index", "Collision tasks")
    _require_contiguous_ids(episodes, "episode_index", "Collision episodes")
    _require_contiguous_ids(stats, "episode_index", "Collision episode stats")

    task_to_index: dict[str, int] = {}
    for row in tasks:
        task = row.get("task")
        if not isinstance(task, str) or not task:
            raise MaterializationError("Collision task text must be non-empty")
        if task in task_to_index:
            raise MaterializationError(f"duplicate Collision task text: {task!r}")
        task_to_index[task] = int(row["task_index"])

    offsets = [0]
    normalized_episodes: list[dict[str, Any]] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        length = row.get("length")
        episode_tasks = row.get("tasks")
        if type(length) is not int or length <= 0:
            raise MaterializationError(f"Collision episode {episode_index} has invalid length")
        if not isinstance(episode_tasks, list) or len(episode_tasks) != 1 or episode_tasks[0] not in task_to_index:
            raise MaterializationError(f"Collision episode {episode_index} must reference one known task")
        offsets.append(offsets[-1] + length)
        normalized_episodes.append(
            {
                "episode_index": episode_index,
                "length": length,
                "task": episode_tasks[0],
                "task_index": task_to_index[episode_tasks[0]],
                "from_index": offsets[-2],
                "to_index": offsets[-1],
            }
        )
    referenced_tasks = {episode["task"] for episode in normalized_episodes}
    if referenced_tasks != set(task_to_index):
        raise MaterializationError("Collision task metadata and episode task set differ")
    if offsets[-1] != contract.target_frames:
        raise MaterializationError(f"Collision episode lengths total {offsets[-1]}, expected {contract.target_frames}")

    for episode, stats_row in zip(normalized_episodes, stats, strict=True):
        payload = stats_row.get("stats")
        if not isinstance(payload, Mapping):
            raise MaterializationError(f"Collision episode {episode['episode_index']} stats missing")
        for feature in (
            "state",
            "actions",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ):
            feature_stats = payload.get(feature)
            if not isinstance(feature_stats, Mapping):
                raise MaterializationError(f"Collision episode {episode['episode_index']} missing {feature} stats")
            if feature_stats.get("count") != [episode["length"]]:
                raise MaterializationError(f"Collision episode {episode['episode_index']} {feature} count mismatch")
        state_stats = payload["state"]
        action_stats = payload["actions"]
        for aggregate in ("min", "max", "mean"):
            if len(state_stats.get(aggregate, ())) != 8:
                raise MaterializationError(f"Collision episode {episode['episode_index']} state stats width mismatch")
            if len(action_stats.get(aggregate, ())) != 7:
                raise MaterializationError(f"Collision episode {episode['episode_index']} action stats width mismatch")
            if float(state_stats[aggregate][6]) != 0.0:
                raise MaterializationError(f"Collision episode {episode['episode_index']} state pad stats are not zero")

    return {
        "info": info,
        "tasks": tasks,
        "episodes": normalized_episodes,
        "stats": stats,
    }


def _validate_collision_tree(
    root: Path,
    contract: CalvinABCContract,
    *,
    hash_workers: int,
) -> tuple[str, tuple[SourceArtifact, ...], tuple[int, ...], tuple[str, ...]]:
    tree_path = root / ".cache" / "huggingface" / "trees" / f"{COLLISION_REVISION}.json"
    if not tree_path.is_file():
        raise MaterializationError(f"missing pinned Collision tree manifest: {tree_path}")
    tree = _read_json(tree_path)
    files = tree.get("files")
    if not isinstance(files, Mapping) or not files:
        raise MaterializationError("Collision tree manifest has no files")

    paths = tuple(sorted(str(path) for path in files))
    data_ids: list[int] = []
    video_ids: dict[str, list[int]] = defaultdict(list)
    for relative_path in paths:
        data_match = _DATA_PATTERN.fullmatch(relative_path)
        if data_match:
            chunk, episode = map(int, data_match.groups())
            if chunk != episode // contract.chunks_size:
                raise MaterializationError(f"wrong data chunk for episode {episode}: {relative_path}")
            data_ids.append(episode)
            continue
        video_match = _VIDEO_PATTERN.fullmatch(relative_path)
        if video_match:
            chunk = int(video_match.group(1))
            view = video_match.group(2)
            episode = int(video_match.group(3))
            if chunk != episode // contract.chunks_size:
                raise MaterializationError(f"wrong video chunk for episode {episode}: {relative_path}")
            video_ids[view].append(episode)

    if len(data_ids) != contract.target_present_parquets:
        raise MaterializationError(
            f"Collision pinned tree has {len(data_ids)} Parquets, expected {contract.target_present_parquets}"
        )
    if len(set(data_ids)) != len(data_ids):
        raise MaterializationError("Collision tree has duplicate episode Parquets")
    expected_ids = list(range(contract.target_episodes))
    for view in TARGET_VIEWS:
        if sorted(video_ids[view]) != expected_ids:
            raise MaterializationError(f"Collision tree does not contain one {view} video per episode")
    if sum(len(ids) for ids in video_ids.values()) != contract.target_videos:
        raise MaterializationError("Collision tree video count mismatch")

    physical_data = {path.relative_to(root).as_posix() for path in (root / "data").rglob("*.parquet") if path.is_file()}
    physical_videos = {path.relative_to(root).as_posix() for path in (root / "videos").rglob("*.mp4") if path.is_file()}
    manifest_data = {path for path in paths if path.startswith("data/")}
    manifest_videos = {path for path in paths if path.startswith("videos/")}
    if physical_data != manifest_data:
        raise MaterializationError("physical Collision Parquet set differs from pinned tree")
    if physical_videos != manifest_videos:
        missing = sorted(manifest_videos - physical_videos)[:10]
        extra = sorted(physical_videos - manifest_videos)[:10]
        raise MaterializationError(
            f"physical Collision video set differs from pinned tree; missing={missing}, extra={extra}"
        )

    def verify(relative_path: str) -> SourceArtifact:
        metadata = files[relative_path]
        if not isinstance(metadata, Mapping):
            raise MaterializationError(f"invalid tree entry for {relative_path}")
        source_path = root / relative_path
        if source_path.is_symlink() or not source_path.is_file():
            raise MaterializationError(f"Collision artifact is missing or a symlink: {relative_path}")
        expected_size = metadata.get("size")
        if source_path.stat().st_size != expected_size:
            raise MaterializationError(f"Collision artifact size mismatch: {relative_path}")
        sha256, git_blob = _file_digests(source_path)
        lfs_sha256 = metadata.get("lfs_sha256")
        if lfs_sha256 is not None:
            if sha256 != lfs_sha256 or metadata.get("lfs_size") != expected_size:
                raise MaterializationError(f"Collision LFS digest mismatch: {relative_path}")
            identity_kind = "lfs_sha256"
            identity = str(lfs_sha256)
        else:
            blob_id = metadata.get("blob_id")
            if git_blob != blob_id:
                raise MaterializationError(f"Collision Git blob mismatch: {relative_path}")
            identity_kind = "git_blob"
            identity = str(blob_id)
        return SourceArtifact(
            source="collision",
            relative_path=relative_path,
            size_bytes=int(expected_size),
            sha256=sha256,
            remote_identity_kind=identity_kind,
            remote_identity=identity,
        )

    with ThreadPoolExecutor(max_workers=hash_workers) as executor:
        artifacts = tuple(executor.map(verify, paths))

    static_paths = tuple(path for path in paths if not path.startswith("data/") and not path.startswith("videos/"))
    return (
        _file_sha256(tree_path),
        artifacts,
        tuple(sorted(data_ids)),
        static_paths,
    )


def _validate_donor_artifacts(root: Path) -> tuple[SourceArtifact, ...]:
    artifacts: list[SourceArtifact] = []
    for relative_path in TRALY_REQUIRED_PATHS:
        source_path = root / relative_path
        metadata_path = root / ".cache" / "huggingface" / "download" / f"{relative_path}.metadata"
        if source_path.is_symlink() or not source_path.is_file():
            raise MaterializationError(f"Traly artifact is missing or a symlink: {relative_path}")
        if not metadata_path.is_file():
            raise MaterializationError(f"Traly artifact lacks download metadata: {relative_path}")
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0] != TRALY_REVISION:
            raise MaterializationError(f"Traly revision mismatch for {relative_path}")
        sha256, git_blob = _file_digests(source_path)
        identity = lines[1]
        if len(identity) == 64:
            if sha256 != identity:
                raise MaterializationError(f"Traly LFS digest mismatch: {relative_path}")
            identity_kind = "lfs_sha256"
        elif len(identity) == 40:
            if git_blob != identity:
                raise MaterializationError(f"Traly Git blob mismatch: {relative_path}")
            identity_kind = "git_blob"
        else:
            raise MaterializationError(f"unrecognized Traly remote identity: {relative_path}")
        artifacts.append(
            SourceArtifact(
                source="traly",
                relative_path=relative_path,
                size_bytes=source_path.stat().st_size,
                sha256=sha256,
                remote_identity_kind=identity_kind,
                remote_identity=identity,
            )
        )
    return tuple(artifacts)


def _validate_donor_metadata(root: Path, contract: CalvinABCContract) -> tuple[dict[str, Any], ...]:
    pa, pq = _require_pyarrow()
    del pa
    info = _read_json(root / "meta" / "info.json")
    expected = {
        "codebase_version": "v3.0",
        "total_episodes": contract.donor_episodes,
        "total_frames": contract.donor_frames,
        "total_tasks": contract.donor_tasks,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise MaterializationError(f"Traly info field {key!r} is {info.get(key)!r}, expected {value!r}")

    tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
    if tasks_table.num_rows != contract.donor_tasks:
        raise MaterializationError("Traly task count mismatch")
    if "__index_level_0__" not in tasks_table.column_names:
        raise MaterializationError("Traly task text index column is missing")
    donor_tasks = tasks_table["__index_level_0__"].to_pylist()
    if len(set(donor_tasks)) != len(donor_tasks):
        raise MaterializationError("Traly task text is not unique")

    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for base in ("stats/observation.state", "stats/action.relative"):
        for aggregate in ("min", "max", "mean", "count"):
            columns.append(f"{base}/{aggregate}")
    table = pq.read_table(root / TRALY_EPISODES_PATH, columns=columns)
    if table.num_rows != contract.donor_episodes:
        raise MaterializationError("Traly episode count mismatch")
    rows = tuple(table.to_pylist())
    _require_contiguous_ids(rows, "episode_index", "Traly episodes")

    cursor = 0
    per_file_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        episode_index = int(row["episode_index"])
        length = row.get("length")
        start = row.get("dataset_from_index")
        end = row.get("dataset_to_index")
        file_index = row.get("data/file_index")
        if type(length) is not int or length <= 0:
            raise MaterializationError(f"Traly episode {episode_index} has invalid length")
        if start != cursor or end != cursor + length:
            raise MaterializationError(f"Traly episode {episode_index} has non-contiguous global range")
        if row.get("data/chunk_index") != 0:
            raise MaterializationError("Traly numeric data must use chunk 000")
        if type(file_index) is not int or not 0 <= file_index < len(TRALY_DATA_PATHS):
            raise MaterializationError(f"Traly episode {episode_index} has invalid data file index")
        episode_tasks = row.get("tasks")
        if not isinstance(episode_tasks, list) or len(episode_tasks) != 1 or episode_tasks[0] not in donor_tasks:
            raise MaterializationError(f"Traly episode {episode_index} must reference one known task")
        for feature in ("stats/observation.state", "stats/action.relative"):
            if row[f"{feature}/count"] != [length]:
                raise MaterializationError(f"Traly episode {episode_index} {feature} count mismatch")
        state_widths = [len(row[f"stats/observation.state/{aggregate}"]) for aggregate in ("min", "max", "mean")]
        action_widths = [len(row[f"stats/action.relative/{aggregate}"]) for aggregate in ("min", "max", "mean")]
        if state_widths != [15, 15, 15] or action_widths != [7, 7, 7]:
            raise MaterializationError(f"Traly episode {episode_index} stats width mismatch")
        per_file_ranges[file_index].append((start, end))
        cursor = end
    referenced_tasks = {row["tasks"][0] for row in rows}
    if referenced_tasks != set(donor_tasks):
        raise MaterializationError("Traly task metadata and episode task set differ")
    if cursor != contract.donor_frames:
        raise MaterializationError(f"Traly episode ranges end at {cursor}, expected {contract.donor_frames}")

    seen_files = sorted(per_file_ranges)
    if seen_files != list(range(len(TRALY_DATA_PATHS))):
        raise MaterializationError(f"Traly metadata references data files {seen_files}")
    previous_end = 0
    for file_index in seen_files:
        ranges = per_file_ranges[file_index]
        if ranges[0][0] != previous_end:
            raise MaterializationError("Traly data-file global ranges are not contiguous")
        for left, right in zip(ranges, ranges[1:]):
            if left[1] != right[0]:
                raise MaterializationError(f"Traly data file {file_index} has a gap")
        previous_end = ranges[-1][1]

    return rows


def _build_episode_mappings(
    target: Mapping[str, Any],
    donor_rows: Sequence[Mapping[str, Any]],
    contract: CalvinABCContract,
) -> tuple[EpisodeMapping, ...]:
    donor_by_signature: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in donor_rows:
        signature = _mapping_signature(
            task=row["tasks"][0],
            length=int(row["length"]),
            state_min=row["stats/observation.state/min"][:7],
            state_max=row["stats/observation.state/max"][:7],
            state_mean=row["stats/observation.state/mean"][:7],
            action_min=row["stats/action.relative/min"],
            action_max=row["stats/action.relative/max"],
            action_mean=row["stats/action.relative/mean"],
        )
        donor_by_signature[signature].append(row)

    mappings: list[EpisodeMapping] = []
    unmatched: list[int] = []
    ambiguous: list[tuple[int, list[int]]] = []
    for episode, stats_row in zip(target["episodes"], target["stats"], strict=True):
        stats = stats_row["stats"]
        state_indices = (0, 1, 2, 3, 4, 5, 7)
        signature = _mapping_signature(
            task=episode["task"],
            length=episode["length"],
            state_min=[stats["state"]["min"][index] for index in state_indices],
            state_max=[stats["state"]["max"][index] for index in state_indices],
            state_mean=[stats["state"]["mean"][index] for index in state_indices],
            action_min=stats["actions"]["min"],
            action_max=stats["actions"]["max"],
            action_mean=stats["actions"]["mean"],
        )
        matches = donor_by_signature.get(signature, [])
        if not matches:
            unmatched.append(episode["episode_index"])
            continue
        if len(matches) != 1:
            ambiguous.append(
                (
                    episode["episode_index"],
                    [int(row["episode_index"]) for row in matches],
                )
            )
            continue
        donor = matches[0]
        mappings.append(
            EpisodeMapping(
                target_episode_index=episode["episode_index"],
                donor_episode_index=int(donor["episode_index"]),
                donor_from_index=int(donor["dataset_from_index"]),
                donor_to_index=int(donor["dataset_to_index"]),
                length=episode["length"],
                task=episode["task"],
                signature_sha256=hashlib.sha256(_canonical_json(signature).encode("utf-8")).hexdigest(),
                data_file_index=int(donor["data/file_index"]),
                target_task_index=episode["task_index"],
                target_from_index=episode["from_index"],
                target_to_index=episode["to_index"],
            )
        )

    if unmatched or ambiguous:
        raise MaterializationError(
            "episode mapping is not total and unique: "
            f"unmatched={unmatched[:10]} ({len(unmatched)} total), "
            f"ambiguous={ambiguous[:10]} ({len(ambiguous)} total)"
        )
    if len(mappings) != contract.target_episodes:
        raise MaterializationError("episode mapping count mismatch")
    donor_ids = [mapping.donor_episode_index for mapping in mappings]
    if len(set(donor_ids)) != contract.donor_episodes:
        raise MaterializationError("episode mapping is not a donor bijection")
    if len(donor_by_signature) != contract.donor_episodes:
        duplicates = sum(len(rows) - 1 for rows in donor_by_signature.values() if len(rows) > 1)
        raise MaterializationError(f"Traly mapping signatures are not unique: {duplicates} duplicates")
    return tuple(mappings)


def _mapping_signature(
    *,
    task: str,
    length: int,
    state_min: Sequence[Any],
    state_max: Sequence[Any],
    state_mean: Sequence[Any],
    action_min: Sequence[Any],
    action_max: Sequence[Any],
    action_mean: Sequence[Any],
) -> tuple[Any, ...]:
    return (
        task,
        int(length),
        _float_tokens(state_min),
        _float_tokens(state_max),
        _float_tokens(state_mean),
        _float_tokens(action_min),
        _float_tokens(action_max),
        _float_tokens(action_mean),
    )


def _float_tokens(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(struct.pack(">d", float(value)).hex() for value in values)


class _DonorStore:
    def __init__(self, plan: CalvinABCMaterializationPlan) -> None:
        pa, pq = _require_pyarrow()
        self._files: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
        mappings_by_file: dict[int, list[EpisodeMapping]] = defaultdict(list)
        for mapping in plan.mappings:
            mappings_by_file[mapping.data_file_index].append(mapping)

        for file_index, relative_path in enumerate(TRALY_DATA_PATHS):
            mappings = mappings_by_file[file_index]
            start = min(mapping.donor_from_index for mapping in mappings)
            end = max(mapping.donor_to_index for mapping in mappings)
            table = pq.read_table(
                plan.donor_root / relative_path,
                columns=[
                    "observation.state",
                    "action.relative",
                    "episode_index",
                    "frame_index",
                    "index",
                ],
            )
            if table.num_rows != end - start:
                raise MaterializationError(f"Traly data file {file_index} row count mismatch")
            states = _list_column_to_numpy(table["observation.state"], 15, pa=pa)
            actions = _list_column_to_numpy(table["action.relative"], 7, pa=pa)
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise MaterializationError(f"Traly data file {file_index} contains non-finite numeric values")
            if not np.isin(actions[:, 6], (-1.0, 1.0)).all():
                raise MaterializationError(f"Traly data file {file_index} has invalid relative gripper actions")
            indices = _scalar_column(table, "index", np.int64)
            if not np.array_equal(indices, np.arange(start, end, dtype=np.int64)):
                raise MaterializationError(f"Traly data file {file_index} has invalid global indices")
            episode_indices = _scalar_column(table, "episode_index", np.int64)
            frame_indices = _scalar_column(table, "frame_index", np.int64)
            for mapping in mappings:
                local_from = mapping.donor_from_index - start
                local_to = mapping.donor_to_index - start
                if not np.array_equal(
                    episode_indices[local_from:local_to],
                    np.full(mapping.length, mapping.donor_episode_index, dtype=np.int64),
                ):
                    raise MaterializationError(f"Traly donor episode column mismatch for {mapping.donor_episode_index}")
                if not np.array_equal(
                    frame_indices[local_from:local_to],
                    np.arange(mapping.length, dtype=np.int64),
                ):
                    raise MaterializationError(f"Traly donor frame column mismatch for {mapping.donor_episode_index}")
                donor_state = states[local_from:local_to, :7]
                donor_action = actions[local_from:local_to]
                signature = _mapping_signature(
                    task=mapping.task,
                    length=mapping.length,
                    state_min=np.min(donor_state, axis=0),
                    state_max=np.max(donor_state, axis=0),
                    state_mean=np.mean(donor_state, axis=0),
                    action_min=np.min(donor_action, axis=0),
                    action_max=np.max(donor_action, axis=0),
                    action_mean=np.mean(donor_action, axis=0),
                )
                signature_sha256 = hashlib.sha256(_canonical_json(signature).encode("utf-8")).hexdigest()
                if signature_sha256 != mapping.signature_sha256:
                    raise MaterializationError(
                        f"Traly numeric rows do not reproduce mapped stats for "
                        f"donor episode {mapping.donor_episode_index}"
                    )
            self._files[file_index] = (start, states, actions)

    def episode(self, mapping: EpisodeMapping) -> tuple[np.ndarray, np.ndarray]:
        start, states, actions = self._files[mapping.data_file_index]
        local_from = mapping.donor_from_index - start
        local_to = mapping.donor_to_index - start
        donor_state = states[local_from:local_to]
        state = np.concatenate(
            (
                donor_state[:, :6],
                np.zeros((mapping.length, 1), dtype=np.float32),
                donor_state[:, 6:7],
            ),
            axis=1,
        )
        return np.ascontiguousarray(state), np.ascontiguousarray(actions[local_from:local_to])


def _validate_existing_gate(
    plan: CalvinABCMaterializationPlan,
    donor_store: _DonorStore,
    *,
    progress: _PROGRESS | None,
) -> tuple[Any, ExistingGateReport]:
    _, pq = _require_pyarrow()
    if not plan.present_episode_ids:
        raise MaterializationError("no Collision Parquet is available as schema authority")
    first_path = plan.collision_root / _target_data_path(plan.present_episode_ids[0], plan.contract.chunks_size)
    schema = pq.read_schema(first_path)
    _validate_target_schema(schema)
    fingerprint = hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()

    frames = 0
    for index, episode_id in enumerate(plan.present_episode_ids):
        mapping = plan.mappings[episode_id]
        path = plan.collision_root / _target_data_path(episode_id, plan.contract.chunks_size)
        table = pq.read_table(path)
        if not table.schema.equals(schema, check_metadata=True):
            raise MaterializationError(f"Collision schema mismatch for episode {episode_id}")
        _validate_episode_table(table, mapping, donor_store)
        frames += mapping.length
        if progress is not None and ((index + 1) % 1_000 == 0 or index + 1 == len(plan.present_episode_ids)):
            progress(f"existing bit-exact gate: {index + 1}/{len(plan.present_episode_ids)}")
    return schema, ExistingGateReport(
        checked_episodes=len(plan.present_episode_ids),
        checked_frames=frames,
        schema_fingerprint=fingerprint,
    )


def _validate_episode_table(
    table: Any,
    mapping: EpisodeMapping,
    donor_store: _DonorStore,
) -> None:
    pa, _ = _require_pyarrow()
    if table.column_names != list(PARQUET_COLUMNS):
        raise MaterializationError(f"episode {mapping.target_episode_index} column order mismatch")
    if table.num_rows != mapping.length:
        raise MaterializationError(f"episode {mapping.target_episode_index} row count mismatch")
    expected_state, expected_actions = donor_store.episode(mapping)
    actual_state = _list_column_to_numpy(table["state"], 8, pa=pa)
    actual_actions = _list_column_to_numpy(table["actions"], 7, pa=pa)
    if not _float32_bits_equal(actual_state, expected_state):
        raise MaterializationError(f"episode {mapping.target_episode_index} state is not bit-exact")
    if not _float32_bits_equal(actual_actions, expected_actions):
        raise MaterializationError(f"episode {mapping.target_episode_index} actions are not bit-exact")
    if not _float32_bits_equal(actual_actions[:1], expected_actions[:1]):
        raise MaterializationError(f"episode {mapping.target_episode_index} first action is not bit-exact")

    length = mapping.length
    expected_timestamp = np.arange(length, dtype=np.float32) / np.float32(TARGET_FPS)
    timestamp = _scalar_column(table, "timestamp", np.float32)
    if not _float32_bits_equal(timestamp, expected_timestamp):
        raise MaterializationError(f"episode {mapping.target_episode_index} timestamp mismatch")
    expected_scalars = {
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, mapping.target_episode_index, dtype=np.int64),
        "index": np.arange(mapping.target_from_index, mapping.target_to_index, dtype=np.int64),
        "task_index": np.full(length, mapping.target_task_index, dtype=np.int64),
    }
    for name, expected in expected_scalars.items():
        actual = _scalar_column(table, name, np.int64)
        if not np.array_equal(actual, expected):
            raise MaterializationError(f"episode {mapping.target_episode_index} {name} mismatch")


def _publish_generated_episode(
    target: Path,
    mapping: EpisodeMapping,
    donor_store: _DonorStore,
    schema: Any,
) -> None:
    _, pq = _require_pyarrow()
    if target.exists():
        table = pq.read_table(target)
        if not table.schema.equals(schema, check_metadata=True):
            raise MaterializationError(f"orphan generated schema mismatch: {target}")
        _validate_episode_table(table, mapping, donor_store)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise MaterializationError(f"stale temporary artifact exists: {temporary}")
    table = _build_target_table(mapping, donor_store, schema)
    try:
        pq.write_table(table, temporary)
        _fsync_file(temporary)
        written = pq.read_table(temporary)
        if not written.schema.equals(schema, check_metadata=True):
            raise MaterializationError(f"generated schema mismatch: {temporary}")
        _validate_episode_table(written, mapping, donor_store)
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = pq.read_table(target)
            if not existing.schema.equals(schema, check_metadata=True):
                raise MaterializationError(f"concurrent generated artifact differs: {target}")
            _validate_episode_table(existing, mapping, donor_store)
        _fsync_dir(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_target_table(mapping: EpisodeMapping, donor_store: _DonorStore, schema: Any) -> Any:
    pa, _ = _require_pyarrow()
    states, actions = donor_store.episode(mapping)
    length = mapping.length
    arrays = [
        pa.FixedSizeListArray.from_arrays(pa.array(states.reshape(-1), type=pa.float32()), 8),
        pa.FixedSizeListArray.from_arrays(pa.array(actions.reshape(-1), type=pa.float32()), 7),
        pa.array(
            np.arange(length, dtype=np.float32) / np.float32(TARGET_FPS),
            type=pa.float32(),
        ),
        pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
        pa.array(
            np.full(length, mapping.target_episode_index, dtype=np.int64),
            type=pa.int64(),
        ),
        pa.array(
            np.arange(
                mapping.target_from_index,
                mapping.target_to_index,
                dtype=np.int64,
            ),
            type=pa.int64(),
        ),
        pa.array(
            np.full(length, mapping.target_task_index, dtype=np.int64),
            type=pa.int64(),
        ),
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _validate_episode_journal(
    plan: CalvinABCMaterializationPlan,
    staging: Path,
    mapping: EpisodeMapping,
    journal_path: Path,
    artifact_by_path: Mapping[str, SourceArtifact],
    donor_store: _DonorStore,
    schema: Any,
) -> None:
    _, pq = _require_pyarrow()
    journal = _read_json(journal_path)
    if journal.get("plan_sha256") != plan.sha256:
        raise MaterializationError(f"journal belongs to another run: {journal_path}")
    if journal.get("mapping") != mapping.to_dict():
        raise MaterializationError(f"journal mapping differs: {journal_path}")
    data = journal.get("data")
    if not isinstance(data, Mapping):
        raise MaterializationError(f"journal data record is missing: {journal_path}")
    data_path = staging / str(data.get("path"))
    if data.get("mode") == "hardlink":
        relative = str(data["path"])
        _link_or_validate(plan.collision_root / relative, data_path)
        expected_sha = artifact_by_path[relative].sha256
    elif data.get("mode") == "generated":
        table = pq.read_table(data_path)
        if not table.schema.equals(schema, check_metadata=True):
            raise MaterializationError(f"journaled generated schema differs: {data_path}")
        _validate_episode_table(table, mapping, donor_store)
        expected_sha = _file_sha256(data_path)
    else:
        raise MaterializationError(f"journal has invalid data mode: {journal_path}")
    if data.get("sha256") != expected_sha:
        raise MaterializationError(f"journaled data digest differs: {data_path}")
    if data.get("size_bytes") != data_path.stat().st_size:
        raise MaterializationError(f"journaled data size differs: {data_path}")
    if data.get("rows") != mapping.length:
        raise MaterializationError(f"journaled row count differs: {data_path}")

    expected_videos = []
    for view in TARGET_VIEWS:
        relative = _target_video_path(mapping.target_episode_index, view, plan.contract.chunks_size)
        _link_or_validate(plan.collision_root / relative, staging / relative)
        expected_videos.append(
            {
                "path": relative,
                "sha256": artifact_by_path[relative].sha256,
                "mode": "hardlink",
            }
        )
    if journal.get("videos") != expected_videos:
        raise MaterializationError(f"journaled videos differ: {journal_path}")


def _validate_full_root(
    plan: CalvinABCMaterializationPlan,
    root: Path,
    donor_store: _DonorStore,
    schema: Any,
    *,
    decode_samples: bool,
    progress: _PROGRESS | None,
) -> dict[str, Any]:
    _, pq = _require_pyarrow()
    expected_data = {
        _target_data_path(index, plan.contract.chunks_size) for index in range(plan.contract.target_episodes)
    }
    expected_videos = {
        _target_video_path(index, view, plan.contract.chunks_size)
        for index in range(plan.contract.target_episodes)
        for view in TARGET_VIEWS
    }
    physical_data = {path.relative_to(root).as_posix() for path in (root / "data").rglob("*.parquet") if path.is_file()}
    physical_videos = {path.relative_to(root).as_posix() for path in (root / "videos").rglob("*.mp4") if path.is_file()}
    stale_temps = [path for path in root.rglob("*") if ".tmp-" in path.name]
    if stale_temps:
        raise MaterializationError(f"final dataset contains temporary artifacts: {stale_temps[:10]}")
    if physical_data != expected_data:
        raise MaterializationError("final Parquet file set is not exact")
    if physical_videos != expected_videos:
        raise MaterializationError("final video file set is not exact")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise MaterializationError("final dataset contains a symlink")

    frames = 0
    for index, mapping in enumerate(plan.mappings):
        table = pq.read_table(root / _target_data_path(mapping.target_episode_index, plan.contract.chunks_size))
        if not table.schema.equals(schema, check_metadata=True):
            raise MaterializationError(f"final schema mismatch for episode {mapping.target_episode_index}")
        _validate_episode_table(table, mapping, donor_store)
        frames += table.num_rows
        for view in TARGET_VIEWS:
            relative = _target_video_path(mapping.target_episode_index, view, plan.contract.chunks_size)
            if not os.path.samefile(plan.collision_root / relative, root / relative):
                raise MaterializationError(f"final video is not the verified source hardlink: {relative}")
        if progress is not None and ((index + 1) % 2_000 == 0 or index + 1 == len(plan.mappings)):
            progress(f"final numeric validation: {index + 1}/{len(plan.mappings)}")
    if frames != plan.contract.target_frames:
        raise MaterializationError(f"final frame count is {frames}, expected {plan.contract.target_frames}")

    info = _read_json(root / "meta" / "info.json")
    if info.get("total_episodes") != plan.contract.target_episodes:
        raise MaterializationError("final info episode count mismatch")
    if info.get("total_frames") != plan.contract.target_frames:
        raise MaterializationError("final info frame count mismatch")
    if len(_read_jsonl(root / "meta" / "episodes.jsonl")) != (plan.contract.target_episodes):
        raise MaterializationError("final episodes metadata count mismatch")

    decoded: list[dict[str, Any]] = []
    if decode_samples:
        from experiments.calvin.data import CALVIN_DATA_SPEC
        from prism.data.lerobot import LeRobotDataset

        sample_ids = tuple(
            dict.fromkeys(
                (
                    0,
                    plan.contract.target_episodes // 2,
                    plan.contract.target_episodes - 1,
                )
            )
        )
        with LeRobotDataset(root, CALVIN_DATA_SPEC, verify_files=True) as dataset:
            for episode_id in sample_ids:
                numeric = dataset.read_numeric_episode(episode_id)
                if numeric.states.shape[0] != plan.mappings[episode_id].length:
                    raise MaterializationError(f"LeRobot numeric decode length mismatch: {episode_id}")
                images = dataset.read_images(episode_id, [0])
                decoded.append(
                    {
                        "episode_index": episode_id,
                        "numeric_rows": int(numeric.states.shape[0]),
                        "primary_shape": list(images["primary"].shape),
                        "wrist_shape": list(images["wrist"].shape),
                    }
                )

    return {
        "schema_version": "prism-calvin-abc-v21-validation-v1",
        "plan_sha256": plan.sha256,
        "mapping_sha256": plan.mapping_sha256,
        "total_episodes": len(plan.mappings),
        "total_frames": frames,
        "total_parquets": len(physical_data),
        "total_videos": len(physical_videos),
        "decoded_samples": decoded,
    }


def _validate_completed_run(
    plan: CalvinABCMaterializationPlan,
    output: Path,
    *,
    decode_samples: bool,
    progress: _PROGRESS | None,
) -> None:
    if output.is_symlink() or not output.is_dir():
        raise MaterializationError(f"completed output is not a real directory: {output}")
    run = _read_json(output / ".materialization" / "run.json")
    if run != _run_spec(plan):
        raise MaterializationError(f"completed output belongs to another plan: {output}")
    donor_store = _DonorStore(plan)
    _, pq = _require_pyarrow()
    schema = pq.read_schema(output / _target_data_path(plan.present_episode_ids[0], plan.contract.chunks_size))
    _validate_full_root(
        plan,
        output,
        donor_store,
        schema,
        decode_samples=decode_samples,
        progress=progress,
    )


def _run_spec(plan: CalvinABCMaterializationPlan) -> dict[str, Any]:
    content = {
        "schema_version": "prism-calvin-abc-v21-run-v1",
        "plan_sha256": plan.sha256,
        "mapping_sha256": plan.mapping_sha256,
        "target_revision": COLLISION_REVISION,
        "donor_revision": TRALY_REVISION,
    }
    content["run_sha256"] = _json_sha256(content)
    return content


def _validate_output_location(plan: CalvinABCMaterializationPlan, output: Path) -> None:
    for source in (plan.collision_root, plan.donor_root):
        if output == source or _is_relative_to(output, source):
            raise MaterializationError(f"output must not be inside an input root: {output}")
    if output.parent.stat().st_dev != plan.collision_root.stat().st_dev:
        raise MaterializationError("output and Collision root must share a filesystem for hardlinks")


def _validate_target_schema(schema: Any) -> None:
    pa, _ = _require_pyarrow()
    if schema.names != list(PARQUET_COLUMNS):
        raise MaterializationError("Collision Parquet column order is unexpected")
    expected_types = (
        pa.list_(pa.float32(), 8),
        pa.list_(pa.float32(), 7),
        pa.float32(),
        pa.int64(),
        pa.int64(),
        pa.int64(),
        pa.int64(),
    )
    if tuple(field.type for field in schema) != expected_types:
        raise MaterializationError("Collision Parquet field types are unexpected")
    if b"huggingface" not in (schema.metadata or {}):
        raise MaterializationError("Collision Parquet lacks Hugging Face schema metadata")


def _list_column_to_numpy(column: Any, width: int, *, pa: Any) -> np.ndarray:
    array = column.combine_chunks()
    if array.null_count:
        raise MaterializationError("numeric list column contains nulls")
    if pa.types.is_fixed_size_list(array.type):
        if array.type.list_size != width:
            raise MaterializationError(f"fixed list width is {array.type.list_size}, expected {width}")
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        if not np.all(np.diff(offsets) == width):
            raise MaterializationError(f"list column does not have width {width}")
    else:
        raise MaterializationError(f"expected list column, got {array.type}")
    values = array.flatten().to_numpy(zero_copy_only=False)
    if values.dtype != np.float32:
        values = values.astype(np.float32, copy=False)
    return np.ascontiguousarray(values.reshape(-1, width))


def _scalar_column(table: Any, name: str, dtype: Any) -> np.ndarray:
    column = table[name].combine_chunks()
    if column.null_count:
        raise MaterializationError(f"scalar column {name!r} contains nulls")
    values = column.to_numpy(zero_copy_only=False)
    if values.dtype != np.dtype(dtype):
        raise MaterializationError(f"scalar column {name!r} has dtype {values.dtype}, expected {np.dtype(dtype)}")
    return np.ascontiguousarray(values)


def _float32_bits_equal(left: Any, right: Any) -> bool:
    a = np.ascontiguousarray(left, dtype=np.float32)
    b = np.ascontiguousarray(right, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def _target_data_path(episode_index: int, chunks_size: int) -> str:
    return f"data/chunk-{episode_index // chunks_size:03d}/episode_{episode_index:06d}.parquet"


def _target_video_path(episode_index: int, view: str, chunks_size: int) -> str:
    return f"videos/chunk-{episode_index // chunks_size:03d}/{view}/episode_{episode_index:06d}.mp4"


def _link_or_validate(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise MaterializationError(f"hardlink source is missing or a symlink: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        _fsync_dir(target.parent)
    except FileExistsError:
        if target.is_symlink() or not target.is_file():
            raise MaterializationError(f"hardlink target is invalid: {target}")
        if not os.path.samefile(source, target):
            raise MaterializationError(f"existing target is not the source hardlink: {target}")


def _write_json_idempotent(path: Path, value: Any) -> None:
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_bytes_idempotent(path, data)


def _write_jsonl_idempotent(path: Path, rows: Sequence[Mapping[str, Any]] | Any) -> None:
    data = b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for row in rows)
    _write_bytes_idempotent(path, data)


def _write_bytes_idempotent(path: Path, data: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise MaterializationError(f"existing run artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise MaterializationError(f"stale temporary run artifact exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise MaterializationError(f"concurrent run artifact differs: {path}")
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cleanup_stale_temps(staging: Path) -> None:
    for path in staging.rglob("*"):
        if ".tmp-" not in path.name:
            continue
        if path.is_symlink() or not path.is_file():
            raise MaterializationError(f"invalid stale temporary artifact: {path}")
        path.unlink()
    for directory, _, _ in os.walk(staging, topdown=False):
        candidate = Path(directory)
        if candidate != staging and not any(candidate.iterdir()):
            candidate.rmdir()


def _rename_dir_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MaterializationError("renameat2 is required for no-overwrite finalization")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"refusing to overwrite completed dataset: {target}")
        raise OSError(error, os.strerror(error), str(target))


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_metadata(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=False):
        _fsync_dir(Path(directory))


def _file_digests(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            sha256.update(block)
            git_blob.update(block)
    return sha256.hexdigest(), git_blob.hexdigest()


def _jsonl_sha256(rows: Any) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _require_contiguous_ids(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> None:
    values = [row.get(key) for row in rows]
    if values != list(range(len(rows))):
        raise MaterializationError(f"{label} IDs are not contiguous and ordered")


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for CALVIN materialization") from exc
    return pa, pq


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _emit(progress: _PROGRESS | None, message: str) -> None:
    if progress is not None:
        progress(message)


__all__ = [
    "CALVIN_ABC_CONTRACT",
    "COLLISION_REPOSITORY",
    "COLLISION_REVISION",
    "TRALY_REPOSITORY",
    "TRALY_REVISION",
    "CalvinABCContract",
    "CalvinABCMaterializationPlan",
    "EpisodeMapping",
    "ExistingGateReport",
    "SourceArtifact",
    "build_calvin_abc_v21_plan",
    "materialize_calvin_abc_v21",
]
