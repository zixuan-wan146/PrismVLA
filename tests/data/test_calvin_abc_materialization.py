from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import prism.data.materialization.calvin_abc_v21 as calvin_v21
from prism.data.materialization.calvin_abc_v21 import CalvinABCContract
from prism.data.materialization.calvin_abc_v21 import build_calvin_abc_v21_plan
from prism.data.materialization.calvin_abc_v21 import materialize_calvin_abc_v21
from prism.data.materialization.libero_v21 import MaterializationError


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@dataclass(frozen=True)
class _SyntheticSources:
    collision_root: Path
    donor_root: Path
    contract: CalvinABCContract
    target_states: tuple[np.ndarray, ...]
    relative_actions: tuple[np.ndarray, ...]
    absolute_actions: tuple[np.ndarray, ...]
    source_hashes: dict[Path, str]


def test_materializes_permuted_traly_relative_actions_and_preserves_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _write_synthetic_sources(tmp_path, monkeypatch)
    plan = build_calvin_abc_v21_plan(
        sources.collision_root,
        sources.donor_root,
        contract=sources.contract,
        hash_workers=1,
    )

    assert [(mapping.target_episode_index, mapping.donor_episode_index) for mapping in plan.mappings] == [
        (0, 1),
        (1, 0),
    ]

    output = materialize_calvin_abc_v21(
        plan,
        tmp_path / "complete_calvin_abc",
        decode_samples=False,
    )

    present_relative = "data/chunk-000/episode_000000.parquet"
    missing_relative = "data/chunk-000/episode_000001.parquet"
    assert os.path.samefile(
        sources.collision_root / present_relative,
        output / present_relative,
    )
    assert not (sources.collision_root / missing_relative).exists()
    assert (output / missing_relative).is_file()

    generated = pq.read_table(output / missing_relative)
    generated_state = _list_column(generated, "state", 8)
    generated_actions = _list_column(generated, "actions", 7)
    np.testing.assert_array_equal(generated_state, sources.target_states[1])
    np.testing.assert_array_equal(generated_actions, sources.relative_actions[1])
    np.testing.assert_array_equal(generated_actions[0], sources.relative_actions[1][0])
    assert not np.array_equal(generated_actions, sources.absolute_actions[1])
    np.testing.assert_array_equal(
        generated["timestamp"].combine_chunks().to_numpy(),
        np.array([0.0, 0.1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        generated["frame_index"].combine_chunks().to_numpy(),
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        generated["episode_index"].combine_chunks().to_numpy(),
        np.array([1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        generated["index"].combine_chunks().to_numpy(),
        np.array([2, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        generated["task_index"].combine_chunks().to_numpy(),
        np.array([0, 0], dtype=np.int64),
    )

    for episode_index in range(2):
        for view in calvin_v21.TARGET_VIEWS:
            relative = f"videos/chunk-000/{view}/episode_{episode_index:06d}.mp4"
            assert os.path.samefile(
                sources.collision_root / relative,
                output / relative,
            )
    for relative in (
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
    ):
        assert os.path.samefile(
            sources.collision_root / relative,
            output / relative,
        )

    present_journal = _read_json(output / ".materialization/journal/episode_000000.json")
    generated_journal = _read_json(output / ".materialization/journal/episode_000001.json")
    assert present_journal["data"]["mode"] == "hardlink"
    assert generated_journal["data"]["mode"] == "generated"

    assert {path: _sha256(path) for path in sources.source_hashes} == sources.source_hashes


def test_ambiguous_stats_mapping_fails_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _write_synthetic_sources(tmp_path, monkeypatch, ambiguous=True)
    output = tmp_path / "must_not_exist"

    with pytest.raises(MaterializationError, match="mapping is not total and unique"):
        build_calvin_abc_v21_plan(
            sources.collision_root,
            sources.donor_root,
            contract=sources.contract,
            hash_workers=1,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".must_not_exist.calvin-v21.partial-*"))


def test_no_resume_rejects_matching_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _write_synthetic_sources(tmp_path, monkeypatch)
    plan = build_calvin_abc_v21_plan(
        sources.collision_root,
        sources.donor_root,
        contract=sources.contract,
        hash_workers=1,
    )
    output = tmp_path / "complete_calvin_abc"
    partial = tmp_path / (f".complete_calvin_abc.calvin-v21.partial-{plan.sha256[:16]}")
    partial.mkdir()
    sentinel = partial / "sentinel"
    sentinel.write_text("do not touch", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to resume partial"):
        materialize_calvin_abc_v21(plan, output, resume=False, decode_samples=False)

    assert sentinel.read_text(encoding="utf-8") == "do not touch"
    assert not output.exists()


def _write_synthetic_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ambiguous: bool = False,
) -> _SyntheticSources:
    collision_root = tmp_path / "collision"
    donor_root = tmp_path / "traly"
    donor_data_path = "data/chunk-000/file-000.parquet"
    donor_required_paths = (
        "meta/info.json",
        "meta/tasks.parquet",
        calvin_v21.TRALY_EPISODES_PATH,
        donor_data_path,
    )
    monkeypatch.setattr(calvin_v21, "TRALY_DATA_PATHS", (donor_data_path,))
    monkeypatch.setattr(calvin_v21, "TRALY_REQUIRED_PATHS", donor_required_paths)

    contract = CalvinABCContract(
        target_episodes=2,
        target_frames=4,
        target_tasks=1,
        target_videos=4,
        target_present_parquets=1,
        donor_episodes=2,
        donor_frames=4,
        donor_tasks=1,
        chunks_size=1_000,
    )
    donor_states_by_target = (
        np.array(
            [
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.070, 1, 2, 3, 4, 5, 6, 7, 1],
                [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.080, 2, 3, 4, 5, 6, 7, 8, -1],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [-0.30, 0.70, 0.20, -0.40, 0.90, -0.60, 0.050, 8, 7, 6, 5, 4, 3, 2, -1],
                [-0.25, 0.75, 0.25, -0.35, 0.95, -0.55, 0.060, 7, 6, 5, 4, 3, 2, 1, 1],
            ],
            dtype=np.float32,
        ),
    )
    relative_actions = (
        np.array(
            [
                [0.25, -0.20, 0.15, -0.10, 0.05, -0.01, 1.0],
                [0.20, -0.15, 0.10, -0.05, 0.01, -0.02, -1.0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [-0.50, 0.40, -0.30, 0.20, -0.10, 0.05, -1.0],
                [-0.45, 0.35, -0.25, 0.15, -0.05, 0.01, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    absolute_actions = tuple(_absolute_actions(value) for value in relative_actions)
    target_states = tuple(_target_state(value) for value in donor_states_by_target)

    _write_collision_metadata(
        collision_root,
        contract,
        target_states,
        relative_actions,
    )
    _write_target_parquet(
        collision_root / "data/chunk-000/episode_000000.parquet",
        target_states[0],
        relative_actions[0],
        episode_index=0,
        global_from=0,
    )
    for episode_index in range(2):
        for view in calvin_v21.TARGET_VIEWS:
            path = collision_root / f"videos/chunk-000/{view}/episode_{episode_index:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic-{episode_index}-{view}".encode("ascii"))
    collision_paths = (
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "data/chunk-000/episode_000000.parquet",
        *(
            f"videos/chunk-000/{view}/episode_{episode_index:06d}.mp4"
            for episode_index in range(2)
            for view in calvin_v21.TARGET_VIEWS
        ),
    )
    _write_collision_tree(collision_root, collision_paths)

    _write_json(
        donor_root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "total_episodes": 2,
            "total_frames": 4,
            "total_tasks": 1,
        },
    )
    _write_parquet(
        donor_root / "meta/tasks.parquet",
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "__index_level_0__": pa.array(["shared task"], type=pa.string()),
            }
        ),
    )

    donor_order = (1, 0)
    donor_episode_rows = []
    donor_cursor = 0
    for donor_episode_index, target_episode_index in enumerate(donor_order):
        stats_target_index = 1 if ambiguous else target_episode_index
        donor_episode_rows.append(
            {
                "episode_index": donor_episode_index,
                "tasks": ["shared task"],
                "length": 2,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": donor_cursor,
                "dataset_to_index": donor_cursor + 2,
                **_flatten_stats(
                    "stats/observation.state",
                    donor_states_by_target[stats_target_index],
                ),
                **_flatten_stats(
                    "stats/action.relative",
                    relative_actions[stats_target_index],
                ),
            }
        )
        donor_cursor += 2
    _write_parquet(
        donor_root / calvin_v21.TRALY_EPISODES_PATH,
        pa.Table.from_pylist(donor_episode_rows),
    )

    donor_states = np.concatenate([donor_states_by_target[index] for index in donor_order], axis=0)
    donor_relative = np.concatenate([relative_actions[index] for index in donor_order], axis=0)
    donor_absolute = np.concatenate([absolute_actions[index] for index in donor_order], axis=0)
    donor_table = pa.table(
        {
            "observation.state": pa.array(donor_states.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(donor_absolute.tolist(), type=pa.list_(pa.float32())),
            "action.absolute": pa.array(donor_absolute.tolist(), type=pa.list_(pa.float32())),
            "action.relative": pa.array(donor_relative.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array([0.0, 1.0 / 30.0, 0.0, 1.0 / 30.0], type=pa.float32()),
            "frame_index": pa.array([0, 1, 0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
            "index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "task_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    _write_parquet(donor_root / donor_data_path, donor_table)
    _write_donor_metadata_files(donor_root, donor_required_paths)

    source_paths = [collision_root / path for path in collision_paths]
    source_paths.extend(donor_root / path for path in donor_required_paths)
    source_hashes = {path: _sha256(path) for path in source_paths}
    return _SyntheticSources(
        collision_root=collision_root,
        donor_root=donor_root,
        contract=contract,
        target_states=target_states,
        relative_actions=relative_actions,
        absolute_actions=absolute_actions,
        source_hashes=source_hashes,
    )


def _write_collision_metadata(
    root: Path,
    contract: CalvinABCContract,
    target_states: tuple[np.ndarray, ...],
    relative_actions: tuple[np.ndarray, ...],
) -> None:
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "panda",
            "total_episodes": contract.target_episodes,
            "total_frames": contract.target_frames,
            "total_tasks": contract.target_tasks,
            "total_videos": contract.target_videos,
            "total_chunks": 1,
            "chunks_size": contract.chunks_size,
            "fps": 10,
            "splits": {"train": "0:2"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "state": {"dtype": "float32", "shape": [8]},
                "actions": {"dtype": "float32", "shape": [7]},
                "image": {"dtype": "video", "shape": [16, 16, 3]},
                "wrist_image": {"dtype": "video", "shape": [16, 16, 3]},
            },
        },
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "shared task"}])
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [{"episode_index": index, "tasks": ["shared task"], "length": 2} for index in range(2)],
    )
    stats_rows = []
    for episode_index in range(2):
        length = 2
        frame_index = np.arange(length, dtype=np.int64)
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "state": _array_stats(target_states[episode_index]),
                    "actions": _array_stats(relative_actions[episode_index]),
                    "timestamp": _array_stats((frame_index.astype(np.float32) / np.float32(10))[:, None]),
                    "frame_index": _array_stats(frame_index[:, None]),
                    "episode_index": _array_stats(np.full((length, 1), episode_index, dtype=np.int64)),
                    "index": _array_stats(np.arange(episode_index * 2, episode_index * 2 + 2, dtype=np.int64)[:, None]),
                    "task_index": _array_stats(np.zeros((length, 1), dtype=np.int64)),
                },
            }
        )
    _write_jsonl(root / "meta/episodes_stats.jsonl", stats_rows)


def _write_target_parquet(
    path: Path,
    state: np.ndarray,
    actions: np.ndarray,
    *,
    episode_index: int,
    global_from: int,
) -> None:
    length = len(state)
    schema = pa.schema(
        [
            pa.field("state", pa.list_(pa.float32(), 8)),
            pa.field("actions", pa.list_(pa.float32(), 7)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ],
        metadata={b"huggingface": b'{"synthetic":true}'},
    )
    arrays = [
        pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), type=pa.float32()), 8),
        pa.FixedSizeListArray.from_arrays(pa.array(actions.reshape(-1), type=pa.float32()), 7),
        pa.array(
            np.arange(length, dtype=np.float32) / np.float32(10),
            type=pa.float32(),
        ),
        pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(
            np.arange(global_from, global_from + length, dtype=np.int64),
            type=pa.int64(),
        ),
        pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
    ]
    _write_parquet(path, pa.Table.from_arrays(arrays, schema=schema))


def _write_collision_tree(root: Path, relative_paths: tuple[str, ...]) -> None:
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        payload = path.read_bytes()
        entry: dict[str, Any] = {"size": len(payload)}
        if relative.startswith(("data/", "videos/")):
            digest = hashlib.sha256(payload).hexdigest()
            entry.update({"lfs_sha256": digest, "lfs_size": len(payload)})
        else:
            entry["blob_id"] = _git_blob(payload)
        files[relative] = entry
    _write_json(
        root / ".cache/huggingface/trees" / f"{calvin_v21.COLLISION_REVISION}.json",
        {"files": files},
    )


def _write_donor_metadata_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative in relative_paths:
        payload = (root / relative).read_bytes()
        identity = hashlib.sha256(payload).hexdigest() if relative.endswith(".parquet") else _git_blob(payload)
        path = root / ".cache/huggingface/download" / f"{relative}.metadata"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"{calvin_v21.TRALY_REVISION}\n{identity}\n",
            encoding="utf-8",
        )


def _flatten_stats(prefix: str, values: np.ndarray) -> dict[str, list[Any]]:
    stats = _array_stats(values)
    return {
        f"{prefix}/min": stats["min"],
        f"{prefix}/max": stats["max"],
        f"{prefix}/mean": stats["mean"],
        f"{prefix}/count": stats["count"],
    }


def _array_stats(values: np.ndarray) -> dict[str, list[Any]]:
    array = np.asarray(values)
    return {
        "min": _native_list(np.min(array, axis=0)),
        "max": _native_list(np.max(array, axis=0)),
        "mean": _native_list(np.mean(array, axis=0)),
        "std": _native_list(np.std(array, axis=0)),
        "count": [int(array.shape[0])],
    }


def _native_list(values: np.ndarray) -> list[Any]:
    return [value.item() for value in np.asarray(values).reshape(-1)]


def _target_state(donor_state: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate(
            [
                donor_state[:, :6],
                np.zeros((len(donor_state), 1), dtype=np.float32),
                donor_state[:, 6:7],
            ],
            axis=1,
        )
    )


def _absolute_actions(relative: np.ndarray) -> np.ndarray:
    absolute = relative.copy()
    absolute[:, :6] += np.float32(50.0)
    return absolute


def _list_column(table: Any, name: str, width: int) -> np.ndarray:
    array = table[name].combine_chunks()
    return np.asarray(array.values).reshape(-1, width)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, table: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
