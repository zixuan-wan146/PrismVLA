from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.libero.data import LIBERO_DATA_SPEC
import prism.data.materialization.libero_v21 as libero_v21
from prism.data.lerobot import LeRobotDataset
from prism.data.materialization.libero_v21 import MaterializationError
from prism.data.materialization.libero_v21 import VideoEncodingConfig
from prism.data.materialization.libero_v21 import build_libero_v21_plan
from prism.data.materialization.libero_v21 import materialize_libero_v21_plan


av = pytest.importorskip("av")
h5py = pytest.importorskip("h5py")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


TEST_ENCODING = VideoEncodingConfig(
    codec="libx264",
    pixel_format="yuv420p",
    crf=35,
    gop=2,
)


def test_materializes_v21_schema_indices_actions_and_two_video_views(tmp_path: Path):
    source_root = _write_libero_source(tmp_path)
    plan = build_libero_v21_plan(
        source_root,
        suite="libero_spatial",
        image_transform="none",
    )
    repeated_plan = build_libero_v21_plan(
        source_root,
        suite="libero_spatial",
        image_transform="none",
    )

    assert plan.sha256 == repeated_plan.sha256
    assert [episode.episode_index for episode in plan.episodes] == [0, 1]
    assert [(episode.global_index_from, episode.global_index_to) for episode in plan.episodes] == [(0, 3), (3, 7)]
    assert plan.total_frames == 7

    output = materialize_libero_v21_plan(
        plan,
        tmp_path / "output" / "libero_spatial",
        video_encoding=TEST_ENCODING,
    )

    info = _read_json(output / "meta" / "info.json")
    assert info["codebase_version"] == "v2.1"
    assert info["robot_type"] == "franka"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 7
    assert info["total_videos"] == 4
    assert set(info["features"]) == {
        "observation.state",
        "action",
        "observation.images.image",
        "observation.images.wrist_image",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    image_feature = info["features"]["observation.images.image"]
    assert image_feature["names"] == ["height", "width", "channel"]
    assert image_feature["info"]["video.height"] == 128
    assert image_feature["info"]["video.width"] == 128
    assert image_feature["info"]["video.channels"] == 3
    assert image_feature["info"]["video.fps"] == 20.0
    assert "video_info" not in image_feature

    first = pq.read_table(output / "data" / "chunk-000" / "episode_000000.parquet")
    second = pq.read_table(output / "data" / "chunk-000" / "episode_000001.parquet")
    assert first.column_names == list(libero_v21.PARQUET_COLUMNS)
    assert first.schema.field("observation.state").type == pa.list_(pa.float32(), 8)
    assert first.schema.field("action").type == pa.list_(pa.float32(), 7)
    assert b"huggingface" in (first.schema.metadata or {})
    assert first["index"].to_pylist() == [0, 1, 2]
    assert second["index"].to_pylist() == [3, 4, 5, 6]
    assert first["frame_index"].to_pylist() == [0, 1, 2]
    assert first["timestamp"].to_pylist() == pytest.approx([0.0, 0.05, 0.1])
    assert [row[-1] for row in first["action"].to_pylist()] == [1.0, 0.0, 1.0]
    assert first["action"].to_pylist()[0][0:6] == [0.0] * 6
    assert first.num_rows + second.num_rows == 7

    for episode_index, expected_frames in ((0, 3), (1, 4)):
        for feature_key in (
            "observation.images.image",
            "observation.images.wrist_image",
        ):
            video_path = output / "videos" / "chunk-000" / feature_key / f"episode_{episode_index:06d}.mp4"
            assert _decode_video(video_path) == (
                expected_frames,
                libero_v21.IMAGE_WIDTH,
                libero_v21.IMAGE_HEIGHT,
            )

    episodes = _read_jsonl(output / "meta" / "episodes.jsonl")
    episode_stats = _read_jsonl(output / "meta" / "episodes_stats.jsonl")
    tasks = _read_jsonl(output / "meta" / "tasks.jsonl")
    provenance = _read_json(output / "meta" / "materialization.json")
    assert episodes == [
        {"episode_index": 0, "length": 3, "tasks": ["pick up the red block"]},
        {"episode_index": 1, "length": 4, "tasks": ["pick up the red block"]},
    ]
    assert [row["episode_index"] for row in episode_stats] == [0, 1]
    first_stats = episode_stats[0]["stats"]
    assert np.asarray(first_stats["observation.state"]["min"]).shape == (8,)
    assert np.asarray(first_stats["observation.images.image"]["min"]).shape == (3, 1, 1)
    assert np.asarray(first_stats["timestamp"]["min"]).shape == (1,)
    assert tasks == [{"task": "pick up the red block", "task_index": 0}]
    assert provenance["plan_sha256"] == plan.sha256
    assert provenance["source_episodes"][0]["source_file"] == "libero_spatial/task_00_demo.hdf5"
    assert provenance["source_episodes"][0]["demo_key"] == "demo_0"
    assert provenance["image_transform"] == "none"
    assert provenance["video_encoding"]["codec"] == "libx264"
    assert len(provenance["artifacts"]) == 6

    with LeRobotDataset(output, LIBERO_DATA_SPEC) as dataset:
        frame = dataset.read_numeric_frame(0, 0)
        assert frame.state.shape == (8,)
        assert frame.action.shape == (7,)
        assert frame.action[-1] == 1.0
        assert dataset.read_instruction(0, 0) == "pick up the red block"
        assert dataset.read_actions(1, 0, 4).shape == (4, 7)
        images = dataset.read_images(1, [3, 1])
        assert images["primary"].shape == (2, 128, 128, 3)
        assert images["wrist"].shape == (2, 128, 128, 3)


def test_resume_uses_completed_journal_and_never_overwrites_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_root = _write_libero_source(tmp_path)
    plan = build_libero_v21_plan(
        source_root,
        suite="libero_spatial",
        image_transform="none",
    )
    output = tmp_path / "output" / "libero_spatial"
    original = libero_v21._materialize_episode

    def fail_on_second_episode(*args, **kwargs):
        episode = kwargs["episode"]
        if episode.episode_index == 1:
            raise RuntimeError("injected interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(libero_v21, "_materialize_episode", fail_on_second_episode)
    with pytest.raises(RuntimeError, match="injected interruption"):
        materialize_libero_v21_plan(
            plan,
            output,
            video_encoding=TEST_ENCODING,
        )

    partials = list(output.parent.glob(".libero_spatial.lerobot-v2.1.partial-*"))
    assert len(partials) == 1
    assert not output.exists()
    journal_paths = sorted((partials[0] / ".materialization" / "journal").glob("*.json"))
    assert [path.name for path in journal_paths] == ["episode_000000.json"]

    monkeypatch.setattr(libero_v21, "_materialize_episode", original)
    completed = materialize_libero_v21_plan(
        plan,
        output,
        video_encoding=TEST_ENCODING,
        resume=True,
    )
    assert completed == output.resolve()
    assert not partials[0].exists()
    assert len(_read_jsonl(output / "meta" / "episodes.jsonl")) == 2

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_libero_v21_plan(
            plan,
            output,
            video_encoding=TEST_ENCODING,
            resume=False,
        )


@pytest.mark.parametrize(
    ("source_option", "message"),
    [
        ({"invalid_gripper": True}, "exactly -1 or \\+1"),
        ({"invalid_wrist_shape": True}, "must have shape"),
    ],
)
def test_invalid_source_fails_without_fallback(
    tmp_path: Path,
    source_option: dict[str, bool],
    message: str,
):
    source_root = _write_libero_source(tmp_path, **source_option)

    with pytest.raises(MaterializationError, match=message):
        build_libero_v21_plan(
            source_root,
            suite="libero_spatial",
            image_transform="none",
        )
    assert not (tmp_path / "output").exists()


def test_image_transform_is_explicit_and_supports_only_none_or_rotate_180():
    image = np.zeros(
        (libero_v21.IMAGE_HEIGHT, libero_v21.IMAGE_WIDTH, 3),
        dtype=np.uint8,
    )
    image[0, 0] = [1, 2, 3]
    image[-1, -1] = [4, 5, 6]

    unchanged = libero_v21._apply_image_transform(image, "none")
    rotated = libero_v21._apply_image_transform(image, "rotate_180")

    np.testing.assert_array_equal(unchanged, image)
    np.testing.assert_array_equal(rotated[0, 0], [4, 5, 6])
    np.testing.assert_array_equal(rotated[-1, -1], [1, 2, 3])
    with pytest.raises(ValueError, match="image_transform"):
        libero_v21._apply_image_transform(image, "implicit")


def _write_libero_source(
    tmp_path: Path,
    *,
    invalid_gripper: bool = False,
    invalid_wrist_shape: bool = False,
) -> Path:
    source_root = tmp_path / "raw"
    suite_dir = source_root / "libero_spatial"
    suite_dir.mkdir(parents=True)
    path = suite_dir / "task_00_demo.hdf5"

    lengths = (3, 4)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {
                "env_kwargs": {
                    "control_freq": 20,
                    "controller_configs": {
                        "type": "OSC_POSE",
                        "control_delta": True,
                    },
                }
            }
        )
        data.attrs["problem_info"] = json.dumps({"language_instruction": "pick up the red block"})
        data.attrs["num_demos"] = len(lengths)
        data.attrs["total"] = sum(lengths)

        for demo_index, length in enumerate(lengths):
            demo = data.create_group(f"demo_{demo_index}")
            actions = np.zeros((length, 7), dtype=np.float64)
            actions[:, :6] = np.arange(length * 6, dtype=np.float64).reshape(length, 6) + demo_index * 100
            if demo_index == 0:
                actions[0, :6] = 0.0
            actions[:, 6] = np.resize(np.array([-1.0, 1.0]), length)
            if invalid_gripper and demo_index == 1:
                actions[-1, 6] = 0.0
            demo.create_dataset("actions", data=actions)
            demo.attrs["num_samples"] = length

            obs = demo.create_group("obs")
            ee_state = np.arange(length * 6, dtype=np.float64).reshape(length, 6) + demo_index * 1000
            gripper_state = np.linspace(
                0.0,
                1.0,
                num=length * 2,
                dtype=np.float64,
            ).reshape(length, 2)
            obs.create_dataset("ee_states", data=ee_state)
            obs.create_dataset("gripper_states", data=gripper_state)

            agent = _test_images(length, channel=0, offset=demo_index * 10)
            wrist = _test_images(length, channel=1, offset=demo_index * 10 + 2)
            if invalid_wrist_shape and demo_index == 1:
                wrist = wrist[:, :, :-1, :]
            obs.create_dataset("agentview_rgb", data=agent)
            obs.create_dataset("eye_in_hand_rgb", data=wrist)

    return source_root


def _test_images(length: int, *, channel: int, offset: int) -> np.ndarray:
    images = np.zeros(
        (
            length,
            libero_v21.IMAGE_HEIGHT,
            libero_v21.IMAGE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )
    for frame_index in range(length):
        images[frame_index, :, :, channel] = frame_index + offset
        images[frame_index, 0, 0] = [255, 128, 64]
    return images


def _decode_video(path: Path) -> tuple[int, int, int]:
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert frames
    return len(frames), frames[0].width, frames[0].height


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
