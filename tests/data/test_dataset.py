from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from experiments.calvin.data import CALVIN_DATA_SPEC
from prism.data.dataset import SingleVLADataset
from prism.data.schema import ViewSpec


@dataclass(frozen=True)
class _RawFrame:
    state: np.ndarray
    actions: np.ndarray
    instruction: str


class FakeBackend:
    spec = CALVIN_DATA_SPEC

    def __init__(self, lengths: tuple[int, ...] = (4, 7)) -> None:
        self.lengths = lengths
        self.fail_images_at: tuple[int, int] | None = None

    def episode_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self.lengths)))

    def episode_length(self, episode_id: int) -> int:
        return self.lengths[episode_id]

    def read_training_window(self, episode_id: int, start: int, end: int) -> _RawFrame:
        return _RawFrame(
            state=np.arange(8, dtype=np.float32) + start,
            actions=np.stack([self._action(index) for index in range(start, end)], axis=0),
            instruction=f"episode {episode_id} frame {start}",
        )

    def read_images(
        self,
        episode_id: int,
        frame_indices: Sequence[int],
        views: Sequence[ViewSpec] | None = None,
    ) -> dict[str, np.ndarray]:
        if self.fail_images_at is not None and (episode_id, frame_indices[0]) == self.fail_images_at:
            raise RuntimeError("synthetic decode failure")
        requested = self.spec.views if views is None else tuple(views)
        shapes = {"primary": (6, 8, 3), "wrist": (3, 4, 3)}
        bases = {"primary": 10, "wrist": 100}
        return {
            view.name: np.stack(
                [np.full(shapes[view.name], bases[view.name] + index, dtype=np.uint8) for index in frame_indices],
                axis=0,
            )
            for view in requested
        }

    @staticmethod
    def _action(frame_index: int) -> np.ndarray:
        action = np.arange(7, dtype=np.float32) + frame_index * 10
        action[-1] = 1.0 if frame_index % 2 == 0 else -1.0
        return action


class FakeNormalizer:
    statistics_group = "calvin_abc"

    def normalize_state(self, raw_state: np.ndarray) -> np.ndarray:
        return np.asarray(raw_state, dtype=np.float32) / 10.0

    def normalize_action(self, raw_action: np.ndarray) -> np.ndarray:
        output = np.asarray(raw_action, dtype=np.float32).copy()
        output[..., :6] /= 100.0
        output[..., 6] = (output[..., 6] + 1.0) / 2.0
        return output


def _dataset(
    *,
    backend: FakeBackend | None = None,
    action_horizon: int = 4,
    history_step_ages: tuple[int, ...] = (3, 1),
    anchor_stride: int = 1,
    include_tail: bool = True,
) -> SingleVLADataset:
    return SingleVLADataset(
        name="calvin_a",
        backend=FakeBackend() if backend is None else backend,
        normalizer=FakeNormalizer(),
        action_horizon=action_horizon,
        history_step_ages=history_step_ages,
        anchor_stride=anchor_stride,
        include_tail=include_tail,
    )


def test_temporal_sample_zero_fills_invalid_history_and_action_tail() -> None:
    sample = _dataset()[1]

    assert (sample.episode_index, sample.frame_index) == (0, 1)
    assert sample.dataset_name == "calvin_a"
    assert sample.statistics_group == "calvin_abc"
    assert sample.policy_input.prompt == "episode 0 frame 1"
    assert tuple(sample.policy_input.images_by_view) == ("primary", "wrist")
    assert sample.policy_input.images_by_view["primary"][0, 0, 0] == 11
    assert sample.policy_input.history_step_ages.tolist() == [3, 1]
    assert sample.policy_input.history_valid_mask.tolist() == [False, True]
    assert not sample.policy_input.history_images_by_view["primary"][0].any()
    assert sample.policy_input.history_images_by_view["primary"][1, 0, 0, 0] == 10
    assert sample.policy_input.history_images_by_view["wrist"].shape == (2, 3, 4, 3)
    np.testing.assert_array_equal(
        sample.policy_input.state,
        (np.arange(8, dtype=np.float32) + 1) / 10.0,
    )

    assert sample.target_actions.shape == (4, 7)
    assert sample.action_valid_mask.tolist() == [True, True, True, False]
    assert sample.target_actions[:, 6].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert not sample.target_actions[-1].any()


def test_anchor_stride_and_tail_policy_are_explicit() -> None:
    with_tail = _dataset(anchor_stride=2, include_tail=True)
    without_tail = _dataset(anchor_stride=1, include_tail=False)

    assert [with_tail.anchor_identity(index).frame_index for index in range(len(with_tail))] == [
        0,
        2,
        0,
        2,
        4,
        6,
    ]
    assert [without_tail.anchor_identity(index).frame_index for index in range(len(without_tail))] == [
        0,
        0,
        1,
        2,
        3,
    ]


def test_negative_dataset_index_matches_map_style_semantics() -> None:
    dataset = _dataset(anchor_stride=2)

    assert dataset.anchor_identity(-1) == dataset.anchor_identity(len(dataset) - 1)
    with pytest.raises(IndexError, match="outside"):
        dataset.anchor_identity(len(dataset))


def test_storage_errors_are_not_retried_with_another_anchor() -> None:
    backend = FakeBackend()
    backend.fail_images_at = (0, 1)
    dataset = _dataset(backend=backend)

    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        dataset[1]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("action_horizon", 0, "positive"),
        ("history_step_ages", (), "at least one"),
        ("history_step_ages", (1, 1), "duplicates"),
        ("anchor_stride", 0, "positive"),
    ],
)
def test_invalid_temporal_configuration_is_rejected(
    keyword: str,
    value: object,
    message: str,
) -> None:
    arguments = {
        "action_horizon": 4,
        "history_step_ages": (3, 1),
        "anchor_stride": 1,
        "include_tail": True,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        _dataset(**arguments)


def test_excluding_tail_rejects_dataset_with_no_full_horizon() -> None:
    with pytest.raises(ValueError, match="contains no anchors"):
        _dataset(
            backend=FakeBackend(lengths=(2,)),
            action_horizon=4,
            include_tail=False,
        )
