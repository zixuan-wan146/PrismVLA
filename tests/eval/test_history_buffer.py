from __future__ import annotations

import numpy as np
import pytest

from prism.serve.history import SparseHistoryBuffer


VIEW_NAMES = ("static", "wrist")


def images(value: int) -> dict[str, np.ndarray]:
    return {
        "static": np.full((3, 4, 3), value, dtype=np.uint8),
        "wrist": np.full((2, 2, 3), value + 1, dtype=np.uint8),
    }


def test_sparse_history_buffer_captures_only_offsets_two_and_five():
    buffer = SparseHistoryBuffer(VIEW_NAMES)
    for step in range(1, 8):
        captured = buffer.capture(step, images(step))
        assert captured is (step in {2, 5})

    assert buffer.captured_offsets == (2, 5)
    payload = buffer.consume(images(8))

    assert payload.step_ages.tolist() == [6, 3]
    assert payload.valid_mask.tolist() == [True, True]
    np.testing.assert_array_equal(payload.images_by_view["static"][:, 0, 0, 0], [2, 5])
    np.testing.assert_array_equal(payload.images_by_view["wrist"][:, 0, 0, 0], [3, 6])
    assert buffer.captured_offsets == ()


def test_initial_history_uses_zero_slots_and_invalid_mask():
    payload = SparseHistoryBuffer(VIEW_NAMES).consume(images(0))

    assert payload.valid_mask.tolist() == [False, False]
    assert not payload.images_by_view["static"].any()
    assert not payload.images_by_view["wrist"].any()


def test_history_buffer_copies_captured_images():
    source = images(2)
    buffer = SparseHistoryBuffer(VIEW_NAMES)
    buffer.capture(2, source)
    source["static"].fill(99)
    payload = buffer.consume(images(8))

    assert payload.images_by_view["static"][0, 0, 0, 0] == 2


def test_history_buffer_rejects_duplicate_capture():
    buffer = SparseHistoryBuffer(VIEW_NAMES)
    buffer.capture(2, images(2))
    with pytest.raises(ValueError, match="more than once"):
        buffer.capture(2, images(2))
