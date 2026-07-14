import numpy as np
import pytest

from prism.serve.wire import (
    WireProtocolError,
    error_envelope,
    pack_message,
    request_envelope,
    success_envelope,
    unpack_message,
    validate_envelope,
)


def test_msgpack_numpy_round_trip_preserves_arrays_without_json_lists():
    payload = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.arange(8, dtype=np.float32),
        "mask": np.array([True, False], dtype=bool),
    }

    encoded = pack_message(payload)
    decoded = unpack_message(encoded)

    assert isinstance(encoded, bytes)
    np.testing.assert_array_equal(decoded["image"], payload["image"])
    np.testing.assert_array_equal(decoded["state"], payload["state"])
    np.testing.assert_array_equal(decoded["mask"], payload["mask"])
    assert decoded["image"].dtype == np.uint8
    assert decoded["state"].dtype == np.float32


def test_msgpack_numpy_rejects_object_arrays():
    with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
        pack_message({"bad": np.array([object()], dtype=object)})


def test_envelope_validation_rejects_wrong_version_and_type():
    message = request_envelope("infer", 3, {"benchmark": "libero"})
    assert validate_envelope(message, expected_type="infer")["request_id"] == 3

    with pytest.raises(WireProtocolError, match="version"):
        validate_envelope({**message, "version": 99})
    with pytest.raises(WireProtocolError, match="message type"):
        validate_envelope(message, expected_type="metadata")


def test_protocol_v2_routes_generic_results_by_request_type():
    success = success_envelope(4, "push_history_observation", {"memory_ready": True})
    error = error_envelope(5, "reset_history", "bad stream")

    assert validate_envelope(success, expected_type="result")["request_type"] == "push_history_observation"
    assert error["type"] == "result"
    assert error["request_type"] == "reset_history"
    assert error["ok"] is False


def test_request_envelope_rejects_unknown_request_type():
    with pytest.raises(WireProtocolError, match="Unsupported request type"):
        request_envelope("legacy_infer", 1, {})


def test_unpack_rejects_text_websocket_frames():
    with pytest.raises(WireProtocolError, match="binary WebSocket frame"):
        unpack_message("not binary")
