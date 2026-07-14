from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


PROTOCOL_VERSION = 2
WIRE_FORMAT = "msgpack-numpy"
REQUEST_TYPES = frozenset({"infer", "push_history_observation", "reset_history"})
# Two uncompressed 448x448 RGB views occupy about 1.2 MiB. This limit keeps
# accepted benchmark requests comfortably below an accidental memory-DoS size.
DEFAULT_MAX_MESSAGE_SIZE_BYTES = 16 * 1024 * 1024


class WireProtocolError(RuntimeError):
    pass


def positive_message_size_bytes(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def pack_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_numpy, use_bin_type=True)


def unpack_message(payload: bytes | bytearray | memoryview) -> Any:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise WireProtocolError(f"Expected a binary WebSocket frame, got {type(payload).__name__}")
    try:
        return msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise WireProtocolError(f"Invalid {WIRE_FORMAT} payload: {exc}") from exc


def metadata_envelope(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "metadata",
        "metadata": metadata,
    }


def request_envelope(request_type: str, request_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    request_type = _validate_request_type(request_type)
    return {
        "version": PROTOCOL_VERSION,
        "type": request_type,
        "request_id": int(request_id),
        "payload": payload,
    }


def success_envelope(request_id: int, request_type: str, data: Any) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "result",
        "request_id": int(request_id),
        "request_type": _validate_request_type(request_type),
        "ok": True,
        "data": data,
    }


def error_envelope(request_id: int, request_type: str, message: str) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": "result",
        "request_id": int(request_id),
        "request_type": _validate_request_type(request_type),
        "ok": False,
        "error": {"message": str(message)},
    }


def validate_envelope(message: Any, *, expected_type: str | None = None) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise WireProtocolError(f"Protocol envelope must be a mapping, got {type(message).__name__}")
    version = message.get("version")
    if version != PROTOCOL_VERSION:
        raise WireProtocolError(f"Unsupported protocol version {version!r}; expected {PROTOCOL_VERSION}")
    if expected_type is not None and message.get("type") != expected_type:
        raise WireProtocolError(f"Expected message type {expected_type!r}, got {message.get('type')!r}")
    return message


def _validate_request_type(request_type: str) -> str:
    if request_type not in REQUEST_TYPES:
        raise WireProtocolError(
            f"Unsupported request type {request_type!r}; expected one of {sorted(REQUEST_TYPES)}"
        )
    return request_type


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"V", "O", "c"}:
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "data": array.tobytes(),
            "dtype": array.dtype.str,
            "shape": array.shape,
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in {"V", "O", "c"}:
            raise ValueError(f"Unsupported NumPy scalar dtype: {value.dtype}")
        return {
            "__npgeneric__": True,
            "data": value.item(),
            "dtype": value.dtype.str,
        }
    raise TypeError(f"Cannot encode {type(value).__name__} in {WIRE_FORMAT}")


def _unpack_numpy(value: dict[str, Any]) -> Any:
    if value.get("__ndarray__") is True:
        try:
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
            return np.frombuffer(value["data"], dtype=dtype).reshape(shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise WireProtocolError(f"Invalid NumPy array payload: {exc}") from exc
    if value.get("__npgeneric__") is True:
        try:
            return np.dtype(value["dtype"]).type(value["data"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WireProtocolError(f"Invalid NumPy scalar payload: {exc}") from exc
    return value
