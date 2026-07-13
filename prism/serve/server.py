from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
from typing import Any

import numpy as np
import websockets

from prism.serve.backend import PolicyBackend
from prism.serve.protocol import policy_request_from_mapping
from prism.serve.wire import (
    PROTOCOL_VERSION,
    WIRE_FORMAT,
    error_envelope,
    metadata_envelope,
    pack_message,
    success_envelope,
    unpack_message,
    validate_envelope,
)


def build_server_metadata(backend: PolicyBackend) -> dict[str, Any]:
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "wire_format": WIRE_FORMAT,
    }
    metadata.update(dict(backend.metadata))
    return metadata


async def handle_request(websocket, backend: PolicyBackend, inference_lock: asyncio.Lock) -> None:
    """Serve one benchmark client without retaining model-side episode state."""

    await websocket.send(pack_message(metadata_envelope(build_server_metadata(backend))))
    async for raw_message in websocket:
        request_id = -1
        try:
            message = validate_envelope(unpack_message(raw_message), expected_type="infer")
            request_id = int(message.get("request_id", -1))
            payload = message.get("payload")
            if not isinstance(payload, Mapping):
                raise ValueError("Inference payload must be a mapping")
            request = policy_request_from_mapping(payload)
            async with inference_lock:
                result = backend.infer(request)
            if isinstance(result, Mapping):
                response = dict(result)
                if "actions" in response:
                    response["actions"] = np.asarray(response["actions"], dtype=np.float32)
            else:
                response = {"actions": np.asarray(result, dtype=np.float32)}
            await websocket.send(pack_message(success_envelope(request_id, response)))
        except Exception as exc:
            logging.exception("Failed to handle request_id=%s", request_id)
            await websocket.send(pack_message(error_envelope(request_id, str(exc))))


async def serve(backend: PolicyBackend, *, host: str, port: int) -> None:
    """Expose a future policy implementation through the stable benchmark protocol."""

    inference_lock = asyncio.Lock()
    async with websockets.serve(
        lambda websocket: handle_request(websocket, backend, inference_lock),
        host,
        port,
        compression=None,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
    ):
        await asyncio.Future()
