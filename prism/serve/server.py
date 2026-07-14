from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any

import numpy as np
import websockets

from prism.serve.backend import CheckpointPolicyBackend, PolicyBackend, PolicyBackendInference
from prism.models.task_state_planner import TaskStatePlannerRuntimeState
from prism.serve.history import ConnectionHistoryState
from prism.serve.protocol import (
    history_observation_from_mapping,
    history_reset_from_mapping,
    policy_request_from_mapping,
)
from prism.serve.wire import (
    PROTOCOL_VERSION,
    REQUEST_TYPES,
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
    """Serve one client with bounded, connection-local token history state."""

    history_state = ConnectionHistoryState()
    planning_states: dict[str, TaskStatePlannerRuntimeState] = {}
    await websocket.send(pack_message(metadata_envelope(build_server_metadata(backend))))
    try:
        async for raw_message in websocket:
            request_id = -1
            request_type = "infer"
            try:
                decoded = unpack_message(raw_message)
                if isinstance(decoded, Mapping) and decoded.get("type") in REQUEST_TYPES:
                    request_type = str(decoded["type"])
                message = validate_envelope(decoded)
                if message.get("type") not in REQUEST_TYPES:
                    raise ValueError(f"Unsupported request type {message.get('type')!r}")
                request_type = str(message["type"])
                raw_request_id = message.get("request_id")
                if isinstance(raw_request_id, bool) or not isinstance(raw_request_id, int):
                    raise ValueError(f"request_id must be an integer, got {raw_request_id!r}")
                if raw_request_id < 0:
                    raise ValueError(f"request_id must be non-negative, got {raw_request_id}")
                request_id = raw_request_id
                payload = message.get("payload")
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{request_type} payload must be a mapping")
                response = await _dispatch_request(
                    request_type=request_type,
                    payload=payload,
                    backend=backend,
                    history_state=history_state,
                    planning_states=planning_states,
                    inference_lock=inference_lock,
                )
                await websocket.send(
                    pack_message(success_envelope(request_id, request_type, response))
                )
            except Exception as exc:
                logging.exception(
                    "Failed to handle request_type=%s request_id=%s",
                    request_type,
                    request_id,
                )
                await websocket.send(
                    pack_message(error_envelope(request_id, request_type, str(exc)))
                )
    finally:
        history_state.clear()
        planning_states.clear()


async def _dispatch_request(
    *,
    request_type: str,
    payload: Mapping[str, Any],
    backend: PolicyBackend,
    history_state: ConnectionHistoryState,
    planning_states: dict[str, TaskStatePlannerRuntimeState],
    inference_lock: asyncio.Lock,
) -> dict[str, Any]:
    if request_type == "reset_history":
        request = history_reset_from_mapping(payload)
        history_state.reset(request.stream_id)
        # ConnectionHistoryState permits one active stream, so a reset must drop
        # every prior stream key to keep planning state bounded across subtasks.
        planning_states.clear()
        return {
            "stream_id": request.stream_id,
            "reset": True,
            "planning_state_reset": True,
        }

    if request_type == "push_history_observation":
        request = history_observation_from_mapping(payload)
        async with inference_lock:
            observation = backend.encode_history_observation(request)
            memory_ready = history_state.add_observation(
                stream_id=request.stream_id,
                target_generation=request.target_generation,
                slot=request.slot,
                observation=observation,
                build_memory=backend.build_history_memory,
            )
        return {
            "stream_id": request.stream_id,
            "target_generation": request.target_generation,
            "slot": request.slot,
            "memory_ready": memory_ready,
        }

    if request_type == "infer":
        request = policy_request_from_mapping(payload)
        async with inference_lock:
            memory = history_state.memory_for_inference(
                stream_id=request.stream_id,
                generation=request.memory_generation,
                empty_memory=backend.empty_history_memory,
            )
            cycle_result = backend.infer_cycle(
                request,
                memory,
                planning_state=planning_states.get(request.stream_id),
            )
            if not isinstance(cycle_result, PolicyBackendInference):
                raise TypeError("PolicyBackend.infer_cycle() must return PolicyBackendInference")
            result = cycle_result.response
            next_planning_state = cycle_result.planning_state
            history_state.mark_inference_complete(
                stream_id=request.stream_id,
                generation=request.memory_generation,
            )
            if next_planning_state is None:
                planning_states.pop(request.stream_id, None)
            else:
                planning_states[request.stream_id] = next_planning_state
        if isinstance(result, Mapping):
            response = dict(result)
            if "actions" in response:
                response["actions"] = np.asarray(response["actions"], dtype=np.float32)
            return response
        return {"actions": np.asarray(result, dtype=np.float32)}

    raise ValueError(f"Unsupported request type {request_type!r}")


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


def run_checkpoint_server(
    checkpoint_path: str | Path,
    *,
    host: str,
    port: int,
    device: str | None = None,
    local_files_only: bool | None = None,
) -> None:
    """Load one verified checkpoint and serve it through the stable protocol."""

    if not isinstance(host, str) or not host:
        raise ValueError("host must be non-empty text")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(f"port must be in [1, 65535], got {port!r}")
    backend = CheckpointPolicyBackend.from_checkpoint(
        checkpoint_path,
        device=device,
        local_files_only=local_files_only,
    )
    asyncio.run(serve(backend, host=host, port=port))


__all__ = ["build_server_metadata", "handle_request", "run_checkpoint_server", "serve"]
