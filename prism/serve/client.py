from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Protocol

import websockets

from prism.serve.protocol import PolicyRequest, policy_request_to_mapping
from prism.serve.wire import (
    WireProtocolError,
    pack_message,
    request_envelope,
    unpack_message,
    validate_envelope,
)


class PolicyClient(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    async def __aenter__(self) -> "PolicyClient": ...

    async def __aexit__(self, exc_type, exc, traceback) -> None: ...

    async def infer(self, request: PolicyRequest) -> Mapping[str, Any]: ...


class WebSocketPolicyClient:
    """Async MessagePack/NumPy transport adapted from StarVLA's policy client."""

    def __init__(self, server_url: str) -> None:
        self.server_url = str(server_url)
        self._connection = None
        self._websocket = None
        self._metadata: dict[str, Any] = {}
        self._next_request_id = 1

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    async def __aenter__(self) -> "WebSocketPolicyClient":
        self._connection = websockets.connect(
            self.server_url,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        )
        self._websocket = await self._connection.__aenter__()
        metadata_message = validate_envelope(
            unpack_message(await self._websocket.recv()),
            expected_type="metadata",
        )
        metadata = metadata_message.get("metadata")
        if not isinstance(metadata, dict):
            raise WireProtocolError("Server metadata payload must be a mapping")
        self._metadata = metadata
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._connection is not None:
            await self._connection.__aexit__(exc_type, exc, traceback)
        self._connection = None
        self._websocket = None
        self._metadata = {}

    async def infer(self, request: PolicyRequest) -> Mapping[str, Any]:
        if self._websocket is None:
            raise RuntimeError("WebSocketPolicyClient must be entered before infer()")

        request_id = self._next_request_id
        self._next_request_id += 1
        message = request_envelope(request_id, policy_request_to_mapping(request))
        await self._websocket.send(pack_message(message))

        response = validate_envelope(
            unpack_message(await self._websocket.recv()),
            expected_type="inference_result",
        )
        if response.get("request_id") != request_id:
            raise WireProtocolError(
                f"Response request_id={response.get('request_id')!r} does not match request_id={request_id}"
            )
        if response.get("ok") is not True:
            error = response.get("error")
            message = error.get("message") if isinstance(error, Mapping) else str(error)
            raise RuntimeError(f"Prism server inference failed: {message}")

        data = response.get("data")
        if not isinstance(data, Mapping):
            raise WireProtocolError(f"Inference response data must be a mapping, got {type(data).__name__}")
        return data


class InProcessPolicyClient:
    """Policy adapter for tests and dependency-compatible local evaluation."""

    def __init__(self, infer: Callable[[PolicyRequest], Any], metadata: Mapping[str, Any] | None = None) -> None:
        self._infer = infer
        self._metadata = dict(metadata or {})

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    async def __aenter__(self) -> "InProcessPolicyClient":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def infer(self, request: PolicyRequest) -> Mapping[str, Any]:
        response = self._infer(request)
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response, Mapping):
            return response
        return {"actions": response}
