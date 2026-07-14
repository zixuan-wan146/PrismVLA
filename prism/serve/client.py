from __future__ import annotations

import asyncio
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
from prism.utils.evaluation import (
    DEFAULT_POLICY_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_POLICY_INFERENCE_TIMEOUT_SECONDS,
    finite_positive_seconds,
)


class PolicyClientTimeoutError(TimeoutError):
    """Fatal policy infrastructure timeout; clients do not reconnect implicitly."""


class PolicyClient(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    async def __aenter__(self) -> "PolicyClient": ...

    async def __aexit__(self, exc_type, exc, traceback) -> None: ...

    async def infer(self, request: PolicyRequest) -> Mapping[str, Any]: ...


class WebSocketPolicyClient:
    """Async MessagePack/NumPy transport with fatal, bounded request waits."""

    def __init__(
        self,
        server_url: str,
        *,
        connect_timeout_seconds: float = DEFAULT_POLICY_CONNECT_TIMEOUT_SECONDS,
        inference_timeout_seconds: float = DEFAULT_POLICY_INFERENCE_TIMEOUT_SECONDS,
    ) -> None:
        self.server_url = str(server_url)
        self.connect_timeout_seconds = finite_positive_seconds(
            connect_timeout_seconds,
            "connect_timeout_seconds",
        )
        self.inference_timeout_seconds = finite_positive_seconds(
            inference_timeout_seconds,
            "inference_timeout_seconds",
        )
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
            open_timeout=self.connect_timeout_seconds,
        )

        async def connect_and_receive_metadata() -> Any:
            self._websocket = await self._connection.__aenter__()
            return await self._websocket.recv()

        try:
            raw_metadata = await asyncio.wait_for(
                connect_and_receive_metadata(),
                timeout=self.connect_timeout_seconds,
            )
            metadata_message = validate_envelope(
                unpack_message(raw_metadata),
                expected_type="metadata",
            )
            metadata = metadata_message.get("metadata")
            if not isinstance(metadata, dict):
                raise WireProtocolError("Server metadata payload must be a mapping")
            self._metadata = metadata
            return self
        except asyncio.TimeoutError:
            await self._close_after_enter_failure()
            raise PolicyClientTimeoutError(
                "policy connection or metadata handshake timed out after "
                f"{self.connect_timeout_seconds:g} seconds; this is a fatal infrastructure error"
            ) from None
        except BaseException:
            await self._close_after_enter_failure()
            raise

    async def _close_after_enter_failure(self) -> None:
        try:
            if self._connection is not None and self._websocket is not None:
                await self._connection.__aexit__(None, None, None)
        finally:
            self._connection = None
            self._websocket = None
            self._metadata = {}

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

        async def send_and_receive() -> Any:
            await self._websocket.send(pack_message(message))
            return await self._websocket.recv()

        try:
            raw_response = await asyncio.wait_for(
                send_and_receive(),
                timeout=self.inference_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise PolicyClientTimeoutError(
                f"policy inference request {request_id} timed out after "
                f"{self.inference_timeout_seconds:g} seconds; this is a fatal infrastructure error "
                "and the client will not reconnect implicitly"
            ) from None

        response = validate_envelope(
            unpack_message(raw_response),
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


__all__ = [
    "InProcessPolicyClient",
    "PolicyClient",
    "PolicyClientTimeoutError",
    "WebSocketPolicyClient",
]
