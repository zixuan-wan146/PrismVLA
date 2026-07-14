from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
import inspect
from typing import Any, Protocol

import websockets

from prism.serve.protocol import (
    HistoryObservationRequest,
    HistoryResetRequest,
    PolicyRequest,
    history_observation_to_mapping,
    history_reset_to_mapping,
    policy_request_to_mapping,
)
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

    async def reset_history(self, stream_id: str) -> None: ...

    async def push_history_observation(self, request: HistoryObservationRequest) -> None: ...

    async def infer(self, request: PolicyRequest) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _PendingRequest:
    request_type: str
    future: asyncio.Future[Any]


class WebSocketPolicyClient:
    """Protocol-v2 client with background response routing and history prefetch."""

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
        self._send_lock: asyncio.Lock | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._pending: dict[int, _PendingRequest] = {}
        self._history_acknowledgements: dict[tuple[str, int, int], asyncio.Future[Any]] = {}
        self._ready_history_generations: set[tuple[str, int]] = set()
        self._active_stream_id: str | None = None
        self._last_inferred_generation = -1
        self._receiver_error: BaseException | None = None
        self._fatal_error: BaseException | None = None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    async def __aenter__(self) -> "WebSocketPolicyClient":
        if self._websocket is not None:
            raise RuntimeError("WebSocketPolicyClient is already entered")
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
            self._send_lock = asyncio.Lock()
            self._receiver_error = None
            self._fatal_error = None
            self._receiver_task = asyncio.create_task(
                self._receive_responses(),
                name="prism-policy-response-router",
            )
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
            self._reset_connection_state()

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        receiver_task = self._receiver_task
        self._receiver_task = None
        if receiver_task is not None:
            receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_task
        self._cancel_pending()
        if self._connection is not None:
            await self._connection.__aexit__(exc_type, exc, traceback)
        self._reset_connection_state()

    async def reset_history(self, stream_id: str) -> None:
        self._ensure_usable("reset_history()")
        request = HistoryResetRequest(stream_id=stream_id)
        _request_id, future = await self._send_request(
            "reset_history",
            history_reset_to_mapping(request),
        )
        await self._wait_for_result(future, operation="history reset")
        self._discard_history_acknowledgements()
        self._active_stream_id = stream_id
        self._last_inferred_generation = -1

    async def push_history_observation(self, request: HistoryObservationRequest) -> None:
        self._ensure_usable("push_history_observation()")
        if not isinstance(request, HistoryObservationRequest):
            raise TypeError(f"request must be HistoryObservationRequest, got {type(request).__name__}")
        self._require_active_stream(request.stream_id)
        expected_generation = self._last_inferred_generation + 1
        if expected_generation <= 0:
            raise RuntimeError("Run initial generation-0 inference before pushing history observations")
        if request.target_generation != expected_generation:
            raise ValueError(
                f"Expected history target_generation {expected_generation}, got {request.target_generation}"
            )
        key = (request.stream_id, request.target_generation, request.slot)
        if key in self._history_acknowledgements:
            raise ValueError(
                f"History slot {request.slot} for generation {request.target_generation} was already pushed"
            )
        _request_id, future = await self._send_request(
            "push_history_observation",
            history_observation_to_mapping(request),
        )
        self._history_acknowledgements[key] = future

    async def infer(self, request: PolicyRequest) -> Mapping[str, Any]:
        self._ensure_usable("infer()")
        if not isinstance(request, PolicyRequest):
            raise TypeError(f"request must be PolicyRequest, got {type(request).__name__}")
        self._require_active_stream(request.stream_id)
        expected_generation = self._last_inferred_generation + 1
        if request.memory_generation != expected_generation:
            raise ValueError(
                f"Expected inference memory_generation {expected_generation}, got {request.memory_generation}"
            )
        if request.memory_generation > 0:
            history_key = (request.stream_id, request.memory_generation)
            if history_key not in self._ready_history_generations:
                await self._wait_for_history(request.stream_id, request.memory_generation)

        request_id, future = await self._send_request(
            "infer",
            policy_request_to_mapping(request),
        )
        data = await self._wait_for_result(
            future,
            operation=f"policy inference request {request_id}",
        )
        if not isinstance(data, Mapping):
            error = WireProtocolError(
                f"Inference response data must be a mapping, got {type(data).__name__}"
            )
            self._fatal_error = error
            raise error
        self._last_inferred_generation = request.memory_generation
        self._ready_history_generations.discard(
            (request.stream_id, request.memory_generation)
        )
        return data

    async def _send_request(
        self,
        request_type: str,
        payload: dict[str, Any],
    ) -> tuple[int, asyncio.Future[Any]]:
        self._ensure_usable(f"{request_type} request")
        assert self._websocket is not None
        assert self._send_lock is not None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        future.add_done_callback(_consume_future_exception)
        async with self._send_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._pending[request_id] = _PendingRequest(request_type=request_type, future=future)
            try:
                await self._websocket.send(
                    pack_message(request_envelope(request_type, request_id, payload))
                )
            except BaseException:
                self._pending.pop(request_id, None)
                future.cancel()
                raise
        return request_id, future

    async def _receive_responses(self) -> None:
        assert self._websocket is not None
        try:
            async for raw_response in self._websocket:
                response = validate_envelope(unpack_message(raw_response), expected_type="result")
                request_id = response.get("request_id")
                if isinstance(request_id, bool) or not isinstance(request_id, int):
                    raise WireProtocolError(f"Response request_id must be an integer, got {request_id!r}")
                pending = self._pending.get(request_id)
                if pending is None:
                    raise WireProtocolError(f"Received response for unknown request_id={request_id}")
                if response.get("request_type") != pending.request_type:
                    raise WireProtocolError(
                        f"Response request_type={response.get('request_type')!r} does not match "
                        f"request type {pending.request_type!r} for request_id={request_id}"
                    )
                self._pending.pop(request_id)
                if response.get("ok") is not True:
                    error = response.get("error")
                    message = error.get("message") if isinstance(error, Mapping) else str(error)
                    pending.future.set_exception(
                        RuntimeError(f"Prism server {pending.request_type} failed: {message}")
                    )
                else:
                    pending.future.set_result(response.get("data"))
            raise WireProtocolError("Policy server closed the response stream")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._receiver_error = exc
            self._fail_pending(exc)

    async def _wait_for_history(self, stream_id: str, generation: int) -> None:
        keys = tuple((stream_id, generation, slot) for slot in (0, 1))
        missing = [slot for slot, key in enumerate(keys) if key not in self._history_acknowledgements]
        if missing:
            raise RuntimeError(
                f"Cannot infer generation {generation}; history slots {missing} were not pushed"
            )
        futures = tuple(self._history_acknowledgements[key] for key in keys)
        try:
            await asyncio.wait_for(
                asyncio.gather(*(asyncio.shield(future) for future in futures)),
                timeout=self.inference_timeout_seconds,
            )
            self._ready_history_generations.add((stream_id, generation))
        except asyncio.TimeoutError:
            error = PolicyClientTimeoutError(
                f"history precompute for generation {generation} timed out after "
                f"{self.inference_timeout_seconds:g} seconds; this is a fatal infrastructure error "
                "and the client will not reconnect implicitly"
            )
            self._fatal_error = error
            raise error from None
        finally:
            for key in keys:
                self._history_acknowledgements.pop(key, None)

    async def _wait_for_result(self, future: asyncio.Future[Any], *, operation: str) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.inference_timeout_seconds,
            )
        except asyncio.TimeoutError:
            error = PolicyClientTimeoutError(
                f"{operation} timed out after {self.inference_timeout_seconds:g} seconds; "
                "this is a fatal infrastructure error and the client will not reconnect implicitly"
            )
            self._fatal_error = error
            raise error from None

    def _ensure_usable(self, operation: str) -> None:
        if self._websocket is None:
            raise RuntimeError(f"WebSocketPolicyClient must be entered before {operation}")
        if self._fatal_error is not None:
            raise RuntimeError("WebSocketPolicyClient is unusable after a fatal request failure") from self._fatal_error
        if self._receiver_error is not None:
            raise RuntimeError("Policy response router has stopped") from self._receiver_error

    def _require_active_stream(self, stream_id: str) -> None:
        if self._active_stream_id is None:
            raise RuntimeError("History stream is not initialized; call reset_history() first")
        if stream_id != self._active_stream_id:
            raise ValueError(f"Active stream is {self._active_stream_id!r}, got {stream_id!r}")

    def _discard_history_acknowledgements(self) -> None:
        self._history_acknowledgements.clear()
        self._ready_history_generations.clear()

    def _fail_pending(self, exc: BaseException) -> None:
        pending = tuple(self._pending.values())
        self._pending.clear()
        for item in pending:
            if not item.future.done():
                item.future.set_exception(exc)

    def _cancel_pending(self) -> None:
        pending = tuple(self._pending.values())
        self._pending.clear()
        for item in pending:
            if not item.future.done():
                item.future.cancel()
        self._discard_history_acknowledgements()

    def _reset_connection_state(self) -> None:
        self._connection = None
        self._websocket = None
        self._metadata = {}
        self._send_lock = None
        self._receiver_task = None
        self._pending.clear()
        self._history_acknowledgements.clear()
        self._ready_history_generations.clear()
        self._active_stream_id = None
        self._last_inferred_generation = -1
        self._receiver_error = None
        self._fatal_error = None


class InProcessPolicyClient:
    """Policy adapter for tests and dependency-compatible local evaluation."""

    def __init__(
        self,
        infer: Callable[[PolicyRequest], Any],
        metadata: Mapping[str, Any] | None = None,
        *,
        reset_history: Callable[[str], Any] | None = None,
        push_history_observation: Callable[[HistoryObservationRequest], Any] | None = None,
    ) -> None:
        self._infer = infer
        self._reset_history = reset_history
        self._push_history_observation = push_history_observation
        self._metadata = dict(metadata or {})

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    async def __aenter__(self) -> "InProcessPolicyClient":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def reset_history(self, stream_id: str) -> None:
        if self._reset_history is not None:
            await _maybe_await(self._reset_history(stream_id))

    async def push_history_observation(self, request: HistoryObservationRequest) -> None:
        if self._push_history_observation is not None:
            await _maybe_await(self._push_history_observation(request))

    async def infer(self, request: PolicyRequest) -> Mapping[str, Any]:
        response = await _maybe_await(self._infer(request))
        if isinstance(response, Mapping):
            return response
        return {"actions": response}


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = [
    "InProcessPolicyClient",
    "PolicyClient",
    "PolicyClientTimeoutError",
    "WebSocketPolicyClient",
]
