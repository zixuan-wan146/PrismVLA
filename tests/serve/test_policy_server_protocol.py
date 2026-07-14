from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest
import torch
import websockets

from prism.models.task_state_planner import MambaStreamingCache, TaskStatePlannerRuntimeState
from prism.serve.backend import PolicyBackendInference
from prism.serve.client import WebSocketPolicyClient
from prism.serve.history import ConnectionHistoryState
from prism.serve.protocol import (
    HistoryObservationRequest,
    PolicyRequest,
    history_observation_to_mapping,
    policy_request_to_mapping,
)
from prism.serve.server import _dispatch_request, handle_request


class _Backend:
    metadata = {"action_horizon": 2, "action_dim": 7}

    def __init__(self) -> None:
        self.encoded_slots = []
        self.memory_builds = []
        self.inferences = []
        self.fail_generation_one_once = False
        self.fail_encode_slot_once = None
        self.encode_attempts = []

    def encode_history_observation(self, request):
        self.encode_attempts.append(request.slot)
        if request.slot == self.fail_encode_slot_once:
            self.fail_encode_slot_once = None
            raise RuntimeError("transient history encoding failure")
        encoded = f"visual-slot-{request.slot}"
        self.encoded_slots.append((request.slot, encoded))
        return encoded

    def build_history_memory(self, observations):
        self.memory_builds.append(observations)
        return f"memory:{observations[0]}:{observations[1]}"

    def empty_history_memory(self):
        return "empty-memory"

    def infer(self, request, memory):
        self.inferences.append((request.memory_generation, memory))
        if request.memory_generation == 1 and self.fail_generation_one_once:
            self.fail_generation_one_once = False
            raise RuntimeError("transient inference failure")
        return np.full((2, 7), request.memory_generation, dtype=np.float32)

    def infer_cycle(self, request, memory, *, planning_state):
        return PolicyBackendInference(
            response=self.infer(request, memory),
            planning_state=planning_state,
        )


class _PlanningBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.previous_planning_values = []
        self.fail_generation_one_once = True

    def infer_cycle(self, request, memory, *, planning_state):
        del memory
        previous_value = (
            None
            if planning_state is None
            else float(planning_state.task_state.item())
        )
        self.previous_planning_values.append(previous_value)
        if request.memory_generation == 1 and self.fail_generation_one_once:
            self.fail_generation_one_once = False
            raise RuntimeError("transient planning failure")
        next_value = 1.0 if previous_value is None else previous_value + 1.0
        next_state = TaskStatePlannerRuntimeState(
            task_state=torch.tensor([next_value]),
            mamba_cache=MambaStreamingCache(
                conv_state=torch.tensor([next_value]),
                ssm_state=torch.tensor([next_value]),
            ),
        )
        return PolicyBackendInference(
            response={
                "actions": np.full(
                    (2, 7),
                    request.memory_generation,
                    dtype=np.float32,
                )
            },
            planning_state=next_state,
        )


def _request(generation: int, *, stream_id: str = "episode:1") -> PolicyRequest:
    return PolicyRequest(
        benchmark="libero",
        prompt="pick up",
        images_by_view={"agentview_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
        state=np.zeros(8, dtype=np.float32),
        action_dim=7,
        stream_id=stream_id,
        memory_generation=generation,
    )


def _history(slot: int) -> HistoryObservationRequest:
    return HistoryObservationRequest(
        benchmark="libero",
        images_by_view={"agentview_rgb": np.full((2, 2, 3), slot, dtype=np.uint8)},
        stream_id="episode:1",
        target_generation=1,
        slot=slot,
    )


def test_policy_server_and_client_precompute_session_memory():
    async def run_test():
        backend = _Backend()
        inference_lock = asyncio.Lock()

        async def handler(websocket):
            await handle_request(websocket, backend, inference_lock)

        async with websockets.serve(
            handler,
            "127.0.0.1",
            0,
            compression=None,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                await client.reset_history("episode:1")
                initial = await client.infer(_request(0))
                await client.push_history_observation(_history(0))
                await client.push_history_observation(_history(1))
                precomputed = await client.infer(_request(1))
                metadata = dict(client.metadata)
        return initial, precomputed, metadata, backend

    initial, precomputed, metadata, backend = asyncio.run(run_test())
    assert metadata["protocol_version"] == 2
    assert metadata["wire_format"] == "msgpack-numpy"
    np.testing.assert_array_equal(initial["actions"], np.zeros((2, 7), dtype=np.float32))
    np.testing.assert_array_equal(precomputed["actions"], np.ones((2, 7), dtype=np.float32))
    assert backend.encoded_slots == [(0, "visual-slot-0"), (1, "visual-slot-1")]
    assert backend.memory_builds == [("visual-slot-0", "visual-slot-1")]
    assert backend.inferences == [
        (0, "empty-memory"),
        (1, "memory:visual-slot-0:visual-slot-1"),
    ]


def test_client_rejects_inference_when_a_history_slot_was_not_pushed():
    async def run_test():
        backend = _Backend()
        inference_lock = asyncio.Lock()

        async def handler(websocket):
            await handle_request(websocket, backend, inference_lock)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                await client.reset_history("episode:1")
                await client.infer(_request(0))
                await client.push_history_observation(_history(0))
                try:
                    await client.infer(_request(1))
                except RuntimeError as exc:
                    return str(exc)
        raise AssertionError("inference unexpectedly succeeded")

    message = asyncio.run(run_test())
    assert "history slots [1] were not pushed" in message


def test_failed_inference_can_retry_the_same_precomputed_generation():
    async def run_test():
        backend = _Backend()
        backend.fail_generation_one_once = True
        inference_lock = asyncio.Lock()

        async def handler(websocket):
            await handle_request(websocket, backend, inference_lock)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                await client.reset_history("episode:1")
                await client.infer(_request(0))
                await client.push_history_observation(_history(0))
                await client.push_history_observation(_history(1))
                try:
                    await client.infer(_request(1))
                except RuntimeError as exc:
                    assert "transient inference failure" in str(exc)
                else:
                    raise AssertionError("first inference unexpectedly succeeded")
                response = await client.infer(_request(1))
        return response, backend

    response, backend = asyncio.run(run_test())
    np.testing.assert_array_equal(response["actions"], np.ones((2, 7), dtype=np.float32))
    assert backend.memory_builds == [("visual-slot-0", "visual-slot-1")]
    assert [generation for generation, _memory in backend.inferences] == [0, 1, 1]


def test_planning_state_commits_only_after_success_and_reset_clears_it():
    async def run_test():
        backend = _PlanningBackend()
        inference_lock = asyncio.Lock()

        async def handler(websocket):
            await handle_request(websocket, backend, inference_lock)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                await client.reset_history("episode:1")
                await client.infer(_request(0))
                await client.push_history_observation(_history(0))
                await client.push_history_observation(_history(1))
                with pytest.raises(RuntimeError, match="transient planning failure"):
                    await client.infer(_request(1))
                await client.infer(_request(1))
                await client.reset_history("episode:2")
                await client.infer(_request(0, stream_id="episode:2"))
        return backend

    backend = asyncio.run(run_test())
    assert backend.previous_planning_values == [None, 1.0, 1.0, None]


def test_invalid_history_request_is_rejected_before_waiting_for_gpu_lock():
    async def run_test():
        backend = _Backend()
        history_state = ConnectionHistoryState()
        planning_states = {}
        inference_lock = asyncio.Lock()
        await _dispatch_request(
            request_type="reset_history",
            payload={"stream_id": "episode:1"},
            backend=backend,
            history_state=history_state,
            planning_states=planning_states,
            inference_lock=inference_lock,
        )
        await _dispatch_request(
            request_type="infer",
            payload=policy_request_to_mapping(_request(0)),
            backend=backend,
            history_state=history_state,
            planning_states=planning_states,
            inference_lock=inference_lock,
        )
        first_payload = history_observation_to_mapping(_history(0))
        await _dispatch_request(
            request_type="push_history_observation",
            payload=first_payload,
            backend=backend,
            history_state=history_state,
            planning_states=planning_states,
            inference_lock=inference_lock,
        )

        await inference_lock.acquire()
        try:
            with pytest.raises(ValueError, match="pushed more than once"):
                await asyncio.wait_for(
                    _dispatch_request(
                        request_type="push_history_observation",
                        payload=first_payload,
                        backend=backend,
                        history_state=history_state,
                        planning_states=planning_states,
                        inference_lock=inference_lock,
                    ),
                    timeout=0.1,
                )
            future_payload = history_observation_to_mapping(
                replace(_history(1), target_generation=2)
            )
            with pytest.raises(ValueError, match="Expected history for generation 1"):
                await asyncio.wait_for(
                    _dispatch_request(
                        request_type="push_history_observation",
                        payload=future_payload,
                        backend=backend,
                        history_state=history_state,
                        planning_states=planning_states,
                        inference_lock=inference_lock,
                    ),
                    timeout=0.1,
                )
            wrong_stream_payload = history_observation_to_mapping(
                replace(_history(1), stream_id="episode:2")
            )
            with pytest.raises(ValueError, match="Active stream"):
                await asyncio.wait_for(
                    _dispatch_request(
                        request_type="push_history_observation",
                        payload=wrong_stream_payload,
                        backend=backend,
                        history_state=history_state,
                        planning_states=planning_states,
                        inference_lock=inference_lock,
                    ),
                    timeout=0.1,
                )
            invalid_slot_payload = history_observation_to_mapping(_history(1))
            invalid_slot_payload["slot"] = 2
            with pytest.raises(ValueError, match="slot must be 0 or 1"):
                await asyncio.wait_for(
                    _dispatch_request(
                        request_type="push_history_observation",
                        payload=invalid_slot_payload,
                        backend=backend,
                        history_state=history_state,
                        planning_states=planning_states,
                        inference_lock=inference_lock,
                    ),
                    timeout=0.1,
                )
        finally:
            inference_lock.release()
        return backend

    backend = asyncio.run(run_test())
    assert backend.encode_attempts == [0]
    assert backend.encoded_slots == [(0, "visual-slot-0")]


def test_history_encoding_failure_rolls_back_reservation_for_retry():
    async def run_test():
        backend = _Backend()
        backend.fail_encode_slot_once = 0
        history_state = ConnectionHistoryState()
        planning_states = {}
        inference_lock = asyncio.Lock()
        await _dispatch_request(
            request_type="reset_history",
            payload={"stream_id": "episode:1"},
            backend=backend,
            history_state=history_state,
            planning_states=planning_states,
            inference_lock=inference_lock,
        )
        await _dispatch_request(
            request_type="infer",
            payload=policy_request_to_mapping(_request(0)),
            backend=backend,
            history_state=history_state,
            planning_states=planning_states,
            inference_lock=inference_lock,
        )
        payload = history_observation_to_mapping(_history(0))
        with pytest.raises(RuntimeError, match="transient history encoding failure"):
            await _dispatch_request(
                request_type="push_history_observation",
                payload=payload,
                backend=backend,
                history_state=history_state,
                planning_states=planning_states,
                inference_lock=inference_lock,
            )
        assert history_state.reserved_slots == ()
        response = await _dispatch_request(
            request_type="push_history_observation",
            payload=payload,
            backend=backend,
            history_state=history_state,
            planning_states=planning_states,
            inference_lock=inference_lock,
        )
        return backend, response

    backend, response = asyncio.run(run_test())
    assert backend.encode_attempts == [0, 0]
    assert backend.encoded_slots == [(0, "visual-slot-0")]
    assert response["memory_ready"] is False
