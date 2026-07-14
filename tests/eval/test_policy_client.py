from __future__ import annotations

import asyncio

import numpy as np
import pytest
import websockets

from prism.serve.client import (
    InProcessPolicyClient,
    PolicyClientTimeoutError,
    WebSocketPolicyClient,
)
from prism.serve.protocol import HistoryObservationRequest, PolicyRequest
from prism.serve.wire import (
    metadata_envelope,
    pack_message,
    success_envelope,
    unpack_message,
    validate_envelope,
)


def _request(generation: int = 0) -> PolicyRequest:
    return PolicyRequest(
        benchmark="libero",
        prompt="pick up",
        images_by_view={"agentview_rgb": np.zeros((1, 1, 3), dtype=np.uint8)},
        state=np.zeros(8, dtype=np.float32),
        action_dim=7,
        stream_id="episode:1",
        memory_generation=generation,
        robot_key="libero",
    )


def _history(slot: int) -> HistoryObservationRequest:
    return HistoryObservationRequest(
        benchmark="libero",
        images_by_view={"agentview_rgb": np.full((1, 1, 3), slot, dtype=np.uint8)},
        stream_id="episode:1",
        target_generation=1,
        slot=slot,
        robot_key="libero",
    )


def test_in_process_policy_client_exposes_history_lifecycle():
    observed = []

    def infer(request):
        observed.append(("infer", request.memory_generation))
        return np.zeros((1, request.action_dim), dtype=np.float32)

    async def run_test():
        async with InProcessPolicyClient(
            infer,
            reset_history=lambda stream_id: observed.append(("reset", stream_id)),
            push_history_observation=lambda request: observed.append(("push", request.slot)),
        ) as client:
            await client.reset_history("episode:1")
            await client.push_history_observation(_history(0))
            return await client.infer(_request())

    response = asyncio.run(run_test())
    assert observed == [("reset", "episode:1"), ("push", 0), ("infer", 0)]
    np.testing.assert_array_equal(response["actions"], np.zeros((1, 7), dtype=np.float32))


def test_websocket_client_routes_out_of_order_history_responses_before_infer():
    observed = []

    async def run_test():
        async def handler(websocket):
            await websocket.send(pack_message(metadata_envelope({"action_horizon": 1, "action_dim": 7})))

            reset = validate_envelope(unpack_message(await websocket.recv()), expected_type="reset_history")
            await websocket.send(
                pack_message(success_envelope(reset["request_id"], "reset_history", {"reset": True}))
            )

            initial = validate_envelope(unpack_message(await websocket.recv()), expected_type="infer")
            observed.append(("infer", initial["payload"]["memory_generation"]))
            assert "history_images_by_view" not in initial["payload"]
            await websocket.send(
                pack_message(
                    success_envelope(
                        initial["request_id"],
                        "infer",
                        {"actions": np.zeros((1, 7), dtype=np.float32)},
                    )
                )
            )

            pushes = [
                validate_envelope(
                    unpack_message(await websocket.recv()),
                    expected_type="push_history_observation",
                )
                for _ in range(2)
            ]
            observed.extend(("push", message["payload"]["slot"]) for message in pushes)
            for message in reversed(pushes):
                await websocket.send(
                    pack_message(
                        success_envelope(
                            message["request_id"],
                            "push_history_observation",
                            {"memory_ready": message["payload"]["slot"] == 1},
                        )
                    )
                )

            precomputed = validate_envelope(unpack_message(await websocket.recv()), expected_type="infer")
            observed.append(("infer", precomputed["payload"]["memory_generation"]))
            await websocket.send(
                pack_message(
                    success_envelope(
                        precomputed["request_id"],
                        "infer",
                        {"actions": np.ones((1, 7), dtype=np.float32)},
                    )
                )
            )

        async with websockets.serve(handler, "127.0.0.1", 0, compression=None, max_size=None) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                await client.reset_history("episode:1")
                await client.infer(_request(0))
                await client.push_history_observation(_history(0))
                await client.push_history_observation(_history(1))
                response = await client.infer(_request(1))
                metadata = dict(client.metadata)
            return response, metadata

    response, metadata = asyncio.run(run_test())
    assert observed == [("infer", 0), ("push", 0), ("push", 1), ("infer", 1)]
    assert metadata == {"action_horizon": 1, "action_dim": 7}
    np.testing.assert_array_equal(response["actions"], np.ones((1, 7), dtype=np.float32))


def test_websocket_policy_client_requires_context_and_history_reset():
    client = WebSocketPolicyClient("ws://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="must be entered"):
        asyncio.run(client.infer(_request()))


def test_websocket_policy_client_connection_metadata_timeout_is_fatal():
    async def run_test():
        async def handler(websocket):
            await websocket.wait_closed()

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = WebSocketPolicyClient(
                f"ws://127.0.0.1:{port}",
                connect_timeout_seconds=0.05,
            )
            with pytest.raises(PolicyClientTimeoutError, match="fatal infrastructure error"):
                async with client:
                    raise AssertionError("client unexpectedly completed metadata handshake")

    asyncio.run(run_test())


def test_websocket_policy_client_inference_timeout_is_fatal_and_does_not_reconnect():
    connections = 0

    async def run_test():
        async def handler(websocket):
            nonlocal connections
            connections += 1
            await websocket.send(pack_message(metadata_envelope({"action_horizon": 1, "action_dim": 7})))
            reset = validate_envelope(unpack_message(await websocket.recv()), expected_type="reset_history")
            await websocket.send(
                pack_message(success_envelope(reset["request_id"], "reset_history", {"reset": True}))
            )
            await websocket.recv()
            await websocket.wait_closed()

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(
                f"ws://127.0.0.1:{port}",
                inference_timeout_seconds=0.05,
            ) as client:
                await client.reset_history("episode:1")
                with pytest.raises(PolicyClientTimeoutError, match="will not reconnect implicitly"):
                    await client.infer(_request())

    asyncio.run(run_test())
    assert connections == 1


def test_websocket_policy_client_rejects_missing_history_slot_before_next_inference():
    async def run_test():
        async def handler(websocket):
            await websocket.send(pack_message(metadata_envelope({})))
            reset = validate_envelope(unpack_message(await websocket.recv()), expected_type="reset_history")
            await websocket.send(
                pack_message(success_envelope(reset["request_id"], "reset_history", {"reset": True}))
            )
            initial = validate_envelope(unpack_message(await websocket.recv()), expected_type="infer")
            await websocket.send(
                pack_message(success_envelope(initial["request_id"], "infer", {"actions": []}))
            )
            push = validate_envelope(
                unpack_message(await websocket.recv()),
                expected_type="push_history_observation",
            )
            await websocket.send(
                pack_message(
                    success_envelope(push["request_id"], "push_history_observation", {"memory_ready": False})
                )
            )
            await websocket.wait_closed()

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                await client.reset_history("episode:1")
                await client.infer(_request(0))
                await client.push_history_observation(_history(0))
                with pytest.raises(RuntimeError, match=r"history slots \[1\] were not pushed"):
                    await client.infer(_request(1))

    asyncio.run(run_test())


def test_websocket_policy_client_rejects_nonpositive_timeouts():
    with pytest.raises(ValueError, match="finite and positive"):
        WebSocketPolicyClient("ws://127.0.0.1:9000", inference_timeout_seconds=0)
