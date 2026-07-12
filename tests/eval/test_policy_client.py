from __future__ import annotations

import asyncio

import numpy as np
import pytest
import websockets

from prism.eval.policy_client import InProcessPolicyClient, WebSocketPolicyClient
from prism.serve.protocol import policy_request_from_mapping
from prism.serve.wire import (
    metadata_envelope,
    pack_message,
    success_envelope,
    unpack_message,
    validate_envelope,
)


def _request():
    return policy_request_from_mapping(
        {
            "benchmark": "libero",
            "prompt": "pick up",
            "images_by_view": {"agentview_rgb": np.zeros((1, 1, 3), dtype=np.uint8)},
            "state": np.zeros(8, dtype=np.float32),
            "action_dim": 7,
            "robot_key": "libero",
        }
    )


def test_in_process_policy_client_returns_structured_actions():
    observed = {}

    def infer(request):
        observed["benchmark"] = request.benchmark
        return np.zeros((1, request.action_dim), dtype=np.float32)

    async def run_test():
        async with InProcessPolicyClient(infer) as client:
            return await client.infer(_request())

    response = asyncio.run(run_test())
    assert observed == {"benchmark": "libero"}
    np.testing.assert_array_equal(response["actions"], np.zeros((1, 7), dtype=np.float32))


def test_websocket_policy_client_msgpack_numpy_round_trip_and_metadata():
    observed = {}

    async def run_test():
        async def handler(websocket):
            await websocket.send(pack_message(metadata_envelope({"action_horizon": 1, "action_dim": 7})))
            message = validate_envelope(unpack_message(await websocket.recv()), expected_type="infer")
            payload = message["payload"]
            observed["benchmark"] = payload["benchmark"]
            observed["image_dtype"] = str(payload["images_by_view"]["agentview_rgb"].dtype)
            await websocket.send(
                pack_message(
                    success_envelope(
                        message["request_id"],
                        {"actions": np.ones((1, payload["action_dim"]), dtype=np.float32)},
                    )
                )
            )

        async with websockets.serve(
            handler,
            "127.0.0.1",
            0,
            compression=None,
            max_size=None,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                response = await client.infer(_request())
                metadata = dict(client.metadata)
            return response, metadata

    response, metadata = asyncio.run(run_test())
    assert observed == {"benchmark": "libero", "image_dtype": "uint8"}
    assert metadata == {"action_horizon": 1, "action_dim": 7}
    np.testing.assert_array_equal(response["actions"], np.ones((1, 7), dtype=np.float32))


def test_websocket_policy_client_requires_context_manager():
    client = WebSocketPolicyClient("ws://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="must be entered"):
        asyncio.run(client.infer(_request()))
