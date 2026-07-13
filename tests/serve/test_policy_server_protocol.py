from __future__ import annotations

import asyncio

import numpy as np
import websockets

from prism.serve.client import WebSocketPolicyClient
from prism.serve.protocol import policy_request_from_mapping
from prism.serve.server import handle_request


class _Backend:
    metadata = {"action_horizon": 2, "action_dim": 7}

    def infer(self, request):
        assert request.benchmark == "libero"
        return np.zeros((2, 7), dtype=np.float32)


def _request():
    return policy_request_from_mapping(
        {
            "benchmark": "libero",
            "prompt": "pick up",
            "images_by_view": {"agentview_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            "history_images_by_view": {"agentview_rgb": np.zeros((2, 2, 2, 3), dtype=np.uint8)},
            "history_step_ages": np.array([6, 3], dtype=np.int32),
            "history_valid_mask": np.array([False, False]),
            "state": np.zeros(8, dtype=np.float32),
            "action_dim": 7,
        }
    )


def test_policy_server_and_client_use_structured_binary_protocol():
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
            max_size=None,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with WebSocketPolicyClient(f"ws://127.0.0.1:{port}") as client:
                response = await client.infer(_request())
                metadata = dict(client.metadata)
        return response, metadata

    response, metadata = asyncio.run(run_test())
    assert metadata["wire_format"] == "msgpack-numpy"
    assert metadata["action_horizon"] == 2
    assert metadata["action_dim"] == 7
    np.testing.assert_array_equal(response["actions"], np.zeros((2, 7), dtype=np.float32))
