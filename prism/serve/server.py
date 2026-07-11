from __future__ import annotations

# --- migrated from src/prism/server_protocol.py ---
"""Compatibility shim for older imports.

New runtime code should use :mod:`prism.serve.engine`.
"""

from collections.abc import Mapping
from typing import Any

from prism.serve.engine import (
    PolicyInferenceEngine,
    PolicyRequest,
    RuntimePolicyState,
    checkpoint_normalizer_dim,
    policy_request_from_json,
)

def validate_inference_request(data: Mapping[str, Any], **_kwargs) -> PolicyRequest:
    return policy_request_from_json(data)


__all__ = ["PolicyRequest", "checkpoint_normalizer_dim", "policy_request_from_json", "validate_inference_request"]

# --- migrated from src/prism/runtime/websocket_server.py ---
import argparse
import asyncio
import json
import logging
from pathlib import Path

import torch
import websockets

from prism.models.policy import PrismPolicy
from prism.config import DEFAULT_MAX_MESSAGE_SIZE
from prism.config import DEFAULT_SERVER_HOST
from prism.config import DEFAULT_SERVER_PORT
from prism.config import TARGET_STATE_DIM
from prism.utils import NormalizationStats


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
    return resolved


def load_checkpoint_payload(ckpt_path: Path, *, allow_unsafe_checkpoint_load: bool):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe_checkpoint_load:
            raise RuntimeError(
                "Checkpoint could not be loaded with torch.load(weights_only=True). "
                "Only enable unsafe pickle loading for trusted local checkpoints."
            ) from exc
        logging.warning(
            "Falling back to torch.load(weights_only=False). Only use this with trusted local checkpoints. "
            "Original safe-load error: %s",
            exc,
        )
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def load_model_and_normalizer(
    ckpt_dir,
    device: str = "cuda",
    inference_steps: int = 15,
    allow_unsafe_checkpoint_load: bool = False,
    vlm_name: str | None = None,
    vlm_local_files_only: bool = False,
):
    device = resolve_device(device)
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    stats_path = ckpt_dir / "norm_stats.json"
    ckpt_path = ckpt_dir / "model.pt"

    for path in (config_path, stats_path, ckpt_path):
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint file not found: {path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    checkpoint_load_vlm = bool(config.get("load_vlm", True))
    if vlm_name:
        config["vlm_name"] = vlm_name
    config["device"] = str(device)
    config["load_vlm"] = True
    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = inference_steps
    config["vlm_local_files_only"] = bool(vlm_local_files_only or config.get("vlm_local_files_only", False))
    logging.info(
        "Runtime VLM config: vlm_name=%s local_files_only=%s",
        config.get("vlm_name"),
        config["vlm_local_files_only"],
    )

    model = PrismPolicy(config).eval()
    checkpoint = load_checkpoint_payload(
        ckpt_path,
        allow_unsafe_checkpoint_load=allow_unsafe_checkpoint_load,
    )
    if isinstance(checkpoint, dict) and checkpoint.get("format") == "stage1_torch_checkpoint":
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    load_result = model.load_state_dict(state_dict, strict=checkpoint_load_vlm)
    if not checkpoint_load_vlm:
        bad_missing = [key for key in load_result.missing_keys if not key.startswith("embedder.")]
        if bad_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected non-VLM checkpoint mismatch while loading token-cache checkpoint: "
                f"missing={bad_missing[:20]}, unexpected={load_result.unexpected_keys[:20]}"
            )
        logging.info(
            "Loaded token-cache checkpoint action-side weights with VLM initialized from base model "
            "(ignored %d missing embedder keys).",
            len(load_result.missing_keys),
        )
    model = model.to(device)

    normalizer_dim = checkpoint_normalizer_dim(config)
    normalizer = NormalizationStats(stats, target_dim=normalizer_dim)
    logging.info("Loaded normalization stats robot_keys=%s default_robot_key=%s", normalizer.robot_keys, normalizer.robot_key)
    return model, normalizer


async def handle_request(websocket, engine: PolicyInferenceEngine, inference_lock: asyncio.Lock):
    logging.info("Client connected")
    runtime_state = RuntimePolicyState()
    try:
        async for message in websocket:
            try:
                request = policy_request_from_json(json.loads(message))
                logging.info("Received policy request benchmark=%s", request.benchmark)
                async with inference_lock:
                    actions = engine.infer(request, runtime_state)
                await websocket.send(json.dumps(actions))
                logging.info("Sent action chunk")
            except Exception as exc:
                logging.exception("Failed to handle request")
                await websocket.send(json.dumps({"error": str(exc)}))
    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected.")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the PrismVLA websocket inference server.")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory containing config.json and weights.")
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference_steps", type=int, default=15)
    parser.add_argument(
        "--vlm_name",
        default=None,
        help="Override checkpoint config.vlm_name. May be a local model directory or a cached HF repo id.",
    )
    parser.add_argument(
        "--vlm_local_files_only",
        action="store_true",
        help="Load the VLM only from local files/cache and never contact Hugging Face.",
    )
    parser.add_argument(
        "--allow_unsafe_checkpoint_load",
        action="store_true",
        help="Allow torch.load(weights_only=False) fallback for trusted local checkpoints.",
    )
    return parser.parse_args(argv)


async def serve(engine: PolicyInferenceEngine, *, host: str, port: int) -> None:
    logging.info("PrismVLA server running at ws://%s:%s", host, port)
    inference_lock = asyncio.Lock()
    async with websockets.serve(
        lambda ws: handle_request(ws, engine, inference_lock),
        host,
        port,
        max_size=DEFAULT_MAX_MESSAGE_SIZE,
        ping_interval=None,
        ping_timeout=None,
    ):
        await asyncio.Future()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("Loading PrismVLA model...")
    model, normalizer = load_model_and_normalizer(
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        inference_steps=args.inference_steps,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        vlm_name=args.vlm_name,
        vlm_local_files_only=args.vlm_local_files_only,
    )
    engine = PolicyInferenceEngine(
        model,
        normalizer,
        state_dim=int(model.config.get("state_dim", TARGET_STATE_DIM)),
    )
    asyncio.run(serve(engine, host=args.host, port=args.port))
    return 0




def serve_from_config(cfg):
    """Config-dispatched serving entry point used by scripts/serve.py."""

    raw = getattr(cfg, "raw", {})
    return main(raw) if callable(globals().get("main")) else None
