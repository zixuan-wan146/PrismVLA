from __future__ import annotations

# --- migrated from src/prism/server_protocol.py ---
"""Compatibility shim for older imports.

New runtime code should use :mod:`prism.serve.engine`.
"""

from collections.abc import Mapping
from typing import Any

from prism.serve.protocol import PolicyRequest, checkpoint_normalizer_dim, policy_request_from_mapping
from prism.serve.engine import (
    PolicyInferenceEngine,
    RuntimePolicyState,
)


def validate_inference_request(data: Mapping[str, Any], **_kwargs) -> PolicyRequest:
    return policy_request_from_mapping(data)


__all__ = ["PolicyRequest", "checkpoint_normalizer_dim", "policy_request_from_mapping", "validate_inference_request"]

# --- migrated from src/prism/runtime/websocket_server.py ---
import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import torch
import websockets

from prism.models.policy import PrismPolicy
from prism.config import DEFAULT_SERVER_HOST
from prism.config import DEFAULT_SERVER_PORT
from prism.config import TARGET_STATE_DIM
from prism.utils import NormalizationStats
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
    logging.info(
        "Loaded normalization stats robot_keys=%s default_robot_key=%s", normalizer.robot_keys, normalizer.robot_key
    )
    return model, normalizer


def build_server_metadata(engine: PolicyInferenceEngine) -> dict[str, Any]:
    model = getattr(engine, "model", None)
    config = getattr(model, "config", {}) if model is not None else {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "wire_format": WIRE_FORMAT,
        "action_horizon": int(getattr(model, "action_horizon", config.get("horizon", 0)) or 0),
        "action_dim": int(getattr(model, "per_action_dim", config.get("per_action_dim", 0)) or 0),
    }


async def handle_request(websocket, engine: PolicyInferenceEngine, inference_lock: asyncio.Lock):
    logging.info("Client connected")
    runtime_state = RuntimePolicyState()
    await websocket.send(pack_message(metadata_envelope(build_server_metadata(engine))))
    try:
        async for raw_message in websocket:
            request_id = -1
            try:
                message = validate_envelope(unpack_message(raw_message), expected_type="infer")
                request_id = int(message.get("request_id", -1))
                payload = message.get("payload")
                if not isinstance(payload, Mapping):
                    raise ValueError("Inference payload must be a mapping")
                request = policy_request_from_mapping(payload)
                logging.info("Received policy request benchmark=%s request_id=%s", request.benchmark, request_id)
                async with inference_lock:
                    result = engine.infer(request, runtime_state)
                if isinstance(result, Mapping):
                    data = dict(result)
                    if "actions" in data:
                        data["actions"] = np.asarray(data["actions"], dtype=np.float32)
                else:
                    data = {"actions": np.asarray(result, dtype=np.float32)}
                await websocket.send(pack_message(success_envelope(request_id, data)))
                logging.info("Sent action chunk request_id=%s", request_id)
            except Exception as exc:
                logging.exception("Failed to handle request_id=%s", request_id)
                await websocket.send(pack_message(error_envelope(request_id, str(exc))))
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
        compression=None,
        max_size=None,
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


def server_argv_from_config(cfg) -> list[str]:
    raw = dict(getattr(cfg, "raw", {}) or {})
    ckpt_dir = raw.get("ckpt_dir")
    if not ckpt_dir:
        raise ValueError("Serving config must define ckpt_dir")

    argv = ["--ckpt_dir", str(ckpt_dir)]
    value_options = {
        "host": "--host",
        "port": "--port",
        "device": "--device",
        "inference_steps": "--inference_steps",
        "vlm_name": "--vlm_name",
    }
    for key, option in value_options.items():
        value = raw.get(key)
        if value is not None and str(value) != "":
            argv.extend((option, str(value)))

    if bool(raw.get("vlm_local_files_only", False)):
        argv.append("--vlm_local_files_only")
    if bool(raw.get("allow_unsafe_checkpoint_load", False)):
        argv.append("--allow_unsafe_checkpoint_load")
    return argv


def serve_from_config(cfg) -> int:
    """Config-dispatched serving entry point used by scripts/serve.py."""

    return main(server_argv_from_config(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
