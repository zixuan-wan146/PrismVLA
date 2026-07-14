#!/usr/bin/env python3
"""Serve a PrismVLA training checkpoint through the benchmark WebSocket API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.serve.server import run_checkpoint_server
from prism.serve.wire import DEFAULT_MAX_MESSAGE_SIZE_BYTES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="completed training checkpoint directory")
    parser.add_argument("--host", default="127.0.0.1", help="WebSocket bind host")
    parser.add_argument("--port", type=int, default=9000, help="WebSocket bind port")
    parser.add_argument(
        "--max-message-size-bytes",
        type=int,
        default=DEFAULT_MAX_MESSAGE_SIZE_BYTES,
        help="maximum accepted WebSocket message size",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="acknowledge the risk of binding this unauthenticated server beyond localhost",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="inference device: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=None,
        help="require the pretrained Qwen base files to already exist in the configured cache",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_checkpoint_server(
        args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
        local_files_only=args.local_files_only,
        max_message_size_bytes=args.max_message_size_bytes,
        allow_non_loopback=args.allow_non_loopback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
