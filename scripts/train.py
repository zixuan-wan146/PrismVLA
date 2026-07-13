#!/usr/bin/env python3
"""Thin CLI for the single resolved PrismVLA training path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prism.training.runner import run_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="resolved training YAML")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="completed checkpoint directory to resume",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(
        args.config,
        resume_from=args.resume,
        project_root=PROJECT_ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
