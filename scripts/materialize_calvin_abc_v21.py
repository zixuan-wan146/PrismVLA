"""Materialize the complete Collision CALVIN ABC-D LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from prism.data.materialization.calvin_abc_v21 import build_calvin_abc_v21_plan
from prism.data.materialization.calvin_abc_v21 import materialize_calvin_abc_v21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "collision_root",
        type=Path,
        help="Pinned CollisionCode v2.1 root with metadata, videos, and existing Parquets",
    )
    parser.add_argument(
        "traly_root",
        type=Path,
        help="Pinned Traly v3 numeric donor root with the three aggregate Parquets",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="New sibling destination; it is never overwritten",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=8,
        help="Workers used to verify the pinned Collision file manifest",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Refuse a matching partial or completed run instead of resuming it",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(message: str) -> None:
        print(message, flush=True)

    plan = build_calvin_abc_v21_plan(
        args.collision_root,
        args.traly_root,
        hash_workers=args.hash_workers,
        progress=progress,
    )
    print(f"plan_sha256={plan.sha256}", flush=True)
    print(f"mapping_sha256={plan.mapping_sha256}", flush=True)
    output = materialize_calvin_abc_v21(
        plan,
        args.output_root,
        resume=args.resume,
        decode_samples=True,
        progress=progress,
    )
    print(output)


if __name__ == "__main__":
    main()
