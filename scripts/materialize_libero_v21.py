"""Materialize raw LIBERO HDF5 suites into LeRobot v2.1 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from prism.data.materialization.libero_v21 import IMAGE_TRANSFORMS
from prism.data.materialization.libero_v21 import LIBERO_SUITES
from prism.data.materialization.libero_v21 import VideoEncodingConfig
from prism.data.materialization.libero_v21 import materialize_libero_v21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Raw LIBERO directory containing suite subdirectories")
    parser.add_argument("output_root", type=Path, help="Destination parent for separate suite datasets")
    parser.add_argument(
        "--suite",
        action="append",
        choices=LIBERO_SUITES,
        dest="suites",
        help="Suite to materialize; repeat as needed (default: all four suites)",
    )
    parser.add_argument(
        "--image-transform",
        required=True,
        choices=sorted(IMAGE_TRANSFORMS),
        help="Explicit transform applied to both RGB views before encoding",
    )
    parser.add_argument("--codec", default="libsvtav1")
    parser.add_argument("--pixel-format", default="yuv420p")
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--gop", type=int, default=2)
    parser.add_argument(
        "--video-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional encoder option; repeat as needed",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Fail if a matching partial or completed dataset already exists",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_encoding = VideoEncodingConfig(
        codec=args.codec,
        pixel_format=args.pixel_format,
        crf=args.crf,
        gop=args.gop,
        options=_parse_video_options(args.video_option),
    )
    outputs = materialize_libero_v21(
        args.source_root,
        args.output_root,
        image_transform=args.image_transform,
        suites=tuple(args.suites or LIBERO_SUITES),
        video_encoding=video_encoding,
        resume=args.resume,
    )
    for output in outputs:
        print(output)


def _parse_video_options(values: list[str]) -> tuple[tuple[str, str], ...]:
    options = []
    for value in values:
        key, separator, option_value = value.partition("=")
        if not separator or not key:
            raise ValueError(f"--video-option must use KEY=VALUE syntax, got {value!r}")
        options.append((key, option_value))
    return tuple(options)


if __name__ == "__main__":
    main()
