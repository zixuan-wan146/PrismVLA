from __future__ import annotations

import argparse

from prism.config import load_config
from prism.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    set_global_seed(cfg.runtime.seed)
    from prism.data.cache import build_cache_from_config
    build_cache_from_config(cfg)


if __name__ == "__main__":
    main()
