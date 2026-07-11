from __future__ import annotations

import argparse
from dataclasses import fields

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
    from prism.training.warmup import ProgressWarmupTrainingConfig
    from prism.training.warmup import run_progress_warmup_training

    warmup_fields = {field.name for field in fields(ProgressWarmupTrainingConfig)}
    warmup_config = {key: value for key, value in cfg.raw.items() if key in warmup_fields}
    run_progress_warmup_training(warmup_config)


if __name__ == "__main__":
    main()
