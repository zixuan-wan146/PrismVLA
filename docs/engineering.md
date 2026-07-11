# PrismVLA Engineering Notes

This document records the current runnable engineering surface after the
reconstruction. Historical `src/`, nested training, cache, quality, and eval
script paths are retired; use the root package and canonical scripts below.

## Package Boundaries

```text
prism/config.py       config schema and YAML loading
prism/models/         VLM wrapper, memory, planner, action head, policy
prism/data/           LIBERO/CALVIN readers, cache IO, action segments
prism/training/       generic trainer, warmup trainer, losses
prism/eval/           LIBERO/CALVIN benchmark adapters and runner
prism/serve/          websocket server and inference engine
prism/utils/          paths, seeding, logging, normalization helpers
```

Scripts are entry points only. Business logic belongs under `prism/`.

## Canonical Commands

```bash
scripts/check.sh
python scripts/build_cache.py --config configs/experiment/libero_stage1.yaml
python scripts/warmup.py --config configs/experiment/libero_warmup_w4.yaml
python scripts/train.py --config configs/experiment/libero_stage1.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/serve.py --config configs/experiment/libero_smoke.yaml
```

CALVIN uses the same entry points:

```bash
python scripts/build_cache.py --config configs/experiment/calvin_stage1.yaml
python scripts/warmup.py --config configs/experiment/calvin_warmup_w4.yaml
python scripts/train.py --config configs/experiment/calvin_stage1.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

## Config Rules

- Shared model defaults live in `configs/model/prism_base.yaml`.
- Dataset defaults live in `configs/data/libero.yaml` and
  `configs/data/calvin.yaml`.
- Runnable profiles live in `configs/experiment/`.
- Paths inside YAML must stay project-relative.
- Do not create per-stage script trees or duplicate CLI modules.

## Runtime Data

Large data, caches, checkpoints, and logs are not repository files. Use relative
paths such as `local_data/...`, `run_outputs/...`, `checkpoints/...`, or
`logs/...`; these locations are ignored by git.

## Verification

Run the repository gate from the project root:

```bash
scripts/check.sh
```

The gate runs `ruff`, `pytest`, and config loading for all LIBERO and CALVIN
experiment YAMLs.
