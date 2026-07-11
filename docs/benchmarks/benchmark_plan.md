# PrismVLA Benchmark Plan

PrismVLA targets two runnable benchmarks:

- LIBERO
- CALVIN ABC->D

Other benchmarks are out of scope for this reconstruction.

## Current Entry Points

Build cache:

```bash
python scripts/build_cache.py --config configs/experiment/libero_stage1.yaml
python scripts/build_cache.py --config configs/experiment/calvin_stage1.yaml
```

Progress warmup:

```bash
python scripts/warmup.py --config configs/experiment/libero_warmup_w4.yaml
python scripts/warmup.py --config configs/experiment/calvin_warmup_w4.yaml
```

Stage1 training:

```bash
python scripts/train.py --config configs/experiment/libero_stage1.yaml
python scripts/train.py --config configs/experiment/calvin_stage1.yaml
```

Smoke eval:

```bash
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

## Module Ownership

- LIBERO data: `prism/data/libero.py`
- CALVIN data: `prism/data/calvin.py`
- Cache IO: `prism/data/cache.py`
- LIBERO eval: `prism/eval/libero.py`
- CALVIN eval: `prism/eval/calvin.py`
- Shared runner: `prism/eval/runner.py`

## Data Paths

Dataset and output paths are configured as project-relative YAML values. Typical
ignored local locations are:

```text
local_data/datasets/...
local_data/token_caches/...
local_data/runs/...
run_outputs/...
```

Do not hard-code machine-specific absolute paths in configs, scripts, docs, or
tests.
