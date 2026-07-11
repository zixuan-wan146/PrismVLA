# Benchmark Research Summary

This note is retained as the benchmark-selection record. The reconstructed
PrismVLA codebase targets LIBERO and CALVIN with one shared script surface.

## Selected Benchmarks

- LIBERO: short-horizon and task-suite manipulation evaluation.
- CALVIN ABC->D: long-horizon language-conditioned manipulation evaluation.

## Current Runnable Surface

```bash
python scripts/build_cache.py --config configs/experiment/libero_stage1.yaml
python scripts/build_cache.py --config configs/experiment/calvin_stage1.yaml
python scripts/warmup.py --config configs/experiment/libero_warmup_w4.yaml
python scripts/warmup.py --config configs/experiment/calvin_warmup_w4.yaml
python scripts/train.py --config configs/experiment/libero_stage1.yaml
python scripts/train.py --config configs/experiment/calvin_stage1.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

Benchmark-specific implementation lives under `prism/data/` and `prism/eval/`.
Dataset paths, cache paths, and output paths must remain project-relative.
