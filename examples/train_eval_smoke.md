# Train And Eval Smoke

This smoke path follows the same high-level pattern as the vendored StarVLA
LIBERO recipe: a shell-safe script calls a config-driven Python trainer, and all
data/output paths are provided by YAML rather than hard-coded in code.

From the repository root:

```bash
python scripts/build_cache.py --config configs/experiment/libero_train_smoke.yaml
python scripts/train.py --config configs/experiment/libero_train_smoke.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

The train smoke config builds a tiny deterministic episode-feature cache under
`local_data/smoke/`, runs one real Stage1 optimizer step, and writes
`local_data/smoke/runs/stage1_train_smoke/step_final`.

The eval smoke configs run in `dry_run` mode. They validate the LIBERO/CALVIN
runtime profiles through `scripts/eval.py` without requiring simulator packages,
datasets, or a live policy server. Disable `dry_run` only after those external
runtime dependencies are installed and configured.
