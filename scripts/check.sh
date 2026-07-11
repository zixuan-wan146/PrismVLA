#!/usr/bin/env bash
set -euo pipefail
python -m ruff check .
python -m pytest
python - <<'PY'
from prism.config import load_config
for path in [
    'configs/experiment/libero_stage1.yaml',
    'configs/experiment/libero_warmup_w4.yaml',
    'configs/experiment/libero_smoke.yaml',
    'configs/experiment/calvin_stage1.yaml',
    'configs/experiment/calvin_warmup_w4.yaml',
    'configs/experiment/calvin_smoke.yaml',
    'configs/experiment/libero_train_smoke.yaml',
]:
    load_config(path)
print('config validation ok')
PY
