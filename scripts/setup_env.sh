#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$(dirname "$PROJECT_ROOT")"
ENV_ROOT="${PRSIM_ENV_ROOT:-$DATA_ROOT/envs/prsim}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/pip-cache}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
PIP_INDEX_URL="${PRSIM_PIP_INDEX_URL:-https://pypi.org/simple}"

mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR" "$(dirname "$ENV_ROOT")"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_ROOT" python=3.10 pip
fi

"$ENV_ROOT/bin/python" -m pip install --index-url "$PIP_INDEX_URL" --upgrade pip
"$ENV_ROOT/bin/python" -m pip install --index-url "$PIP_INDEX_URL" -e "$PROJECT_ROOT[data,dev]"

"$ENV_ROOT/bin/python" - <<'PY'
import torch
import transformers

print(f"prsim python ready: torch={torch.__version__} transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
