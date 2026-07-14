#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="$(cd "$repo_root/.." && pwd)"
env_prefix="${PRISM_LIBERO_ENV_PREFIX:-$data_root/envs/libero}"
conda_bin="${PRISM_CONDA_BIN:-$data_root/miniforge3/bin/conda}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$data_root/pip-cache}"
export TMPDIR="${TMPDIR:-$data_root/tmp}"
export PIP_INDEX_URL="${PRISM_PIP_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$env_prefix")"

if [[ "${PRISM_ENABLE_NETWORK_TURBO:-0}" == "1" && -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

if [[ ! -x "$conda_bin" ]]; then
    echo "Conda executable is not available: $conda_bin" >&2
    exit 1
fi
if [[ ! -x "$env_prefix/bin/python" ]]; then
    "$conda_bin" create -y -p "$env_prefix" python=3.8.13 pip=24.2
fi

"$env_prefix/bin/python" -m pip install --requirement "$repo_root/requirements/libero-eval.lock.txt"
"$env_prefix/bin/python" -m pip check

cd "$repo_root"
"$env_prefix/bin/python" -m experiments.libero.eval --config experiments/libero/configs/smoke.yaml
