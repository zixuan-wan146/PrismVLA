#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PRISM_LIBERO_PYTHON:-$repo_root/../envs/libero/bin/python}"
config="${1:-experiments/libero/configs/eval.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -x "$python_bin" ]]; then
    echo "LIBERO Python is not executable: $python_bin" >&2
    exit 1
fi

cd "$repo_root"
exec "$python_bin" -m experiments.libero.eval --config "$config" "$@"
