#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="$(cd "$repo_root/.." && pwd)"
env_prefix="${PRISM_CALVIN_ENV_PREFIX:-$data_root/envs/calvin}"
runtime_root="${PRISM_CALVIN_ROOT:-$data_root/benchmarks/calvin/runtime}"
conda_bin="${PRISM_CONDA_BIN:-$data_root/miniforge3/bin/conda}"
calvin_repository="${PRISM_CALVIN_REPOSITORY:-https://github.com/mees/calvin.git}"
calvin_commit="fa03f01f19c65920e18cf37398a9ce859274af76"
calvin_env_commit="1431a46bd36bde5903fb6345e68b5ccc30def666"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$data_root/pip-cache}"
export TMPDIR="${TMPDIR:-$data_root/tmp}"
export PIP_INDEX_URL="${PRISM_PIP_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.org/simple}}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$env_prefix")" "$(dirname "$runtime_root")"

if [[ "${PRISM_ENABLE_NETWORK_TURBO:-0}" == "1" && -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

if [[ ! -d "$runtime_root/.git" ]]; then
    if [[ -e "$runtime_root" ]]; then
        echo "CALVIN runtime exists but is not a Git checkout: $runtime_root" >&2
        exit 1
    fi
    git clone --recurse-submodules "$calvin_repository" "$runtime_root"
    git -C "$runtime_root" checkout --detach "$calvin_commit"
    git -C "$runtime_root" submodule update --init --recursive
fi

actual_calvin_commit="$(git -C "$runtime_root" rev-parse HEAD)"
actual_calvin_env_commit="$(git -C "$runtime_root/calvin_env" rev-parse HEAD)"
if [[ "$actual_calvin_commit" != "$calvin_commit" ]]; then
    echo "CALVIN runtime commit mismatch: expected $calvin_commit, got $actual_calvin_commit" >&2
    exit 1
fi
if [[ "$actual_calvin_env_commit" != "$calvin_env_commit" ]]; then
    echo "calvin_env commit mismatch: expected $calvin_env_commit, got $actual_calvin_env_commit" >&2
    exit 1
fi
if [[ -n "$(git -C "$runtime_root" status --porcelain --untracked-files=no)" ]]; then
    echo "CALVIN runtime has tracked modifications; refusing a non-reproducible install" >&2
    exit 1
fi

if [[ ! -x "$conda_bin" ]]; then
    echo "Conda executable is not available: $conda_bin" >&2
    exit 1
fi
if [[ ! -x "$env_prefix/bin/python" ]]; then
    "$conda_bin" create -y -p "$env_prefix" python=3.8.13 pip=24.0
fi

"$env_prefix/bin/python" -m pip install --requirement "$repo_root/requirements/calvin-eval.lock.txt"
# Prism imports the two pinned source trees directly. Removing legacy editable
# metadata keeps pip's dependency audit scoped to the smaller evaluation client.
"$env_prefix/bin/python" -m pip uninstall -y calvin calvin-env >/dev/null 2>&1 || true
"$env_prefix/bin/python" -m pip check

cd "$repo_root"
PYTHONPATH="$repo_root:$runtime_root/calvin_env:$runtime_root/calvin_models${PYTHONPATH:+:$PYTHONPATH}" "$env_prefix/bin/python" -m experiments.calvin.eval --config experiments/calvin/configs/smoke.yaml
