#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 {libero|calvin} CONFIG [--overrides KEY=VALUE ...]" >&2
    exit 2
fi

benchmark="$1"
config="$2"
shift 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$benchmark" in
    libero)
        python_bin="${PRISM_LIBERO_PYTHON:-$repo_root/../envs/libero/bin/python}"
        ;;
    calvin)
        python_bin="${PRISM_CALVIN_PYTHON:-$repo_root/../envs/calvin/bin/python}"
        ;;
    *)
        echo "unsupported benchmark: $benchmark" >&2
        exit 2
        ;;
esac

if [[ ! -x "$python_bin" ]]; then
    echo "benchmark Python is not executable: $python_bin" >&2
    exit 1
fi

cd "$repo_root"
exec "$python_bin" -m scripts.eval --config "$config" "$@"
