from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from prism.utils.git_metadata import collect_optional_git_identity
from prism.utils.paths import display_project_path, sanitize_project_paths


def build_run_metadata(
    *,
    repo_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    repo_path = Path(repo_root).expanduser().resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    environ = os.environ if environ is None else environ
    argv = sys.argv if argv is None else argv
    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    argv_items = [str(item) for item in argv]
    sanitized_argv = sanitize_project_paths(argv_items, repo_path)
    git_identity = collect_optional_git_identity(repo_path)
    return {
        "created_at_utc": created_at_utc,
        "cwd": display_project_path(Path.cwd(), repo_path),
        "argv": sanitized_argv,
        "command": " ".join(str(item) for item in sanitized_argv),
        "python": {
            "executable": display_project_path(sys.executable, repo_path),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git": {
            "repo_root": ".",
            "commit": git_identity["commit"],
            "branch": git_identity["branch"],
            "is_dirty": git_identity["dirty"],
        },
        "environment": _safe_environment(environ, repo_path),
    }

def _safe_environment(environ: Mapping[str, str], repo_root: Path) -> dict[str, str]:
    allowed_exact = {
        "HF_ENDPOINT",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "LIBERO_DATASETS_DIR",
        "LIBERO_ENV_PREFIX",
        "LIBERO_PYTHON",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
    }
    allowed_prefixes = ("PRISM_",)
    blocked_fragments = ("TOKEN", "SECRET", "PASSWORD", "KEY")

    safe_items = {}
    for key, value in environ.items():
        if any(fragment in key.upper() for fragment in blocked_fragments):
            continue
        if key in allowed_exact or any(key.startswith(prefix) for prefix in allowed_prefixes):
            safe_items[key] = str(sanitize_project_paths(str(value), repo_root))
    return dict(sorted(safe_items.items()))
