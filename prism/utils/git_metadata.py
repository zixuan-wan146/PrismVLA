"""Shared best-effort Git identity used by non-checkpoint run metadata."""

from __future__ import annotations

from pathlib import Path
import subprocess


def collect_optional_git_identity(
    repository_root: Path | None,
) -> dict[str, str | bool | None]:
    """Return commit, branch, and dirty state without requiring a Git checkout."""

    commit = _git_output(repository_root, "rev-parse", "HEAD")
    branch = _git_output(repository_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output(repository_root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def _git_output(repository_root: Path | None, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


__all__ = ["collect_optional_git_identity"]
