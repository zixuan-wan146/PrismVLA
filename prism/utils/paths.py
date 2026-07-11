from __future__ import annotations

# --- migrated from src/prism/path_utils.py ---
import os
import re
from pathlib import Path
from typing import Any, Mapping


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "prism").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {current}")


def normalize_project_relative_path(value: str | Path, repo_root: str | Path, *, label: str = "path") -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path_inside_project(path, repo_root, label=label).as_posix()
    normalized = Path(os.path.normpath(path.as_posix()))
    if normalized.as_posix() == ".":
        return "."
    if normalized.as_posix().startswith("../") or normalized.as_posix() == "..":
        raise ValueError(f"{label} must be project-relative, got {value!r}")
    return normalized.as_posix()


def project_path(value: str | Path, repo_root: str | Path, *, label: str = "path") -> Path:
    root = Path(repo_root).expanduser().resolve()
    path = Path(value).expanduser()
    if path.is_absolute():
        rel = path_inside_project(path, root, label=label)
        return (root / rel).resolve()
    normalized = normalize_project_relative_path(path, root, label=label)
    return (root / normalized).resolve()


def path_inside_project(path: str | Path, repo_root: str | Path, *, label: str = "path") -> Path:
    root = Path(repo_root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be project-relative, got {path!s}") from exc


def display_project_path(path: str | Path, repo_root: str | Path, *, fallback_name: bool = True) -> str:
    raw_path = Path(path).expanduser()
    root = Path(repo_root).expanduser().resolve()
    try:
        rel = raw_path.resolve().relative_to(root)
    except (OSError, ValueError):
        if raw_path.is_absolute():
            return raw_path.name if fallback_name else "."
        return normalize_project_relative_path(raw_path, root)
    return rel.as_posix() if rel.as_posix() != "." else "."


def sanitize_project_paths(value: Any, repo_root: str | Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_project_paths(item, repo_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_project_paths(item, repo_root) for item in value]
    if isinstance(value, Path):
        return display_project_path(value, repo_root)
    if isinstance(value, str):
        if _looks_like_path(value):
            try:
                return display_project_path(value, repo_root)
            except ValueError:
                return value
        return value
    return value


def _looks_like_path(value: str) -> bool:
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        return False
    if value.startswith(("/", "./", "../", "~")):
        return True
    return "/" in value or "\\" in value

