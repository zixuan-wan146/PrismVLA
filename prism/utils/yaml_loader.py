"""Safe YAML loading with repository-wide duplicate-key rejection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe loader that treats duplicate mapping keys as configuration errors."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise TypeError(f"YAML mapping keys must be strings, got {key!r}")
        if key in mapping:
            raise ValueError(f"duplicate YAML mapping key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_unique_yaml(path: str | Path, *, label: str) -> Any:
    """Load one YAML document safely and reject ambiguous duplicate keys."""

    source = Path(path)
    try:
        return yaml.load(
            source.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid {label} in {source}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid {label} in {source}: {exc}") from exc


__all__ = ["load_unique_yaml"]
