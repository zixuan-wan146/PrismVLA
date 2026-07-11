from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MEMORY_TOKEN_CACHE_FORMAT = "memory_replay_visual_token_cache"
MEMORY_TOKEN_CACHE_VERSION = 1
EPISODE_FEATURE_CACHE_FORMAT = "libero_episode_feature_cache"
EPISODE_FEATURE_CACHE_VERSION = 1
DEFAULT_TOKEN_CACHE_SHARD_SIZE = 1024


@dataclass(frozen=True)
class TokenCacheShard:
    path: Path
    sample_count: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class TokenCacheBuildResult:
    output_root: Path
    manifest_path: Path
    sample_count: int
    shards: tuple[TokenCacheShard, ...]


@dataclass(frozen=True)
class TokenCacheDatasetConfig:
    manifest_path: Path
    output_root: Path
    benchmark: str
    sample_count: int
    hidden_dim: int
    storage_dtype: str


@dataclass(frozen=True)
class VLMCurrentFeatures:
    hidden_states: tuple[Any, ...]
    planner_vl_summary: Any | None = None


