from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from prism.data.token_cache_core import (
    EPISODE_FEATURE_CACHE_FORMAT,
    MEMORY_TOKEN_CACHE_FORMAT,
    TokenCacheDatasetConfig,
)
from prism.data.token_cache_io import (
    _mapping_get_step,
    _pad_bool_sequences,
    _pad_executed_actions,
    _pad_future_actions,
    _pad_long_sequences,
    _pad_token_sequences,
    _require_torch,
    _resize_action_horizon,
    _resolve_episode_feature_shards,
    _resolve_manifest_shards,
    _short_steps_tensor,
    _stack_optional_hidden_states,
    _stack_optional_rank1,
    _stack_optional_rank2,
    _torch_load,
    _validate_episode_feature_cache_manifest,
    _validate_token_cache_manifest,
    flatten_visual_tokens,
    read_token_cache_manifest,
    resolve_token_cache_manifest_path,
)

def _parse_token_cache_normalization(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    normalization = manifest.get("normalization")
    if not isinstance(normalization, Mapping) or not bool(normalization.get("enabled", False)):
        return None
    if normalization.get("type") != "train_split_minmax_to_minus_one_one":
        raise ValueError(f"unsupported token-cache normalization type: {normalization.get('type')!r}")
    stats = normalization.get("stats")
    if not isinstance(stats, Mapping) or not stats:
        raise ValueError("token-cache normalization must contain non-empty stats")
    robot_key = str(normalization.get("robot_key") or next(iter(stats)))
    if robot_key not in stats:
        raise KeyError(f"normalization robot_key {robot_key!r} not found in stats")
    robot_stats = stats[robot_key]
    for group_name in ("observation.state", "action"):
        group = robot_stats.get(group_name)
        if not isinstance(group, Mapping) or "min" not in group or "max" not in group:
            raise ValueError(f"normalization stats missing {group_name}.min/max")
    return {
        "type": normalization["type"],
        "robot_key": robot_key,
        "stats": {str(key): value for key, value in stats.items()},
        "clip_after_normalization": bool(normalization.get("clip_after_normalization", True)),
    }


class MemoryTokenCacheDataset:
    """PyTorch-compatible dataset over replay visual-token cache shards."""

    def __init__(self, manifest_path: str | Path, *, max_samples: int | None = None) -> None:
        self.manifest_path = resolve_token_cache_manifest_path(manifest_path)
        self.manifest = read_token_cache_manifest(self.manifest_path)
        _validate_token_cache_manifest(self.manifest, self.manifest_path)
        self.output_root = self.manifest_path.parent
        self.shards = tuple(_resolve_manifest_shards(self.manifest, self.output_root))
        self.shard_end_indices = tuple(shard.end_index for shard in self.shards)
        self.normalization = _parse_token_cache_normalization(self.manifest)
        self.arm2stats_dict = None if self.normalization is None else dict(self.normalization["stats"])

        sample_count = int(self.manifest["sample_count"])
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive when provided")
            sample_count = min(sample_count, int(max_samples))
        self.sample_count = sample_count
        self.config = TokenCacheDatasetConfig(
            manifest_path=self.manifest_path,
            output_root=self.output_root,
            benchmark=str(self.manifest["benchmark"]),
            sample_count=self.sample_count,
            hidden_dim=int(self.manifest["hidden_dim"]),
            storage_dtype=str(self.manifest["storage_dtype"]),
        )
        self._loaded_shard_index: int | None = None
        self._loaded_shard_samples: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self.sample_count
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        shard_index = bisect_right(self.shard_end_indices, index)
        if shard_index >= len(self.shards):
            raise IndexError(index)
        shard = self.shards[shard_index]
        samples = self._load_shard_samples(shard_index)
        local_index = index - shard.start_index
        sample = normalize_token_cache_sample(samples[local_index])
        return _apply_token_cache_normalization(sample, self.normalization)

    def _load_shard_samples(self, shard_index: int) -> list[dict[str, Any]]:
        if self._loaded_shard_index == shard_index and self._loaded_shard_samples is not None:
            return self._loaded_shard_samples
        shard = self.shards[shard_index]
        payload = _torch_load(shard.path)
        if payload.get("format") != MEMORY_TOKEN_CACHE_FORMAT:
            raise ValueError(f"invalid token cache shard format in {shard.path}")
        samples = list(payload.get("samples", []))
        if len(samples) != shard.sample_count:
            raise ValueError(
                f"shard {shard.path} manifest sample_count={shard.sample_count} "
                f"but file has {len(samples)} samples"
            )
        self._loaded_shard_index = shard_index
        self._loaded_shard_samples = samples
        return samples


class MemoryTokenCacheTrajectoryDataset:
    """Trajectory-window view over a memory-token cache.

    The underlying cache is one row per replan point. This wrapper groups rows
    by episode and returns burn-in + loss windows so recurrent progress memory
    is updated chronologically instead of being reset for random frame batches.
    """

    INDEX_VERSION = 1

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        burnin_replan_steps: int = 8,
        loss_replan_steps: int = 8,
        allow_short_burnin: bool = True,
        action_horizon: int = 32,
        window_stride: int = 1,
        max_samples: int | None = None,
    ) -> None:
        burnin_replan_steps = int(burnin_replan_steps)
        loss_replan_steps = int(loss_replan_steps)
        action_horizon = int(action_horizon)
        window_stride = int(window_stride)
        if burnin_replan_steps < 0:
            raise ValueError("burnin_replan_steps must be non-negative")
        if loss_replan_steps <= 0:
            raise ValueError("loss_replan_steps must be positive")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")

        self.base = MemoryTokenCacheDataset(manifest_path, max_samples=max_samples)
        self.manifest_path = self.base.manifest_path
        self.manifest = self.base.manifest
        self.config = self.base.config
        self.normalization = self.base.normalization
        self.arm2stats_dict = self.base.arm2stats_dict
        self.burnin_replan_steps = burnin_replan_steps
        self.loss_replan_steps = loss_replan_steps
        self.allow_short_burnin = bool(allow_short_burnin)
        self.action_horizon = action_horizon
        self.window_stride = window_stride

        rows = self._load_or_build_trajectory_index()
        if self.base.sample_count < int(self.manifest["sample_count"]):
            rows = [row for row in rows if int(row["sample_index"]) < self.base.sample_count]
        self.windows = tuple(self._build_windows(rows))
        if not self.windows:
            raise ValueError(
                "trajectory token-cache dataset produced no windows; check burnin/loss length and action horizon"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[int(index)]
        return {
            "samples": [self.base[int(sample_index)] for sample_index in window["sample_indices"]],
            "loss_mask": list(window["loss_mask"]),
            "episode_id": str(window["episode_id"]),
            "start_step": int(window["start_step"]),
        }

    def _load_or_build_trajectory_index(self) -> list[dict[str, Any]]:
        index_path = self.manifest_path.parent / "trajectory_index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("rows")
            if (
                payload.get("format") == "memory_token_cache_trajectory_index"
                and int(payload.get("version", -1)) == self.INDEX_VERSION
                and int(payload.get("sample_count", -1)) == int(self.manifest["sample_count"])
                and isinstance(rows, list)
            ):
                return rows

        rows: list[dict[str, Any]] = []
        for shard in self.base.shards:
            payload = _torch_load(shard.path)
            samples = list(payload.get("samples", []))
            if len(samples) != shard.sample_count:
                raise ValueError(f"invalid sample count in shard {shard.path}")
            for local_index, sample in enumerate(samples):
                rows.append(
                    {
                        "sample_index": int(sample.get("sample_index", shard.start_index + local_index)),
                        "benchmark": str(sample["benchmark"]),
                        "episode_id": str(sample["episode_id"]),
                        "current_step": int(sample["current_step"]),
                        "action_valid_count": int(sample["action_valid_count"]),
                    }
                )
        payload = {
            "format": "memory_token_cache_trajectory_index",
            "version": self.INDEX_VERSION,
            "sample_count": int(self.manifest["sample_count"]),
            "rows": rows,
        }
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
        return rows

    def _build_windows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        episodes: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            episodes[(str(row["benchmark"]), str(row["episode_id"]))].append(row)

        windows: list[dict[str, Any]] = []
        for (_benchmark, episode_id), episode_rows in episodes.items():
            ordered = sorted(episode_rows, key=lambda row: (int(row["current_step"]), int(row["sample_index"])))
            valid_loss_positions = {
                pos for pos, row in enumerate(ordered) if int(row["action_valid_count"]) >= self.action_horizon
            }
            max_start = len(ordered) - self.loss_replan_steps
            for start_pos in range(0, max_start + 1, self.window_stride):
                loss_positions = range(start_pos, start_pos + self.loss_replan_steps)
                if any(pos not in valid_loss_positions for pos in loss_positions):
                    continue
                if not self.allow_short_burnin and start_pos < self.burnin_replan_steps:
                    continue
                burnin_start = max(0, start_pos - self.burnin_replan_steps)
                selected = ordered[burnin_start : start_pos + self.loss_replan_steps]
                burnin_count = start_pos - burnin_start
                windows.append(
                    {
                        "episode_id": episode_id,
                        "start_step": int(ordered[start_pos]["current_step"]),
                        "sample_indices": [int(row["sample_index"]) for row in selected],
                        "loss_mask": [False] * burnin_count + [True] * self.loss_replan_steps,
                    }
                )
        return windows


class EpisodeFeatureCacheTrajectoryDataset:
    """Episode-level trajectory view over processed LIBERO feature cache shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        action_horizon: int = 32,
        max_episodes: int | None = None,
    ) -> None:
        self.manifest_path = resolve_token_cache_manifest_path(manifest_path)
        self.manifest = read_token_cache_manifest(self.manifest_path)
        _validate_episode_feature_cache_manifest(self.manifest, self.manifest_path)
        self.output_root = self.manifest_path.parent
        self.shards = tuple(_resolve_episode_feature_shards(self.manifest, self.output_root))
        self.shard_end_indices = tuple(shard.end_index for shard in self.shards)
        self.normalization = _parse_token_cache_normalization(self.manifest)
        self.arm2stats_dict = None if self.normalization is None else dict(self.normalization["stats"])
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        episode_count = int(self.manifest["episode_count"])
        if max_episodes is not None:
            if int(max_episodes) <= 0:
                raise ValueError("max_episodes must be positive when provided")
            episode_count = min(episode_count, int(max_episodes))
        self.episode_count = episode_count
        self.config = TokenCacheDatasetConfig(
            manifest_path=self.manifest_path,
            output_root=self.output_root,
            benchmark=str(self.manifest["benchmark"]),
            sample_count=self.episode_count,
            hidden_dim=int(self.manifest["hidden_dim"]),
            storage_dtype=str(self.manifest["storage_dtype"]),
        )
        self._loaded_shard_index: int | None = None
        self._loaded_shard_episodes: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return self.episode_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self.episode_count
        if index < 0 or index >= self.episode_count:
            raise IndexError(index)
        shard_index = bisect_right(self.shard_end_indices, index)
        if shard_index >= len(self.shards):
            raise IndexError(index)
        shard = self.shards[shard_index]
        episodes = self._load_shard_episodes(shard_index)
        local_index = index - shard.start_index
        episode = episodes[local_index]
        samples = [
            self._node_to_sample(episode, node, sample_index=index * 100000 + node_index)
            for node_index, node in enumerate(episode["nodes"])
        ]
        loss_mask = [
            int(node["action_valid_count"]) >= self.action_horizon
            for node in episode["nodes"]
        ]
        if not any(loss_mask):
            raise ValueError(f"episode {episode.get('episode_id')} has no full-horizon Stage1 loss nodes")
        return {
            "samples": samples,
            "loss_mask": loss_mask,
            "episode_id": str(episode["episode_id"]),
            "start_step": int(episode["nodes"][0]["current_step"]),
        }

    def _load_shard_episodes(self, shard_index: int) -> list[dict[str, Any]]:
        if self._loaded_shard_index == shard_index and self._loaded_shard_episodes is not None:
            return self._loaded_shard_episodes
        shard = self.shards[shard_index]
        payload = _torch_load(shard.path)
        if payload.get("format") != EPISODE_FEATURE_CACHE_FORMAT:
            raise ValueError(f"invalid episode feature cache shard format in {shard.path}")
        episodes = list(payload.get("episodes", []))
        if len(episodes) != shard.sample_count:
            raise ValueError(
                f"shard {shard.path} manifest episode_count={shard.sample_count} "
                f"but file has {len(episodes)} episodes"
            )
        self._loaded_shard_index = shard_index
        self._loaded_shard_episodes = episodes
        return episodes

    def _node_to_sample(
        self,
        episode: Mapping[str, Any],
        node: Mapping[str, Any],
        *,
        sample_index: int,
    ) -> dict[str, Any]:
        torch = _require_torch()
        current_step = int(node["current_step"])
        actions = torch.as_tensor(episode["actions"], dtype=torch.float32).cpu()
        visual_tokens_by_step = episode["visual_tokens_by_step"]
        state_by_step = episode["state_by_step"]
        current_features_by_step = episode["current_features_by_step"]

        short_steps = [None if step is None else int(step) for step in node.get("short_visual_steps", [])]
        short_mask = [bool(value) for value in node.get("short_mask", [])]
        short_tokens = tuple(
            None
            if step is None or index >= len(short_mask) or not short_mask[index]
            else _mapping_get_step(visual_tokens_by_step, int(step), label="visual_tokens_by_step")
            for index, step in enumerate(short_steps)
        )
        executed_start, executed_end = [int(value) for value in node["executed_action_range"]]
        executed_actions, executed_action_mask = _pad_executed_actions(
            actions[executed_start:executed_end],
            valid_count=int(node["executed_action_valid_count"]),
            action_dim=int(actions.shape[-1]),
            target_length=max(1, executed_end - executed_start, int(self.manifest["source_executed_action_stride"])),
        )
        future_start, future_end = [int(value) for value in node["future_action_range"]]
        features = _mapping_get_step(current_features_by_step, current_step, label="current_features_by_step")
        sample = {
            "sample_index": int(sample_index),
            "benchmark": str(self.manifest.get("benchmark", "LIBERO")),
            "episode_id": str(episode["episode_id"]),
            "prompt": str(episode.get("prompt", "")),
            "current_step": current_step,
            "current_tokens_by_view": _mapping_get_step(
                visual_tokens_by_step,
                current_step,
                label="visual_tokens_by_step",
            ),
            "current_state": _mapping_get_step(state_by_step, current_step, label="state_by_step"),
            "short_tokens_by_view": short_tokens,
            "short_steps": [-1 if step is None else int(step) for step in short_steps],
            "short_mask": short_mask,
            "executed_actions": executed_actions,
            "executed_action_mask": executed_action_mask,
            "future_actions": actions[future_start:future_end].contiguous(),
            "action_valid_count": int(node["action_valid_count"]),
            "current_hidden_states": tuple(features["hidden_states"]),
            "planner_vl_summary": features["planner_vl_summary"],
        }
        return _apply_token_cache_normalization(normalize_token_cache_sample(sample), self.normalization)


def collate_direct_bridge_token_cache_windows(
    batch: Sequence[Mapping[str, Any]],
    *,
    memory_entry_tokens: int = 16,
    action_horizon: int | None = None,
    view_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collate episode/node sequences into per-timestep active mini-batches."""

    if not batch:
        raise ValueError("batch must contain at least one episode sequence")
    torch = _require_torch()
    max_length = max(len(window["samples"]) for window in batch)
    steps = []
    for step_index in range(max_length):
        active_samples = []
        batch_indices = []
        loss_mask = []
        for batch_index, window in enumerate(batch):
            samples = list(window["samples"])
            if step_index >= len(samples):
                continue
            active_samples.append(samples[step_index])
            batch_indices.append(batch_index)
            loss_mask.append(bool(window["loss_mask"][step_index]))
        if not active_samples:
            continue
        step_batch = collate_direct_bridge_token_cache_samples(
            active_samples,
            memory_entry_tokens=memory_entry_tokens,
            action_horizon=action_horizon,
            view_names=view_names,
        )
        step_batch["batch_indices"] = torch.tensor(batch_indices, dtype=torch.long)
        step_batch["loss_mask"] = torch.tensor(loss_mask, dtype=torch.bool)
        steps.append(step_batch)
    if not steps:
        raise ValueError("episode sequence batch contains no active steps")
    return {
        "trajectory_steps": steps,
        "batch_size": len(batch),
        "episode_id": [str(window["episode_id"]) for window in batch],
        "start_step": torch.tensor([int(window["start_step"]) for window in batch], dtype=torch.long),
    }


def collate_memory_token_cache_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must contain at least one item")
    torch = _require_torch()
    future_actions, action_mask = _pad_future_actions([sample["future_actions"] for sample in batch])
    output = {
        "benchmark": [str(sample["benchmark"]) for sample in batch],
        "episode_id": [str(sample["episode_id"]) for sample in batch],
        "sample_index": torch.tensor([int(sample["sample_index"]) for sample in batch], dtype=torch.long),
        "current_step": torch.tensor([int(sample["current_step"]) for sample in batch], dtype=torch.long),
        "current_tokens_by_view": [sample["current_tokens_by_view"] for sample in batch],
        "current_state": torch.stack(
            [torch.as_tensor(sample["current_state"], dtype=torch.float32) for sample in batch]
        ),
        "short_tokens_by_view": [sample["short_tokens_by_view"] for sample in batch],
        "short_steps": torch.stack([torch.as_tensor(sample["short_steps"], dtype=torch.long) for sample in batch]),
        "short_mask": torch.stack([torch.as_tensor(sample["short_mask"], dtype=torch.bool) for sample in batch]),
        "future_actions": future_actions,
        "action_mask": action_mask,
        "action_valid_count": torch.tensor([int(sample["action_valid_count"]) for sample in batch], dtype=torch.long),
    }
    executed_actions = _stack_optional_rank2(batch, "executed_actions", dtype=torch.float32)
    executed_action_mask = _stack_optional_rank1(batch, "executed_action_mask", dtype=torch.bool)
    if executed_actions is not None:
        output["executed_actions"] = executed_actions
    if executed_action_mask is not None:
        output["executed_action_mask"] = executed_action_mask
    hidden_states = _stack_optional_hidden_states(batch, "current_hidden_states")
    if hidden_states is not None:
        output["vlm_hidden_states"] = hidden_states
    planner_vl_summary = _stack_optional_rank1(batch, "planner_vl_summary", dtype=torch.float32)
    if planner_vl_summary is not None:
        output["planner_vl_summary"] = planner_vl_summary
    return output


def collate_direct_bridge_token_cache_samples(
    batch: Sequence[Mapping[str, Any]],
    *,
    memory_entry_tokens: int = 16,
    action_horizon: int | None = None,
    view_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collate visual-token cache rows into the direct bridge-attn training contract.

    The cache stores raw visual tokens by view. This collate function builds:

    - ``fused_tokens`` from the current frame visual tokens.
    - ``memory_context`` from fixed-size short-memory entries.
    - ``short_memory_time_ids`` with one id per short-memory entry.
    - per-dimension ``action_mask`` matching ``future_actions``.
    """

    if not batch:
        raise ValueError("batch must contain at least one item")
    memory_entry_tokens = int(memory_entry_tokens)
    if memory_entry_tokens <= 0:
        raise ValueError("memory_entry_tokens must be positive")

    torch = _require_torch()
    current_tokens = []
    memory_context = []
    memory_context_mask = []
    short_time_ids = []

    for sample in batch:
        current = concat_tokens_by_view(sample["current_tokens_by_view"], view_names=view_names)
        current_tokens.append(current)

        sample_short_tokens = []
        sample_short_mask = []
        sample_time_ids = []
        short_entries = tuple(sample.get("short_tokens_by_view", ()))
        short_valid = torch.as_tensor(sample.get("short_mask", [entry is not None for entry in short_entries])).bool()
        if short_valid.numel() != len(short_entries):
            raise ValueError("short_mask length must match short_tokens_by_view length")

        for entry_index, entry in enumerate(short_entries):
            if entry is None or not bool(short_valid[entry_index].item()):
                packed = torch.zeros(memory_entry_tokens, current.shape[-1], dtype=current.dtype)
                packed_mask = torch.zeros(memory_entry_tokens, dtype=torch.bool)
            else:
                packed, packed_mask = pack_visual_tokens(
                    concat_tokens_by_view(entry, view_names=view_names),
                    target_tokens=memory_entry_tokens,
                )
            sample_short_tokens.append(packed)
            sample_short_mask.append(packed_mask)
            sample_time_ids.append(torch.full((memory_entry_tokens,), entry_index, dtype=torch.long))

        if sample_short_tokens:
            memory_context.append(torch.cat(sample_short_tokens, dim=0))
            memory_context_mask.append(torch.cat(sample_short_mask, dim=0))
            short_time_ids.append(torch.cat(sample_time_ids, dim=0))
        else:
            memory_context.append(torch.zeros(0, current.shape[-1], dtype=current.dtype))
            memory_context_mask.append(torch.zeros(0, dtype=torch.bool))
            short_time_ids.append(torch.zeros(0, dtype=torch.long))

    fused_tokens = _pad_token_sequences(current_tokens)
    future_actions, step_mask = _pad_future_actions([sample["future_actions"] for sample in batch])
    if action_horizon is not None:
        future_actions, step_mask = _resize_action_horizon(
            future_actions,
            step_mask,
            action_horizon=int(action_horizon),
        )
    action_mask = step_mask.unsqueeze(-1).expand_as(future_actions).clone()
    executed_actions = _stack_optional_rank2(batch, "executed_actions", dtype=torch.float32)
    executed_action_mask = _stack_optional_rank1(batch, "executed_action_mask", dtype=torch.bool)

    output = {
        "benchmark": [str(sample["benchmark"]) for sample in batch],
        "episode_id": [str(sample["episode_id"]) for sample in batch],
        "sample_index": torch.tensor([int(sample["sample_index"]) for sample in batch], dtype=torch.long),
        "current_step": torch.tensor([int(sample["current_step"]) for sample in batch], dtype=torch.long),
        "fused_tokens": fused_tokens,
        "states": torch.stack([torch.as_tensor(sample["current_state"], dtype=torch.float32) for sample in batch]),
        "actions": future_actions,
        "action_mask": action_mask,
        "memory_context": _pad_token_sequences(memory_context),
        "memory_context_mask": _pad_bool_sequences(memory_context_mask),
        "short_memory_time_ids": _pad_long_sequences(short_time_ids),
        "action_valid_count": torch.tensor([int(sample["action_valid_count"]) for sample in batch], dtype=torch.long),
    }
    if executed_actions is not None:
        output["executed_actions"] = executed_actions
    if executed_action_mask is not None:
        output["executed_action_mask"] = executed_action_mask
    hidden_states = _stack_optional_hidden_states(batch, "current_hidden_states")
    if hidden_states is not None:
        output["vlm_hidden_states"] = hidden_states
    planner_vl_summary = _stack_optional_rank1(batch, "planner_vl_summary", dtype=torch.float32)
    if planner_vl_summary is not None:
        output["planner_vl_summary"] = planner_vl_summary
    return output


def concat_tokens_by_view(tokens_by_view: Mapping[str, Any], *, view_names: Sequence[str] | None = None) -> Any:
    torch = _require_torch()
    if view_names is None:
        view_names = sorted(str(name) for name in tokens_by_view)
    tensors = []
    for view_name in view_names:
        if view_name not in tokens_by_view:
            raise KeyError(f"missing visual tokens for view {view_name!r}")
        tensors.append(flatten_visual_tokens(torch.as_tensor(tokens_by_view[view_name], dtype=torch.float32).cpu()))
    if not tensors:
        raise ValueError("tokens_by_view must contain at least one view")
    hidden_dim = int(tensors[0].shape[-1])
    if any(int(tensor.shape[-1]) != hidden_dim for tensor in tensors):
        raise ValueError("all view token tensors must share hidden dim")
    return torch.cat(tensors, dim=0)


def pack_visual_tokens(tokens: Any, *, target_tokens: int) -> tuple[Any, Any]:
    """Convert an arbitrary token sequence into a fixed-size token entry.

    Long sequences are reduced with deterministic mean pooling over equally
    spaced bins. Short sequences are zero-padded and expose a validity mask.
    """

    torch = _require_torch()
    tokens = flatten_visual_tokens(torch.as_tensor(tokens, dtype=torch.float32).cpu())
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    token_count, hidden_dim = int(tokens.shape[0]), int(tokens.shape[1])
    if token_count <= 0:
        raise ValueError("visual token sequence must be non-empty")

    if token_count == target_tokens:
        return tokens.contiguous(), torch.ones(target_tokens, dtype=torch.bool)
    if token_count < target_tokens:
        output = torch.zeros(target_tokens, hidden_dim, dtype=tokens.dtype)
        output[:token_count] = tokens
        mask = torch.zeros(target_tokens, dtype=torch.bool)
        mask[:token_count] = True
        return output, mask

    boundaries = torch.linspace(0, token_count, steps=target_tokens + 1).round().long()
    output = torch.zeros(target_tokens, hidden_dim, dtype=tokens.dtype)
    for index in range(target_tokens):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            end = min(token_count, start + 1)
        output[index] = tokens[start:end].mean(dim=0)
    return output, torch.ones(target_tokens, dtype=torch.bool)


def normalize_token_cache_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    torch = _require_torch()
    normalized = dict(sample)
    short_tokens = tuple(normalized.get("short_tokens_by_view", ()))
    normalized["short_tokens_by_view"] = short_tokens
    normalized["short_mask"] = torch.as_tensor(normalized["short_mask"], dtype=torch.bool).cpu()
    normalized["short_steps"] = _short_steps_tensor(normalized, len(short_tokens))
    normalized["current_state"] = torch.as_tensor(normalized["current_state"], dtype=torch.float32).cpu()
    if "current_hidden_states" in normalized:
        normalized["current_hidden_states"] = tuple(
            flatten_visual_tokens(torch.as_tensor(hidden_state, dtype=torch.float32).cpu())
            for hidden_state in normalized["current_hidden_states"]
        )
    if "planner_vl_summary" in normalized:
        normalized["planner_vl_summary"] = torch.as_tensor(
            normalized["planner_vl_summary"],
            dtype=torch.float32,
        ).reshape(-1).cpu()
    normalized["future_actions"] = torch.as_tensor(normalized["future_actions"], dtype=torch.float32).cpu()
    if "executed_actions" in normalized:
        normalized["executed_actions"] = torch.as_tensor(normalized["executed_actions"], dtype=torch.float32).cpu()
    if "executed_action_mask" in normalized:
        normalized["executed_action_mask"] = torch.as_tensor(normalized["executed_action_mask"], dtype=torch.bool).cpu()
    normalized["action_valid_count"] = int(normalized["action_valid_count"])
    normalized["current_step"] = int(normalized["current_step"])
    normalized["sample_index"] = int(normalized.get("sample_index", -1))
    normalized["benchmark"] = str(normalized["benchmark"])
    normalized["episode_id"] = str(normalized["episode_id"])
    normalized["prompt"] = str(normalized.get("prompt", ""))
    return normalized


def _apply_token_cache_normalization(
    sample: dict[str, Any],
    normalization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if normalization is None:
        return sample

    torch = _require_torch()
    robot_key = str(normalization["robot_key"])
    stats = normalization["stats"][robot_key]
    clip = bool(normalization.get("clip_after_normalization", True))

    state_min = torch.as_tensor(stats["observation.state"]["min"], dtype=torch.float32)
    state_max = torch.as_tensor(stats["observation.state"]["max"], dtype=torch.float32)
    action_min = torch.as_tensor(stats["action"]["min"], dtype=torch.float32)
    action_max = torch.as_tensor(stats["action"]["max"], dtype=torch.float32)

    sample["current_state"] = _minmax_normalize_tensor(
        torch.as_tensor(sample["current_state"], dtype=torch.float32).cpu(),
        state_min,
        state_max,
        clip=clip,
        name="current_state",
    )
    sample["future_actions"] = _minmax_normalize_tensor(
        torch.as_tensor(sample["future_actions"], dtype=torch.float32).cpu(),
        action_min,
        action_max,
        clip=clip,
        name="future_actions",
    )
    if "executed_actions" in sample:
        executed = _minmax_normalize_tensor(
            torch.as_tensor(sample["executed_actions"], dtype=torch.float32).cpu(),
            action_min,
            action_max,
            clip=clip,
            name="executed_actions",
        )
        if "executed_action_mask" in sample:
            mask = torch.as_tensor(sample["executed_action_mask"], dtype=torch.bool).cpu()
            if mask.shape != executed.shape[:1]:
                raise ValueError(f"executed_action_mask shape {tuple(mask.shape)} does not match executed_actions")
            executed = executed * mask.unsqueeze(-1).to(dtype=executed.dtype)
        sample["executed_actions"] = executed
    return sample


def _minmax_normalize_tensor(
    value: Any,
    min_value: Any,
    max_value: Any,
    *,
    clip: bool,
    name: str,
) -> Any:
    torch = _require_torch()
    tensor = torch.as_tensor(value, dtype=torch.float32).cpu()
    min_tensor = torch.as_tensor(min_value, dtype=torch.float32).cpu()
    max_tensor = torch.as_tensor(max_value, dtype=torch.float32).cpu()
    dim = int(tensor.shape[-1]) if tensor.ndim > 0 else 1
    if dim > int(min_tensor.shape[0]) or dim > int(max_tensor.shape[0]):
        raise ValueError(
            f"{name} dim {dim} exceeds normalization stats dims "
            f"{tuple(min_tensor.shape)}, {tuple(max_tensor.shape)}"
        )
    min_tensor = min_tensor[:dim].to(dtype=tensor.dtype)
    max_tensor = max_tensor[:dim].to(dtype=tensor.dtype)
    normalized = 2.0 * (tensor - min_tensor) / (max_tensor - min_tensor + 1e-8) - 1.0
    if clip:
        normalized = torch.clamp(normalized, -1.0, 1.0)
    return normalized.contiguous()

