from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prism.data.token_cache_core import (
    EPISODE_FEATURE_CACHE_FORMAT,
    EPISODE_FEATURE_CACHE_VERSION,
    MEMORY_TOKEN_CACHE_FORMAT,
    MEMORY_TOKEN_CACHE_VERSION,
    TokenCacheShard,
)

def ensure_rank2_tokens(tokens: Any, *, storage_dtype: Any) -> Any:
    torch = _require_torch()
    tensor = torch.as_tensor(tokens).detach().cpu()
    tensor = flatten_visual_tokens(tensor)
    if tensor.ndim != 2:
        raise ValueError(f"visual tokens must have shape [num_tokens, hidden_dim], got {tuple(tensor.shape)}")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"visual tokens must be non-empty, got {tuple(tensor.shape)}")
    return tensor.to(dtype=storage_dtype).contiguous()


def flatten_visual_tokens(tokens: Any) -> Any:
    torch = _require_torch()
    tensor = torch.as_tensor(tokens)
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
    if tensor.ndim == 4:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1] * tensor.shape[2], tensor.shape[3])
    raise ValueError(f"unsupported visual token tensor rank: {tensor.ndim}")


def read_token_cache_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = resolve_token_cache_manifest_path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_token_cache_manifest_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        resolved = resolved / "manifest.json"
    return resolved


def resolve_torch_dtype(name: str) -> Any:
    torch = _require_torch()
    normalized = str(name).lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported storage dtype: {name!r}")


def _write_token_cache_shard(shard_dir: Path, samples: Sequence[Mapping[str, Any]], *, start_index: int) -> TokenCacheShard:
    torch = _require_torch()
    if not samples:
        raise ValueError("cannot write an empty token cache shard")
    end_index = int(start_index) + len(samples)
    shard_path = shard_dir / f"shard_{int(start_index):09d}_{end_index:09d}.pt"
    torch.save(
        {"format": MEMORY_TOKEN_CACHE_FORMAT, "version": MEMORY_TOKEN_CACHE_VERSION, "samples": list(samples)},
        shard_path,
    )
    return TokenCacheShard(
        path=shard_path,
        sample_count=len(samples),
        start_index=int(start_index),
        end_index=end_index,
    )


def _serialize_layer_selector(layer: int | str) -> int | str:
    if isinstance(layer, (int, np.integer)):
        return int(layer)
    text = str(layer)
    try:
        return int(text)
    except ValueError:
        return text


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("memory token cache utilities require torch") from exc
    return torch


def _short_steps_tensor(item: Mapping[str, Any], short_count: int) -> Any:
    torch = _require_torch()
    if "short_steps" not in item:
        return torch.full((int(short_count),), -1, dtype=torch.long)
    raw_steps = item["short_steps"]
    if isinstance(raw_steps, torch.Tensor):
        steps = raw_steps.to(dtype=torch.long).reshape(-1).cpu()
    else:
        steps = torch.as_tensor(raw_steps, dtype=torch.long).reshape(-1).cpu()
    if steps.numel() != int(short_count):
        raise ValueError(f"short_steps has {steps.numel()} values for {short_count} short entries")
    return steps


def _validate_token_cache_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    if manifest.get("format") != MEMORY_TOKEN_CACHE_FORMAT:
        raise ValueError(f"invalid token cache format in {manifest_path}: {manifest.get('format')!r}")
    if int(manifest.get("version", -1)) != MEMORY_TOKEN_CACHE_VERSION:
        raise ValueError(f"unsupported token cache version in {manifest_path}: {manifest.get('version')!r}")
    if int(manifest.get("sample_count", -1)) < 0:
        raise ValueError(f"token cache manifest has invalid sample_count: {manifest.get('sample_count')!r}")
    if int(manifest.get("hidden_dim", 0)) <= 0:
        raise ValueError(f"token cache manifest has invalid hidden_dim: {manifest.get('hidden_dim')!r}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"token cache manifest has no shards: {manifest_path}")


def _validate_episode_feature_cache_manifest(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    if manifest.get("format") != EPISODE_FEATURE_CACHE_FORMAT:
        raise ValueError(f"invalid episode feature cache format in {manifest_path}: {manifest.get('format')!r}")
    if int(manifest.get("version", -1)) != EPISODE_FEATURE_CACHE_VERSION:
        raise ValueError(f"unsupported episode feature cache version in {manifest_path}: {manifest.get('version')!r}")
    if not str(manifest.get("benchmark", "")).strip():
        raise ValueError(f"episode feature cache manifest must include benchmark: {manifest_path}")
    if int(manifest.get("episode_count", -1)) < 0:
        raise ValueError(f"episode feature cache manifest has invalid episode_count: {manifest.get('episode_count')!r}")
    if int(manifest.get("node_count", -1)) < 0:
        raise ValueError(f"episode feature cache manifest has invalid node_count: {manifest.get('node_count')!r}")
    if int(manifest.get("hidden_dim", 0)) <= 0:
        raise ValueError(f"episode feature cache manifest has invalid hidden_dim: {manifest.get('hidden_dim')!r}")
    if int(manifest.get("source_executed_action_stride", 0)) <= 0:
        raise ValueError("episode feature cache manifest must include source_executed_action_stride")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"episode feature cache manifest has no shards: {manifest_path}")


def _resolve_manifest_shards(manifest: Mapping[str, Any], output_root: Path) -> list[TokenCacheShard]:
    shards: list[TokenCacheShard] = []
    expected_start = 0
    for raw_shard in manifest["shards"]:
        shard = TokenCacheShard(
            path=output_root / str(raw_shard["path"]),
            sample_count=int(raw_shard["sample_count"]),
            start_index=int(raw_shard["start_index"]),
            end_index=int(raw_shard["end_index"]),
        )
        if shard.start_index != expected_start:
            raise ValueError(
                f"non-contiguous token cache shard starts at {shard.start_index}, expected {expected_start}"
            )
        if shard.end_index - shard.start_index != shard.sample_count:
            raise ValueError(f"token cache shard has inconsistent index range: {shard}")
        if not shard.path.exists():
            raise FileNotFoundError(f"token cache shard does not exist: {shard.path}")
        shards.append(shard)
        expected_start = shard.end_index
    if expected_start != int(manifest["sample_count"]):
        raise ValueError(f"manifest sample_count={manifest['sample_count']} but shards end at {expected_start}")
    return shards


def _resolve_episode_feature_shards(manifest: Mapping[str, Any], output_root: Path) -> list[TokenCacheShard]:
    shards: list[TokenCacheShard] = []
    expected_start = 0
    for raw_shard in manifest["shards"]:
        shard = TokenCacheShard(
            path=output_root / str(raw_shard["path"]),
            sample_count=int(raw_shard["episode_count"]),
            start_index=int(raw_shard["start_index"]),
            end_index=int(raw_shard["end_index"]),
        )
        if shard.start_index != expected_start:
            raise ValueError(
                f"non-contiguous episode feature shard starts at {shard.start_index}, expected {expected_start}"
            )
        if shard.end_index - shard.start_index != shard.sample_count:
            raise ValueError(f"episode feature shard has inconsistent index range: {shard}")
        if not shard.path.exists():
            raise FileNotFoundError(f"episode feature shard does not exist: {shard.path}")
        shards.append(shard)
        expected_start = shard.end_index
    if expected_start != int(manifest["episode_count"]):
        raise ValueError(f"manifest episode_count={manifest['episode_count']} but shards end at {expected_start}")
    return shards


def _mapping_get_step(mapping: Mapping[Any, Any], step: int, *, label: str) -> Any:
    if step in mapping:
        return mapping[step]
    text_step = str(step)
    if text_step in mapping:
        return mapping[text_step]
    raise KeyError(f"{label} missing step {step}")


def _pad_executed_actions(
    actions: Any,
    *,
    valid_count: int,
    action_dim: int,
    target_length: int,
) -> tuple[Any, Any]:
    torch = _require_torch()
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    action_dim = int(action_dim)
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    tensor = torch.as_tensor(actions, dtype=torch.float32).reshape(-1, action_dim).cpu()
    valid = min(max(int(valid_count), 0), int(tensor.shape[0]), target_length)
    output = torch.zeros(target_length, action_dim, dtype=torch.float32)
    mask = torch.zeros(target_length, dtype=torch.bool)
    if valid > 0:
        output[-valid:] = tensor[-valid:]
        mask[-valid:] = True
    return output, mask


def _pad_future_actions(actions: Sequence[Any]) -> tuple[Any, Any]:
    torch = _require_torch()
    tensors = [torch.as_tensor(action, dtype=torch.float32).cpu() for action in actions]
    if not tensors:
        raise ValueError("actions must not be empty")
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("future_actions must have shape [T, A]")
    action_dim = int(tensors[0].shape[-1])
    if any(int(tensor.shape[-1]) != action_dim for tensor in tensors):
        raise ValueError("all future_actions tensors in a batch must share action dim")
    max_steps = max(int(tensor.shape[0]) for tensor in tensors)
    batch = torch.zeros(len(tensors), max_steps, action_dim, dtype=torch.float32)
    mask = torch.zeros(len(tensors), max_steps, dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        step_count = int(tensor.shape[0])
        batch[index, :step_count] = tensor
        mask[index, :step_count] = True
    return batch, mask


def _stack_optional_rank2(batch: Sequence[Mapping[str, Any]], key: str, *, dtype: Any) -> Any | None:
    if not all(key in sample for sample in batch):
        return None
    torch = _require_torch()
    tensors = [torch.as_tensor(sample[key], dtype=dtype).cpu() for sample in batch]
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError(f"{key} must have shape [T, D]")
    expected_shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != expected_shape for tensor in tensors):
        raise ValueError(f"all {key} tensors in a batch must share shape")
    return torch.stack(tensors, dim=0)


def _stack_optional_rank1(batch: Sequence[Mapping[str, Any]], key: str, *, dtype: Any) -> Any | None:
    if not all(key in sample for sample in batch):
        return None
    torch = _require_torch()
    tensors = [torch.as_tensor(sample[key], dtype=dtype).reshape(-1).cpu() for sample in batch]
    expected_shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != expected_shape for tensor in tensors):
        raise ValueError(f"all {key} tensors in a batch must share shape")
    return torch.stack(tensors, dim=0)


def _stack_optional_hidden_states(batch: Sequence[Mapping[str, Any]], key: str) -> list[Any] | None:
    if not all(key in sample for sample in batch):
        return None
    torch = _require_torch()
    per_sample = [tuple(sample[key]) for sample in batch]
    layer_count = len(per_sample[0])
    if layer_count <= 0:
        raise ValueError(f"{key} must contain at least one hidden-state layer")
    if any(len(sample_layers) != layer_count for sample_layers in per_sample):
        raise ValueError(f"all samples must provide the same number of {key} layers")
    hidden_states = []
    for layer_index in range(layer_count):
        hidden_states.append(
            _pad_token_sequences(
                [
                    flatten_visual_tokens(torch.as_tensor(sample_layers[layer_index], dtype=torch.float32).cpu())
                    for sample_layers in per_sample
                ]
            )
        )
    return hidden_states


def _resize_action_horizon(actions: Any, step_mask: Any, *, action_horizon: int) -> tuple[Any, Any]:
    torch = _require_torch()
    action_horizon = int(action_horizon)
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if actions.shape[1] == action_horizon:
        return actions, step_mask
    if actions.shape[1] > action_horizon:
        return actions[:, :action_horizon].contiguous(), step_mask[:, :action_horizon].contiguous()
    padded_actions = torch.zeros(actions.shape[0], action_horizon, actions.shape[2], dtype=actions.dtype)
    padded_mask = torch.zeros(step_mask.shape[0], action_horizon, dtype=step_mask.dtype)
    padded_actions[:, : actions.shape[1]] = actions
    padded_mask[:, : step_mask.shape[1]] = step_mask
    return padded_actions, padded_mask


def _pad_token_sequences(sequences: Sequence[Any]) -> Any:
    torch = _require_torch()
    tensors = [flatten_visual_tokens(torch.as_tensor(sequence, dtype=torch.float32).cpu()) for sequence in sequences]
    if not tensors:
        raise ValueError("sequences must not be empty")
    hidden_dim = int(tensors[0].shape[-1])
    if any(int(tensor.shape[-1]) != hidden_dim for tensor in tensors):
        raise ValueError("all token sequences in a batch must share hidden dim")
    max_tokens = max(int(tensor.shape[0]) for tensor in tensors)
    output = torch.zeros(len(tensors), max_tokens, hidden_dim, dtype=torch.float32)
    for index, tensor in enumerate(tensors):
        output[index, : tensor.shape[0]] = tensor
    return output


def _pad_bool_sequences(sequences: Sequence[Any]) -> Any:
    torch = _require_torch()
    tensors = [torch.as_tensor(sequence, dtype=torch.bool).reshape(-1).cpu() for sequence in sequences]
    if not tensors:
        raise ValueError("sequences must not be empty")
    max_tokens = max(int(tensor.numel()) for tensor in tensors)
    output = torch.zeros(len(tensors), max_tokens, dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        output[index, : tensor.numel()] = tensor
    return output


def _pad_long_sequences(sequences: Sequence[Any]) -> Any:
    torch = _require_torch()
    tensors = [torch.as_tensor(sequence, dtype=torch.long).reshape(-1).cpu() for sequence in sequences]
    if not tensors:
        raise ValueError("sequences must not be empty")
    max_tokens = max(int(tensor.numel()) for tensor in tensors)
    output = torch.zeros(len(tensors), max_tokens, dtype=torch.long)
    for index, tensor in enumerate(tensors):
        output[index, : tensor.numel()] = tensor
    return output


def _torch_load(path: str | Path) -> Any:
    torch = _require_torch()
    try:
        return torch.load(path, weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError):
        return torch.load(path, weights_only=False)




