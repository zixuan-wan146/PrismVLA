from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from prism.data.replay_dataset import MemoryReplayFrameDataset
from prism.data.token_cache_core import (
    DEFAULT_TOKEN_CACHE_SHARD_SIZE,
    MEMORY_TOKEN_CACHE_FORMAT,
    MEMORY_TOKEN_CACHE_VERSION,
    TokenCacheBuildResult,
    TokenCacheShard,
    VLMCurrentFeatures,
)
from prism.data.token_cache_io import (
    _require_torch,
    _serialize_layer_selector,
    _short_steps_tensor,
    _write_token_cache_shard,
    ensure_rank2_tokens,
    flatten_visual_tokens,
    resolve_torch_dtype,
)

class VisualTokenEncoder(Protocol):
    """Encode one RGB image into visual tokens with shape [num_tokens, hidden_dim]."""

    name: str
    hidden_dim: int
    tokens_per_view: int | None

    def encode_image(self, image: Image.Image) -> Any:
        ...

    def encode_images(self, images: Sequence[Image.Image]) -> Sequence[Any]:
        ...


class VLMHiddenStateEncoder(Protocol):
    """Encode current observation and prompt into selected VLM hidden-state layers."""

    name: str
    hidden_dim: int
    selected_layers: tuple[int | str, ...]

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> tuple[Any, ...]:
        ...


class ImageStatsVisualTokenEncoder:
    """Small deterministic encoder used for tests and pipeline smoke checks.

    This encoder is intentionally not a training feature extractor. It lets the
    replay-cache IO path run without model downloads or GPU allocation.
    """

    name = "image_stats"

    def __init__(self, *, hidden_dim: int = 16, tokens_per_view: int = 1) -> None:
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(tokens_per_view) <= 0:
            raise ValueError("tokens_per_view must be positive")
        self.hidden_dim = int(hidden_dim)
        self.tokens_per_view = int(tokens_per_view)

    def encode_image(self, image: Image.Image) -> Any:
        torch = _require_torch()
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        flat = rgb.reshape(-1, 3)
        stats = np.concatenate(
            [
                flat.mean(axis=0),
                flat.std(axis=0),
                flat.min(axis=0),
                flat.max(axis=0),
            ],
            axis=0,
        )
        values = np.resize(stats, self.hidden_dim * self.tokens_per_view).reshape(
            self.tokens_per_view,
            self.hidden_dim,
        )
        return torch.tensor(values, dtype=torch.float32)

    def encode_images(self, images: Sequence[Image.Image]) -> list[Any]:
        return [self.encode_image(image) for image in images]


class ImageStatsVLMHiddenStateEncoder:
    """Deterministic hidden-state stand-in for cache IO tests."""

    name = "image_stats_vlm_hidden_states"

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        tokens_per_view: int = 1,
        selected_layers: Sequence[int | str] = (3, 6, 9, 12),
    ) -> None:
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(tokens_per_view) <= 0:
            raise ValueError("tokens_per_view must be positive")
        self.hidden_dim = int(hidden_dim)
        self.tokens_per_view = int(tokens_per_view)
        self.selected_layers = tuple(selected_layers)
        self._visual = ImageStatsVisualTokenEncoder(hidden_dim=self.hidden_dim, tokens_per_view=self.tokens_per_view)

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> tuple[Any, ...]:
        return self.encode_current_features(images_by_view, prompt).hidden_states

    def encode_current_features(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> VLMCurrentFeatures:
        torch = _require_torch()
        base_tokens = torch.cat(
            [self._visual.encode_image(image) for _view_name, image in sorted(images_by_view.items())],
            dim=0,
        )
        prompt_offset = min(len(str(prompt)), 512) / 512.0
        hidden_states = tuple(
            base_tokens + (layer_index + 1) * 0.01 + prompt_offset
            for layer_index, _ in enumerate(self.selected_layers)
        )
        return VLMCurrentFeatures(
            hidden_states=hidden_states,
            planner_vl_summary=hidden_states[-1][-1].to(dtype=torch.float32),
        )


class InternVL3VisualTokenEncoder:
    """InternVL3 visual-tower encoder for replay visual token caches."""

    name = "internvl3"
    tokens_per_view = None

    def __init__(
        self,
        *,
        model_name: str = "OpenGVLab/InternVL3-1B",
        image_size: int = 448,
        device: str = "cuda",
        storage_dtype: str = "bfloat16",
    ) -> None:
        torch = _require_torch()
        from prism.models.vlm import InternVL3Embedder

        self.embedder = InternVL3Embedder(model_name=model_name, image_size=image_size, device=device)
        self.embedder.eval()
        self.device = str(device)
        self.storage_dtype = resolve_torch_dtype(storage_dtype)
        self.hidden_dim = int(getattr(self.embedder.model, "llm_hidden_size", 0) or 0)
        if self.hidden_dim <= 0:
            with torch.no_grad():
                tokens = self.encode_image(Image.new("RGB", (image_size, image_size)))
            self.hidden_dim = int(tokens.shape[-1])

    def encode_image(self, image: Image.Image) -> Any:
        torch = _require_torch()
        with torch.no_grad():
            pixel_values, _num_tiles = self.embedder._preprocess_images([image])
            tokens = self.embedder.model.extract_feature(pixel_values)
        tokens = flatten_visual_tokens(tokens).to(dtype=self.storage_dtype).cpu()
        if tokens.ndim != 2:
            raise ValueError(
                f"InternVL3 visual tokens must be rank-2 after flattening, got {tuple(tokens.shape)}"
            )
        return tokens

    def encode_images(self, images: Sequence[Image.Image]) -> list[Any]:
        torch = _require_torch()
        images = list(images)
        if not images:
            return []
        with torch.no_grad():
            pixel_values, num_tiles_list = self.embedder._preprocess_images(images)
            tokens = self.embedder.model.extract_feature(pixel_values)
        token_tensor = torch.as_tensor(tokens)
        encoded: list[Any] = []
        cursor = 0
        for tile_count in num_tiles_list:
            tile_count = int(tile_count)
            image_tokens = token_tensor[cursor : cursor + tile_count]
            encoded.append(flatten_visual_tokens(image_tokens).to(dtype=self.storage_dtype).cpu())
            cursor += tile_count
        if cursor != int(token_tensor.shape[0]):
            raise ValueError(
                f"InternVL3 visual batch split consumed {cursor} tiles, "
                f"but encoder returned {int(token_tensor.shape[0])}"
            )
        return encoded


class InternVL3VLMHiddenStateEncoder:
    """InternVL3 language-model hidden-state encoder for current replay observations."""

    name = "internvl3_vlm_hidden_states"

    def __init__(
        self,
        *,
        model_name: str = "OpenGVLab/InternVL3-1B",
        image_size: int = 448,
        device: str = "cuda",
        storage_dtype: str = "bfloat16",
        selected_layers: Sequence[int | str] = (3, 6, 9, 12),
        embedder: Any | None = None,
    ) -> None:
        from prism.models.vlm import InternVL3Embedder

        self.embedder = embedder or InternVL3Embedder(model_name=model_name, image_size=image_size, device=device)
        self.embedder.eval()
        self.device = str(device)
        self.storage_dtype = resolve_torch_dtype(storage_dtype)
        self.selected_layers = tuple(selected_layers)
        self.hidden_dim = int(getattr(self.embedder.model, "llm_hidden_size", 0) or 0)

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> tuple[Any, ...]:
        return self.encode_current_features(images_by_view, prompt).hidden_states

    def encode_current_features(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> VLMCurrentFeatures:
        torch = _require_torch()
        images = list(images_by_view.values())
        if not images:
            raise ValueError("images_by_view must contain at least one image")
        image_mask = torch.ones(len(images), dtype=torch.bool)
        with torch.no_grad():
            output = self.embedder.get_fused_image_text_embedding_from_tensor_images(
                image_tensors=images,
                image_mask=image_mask,
                text_prompt=str(prompt),
                return_cls_only=False,
                return_hidden_states=True,
                selected_layers=self.selected_layers,
            )
        hidden_states = tuple(
            flatten_visual_tokens(hidden_state).to(dtype=self.storage_dtype).cpu()
            for hidden_state in output.hidden_states
        )
        if not hidden_states:
            raise ValueError("InternVL3 returned no hidden states")
        if self.hidden_dim <= 0:
            self.hidden_dim = int(hidden_states[0].shape[-1])
        planner_vl_summary = getattr(output, "planner_vl_summary", None)
        if planner_vl_summary is not None:
            planner_vl_summary = torch.as_tensor(planner_vl_summary).reshape(-1, self.hidden_dim)[0].to(
                dtype=self.storage_dtype
            ).cpu()
        return VLMCurrentFeatures(hidden_states=hidden_states, planner_vl_summary=planner_vl_summary)


def build_memory_replay_token_cache(
    *,
    benchmark: str,
    data_root: str | Path,
    index_path: str | Path,
    output_root: str | Path,
    encoder: VisualTokenEncoder,
    hidden_state_encoder: VLMHiddenStateEncoder | None = None,
    view_names: Sequence[str] | None = None,
    max_samples: int | None = None,
    max_samples_per_shard: int = DEFAULT_TOKEN_CACHE_SHARD_SIZE,
    storage_dtype: str = "bfloat16",
    manifest_extra: Mapping[str, Any] | None = None,
) -> TokenCacheBuildResult:
    if str(benchmark).upper() == "LIBERO":
        raise ValueError(
            "LIBERO Stage1 no longer uses memory_replay_visual_token_cache. "
            "Build the active cache with build_libero_episode_replay_index.py followed by "
            "build_libero_episode_feature_cache.py."
        )
    max_samples_per_shard = int(max_samples_per_shard)
    if max_samples_per_shard <= 0:
        raise ValueError("max_samples_per_shard must be positive")

    dataset = MemoryReplayFrameDataset(
        benchmark=benchmark,
        data_root=data_root,
        index_path=index_path,
        view_names=view_names,
        max_samples=max_samples,
    )
    output_path = Path(output_root).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    target_dtype = resolve_torch_dtype(storage_dtype)
    pending: list[dict[str, Any]] = []
    shards: list[TokenCacheShard] = []
    sample_count = 0
    visual_token_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    hidden_state_cache: dict[tuple[str, str, int, str], VLMCurrentFeatures] = {}
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None

    for dataset_index in range(len(dataset)):
        item = dataset[dataset_index]
        action_min, action_max = _update_running_minmax(action_min, action_max, item["future_actions"], name="future_actions")
        state_min, state_max = _update_running_minmax(state_min, state_max, item["current_state"], name="current_state")
        pending.append(
            encode_memory_replay_item(
                item,
                encoder=encoder,
                hidden_state_encoder=hidden_state_encoder,
                storage_dtype=target_dtype,
                sample_index=dataset_index,
                visual_token_cache=visual_token_cache,
                hidden_state_cache=hidden_state_cache,
            )
        )
        sample_count += 1
        if len(pending) >= max_samples_per_shard:
            shards.append(_write_token_cache_shard(shard_dir, pending, start_index=sample_count - len(pending)))
            pending = []

    if pending:
        shards.append(_write_token_cache_shard(shard_dir, pending, start_index=sample_count - len(pending)))

    extra = dict(manifest_extra or {})
    extra.setdefault("builder_mode", "frame_token_dedup")
    extra["visual_token_cache_entries"] = len(visual_token_cache)
    if hidden_state_encoder is not None:
        extra["hidden_state_cache_entries"] = len(hidden_state_cache)
    if action_min is not None and action_max is not None and state_min is not None and state_max is not None:
        extra["normalization"] = _build_minmax_normalization_manifest(
            benchmark=benchmark,
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
        )
        extra["action_normalization"] = {
            "enabled": True,
            "type": "train_split_minmax_to_minus_one_one",
            "clip_after_normalization": True,
            "clip_range": [-1.0, 1.0],
            "statistics_from": "cache_build_rows",
        }

    manifest = build_token_cache_manifest(
        benchmark=benchmark,
        data_root=data_root,
        index_path=index_path,
        output_root=output_path,
        encoder=encoder,
        hidden_state_encoder=hidden_state_encoder,
        storage_dtype=storage_dtype,
        sample_count=sample_count,
        max_samples=max_samples,
        max_samples_per_shard=max_samples_per_shard,
        view_names=view_names,
        shards=shards,
        extra=extra,
    )
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return TokenCacheBuildResult(
        output_root=output_path,
        manifest_path=manifest_path,
        sample_count=sample_count,
        shards=tuple(shards),
    )


def encode_memory_replay_item(
    item: Mapping[str, Any],
    *,
    encoder: VisualTokenEncoder,
    hidden_state_encoder: VLMHiddenStateEncoder | None = None,
    storage_dtype: Any,
    sample_index: int,
    visual_token_cache: dict[tuple[str, str, int], dict[str, Any]] | None = None,
    hidden_state_cache: dict[tuple[str, str, int, str], VLMCurrentFeatures] | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    benchmark = str(item["benchmark"])
    episode_id = str(item["episode_id"])
    current_step = int(item["current_step"])
    current_tokens = _get_or_encode_frame_tokens(
        item["current_images"],
        cache_key=(benchmark, episode_id, current_step),
        encoder=encoder,
        storage_dtype=storage_dtype,
        visual_token_cache=visual_token_cache,
    )
    short_entries = tuple(item["short_images"])
    short_steps = _short_steps_tensor(item, len(short_entries))
    short_tokens = tuple(
        None
        if images_by_view is None or int(short_steps[entry_index].item()) < 0
        else _get_or_encode_frame_tokens(
            images_by_view,
            cache_key=(benchmark, episode_id, int(short_steps[entry_index].item())),
            encoder=encoder,
            storage_dtype=storage_dtype,
            visual_token_cache=visual_token_cache,
        )
        for entry_index, images_by_view in enumerate(short_entries)
    )
    encoded = {
        "sample_index": int(sample_index),
        "benchmark": benchmark,
        "episode_id": episode_id,
        "prompt": str(item.get("prompt", "")),
        "current_step": current_step,
        "current_tokens_by_view": current_tokens,
        "current_state": torch.as_tensor(item["current_state"], dtype=torch.float32).cpu(),
        "short_tokens_by_view": short_tokens,
        "short_steps": short_steps,
        "short_mask": torch.as_tensor(item["short_mask"], dtype=torch.bool).cpu(),
        "executed_actions": torch.as_tensor(item["executed_actions"], dtype=torch.float32).cpu(),
        "executed_action_mask": torch.as_tensor(item["executed_action_mask"], dtype=torch.bool).cpu(),
        "future_actions": torch.as_tensor(item["future_actions"], dtype=torch.float32).cpu(),
        "action_valid_count": int(item["action_valid_count"]),
    }
    if hidden_state_encoder is not None:
        current_features = _get_or_encode_current_features(
            item["current_images"],
            prompt=str(item.get("prompt", "")),
            cache_key=(benchmark, episode_id, current_step, str(item.get("prompt", ""))),
            hidden_state_encoder=hidden_state_encoder,
            storage_dtype=storage_dtype,
            hidden_state_cache=hidden_state_cache,
        )
        encoded["current_hidden_states"] = current_features.hidden_states
        if current_features.planner_vl_summary is not None:
            encoded["planner_vl_summary"] = torch.as_tensor(
                current_features.planner_vl_summary,
                dtype=storage_dtype,
            ).cpu()
    return encoded


def _get_or_encode_frame_tokens(
    images_by_view: Mapping[str, Image.Image],
    *,
    cache_key: tuple[str, str, int],
    encoder: VisualTokenEncoder,
    storage_dtype: Any,
    visual_token_cache: dict[tuple[str, str, int], dict[str, Any]] | None,
) -> dict[str, Any]:
    if visual_token_cache is not None and cache_key in visual_token_cache:
        return visual_token_cache[cache_key]
    tokens = encode_images_by_view(images_by_view, encoder=encoder, storage_dtype=storage_dtype)
    if visual_token_cache is not None:
        visual_token_cache[cache_key] = tokens
    return tokens


def _get_or_encode_current_features(
    images_by_view: Mapping[str, Image.Image],
    *,
    prompt: str,
    cache_key: tuple[str, str, int, str],
    hidden_state_encoder: VLMHiddenStateEncoder,
    storage_dtype: Any,
    hidden_state_cache: dict[tuple[str, str, int, str], VLMCurrentFeatures] | None,
) -> VLMCurrentFeatures:
    if hidden_state_cache is not None and cache_key in hidden_state_cache:
        return hidden_state_cache[cache_key]
    if hasattr(hidden_state_encoder, "encode_current_features"):
        raw_features = hidden_state_encoder.encode_current_features(images_by_view, prompt)
        raw_hidden_states = raw_features.hidden_states
        raw_planner_vl_summary = raw_features.planner_vl_summary
    else:
        raw_hidden_states = hidden_state_encoder.encode_current(images_by_view, prompt)
        raw_planner_vl_summary = None
    hidden_states = tuple(
        ensure_rank2_tokens(hidden_state, storage_dtype=storage_dtype)
        for hidden_state in raw_hidden_states
    )
    planner_vl_summary = None
    if raw_planner_vl_summary is not None:
        torch = _require_torch()
        planner_vl_summary = torch.as_tensor(raw_planner_vl_summary).detach().cpu().reshape(-1).to(dtype=storage_dtype)
        if planner_vl_summary.numel() <= 0:
            raise ValueError("planner_vl_summary must not be empty")
    features = VLMCurrentFeatures(hidden_states=hidden_states, planner_vl_summary=planner_vl_summary)
    if hidden_state_cache is not None:
        hidden_state_cache[cache_key] = features
    return features


def encode_images_by_view(
    images_by_view: Mapping[str, Image.Image],
    *,
    encoder: VisualTokenEncoder,
    storage_dtype: Any,
) -> dict[str, Any]:
    tokens_by_view = {}
    items = list(images_by_view.items())
    if hasattr(encoder, "encode_images"):
        encoded_tokens = list(encoder.encode_images([image for _view_name, image in items]))
        if len(encoded_tokens) != len(items):
            raise ValueError(
                f"visual encoder returned {len(encoded_tokens)} images for {len(items)} input images"
            )
    else:
        encoded_tokens = [encoder.encode_image(image) for _view_name, image in items]
    for (view_name, _image), tokens in zip(items, encoded_tokens, strict=True):
        tokens_by_view[str(view_name)] = ensure_rank2_tokens(tokens, storage_dtype=storage_dtype)
    return tokens_by_view


def build_token_cache_manifest(
    *,
    benchmark: str,
    data_root: str | Path,
    index_path: str | Path,
    output_root: str | Path,
    encoder: VisualTokenEncoder,
    hidden_state_encoder: VLMHiddenStateEncoder | None = None,
    storage_dtype: str,
    sample_count: int,
    max_samples: int | None,
    max_samples_per_shard: int,
    view_names: Sequence[str] | None,
    shards: Sequence[TokenCacheShard],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": MEMORY_TOKEN_CACHE_FORMAT,
        "version": MEMORY_TOKEN_CACHE_VERSION,
        "benchmark": str(benchmark).upper(),
        "data_root": str(Path(data_root).expanduser()),
        "index_path": str(Path(index_path).expanduser()),
        "output_root": str(Path(output_root).expanduser()),
        "encoder": encoder.name,
        "hidden_state_encoder": None if hidden_state_encoder is None else hidden_state_encoder.name,
        "hidden_state_layers": None
        if hidden_state_encoder is None
        else [_serialize_layer_selector(layer) for layer in hidden_state_encoder.selected_layers],
        "planner_vl_summary": None
        if hidden_state_encoder is None
        else {
            "enabled": bool(hasattr(hidden_state_encoder, "encode_current_features")),
            "source": "vlm_last_valid_token",
            "encoder": hidden_state_encoder.name,
        },
        "hidden_dim": int(encoder.hidden_dim),
        "tokens_per_view": None if encoder.tokens_per_view is None else int(encoder.tokens_per_view),
        "storage_dtype": str(storage_dtype),
        "sample_count": int(sample_count),
        "max_samples": None if max_samples is None else int(max_samples),
        "max_samples_per_shard": int(max_samples_per_shard),
        "view_names": None if view_names is None else [str(name) for name in view_names],
        "shards": [
            {
                "path": str(shard.path.relative_to(Path(output_root).expanduser())),
                "sample_count": shard.sample_count,
                "start_index": shard.start_index,
                "end_index": shard.end_index,
            }
            for shard in shards
        ],
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def _update_running_minmax(
    current_min: np.ndarray | None,
    current_max: np.ndarray | None,
    values: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty when building token-cache normalization stats")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim >= 2:
        array = array.reshape(-1, array.shape[-1])
    else:
        raise ValueError(f"{name} must have at least one dimension")
    if array.shape[-1] <= 0:
        raise ValueError(f"{name} last dimension must be positive")
    finite = np.isfinite(array)
    if not bool(finite.all()):
        raise ValueError(f"{name} contains non-finite values")

    value_min = array.min(axis=0)
    value_max = array.max(axis=0)
    if current_min is None or current_max is None:
        return value_min, value_max
    if current_min.shape != value_min.shape or current_max.shape != value_max.shape:
        raise ValueError(
            f"{name} dimension changed while building normalization stats: "
            f"{current_min.shape} -> {value_min.shape}"
        )
    return np.minimum(current_min, value_min), np.maximum(current_max, value_max)


def _build_minmax_normalization_manifest(
    *,
    benchmark: str,
    action_min: np.ndarray,
    action_max: np.ndarray,
    state_min: np.ndarray,
    state_max: np.ndarray,
) -> dict[str, Any]:
    robot_key = _normalization_robot_key(benchmark)
    return {
        "enabled": True,
        "type": "train_split_minmax_to_minus_one_one",
        "statistics_from": "cache_build_rows",
        "clip_after_normalization": True,
        "clip_range": [-1.0, 1.0],
        "robot_key": robot_key,
        "stats": {
            robot_key: {
                "observation.state": {
                    "min": np.asarray(state_min, dtype=np.float32).astype(float).tolist(),
                    "max": np.asarray(state_max, dtype=np.float32).astype(float).tolist(),
                },
                "action": {
                    "min": np.asarray(action_min, dtype=np.float32).astype(float).tolist(),
                    "max": np.asarray(action_max, dtype=np.float32).astype(float).tolist(),
                },
            }
        },
    }


def _normalization_robot_key(benchmark: str) -> str:
    return str(benchmark).strip().lower() or "default"


