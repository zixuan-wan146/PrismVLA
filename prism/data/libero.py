from __future__ import annotations

# --- migrated from src/prism/dataset/libero.py ---
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_LIBERO_VIEW_NAMES = ("agentview_rgb", "eye_in_hand_rgb")


@dataclass(frozen=True)
class LiberoFrame:
    tau: int
    images_by_view: Mapping[str, Image.Image]
    action: np.ndarray
    state_vector: np.ndarray


class LiberoEpisodeReader:
    """Read one LIBERO demonstration from a local HDF5 file without importing the simulator."""

    def __init__(
        self,
        hdf5_path: str | Path,
        *,
        demo_key: str,
        view_names: Sequence[str] = DEFAULT_LIBERO_VIEW_NAMES,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser()
        if not self.hdf5_path.exists():
            raise FileNotFoundError(self.hdf5_path)
        self.demo_key = str(demo_key)
        self.view_names = tuple(str(name) for name in view_names)
        if not self.view_names:
            raise ValueError("view_names must contain at least one view")

        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            self.demo_path = f"data/{self.demo_key}"
            if self.demo_path not in handle:
                raise KeyError(f"demo {self.demo_key!r} is missing from {self.hdf5_path}")
            demo = handle[self.demo_path]
            self.length = _dataset_length(demo, "actions")
            self.action_dim = int(np.asarray(demo["actions"].shape[-1]).item())
            self._validate_view_lengths(demo)

    def __len__(self) -> int:
        return self.length

    def read_frame(self, index: int) -> LiberoFrame:
        index = int(index)
        if index < 0 or index >= self.length:
            raise IndexError(f"frame index {index} out of range for episode length {self.length}")

        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            demo = handle[self.demo_path]
            images_by_view = {
                view_name: Image.fromarray(np.asarray(demo[f"obs/{view_name}"][index], dtype=np.uint8)).convert("RGB")
                for view_name in self.view_names
            }
            action = np.asarray(demo["actions"][index], dtype=np.float32).reshape(-1)
            state_vector = read_libero_state_vector(demo, index)
        return LiberoFrame(
            tau=index,
            images_by_view=images_by_view,
            action=action,
            state_vector=state_vector,
        )

    def read_future_actions(self, start: int, end: int) -> np.ndarray:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.length:
            raise IndexError(f"invalid action slice [{start}, {end}) for episode length {self.length}")
        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            return np.asarray(handle[self.demo_path]["actions"][start:end], dtype=np.float32)

    def _validate_view_lengths(self, demo: Any) -> None:
        for view_name in self.view_names:
            key = f"obs/{view_name}"
            if key not in demo:
                raise KeyError(f"LIBERO demo {self.demo_key!r} is missing image view: {key}")
            view_length = int(demo[key].shape[0])
            if view_length != self.length:
                raise ValueError(f"view {view_name!r} length {view_length} does not match action length {self.length}")


def read_libero_state_vector(demo: Any, index: int) -> np.ndarray:
    parts = []
    for key in ("obs/ee_states", "obs/gripper_states"):
        if key in demo:
            parts.append(np.asarray(demo[key][index], dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def _dataset_length(handle: Any, key: str) -> int:
    if key not in handle:
        raise KeyError(f"dataset key {key!r} is missing")
    shape = handle[key].shape
    if not shape:
        raise ValueError(f"dataset key {key!r} must have a time dimension")
    return int(shape[0])


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LiberoEpisodeReader requires h5py to read HDF5 demonstrations") from exc
    return h5py

# --- migrated from src/prism/dataset/libero_progress_warmup.py ---
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from prism.data.cache import read_memory_replay_jsonl
from prism.data.cache import MemoryReplayFrameReader
from prism.models.planner import ActionSegmentAutoencoder
from prism.models.planner import ActionSegmentAutoencoderConfig
from prism.utils import NormalizationStats


LIBERO_PROGRESS_WARMUP_FORMAT = "libero_progress_vl_embedding_warmup_cache"
LIBERO_PROGRESS_WARMUP_VERSION = 2


ActionNormalizer = Callable[[torch.Tensor], torch.Tensor]


class VLSummaryEncoder(Protocol):
    name: str
    hidden_dim: int

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> torch.Tensor:
        ...

    def encode_batch(self, batch: Sequence[tuple[Mapping[str, Image.Image], str]]) -> list[torch.Tensor]:
        ...


@dataclass(frozen=True)
class LiberoProgressWarmupBuildResult:
    output_root: Path
    manifest_path: Path
    step_count: int
    window_count: int


class ImageStatsVLSummaryEncoder:
    """Deterministic tiny encoder for tests and pipeline smoke checks."""

    name = "image_stats_vl_summary"

    def __init__(self, *, hidden_dim: int = 16) -> None:
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> torch.Tensor:
        values: list[float] = []
        for view_name in sorted(images_by_view):
            rgb = np.asarray(images_by_view[view_name].convert("RGB"), dtype=np.float32) / 255.0
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
            values.extend(float(value) for value in stats.tolist())
        values.append(float(len(str(prompt))) / 256.0)
        return torch.tensor(np.resize(np.asarray(values, dtype=np.float32), self.hidden_dim), dtype=torch.float32)

    def encode_batch(self, batch: Sequence[tuple[Mapping[str, Image.Image], str]]) -> list[torch.Tensor]:
        return [self.encode_current(images_by_view, prompt) for images_by_view, prompt in batch]


class InternVL3VLSummaryEncoder:
    """InternVL3 image-language summary encoder for progress warm-up data."""

    name = "internvl3_vl_summary"

    def __init__(
        self,
        *,
        model_name: str = "OpenGVLab/InternVL3-1B",
        image_size: int = 448,
        device: str = "cuda",
        storage_dtype: str = "bfloat16",
    ) -> None:
        from prism.models.vlm import InternVL3Embedder

        self.embedder = InternVL3Embedder(model_name=model_name, image_size=image_size, device=device)
        self.embedder.eval()
        self.device = str(device)
        self.storage_dtype = resolve_storage_dtype(storage_dtype)
        self.hidden_dim = int(getattr(self.embedder.model, "llm_hidden_size", 0) or 0)

    def encode_current(self, images_by_view: Mapping[str, Image.Image], prompt: str) -> torch.Tensor:
        return self.encode_batch([(images_by_view, prompt)])[0]

    def encode_batch(self, batch: Sequence[tuple[Mapping[str, Image.Image], str]]) -> list[torch.Tensor]:
        if not batch:
            return []
        images_per_sample = [list(images_by_view.values()) for images_by_view, _prompt in batch]
        if any(not images for images in images_per_sample):
            raise ValueError("each VL summary sample must contain at least one image")
        flat_images = [image for images in images_per_sample for image in images]
        image_counts = [len(images) for images in images_per_sample]
        prompts = [str(prompt) for _images_by_view, prompt in batch]
        with torch.no_grad():
            summaries = self._encode_batch(flat_images, image_counts=image_counts, prompts=prompts)
        return [summary.detach().to(dtype=self.storage_dtype).cpu() for summary in summaries]

    def _encode_batch(self, images: Sequence[Image.Image], *, image_counts: Sequence[int], prompts: Sequence[str]) -> torch.Tensor:
        if not images:
            raise ValueError("images_by_view must not be empty")
        if sum(int(count) for count in image_counts) != len(images):
            raise ValueError("image_counts must sum to the number of images")
        if len(image_counts) != len(prompts):
            raise ValueError("image_counts and prompts must have the same length")

        pixel_values, flat_num_tiles = self.embedder._preprocess_images(list(images))
        vit_embeds = self.embedder.model.extract_feature(pixel_values)
        tokens_per_tile = int(self.embedder.model.num_image_token)
        prompts_with_image_tokens = []
        sample_tile_counts: list[list[int]] = []
        cursor = 0
        for image_count, prompt in zip(image_counts, prompts, strict=True):
            tile_counts = [int(count) for count in flat_num_tiles[cursor : cursor + int(image_count)]]
            sample_tile_counts.append(tile_counts)
            prompts_with_image_tokens.append(self.embedder._build_multimodal_prompt(tile_counts, str(prompt)))
            cursor += int(image_count)

        model_inputs = self.embedder.tokenizer(
            prompts_with_image_tokens,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.embedder.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        input_embeds = self.embedder.model.language_model.get_input_embeddings()(input_ids).clone()
        if input_embeds.ndim != 3:
            raise ValueError(f"language input embeddings must have shape [B, N, D], got {tuple(input_embeds.shape)}")
        batch_size, _seq_len, hidden_dim = input_embeds.shape
        vit_embeds = vit_embeds.reshape(-1, hidden_dim)
        image_token_mask = input_ids == self.embedder.img_context_token_id
        token_cursor = 0
        for sample_index, tile_counts in enumerate(sample_tile_counts):
            selected = image_token_mask[sample_index]
            selected_count = int(selected.sum().item())
            vit_token_count = int(sum(tile_counts) * tokens_per_tile)
            message = (
                "Image/text embedding token mismatch: "
                f"sample_index={sample_index}, "
                f"selected_img_context_tokens={selected_count}, "
                f"vit_tokens={vit_token_count}, "
                f"max_text_length={self.embedder.max_text_length}, "
                f"image_count={len(tile_counts)}, "
                f"num_tiles_list={tile_counts}"
            )
            if selected_count != vit_token_count:
                if not self.embedder.allow_image_token_truncation:
                    raise ValueError(message)
                copy_count = min(selected_count, vit_token_count)
                if copy_count > 0:
                    selected_indices = selected.nonzero(as_tuple=False).reshape(-1)
                    input_embeds[sample_index, selected_indices[:copy_count]] = vit_embeds[
                        token_cursor : token_cursor + copy_count
                    ]
            else:
                input_embeds[sample_index, selected] = vit_embeds[token_cursor : token_cursor + vit_token_count]
            token_cursor += vit_token_count
        if token_cursor != int(vit_embeds.shape[0]):
            raise ValueError(f"assigned {token_cursor} visual tokens but encoder produced {int(vit_embeds.shape[0])}")

        outputs = self.embedder.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        final_hidden = outputs.hidden_states[-1]
        positions = torch.arange(final_hidden.shape[1], device=final_hidden.device).unsqueeze(0)
        last_indices = (attention_mask.to(device=final_hidden.device).long() * positions).max(dim=1).values
        summary = final_hidden[torch.arange(batch_size, device=final_hidden.device), last_indices].to(torch.float32)
        if summary.ndim != 2 or int(summary.shape[0]) != batch_size:
            raise ValueError(f"VL summaries must have shape [B, D], got {tuple(summary.shape)}")
        if self.hidden_dim <= 0:
            self.hidden_dim = int(summary.shape[-1])
        return summary


class LiberoProgressWarmupDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = resolve_libero_progress_manifest_path(manifest_path)
        self.manifest = read_libero_progress_warmup_manifest(self.manifest_path)
        self.output_root = self.manifest_path.parent
        payload = _torch_load(self.output_root / self.manifest["data_path"])
        if payload.get("format") != LIBERO_PROGRESS_WARMUP_FORMAT:
            raise ValueError(f"invalid LIBERO progress warmup payload: {self.output_root / self.manifest['data_path']}")
        self.steps = tuple(payload["steps"])
        self.windows = tuple(payload["windows"])
        if len(self.steps) != int(self.manifest["step_count"]):
            raise ValueError("manifest step_count does not match payload")
        if len(self.windows) != int(self.manifest["window_count"]):
            raise ValueError("manifest window_count does not match payload")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = dict(self.windows[int(index)])
        burnin_steps = [self.steps[int(step_index)] for step_index in window["burnin_step_indices"]]
        loss_steps = [self.steps[int(step_index)] for step_index in window["loss_step_indices"]]
        return {
            "window_index": int(index),
            "episode_id": str(window["episode_id"]),
            "suite": str(window["suite"]),
            "task_name": str(window["task_name"]),
            "start_k": int(window["start_k"]),
            "ctx_start": int(window["ctx_start"]),
            "burnin": burnin_steps,
            "loss": loss_steps,
        }


class TemperatureSuiteSampler(Sampler[int]):
    def __init__(
        self,
        dataset: LiberoProgressWarmupDataset,
        *,
        samples_per_epoch: int,
        alpha: float = 0.5,
        seed: int = 0,
        indices: Sequence[int] | None = None,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        self.dataset = dataset
        self.samples_per_epoch = int(samples_per_epoch)
        self.alpha = float(alpha)
        self.seed = int(seed)
        by_suite: dict[str, list[int]] = defaultdict(list)
        selected_indices = range(len(dataset.windows)) if indices is None else [int(index) for index in indices]
        for index in selected_indices:
            if index < 0 or index >= len(dataset.windows):
                raise IndexError(f"window index {index} is out of range for dataset of size {len(dataset.windows)}")
            window = dataset.windows[index]
            by_suite[str(window["suite"])].append(index)
        if not by_suite:
            raise ValueError("dataset has no windows")
        self.suites = tuple(sorted(by_suite))
        self.indices_by_suite = {suite: tuple(indices) for suite, indices in by_suite.items()}
        weights = torch.tensor(
            [float(len(self.indices_by_suite[suite])) ** self.alpha for suite in self.suites],
            dtype=torch.double,
        )
        self.probabilities = (weights / weights.sum()).tolist()

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.samples_per_epoch):
            suite = rng.choices(self.suites, weights=self.probabilities, k=1)[0]
            yield rng.choice(self.indices_by_suite[suite])

    def __len__(self) -> int:
        return self.samples_per_epoch


def collate_libero_progress_warmup_windows(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty")
    max_burnin = max(len(item["burnin"]) for item in batch)
    loss_len = len(batch[0]["loss"])
    if any(len(item["loss"]) != loss_len for item in batch):
        raise ValueError("all loss windows in a batch must have the same length")
    hidden_dim = int(batch[0]["loss"][0]["vl_summary"].shape[-1])
    state_dim = int(batch[0]["loss"][0]["state"].shape[-1])
    action_shape = tuple(batch[0]["loss"][0]["executed_actions"].shape)
    latent_dim = int(batch[0]["loss"][0]["target_intent"].shape[-1])

    burnin = _empty_window_tensors(len(batch), max_burnin, hidden_dim, state_dim, action_shape, latent_dim)
    loss = _empty_window_tensors(len(batch), loss_len, hidden_dim, state_dim, action_shape, latent_dim)
    burnin_mask = torch.zeros(len(batch), max_burnin, dtype=torch.bool)
    loss_mask = torch.ones(len(batch), loss_len, dtype=torch.bool)
    burnin_replan_indices = torch.full((len(batch), max_burnin), -1, dtype=torch.long)
    loss_replan_indices = torch.zeros(len(batch), loss_len, dtype=torch.long)

    for batch_index, item in enumerate(batch):
        _fill_steps(burnin, item["burnin"], batch_index)
        burnin_count = len(item["burnin"])
        if burnin_count:
            burnin_mask[batch_index, :burnin_count] = True
            burnin_replan_indices[batch_index, :burnin_count] = torch.tensor(
                [int(step["replan_index"]) for step in item["burnin"]],
                dtype=torch.long,
            )
        _fill_steps(loss, item["loss"], batch_index)
        loss_replan_indices[batch_index] = torch.tensor([int(step["replan_index"]) for step in item["loss"]], dtype=torch.long)

    return {
        "episode_id": [str(item["episode_id"]) for item in batch],
        "suite": [str(item["suite"]) for item in batch],
        "task_name": [str(item["task_name"]) for item in batch],
        "window_index": torch.tensor([int(item["window_index"]) for item in batch], dtype=torch.long),
        "start_k": torch.tensor([int(item["start_k"]) for item in batch], dtype=torch.long),
        "ctx_start": torch.tensor([int(item["ctx_start"]) for item in batch], dtype=torch.long),
        "burnin": burnin,
        "burnin_mask": burnin_mask,
        "burnin_replan_indices": burnin_replan_indices,
        "loss": loss,
        "loss_mask": loss_mask,
        "loss_replan_indices": loss_replan_indices,
    }


def build_libero_progress_vl_embedding_cache(
    *,
    data_root: str | Path,
    index_path: str | Path,
    output_root: str | Path,
    vl_encoder: VLSummaryEncoder,
    action_horizon: int = 32,
    replan_stride: int = 16,
    burnin_replan_steps: int = 8,
    loss_replan_steps: int = 8,
    allow_short_burnin: bool = True,
    intent_encoder: ActionSegmentAutoencoder | None = None,
    intent_encoder_checkpoint: str | Path | None = None,
    action_normalizer: ActionNormalizer | None = None,
    norm_stats_path: str | Path | None = None,
    robot_key: str | None = None,
    storage_dtype: torch.dtype = torch.float32,
    view_names: Sequence[str] | None = None,
    max_steps: int | None = None,
    progress_interval: int | None = 100,
    vl_batch_size: int = 1,
) -> LiberoProgressWarmupBuildResult:
    if action_horizon <= 0 or replan_stride <= 0 or burnin_replan_steps < 0 or loss_replan_steps <= 0:
        raise ValueError("invalid horizon/stride/window configuration")
    if int(vl_batch_size) <= 0:
        raise ValueError("vl_batch_size must be positive")
    rows = read_memory_replay_jsonl(index_path)
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive when provided")
        rows = rows[: int(max_steps)]
    if not rows:
        raise ValueError(f"LIBERO replay index has no rows: {index_path}")

    data_path_root = Path(data_root).expanduser()
    normalizer = action_normalizer or (lambda tensor: tensor)
    reader = MemoryReplayFrameReader(benchmark="LIBERO", data_root=data_path_root, view_names=view_names)
    row_by_episode_step = {
        (str(row["episode_id"]), int(row["current_step"])): row
        for row in rows
        if str(row.get("benchmark", "LIBERO")).upper() == "LIBERO"
    }
    prompt_cache: dict[str, str] = {}

    steps: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        if str(row.get("benchmark", "LIBERO")).upper() != "LIBERO":
            continue
        current_step = int(row["current_step"])
        episode_id = str(row["episode_id"])
        if current_step % int(replan_stride) != 0:
            continue
        if int(row["action_valid_count"]) < int(action_horizon):
            continue

        sample = reader.read(row)
        future_actions = torch.as_tensor(sample.future_actions, dtype=torch.float32)
        if future_actions.shape[0] < int(action_horizon):
            continue
        target_actions = normalizer(future_actions[:action_horizon]).float()

        if current_step == 0:
            executed_actions = torch.zeros(replan_stride, target_actions.shape[-1], dtype=torch.float32)
            executed_mask = torch.zeros(replan_stride, dtype=torch.bool)
        else:
            prev_row = row_by_episode_step.get((episode_id, current_step - int(replan_stride)))
            if prev_row is None:
                continue
            executed_raw = _read_libero_action_slice(
                data_path_root,
                prev_row,
                start=current_step - int(replan_stride),
                end=current_step,
            )
            if executed_raw.shape[0] != int(replan_stride):
                continue
            executed_actions = normalizer(torch.as_tensor(executed_raw, dtype=torch.float32)).float()
            executed_mask = torch.ones(replan_stride, dtype=torch.bool)

        prompt = _libero_prompt_for_row(data_path_root, row, prompt_cache)
        target_intent = _encode_target_intent(intent_encoder, target_actions)
        pending.append(
            {
                "images_by_view": sample.current.images_by_view,
                "step": {
                    "step_index": -1,
                    "sample_index": int(sample_index),
                    "episode_id": episode_id,
                    "suite": _suite_from_episode_id(episode_id),
                    "task_name": str(row.get("task_name", "")),
                    "prompt": prompt,
                    "current_step": current_step,
                    "replan_index": current_step // int(replan_stride),
                    "state": torch.as_tensor(sample.current.state_vector, dtype=torch.float32).cpu(),
                    "executed_actions": executed_actions.cpu(),
                    "executed_action_mask": executed_mask.cpu(),
                    "target_intent": target_intent.cpu(),
                },
            }
        )
        if len(pending) >= int(vl_batch_size):
            _flush_vl_summary_batch(
                pending,
                steps,
                vl_encoder=vl_encoder,
                storage_dtype=storage_dtype,
                progress_interval=progress_interval,
            )
            pending = []

    if pending:
        _flush_vl_summary_batch(
            pending,
            steps,
            vl_encoder=vl_encoder,
            storage_dtype=storage_dtype,
            progress_interval=progress_interval,
        )

    windows = build_libero_progress_windows(
        steps,
        burnin_replan_steps=burnin_replan_steps,
        loss_replan_steps=loss_replan_steps,
        allow_short_burnin=allow_short_burnin,
    )
    output_path = Path(output_root).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = output_path / "data.pt"
    torch.save(
        {
            "format": LIBERO_PROGRESS_WARMUP_FORMAT,
            "version": LIBERO_PROGRESS_WARMUP_VERSION,
            "steps": steps,
            "windows": windows,
        },
        data_path,
    )
    suite_counts = _window_suite_counts(windows)
    manifest = {
        "format": LIBERO_PROGRESS_WARMUP_FORMAT,
        "version": LIBERO_PROGRESS_WARMUP_VERSION,
        "data_root": str(data_path_root),
        "index_path": str(Path(index_path).expanduser()),
        "data_path": data_path.name,
        "embedding": "vl_summary",
        "encoder": str(getattr(vl_encoder, "name", vl_encoder.__class__.__name__)),
        "hidden_dim": int(getattr(vl_encoder, "hidden_dim", int(steps[0]["vl_summary"].shape[-1]) if steps else 0)),
        "view_names": None if view_names is None else [str(name) for name in view_names],
        "action_horizon": int(action_horizon),
        "replan_stride": int(replan_stride),
        "burnin_replan_steps": int(burnin_replan_steps),
        "loss_replan_steps": int(loss_replan_steps),
        "allow_short_burnin": bool(allow_short_burnin),
        "vl_batch_size": int(vl_batch_size),
        "intent_encoder_checkpoint": None if intent_encoder_checkpoint is None else str(Path(intent_encoder_checkpoint).expanduser()),
        "norm_stats_path": None if norm_stats_path is None else str(Path(norm_stats_path).expanduser()),
        "robot_key": robot_key,
        "step_count": len(steps),
        "window_count": len(windows),
        "suite_window_counts": suite_counts,
        "sampler": {
            "default": "temperature_suite",
            "sampling_alpha": 0.5,
            "samples_per_epoch": 8192,
        },
    }
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return LiberoProgressWarmupBuildResult(
        output_root=output_path,
        manifest_path=manifest_path,
        step_count=len(steps),
        window_count=len(windows),
    )


def _flush_vl_summary_batch(
    pending: Sequence[Mapping[str, Any]],
    steps: list[dict[str, Any]],
    *,
    vl_encoder: VLSummaryEncoder,
    storage_dtype: torch.dtype,
    progress_interval: int | None,
) -> None:
    summaries = vl_encoder.encode_batch(
        [(item["images_by_view"], item["step"]["prompt"]) for item in pending]
    )
    if len(summaries) != len(pending):
        raise ValueError(f"VL encoder returned {len(summaries)} summaries for {len(pending)} pending samples")
    for item, vl_summary in zip(pending, summaries, strict=True):
        step = dict(item["step"])
        step["step_index"] = len(steps)
        step["vl_summary"] = torch.as_tensor(vl_summary).to(dtype=storage_dtype).cpu()
        steps.append(step)
        if progress_interval and len(steps) % int(progress_interval) == 0:
            print(
                json.dumps(
                    {
                        "event": "progress_vl_embedding_build",
                        "steps": len(steps),
                        "sample_index": int(step["sample_index"]),
                        "episode_id": str(step["episode_id"]),
                        "current_step": int(step["current_step"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


build_libero_progress_warmup_cache = build_libero_progress_vl_embedding_cache


def build_libero_progress_windows(
    steps: Sequence[Mapping[str, Any]],
    *,
    burnin_replan_steps: int = 8,
    loss_replan_steps: int = 8,
    allow_short_burnin: bool = True,
) -> list[dict[str, Any]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for step in steps:
        by_episode[str(step["episode_id"])].append(step)
    windows: list[dict[str, Any]] = []
    for episode_id, episode_steps in sorted(by_episode.items()):
        ordered = sorted(episode_steps, key=lambda step: int(step["replan_index"]))
        for start_pos in range(0, len(ordered) - int(loss_replan_steps) + 1):
            loss_steps = ordered[start_pos : start_pos + int(loss_replan_steps)]
            expected = list(range(int(loss_steps[0]["replan_index"]), int(loss_steps[0]["replan_index"]) + int(loss_replan_steps)))
            if [int(step["replan_index"]) for step in loss_steps] != expected:
                continue
            if not allow_short_burnin and start_pos < int(burnin_replan_steps):
                continue
            ctx_start_pos = max(0, start_pos - int(burnin_replan_steps))
            burnin_steps = ordered[ctx_start_pos:start_pos]
            start_k = int(loss_steps[0]["replan_index"])
            windows.append(
                {
                    "window_index": len(windows),
                    "episode_id": episode_id,
                    "suite": str(loss_steps[0].get("suite", _suite_from_episode_id(episode_id))),
                    "task_name": str(loss_steps[0].get("task_name", "")),
                    "ctx_start": int(ordered[ctx_start_pos]["replan_index"]) if burnin_steps else start_k,
                    "start_k": start_k,
                    "burnin_step_indices": [int(step["step_index"]) for step in burnin_steps],
                    "loss_step_indices": [int(step["step_index"]) for step in loss_steps],
                }
            )
    return windows


def load_action_segment_autoencoder(checkpoint_path: str | Path, *, device: str | torch.device = "cpu") -> ActionSegmentAutoencoder:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device, weights_only=False)
    raw_config = checkpoint.get("segment_autoencoder_config")
    if raw_config is None:
        raise KeyError(f"checkpoint lacks segment_autoencoder_config: {checkpoint_path}")
    model = ActionSegmentAutoencoder(ActionSegmentAutoencoderConfig(**raw_config)).to(device)
    model.load_state_dict(checkpoint["segment_autoencoder_state_dict"])
    model.eval()
    return model


def action_normalizer_from_stats(norm_stats_path: str | Path | None, *, robot_key: str | None = None) -> ActionNormalizer:
    if norm_stats_path is None:
        return lambda tensor: tensor
    stats = NormalizationStats(norm_stats_path, robot_key=robot_key)
    return lambda tensor: stats.normalize_action(tensor, robot_key=robot_key)


def read_libero_progress_warmup_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = resolve_libero_progress_manifest_path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != LIBERO_PROGRESS_WARMUP_FORMAT:
        raise ValueError(f"invalid LIBERO progress warmup manifest format: {manifest.get('format')!r}")
    if int(manifest.get("version", -1)) != LIBERO_PROGRESS_WARMUP_VERSION:
        raise ValueError(f"unsupported LIBERO progress warmup version: {manifest.get('version')!r}")
    return manifest


def resolve_libero_progress_manifest_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        resolved = resolved / "manifest.json"
    return resolved


def resolve_storage_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported storage dtype: {name!r}")


def _encode_target_intent(intent_encoder: ActionSegmentAutoencoder | None, target_actions: torch.Tensor) -> torch.Tensor:
    if intent_encoder is None:
        flat = target_actions.reshape(-1)
        if flat.numel() < 128:
            flat = F.pad(flat, (0, 128 - flat.numel()))
        return F.normalize(flat[:128], dim=-1).float()
    device = next(intent_encoder.parameters()).device
    with torch.no_grad():
        latent = intent_encoder.encode(target_actions.to(device).unsqueeze(0)).squeeze(0)
    return F.normalize(latent.detach().cpu().float(), dim=-1)


def _empty_window_tensors(
    batch_size: int,
    steps: int,
    hidden_dim: int,
    state_dim: int,
    action_shape: tuple[int, ...],
    latent_dim: int,
) -> dict[str, torch.Tensor]:
    return {
        "vl_summary": torch.zeros(batch_size, steps, hidden_dim, dtype=torch.float32),
        "state": torch.zeros(batch_size, steps, state_dim, dtype=torch.float32),
        "executed_actions": torch.zeros(batch_size, steps, *action_shape, dtype=torch.float32),
        "executed_action_mask": torch.zeros(batch_size, steps, action_shape[0], dtype=torch.bool),
        "target_intent": torch.zeros(batch_size, steps, latent_dim, dtype=torch.float32),
    }


def _fill_steps(tensors: dict[str, torch.Tensor], steps: Sequence[Mapping[str, Any]], batch_index: int) -> None:
    for step_index, step in enumerate(steps):
        tensors["vl_summary"][batch_index, step_index] = torch.as_tensor(step["vl_summary"], dtype=torch.float32)
        tensors["state"][batch_index, step_index] = torch.as_tensor(step["state"], dtype=torch.float32)
        tensors["executed_actions"][batch_index, step_index] = torch.as_tensor(step["executed_actions"], dtype=torch.float32)
        tensors["executed_action_mask"][batch_index, step_index] = torch.as_tensor(step["executed_action_mask"], dtype=torch.bool)
        tensors["target_intent"][batch_index, step_index] = torch.as_tensor(step["target_intent"], dtype=torch.float32)


def _libero_prompt_for_row(data_root: Path, row: Mapping[str, Any], cache: dict[str, str]) -> str:
    source_path = str(row.get("source_path", ""))
    if source_path and source_path in cache:
        return cache[source_path]
    prompt = ""
    if source_path:
        hdf5_path = data_root / source_path
        try:
            import h5py

            with h5py.File(hdf5_path, "r") as handle:
                problem_info = handle["data"].attrs.get("problem_info")
            if isinstance(problem_info, bytes):
                problem_info = problem_info.decode("utf-8")
            if problem_info:
                payload = json.loads(str(problem_info))
                prompt = str(payload.get("language_instruction") or "")
        except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError):
            prompt = ""
    if not prompt:
        prompt = str(row.get("task_name") or "").replace("_", " ").strip()
    if source_path:
        cache[source_path] = prompt
    return prompt


def _read_libero_action_slice(data_root: Path, row: Mapping[str, Any], *, start: int, end: int) -> np.ndarray:
    import h5py

    source_path = str(row["source_path"])
    demo_key = str(row.get("episode_key") or str(row["episode_id"]).split(":")[-1])
    with h5py.File(data_root / source_path, "r") as handle:
        actions = np.asarray(handle[f"data/{demo_key}/actions"][int(start) : int(end)], dtype=np.float32)
    return actions


def _suite_from_episode_id(episode_id: str) -> str:
    return str(episode_id).split(":")[0]


def _window_suite_counts(windows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for window in windows:
        counts[str(window["suite"])] += 1
    return dict(sorted(counts.items()))


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)

