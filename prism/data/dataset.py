"""Temporal VLA samples, stateless mixtures, and deterministic loading."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import operator
import random
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from prism.data.schema import DataSpec, VLASample, ViewSpec
from prism.schema import PolicyInput


class StorageBackend(Protocol):
    """The model-neutral storage operations consumed by temporal sampling."""

    spec: DataSpec

    def episode_ids(self) -> tuple[int, ...]: ...

    def episode_length(self, episode_id: int) -> int: ...

    def read_training_window(self, episode_id: int, start: int, end: int) -> Any: ...

    def read_images(
        self,
        episode_id: int,
        frame_indices: Sequence[int],
        views: Sequence[ViewSpec] | None = None,
    ) -> Mapping[str, np.ndarray]: ...

class FeatureNormalizer(Protocol):
    """A bound statistics group that canonicalizes and normalizes raw values."""

    statistics_group: str

    def normalize_state(self, raw_state: np.ndarray) -> np.ndarray: ...

    def normalize_action(self, raw_action: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class AnchorIdentity:
    dataset_name: str
    episode_index: int
    frame_index: int


@dataclass(frozen=True)
class MixtureSelection:
    virtual_index: int
    dataset_index: int
    anchor_index: int
    identity: AnchorIdentity


class SingleVLADataset(Dataset[VLASample]):
    """All configured temporal anchors from one physical LeRobot root."""

    def __init__(
        self,
        *,
        name: str,
        backend: StorageBackend,
        normalizer: FeatureNormalizer,
        action_horizon: int,
        history_step_ages: Sequence[int],
        anchor_stride: int,
        include_tail: bool,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(backend.spec, DataSpec):
            raise TypeError("backend.spec must be a DataSpec")
        backend.spec.validate()
        if not isinstance(normalizer.statistics_group, str) or not normalizer.statistics_group.strip():
            raise ValueError("normalizer.statistics_group must be a non-empty string")
        self.name = name
        self.backend = backend
        self.spec = backend.spec
        self.normalizer = normalizer
        self.statistics_group = normalizer.statistics_group
        self.action_horizon = _positive_int(action_horizon, "action_horizon")
        self.anchor_stride = _positive_int(anchor_stride, "anchor_stride")
        if type(include_tail) is not bool:
            raise TypeError("include_tail must be a boolean")
        self.include_tail = include_tail

        ages = tuple(_non_negative_int(age, "history_step_ages") for age in history_step_ages)
        if not ages:
            raise ValueError("history_step_ages must contain at least one age")
        if len(set(ages)) != len(ages):
            raise ValueError("history_step_ages must not contain duplicates")
        self.history_step_ages = ages

        episode_ids = tuple(int(episode_id) for episode_id in backend.episode_ids())
        if not episode_ids:
            raise ValueError(f"dataset={name} contains no episodes")
        anchor_episode_ids: list[int] = []
        anchor_prefix: list[int] = []
        total = 0
        for episode_id in episode_ids:
            length = int(backend.episode_length(episode_id))
            count = self._anchor_count(length)
            if count == 0:
                continue
            anchor_episode_ids.append(episode_id)
            total += count
            anchor_prefix.append(total)
        if total == 0:
            raise ValueError(
                f"dataset={name} contains no anchors for horizon={self.action_horizon}, "
                f"stride={self.anchor_stride}, include_tail={self.include_tail}"
            )
        self._anchor_episode_ids = tuple(anchor_episode_ids)
        self._anchor_prefix = tuple(anchor_prefix)
        self._anchor_count_total = total

    def __len__(self) -> int:
        return self._anchor_count_total

    def __getitem__(self, index: int) -> VLASample:
        identity = self.anchor_identity(index)
        episode_id = identity.episode_index
        frame_index = identity.frame_index
        episode_length = self.backend.episode_length(episode_id)

        current_and_history = [frame_index]
        valid_history_slots: list[tuple[int, int]] = []
        history_valid_mask = np.zeros((len(self.history_step_ages),), dtype=np.bool_)
        for slot, age in enumerate(self.history_step_ages):
            history_index = frame_index - age
            if history_index < 0:
                continue
            history_valid_mask[slot] = True
            valid_history_slots.append((slot, len(current_and_history)))
            current_and_history.append(history_index)

        decoded = self.backend.read_images(episode_id, current_and_history)
        if tuple(decoded) != self.spec.view_names:
            raise ValueError(
                f"dataset={self.name} episode={episode_id} frame={frame_index} returned "
                f"view order {tuple(decoded)}, expected {self.spec.view_names}"
            )
        current_images: dict[str, np.ndarray] = {}
        history_images: dict[str, np.ndarray] = {}
        for view in self.spec.views:
            frames = np.asarray(decoded[view.name])
            if frames.ndim != 4 or frames.shape[0] != len(current_and_history) or frames.shape[-1] != 3:
                raise ValueError(
                    f"dataset={self.name} episode={episode_id} frame={frame_index} view={view.name} "
                    f"returned invalid temporal image shape {frames.shape}"
                )
            if frames.dtype != np.uint8:
                raise ValueError(
                    f"dataset={self.name} episode={episode_id} frame={frame_index} view={view.name} "
                    f"returned image dtype {frames.dtype}, expected uint8"
                )
            current = np.ascontiguousarray(frames[0])
            history = np.zeros((len(self.history_step_ages), *current.shape), dtype=np.uint8)
            for slot, decoded_index in valid_history_slots:
                history[slot] = frames[decoded_index]
            current_images[view.name] = current
            history_images[view.name] = np.ascontiguousarray(history)

        valid_end = min(frame_index + self.action_horizon, episode_length)
        numeric_window = self.backend.read_training_window(episode_id, frame_index, valid_end)
        state = _normalized_vector(
            self.normalizer.normalize_state(np.asarray(numeric_window.state, dtype=np.float32)),
            expected_dim=self.spec.state_dim,
            label=(f"dataset={self.name} episode={episode_id} frame={frame_index} normalized state"),
        )

        raw_actions = np.asarray(numeric_window.actions, dtype=np.float32)
        expected_raw_shape = (valid_end - frame_index, self.spec.action_dim)
        if raw_actions.shape != expected_raw_shape:
            raise ValueError(
                f"dataset={self.name} episode={episode_id} frame={frame_index} returned "
                f"action shape {raw_actions.shape}, expected {expected_raw_shape}"
            )
        valid_actions = _normalized_matrix(
            self.normalizer.normalize_action(raw_actions),
            expected_shape=expected_raw_shape,
            label=(f"dataset={self.name} episode={episode_id} frame={frame_index} normalized action"),
        )
        target_actions = np.zeros(
            (self.action_horizon, self.spec.action_dim),
            dtype=np.float32,
        )
        valid_count = valid_actions.shape[0]
        target_actions[:valid_count] = valid_actions
        action_valid_mask = np.zeros((self.action_horizon,), dtype=np.bool_)
        action_valid_mask[:valid_count] = True

        policy_input = PolicyInput(
            benchmark=self.spec.benchmark,
            prompt=numeric_window.instruction,
            images_by_view=current_images,
            history_images_by_view=history_images,
            history_step_ages=np.asarray(self.history_step_ages, dtype=np.int32),
            history_valid_mask=history_valid_mask,
            state=state,
            action_dim=self.spec.action_dim,
            robot_key=self.spec.robot_key,
        )
        return VLASample(
            policy_input=policy_input,
            dataset_name=self.name,
            statistics_group=self.statistics_group,
            episode_index=episode_id,
            frame_index=frame_index,
            target_actions=target_actions,
            action_valid_mask=action_valid_mask,
        )

    def anchor_identity(self, index: int) -> AnchorIdentity:
        index = _dataset_index(index, len(self))
        episode_position = bisect_right(self._anchor_prefix, index)
        previous = 0 if episode_position == 0 else self._anchor_prefix[episode_position - 1]
        local_anchor = index - previous
        episode_id = self._anchor_episode_ids[episode_position]
        return AnchorIdentity(
            dataset_name=self.name,
            episode_index=episode_id,
            frame_index=local_anchor * self.anchor_stride,
        )

    def _anchor_count(self, episode_length: int) -> int:
        if episode_length <= 0:
            raise ValueError(f"dataset={self.name} contains non-positive episode length {episode_length}")
        if self.include_tail:
            final_anchor = episode_length - 1
        else:
            final_anchor = episode_length - self.action_horizon
        if final_anchor < 0:
            return 0
        return final_anchor // self.anchor_stride + 1


class VLAMixtureDataset(Dataset[VLASample]):
    """A fixed-size virtual epoch mapped to physical anchors by stable SHA-256."""

    HASH_DOMAIN = b"prism-vla-mixture-v1"

    def __init__(
        self,
        datasets: Sequence[SingleVLADataset],
        weights: Sequence[float],
        *,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("datasets must contain at least one SingleVLADataset")
        if any(not isinstance(dataset, SingleVLADataset) for dataset in self.datasets):
            raise TypeError("datasets entries must be SingleVLADataset instances")
        names = [dataset.name for dataset in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError(f"dataset names must be unique, got {names}")
        if len(weights) != len(self.datasets):
            raise ValueError("weights length must match datasets length")
        parsed_weights = np.asarray(weights, dtype=np.float64)
        if parsed_weights.shape != (len(self.datasets),):
            raise ValueError("weights must be a one-dimensional sequence")
        if not np.isfinite(parsed_weights).all() or np.any(parsed_weights <= 0.0):
            raise ValueError("weights must contain finite positive values")
        self._cumulative_weights = np.cumsum(parsed_weights / parsed_weights.sum())
        self._cumulative_weights[-1] = 1.0
        self.samples_per_epoch = _positive_int(samples_per_epoch, "samples_per_epoch")
        self.seed = _non_negative_int(seed, "seed")
        self._epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, virtual_index: int) -> VLASample:
        selection = self.resolve_index(virtual_index)
        return self.datasets[selection.dataset_index][selection.anchor_index]

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = _non_negative_int(epoch, "epoch")

    def resolve_index(self, virtual_index: int) -> MixtureSelection:
        virtual_index = _dataset_index(virtual_index, len(self))
        payload = b"\0".join(
            (
                self.HASH_DOMAIN,
                str(self.seed).encode("ascii"),
                str(self._epoch).encode("ascii"),
                str(virtual_index).encode("ascii"),
            )
        )
        digest = hashlib.sha256(payload).digest()
        dataset_draw = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
        dataset_index = int(np.searchsorted(self._cumulative_weights, dataset_draw, side="right"))
        anchor_draw = int.from_bytes(digest[8:16], byteorder="big")
        dataset = self.datasets[dataset_index]
        anchor_index = (anchor_draw * len(dataset)) >> 64
        return MixtureSelection(
            virtual_index=virtual_index,
            dataset_index=dataset_index,
            anchor_index=anchor_index,
            identity=dataset.anchor_identity(anchor_index),
        )


def raw_sample_collate(samples: list[VLASample]) -> list[VLASample]:
    """Preserve variable image sizes for the later model-owned collator."""

    return samples


def seed_data_worker(worker_id: int) -> None:
    """Seed libraries for a newly-created worker from PyTorch's worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (1 << 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_vla_dataloader(
    dataset: Dataset[VLASample],
    *,
    batch_size_per_rank: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    seed: int,
    num_replicas: int = 1,
    rank: int = 0,
) -> DataLoader[list[VLASample]]:
    """Build a non-persistent map-style loader with unique virtual indices per rank."""

    batch_size = _positive_int(batch_size_per_rank, "batch_size_per_rank")
    workers = _non_negative_int(num_workers, "num_workers")
    replicas = _positive_int(num_replicas, "num_replicas")
    rank = _non_negative_int(rank, "rank")
    base_seed = _non_negative_int(seed, "seed")
    if rank >= replicas:
        raise ValueError(f"rank must be smaller than num_replicas, got rank={rank}, replicas={replicas}")
    if type(pin_memory) is not bool or type(drop_last) is not bool:
        raise TypeError("pin_memory and drop_last must be booleans")
    if replicas > 1 and len(dataset) % replicas != 0:
        raise ValueError(
            f"dataset length {len(dataset)} must be divisible by num_replicas={replicas} "
            "to avoid duplicated virtual indices"
        )

    sampler: DistributedSampler[VLASample] | None = None
    if replicas > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=replicas,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(base_seed + rank)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        collate_fn=raw_sample_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,
        worker_init_fn=seed_data_worker,
        generator=generator,
    )


def set_data_epoch(dataset: Dataset[VLASample], loader: DataLoader[Any], epoch: int) -> None:
    """Propagate epoch before creating a fresh non-persistent worker iterator."""

    epoch = _non_negative_int(epoch, "epoch")
    if isinstance(dataset, VLAMixtureDataset):
        dataset.set_epoch(epoch)
    sampler = loader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def _dataset_index(index: int, length: int) -> int:
    try:
        parsed = operator.index(index)
    except TypeError as exc:
        raise TypeError(f"dataset index must be an integer, got {index!r}") from exc
    if parsed < 0:
        parsed += length
    if parsed < 0 or parsed >= length:
        raise IndexError(f"dataset index {parsed} is outside [0, {length})")
    return parsed


def _positive_int(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _normalized_vector(value: np.ndarray, *, expected_dim: int, label: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float32)
    if output.shape != (expected_dim,):
        raise ValueError(f"{label} has shape {output.shape}, expected {(expected_dim,)}")
    if not np.isfinite(output).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(output)


def _normalized_matrix(
    value: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    output = np.asarray(value, dtype=np.float32)
    if output.shape != expected_shape:
        raise ValueError(f"{label} has shape {output.shape}, expected {expected_shape}")
    if not np.isfinite(output).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(output)


__all__ = [
    "AnchorIdentity",
    "FeatureNormalizer",
    "MixtureSelection",
    "SingleVLADataset",
    "StorageBackend",
    "VLAMixtureDataset",
    "build_vla_dataloader",
    "raw_sample_collate",
    "seed_data_worker",
    "set_data_epoch",
]
