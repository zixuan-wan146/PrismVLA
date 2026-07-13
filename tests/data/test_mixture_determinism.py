from __future__ import annotations

import multiprocessing
from queue import Empty

import pytest

from prism.data.dataset import (
    VLAMixtureDataset,
    build_vla_dataloader,
    set_data_epoch,
)
from tests.data.test_dataset import FakeBackend, FakeNormalizer
from prism.data.dataset import SingleVLADataset


def _single(name: str, lengths: tuple[int, ...]) -> SingleVLADataset:
    return SingleVLADataset(
        name=name,
        backend=FakeBackend(lengths=lengths),
        normalizer=FakeNormalizer(),
        action_horizon=4,
        history_step_ages=(3, 1),
        anchor_stride=1,
        include_tail=True,
    )


def _mixture(*, samples_per_epoch: int = 24) -> VLAMixtureDataset:
    return VLAMixtureDataset(
        (_single("short", (4,)), _single("long", (19,))),
        (1.0, 3.0),
        samples_per_epoch=samples_per_epoch,
        seed=1234,
    )


def _identity_rows(dataset: VLAMixtureDataset, count: int) -> list[tuple[object, ...]]:
    return [
        (
            selection.dataset_index,
            selection.anchor_index,
            selection.identity.dataset_name,
            selection.identity.episode_index,
            selection.identity.frame_index,
        )
        for selection in (dataset.resolve_index(index) for index in range(count))
    ]


def _spawn_resolve(dataset: VLAMixtureDataset, count: int, queue: object) -> None:
    queue.put(_identity_rows(dataset, count))


def _loader_identities(loader: object) -> list[tuple[str, int, int]]:
    return [(sample.dataset_name, sample.episode_index, sample.frame_index) for batch in loader for sample in batch]


def test_same_seed_epoch_and_index_are_stable_after_spawned_restart() -> None:
    dataset = _mixture()
    dataset.set_epoch(7)
    expected = _identity_rows(dataset, 16)

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_resolve, args=(dataset, 16, queue))
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("spawned deterministic-mixture check timed out")
    assert process.exitcode == 0
    try:
        observed = queue.get(timeout=5)
    except Empty:
        pytest.fail("spawned deterministic-mixture check returned no result")
    assert observed == expected


def test_epoch_changes_stateless_mapping_without_rng_state() -> None:
    dataset = _mixture()
    epoch_zero = _identity_rows(dataset, 16)
    dataset.set_epoch(1)
    epoch_one = _identity_rows(dataset, 16)
    dataset.set_epoch(0)

    assert epoch_zero != epoch_one
    assert _identity_rows(dataset, 16) == epoch_zero


def test_weights_are_selection_probabilities_not_length_multipliers() -> None:
    dataset = VLAMixtureDataset(
        (_single("tiny", (2,)), _single("large", (200,))),
        (1.0, 1.0),
        samples_per_epoch=10_000,
        seed=99,
    )

    tiny_count = sum(dataset.resolve_index(index).dataset_index == 0 for index in range(len(dataset)))
    assert 0.48 < tiny_count / len(dataset) < 0.52


def test_worker_count_does_not_change_sample_identity() -> None:
    dataset = _mixture(samples_per_epoch=24)
    loader_zero = build_vla_dataloader(
        dataset,
        batch_size_per_rank=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        seed=41,
    )
    loader_two = build_vla_dataloader(
        dataset,
        batch_size_per_rank=4,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        seed=41,
    )
    set_data_epoch(dataset, loader_zero, 3)
    expected = _loader_identities(loader_zero)
    set_data_epoch(dataset, loader_two, 3)

    assert _loader_identities(loader_two) == expected


def test_distributed_sampler_partitions_virtual_indices_without_overlap() -> None:
    dataset = _mixture(samples_per_epoch=24)
    rank_zero = build_vla_dataloader(
        dataset,
        batch_size_per_rank=3,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        seed=12,
        num_replicas=2,
        rank=0,
    )
    rank_one = build_vla_dataloader(
        dataset,
        batch_size_per_rank=3,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        seed=12,
        num_replicas=2,
        rank=1,
    )

    indices_zero = set(rank_zero.sampler)
    indices_one = set(rank_one.sampler)
    assert indices_zero.isdisjoint(indices_one)
    assert indices_zero | indices_one == set(range(len(dataset)))


def test_distributed_loader_rejects_padding_that_would_duplicate_indices() -> None:
    dataset = _mixture(samples_per_epoch=23)

    with pytest.raises(ValueError, match="divisible"):
        build_vla_dataloader(
            dataset,
            batch_size_per_rank=3,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            seed=12,
            num_replicas=2,
            rank=0,
        )
