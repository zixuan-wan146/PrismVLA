from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import prism.serve.backend as backend_module
from experiments.calvin.data import CALVIN_DATA_SPEC
from prism.data.normalization import (
    DataSpecNormalizer,
    canonical_sha256,
    compute_statistics,
    denormalize_action,
    normalize_action,
)
from prism.models.batch import PolicyBatchCollator, PolicyInferenceBatch
from prism.models.config import DirectActionHeadConfig, PrismArchitectureConfig
from prism.models.vlm import PreparedQueryMemoryBatch
from prism.serve.backend import CheckpointPolicyBackend
from prism.serve.protocol import PolicyRequest
from prism.training.checkpoint import CheckpointMetadata, TrainingProgress


STATISTICS_GROUP = "calvin_abc"
DATASET_NAME = "task_ABC_D"


def _architecture() -> PrismArchitectureConfig:
    return PrismArchitectureConfig(
        action_head=DirectActionHeadConfig(
            action_hidden_size=32,
            num_attention_heads=4,
            ffn_ratio=2,
        )
    )


def _statistics(
    *,
    robot_key: str = "calvin",
    schema_hash: str | None = None,
) -> dict:
    states = np.asarray(
        [
            [-2.0, -1.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.01],
            [-1.0, 0.0, 1.0, 0.2, 0.3, 0.4, 0.0, 0.02],
            [0.0, 1.0, 2.0, 0.3, 0.4, 0.5, 0.0, 0.03],
            [1.0, 2.0, 3.0, 0.4, 0.5, 0.6, 0.0, 0.04],
        ],
        dtype=np.float32,
    )
    actions = np.asarray(
        [
            [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.0],
            [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 1.0],
            [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 1.0],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    return compute_statistics(
        states,
        actions,
        group=STATISTICS_GROUP,
        robot_key=robot_key,
        datasets=(DATASET_NAME,),
        schema_hash=(canonical_sha256(CALVIN_DATA_SPEC) if schema_hash is None else schema_hash),
        state_continuous_indices=(0, 1, 2, 3, 4, 5, 7),
    )


def _metadata(
    statistics: dict,
    architecture: PrismArchitectureConfig,
) -> CheckpointMetadata:
    data_spec_hash = canonical_sha256(CALVIN_DATA_SPEC)
    snapshot = {
        "model": {"architecture": asdict(architecture)},
        "data": {
            "data_spec": asdict(CALVIN_DATA_SPEC),
            "datasets": [{"name": DATASET_NAME}],
            "normalization": {
                "group": STATISTICS_GROUP,
                "statistics": statistics,
            },
        },
    }
    return CheckpointMetadata(
        created_at_utc="2026-07-13T00:00:00+00:00",
        world_size=1,
        progress=TrainingProgress(
            completed_optimizer_steps=1,
            gradient_accumulation_micro_step=0,
            epoch=0,
            virtual_sample_cursor=1,
            virtual_batch_cursor=1,
        ),
        resolved_train_snapshot=snapshot,
        resolved_train_snapshot_sha256=canonical_sha256(snapshot),
        architecture_sha256=canonical_sha256(architecture),
        data_spec_sha256=data_spec_hash,
        statistics_sha256=statistics["content_sha256"],
        git={},
        environment={},
        rng_rank_files=(),
    )


def _prepared_batch(batch_size: int) -> PreparedQueryMemoryBatch:
    return PreparedQueryMemoryBatch(
        current_inputs={
            "input_ids": torch.zeros(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
            "pixel_values": torch.zeros(batch_size * 2, 3, 4, 4),
            "image_grid_thw": torch.ones(batch_size * 2, 3, dtype=torch.long),
        },
        history_inputs={
            "pixel_values": torch.zeros(batch_size * 4, 3, 4, 4),
            "image_grid_thw": torch.ones(batch_size * 4, 3, dtype=torch.long),
        },
        history_step_ages=torch.tensor([[6, 3]]).expand(batch_size, -1).clone(),
        history_valid_mask=torch.ones(batch_size, 2, dtype=torch.bool),
    )


class _PrepareEncoder:
    def __init__(self) -> None:
        self.requests: list[PolicyRequest] = []

    def prepare_requests(
        self,
        requests: tuple[PolicyRequest, ...],
    ) -> PreparedQueryMemoryBatch:
        self.requests.extend(requests)
        return _prepared_batch(len(requests))


class _LoadedPolicyFixture(nn.Module):
    def __init__(
        self,
        architecture: PrismArchitectureConfig,
        normalized_prediction: np.ndarray,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.state_dim = CALVIN_DATA_SPEC.state_dim
        self.anchor = nn.Parameter(torch.zeros(()))
        self.encoder = _PrepareEncoder()
        self.normalized_prediction = torch.from_numpy(normalized_prediction).unsqueeze(0)
        self.prediction_batches: list[PolicyInferenceBatch] = []
        self.prediction_devices: list[set[torch.device]] = []

    def make_collator(self) -> PolicyBatchCollator:
        return PolicyBatchCollator(
            self.architecture,
            self.encoder,
            state_dim=self.state_dim,
        )

    def predict(self, batch: PolicyInferenceBatch) -> torch.Tensor:
        self.prediction_batches.append(batch)
        tensors = [
            *batch.current_inputs.values(),
            *batch.history_inputs.values(),
            batch.history_step_ages,
            batch.history_valid_mask,
            batch.state,
        ]
        self.prediction_devices.append({tensor.device for tensor in tensors})
        return self.normalized_prediction + self.anchor * 0.0


def _request(**changes) -> PolicyRequest:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    request = PolicyRequest(
        benchmark="calvin",
        prompt="move the block",
        images_by_view={"primary": image, "wrist": image.copy()},
        history_images_by_view={
            "primary": np.zeros((2, 4, 5, 3), dtype=np.uint8),
            "wrist": np.zeros((2, 4, 5, 3), dtype=np.uint8),
        },
        history_step_ages=np.asarray([6, 3], dtype=np.int32),
        history_valid_mask=np.asarray([True, False], dtype=np.bool_),
        state=np.asarray([-0.5, 0.5, 1.5, 0.25, 0.35, 0.45, 0.0, 0.025], dtype=np.float32),
        action_dim=7,
        robot_key="calvin",
        return_debug=True,
    )
    return replace(request, **changes)


def _normalized_prediction(statistics: dict) -> np.ndarray:
    canonical_actions = np.tile(
        np.asarray([-0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.0], dtype=np.float32),
        (8, 1),
    )
    group = statistics["groups"][STATISTICS_GROUP]
    prediction = normalize_action(canonical_actions, group)
    prediction[:, 6] = np.asarray(
        [0.0, 0.5, 0.5001, 1.0, 0.49, 0.9, -0.2, 1.2],
        dtype=np.float32,
    )
    return prediction


def _backend(monkeypatch: pytest.MonkeyPatch):
    statistics = _statistics()
    architecture = _architecture()
    metadata = _metadata(statistics, architecture)
    policy = _LoadedPolicyFixture(
        architecture,
        _normalized_prediction(statistics),
    )
    seen_paths: list[Path] = []

    def read_metadata(path):
        seen_paths.append(Path(path))
        return metadata

    monkeypatch.setattr("prism.serve.backend.read_checkpoint_metadata", read_metadata)
    backend = CheckpointPolicyBackend.from_checkpoint(
        "checkpoints/step-00000001",
        loaded_policy=policy,
        data_spec=CALVIN_DATA_SPEC,
        statistics_group=STATISTICS_GROUP,
    )
    return backend, policy, statistics, seen_paths


def test_checkpoint_backend_normalizes_predicts_and_denormalizes(monkeypatch):
    move_calls: list[torch.device] = []
    move_batch = backend_module._move_inference_batch_to_device

    def recording_move(batch, device):
        move_calls.append(device)
        return move_batch(batch, device)

    monkeypatch.setattr(
        backend_module,
        "_move_inference_batch_to_device",
        recording_move,
    )
    backend, policy, statistics, seen_paths = _backend(monkeypatch)
    request = _request()

    response = backend.infer(request)

    assert seen_paths == [Path("checkpoints/step-00000001")]
    assert len(policy.encoder.requests) == 1
    normalized_request = policy.encoder.requests[0]
    expected_state = DataSpecNormalizer(
        CALVIN_DATA_SPEC,
        statistics,
        STATISTICS_GROUP,
    ).normalize_state(request.state)
    np.testing.assert_allclose(normalized_request.state, expected_state)
    assert len(policy.prediction_batches) == 1
    assert isinstance(policy.prediction_batches[0], PolicyInferenceBatch)
    assert not hasattr(policy.prediction_batches[0], "target_actions")
    assert move_calls == [policy.anchor.device]
    assert policy.prediction_devices == [{policy.anchor.device}]

    prediction = policy.normalized_prediction.detach().numpy()[0]
    expected_actions = denormalize_action(
        prediction,
        statistics["groups"][STATISTICS_GROUP],
    )
    np.testing.assert_allclose(response["actions"], expected_actions)
    np.testing.assert_array_equal(response["actions"][:, 6], prediction[:, 6])
    assert response["actions"][1, 6] == pytest.approx(0.5)
    assert response["actions"][2, 6] > 0.5
    assert response["debug"]["statistics_sha256"] == statistics["content_sha256"]
    assert backend.metadata["weights_state"] == "injected_loaded_policy"
    assert backend.metadata["device"] == str(policy.anchor.device)
    assert backend.metadata["ordered_views"] == ("primary", "wrist")


def test_checkpoint_backend_rejects_contract_mismatches(monkeypatch):
    backend, _, _, _ = _backend(monkeypatch)

    with pytest.raises(ValueError, match="benchmark"):
        backend.infer(_request(benchmark="libero"))
    with pytest.raises(ValueError, match="robot_key"):
        backend.infer(_request(robot_key=None))
    with pytest.raises(ValueError, match="ordered views"):
        backend.infer(
            _request(
                images_by_view={
                    "image": np.zeros((4, 5, 3), dtype=np.uint8),
                    "wrist_image": np.zeros((4, 5, 3), dtype=np.uint8),
                }
            )
        )
    with pytest.raises(ValueError, match="state"):
        backend.infer(_request(state=np.zeros(7, dtype=np.float32)))
    with pytest.raises(ValueError, match="action_dim"):
        backend.infer(_request(action_dim=6))


def test_checkpoint_factory_rejects_group_and_statistics_hash_drift(monkeypatch):
    statistics = _statistics()
    architecture = _architecture()
    metadata = _metadata(statistics, architecture)
    policy = _LoadedPolicyFixture(architecture, _normalized_prediction(statistics))
    monkeypatch.setattr(
        "prism.serve.backend.read_checkpoint_metadata",
        lambda path: metadata,
    )

    with pytest.raises(ValueError, match="statistics group"):
        CheckpointPolicyBackend.from_checkpoint(
            "checkpoint",
            loaded_policy=policy,
            data_spec=CALVIN_DATA_SPEC,
            statistics_group="wrong_group",
        )

    corrupted = replace(metadata, statistics_sha256="0" * 64)
    monkeypatch.setattr(
        "prism.serve.backend.read_checkpoint_metadata",
        lambda path: corrupted,
    )
    with pytest.raises(ValueError, match="content_sha256"):
        CheckpointPolicyBackend.from_checkpoint(
            "checkpoint",
            loaded_policy=policy,
            data_spec=CALVIN_DATA_SPEC,
            statistics_group=STATISTICS_GROUP,
        )


def test_checkpoint_factory_requires_a_parameter_device(monkeypatch):
    statistics = _statistics()
    architecture = _architecture()
    metadata = _metadata(statistics, architecture)
    policy = _LoadedPolicyFixture(architecture, _normalized_prediction(statistics))
    del policy.anchor
    monkeypatch.setattr(
        "prism.serve.backend.read_checkpoint_metadata",
        lambda path: metadata,
    )

    with pytest.raises(ValueError, match="at least one parameter"):
        CheckpointPolicyBackend.from_checkpoint(
            "checkpoint",
            loaded_policy=policy,
            data_spec=CALVIN_DATA_SPEC,
            statistics_group=STATISTICS_GROUP,
        )


def test_checkpoint_factory_rejects_statistics_robot_drift(monkeypatch):
    statistics = _statistics(robot_key="libero")
    architecture = _architecture()
    metadata = _metadata(statistics, architecture)
    policy = _LoadedPolicyFixture(architecture, _normalized_prediction(statistics))
    monkeypatch.setattr(
        "prism.serve.backend.read_checkpoint_metadata",
        lambda path: metadata,
    )

    with pytest.raises(ValueError, match="robot_key mismatch"):
        CheckpointPolicyBackend.from_checkpoint(
            "checkpoint",
            loaded_policy=policy,
            data_spec=CALVIN_DATA_SPEC,
            statistics_group=STATISTICS_GROUP,
        )


def test_checkpoint_factory_rejects_statistics_schema_drift(monkeypatch):
    statistics = _statistics(schema_hash="1" * 64)
    architecture = _architecture()
    metadata = _metadata(statistics, architecture)
    policy = _LoadedPolicyFixture(architecture, _normalized_prediction(statistics))
    monkeypatch.setattr(
        "prism.serve.backend.read_checkpoint_metadata",
        lambda path: metadata,
    )

    with pytest.raises(ValueError, match="schema hash mismatch"):
        CheckpointPolicyBackend.from_checkpoint(
            "checkpoint",
            loaded_policy=policy,
            data_spec=CALVIN_DATA_SPEC,
            statistics_group=STATISTICS_GROUP,
        )
