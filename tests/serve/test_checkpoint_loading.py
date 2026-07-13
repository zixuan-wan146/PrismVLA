from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from accelerate import Accelerator

from experiments.calvin.data import CALVIN_DATA_SPEC
from prism.data.normalization import canonical_sha256
from prism.models.config import DirectActionHeadConfig, PrismArchitectureConfig
from prism.serve.loading import load_accelerate_model_weights, load_policy_checkpoint


def _architecture() -> PrismArchitectureConfig:
    return PrismArchitectureConfig(
        action_head=DirectActionHeadConfig(
            action_hidden_size=32,
            num_attention_heads=4,
            ffn_ratio=2,
        )
    )


def test_real_accelerate_model_state_is_strictly_restored(tmp_path: Path) -> None:
    accelerator = Accelerator(cpu=True)
    source = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        source.bias.copy_(torch.tensor([7.0, 8.0]))
    source = accelerator.prepare(source)
    checkpoint = tmp_path / "accelerate-state"
    accelerator.save_state(checkpoint)

    target = nn.Linear(3, 2)
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()
    load_accelerate_model_weights(target, checkpoint)

    assert torch.equal(target.weight, accelerator.unwrap_model(source).weight)
    assert torch.equal(target.bias, accelerator.unwrap_model(source).bias)


def test_policy_loader_reconstructs_embedded_contract_and_loads_same_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    architecture = _architecture()
    snapshot = {
        "model": {"architecture": asdict(architecture)},
        "data": {
            "data_spec": asdict(CALVIN_DATA_SPEC),
            "normalization": {"group": "calvin_abc"},
        },
    }
    metadata = SimpleNamespace(
        resolved_train_snapshot=snapshot,
        architecture_sha256=canonical_sha256(architecture),
        data_spec_sha256=canonical_sha256(CALVIN_DATA_SPEC),
    )
    checkpoint = tmp_path / "checkpoint"
    policy = nn.Linear(1, 1)
    observed: dict[str, object] = {}

    monkeypatch.setattr("prism.serve.loading.read_checkpoint_metadata", lambda path: metadata)

    def build(resolved_architecture, *, state_dim, local_files_only):
        observed["architecture"] = resolved_architecture
        observed["state_dim"] = state_dim
        observed["local_files_only"] = local_files_only
        return policy

    def load_weights(model, path):
        observed["model"] = model
        observed["checkpoint"] = path

    monkeypatch.setattr("prism.serve.loading.build_prism_policy", build)
    monkeypatch.setattr("prism.serve.loading.load_accelerate_model_weights", load_weights)

    loaded = load_policy_checkpoint(
        checkpoint,
        device="cpu",
        local_files_only=True,
    )

    assert loaded.policy is policy
    assert loaded.data_spec == CALVIN_DATA_SPEC
    assert loaded.statistics_group == "calvin_abc"
    assert loaded.checkpoint_path == checkpoint.resolve()
    assert observed == {
        "architecture": architecture,
        "state_dim": CALVIN_DATA_SPEC.state_dim,
        "local_files_only": True,
        "model": policy,
        "checkpoint": checkpoint.resolve(),
    }
    assert not policy.training
    assert all(not parameter.requires_grad for parameter in policy.parameters())
