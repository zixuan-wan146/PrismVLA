from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.calvin.data import CALVIN_DATA_SPEC
from experiments.libero.data import LIBERO_DATA_SPEC
import prism.training.config as training_config
from prism.data.normalization import canonical_sha256
from prism.data.normalization import compute_statistics
from prism.data.normalization import save_statistics
from prism.training.config import ResolvedTrainConfig
from prism.training.config import build_checkpoint_snapshot
from prism.training.config import load_train_config


def test_calvin_config_resolves_contract_and_complete_checkpoint_snapshot(
    tmp_path: Path,
):
    raw, artifact = _project_fixture(tmp_path, benchmark="calvin")
    raw["data"]["loader"]["global_samples_per_epoch"] = 5
    config_path = _write_config(tmp_path, raw)

    config = load_train_config(config_path, project_root=tmp_path)
    snapshot = config.checkpoint_snapshot()

    assert isinstance(config, ResolvedTrainConfig)
    assert config.project_root == tmp_path.resolve()
    assert config.experiment.output_dir == (tmp_path / "outputs" / "run").resolve()
    assert config.model.architecture.action_head.action_hidden_size == 512
    assert config.model.architecture.action_head.num_attention_heads == 8
    assert config.optimization.language_model.trainable is False
    assert config.optimization.action_queries.learning_rate == pytest.approx(1.0e-4)
    assert config.optimization.action_head.weight_decay == pytest.approx(0.01)
    assert config.data.spec is CALVIN_DATA_SPEC
    assert config.data.train_splits == ("A", "B", "C")
    assert config.data.eval_splits == ("D",)
    assert config.data.datasets[0].splits == ("A", "B", "C")
    assert config.data.loader.global_samples_per_epoch == 5
    assert config.data.loader.persistent_workers is False
    assert config.temporal.action_horizon == 8
    assert config.temporal.replan_stride == 8
    assert config.temporal.history_capture_offsets == (2, 5)
    assert config.temporal.history_step_ages == (6, 3)
    assert config.temporal.num_history_frames == 2
    assert config.temporal.num_ordered_views == 2

    assert snapshot == build_checkpoint_snapshot(config)
    assert snapshot["model"]["architecture"]["action_head"]["objective"] == "direct_masked_l1"
    assert snapshot["model"]["architecture"]["action_head"]["action_hidden_size"] == 512
    assert snapshot["optimization"]["no_decay_rule"] == "bias_and_low_dimensional"
    assert snapshot["optimization"]["vision_encoder"]["trainable"] is False
    assert snapshot["data"]["data_spec"]["action"][-1]["name"] == "action.gripper_open"
    assert snapshot["data"]["normalization"]["statistics"] == artifact
    assert snapshot["data"]["normalization"]["content_sha256"] == artifact["content_sha256"]
    assert snapshot["data"]["data_spec_sha256"] == canonical_sha256(CALVIN_DATA_SPEC)
    assert snapshot["derived"]["source"] == "model.architecture.temporal"
    json.dumps(snapshot, allow_nan=False)

    with pytest.raises(FrozenInstanceError):
        config.experiment.seed = 9
    with pytest.raises(TypeError):
        config.data.normalization.statistics["format"] = "tampered"
    with pytest.raises(TypeError):
        config.data.normalization.statistics["groups"]["calvin_abc"]["robot_key"] = "tampered"


def test_libero_allows_known_explicit_subset_and_preserves_dataset_order(
    tmp_path: Path,
):
    names = ("libero_10", "libero_spatial")
    raw, artifact = _project_fixture(
        tmp_path,
        benchmark="libero",
        dataset_names=names,
    )

    config = load_train_config(
        _write_config(tmp_path, raw),
        project_root=tmp_path,
    )

    assert config.data.spec is LIBERO_DATA_SPEC
    assert config.data.normalization.group == "libero"
    assert tuple(dataset.name for dataset in config.data.datasets) == names
    assert all(dataset.splits is None for dataset in config.data.datasets)
    assert config.data.train_splits is None
    assert config.data.eval_splits is None
    assert artifact["groups"]["libero"]["datasets"] == list(names)


def test_unresolved_architecture_fails_before_training_can_start(tmp_path: Path):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin", resolved=False)

    with pytest.raises(ValueError, match="must resolve every.*action_hidden_size"):
        load_train_config(
            _write_config(tmp_path, raw),
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "pretrained_model", "forbidden"),
        ("model", "objective", "forbidden"),
        ("data", "action_horizon", 8),
        ("trainer", "objective", "direct_masked_l1"),
    ],
)
def test_no_second_model_objective_or_horizon_config_source(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    raw[section][key] = value

    with pytest.raises(ValueError, match=rf"{section} contains unsupported keys.*{key}"):
        load_train_config(
            _write_config(tmp_path, raw),
            project_root=tmp_path,
        )


def test_unknown_missing_and_duplicate_yaml_keys_are_rejected(tmp_path: Path):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    raw["data"]["loader"]["surprise"] = 1
    with pytest.raises(ValueError, match="data.loader contains unsupported keys"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    del raw["trainer"]["save_interval"]
    with pytest.raises(ValueError, match="trainer is missing required keys.*save_interval"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    text = yaml.safe_dump(raw, sort_keys=False)
    text = text.replace("  name: fixture\n", "  name: fixture\n  name: duplicate\n", 1)
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML mapping key 'name'"):
        load_train_config(path, project_root=tmp_path)


@pytest.mark.parametrize(
    ("reference", "error", "message"),
    [
        ("os:path", ValueError, "trusted experiments"),
        (
            "experiments.calvin.data:CALVIN_DATA_SPEC.robot_key",
            ValueError,
            "trusted experiments",
        ),
        (
            "experiments.calvin.data:DOES_NOT_EXIST",
            ImportError,
            "no exact object",
        ),
        (
            "experiments.calvin.data:CALVIN_TRAIN_SPLITS",
            TypeError,
            "must resolve to DataSpec",
        ),
    ],
)
def test_data_spec_import_is_exact_trusted_and_typed(
    tmp_path: Path,
    reference: str,
    error: type[Exception],
    message: str,
):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    raw["data"]["spec"] = reference

    with pytest.raises(error, match=message):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("eval_in_train_splits", "train_splits must be exactly"),
        ("eval_in_dataset", "leaks eval split D"),
        ("missing_training_split", "split union must be exactly"),
        ("missing_dataset_splits", "must explicitly declare splits"),
        ("scene_d_root", "forbidden scene-D training root"),
    ],
)
def test_calvin_split_contract_prevents_scene_d_leakage(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    if mutation == "eval_in_train_splits":
        raw["data"]["train_splits"] = ["A", "B", "C", "D"]
    elif mutation == "eval_in_dataset":
        raw["data"]["datasets"][0]["splits"] = ["A", "B", "C", "D"]
    elif mutation == "missing_training_split":
        raw["data"]["datasets"][0]["splits"] = ["A", "B"]
    elif mutation == "missing_dataset_splits":
        del raw["data"]["datasets"][0]["splits"]
    else:
        (tmp_path / "data" / "task_D_D").mkdir()
        raw["data"]["datasets"][0]["path"] = "task_D_D"

    with pytest.raises(ValueError, match=message):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


def test_calvin_statistics_provenance_must_be_exact(tmp_path: Path):
    raw, _ = _project_fixture(
        tmp_path,
        benchmark="calvin",
        statistics_provenance={
            "train_splits": ["A", "B", "C", "D"],
            "eval_splits": ["D"],
        },
    )

    with pytest.raises(ValueError, match="statistics provenance mismatch"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


def test_statistics_dataset_order_schema_and_robot_are_validated(tmp_path: Path):
    raw, _ = _project_fixture(
        tmp_path,
        benchmark="libero",
        dataset_names=("libero_spatial", "libero_10"),
        statistics_datasets=("libero_10", "libero_spatial"),
    )
    with pytest.raises(ValueError, match="statistics datasets mismatch"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(
        tmp_path,
        benchmark="calvin",
        statistics_schema_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="statistics schema hash mismatch"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(
        tmp_path,
        benchmark="calvin",
        statistics_robot_key="wrong_robot",
    )
    with pytest.raises(ValueError, match="statistics robot_key mismatch"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


def test_libero_rejects_unknown_suite_wrong_group_and_calvin_split_fields(
    tmp_path: Path,
):
    raw, _ = _project_fixture(tmp_path, benchmark="libero")
    (tmp_path / "data" / "libero_unknown").mkdir()
    raw["data"]["datasets"][0]["name"] = "libero_unknown"
    raw["data"]["datasets"][0]["path"] = "libero_unknown"
    with pytest.raises(ValueError, match="unknown suites"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(tmp_path, benchmark="libero")
    raw["data"]["normalization"]["group"] = "libero_spatial"
    with pytest.raises(ValueError, match="group must be 'libero'"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(tmp_path, benchmark="libero")
    raw["data"]["train_splits"] = ["A", "B", "C"]
    raw["data"]["eval_splits"] = ["D"]
    with pytest.raises(ValueError, match="must not declare CALVIN"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


def test_phase_one_loader_rejects_persistent_workers(tmp_path: Path):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    raw["data"]["loader"]["persistent_workers"] = True

    with pytest.raises(ValueError, match="persistent_workers must be false"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


def test_optimization_scope_and_group_values_are_explicit(tmp_path: Path):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    raw["optimization"]["language_model"]["learning_rate"] = 1.0e-5
    with pytest.raises(ValueError, match="must set learning_rate and weight_decay to null when frozen"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    raw["optimization"]["action_head"]["learning_rate"] = None
    with pytest.raises(ValueError, match="must set learning_rate and weight_decay when trainable"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)

    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    for name in (
        "language_model",
        "vision_encoder",
        "action_queries",
        "history_qformer",
        "action_head",
    ):
        raw["optimization"][name] = {
            "trainable": False,
            "learning_rate": None,
            "weight_decay": None,
        }
    with pytest.raises(ValueError, match="at least one parameter group trainable"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("architecture_config", "model.architecture_config must be relative"),
        ("statistics_path", "statistics_path must be relative"),
    ],
)
def test_declared_paths_must_be_relative_to_explicit_project_root(
    tmp_path: Path,
    field: str,
    message: str,
):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    if field == "architecture_config":
        raw["model"][field] = str((tmp_path / "configs" / "architecture.yaml").resolve())
    else:
        raw["data"]["normalization"][field] = str((tmp_path / "artifacts" / "statistics.json").resolve())

    with pytest.raises(ValueError, match=message):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


def test_dataset_path_cannot_escape_data_root(tmp_path: Path):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    (tmp_path / "outside").mkdir()
    raw["data"]["datasets"][0]["path"] = "../outside"

    with pytest.raises(ValueError, match="must remain inside data.root"):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


@pytest.mark.parametrize("contract", ["views", "state", "action"])
def test_dataspec_shape_order_and_action_semantics_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract: str,
):
    raw, _ = _project_fixture(tmp_path, benchmark="calvin")
    raw["data"]["spec"] = "experiments.synthetic.data:SPEC"
    if contract == "views":
        bad_spec = replace(CALVIN_DATA_SPEC, views=tuple(reversed(CALVIN_DATA_SPEC.views)))
        message = "two ordered views"
    elif contract == "state":
        bad_spec = replace(CALVIN_DATA_SPEC, state=CALVIN_DATA_SPEC.state[:-1])
        message = "state dimension must be 8"
    else:
        bad_action = (
            replace(CALVIN_DATA_SPEC.action[0], temporal_semantics="absolute"),
            *CALVIN_DATA_SPEC.action[1:],
        )
        bad_spec = replace(CALVIN_DATA_SPEC, action=bad_action)
        message = "q01_q99 delta continuous"

    original_import = training_config.importlib.import_module

    def import_fixture(name: str):
        if name == "experiments.synthetic.data":
            return SimpleNamespace(SPEC=bad_spec)
        return original_import(name)

    monkeypatch.setattr(training_config.importlib, "import_module", import_fixture)
    with pytest.raises(ValueError, match=message):
        load_train_config(_write_config(tmp_path, raw), project_root=tmp_path)


def _project_fixture(
    root: Path,
    *,
    benchmark: str,
    dataset_names: tuple[str, ...] | None = None,
    resolved: bool = True,
    statistics_datasets: tuple[str, ...] | None = None,
    statistics_provenance: dict | None = None,
    statistics_schema_hash: str | None = None,
    statistics_robot_key: str | None = None,
) -> tuple[dict, dict]:
    (root / "configs").mkdir(exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    data_root = root / "data"
    data_root.mkdir(exist_ok=True)
    architecture_path = root / "configs" / "architecture.yaml"
    architecture_path.write_text(
        yaml.safe_dump(_architecture_values(resolved=resolved), sort_keys=False),
        encoding="utf-8",
    )

    if benchmark == "calvin":
        spec = CALVIN_DATA_SPEC
        names = ("calvin_abc",) if dataset_names is None else dataset_names
        paths = ("task_ABC_D",)
        group = "calvin_abc"
        provenance = (
            {"train_splits": ["A", "B", "C"], "eval_splits": ["D"]}
            if statistics_provenance is None
            else statistics_provenance
        )
    else:
        spec = LIBERO_DATA_SPEC
        names = ("libero_spatial",) if dataset_names is None else dataset_names
        paths = names
        group = "libero"
        provenance = {} if statistics_provenance is None else statistics_provenance

    if len(paths) != len(names):
        paths = names
    for path in paths:
        (data_root / path).mkdir(exist_ok=True)

    statistics_names = names if statistics_datasets is None else statistics_datasets
    artifact = _statistics_artifact(
        spec,
        group=group,
        datasets=statistics_names,
        provenance=provenance,
        schema_hash=statistics_schema_hash,
        robot_key=statistics_robot_key,
    )
    save_statistics(artifact, root / "artifacts" / "statistics.json")

    datasets = []
    for name, path in zip(names, paths, strict=True):
        row = {"name": name, "path": path, "weight": 1.0}
        if benchmark == "calvin":
            row["splits"] = ["A", "B", "C"]
        datasets.append(row)

    raw = {
        "experiment": {
            "name": "fixture",
            "output_dir": "outputs/run",
            "seed": 7,
        },
        "model": {
            "architecture_config": "configs/architecture.yaml",
        },
        "data": {
            "spec": (
                "experiments.calvin.data:CALVIN_DATA_SPEC"
                if benchmark == "calvin"
                else "experiments.libero.data:LIBERO_DATA_SPEC"
            ),
            "root": "data",
            "anchor_stride": 1,
            "include_tail": True,
            "datasets": datasets,
            "normalization": {
                "group": group,
                "statistics_path": "artifacts/statistics.json",
            },
            "loader": {
                "global_samples_per_epoch": 17,
                "batch_size_per_rank": 2,
                "num_workers": 0,
                "preprocessing_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": True,
            },
        },
        "optimization": {
            "optimizer": "adamw",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1.0e-8,
            "no_decay_rule": "bias_and_low_dimensional",
            "language_model": {
                "trainable": False,
                "learning_rate": None,
                "weight_decay": None,
            },
            "vision_encoder": {
                "trainable": False,
                "learning_rate": None,
                "weight_decay": None,
            },
            "action_queries": {
                "trainable": True,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.0,
            },
            "history_qformer": {
                "trainable": True,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.01,
            },
            "action_head": {
                "trainable": True,
                "learning_rate": 1.0e-4,
                "weight_decay": 0.01,
            },
        },
        "trainer": {
            "max_steps": 100,
            "gradient_accumulation_steps": 2,
            "mixed_precision": "bf16",
            "scheduler": "linear_warmup_decay",
            "warmup_steps": 10,
            "max_grad_norm": 1.0,
            "log_interval": 5,
            "save_interval": 20,
        },
    }
    if benchmark == "calvin":
        raw["data"]["train_splits"] = ["A", "B", "C"]
        raw["data"]["eval_splits"] = ["D"]
    return raw, artifact


def _architecture_values(*, resolved: bool) -> dict:
    return {
        "backbone": {
            "model_name": "Qwen/Qwen3.5-0.8B",
            "num_hidden_layers": 16,
            "hidden_size": 1024,
            "num_action_queries": 48,
            "image_size": 384,
            "torch_dtype": "bfloat16",
            "local_files_only": False,
        },
        "history": {
            "input_dim": 1024,
            "hidden_size": 512,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 4,
            "num_memory_tokens": 24,
            "num_history_frames": 2,
            "max_relative_age": 8,
            "dropout": 0.0,
        },
        "temporal": {
            "action_horizon": 8,
            "replan_stride": 8,
            "history_capture_offsets": [2, 5],
        },
        "action_head": {
            "objective": "direct_masked_l1",
            "action_dim": 7,
            "gripper_index": 6,
            "gripper_threshold": 0.5,
            "action_hidden_size": 512 if resolved else None,
            "num_attention_heads": 8 if resolved else None,
            "ffn_ratio": 4 if resolved else None,
        },
        "bridge": {
            "num_layers": 16,
            "memory_gate_init": 0.1,
        },
    }


def _statistics_artifact(
    spec,
    *,
    group: str,
    datasets: tuple[str, ...],
    provenance: dict,
    schema_hash: str | None,
    robot_key: str | None,
) -> dict:
    count = 32
    states = np.arange(count * 8, dtype=np.float64).reshape(count, 8)
    if spec is CALVIN_DATA_SPEC:
        states[:, 6] = 0.0
    actions = np.arange(count * 7, dtype=np.float64).reshape(count, 7)
    actions[:, 6] = np.arange(count) % 2
    state_continuous_indices = tuple(
        index for index, feature in enumerate(spec.state) if feature.normalization == "q01_q99"
    )
    return compute_statistics(
        states,
        actions,
        group=group,
        robot_key=spec.robot_key if robot_key is None else robot_key,
        datasets=datasets,
        schema_hash=canonical_sha256(spec) if schema_hash is None else schema_hash,
        provenance=provenance,
        state_continuous_indices=state_continuous_indices,
    )


def _write_config(root: Path, raw: dict) -> Path:
    path = root / "train.yaml"
    path.write_text(
        yaml.safe_dump(deepcopy(raw), sort_keys=False),
        encoding="utf-8",
    )
    return path
