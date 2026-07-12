from types import SimpleNamespace

import pytest

from prism.serve.server import server_argv_from_config
from prism.serve.server import parse_args


def test_server_argv_from_config_maps_values_and_flags():
    cfg = SimpleNamespace(
        raw={
            "ckpt_dir": "local_data/checkpoints/policy",
            "host": "127.0.0.1",
            "port": 9100,
            "device": "cuda:0",
            "inference_steps": 8,
            "vlm_name": "OpenGVLab/InternVL3-1B",
            "vlm_local_files_only": True,
            "allow_unsafe_checkpoint_load": False,
        }
    )

    assert server_argv_from_config(cfg) == [
        "--ckpt_dir",
        "local_data/checkpoints/policy",
        "--host",
        "127.0.0.1",
        "--port",
        "9100",
        "--device",
        "cuda:0",
        "--inference_steps",
        "8",
        "--vlm_name",
        "OpenGVLab/InternVL3-1B",
        "--vlm_local_files_only",
    ]


def test_server_cli_defaults_to_loopback():
    args = parse_args(["--ckpt_dir", "local_data/checkpoints/policy"])
    assert args.host == "127.0.0.1"


def test_server_argv_from_config_requires_checkpoint():
    with pytest.raises(ValueError, match="ckpt_dir"):
        server_argv_from_config(SimpleNamespace(raw={}))
