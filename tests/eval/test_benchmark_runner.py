from prism.config import load_config
from prism.eval.runner import BenchmarkRunner, parse_profile_env


def test_parse_profile_env_ignores_comments_and_export_prefix():
    assert parse_profile_env(
        """
        # comment
        export PRISM_LIBERO_EPISODES=1
        PRISM_MUJOCO_GL=osmesa
        """
    ) == {
        "PRISM_LIBERO_EPISODES": "1",
        "PRISM_MUJOCO_GL": "osmesa",
    }


def test_libero_eval_smoke_config_dry_run(capsys):
    cfg = load_config("configs/experiment/libero_smoke.yaml")
    assert BenchmarkRunner.from_config(cfg).run() == 0
    assert "libero eval dry-run ok" in capsys.readouterr().out


def test_calvin_eval_smoke_config_dry_run(capsys):
    cfg = load_config("configs/experiment/calvin_smoke.yaml")
    assert BenchmarkRunner.from_config(cfg).run() == 0
    assert "calvin eval dry-run ok" in capsys.readouterr().out
