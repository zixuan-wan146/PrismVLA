from experiments.calvin.eval import main as calvin_main
from experiments.libero.eval import main as libero_main
from prism.config import parse_profile_env


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
    assert libero_main(["--config", "experiments/libero/configs/smoke.yaml"]) == 0
    assert "libero eval dry-run ok" in capsys.readouterr().out


def test_calvin_eval_smoke_config_dry_run(capsys):
    assert calvin_main(["--config", "experiments/calvin/configs/smoke.yaml"]) == 0
    assert "calvin eval dry-run ok" in capsys.readouterr().out
