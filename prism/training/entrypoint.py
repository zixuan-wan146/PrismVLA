from __future__ import annotations

from prism.utils.paths import find_repo_root


class Trainer:
    """Config-dispatched training facade used by thin scripts."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        stage = getattr(getattr(self.cfg, "training", None), "stage", None)
        raw = getattr(self.cfg, "raw", {})
        repo_root = find_repo_root(__file__)
        if stage == "stage1":
            from prism.training.trainer import train_stage1

            return train_stage1(raw, repo_root=repo_root)
        if stage == "stage2":
            from prism.training.trainer import train_stage2

            return train_stage2(raw, repo_root=repo_root)
        raise ValueError(f"Trainer only handles stage1/stage2, got {stage!r}")
