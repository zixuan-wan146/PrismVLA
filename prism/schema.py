from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PolicyInput:
    """Model-facing input shared by training and inference.

    External transports remain responsible for validating and canonicalizing
    their wire payloads before constructing this contract.
    """

    benchmark: str
    prompt: str
    images_by_view: Mapping[str, np.ndarray]
    history_images_by_view: Mapping[str, np.ndarray]
    history_step_ages: np.ndarray
    history_valid_mask: np.ndarray
    state: np.ndarray
    action_dim: int
    robot_key: str | None = None
