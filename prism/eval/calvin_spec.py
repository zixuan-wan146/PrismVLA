from __future__ import annotations

# --- migrated from src/prism/benchmarks/calvin/spec.py ---
from prism.eval.runner import BenchmarkSpec


CALVIN_SPEC = BenchmarkSpec(
    name="calvin",
    view_names=("image", "wrist_image"),
    state_dim=8,
    action_dim=7,
    short_memory_offsets=(16, 8),
    replan_stride=16,
)

