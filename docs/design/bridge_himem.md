# PrismVLA Bridge, Memory, And Planner Design

This document summarizes the active model path after reconstruction. The
behavior-level redesign details are tracked in
`docs/design/prism_architecture_redesign.md`.

## Active Model Path

```text
RGB views + prompt
  -> InternVL3 visual-language embeddings
  -> short visual-token memory
  -> progress-state planner
  -> direct bridge-attention flow-matching action head
```

Retired mechanisms stay retired unless a future design record explicitly brings
them back with tests:

- H64 suffix planner
- transition-trigger memory refresh
- Dual-FIFO long visual memory
- PlanTokenQueue

## Config Entry

Shared model defaults live in:

```text
configs/model/prism_base.yaml
```

Experiment profiles live in:

```text
configs/experiment/
```

Use `scripts/check.sh` after changing config or model contracts.

## Ownership

- `prism/models/memory.py`: short visual-token memory
- `prism/models/planner.py`: progress-state planner and updater
- `prism/models/action_head.py`: direct bridge-attention action head
- `prism/models/policy.py`: top-level wiring
- `prism/config.py`: config schema and loading
- `prism/training/loss.py`: train-time losses

All project-internal paths must stay relative.
