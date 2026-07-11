# PrismVLA Architecture Redesign Notes

Status: design record, details pending
Created: 2026-07-10

This document records the new PrismVLA design decisions that intentionally
change the legacy bridge-VLA architecture. Keep this file updated before or in
the same change as implementation work so future migrations can distinguish
intended redesigns from accidental behavior drift.

## Scope

The redesign scope is limited to three model components:

- short memory mechanism
- progress-state planner
- bridge-attention action head

All other restructuring work should follow `reconstruct.md` unless this file
explicitly updates a contract.

## Legacy Baseline

The current legacy bridge-VLA active path is documented in:

- `docs/bridge_himem_design.md`
- `docs/progress_state_planner_design_zh.md`
- `docs/direct_bridge_attention_design_zh.md`

The legacy model path is:

```text
RGB views + prompt
  -> InternVL3
  -> current VLM hidden states and pooled VL summary
  -> short visual-token memory
  -> progress-state planner
  -> direct bridge-attn flow-matching action head
```

Legacy assumptions to revisit during the redesign:

- short memory stores recent visual-token evidence only
- planner long memory is task-progress state, not a visual FIFO bank
- bridge-attn uses action tokens as queries
- bridge-attn groups sources into visual evidence and action-condition branches
- action horizon `H` and replan stride `R` are fixed by config

Retired mechanisms should stay retired unless a new design decision explicitly
brings them back with tests and documentation:

- H64 suffix planner
- transition-trigger memory refresh
- Dual-FIFO long visual memory
- `PlanTokenQueue`

## Design Principles

- Keep the module boundaries from `reconstruct.md`.
- Put neural components in `prism.models`, not in scripts or training loops.
- Keep changeable behavior in config fields, not hard-coded branches.
- Preserve explicit tensor contracts: shapes, masks, update cadence, and reset
  semantics must be documented before implementation.
- Treat this as a behavior change, not only a file move. Add tests for each
  changed contract.

## Target Ownership

The redesign should land in these PrismVLA modules:

| Concern | Module |
|---|---|
| Short memory | `prism/models/memory.py` |
| Progress-state planner | `prism/models/planner.py` |
| Bridge-attention action head | `prism/models/action_head.py` |
| Top-level wiring | `prism/models/policy.py` |
| Config schema | `prism/config.py` |
| Losses and auxiliary supervision | `prism/training/loss.py` |
| Dataset-side segment and mask support | `prism/data/segments.py` |

Do not create new top-level subpackages for this redesign.

## Shared Interface Contract

The three redesigned components should agree on one explicit interface.

Core inputs:

```text
F_t: current VLM token hidden states
h_t: current VLM pooled or summary representation
s_t: current robot/proprio state
u_t: previous executed action segment summary, if used
M_{t-1}: previous planner state, if recurrent state is used
S_{t-1}: previous short-memory state, if recurrent memory is used
```

Core outputs:

```text
S_t: short-memory tokens or state
M_t: planner state
P_t: plan/action-condition tokens
A_t: predicted action horizon or flow-matching velocity
aux: optional diagnostic tensors and auxiliary losses
```

Before implementation, each component must define:

- tensor shapes
- dtype and device expectations
- masks and padding behavior
- reset behavior at episode boundaries
- update cadence relative to low-level control steps
- train-time versus inference-time differences

## 1. Short Memory Redesign

Status: pending final design.

Legacy behavior:

```text
S_t = ShortVisualMemory(V_{t-R/2}, V_{t-R})
```

The legacy short memory is local visual continuity only. It should not encode
task progress unless this redesign changes that contract explicitly.

New design decisions to record:

- What does short memory store: tokens, compressed slots, recurrent state, or
  another representation?
- How is memory written: fixed stride, every observation, learned gate, or
  externally triggered update?
- How is memory read: concatenation, cross-attention, key-value cache, pooling,
  or another mechanism?
- What is the capacity and eviction policy?
- Does memory reset only at episode boundaries, or also at subtask boundaries?
- Which inputs can write into memory: VLM hidden states, pooled summary,
  proprio state, action history, planner state?
- Which modules may read memory: planner, action head, both, or policy only?
- What supervision or regularization applies to memory?

Implementation contract to fill before coding:

```text
ShortMemory.forward(
    current_tokens,
    current_summary,
    robot_state,
    previous_memory,
    masks,
) -> memory_output, next_memory, aux
```

Test requirements:

- episode reset clears or reinitializes memory deterministically
- batch padding does not leak memory across episodes
- memory output shape is stable across train and inference
- configured capacity is enforced

## 2. Progress-State Planner Redesign

Status: pending final design.

Legacy behavior:

```text
x_t = ProgressEvidenceEncoder(h_t, s_t, u_t)
M_t = ProgressStateUpdater(M_{t-1}, x_t)
P_t = ProgressPlanner(M_t, h_t, s_t)
```

The legacy planner treats long memory as task-progress state rather than a
growing visual-token bank.

New design decisions to record:

- What is the planner state representation?
- Is planner state recurrent, windowed, token-slot based, or stateless?
- Which evidence updates state: VLM summary, short memory, proprio state,
  action history, success signals, language tokens?
- How many plan tokens are emitted, and what does each token mean?
- Does the planner output only conditioning tokens, or also explicit progress
  predictions?
- How does planner update cadence relate to action horizon and replan stride?
- Which auxiliary losses supervise progress state or plan tokens?
- What state is serialized for runtime inference?

Implementation contract to fill before coding:

```text
Planner.forward(
    vl_summary,
    robot_state,
    previous_actions,
    short_memory,
    previous_planner_state,
    masks,
) -> plan_tokens, next_planner_state, aux
```

Test requirements:

- planner state resets correctly at episode boundaries
- planner output token count matches config
- update cadence works for both warmup/training and inference
- auxiliary losses handle missing labels or masks explicitly

## 3. Bridge-Attention Redesign

Status: pending final design.

Legacy behavior:

```text
action tokens
  -> action self-attention
  -> visual cross-attention over current VLM hidden states and short memory
  -> action-condition cross-attention over plan tokens and state token
  -> flow-matching action prediction
```

New design decisions to record:

- Which sources are exposed to bridge-attention?
- Are sources grouped into branches, fused first, gated, or attended
  independently?
- Which tokens query which sources: action tokens, plan tokens, memory tokens,
  or a learned bridge sequence?
- What is the VLM hidden-state layer schedule?
- How are short-memory tokens aligned with current VLM tokens?
- How are plan tokens and robot state injected?
- Where does flow-matching time conditioning enter?
- Which masks are required for variable-length sources?

Implementation contract to fill before coding:

```text
ActionHead.forward(
    noisy_actions,
    flow_time,
    current_vlm_tokens,
    short_memory,
    plan_tokens,
    robot_state,
    masks,
) -> action_velocity, aux
```

Test requirements:

- action horizon and action dimension follow config
- branch/source masks prevent attention to padded tokens
- train-time flow target and inference-time integration use the same contract
- removing any optional source has a defined behavior

## Config Fields To Add Or Confirm

The final design should be expressible through `prism.config` and experiment
YAMLs. Candidate fields:

```text
model.architecture_version
model.hidden_dim
model.action_horizon
model.replan_stride
model.memory.kind
model.memory.capacity
model.memory.update_stride
model.planner.kind
model.planner.state_tokens
model.planner.plan_tokens
model.action_head.kind
model.action_head.vlm_layers
training.loss_weights.*
runtime.inference_steps
```

Do not read environment variables or hard-code experiment switches inside model
or training modules.

## Acceptance Criteria For Final Design

The redesign is ready for implementation when this document records:

- exact tensor shapes for each component input and output
- update cadence for memory and planner
- source grouping and mask semantics for bridge-attention
- config field names and defaults
- train-time losses and inference-time behavior
- migration notes from the legacy bridge-VLA implementation
- focused tests required for the changed behavior

## Change Log

- 2026-07-10: created design record for the upcoming PrismVLA short memory,
  progress-state planner, and bridge-attention redesign.
