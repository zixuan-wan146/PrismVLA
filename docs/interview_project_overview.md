# PrismVLA Interview Project Overview

Created: 2026-07-10

This document is a practical interview-preparation guide for explaining
PrismVLA. It summarizes the motivation, architecture, implementation layout,
training and evaluation workflow, verification status, and the most likely
technical questions.

## 1. Short Pitch

PrismVLA is a reconstructed vision-language-action research codebase for robot
learning. It takes the active `himem-bridge-vla` project path and restructures
it into a maintainable Python package named `prism`, while preserving the
scientific behavior of the current model, data, training, evaluation, and
checkpoint contracts.

The active model path is:

```text
RGB views + language prompt
  -> InternVL3 visual-language embeddings
  -> short visual-token memory
  -> progress-state planner
  -> direct bridge-attention flow-matching action head
  -> action horizon prediction
```

Target benchmarks are LIBERO and CALVIN. RMbench is explicitly abandoned for
this project unless restored later as a separate benchmark.

## 2. What Problem The Project Solves

The project addresses two linked problems:

1. Robot VLA modeling:
   - Use a VLM backbone to process image observations and language.
   - Maintain short-term visual memory for recent observations.
   - Maintain task progress state instead of an unbounded visual-token history.
   - Predict robot actions through a bridge-attention flow-matching action head.

2. Research-code maintainability:
   - Convert a legacy, over-nested project into a flat package layout.
   - Keep scripts thin and move real logic into `prism/`.
   - Preserve checkpoint formats, config fields, dataset contracts, and result
     formats while making modules reviewable.

A good interview framing:

```text
I reconstructed a VLA robot learning codebase into a maintainable package while
preserving its scientific behavior. The core research idea is to combine current
VLM hidden states, recent visual-token memory, and an explicit progress-state
planner, then feed those signals into a direct bridge-attention flow-matching
action head for long-horizon robot control.
```

## 3. Repository Layout

Top-level structure:

```text
PrismVLA/
  prism/                  Python package
  configs/                YAML configs
  scripts/                thin CLI entry points
  tests/                  pytest coverage
  docs/                   design, engineering, benchmark docs
  examples/               runnable examples
  third_party/            reference repos and papers
  reconstruct.md          migration guideline
```

Package structure:

```text
prism/
  config.py               stable config facade
  config_bridge.py        bridge/model schema
  config_experiment.py    bridge YAML resolution into legacy training dicts
  config_loader.py        top-level PrismConfig loader
  config_runtime.py       runtime constants and array helpers
  config_training.py      training config defaults, merging, validation

  models/
    policy.py             top-level PrismPolicy wiring
    vlm.py                InternVL3 wrapper
    memory.py             short visual-token memory
    planner.py            progress-state planner
    bridge_attention.py   bridge attention block
    bridge_adapter.py     bridge adapter
    flow_matching_head.py flow-matching action head
    action_head.py        backward-compatible facade

  data/
    libero.py             LIBERO readers and warmup data
    calvin.py             CALVIN readers and warmup data
    segments.py           action segment utilities
    memory_replay.py      replay-index sample construction
    replay_frames.py      raw replay frame reading
    replay_dataset.py     frame replay dataset
    token_cache_*         token cache build, IO, datasets, core dataclasses
    smoke.py              tiny deterministic train smoke cache
    cache.py              backward-compatible cache facade

  training/
    trainer.py            backward-compatible trainer facade
    entrypoint.py         config-dispatched Trainer
    stage1.py             Stage1 token-cache training
    stage2_data.py        Stage2 raw-episode dataset and collation
    stage2_loop.py        Stage2 full end-to-end training loop
    stage2_config.py      Stage2 CLI/config/contract handling
    warmup.py             progress planner warmup loop
    loss.py               training losses
    checkpointing.py      save/load checkpoint format
    optim.py              param groups
    scheduler.py          LR schedule
    distributed.py        accelerator helpers
    loggers.py            file logging

  eval/
    runner.py             benchmark dispatch facade
    profiles.py           profile_env parsing, dry-run helpers
    metadata.py           result metadata
    libero.py             LIBERO eval path
    calvin_*.py           CALVIN protocol/config/summary/runner modules
    calvin.py             backward-compatible CALVIN facade

  serve/
    engine.py             inference engine and runtime memory inputs
    server.py             websocket protocol server

  utils/
    paths.py              project-relative path helpers
    seeding.py            deterministic seeding
    logging.py            shared logging helpers
```

Important maintainability point:

```text
No Python module under prism/ is currently over 1,000 lines after the refactor.
The old large files remain only as stable import facades where needed.
```

## 4. Thin Script Policy

The scripts are intentionally thin:

```text
scripts/build_cache.py
scripts/warmup.py
scripts/train.py
scripts/eval.py
scripts/serve.py
scripts/check.sh
```

They only parse CLI arguments, load configs, and call package logic. This keeps
business logic in `prism/`, which makes it testable and reusable.

Canonical commands:

```bash
python scripts/build_cache.py --config configs/experiment/libero_stage1.yaml
python scripts/warmup.py --config configs/experiment/libero_warmup_w4.yaml
python scripts/train.py --config configs/experiment/libero_stage1.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
```

Smoke commands:

```bash
python scripts/build_cache.py --config configs/experiment/libero_train_smoke.yaml
python scripts/train.py --config configs/experiment/libero_train_smoke.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

## 5. Model Architecture

### 5.1 VLM Backbone

The model uses an InternVL3 wrapper in `prism/models/vlm.py`.

Responsibilities:

- preprocess image views
- run the VLM backbone
- return fused image/text embeddings
- return selected hidden-state layers for bridge attention
- provide a planner summary token when available

The active VLM hidden-state layers used by bridge attention are:

```text
3, 6, 9, 12
```

These are token sequences, not pooled vectors.

### 5.2 Short Visual-Token Memory

Implemented in `prism/models/memory.py`.

Current concept:

```text
short memory = recent visual-token evidence
```

It is separate from long-term task progress. It stores recent visual evidence,
usually with fixed offsets such as:

```text
R = 16
short offsets = (16, 8)
```

Interview explanation:

```text
I intentionally keep short memory as local visual continuity, not as a task
progress representation. This prevents the model from treating a historical
token FIFO as long-term memory. Long-term task state is handled by the planner.
```

### 5.3 Progress-State Planner

Implemented in `prism/models/planner.py`.

Current concept:

```text
long memory = task-progress state
```

The planner keeps two progress-state tokens:

```text
C_k: completed-events state token
G_k: current-stage state token
```

Planner input signals:

- VLM summary token
- robot/proprio state
- previous executed action segment

Planner outputs:

- updated progress state
- plan/action-condition tokens used by the action head

Fixed time scales:

```text
H = 32 action horizon
R = 16 replan stride
```

A concise explanation:

```text
Instead of maintaining an unbounded visual-token bank, the planner maintains a
compact recurrent task-progress state. This makes long-horizon task structure
explicit and keeps memory bounded.
```

### 5.4 Direct Bridge-Attention Action Head

Core modules:

```text
prism/models/bridge_attention.py
prism/models/bridge_adapter.py
prism/models/flow_matching_head.py
```

The action head uses action tokens as the query sequence. It attends to two
functional branches:

```text
visual evidence branch:
  current VLM hidden states
  short memory tokens

action-condition branch:
  plan tokens
  robot state token
```

It does not implement four separate cross-attention branches or four separate
source-level gates. The current design deliberately groups sources into two
branches for clarity and stability.

Flow matching:

- Training samples a noise-action interpolation.
- The model predicts velocity from noise to ground-truth action.
- Inference integrates velocity over a fixed number of midpoint Euler steps.

Important config defaults:

```text
horizon = 32
per_action_dim = 7
num_inference_timesteps = 15
inference_tau_schedule = midpoint
avoid_endpoint_tau = true
```

## 6. Data Pipeline

### 6.1 Benchmark Targets

Active:

- LIBERO
- CALVIN

Inactive:

- RMbench

### 6.2 LIBERO And CALVIN Readers

Files:

```text
prism/data/libero.py
prism/data/calvin.py
```

They provide benchmark-specific episode/frame/action reading and conversion
into common internal contracts.

### 6.3 Replay And Token Cache

The data pipeline separates raw episode reading from token-cache training.

Important modules:

```text
prism/data/memory_replay.py
prism/data/replay_frames.py
prism/data/replay_dataset.py
prism/data/token_cache_build.py
prism/data/token_cache_io.py
prism/data/token_cache_dataset.py
prism/data/token_cache_core.py
```

Why token caches matter:

- Stage1 can train the bridge/action head without repeatedly running the VLM.
- Caches preserve selected VLM hidden states and planner summaries.
- Training becomes cheaper and reproducible.

Stage1 active cache format:

```text
libero_episode_feature_cache
```

The smoke cache builder in `prism/data/smoke.py` creates a tiny deterministic
episode-feature cache for fast command verification.

## 7. Training Pipeline

### 7.1 Warmup

Entry:

```bash
python scripts/warmup.py --config configs/experiment/libero_warmup_w4.yaml
python scripts/warmup.py --config configs/experiment/calvin_warmup_w4.yaml
```

Purpose:

- train or verify the progress-state planner warmup path
- produce planner checkpoints consumed by Stage1

Implementation:

```text
prism/training/warmup.py
```

### 7.2 Stage1

Entry:

```bash
python scripts/train.py --config configs/experiment/libero_stage1.yaml
python scripts/train.py --config configs/experiment/calvin_stage1.yaml
```

Implementation:

```text
prism/training/stage1.py
```

Stage1 trains from episode feature/token cache:

- `load_vlm = false`
- `finetune_vlm = false`
- `finetune_action_head = true`
- `finetune_progress_planner = false`
- frozen progress planner checkpoint required
- masked flow-matching velocity loss only

Main contract:

```text
episode feature cache
  -> trajectory/node dataloader
  -> update frozen progress-state planner through trajectory
  -> compute flow-matching loss on full-horizon nodes
  -> save stage1_torch_checkpoint
```

### 7.3 Stage2

Implementation:

```text
prism/training/stage2_data.py
prism/training/stage2_loop.py
prism/training/stage2_config.py
```

Stage2 is designed for full end-to-end raw-episode training:

- `load_vlm = true`
- `finetune_vlm = true`
- `finetune_action_head = true`
- `progress_planner_enabled = true`
- `finetune_progress_planner = true`
- raw replay episode dataset

Current Stage2 path is preserved and modularized, but should be treated as the
higher-risk training path because it touches VLM, planner, memory, and action
head together.

### 7.4 Checkpointing

Implementation:

```text
prism/training/checkpointing.py
```

Checkpoint payload format:

```text
format: stage1_torch_checkpoint
model_state_dict
optimizer_state_dict
scheduler_state_dict
client_state
config
```

Sidecar files:

```text
config.json
norm_stats.json
checkpoint.json
```

The refactor preserves these keys and file formats.

## 8. Evaluation Pipeline

Main entry:

```bash
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

Dispatch:

```text
prism/eval/runner.py
```

Shared helpers:

```text
prism/eval/profiles.py   profile_env parsing, dry-run behavior
prism/eval/metadata.py   result metadata and sanitized environment capture
```

LIBERO:

```text
prism/eval/libero.py
```

CALVIN:

```text
prism/eval/calvin_config.py
prism/eval/calvin_action_protocol.py
prism/eval/calvin_observation.py
prism/eval/calvin_request_builder.py
prism/eval/calvin_history.py
prism/eval/calvin_eval_summary.py
prism/eval/calvin_runner.py
```

The eval smoke configs use:

```yaml
dry_run: true
```

This validates runtime profiles without requiring simulator packages, datasets,
or a live websocket policy server.

## 9. Serving And Inference

Modules:

```text
prism/serve/engine.py
prism/serve/server.py
```

The server exposes a websocket policy runtime. Evaluation clients build
benchmark-specific JSON requests and send them to the server.

Request-level concepts:

- benchmark name
- prompt
- images by view
- robot state
- action dimension
- optional short-memory images by offset
- optional executed action segment
- reset-memory flag

## 10. Configuration

Config files:

```text
configs/model/prism_base.yaml
configs/model/prism_smoke.yaml
configs/data/libero.yaml
configs/data/calvin.yaml
configs/experiment/libero_warmup_w4.yaml
configs/experiment/libero_stage1.yaml
configs/experiment/libero_train_smoke.yaml
configs/experiment/libero_smoke.yaml
configs/experiment/calvin_warmup_w4.yaml
configs/experiment/calvin_stage1.yaml
configs/experiment/calvin_smoke.yaml
```

Python config modules:

```text
prism/config.py             stable public facade
prism/config_bridge.py      model/bridge dataclasses
prism/config_experiment.py  bridge config resolution
prism/config_loader.py      load_config()
prism/config_runtime.py     runtime constants
prism/config_training.py    training config defaults and validation
```

Important rule:

```text
Code defines logic. YAML defines experiment values.
```

No personal absolute project paths should appear in configs or source.

## 11. Dependency And Environment Notes

The Python package is `prism`; project package name is `prismvla`.

Pinned core dependencies include:

- Python >= 3.10
- torch 2.5.1
- torchvision 0.20.1
- transformers 4.39.0
- accelerate 1.13.0
- timm 1.0.27
- diffusers 0.38.0
- PyYAML 6.0.3
- websockets 16.0
- pytest 9.0.3 for dev
- ruff 0.15.16 for dev

Work is verified in the remote Evo environment:

```text
PATH=../miniforge3/envs/Evo1/bin:$PATH
```

## 12. Verification Status

Remote gate command:

```bash
scripts/check.sh
```

Current passing result:

```text
ruff: passed
pytest: 116 passed, 1 warning, 3 subtests passed
config validation: ok
```

Smoke commands verified:

```bash
python scripts/build_cache.py --config configs/experiment/libero_train_smoke.yaml
python scripts/train.py --config configs/experiment/libero_train_smoke.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

Smoke behavior:

- builds `local_data/smoke/stage1_cache/manifest.json`
- builds `local_data/smoke/progress_planner_smoke.pt`
- runs one real Stage1 optimizer step
- writes `local_data/smoke/runs/stage1_train_smoke/step_final/model.pt`
- runs LIBERO eval dry-run
- runs CALVIN eval dry-run

Known limitation:

Full simulator evaluation requires external LIBERO/CALVIN runtime packages,
datasets, and a live policy server. The current smoke path intentionally
validates eval profiles in dry-run mode.

## 13. Refactor Work Completed

The repository was refactored for maintainability while preserving behavior.

Major changes:

- `prism/training/trainer.py` reduced to a compatibility facade.
- Stage1, Stage2 data, Stage2 loop, checkpointing, optimizer, scheduler, and
  logging moved into separate training modules.
- `prism/data/cache.py` reduced to a compatibility facade.
- Token-cache, replay, frame-reader, and smoke-cache responsibilities split.
- `prism/models/action_head.py` reduced to a compatibility facade.
- Bridge attention, bridge adapter, and flow-matching head split.
- CALVIN eval split into protocol/config/history/request/summary/runner modules.
- Config split into bridge/runtime/training/experiment/loader modules.

Preserved contracts:

- CLI arguments
- YAML config names
- checkpoint format and keys
- output file formats
- logging fields
- eval result JSON format
- tensor shapes and masks
- loss definitions
- training objectives
- random seed behavior
- dataset filtering and sampling behavior

## 14. StarVLA Relationship

StarVLA is used as a structural and workflow reference, especially for the
LIBERO training recipe style:

- config-driven training
- shell-safe script wrappers
- explicit paths in YAML/shell
- clear separation between recipe and implementation

PrismVLA does not copy StarVLA architecture directly. The influence is mainly
engineering structure and training/eval workflow discipline.

## 15. What Makes The Project Technically Interesting

Strong interview points:

1. Bounded long-horizon memory:
   - long memory is task-progress state, not an ever-growing visual bank

2. Two-branch bridge attention:
   - visual evidence and action-condition sources are separated cleanly

3. Flow-matching action prediction:
   - predicts action velocity instead of direct regression

4. Cache-based Stage1:
   - avoids repeated VLM compute and makes action-head training reproducible

5. Maintainability refactor:
   - preserves scientific behavior while splitting large modules by ownership

6. Benchmark pragmatism:
   - focuses on LIBERO and CALVIN, with RMbench explicitly removed from scope

## 16. Likely Interview Questions

### What is PrismVLA?

PrismVLA is a robot-learning VLA codebase built around InternVL3 embeddings,
short visual memory, a progress-state planner, and a bridge-attention
flow-matching action head. It targets LIBERO and CALVIN.

### Why not keep long memory as a visual-token FIFO?

Because an unbounded visual-token bank is expensive and weakly structured. This
project uses short memory for recent visual continuity and a compact recurrent
progress state for long-horizon task structure.

### What does the progress-state planner store?

It stores task progress as two learned state tokens: one for completed events
and one for the current stage. The planner updates them from current VLM
summary, robot state, and previously executed actions.

### How does bridge attention work?

Action tokens query two context branches. The visual branch contains current VLM
hidden states and short memory. The action-condition branch contains plan tokens
and the robot state token. The output goes through the flow-matching action
decoder.

### Why use flow matching for actions?

Flow matching gives a continuous denoising-style action generation process.
During training, the model predicts velocity from noisy/interpolated action
states toward the ground-truth action sequence. During inference, it integrates
the predicted velocity for a fixed number of midpoint steps.

### What is Stage1 training?

Stage1 trains the action-head path from cached VLM features. The VLM is not
loaded or finetuned, and the progress planner is frozen from a checkpoint.
This makes training cheaper and more reproducible.

### What is Stage2 training?

Stage2 is the full end-to-end path from raw episodes. It loads and finetunes the
VLM, action head, and progress planner together.

### How do you know the refactor preserved behavior?

The refactor kept the old public import paths as facades and moved code by
responsibility. The remote gate passes ruff, all tests, config validation, and
the smoke cache/train/eval commands. Checkpoint keys and output formats were
not changed.

### What are the main risks?

Full simulator evaluation depends on external LIBERO/CALVIN packages, datasets,
and a live server. Stage2 is higher risk because it couples VLM, memory,
planner, and action head. The active redesign of short memory, planner, and
bridge attention must be documented and tested before behavior changes land.

## 17. Current Limitations And Future Work

Current limitations:

- Full LIBERO/CALVIN simulator eval is not part of the lightweight smoke path.
- Stage2 needs more benchmark-specific full-run validation.
- The new short-memory/planner/bridge-attention redesign is documented as
  pending in `docs/design/prism_architecture_redesign.md`.
- Some modules are under 1,000 lines but still dense, especially:
  - `prism/eval/libero.py`
  - `prism/data/libero.py`
  - `prism/models/planner.py`
  - `prism/models/memory.py`

Future work:

- finalize the new short memory contract
- finalize the new progress-state planner contract
- finalize the bridge-attention redesign contract
- add tests for every new tensor/mask/reset contract
- run full LIBERO and CALVIN simulator evaluations once external deps and data
  are installed
- expand Stage2 smoke and full training validation

## 18. Fast Demo Plan

If asked to demo quickly:

1. Show the package layout:

```bash
find prism -maxdepth 2 -type f | sort
```

2. Run the gate:

```bash
scripts/check.sh
```

3. Run smoke:

```bash
python scripts/build_cache.py --config configs/experiment/libero_train_smoke.yaml
python scripts/train.py --config configs/experiment/libero_train_smoke.yaml
python scripts/eval.py --config configs/experiment/libero_smoke.yaml
python scripts/eval.py --config configs/experiment/calvin_smoke.yaml
```

4. Show the output checkpoint:

```text
local_data/smoke/runs/stage1_train_smoke/step_final/model.pt
```

5. Explain that eval smoke is dry-run by design until simulator/runtime
   dependencies and the policy server are available.

## 19. One-Minute Interview Summary

```text
PrismVLA is a reconstructed VLA robot-learning codebase for LIBERO and CALVIN.
The model uses InternVL3 to encode image-language observations, short visual
memory for recent visual evidence, a compact progress-state planner for
long-horizon task state, and a direct bridge-attention flow-matching action
head to generate action horizons.

My main engineering contribution was turning a legacy research code path into a
maintainable package without changing scientific behavior. I split oversized
training, data, model, config, and evaluation files by responsibility; kept old
import paths as compatibility facades; preserved configs, checkpoint keys, and
result formats; and verified the project with ruff, 116 tests, config loading,
and smoke cache/train/eval commands in the remote Evo environment.
```
