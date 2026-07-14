# Training execution plan

This plan separates the runnable direct-action baseline from the experimental
task-state planner. It is based on the verified remote host and data available
on 2026-07-15; it is not a claim that the current hyperparameters are already
benchmark-optimal.

## Verified starting point

- Hardware: one NVIDIA RTX 4090 with 24,564 MiB VRAM.
- Model environment: Python 3.10.20, PyTorch 2.5.1, Transformers 5.13.1,
  Accelerate 1.13.0, and the pinned Qwen3.5 fast-path dependencies.
- LIBERO materialization: 2,000 episodes and 338,575 frames across
  libero_spatial, libero_object, libero_goal, and libero_10.
- CALVIN materialization: 17,870 episodes and 1,071,743 frames; training uses
  splits A/B/C and keeps D for evaluation.
- Both one-step CUDA smokes completed with finite loss and hash-verified
  checkpoints. CALVIN took 60.7 seconds and peaked at 3.34 GiB allocated /
  3.41 GiB reserved. LIBERO took 51.0 seconds and peaked at 3.33 GiB allocated /
  3.40 GiB reserved.
- Each one-step checkpoint is approximately 2.5 GiB. The smoke timings include
  model construction, base-model loading, one update, and a full checkpoint
  write, so they must not be used as steady-state steps/second estimates.

The accepted training path currently optimizes only the direct masked-L1 action
objective. Qwen language and vision weights are frozen; action queries, History
Q-Former, and the action head (including its Bridge stack) are trainable. The
task-state planner is frozen, its plan tokens are not consumed by the action
Bridge, and it has no training target. Unfreezing it today would spend compute
without providing an action-learning path.

## Execution order and gates

### Gate 0: repository and artifact contract

Before every new run:

1. Require a clean Git commit and record its ID.
2. Run the default unit/DDP suite, the real Qwen checkpoint integration, and
   both opt-in simulator runtime integrations documented in
   [Benchmark runtime](benchmarks/runtime.md).
3. Verify dataset statistics and materialized data roots through config loading.
4. Run one-step LIBERO and CALVIN CUDA smokes in new output directories.
5. Read the produced checkpoint metadata back and confirm manifest, config,
   architecture, DataSpec, statistics, world size, and RNG files.

Do not launch a long run if any gate fails or if its output directory already
exists. Checkpoints are immutable by design.

### Gate 1: steady-state throughput pilot

Run CALVIN first and LIBERO second for 100 optimizer steps, with the production
batch size and gradient accumulation (1 x 8 x 1 GPU = 8 samples per optimizer
step), but a unique pilot output directory and only a final checkpoint. Record:

- wall time after model initialization and before final checkpoint writing;
- optimizer steps/hour and data-loader wait time;
- maximum allocated/reserved VRAM;
- total, motion, and gripper losses plus transition recall;
- checkpoint write duration and resulting disk use.

Use the observed steady-state rate to calculate, rather than guess, the long-run
ETA:

~~~text
compute_hours = remaining_optimizer_steps / measured_optimizer_steps_per_hour
I/O_hours = remaining_checkpoint_count * measured_checkpoint_write_hours
~~~

The pilot must complete without non-finite loss, growing memory, data starvation,
or a resume mismatch. Resume the pilot checkpoint once and verify that the next
sample cursor and loss trajectory match an uninterrupted control run.

### Gate 2: direct-action baselines

Use the checked-in baseline configs without source edits:

~~~bash
../envs/prsim/bin/python scripts/train.py --config configs/train/calvin_baseline.yaml
../envs/prsim/bin/python scripts/train.py --config configs/train/libero_baseline.yaml
~~~

Run them sequentially on the single GPU. Do not mix LIBERO and CALVIN in one
checkpoint: they have different state contracts, normalization groups, data
splits, and simulator selection metrics.

With the current virtual epoch and accumulation settings:

| Run | Optimizer steps | Effective samples/step | Approx. virtual epochs | Checkpoints | Approx. checkpoint storage |
| --- | ---: | ---: | ---: | ---: | ---: |
| CALVIN A/B/C | 30,000 | 8 | 7.3 | 6 | 15 GiB |
| LIBERO four-suite mixture | 100,000 | 8 | 24.4 | 20 | 50 GiB |

The storage estimate uses the measured 2.5 GiB checkpoint and the current
5,000-step interval. Confirm free disk space before launch. Keep the existing
full-run config when resuming because exact resume rejects changed config hashes:

~~~bash
../envs/prsim/bin/python scripts/train.py --config configs/train/calvin_baseline.yaml --resume ../outputs/prismvla/calvin_abc_to_d_query_memory_baseline/checkpoints/step-00005000
~~~

### Gate 3: checkpoint selection

Training L1 is a health metric, not the model-selection target.

- At each CALVIN checkpoint, first run a fixed 100-sequence diagnostic slice.
  Run the full 1,000-sequence protocol only for the best two checkpoints and the
  final checkpoint. Select on average successful chain length and per-length
  success rates.
- At each LIBERO checkpoint, run a fixed seeded diagnostic subset from every
  suite. Run the full benchmark for the best two checkpoints and the final
  checkpoint. Select on macro-averaged suite success, while retaining per-suite
  results so the larger suites cannot hide a regression.
- Keep evaluation environment, sequence/episode IDs, server commit, checkpoint
  hash, and profile in the atomic result summary.

Use seed 7 for the first complete recipe. After selecting hyperparameters,
confirm the chosen recipe with at least two additional seeds rather than
running every exploratory setting three times.

## Planner training is a separate experiment

Do not modify the direct-action baseline in place. A planner experiment may
start only after all of the following are implemented and tested:

1. Plan tokens are consumed by the action Bridge/head under a causal mask.
2. A declared objective exists. A reasonable first proposal is the normal
   next-chunk action loss through the connected plan path plus a masked
   auxiliary decoder over the configured 64-action planning horizon; the exact
   target and weight must live in config and be ablated.
3. Training examples contain consecutive planning cycles with executed-action
   masks, and truncated-BPTT/cache-detach semantics are explicit.
4. Reset, padding, failed-action, and subtask-boundary behavior has unit and
   runtime tests.
5. Metrics can show that the planner affects actions; debug plan-token norms
   alone are not evidence.

Start by freezing the accepted direct-action path and training only the new
planner-to-action adapters. If that beats the direct baseline on held-out
success, run a short joint fine-tune with separate learning-rate groups. Keep
the original baseline checkpoint and results unchanged for the required
no-planner ablation.

## Operational checklist

- Keep Hugging Face, Triton, TorchInductor, pip, temporary, dataset, output, and
  log paths on the data disk.
- Run the policy server on loopback and use an SSH tunnel. It has no built-in
  authentication.
- Monitor GPU utilization, VRAM, host RAM, disk free space, and checkpoint
  duration. Stop on non-finite loss or repeated data/driver errors; do not hide
  them with retry loops.
- Never overwrite or delete a checkpoint during a run. Validate a resume target
  before retiring any older artifact.
- Record the final commands and evaluation result paths alongside the selected
  checkpoint.
