# Training and serving

All training, testing, checkpointing, and serving commands run from the remote
repository on the data disk. Dataset roots, model caches, output directories,
and temporary files in the checked-in configurations are relative to the
repository root.

## Current runnable baseline

The current implementation uses:

- provisional action-head width 512, eight attention heads, and FFN ratio four;
- frozen Qwen language-model and vision-encoder parameters;
- trainable learned action queries, History Q-Former, Bridge/action stack, and
  direct action head;
- AdamW parameter groups with explicit per-scope learning rate and weight
  decay;
- global element-count-weighted masked L1 across accumulation steps and ranks.

The action-head values `512 / 8 / 4` make this baseline runnable and shape-complete,
but they are not an experimentally accepted final design. Capacity experiments
must change the architecture configuration and preserve the resolved values in
run/checkpoint metadata.

The complete values live in `configs/model/qwen35_query_memory.yaml` and
`configs/train/*.yaml`. Changing the tuning scope or optimizer values is an
experiment/configuration change, not a source edit.

## Environment

Keep caches and temporary files on the data disk:

```bash
export HF_HOME=../hf-home
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR=../tmp
```

If the Qwen checkpoint is already cached and the overseas endpoint is
unreachable, also use:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The accepted data roots are:

```text
../benchmarks/libero/lerobot-v2.1-rotate180/
../benchmarks/calvin/lerobot/task_ABC_D_complete/
```

CALVIN training uses splits A/B/C; split D remains evaluation-only.

## Train and resume

One-step remote smoke configurations are provided for both benchmarks:

```bash
../envs/prsim/bin/python scripts/train.py \
  --config configs/train/calvin_smoke.yaml

../envs/prsim/bin/python scripts/train.py \
  --config configs/train/libero_smoke.yaml
```

Checkpoints are immutable. A smoke command therefore requires that its
configured output/checkpoint directory does not already exist. Production
baselines use:

```bash
../envs/prsim/bin/python scripts/train.py \
  --config configs/train/calvin_baseline.yaml

../envs/prsim/bin/python scripts/train.py \
  --config configs/train/libero_baseline.yaml
```

Resume validates the complete resolved configuration, architecture, DataSpec,
statistics, world size, manifest, and cursor before loading Accelerate state:

```bash
../envs/prsim/bin/python scripts/train.py \
  --config configs/train/calvin_baseline.yaml \
  --resume ../outputs/prismvla/calvin_abc_to_d_query_memory_baseline/checkpoints/step-00005000
```

The opt-in pytest smoke lets validation jobs choose a new output directory
without modifying a checked-in config:

```bash
PRISM_RUN_TRAIN_SMOKE=1 \
PRISM_TRAIN_SMOKE_OUTPUT_DIR=../outputs/prismvla/smoke/calvin-validation \
../envs/prsim/bin/pytest -q tests/training/test_train_smoke.py
```

## Validate distributed loss math

The real two-process CPU test exercises DDP gradient averaging and two-step
gradient accumulation with unequal valid-element counts:

```bash
../envs/prsim/bin/pytest -q \
  tests/training/test_distributed_loss_integration.py
```

It verifies that all ranks optimize the global error sum divided by the global
valid-element count and that empty local transition populations do not dilute
recall.

## Serve a checkpoint

The server reconstructs architecture and DataSpec from verified checkpoint
metadata, loads the model artifact strictly from that same checkpoint, disables
gradients, and then exposes the MessagePack/WebSocket protocol:

```bash
../envs/prsim/bin/python scripts/serve_policy.py \
  --checkpoint ../outputs/prismvla/calvin_abc_to_d_query_memory_baseline/checkpoints/step-00030000 \
  --device cuda \
  --host 127.0.0.1 \
  --port 9000 \
  --local-files-only
```

LIBERO and CALVIN benchmark clients connect with `PRISM_SERVER_URI`. Their
adapters clip the six de-normalized relative motion dimensions to the verified
environment input range `[-1, 1]` before execution; q01/q99 remains a
statistical normalization range rather than a physical limit.
