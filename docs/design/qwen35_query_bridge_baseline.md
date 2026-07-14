# Qwen3.5 Query-Memory and Task-State Architecture

Status: implemented runtime graph; task-state/plan objectives remain deferred
Date: 2026-07-14

## Purpose and status

This document defines the current PrismVLA graph and its externally visible tensor,
mask, checkpoint, and runtime contracts. It describes a runnable starting point, not
an experimentally validated final architecture.

The action-head capacity values below are especially important to interpret
correctly:

- `action_hidden_size: 512`
- `num_attention_heads: 8`
- `ffn_ratio: 4`

They are explicit configuration values that make the current implementation
runnable and testable. They remain provisional until training and benchmark
evidence supports keeping or changing them. Tests that assert these values verify
configuration and shape consistency; they do not establish that the values are
optimal or final.

Any capacity experiment must change configuration rather than source logic and
must be recorded in the resolved run snapshot.

## Current baseline

| Area | Current choice | Contract status |
| --- | --- | --- |
| VLM | `Qwen/Qwen3.5-0.8B`, first 16 of 24 language blocks | implemented baseline |
| VLM hidden width | 1024 | checkpoint-derived |
| Current observation | two ordered camera views plus the complete instruction | benchmark interface |
| Image preprocessing | aspect-preserving smart resize; square target 256 | accepted pipeline |
| Qwen action queries | 32 learned queries after images and instruction | accepted pipeline |
| Layer features | query states `Q1` through `Q16` | Bridge interface |
| History | two observations at ages 6 and 3, with two ordered views each | runtime interface |
| History Q-Former | 2 layers, width 512, 4 heads, MLP ratio 4, 16 output tokens | accepted pipeline |
| Bridge | 16 blocks; one block per retained Qwen layer | alignment contract |
| Memory fusion | separate current/history cross-attention; scalar history gate initialized to 0.1 | implemented baseline |
| Action sequence | 8 parallel action-step queries producing `[B, 8, 7]` | policy/runtime interface |
| Action capacity | width 512, 8 heads, FFN ratio 4 | **provisional experiment default** |
| Objective | direct masked L1 | implemented baseline |
| Replanning | predict 8 actions and execute all 8 before replanning | runtime interface |
| Persistent task state | 8 tokens of width 512, updated once per planning cycle | implemented runtime state |
| Coarse plan | 16 tokens of width 512 representing the next 64-action trend collectively | implemented output |
| State/plan supervision | no objective, loss weighting, or Bridge consumption yet | deliberately deferred |

The rebuilt model does not preserve compatibility with the removed planner,
progress estimator, action autoencoder, flow-matching head, legacy memory
compression, staged-training, or legacy Bridge implementations. MTP and vocabulary
logits are not part of the VLA forward path.

## Model flow

### Current observation and layer features

The causal Qwen input order is:

```text
primary image -> wrist image -> complete instruction/chat tokens -> 32 action queries
```

For a padded sequence length `T`, the retained backbone exposes:

```text
H0: assembled multimodal embeddings, before transformer block 1
H1 ... H16: retained block outputs, with final RMSNorm applied to H16
Hi shape: [B, T, 1024]
Qi = gather(Hi, action_query_mask), i in 1..16
Qi shape: [B, 32, 1024]
```

The Bridge consumes `Q1` through `Q16` exactly once. `H0` is not a Bridge level.
Implementations may stream the query features instead of materializing a
`[B, 16, 32, 1024]` tensor.

The processor, rather than a hard-coded token count, is authoritative for image
grid metadata and masks. For a processed image of height `H` and width `W`:

```text
raw vision patches = (H / 16) * (W / 16)
merged Qwen tokens = raw vision patches / 4
```

A square 256 input therefore yields 64 merged tokens per camera and 128 for two
cameras. Non-square inputs preserve aspect ratio and can yield different counts.

### Sparse history

At the planning observation `O8`, history contains `O2` and `O5`, corresponding to
ages `[6, 3]`. The same schedule repeats for each eight-action chunk. Each history
observation contains ordered `primary` and `wrist` views encoded by the shared Qwen
vision encoder.

At the square baseline resolution:

```text
two times * two cameras -> 256 visual tokens of width 1024
input projection + relative-age embedding -> [B, 256, 512]
16 learned Q-Former queries -> memory M: [B, 16, 512]
```

The training batch keeps the end-to-end raw-image path so the shared vision encoder
and History Q-Former remain part of the trainable graph:

```text
history_images: [B, 2, 2, H, W, 3]
history_step_ages: [B, 2] = [6, 3]
history_valid_mask: [B, 2]
```

Serving uses a split model interface and does not carry those historical images in
the next inference request:

```text
encode_history_observation(two current camera images) -> visual tokens [128, 1024]
build_history_memory(O2 tokens, O5 tokens, ages=[6, 3]) -> memory [1, 16, 512]
predict_with_memory(current images, state, prompt, memory) -> actions [1, 8, 7]
```

The client pushes O2 and O5 while executing the current eight-action chunk. Source
images are transient: after immediate server-side visual encoding, only visual
tokens are retained. Once O5 is encoded, the Q-Former runs before the next planning
request and both visual-token slots are released. The next inference carries only
its current observation and a memory generation identifier.

Every episode or subtask explicitly resets its connection-local stream. Initial
generation 0 constructs zero `[1, 16, 512]` memory with an all-false mask directly,
so it does not run four zero images through the vision encoder. Missing, duplicate,
stale, or cross-stream slots are rejected; invalid memory tokens remain masked from
Bridge attention.

### Bridge and action head

Bridge block `i` receives `Qi` plus the shared history memory:

```text
X = X + CrossAttention_current(X, project_current(Qi))
X = X + gate_i * CrossAttention_memory(X, project_memory(M))
```

Current and history branches use independent projections and padding masks. Each
`gate_i` starts at 0.1. Direct current visual tokens and direct text tokens are not
sent to the Bridge.

Eight learned action-step queries, temporal embeddings, and normalized robot state
form the action stream. After 16 Bridge blocks, final normalization and a linear
projection produce unbounded normalized predictions:

```text
normalized_actions: [B, 8, 7]
```

The first six values are relative motion. The seventh is canonical gripper state,
where `0=close`, `1=open`, and prediction `> 0.5` means open. There is no output
`tanh`, sigmoid, BCE, diffusion, flow-matching, noisy-action, or velocity-target
path.

### Task-state update and plan-token readout

Every planning request also advances a connection-local state. The updater receives
only Qwen layer-12 action-query outputs and the actions that the environment actually
executed in the preceding cycle; robot state/proprioception never enters this branch.
Both the updater and planner reuse one `LayerNorm(Linear(1024, 512))` projection of
`Q12`:

```text
Q12: [B, 32, 1024] -> shared projection -> [B, 32, 512]
executed actions: [B, 8, 7] -> MLP 7->256->512 + 8 learned positions -> LayerNorm
update context: direct concat -> [B, 40, 512]
```

At reset, the previous state is one learned `[8, 512]` tensor. The initial action
positions are all masked; they are not filled by copied or dummy actions. A pre-LN
eight-head cross-attention updates the state from the 40 context tokens, followed by
one pre-LN noncausal state self-attention block with no FFN.

Temporal recurrence is one Mamba-1 layer with `d_model=512`, `d_state=16`,
`d_conv=4`, and `expand=2`. The reshape `[B, 8, 512] -> [B*8, 1, 512]` gives each
state slot independent causal convolution and selective-SSM caches while sharing
weights. The cache and task state are reset at every episode/CALVIN subtask and are
committed only after successful inference.

The planner concatenates `LayerNorm(current_state)` and the already projected Q12.
Sixteen learned plan queries cross-attend to this context, pass through a
`512->1024->512` GELU residual MLP, one noncausal plan self-attention mixer, another
matching residual MLP, and final LayerNorm. The result is `[B, 16, 512]`; these
tokens collectively describe a coarse 64-action trend and do not have per-timestep
roles. There are no modality/type embeddings, pooling branches, planner Mamba, or
explicit robot-state tokens in this path. The module adds 8,686,080 parameters.

## Training contract

The current action baseline freezes Qwen language and vision parameters. It trains learned
Qwen action queries, the History Q-Former, the Bridge/action stack, and the direct
action head. Optimization scopes, learning rates, weight decay, accumulation, and
checkpoint cadence are explicit in `configs/train/*.yaml`.

The task-state/plan module is explicitly present but frozen in current training
profiles. Plan/state/action objectives, target construction, loss weighting,
staged-versus-joint training, Bridge consumption of plan tokens, and checkpoint
migration are deferred. They must be designed together rather than inferred from
this runtime implementation.

All seven action dimensions use global element-count-weighted masked L1:

```text
element_mask = action_valid_mask[:, :, None] & action_dim_mask[:, None, :]
loss = sum(abs(prediction - target) * element_mask) / sum(element_mask)
```

Sufficient statistics are accumulated across micro-batches and reduced across
ranks before division. Empty local populations therefore do not receive equal
weight.

## Required invariants

Remote tests must protect at least these contracts:

1. Sixteen retained blocks expose `H1` through `H16`; the Bridge consumes each once.
2. Each query level is `[B, 32, 1024]` and queries follow both images and instruction.
3. Image token counts and masks come from processor grid metadata.
4. History ages, view ordering, validity masks, and Q-Former output `[B, 16, 512]` agree across training and serving.
5. Serving caches no history images: O2/O5 become visual tokens immediately, become
   fixed memory before the next infer, and are cleared on successful inference,
   reset, failed memory construction, or disconnect.
6. Missing history cannot create invalid attention values or duplicate current frames.
7. Current and memory attention use separate projections and masks; gates initialize to 0.1.
8. The policy and both benchmark clients agree on `[B, 8, 7]`, normalization, gripper decoding, and eight-step replanning.
9. Architecture configuration, checkpoint snapshots, and reconstructed serving models are exact matches.
10. Masked loss and metrics reduce sufficient statistics across accumulation steps and ranks.
11. Removed language-generation, MTP, legacy action-head, and flow/diffusion paths are absent.
12. The updater uses exactly Q12 plus masked executed actions, one shared Q12
    projection, eight persistent state tokens, and no proprioception input.
13. Mamba advances along planning cycles independently for each state slot and its
    cache resets with the stream; failed inference cannot commit advanced state.
14. The planner emits exactly 16 noncausal width-512 tokens without timestep roles,
    type embeddings, pooling, or a planner Mamba.

## Experiment questions and change control

The first remote experiments should measure, rather than assume:

- whether action width 512, 8 heads, and FFN ratio 4 are appropriate;
- whether 32 current queries are sufficient without direct visual conditioning;
- whether 256 input resolution preserves enough task detail;
- how many Qwen layers need training and whether all 16 query levels contribute;
- whether 16 history tokens and the learned memory gates improve closed-loop results;
- how task-state and coarse-plan targets should be built and weighted before the
  new 8.7M-parameter branch is made trainable or consumed by the action Bridge.

A proposed baseline change must update the model YAML, resolved configuration and
checkpoint schema as needed, this document, and the relevant shape/protocol tests in
the same change. Experimental results should determine whether a provisional value
becomes a retained baseline; wording alone must not promote it to a final decision.
