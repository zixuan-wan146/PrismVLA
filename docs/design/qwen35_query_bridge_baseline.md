# Qwen3.5 Query-Bridge Architecture Baseline

Status: implemented research baseline; action-head capacity remains provisional
Date: 2026-07-13

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
| Image preprocessing | aspect-preserving smart resize; square target 384; dimensions aligned to 32 | configurable baseline |
| Qwen action queries | 48 learned queries after images and instruction | implemented baseline |
| Layer features | query states `Q1` through `Q16` | Bridge interface |
| History | two observations at ages 6 and 3, with two ordered views each | runtime interface |
| History Q-Former | 2 layers, width 512, 4 heads, MLP ratio 4, 24 output tokens | implemented baseline |
| Bridge | 16 blocks; one block per retained Qwen layer | alignment contract |
| Memory fusion | separate current/history cross-attention; scalar history gate initialized to 0.1 | implemented baseline |
| Action sequence | 8 parallel action-step queries producing `[B, 8, 7]` | policy/runtime interface |
| Action capacity | width 512, 8 heads, FFN ratio 4 | **provisional experiment default** |
| Objective | direct masked L1 | implemented baseline |
| Replanning | predict 8 actions and execute all 8 before replanning | runtime interface |

The rebuilt model does not preserve compatibility with the removed planner,
progress estimator, action autoencoder, flow-matching head, legacy memory
compression, staged-training, or legacy Bridge implementations. MTP and vocabulary
logits are not part of the VLA forward path.

## Model flow

### Current observation and layer features

The causal Qwen input order is:

```text
primary image -> wrist image -> complete instruction/chat tokens -> 48 action queries
```

For a padded sequence length `T`, the retained backbone exposes:

```text
H0: assembled multimodal embeddings, before transformer block 1
H1 ... H16: retained block outputs, with final RMSNorm applied to H16
Hi shape: [B, T, 1024]
Qi = gather(Hi, action_query_mask), i in 1..16
Qi shape: [B, 48, 1024]
```

The Bridge consumes `Q1` through `Q16` exactly once. `H0` is not a Bridge level.
Implementations may stream the query features instead of materializing a
`[B, 16, 48, 1024]` tensor.

The processor, rather than a hard-coded token count, is authoritative for image
grid metadata and masks. For a processed image of height `H` and width `W`:

```text
raw vision patches = (H / 16) * (W / 16)
merged Qwen tokens = raw vision patches / 4
```

A square 384 input therefore yields 144 merged tokens per camera and 288 for two
cameras. Non-square inputs preserve aspect ratio and can yield different counts.

### Sparse history

At the planning observation `O8`, history contains `O2` and `O5`, corresponding to
ages `[6, 3]`. The same schedule repeats for each eight-action chunk. Each history
observation contains ordered `primary` and `wrist` views encoded by the shared Qwen
vision encoder.

At the square baseline resolution:

```text
two times * two cameras -> 576 visual tokens of width 1024
input projection + relative-age embedding -> [B, 576, 512]
24 learned Q-Former queries -> memory M: [B, 24, 512]
```

The request boundary carries history explicitly:

```text
history_images: [B, 2, 2, H, W, 3]
history_step_ages: [B, 2] = [6, 3]
history_valid_mask: [B, 2]
```

The client captures the needed intermediate frames, clears the sparse buffer on
episode reset, and never substitutes the current observation for missing initial
history. Invalid slots are masked from memory attention.

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

## Training contract

The current baseline freezes Qwen language and vision parameters. It trains learned
Qwen action queries, the History Q-Former, the Bridge/action stack, and the direct
action head. Optimization scopes, learning rates, weight decay, accumulation, and
checkpoint cadence are explicit in `configs/train/*.yaml`.

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
2. Each query level is `[B, 48, 1024]` and queries follow both images and instruction.
3. Image token counts and masks come from processor grid metadata.
4. History ages, view ordering, validity masks, and Q-Former output `[B, 24, 512]` agree across training and serving.
5. Missing history cannot create invalid attention values or duplicate current frames.
6. Current and memory attention use separate projections and masks; gates initialize to 0.1.
7. The policy and both benchmark clients agree on `[B, 8, 7]`, normalization, gripper decoding, and eight-step replanning.
8. Architecture configuration, checkpoint snapshots, and reconstructed serving models are exact matches.
9. Masked loss and metrics reduce sufficient statistics across accumulation steps and ranks.
10. Removed language-generation, MTP, legacy action-head, and flow/diffusion paths are absent.

## Experiment questions and change control

The first remote experiments should measure, rather than assume:

- whether action width 512, 8 heads, and FFN ratio 4 are appropriate;
- whether 48 current queries are sufficient without direct visual conditioning;
- whether 384 input resolution justifies its cost relative to 320 or 256;
- how many Qwen layers need training and whether all 16 query levels contribute;
- whether 24 history tokens and the learned memory gates improve closed-loop results.

A proposed baseline change must update the model YAML, resolved configuration and
checkpoint schema as needed, this document, and the relevant shape/protocol tests in
the same change. Experimental results should determine whether a provisional value
becomes a retained baseline; wording alone must not promote it to a final decision.
