# Qwen3.5 Query-Bridge Architecture Baseline

Status: accepted research baseline
Date: 2026-07-13

## 1. Purpose

This document records the architecture decisions that have been accepted for the PrismVLA rebuild. It is the baseline for implementation, configuration, training experiments, and shape tests.

The design intentionally does not preserve compatibility with the removed PrismVLA action head, planner, progress estimator, legacy memory compression, staged-training, or legacy bridge implementations. The history memory defined here is a new sparse-frame Q-Former branch rather than a restoration of the deleted mechanism. VLA-Adapter and StarVLA are research references, not code dependencies for the new model.

## 2. Accepted decisions

The first implementation shall use the following baseline:

| Component | Accepted choice |
| --- | --- |
| VLM checkpoint | `Qwen/Qwen3.5-0.8B` |
| Language depth | First 16 of the 24 Qwen transformer blocks |
| Hidden size | 1024 |
| Cameras | Two ordered benchmark views |
| Image preprocessing | Aspect-preserving smart resize with a canonical 384 x 384 target for square inputs |
| Size alignment | Height and width aligned to multiples of 32 |
| Action queries | 48 learnable query tokens |
| Query placement | After both images and the complete instruction |
| Layer features | Query-token states from every retained transformer layer, `H1` through `H16` |
| Action horizon | 8 environment actions |
| Replan stride | 8 environment actions; no overlapping action chunks |
| History sampling | Two observations at relative environment-step offsets `[-6, -3]` |
| History cameras | Two ordered views at each sampled observation |
| History Q-Former | 2 layers, hidden size 512, 4 heads, MLP ratio 4 |
| Memory tokens | 24 learnable queries producing 24 memory tokens |
| Bridge depth | 16 blocks, aligned one-to-one with the 16 retained VLM layers |
| Bridge conditioning | 48 current action-query features plus 24 history memory tokens |
| Bridge source fusion | Separate current and memory cross-attention branches |
| Memory residual gate | One learned scalar per Bridge block, initialized to 0.1 |
| Direct vision tokens in Bridge | Disabled in the baseline |
| Direct text tokens in Bridge | Disabled in the baseline |
| CLS or mean pooling | Not used |
| MTP | Not constructed or loaded |
| Language generation path | Not used |

This is a single architecture baseline, not a staged-training recipe. Changeable values must remain external configuration rather than source-code constants.

## 3. Backbone truncation

Qwen3.5-0.8B contains 24 language blocks arranged as six repetitions of:

```text
3 x Gated DeltaNet linear-attention block
1 x full-attention block
```

Retaining the first 16 blocks preserves four complete repetitions:

```text
12 x linear-attention block
4 x full-attention block
```

The retained backbone must include the final RMSNorm after block 16. Blocks 17 through 24 are physically absent from the constructed model rather than merely frozen.

The approximate parameter contract, based on the official checkpoint tensor shapes, is:

| Retained component | Parameters |
| --- | ---: |
| Token embedding | 254,279,680 |
| First 16 language blocks | 332,074,880 |
| Final RMSNorm | 1,024 |
| Truncated language backbone | 586,355,584 |
| Vision encoder and merger | 100,592,896 |
| Total truncated multimodal backbone | 686,948,480 |

The checkpoint MTP branch contains an additional 20,452,864 parameters and is excluded. The tied language output head does not need a separate stored weight, and the new VLA forward path must not produce vocabulary logits.

## 4. Hidden-state terminology

A 16-block transformer produces 17 hidden-state levels when hidden-state capture includes the input to the first block:

```text
H0  = assembled multimodal input embeddings
H1  = output of retained transformer block 1
H2  = output of retained transformer block 2
...
H16 = output of retained transformer block 16 followed by final RMSNorm
```

The model is called a 16-layer backbone, not a 17-layer backbone.

`H0` contains:

- text-token embeddings;
- image features produced by the vision encoder and merger and inserted at image-token positions;
- special-token embeddings.

`H0` has not passed through a Qwen language block and is not a Bridge conditioning level in the accepted baseline. The Bridge consumes `H1` through `H16`.

Use unambiguous configuration and interface names:

```text
num_backbone_layers = 16
num_hidden_state_levels = 17
num_bridge_levels = 16
```

## 5. Image preprocessing and visual-token count

The camera capture resolution and model input resolution are different concepts. The benchmark or camera produces an original image; the Qwen image processor then performs aspect-preserving smart resize and aligns each dimension to a multiple of 32.

For a processed static image of height `H` and width `W`:

```text
raw vision patches = (H / 16) x (W / 16)
LLM visual tokens = raw vision patches / 4
```

The division by four comes from the Qwen spatial merger, which combines each 2 x 2 group of vision patches and projects it to the 1024-dimensional language hidden space.

For the canonical square baseline:

```text
processed image size       = 384 x 384
raw patches per camera     = 24 x 24 = 576
merged tokens per camera   = 12 x 12 = 144
two-camera visual tokens   = 288
```

The canonical 384 x 384 target is a preprocessing budget, not a camera setting. Non-square inputs must preserve aspect ratio, so their exact token count can differ. The processor must return the final grid metadata and valid-token mask; downstream code must not infer token counts from a hard-coded constant.

Resolution is an experimental axis, but 384 is the accepted first baseline. Lower-resolution 256 and 320 inputs may be evaluated later as efficiency ablations.

## 6. Action-query construction

The VLM input sequence must place all learnable action queries after the complete observation and instruction:

```text
camera 1 image
camera 2 image
instruction and required chat tokens
48 learnable action-query tokens
```

This ordering is required because the Qwen language backbone is causal. An action query can attend to preceding visual and language tokens; an image token placed before the instruction cannot attend to the later instruction.

Each retained layer preserves sequence length and hidden width. Let `T` be the padded multimodal sequence length:

```text
Hi shape = [B, T, 1024], for i in 1..16
```

The action-query mask gathers exactly 48 positions from each layer:

```text
Qi = gather(Hi, action_query_mask)
Qi shape = [B, 48, 1024]
```

The layer-wise query feature pyramid is therefore:

```text
Q = [Q1, Q2, ..., Q16]
conceptual shape = [B, 16, 48, 1024]
```

Implementations may stream these features layer by layer instead of materializing the conceptual four-dimensional tensor, provided gradients and layer alignment remain correct.

The 48 queries are a learned information bottleneck. They have access to the complete preceding multimodal sequence but are not assumed to be a mathematically lossless representation of it. At the canonical resolution, 288 current visual tokens condition 48 action queries, giving a `6:1` visual-to-query token ratio. The queries are global multimodal readers and are not assigned rigidly to individual action timesteps.

## 7. Sparse history memory

### 7.1 Planning clock and sampled observations

The accepted runtime schedule predicts eight actions and executes the full chunk before replanning:

```text
action_horizon = 8
replan_stride = 8
```

Nine observations bound the eight executed actions:

```text
observations: O0  O1  O2  O3  O4  O5  O6  O7  O8
actions:          A0  A1  A2  A3  A4  A5  A6  A7
```

At `O0`, the policy predicts `A0` through `A7`. At the next planning observation `O8`, the history branch consumes `O2` and `O5`, while `O8` remains the current VLM observation. The history offsets are expressed in environment steps relative to the current planning observation:

```text
history_step_offsets = [-6, -3]
```

The rule repeats for every action chunk:

```text
current O8:  history = [O2,  O5]
current O16: history = [O10, O13]
current O24: history = [O18, O21]
```

History sampling therefore captures two approximately evenly spaced observations from the immediately preceding action chunk. It is a short-term visual dynamics summary, not a recurrent long-term episode memory.

### 7.2 History visual-token construction

Each sampled observation contains the same two ordered camera views as the current observation. The two images are encoded independently by the shared Qwen vision encoder and merger, then concatenated along the token axis. Raw images must not be stitched into one wider image.

At the canonical resolution:

```text
V_static: [B, 144, 1024]
V_wrist:  [B, 144, 1024]
V_time = concat(V_static, V_wrist): [B, 288, 1024]
```

The fixed `static, wrist` ordering is the camera identity contract. No additional camera embedding is used in the baseline.

The two historical observations are projected and given learned relative-age embeddings:

```text
K_old    = input_projection(V[t-6]) + relative_age_embedding[6]
K_recent = input_projection(V[t-3]) + relative_age_embedding[3]
K_history = concat(K_old, K_recent)
K_history shape = [B, 576, 512]
```

The relative-age embedding is a learned table indexed by environment-step age. The same age vector is added to every token from the corresponding observation. The processor-provided visual masks must be concatenated in the same order as the tokens.

### 7.3 History Q-Former

The History Q-Former is intentionally small:

| Field | Accepted value |
| --- | ---: |
| Input visual width | 1024 |
| Hidden and output width | 512 |
| Layers | 2 |
| Attention heads | 4 |
| Head dimension | 128 |
| MLP ratio | 4 |
| Learnable memory queries | 24 |
| Output memory tokens | 24 |

Each pre-normalized Q-Former block performs:

```text
memory-query self-attention
cross-attention from memory queries to K_history
feed-forward network
```

Starting from 24 learned queries, the two blocks produce:

```text
M shape = [B, 24, 512]
```

The history compression ratio at 384 x 384 is `576:24 = 24:1`. Four heads are sufficient for the two-observation compression problem; changing the head count does not materially change the projection parameter count at fixed hidden width.

### 7.4 Sparse frame buffer and request boundary

The benchmark client must capture observations at every environment step because `O2` and `O5` are not policy-request steps. A shared `SparseHistoryBuffer` records only the two configured offsets and both camera views, so it holds at most four historical images for one environment instance.

At a replan boundary, the canonical request must carry explicit history rather than relying on hidden model-server state:

```text
history_images: [B, 2 history times, 2 cameras, H, W, 3]
history_step_ages: [B, 2] = [6, 3]
history_valid_mask: [B, 2]
```

The buffer is cleared on episode reset and after its frames are transferred into the next request. At the initial `O0` request, both history slots are invalid and the Bridge skips memory attention. Initial observations must not be copied into missing history slots.

## 8. Layer-wise Bridge contract

Bridge depth must match retained VLM depth:

```text
Q1  -> Bridge block 1
Q2  -> Bridge block 2
...
Q16 -> Bridge block 16
```

Every retained contextual VLM layer must be consumed exactly once. The baseline must not:

- concatenate all 16 levels into one large key/value sequence;
- silently reuse the last available VLM layer for unmatched Bridge blocks;
- alternate Bridge blocks in a way that causes half of the VLM levels to be ignored;
- include `H0` as if it were a transformer-layer output.

The minimum Bridge input contract is:

```text
action_latent_i: [B, A, D_action]
query_features_i: [B, 48, 1024]
query_padding_mask: [B, 48]
memory_features: [B, 24, 512]
memory_padding_mask: [B, 24]
```

Each Bridge block uses separate current and history cross-attention residuals:

```text
X_i = X_i + CrossAttention_current_i(X_i, project_current_i(Q_i))
X_i = X_i + g_i * CrossAttention_memory_i(X_i, project_memory_i(M))
```

`Q_i` changes with the aligned VLM layer, while the 24 Q-Former memory tokens are shared across the 16 Bridge blocks. Current and memory branches have independent projections and masks. Each learned scalar `g_i` is initialized to `0.1`, making history an auxiliary signal at initialization instead of allowing it to dominate the current observation.

If the Bridge internal width differs from either conditioning width, each layer uses normalization and learned projections from 1024 and 512 to the configured action width. Projection may reduce channel width but must not pool either token sequence. The Bridge receives 48 current tokens and 24 memory tokens, preserving a `2:1` current-to-history token-count priority.

The action stream contains exactly eight learned action-step tokens, one for each
parallel action in the accepted horizon. The action hidden width,
attention-head count, and feed-forward ratio have not yet been accepted. They
must remain explicit architecture configuration fields and must be resolved
before constructing the policy.

## 9. Relationship to research references

The statements in this section were checked against the following local snapshots:

| Repository | Commit | Relevant implementation |
| --- | --- | --- |
| StarVLA | `a060fd97ddb7e4163d4e4fb17271e4bde74ea3c7` | `starVLA/model/framework/VLM4A/QwenAdapter.py`, `QwenPI_v3.py` |
| VLA-Adapter | `23fa0c9c159e2aa04341cdd3e924f44061311060` | `prismatic/extern/hf/modeling_prismatic.py`, `prismatic/models/action_heads.py` |

These repositories remain isolated under `third_party/`; their Git histories and source trees are references rather than vendored PrismVLA modules.

### 9.1 VLA-Adapter

VLA-Adapter establishes the main layer-wise principle:

- collect the hidden state from every language layer;
- extract spatial task features and action-query features at each level;
- condition the corresponding action block on the corresponding VLM level.

Its Qwen2.5 backbone has 24 transformer blocks and returns 25 hidden-state levels including `H0`. Its 24 action blocks consume `H1` through `H24`.

The new PrismVLA current-observation stream keeps the layer-wise correspondence but differs in token selection: it consumes action-query features instead of direct current visual tokens. Its separate history stream is defined by the History Q-Former contract above.

### 9.2 StarVLA

StarVLA provides two useful engineering references:

- `QwenAdapter` locates action-query positions and dynamic image-token spans instead of relying on fixed token offsets;
- `QwenPI_v3` uses per-layer normalization and projection to decouple VLM hidden width from action-model hidden width.

The new implementation should reuse these ideas at the interface level while avoiding two behaviors that do not match the accepted baseline:

- passing the entire padded VLM sequence to every action block;
- interleaving action-only blocks in a way that leaves some VLM levels unused.

## 10. Why visual tokens are not sent directly to Bridge

At 384 x 384, two cameras provide 288 merged visual tokens to the Qwen language backbone. The 48 action queries attend to these visual tokens and to the instruction inside each retained Qwen layer.

The Bridge receives the 48 current query states from each layer rather than receiving the 288 current visual states again. Historical visual tokens are likewise compressed from 576 tokens to 24 memory tokens before entering the Bridge. These choices:

- make the VLM responsible for current multimodal fusion;
- preserve layer-wise information through `Q1` to `Q16`;
- reduce Bridge cross-attention key/value length;
- avoid duplicating the complete vision sequence across 16 Bridge blocks;
- create explicit current and history bottlenecks that can be measured through ablation;
- give current information a `48:24 = 2:1` token-count priority over history.

Direct visual-token conditioning remains a possible ablation, not part of the accepted baseline.

## 11. Direct action and training contract

The action objective is no longer an open question. The accepted first policy
uses eight learned action-step queries with learned temporal position
embeddings and explicit normalized-state conditioning. The 48 Qwen action
queries remain layer-wise multimodal readers; they are not the eight action-step
queries and are not assigned to timesteps.

Every Bridge block performs non-causal action-step self-attention, current-query
cross-attention, gated history-memory cross-attention, and a feed-forward
update. Sixteen Bridge blocks consume Q1 through Q16 exactly once. A final
normalization and linear projection produce:

```text
normalized_actions: [B, 8, 7]
```

The seven action dimensions are delta position (3), delta rotation (3), and an
absolute canonical gripper command (1). The first six dimensions use
training-split q01/q99 normalization with a hard clip to [-1,1]. The gripper is
stored as 0=close and 1=open, uses identity normalization, and is decoded with
the strict rule prediction > 0.5 means open; exactly 0.5 means close.

All seven dimensions use one masked L1 objective:

```text
element_mask = action_valid_mask[:, :, None] & action_dim_mask[:, None, :]
loss = sum(abs(predicted_actions - target_actions) * element_mask)
       / sum(element_mask)
```

The output projection has no tanh or sigmoid. The baseline does not construct a
flow-matching, diffusion, noisy-action, timestep, velocity-target, BCE, or
classification path.

The accepted first baseline resolves the action hidden width to 512, uses eight
attention heads, and uses an FFN ratio of four. This keeps a 64-dimensional
attention head while aligning the action and history widths. The baseline
freezes the Qwen language model and vision encoder, and trains the learned
action queries, History Q-Former, Bridge/action stack, and direct action head.
Every scope, learning rate, weight decay, and checkpoint cadence remains
explicit in `configs/train/*.yaml`; none is inferred from
`model.parameters()`.

Training reduces sufficient statistics across every accumulation micro-batch
and distributed rank before dividing. In particular, masked L1 is the global
sum of valid element errors divided by the global valid element count, and
transition recall is global true positives divided by global positives. Empty
local populations do not receive equal weight.

There is no legacy stage-1/stage-2 training architecture. Freeze schedules, if
used, are configuration-driven optimization choices rather than separate model
definitions.

## 12. Required implementation tests

The implementation is not complete until remote tests cover all of the following contracts:

1. A 16-block backbone returns 17 hidden-state levels when `H0` is requested.
2. Bridge conditioning contains exactly 16 levels corresponding to `H1` through `H16`.
3. Each query level has shape `[B, 48, 1024]` before any Bridge projection.
4. Every Bridge block receives the matching VLM level exactly once.
5. Action queries occur after both images and the complete instruction.
6. Image, query, text, and padding masks identify the correct token positions.
7. Square 384 x 384 inputs produce 144 merged visual tokens per camera and 288 for two cameras.
8. Non-square smart-resized inputs preserve aspect ratio and derive token counts from grid metadata.
9. Two historical observations with two square 384 x 384 views produce 576 History Q-Former input tokens.
10. Relative-age embeddings for ages 6 and 3 are applied to the matching historical token spans.
11. The two-layer, four-head History Q-Former returns exactly `[B, 24, 512]` memory features.
12. The sparse frame buffer captures local offsets 2 and 5, emits relative ages 6 and 3 at the next replan, and clears on episode reset.
13. Missing initial history disables memory attention without duplicating the current observation or producing invalid attention values.
14. Each Bridge block receives separate current and memory masks, uses the matching current VLM level, and initializes its memory gate to 0.1.
15. The runtime predicts and executes eight actions before the next policy request.
16. MTP, blocks 17 through 24, and vocabulary-logit computation are absent from the VLA forward graph.
17. LIBERO and CALVIN protocol smoke tests preserve ordered current and history two-camera contracts.
18. Eight action-step queries remain distinct from the 48 Qwen query readers and produce exactly `[B, 8, 7]`.
19. Action self-attention, temporal position embeddings, and state conditioning all receive gradients.
20. Masked L1 counts only valid time/dimension elements and uses the same loss for motion and gripper.
21. Gripper decoding uses `prediction > 0.5`; no BCE, sigmoid, noisy-action, timestep, flow, or diffusion path is constructed.

All shape, mask, truncation, parameter-count, and benchmark-protocol tests must run in the remote project environment.

## 13. Open research questions

The first experiments should answer the following questions without changing the baseline definitions silently:

1. Are 48 layer-wise action queries sufficient without direct current visual-token conditioning?
2. Does 384 x 384 improve manipulation accuracy over 320 or 256 enough to justify its compute cost?
3. What is the smallest Bridge hidden width that preserves benchmark performance?
4. Do all 16 query levels contribute, or can a later experiment select a smaller principled subset?
5. How much of the truncated Qwen backbone must be updated for action queries to become effective task-specific readers?
6. Do 24 memory tokens preserve the useful change signal better than 16 without allowing history to dominate?
7. Does the learned memory gate remain small for static scenes and increase for visually meaningful transitions?

Any accepted change must update this document, the external configuration schema, and the corresponding shape tests together.
