# LIBERO-10 Train Recipe

本文记录当前 PrismVLA 在 `libero_10` 上的训练 recipe。当前文档只锁定 Stage 1：基于 VLM token cache 的 action-side flow matching 训练。Stage 2 的 raw image / VLM finetune 设置后续单独补充。

## 1. 训练目标

`libero_10` 当前训练预算按单个 suite 计算：

$$
N_{\mathrm{suite}} = 60000
$$

Stage 1 用于把 frozen VLM cache、short memory、plan condition、state condition 和 direct bridge action head 接起来。当前 episode-level fixed-replan-node 配置训练：

$$
N_{\mathrm{stage1}} = 5000
$$

当前 LIBERO-10 episode feature cache 有 500 条 episode，因此 5000 optimizer steps 约等于 500 episodes × 10 passes。是否切入 Stage 2 不能只看 train loss，还需要看 checkpoint 健康状态、smoke/rollout 和后续 eval。

重要修正：只要 Stage 1 forward 中仍使用 W4 ProgressPlanner 的递推状态：

$$
M_k = f(M_{k-1}, x_k)
$$

训练就不能把 replan frame 完全打散成独立样本。当前 Stage 1 使用 episode-level fixed-replan-node training：每个 batch item 是一条 episode，episode 内按 fixed replan nodes 顺序递推 frozen progress state `M`，并在所有 full-horizon nodes 上计算 flow matching loss。

## 2. Stage 1 Summary

```yaml
stage_1:
  name: frozen_vlm_cache_action_side_training
  max_steps: 5000

  input_mode: token_cache
  use_token_cache: true
  use_raw_images: false
  load_vlm: false

  suite: libero_10
  horizon: 32
  stride: 16
  replan_interval: 16
  action_dim: 7
  state_dim: 8
```

训练时预测未来 `32` 步动作，执行侧按 `16` 步重新规划：

$$
H = 32
$$

$$
R = 16
$$

对于 episode 内第 `k` 个 replan step，当前低层时间步为 `t_k`，监督动作 chunk 为：

$$
A_k = a_{t_k:t_k+H-1}
$$

尾部不足 `H = 32` 的样本直接跳过：

```yaml
drop_tail_shorter_than_horizon: true
```

## 3. Cache Training Boundary

Stage 1 使用 VLM token cache，因此 VLM 不在计算图中：

$$
F_k, S_k = \mathrm{Cache}(o_k)
$$

cached tokens 在训练时视为固定输入：

```yaml
dataset:
  suite: libero_10
  input_mode: token_cache
  cache_type: vlm_token_cache
  detach_cached_tokens: true
```

冻结项：

```yaml
freeze:
  vision_encoder: true
  vlm_backbone: true
  llm_backbone: true
  vlm_multimodal_projector: true
```

这意味着 Stage 1 不能训练 VLM、vision tower、LLM backbone 或 multimodal projector。Stage 1 只训练 non-VLM action-side modules。

当前工程实现中，Stage 1 直接设置：

```yaml
load_vlm: false
```

也就是训练进程不初始化 InternVL3。训练只读取 cache 中的 `fused_tokens` / `vlm_hidden_states`，因此不需要把 VLM 权重占在显存里。

## 3.1 Episode-Level Fixed-Replan-Node Training

Stage 1 token cache 训练按 episode 读取 fixed replan nodes，而不是 frame-level random batch，也不是旧的固定长度 trajectory window 采样。当前 cache 每个 dataset item 是一条 episode：

```yaml
memory_token_cache_sequence_training: true
batch_size: 1
```

`batch_size=1` 表示每个 optimizer step 处理一条 episode。DataLoader collate 后得到按 node 时间顺序排列的 `trajectory_steps`。训练 loop 先初始化该 episode 的 progress state：

$$
M_0
$$

然后对 episode 内 replan nodes 顺序递推：

$$
M_k
=
\mathrm{ProgressPlanner}_{\mathrm{W4}}
\left(
M_{k-1},\ \bar{F}_k,\ s_k,\ A^{\mathrm{exec}}_{k-1}
\right)
$$

每个 node 都会更新 frozen progress state `M`。只有 `action_valid_count >= horizon` 的 full-horizon nodes 进入 action-side flow matching loss：

$$
\mathcal{L}_{\mathrm{stage1}}
=
\frac{1}{|\mathcal{K}_{\mathrm{full}}|}
\sum_{k \in \mathcal{K}_{\mathrm{full}}}
\mathcal{L}_{\mathrm{FM}}(k)
$$

当前 `burnin_replan_steps`、`loss_replan_steps`、`trajectory_window_stride` 是旧 window 路线留下的 schema 字段；在 `libero_episode_feature_cache` 的 episode-level path 中不用于采样。尾部不足 `H=32` 的 nodes 仍会递推 `M`，但不计算 FM loss。

当前训练 profile 使用：

```yaml
batch_size: 1
max_steps: 5000
warmup_steps: 500
```

因此 5000 optimizer steps 对 500 条 episode 约为：

$$
5000 / 500 = 10
$$

也就是约 10 passes。当前从 500-step checkpoint 续训 4500 steps 时使用 `--max_steps 5001`，原因是 checkpoint 内 `next_step=501`，循环跑到 `<5001`，实际新增 `501..5000`。

## 4. Trainable Modules

Stage 1 可训练模块按当前 computation graph 写成：

```yaml
trainable:
  flow_matching_action_head: true
  action_expert: true

  flow_time_embedding: true
  timestep_mlp: true
  noisy_action_encoder: true
  action_pos_embedding: true
  temporal_pos_embedding: true

  bridge_attention: true
  bridge_adapter: true
  gates: true

  short_memory_encoder: true
  short_memory_projector: true

  plan_projector: true
  progress_condition_projector: true
  source_mlp: true

  gripper_head: false
```

Stage 1 不把 long memory tokens 作为单独 `long_memory_context` 输入 action-side graph。长期进展信息通过冻结的 W4 ProgressPlanner 产生 plan token，再进入 action-condition branch。

ProgressPlanner 在 Stage 1 中不作为主训练对象。当前使用已经预热好的 W4 planner checkpoint 在线产生 plan condition，但 planner body 冻结，不计算额外 planner loss：

```yaml
progress_planner:
  body_trainable: false
  source: frozen_w4_checkpoint
  checkpoint: $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
  online_forward: true
```

Stage 1 可训练的是 action-side 的 plan condition 接入层：

```yaml
action_side_plan_condition:
  plan_projector_trainable: true
  progress_condition_projector_trainable: true
```

ProgressPlanner 参与 forward 只是为了产生 plan token。由于 `finetune_progress_planner=false`，它的参数不参与优化，也不会引入 warm-up 阶段的 plan/memory 辅助损失。

## 5. Input Tokens

Stage 1 cache 至少包含：

| field | 说明 |
|---|---|
| `fused_tokens` | 当前 VLM fused tokens |
| `vlm_hidden_states` | 当前 VLM selected hidden states，layers `[3, 6, 9, 12]` |
| `memory_context` | short memory visual tokens |
| `short_memory_time_ids` | short memory 时间偏移标记 |
| `states` | 当前 robot state / proprio |
| `actions` | 未来 `H = 32` 步 normalized action target |
| `action_mask` | action target 有效 mask |
| `executed_actions` | 上一段实际执行动作，长度 `R = 16` |
| `executed_action_mask` | executed action 有效 mask |
| `episode_id` | episode 标识 |
| `task_name` | task 标识 |
| `suite` | suite 标识 |

`plan_tokens` 不作为 cache 字段保存。训练 forward 中使用：

$$
P_k
=
\mathrm{ProgressPlanner}_{\mathrm{W4}}
\left(
\bar{F}_k,\ s_k,\ A^{\mathrm{exec}}_{k-1}
\right)
$$

其中 W4 ProgressPlanner 冻结，只作为 plan token source。

short memory 使用两个时间偏移：

$$
t_k - R
$$

$$
t_k - \frac{R}{2}
$$

在当前设置中就是：

$$
t_k - 16
$$

$$
t_k - 8
$$

## 6. Direct Bridge Context

Direct bridge 使用两个 functional branches，而不是把四类 source 拆成四套独立 cross-attn。

Visual evidence branch：

$$
C^{\mathrm{vis}}_{l,k}
=
\left[
F_{l,k},\ S_k
\right]
$$

Action-condition branch：

$$
C^{\mathrm{act}}_{l,k}
=
\left[
\tilde{P}_{l,k},\ z^s_k
\right]
$$

其中：

- `F_lk` 是第 `l` 个 bridge block 使用的 current VLM hidden states；
- `S_k` 是 short memory tokens；
- `P_k` 是 plan token / plan slots；
- `tilde_P_lk` 是经过第 `l` 层 plan scale 后的 plan token；
- `z_s_k` 是由 state MLP 编码得到的 state token。

每个 bridge block：

$$
X'_l
=
X_l
+
\mathrm{SA}_l(\mathrm{LN}(X_l))
$$

$$
O^{\mathrm{vis}}_l
=
\mathrm{CA}^{\mathrm{vis}}_l
\left(
\mathrm{LN}(X'_l),
C^{\mathrm{vis}}_{l,k}
\right)
$$

$$
O^{\mathrm{act}}_l
=
\mathrm{CA}^{\mathrm{act}}_l
\left(
\mathrm{LN}(X'_l),
C^{\mathrm{act}}_{l,k}
\right)
$$

$$
\bar{X}_l
=
X'_l
+
\alpha^{\mathrm{vis}}_l O^{\mathrm{vis}}_l
+
O^{\mathrm{act}}_l
$$

$$
X_{l+1}
=
\bar{X}_l
+
\mathrm{FFN}_l(\mathrm{LN}(\bar{X}_l))
$$

Visual branch gate：

$$
\alpha^{\mathrm{vis}}_l
=
1
+
\lambda_{\mathrm{vis}}
\tanh(g^{\mathrm{vis}}_l)
$$

当前：

$$
\lambda_{\mathrm{vis}} = 0.5
$$

Plan scale：

$$
\beta^{\mathrm{plan}}_l
=
1
+
\lambda_{\mathrm{plan}}
\tanh(g^{\mathrm{plan}}_l)
$$

当前：

$$
\lambda_{\mathrm{plan}} = 0.25
$$

Plan scale 在 cross-attn 之前只作用到 plan token，不缩放 state token：

$$
\tilde{P}_{l,k}
=
\beta^{\mathrm{plan}}_l
P_k
$$

因此 action-condition branch 使用：

$$
C^{\mathrm{act}}_{l,k}
=
\left[
\tilde{P}_{l,k},\ z^s_k
\right]
$$

`O_act` 不再额外乘 `beta_plan`，因为 plan token 在进入 action-condition branch 前已经被缩放。

## 7. Stage 1 Loss

Stage 1 唯一主损失是 masked flow matching velocity loss：

```yaml
loss:
  type: masked_flow_matching_mse
  target: velocity
  action_dim: 7
  horizon: 32

  flow_time:
    distribution: beta
    alpha: 2.0
    beta: 2.0
    min_t: 0.02
    max_t: 0.98

  noise:
    distribution: uniform
    low: -1.0
    high: 1.0

  gripper:
    separate_head: false
    loss: none
    included_in_action_vector: true
    dim_index: -1
    dim_weight: 1.0
```

不加入 planner loss、memory loss、intent latent loss、gate regularization 或 gripper BCE。

因此：

$$
\mathcal{L}_{\mathrm{stage1}}
=
\mathcal{L}_{\mathrm{FM}}
$$

## 8. Flow Matching Formula

单个样本的 normalized future action chunk：

$$
A_k
\in
\mathbb{R}^{32 \times 7}
$$

采样噪声：

$$
\epsilon
\sim
\mathcal{U}(-1, 1)
$$

采样 flow time：

$$
\tau
\sim
\mathrm{Beta}(2,2)
$$

并截断到：

$$
\tau \in [0.02, 0.98]
$$

构造 noisy action：

$$
X_{\tau}
=
(1-\tau)\epsilon
+
\tau A_k
$$

目标 velocity：

$$
V^\star
=
A_k
-
\epsilon
$$

模型预测：

$$
\hat{V}_{\theta}
=
f_{\theta}
\left(
X_{\tau},
\tau,
F_k,
S_k,
P_k,
s_k
\right)
$$

masked flow matching MSE：

$$
\mathcal{L}_{\mathrm{FM}}
=
\frac{
\sum_i
m_i
\left(
\hat{V}_{\theta,i}
-
V^\star_i
\right)^2
}{
\sum_i m_i + \epsilon_{\mathrm{denom}}
}
$$

其中 `m_i` 来自 `action_mask`。实现中 denominator 应加入极小常数：

$$
\epsilon_{\mathrm{denom}} = 10^{-8}
$$

同时如果 `action_mask.sum() == 0`，应直接报错，因为这说明数据或 mask 生成有问题。

## 9. Gripper Handling

LIBERO action 维度为：

$$
d_a = 7
$$

当前 gripper 是 action vector 的最后一维：

$$
a^{\mathrm{gripper}} = a_{-1}
$$

Stage 1 不拆单独 gripper head：

```yaml
gripper_head: false
gripper_loss: none
gripper_dim_weight: 1.0
```

如果后续发现 gripper 学习明显弱于其他维度，优先考虑 action dim weight，而不是立刻引入独立分类 head：

$$
\mathcal{L}_{\mathrm{FM}}
=
\frac{
\sum_i
w_i m_i
\left(
\hat{V}_{\theta,i}
-
V^\star_i
\right)^2
}{
\sum_i w_i m_i + \epsilon_{\mathrm{denom}}
}
$$

Stage 1 当前不启用该加权版本。

## 10. Action Normalization

Action normalization 必须在 Stage 1、Stage 2 和 rollout inference 之间保持一致：

```yaml
action_normalization:
  enabled: true
  type: train_split_minmax_to_minus_one_one
  statistics_from: train_split
  apply_to_training_targets: true
  apply_to_flow_matching_target_actions: true
  clip_after_normalization: true
  clip_range: [-1.0, 1.0]
  denormalize_for_rollout: true
  save_normalizer_in_checkpoint: true
  require_same_normalizer_for_stage2: true
```

原因是 flow matching 噪声尺度固定为：

$$
\epsilon
\sim
\mathcal{U}(-1, 1)
$$

所以 normalized action `A_k` 的尺度必须和噪声尺度匹配。否则：

$$
V^\star = A_k - \epsilon
$$

的尺度会异常，导致 velocity prediction 不稳定。

当前 Stage 1 不使用 z-score normalization。若后续改成 mean/std z-score，则 flow matching noise 也需要重新讨论，不能默认继续使用 `Uniform(-1, 1)`。

## 11. Rollout Inference

Stage 1 rollout / smoke inference 必须和训练时的 flow matching 定义对齐：

```yaml
rollout_inference:
  sampler: euler
  num_flow_steps: 15
  tau_schedule: midpoint
  avoid_endpoint_tau: true
  init_noise_distribution: uniform
  init_noise_low: -1.0
  init_noise_high: 1.0
  predict: velocity
  denormalize_action_before_env_step: true
  execute_first_n_actions: 16
```

当前 server / rollout / smoke inference 统一使用 `15` 步 Euler 积分。训练阶段不进行多步积分；训练只随机采样单个 flow time，并优化 masked flow matching velocity loss。

推理初始化：

$$
X_0
\sim
\mathcal{U}(-1, 1)
$$

Euler 更新：

$$
N = 15
$$

$$
\Delta \tau
=
\frac{1}{N}
$$

midpoint tau grid：

$$
\tau_i
=
\frac{i + 0.5}{N},
\quad
i = 0,1,\ldots,N-1
$$

当 `N = 15` 时：

$$
\tau_{\min}
=
\frac{1}{30}
\approx
0.033
$$

$$
\tau_{\max}
=
\frac{29}{30}
\approx
0.967
$$

这落在训练时的 flow time 范围内：

$$
[0.02, 0.98]
$$

Euler 更新：

$$
X_{i+1}
=
X_i
+
\Delta \tau
\hat{V}_{\theta}
\left(
X_i, \tau_i, F_k, S_k, P_k, s_k
\right)
$$

得到 normalized action chunk 后，必须先 denormalize，再送入环境执行。每次 rollout 只执行前 `16` 步，和训练中的 `replan_interval = 16` 对齐。

## 12. Optimizer

Stage 1 使用 AdamW：

```yaml
optimizer:
  type: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 1.0e-3
  grad_clip: 1.0
```

不做 weight decay 的参数：

```yaml
no_weight_decay:
  - bias
  - norm
  - layernorm
  - LayerNorm
  - rmsnorm
  - RMSNorm
  - gate
  - gates
  - pos_embedding
  - position_embedding
  - time_embedding
  - timestep_embedding
```

`gate`、norm、bias、pos embedding 和 time embedding 不做 weight decay。

## 13. LR Groups

```yaml
lr:
  flow_matching_action_head: 5.0e-5
  action_expert: 5.0e-5

  flow_time_embedding: 1.0e-4
  timestep_mlp: 1.0e-4
  noisy_action_encoder: 1.0e-4

  bridge_attention: 1.0e-4
  bridge_adapter: 1.0e-4

  short_memory_encoder: 1.0e-4
  short_memory_projector: 1.0e-4

  plan_projector: 1.0e-4
  progress_condition_projector: 1.0e-4
  source_mlp: 1.0e-4

  gates: 5.0e-5
  action_pos_embedding: 5.0e-5
  temporal_pos_embedding: 5.0e-5
```

`flow_matching_action_head` 和 `action_expert` 主体用 `5e-5`，避免 velocity prediction 前期过强振荡。bridge、projector、time/noisy action 相关模块用 `1e-4`，让条件对齐更快。

## 14. Scheduler

```yaml
scheduler:
  type: cosine
  warmup_steps: 500
  min_lr_ratio: 0.1
```

前 `500` optimizer steps 线性 warmup，之后 cosine decay 到 peak learning rate 的 `10%`。本次从 500-step checkpoint resume 时，scheduler 按 checkpoint `next_step=501` 对齐继续退火。

## 15. Dropout

```yaml
dropout:
  action_expert: 0.1
  action_head: 0.1
  bridge_attention: 0.1
  bridge_adapter: 0.1
  short_memory: 0.1
  plan_condition: 0.05
```

如果 `2k` 到 `5k` 后出现明显过拟合，再提高 action expert、bridge attention、action head 的 dropout。

## 16. Batch Size

Stage 1 使用 episode-level fixed-replan-node feature cache。当前 `batch_size` 表示 episode 数量，不是 frame 数量，也不是 replan node 数量。推荐：

```yaml
batch:
  episode_batch_size: 1
  dataset_episodes: 500
  optimizer_steps: 5000
  approximate_passes: 10
```

每个 optimizer step 处理一条 episode，在 episode 内顺序递推 frozen progress state `M`，并对所有 full-horizon nodes 计算 FM loss。有效 loss rows 会随 episode 长度和尾部 full-horizon 可用性变化，不再由 `batch_size * loss_replan_steps` 固定决定。

## 17. Eval And Checkpoint

当前 `scripts/train.py --config configs/experiment/libero_stage1.yaml` 是 Stage 1 的 active 训练入口，只支持 episode-level `libero_episode_feature_cache`、frozen W4 ProgressPlanner、direct bridge action head 和 masked flow-matching loss。它只实现 training loss 记录和 checkpoint 保存；还没有内置 episode-level validation loop。因此当前 `step_best` 是按训练 loss 监控得到的 health checkpoint，不应被解释为 validation-best checkpoint。

当前实现使用：

```yaml
checkpoint:
  save_interval: 1000000
  best_ckpt_min_step: 1
  best_ckpt_interval: 1
  save_best_by: train_fm_loss
```

其中：

```text
save_interval = 1000000:
  不在 5000-step Stage1 过程中保存周期 checkpoint

best_ckpt_min_step = 1:
  从训练开始就允许覆盖 step_best

best_ckpt_interval = 1:
  每次训练 loss 创新低都覆盖 step_best
```

判断是否切 Stage 2 时，不能只看 train loss：

```text
train FM loss 没有数值异常
gate 没有塌缩
source norm 没有爆炸
smoke rollout 没有明显异常
```

如果满足这些条件，可以结束 Stage 1。否则继续完成当前 `5000` optimizer steps。

后续如果加入 validation，split 应按 episode 划分，而不是按 replan node 随机划分：

```yaml
validation:
  split: episode_level
  train_ratio: 0.9
  val_ratio: 0.1
```

原因是 node-level split 会把同一个 episode 的相邻片段同时放进 train 和 val，使 validation loss 偏乐观。

## 18. Metrics

基础指标：

```text
train/fm_loss
lr
grad_norm
```

如果后续实现 validation，再加入：

```text
val/fm_loss
```

Flow matching 相关：

```text
fm/time_mean
fm/noise_norm
fm/target_velocity_norm
fm/pred_velocity_norm
```

Bridge 相关：

```text
bridge/alpha_vis
bridge/beta_plan
```

Source 相关：

```text
source/vlm_norm
source/short_memory_norm
source/plan_norm
source/state_norm
```

Plan 使用情况：

```text
plan_token_norm
plan_token_std
```

如果 `beta_plan` 长期不变化，或者移除 `plan_tokens` 后 validation loss 几乎不变，说明 action head 可能主要依赖当前视觉和 state，没有有效使用 plan condition。

## 19. Stage 1 Full YAML

```yaml
stage_1:
  name: frozen_vlm_cache_action_side_training
  max_steps: 5000

  suite: libero_10
  input_mode: token_cache
  use_token_cache: true
  use_raw_images: false
  load_vlm: false

  horizon: 32
  stride: 16
  replan_interval: 16
  action_dim: 7
  state_dim: 8
  drop_tail_shorter_than_horizon: true
  enable_bridge_aux_loss: false

  dataset:
    cache_type: libero_episode_feature_cache
    detach_cached_tokens: true
    fixed_replan_nodes: true
    episode_batch_size: 1
    loss_nodes: full_horizon_only

  freeze:
    vision_encoder: true
    vlm_backbone: true
    llm_backbone: true
    vlm_multimodal_projector: true

  progress_planner:
    body_trainable: false
    source: frozen_w4_checkpoint
    checkpoint: $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1/best.pt
    online_forward: true
    finetune: false

  action_side_plan_condition:
    plan_projector_trainable: true
    progress_condition_projector_trainable: true

  trainable:
    flow_matching_action_head: true
    action_expert: true
    flow_time_embedding: true
    timestep_mlp: true
    noisy_action_encoder: true
    action_pos_embedding: true
    temporal_pos_embedding: true
    bridge_attention: true
    bridge_adapter: true
    gates: true
    short_memory_encoder: true
    short_memory_projector: true
    plan_projector: true
    progress_condition_projector: true
    source_mlp: true
    gripper_head: false

  loss:
    type: masked_flow_matching_mse
    target: velocity
    auxiliary_losses_enabled: false
    action_dim: 7
    horizon: 32
    denom_eps: 1.0e-8
    flow_time:
      distribution: beta
      alpha: 2.0
      beta: 2.0
      min_t: 0.02
      max_t: 0.98
    noise:
      distribution: uniform
      low: -1.0
      high: 1.0
    gripper:
      separate_head: false
      loss: none
      included_in_action_vector: true
      dim_index: -1
      dim_weight: 1.0

  action_normalization:
    enabled: true
    type: train_split_minmax_to_minus_one_one
    statistics_from: train_split
    apply_to_training_targets: true
    apply_to_flow_matching_target_actions: true
    clip_after_normalization: true
    clip_range: [-1.0, 1.0]
    denormalize_for_rollout: true
    save_normalizer_in_checkpoint: true
    require_same_normalizer_for_stage2: true

  rollout_inference:
    sampler: euler
    num_flow_steps: 15
    tau_schedule: midpoint
    avoid_endpoint_tau: true
    init_noise_distribution: uniform
    init_noise_low: -1.0
    init_noise_high: 1.0
    predict: velocity
    denormalize_action_before_env_step: true
    execute_first_n_actions: 16

  optimizer:
    type: adamw
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay: 1.0e-3
    grad_clip: 1.0

  lr_groups:
    flow_matching_action_head: 5.0e-5
    action_expert: 5.0e-5
    flow_time_embedding: 1.0e-4
    timestep_mlp: 1.0e-4
    noisy_action_encoder: 1.0e-4
    bridge_attention: 1.0e-4
    bridge_adapter: 1.0e-4
    short_memory_encoder: 1.0e-4
    short_memory_projector: 1.0e-4
    plan_projector: 1.0e-4
    progress_condition_projector: 1.0e-4
    source_mlp: 1.0e-4
    gates: 5.0e-5
    action_pos_embedding: 5.0e-5
    temporal_pos_embedding: 5.0e-5

  scheduler:
    type: cosine
    warmup_steps: 500
    min_lr_ratio: 0.1

  batch:
    batch_size: 1
    unit: episode
    dataset_episodes: 500
    approximate_passes_for_5000_steps: 10

  checkpoint:
    save_interval: 1000000
    best_ckpt_min_step: 1
    best_ckpt_interval: 1
    save_best_by: train_fm_loss
```

## 20. 训练前检查

Stage 1 开始前必须确认：

- `libero_10` token cache 已完成；
- replay index 使用 `H=32`、`R=16`；
- short memory offsets 是 `16` 和 `8`；
- selected VLM hidden layers 是 `[3, 6, 9, 12]`；
- action target 已用 train split statistics 归一化；
- checkpoint 中保存 action normalizer；
- plan tokens 来自 W4 progress planner；
- cache dataloader 返回的 tensor shape 与 direct bridge action head 一致；
- `gripper_head=false`；
- 训练 loss 只有 `masked_flow_matching_mse`；
- 训练前通过 direct bridge inference smoke test。
