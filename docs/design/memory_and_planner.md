# Progress-State Planner 与 Long Memory 设计

状态：当前确定设计。本文替代旧的 Dual-FIFO long visual memory、H32 action-latent planner 主线，以及所有把 long memory 表述为历史 token bank 的旧设计。

## 1. 核心定义

当前系统保留两个 memory 概念：

```text
short memory = recent visual-token memory
long memory  = task-progress state
```

short memory 是独立视觉记忆模块，保存最近视觉证据。long memory 不是历史仓库，不从历史 visual tokens 里检索，也不维护不断增长的 bank。long memory 是 planner 内部维护的任务进展状态。

当前固定两个时间尺度：

$$
H = 32
$$

$$
R = 16
$$

其中 `H` 是 action prediction horizon，`R` 是 replan stride。planner 每次对未来 `H` 步形成 intent 监督，long memory 每隔 `R` 个 low-level steps 更新一次。

设 replan index 为 `k`，对应 low-level time：

$$
t_k = kR
$$

系统按 replan step 同步更新：

```text
replan step k:
  输入当前 VL summary、当前机器人状态、上一段实际执行的 R-step action segment
  更新 long memory
  planner 输出当前 H-step horizon 的 intent token
```

整体形式：

$$
S_k =
\operatorname{ShortVisualMemory}
(
V_{t_k-\frac{R}{2}},
V_{t_k-R}
)
$$

$$
x_k = \operatorname{ProgressEvidenceEncoder}(h_k, s_k, u_k)
$$

$$
M_k = \operatorname{ProgressStateUpdater}(M_{k-1}, x_k)
$$

$$
P_k = \operatorname{Planner}(M_k, h_k, s_k)
$$

其中：

```text
S_k: short visual memory tokens
M_k: long memory / task-progress state
P_k: planner intent token
F_k: 当前 VL hidden states，只用于生成 h_k
h_k: 当前 VL pooled summary，warm-up cache 中实际保存的 VL 表征
s_k: 当前机器人状态
u_k: 上一次 replan 之后实际执行的 R-step action summary
```

`P_k` 是 planner 输出，不是 long memory 本身。long memory 是 `M_k`。

## 2. 固定网络参数

当前设计固定使用以下参数：

```text
VL / policy hidden dim D: 896
action horizon H: 32
replan stride R: 16
burnin replan steps: 8
loss replan steps W: 4 or 8
attention heads: 8
head dim: 112
planner cross-attention layers: 2
state encoder hidden dim: 512
action summary MLP hidden dim: 512
updater MLP hidden dim: 1792
planner FFN hidden dim: 3584
intent latent dim: 128
dropout: 0.05
```

代码命名保持一致：

```text
horizon = 32
replan_stride = 16
vl_tokens = F_k
vl_summary = h_k
```

代码里不要用 `H_k` 表示 hidden states，避免和 horizon `H` 混淆。

动作维度按 benchmark 决定：

```text
LIBERO / 单臂默认 d_a = 7
```

## 3. Long Memory 形式

long memory 使用两个 token：

$$
M_k = [C_k, G_k] \in \mathbb{R}^{2 \times 896}
$$

其中：

```text
C_k: completed-events state token
G_k: current-stage state token
```

两个 token 的职责不同：

```text
C_k 记录任务已经完成的事件
G_k 表示当前所处的任务阶段
```

这个分工是结构约束。网络更新时必须先更新 `C_k`，再使用更新后的 `C_k` 更新 `G_k`。

初始状态是可学习参数：

$$
C_0 = \theta_C
$$

$$
G_0 = \theta_G
$$

其中：

$$
\theta_C,\theta_G \in \mathbb{R}^{896}
$$

## 4. VL 表征

progress-state planner 只消费一个 VL summary：

$$
h_k \in \mathbb{R}^{896}
$$

warm-up cache 中的 `h_k` 由 `InternVL3VLSummaryEncoder` 固定生成。设最后一层 language hidden states 为：

$$
F_k^{final}
=
\operatorname{InternVL3}^{final}
(
o_{t_k},\ \text{instruction}
)
$$

其中：

$$
F_k^{final}
\in
\mathbb{R}^{N \times 896}
$$

设 attention mask 为：

$$
m_k \in \{0,1\}^{N}
$$

取最后一个有效 token index：

$$
i_k =
\max
\{i \mid m_{k,i}=1\}
$$

warm-up cache 保存：

$$
h_k =
F^{final}_{k,i_k}
$$

因此 warm-up 阶段不缓存完整 `F_k`。cache 只保存：

```text
vl_summary = h_k: [896]
```

direct bridge-attn 的 raw VLM features 是另一条路径。policy action head 使用多层 token hidden states：

$$
\mathcal{L}_{vlm}
=
\{3,6,9,12\}
$$

这些 token hidden states 进入 visual branch，不进入 progress-state planner。progress planner 的 runtime 接口优先使用显式 `planner_vl_summary`。如果没有显式 summary，代码会对提供的 `planner_fused_tokens` 做确定性 mean pooling：

$$
h_k =
\frac{1}{N}
\sum_{i=1}^{N}
F_{k,i}
$$

这个 fallback 只用于没有单独 summary 的 visual-token cache 或 smoke path。正式复用 warm-up 分布时，应直接传入 `planner_vl_summary`。

## 5. 已执行动作摘要

动作摘要来自上一次 replan 之后实际执行过的动作 segment，只描述已经发生的控制历史。

$$
\tilde A^{exec}_{k-1}
=
\operatorname{Norm}_a
\left(
a_{t_k-R:t_k-1}
\right)
$$

当前固定 `R=16`：

$$
\tilde A^{exec}_{k-1} \in \mathbb{R}^{16 \times d_a}
$$

动作摘要为：

$$
u_k =
\operatorname{ActionSummaryEncoder}
(
\tilde A^{exec}_{k-1},\ m^{exec}_{k-1}
)
$$

其中：

$$
m^{exec}_{k-1} \in \{0,1\}^{16}
$$

episode 开头没有上一段动作时：

$$
\tilde A^{exec}_{k-1} = 0
$$

$$
m^{exec}_{k-1} = 0
$$

ActionSummaryEncoder 使用固定结构：

```text
input: flatten([normalized_executed_actions, executed_action_mask])
MLP: 16 * (d_a + 1) -> 512 -> 896
activation: GELU
dropout: 0.05
output norm: LayerNorm(896)
output: [896]
```

公式：

$$
\bar A^{exec}_{k-1}
=
\tilde A^{exec}_{k-1}
\odot
m^{exec}_{k-1}
$$

$$
f^a_k =
\operatorname{Flatten}
(
[\bar A^{exec}_{k-1},\ m^{exec}_{k-1}]
)
$$

$$
u_k =
\operatorname{LN}
\left(
\operatorname{MLP}_u(f^a_k)
\right)
$$

如果 `m^{exec}_{k-1}` 全 0，则直接置零：

$$
u_k = 0
$$

当前要预测的 normalized target action 不能进入 long memory updater。

## 6. Progress Evidence Encoder

long memory 更新先构造进展证据：

$$
x_k = \operatorname{ProgressEvidenceEncoder}(h_k, s_k, u_k)
$$

状态编码：

$$
e^s_k =
\operatorname{LN}
\left(
\operatorname{MLP}_s(s_k)
\right)
$$

其中 `MLP_s` 结构为：

```text
d_s -> 512 -> 896
activation: GELU
```

证据融合：

$$
x_k =
\operatorname{LN}
\left(
\operatorname{MLP}_x([h_k,\ e^s_k,\ u_k])
\right)
$$

其中 `MLP_x` 结构为：

```text
2688 -> 1792 -> 896
activation: GELU
dropout: 0.05
```

## 7. Completed-Events Token 更新

completed-events token 先更新。

构造输入：

$$
r^C_k = [C_{k-1},\ G_{k-1},\ x_k]
$$

其中：

$$
r^C_k \in \mathbb{R}^{2688}
$$

候选更新：

$$
\Delta C_k =
\operatorname{MLP}_C(r^C_k)
$$

其中 `MLP_C` 结构为：

```text
2688 -> 1792 -> 896
activation: GELU
dropout: 0.05
```

更新门：

$$
g^C_k =
\sigma
\left(
\operatorname{MLP}^{g}_C(r^C_k)
\right)
$$

其中：

$$
g^C_k \in \mathbb{R}^{896}
$$

状态更新：

$$
C_k =
\operatorname{LN}
\left(
C_{k-1} + g^C_k \odot \Delta C_k
\right)
$$

`C_k` 表示已经完成的任务事件，因此它是慢变化 token。`g^C_k` 使用 vector gate，gate head 最后一层 bias 初始化为 `-2.0`。

## 8. Current-Stage Token 更新

current-stage token 使用更新后的 `C_k`。

构造输入：

$$
r^G_k = [G_{k-1},\ C_k,\ x_k]
$$

其中：

$$
r^G_k \in \mathbb{R}^{2688}
$$

候选更新：

$$
\Delta G_k =
\operatorname{MLP}_G(r^G_k)
$$

其中 `MLP_G` 结构为：

```text
2688 -> 1792 -> 896
activation: GELU
dropout: 0.05
```

更新门：

$$
g^G_k =
\sigma
\left(
\operatorname{MLP}^{g}_G(r^G_k)
\right)
$$

其中：

$$
g^G_k \in \mathbb{R}^{896}
$$

状态更新：

$$
G_k =
\operatorname{LN}
\left(
G_{k-1} + g^G_k \odot \Delta G_k
\right)
$$

`G_k` 表示当前任务阶段。`G_k` 依赖新的 `C_k`，因为当前阶段判断取决于已经完成的事件。`g^G_k` 使用 vector gate，gate head 最后一层 bias 初始化为 `-1.0`。

## 9. Planner 网络结构

Planner 读取 long memory、当前 VL 全局摘要和当前状态，输出一个 intent token。Planner 不直接读取完整 `F_k` token sequence。

$$
P_k =
\operatorname{Planner}(M_k,\ h_k,\ s_k)
$$

其中：

$$
P_k \in \mathbb{R}^{1 \times 896}
$$

当前 planner 输出一个 intent token。这个 token 表示当前 horizon 的粗粒度下一阶段动作意图，不拆成多个 plan tokens。对应预训练权重、cache 和日志均按：

```text
planner_token: [B, 1, 896]
intent_projection: 896 -> 128
```

解释。

Planner 使用固定结构：

```text
planner query: learned [1, 896]
condition tokens: [h_k, C_k, G_k, T^s_k]
cross-attention layers: 2
heads: 8
FFN hidden dim: 3584
dropout: 0.05
output norm: LayerNorm(896)
```

state token：

$$
T^s_k =
\operatorname{MLP}_{state}(s_k)
\in \mathbb{R}^{1 \times 896}
$$

condition tokens：

$$
R_k =
[h_k,\ C_k,\ G_k,\ T^s_k]
$$

其中 `h_k` 作为单个 VL summary token 输入 planner。完整 `F_k` 只用于产生 `h_k` 和 long memory update evidence，不作为 planner 的直接 cross-attention context。

planner query 初始化为：

$$
Q^p_0 =
q_p
$$

经过两层 cross-attention block：

$$
Q^p_2 =
\operatorname{PlannerBlock}_2
(
Q^p_0,\ R_k
)
$$

输出：

$$
P_k =
\operatorname{LN}(Q^p_2)
$$

## 10. Intent Encoder

预热阶段使用 frozen action intent encoder 生成监督目标。

当前 horizon 的监督动作：

$$
\tilde A^{tar}_k
=
\operatorname{Norm}_a
\left(
a_{t_k:t_k+H-1}
\right)
$$

当前固定 `H=32`：

$$
\tilde A^{tar}_k \in \mathbb{R}^{32 \times d_a}
$$

replan step 必须满足：

$$
t_k + H \le T_{episode}
$$

只有满足该条件的 step 才进入预热训练。尾部不足 32 步的样本直接 skip，不对 normalized target action 做 padding，也不使用 target mask 训练 intent target。

intent encoder 输出：

$$
z_k =
\operatorname{IntentEncoder}(\tilde A^{tar}_k)
$$

其中：

$$
z_k \in \mathbb{R}^{128}
$$

IntentEncoder 使用当前已有 H32 action-intent autoencoder 的 encoder。预热时该 encoder 冻结，只提供 target。

所有 action 输入必须使用同一套 normalization：

```text
executed action summary input:  tilde A^{exec}_{k-1}
intent encoder target input:    tilde A^{tar}_k
IntentEncoder AE training data: same Norm_a
```


target latent 使用固定 L2 normalization，不使用可学习 LayerNorm：

$$
\bar z_k =
\operatorname{normalize}
\left(
\operatorname{sg}(z_k)
\right)
$$

planner token 投影到 intent latent：

$$
\hat z^P_k =
\operatorname{normalize}
\left(
W_P P_k
\right)
$$

其中：

$$
W_P: \mathbb{R}^{896} \rightarrow \mathbb{R}^{128}
$$

current-stage token 投影到 intent latent：

$$
\hat z^G_k =
\operatorname{normalize}
\left(
W_G G_k
\right)
$$

其中：

$$
W_G: \mathbb{R}^{896} \rightarrow \mathbb{R}^{128}
$$

long memory pooled projection 只作为弱约束：

$$
\bar M_k =
\operatorname{Pool}([C_k,\ G_k])
$$

$$
\hat z^M_k =
\operatorname{normalize}
\left(
W_M \bar M_k
\right)
$$

其中：

$$
W_M: \mathbb{R}^{896} \rightarrow \mathbb{R}^{128}
$$

`G_k` 直接对齐当前 horizon 的 intent target；`C_k` 不直接预测 `z_k`，它通过参与 `G_k` 更新、参与 planner、以及更小的 gate bias 承担慢变化 completed-events state。

## 11. 预热训练

训练顺序必须按 episode window 展开，不能把 replan step 打散成独立样本。随机采样 episode 中间窗口时，loss window 不能直接从 `M_init` 开始。

warm-up dataset 返回：

```text
episode_id
start_k
ctx_start = max(0, start_k - burnin_replan_steps)
burnin_replan_steps = 8
loss_replan_steps = W
burn-in sequence:
  h_k
  state_k
  executed_action_segment_k
loss sequence:
  h_k
  state_k
  executed_action_segment_k
  target_intent_z_k
```

其中 `target_intent_z_k` 是由 frozen IntentEncoder 和相同 `Norm_a` 离线生成的 target latent。warm-up cache 保存：

当前构建两套 window 配置：

```text
W = 4
W = 8
```

`W=4` 提高可用 window 数量，尤其减少短 episode 和小 suite 的样本浪费；`W=8` 保持更长 recurrent loss unroll，用于对照同一结构在更长训练链上的行为。

```text
episode_id
t_k
h_k
state_k
executed_action_segment_k
target_intent_z_k
```

训练时先用 burn-in 恢复当前 episode progress state，再在 loss window 上计算 loss：

```text
M = init_memory(batch_size)

# burn-in: update memory only
for k in burnin_steps:
  with no_grad():
    u_k = ActionSummaryEncoder(executed_action_segment_k)
    x_k = ProgressEvidenceEncoder(h_k, state_k, u_k)
    M = ProgressStateUpdater(M, x_k)

M = detach(M)

# loss window: update memory and train
for k in loss_steps:
  u_k = ActionSummaryEncoder(executed_action_segment_k)
  x_k = ProgressEvidenceEncoder(h_k, state_k, u_k)
  M = ProgressStateUpdater(M, x_k)
  P_k = Planner(M, h_k, state_k)
  loss += lambda_plan L_plan + lambda_stage L_stage + lambda_mem_pool L_mem_pool

if use_order_loss:
  loss += lambda_order L_order
```

只有 `start_k = 0` 的 episode 开头窗口可以直接从 `M_init` 进入 loss window。其他随机窗口必须带 burn-in。

公式形式：

$$
M_k =
\operatorname{ProgressStateUpdater}
(
M_{k-1},\ x_k
)
$$

$$
P_k =
\operatorname{Planner}(M_k,\ h_k,\ s_k)
$$

$$
z_k =
\operatorname{sg}
(
\operatorname{IntentEncoder}(\tilde A^{tar}_k)
)
$$

固定归一化 target：

$$
\bar z_k =
\operatorname{normalize}(z_k)
$$

基础对齐损失：

$$
\ell(\hat z,\bar z)
=
\left\|
\hat z - \bar z
\right\|_2^2
+
0.1
\left(
1 -
\operatorname{cos}(\hat z,\bar z)
\right)
$$

planner loss：

$$
\mathcal{L}_{plan}
=
\ell(\hat z^P_k,\bar z_k)
$$

current-stage loss：

$$
\mathcal{L}_{stage}
=
\ell(\hat z^G_k,\bar z_k)
$$

weak memory-pool loss：

$$
\mathcal{L}_{mem\_pool}
=
\ell(\hat z^M_k,\bar z_k)
$$

progress order head：

$$
\rho_k =
\operatorname{ProgressHead}(\bar M_k)
$$

order loss 可关闭，并且只采样同一个 episode window 内间隔足够远的 pair：

```text
use_order_loss: false
min_order_gap: 2
```

同一个 episode 内取：

$$
j - i \ge 2
$$

$$
\mathcal{L}_{order}
=
-
\log
\sigma
(
\rho_j - \rho_i
)
$$

warm-up loss 权重全部是配置项：

```text
lambda_plan: 1.0
lambda_stage: 0.5
lambda_mem_pool: 0.1
use_order_loss: false
lambda_order: 0.02
min_order_gap: 2
```

其中 `lambda_stage` 默认使用 `0.5`，训练时监控 `G_k` 是否退化：

$$
\lambda_{stage} = 0.5
$$

如果 `cos_{G,P}` 长期接近 1，或者 `G_k` 的 batch variance / effective rank 明显下降，优先把 `lambda_stage` 改为：

```text
0.2 or 0.3
```


预热总损失：

$$
\mathcal{L}_{warmup}
=
\lambda_{plan}\mathcal{L}_{plan}
+
\lambda_{stage}\mathcal{L}_{stage}
+
\lambda_{mem\_pool}\mathcal{L}_{mem\_pool}
+
\mathbb{1}_{order}
\lambda_{order}\mathcal{L}_{order}
$$

必须监控 `G_k` 是否退化成 planner token 的复制。记录：

$$
\operatorname{cos}_{G,P}
=
\operatorname{cos}
\left(
G_k,\ \operatorname{squeeze}(P_k)
\right)
$$

$$
\operatorname{cos}_{\hat z^G,\hat z^P}
=
\operatorname{cos}
\left(
\hat z^G_k,\ \hat z^P_k
\right)
$$

同时记录 batch 内 `G_k` 的 variance 和 effective rank。若 `cos_{G,P}` 长期接近 1 且 `G_k` batch variance 明显下降，说明 `G_k` 正在退化成 planner token，需要降低 `lambda_stage` 或提高 planner query 与 `G_k` 的结构区分。

序列训练使用 TBPTT：

```text
loss unroll replan steps: W
detach M_k every W replan steps
effective low-level span per loss unroll: W * R steps
burn-in span: 8 replan steps
loss span: W replan steps
```

这个预热阶段只训练：

```text
ActionSummaryEncoder
ProgressEvidenceEncoder
ProgressStateUpdater
Planner
W_P
W_G
W_M
ProgressHead
```

IntentEncoder 冻结。

## 12. Short Memory 接口

short memory 保存最近两个历史视觉观测的 visual tokens。当前固定使用：

$$
R = 16
$$

因此 short-memory offsets 是：

$$
\left[
\frac{R}{2},\ R
\right]
=
[8,\ 16]
$$

$$
S_k = [S^1_k,\ S^2_k]
$$

其中：

```text
S^1_k: compressed visual tokens at t_k - R/2
S^2_k: compressed visual tokens at t_k - R
```

每个历史视觉观测先通过共享的 BottleneckSE-style 压缩器压成固定数量的 visual memory tokens：

$$
S^1_k =
\operatorname{Compress}_{vis}
(
V_{t_k-\frac{R}{2}}
)
$$

$$
S^2_k =
\operatorname{Compress}_{vis}
(
V_{t_k-R}
)
$$

其中：

$$
\operatorname{Compress}_{vis}:
\mathbb{R}^{N \times 896}
\rightarrow
\mathbb{R}^{K \times 896}
$$

主线不使用 learnable-query cross-attention 压缩。原因是 query-based compression 在监督较弱时容易退化成固定查询模板。BottleneckSE-style compressor 更接近 MemoryVLA 的 perceptual token 压缩做法：保留视觉 token 的空间网格结构，通过 channel bottleneck 和 squeeze-excitation 做局部视觉证据压缩。

因此输入的视觉 token 应保持视觉塔输出的空间网格结构。若使用 InternVL visual tower，每个 view 的 visual tokens 应按固定 view 顺序分别压缩，再拼接为 short memory context。不要把 text tokens、planner tokens 或 pooled `h_k` 混入 short memory。

固定使用：

$$
K = 16
$$

因此：

$$
S_k \in \mathbb{R}^{32 \times 896}
$$

episode 开头 short memory 不足时使用 padding mask。padding token 保持全零，不使用 learnable null token。

short memory 不进入 long memory updater，也不进入 progress-state warm-up loss。它作为独立视觉记忆模块提供给后续策略侧读取，progress-state warm-up cache 中不保存 short memory tensors。

## 13. 当前训练快照


```text
H = 32
R = 16
burnin_replan_steps = 8
use_order_loss = false
lambda_plan = 1.0
lambda_stage = 0.5
lambda_mem_pool = 0.1
planner_token = 1
```

### 13.1 Cache

```text
LIBERO W=4:
  cache: $AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w4
  step_count:   18199
  window_count: 12199
  suite_window_counts:
    libero_10:      6391
    libero_goal:    1741
    libero_object:  2421
    libero_spatial: 1646

LIBERO W=8:
  cache: $AUTODL_TMP/token_caches/libero_progress_vl_embedding_h32_r16_w8
  step_count:   18199
  window_count: 5429
  suite_window_counts:
    libero_10:      4391
    libero_goal:    430
    libero_object:  481
    libero_spatial: 127

  step_count:   16676
  window_count: 15326

  step_count:   16676
  window_count: 13526
```


```text
vl_summary: [896], bfloat16
state: [16], float32
executed_actions: [16, 14], float32
executed_action_mask: [16], bool
target_intent: [128], float32
```

因此 `data.pt` 只有约 76MB 是合理的。它不是 visual-token cache，也不是图像 cache。


```text
action_dim: 14
horizon: 32
latent_dim: 128
best checkpoint: best.pt
best step: 950
val_loss: 0.015030
val_segment_ae_rec_loss: 0.014776
```

### 13.3 Progress-State Warm-Up

```text
LIBERO W=4:
  run: $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_w4_bs12800_epval_v1
  best step: 310
  val_loss: 0.017872
  val_plan_loss: 0.011099
  val_stage_loss: 0.011190
  val_mem_pool_loss: 0.011780
  val_cos_g_p: -0.021132
  val_stage_effective_rank: 118.357338

LIBERO W=8:
  run: $AUTODL_TMP/runs/progress_warmup/libero_progress_state_planner_h32_r16_bs6656_epval_v1
  best step: 280
  val_loss: 0.021811
  val_plan_loss: 0.013405
  val_stage_loss: 0.013884
  val_mem_pool_loss: 0.014639
  val_cos_g_p: -0.023982
  val_stage_effective_rank: 78.541298

  best step: 590
  val_loss: 0.001225
  val_plan_loss: 0.000732
  val_stage_loss: 0.000779
  val_mem_pool_loss: 0.001029
  val_cos_g_p: 0.005919
  val_stage_effective_rank: 93.020485

  stopped after: step 700
  per-step checkpoints: pruned after summary generation
  best step: 660
  val_loss: 0.001016
  val_plan_loss: 0.000563
  val_stage_loss: 0.000707
  val_mem_pool_loss: 0.000991
  val_cos_g_p: 0.000006
  val_stage_effective_rank: 83.066261
```


$$
0.001225 \rightarrow 0.001016
$$

对应总 validation loss 下降约：

$$
17.1\%
$$

planner 对齐项下降为：

$$
0.000732 \rightarrow 0.000563
$$

`val_cos_g_p` 接近 0，说明 raw `G_k` 与 raw `P_k` 没有直接塌缩成同一个 token。`val_stage_effective_rank` 低于 W=4，说明 W=8 的 stage token 多样性略弱；当前优先使用 W=8 best checkpoint 做后续分析，同时保留 W=4 作为短 window 对照。

## 14. 模块划分

新增模块放在：

```text
prism/model/planner/progress_state.py
```

包含：

```text
ProgressState
ActionSummaryEncoder
ProgressEvidenceEncoder
ProgressStateUpdater
ProgressPlanner
ProgressPretrainHeads
```

### 14.1 ProgressState

```text
completed_events: [B, 896]
current_stage:    [B, 896]
```

### 14.2 ActionSummaryEncoder

```text
input executed_actions: [B, 16, d_a]
input executed_action_mask: [B, 16]
output action_summary: [B, 896]
```

### 14.3 ProgressEvidenceEncoder

```text
input vl_summary: [B, 896]
input state: [B, d_s]
input action_summary: [B, 896]
output progress_evidence: [B, 896]
```

### 14.4 ProgressStateUpdater

```text
input previous progress_state: [B, 2, 896]
input progress_evidence: [B, 896]
output completed_events: [B, 896]
output current_stage: [B, 896]
output gate_stats: g^C_k, g^G_k
```

### 14.5 ProgressPlanner

```text
input progress_state: [B, 2, 896]
input vl_summary: [B, 896]
input state: [B, d_s]
output planner_token: [B, 1, 896]
```

### 14.6 ProgressPretrainHeads

```text
input planner_token: [B, 1, 896]
input progress_state: [B, 2, 896]
output planner_intent: [B, 128]
output stage_intent: [B, 128]
output memory_pool_intent: [B, 128]
output progress_score: [B, 1]
```

## 15. 因果约束

long memory updater 的输入只包含：

```text
progress evidence x_k
  x_k = ProgressEvidenceEncoder(h_k, s_k, u_k)
当前机器人状态 s_k
上一次 replan 之后实际执行的 normalized R-step action segment
上一时刻 long memory M_{k-1}
```

warm-up cache 不保存完整 `F_k`，只保存 `h_k`。`F_k` 只在离线缓存构建时用于生成 `h_k`。

normalized target action 只用于 IntentEncoder target，不进入 updater 输入。尾部不足 `H=32` 的 target 样本不参与预热训练。
