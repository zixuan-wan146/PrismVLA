# Direct Bridge-Attn 最终设计

本文记录当前确定的 bridge-attn 设计。核心原则是：保留 4 类信息源，但只实现 2 个 functional branches。

4 类 source 是：

1. current VLM hidden states
2. short memory tokens
3. plan tokens
4. state / proprio token

最终分组是：

```text
visual evidence branch:
  current VLM hidden states
  short memory tokens

action-condition branch:
  plan tokens
  state token
```

不要实现 4 套 cross-attn，也不要实现 4 套 source-level gate。

## 1. Transformer Hidden States

Transformer 输入是一整个 token sequence。

设第 `l - 1` 层输入为：

$$
X^{l-1}
\in
\mathbb{R}^{B \times N \times D}
$$

经过 self-attention 和 FFN 后，第 `l` 层输出仍然是：

$$
X^l
\in
\mathbb{R}^{B \times N \times D}
$$

也就是说，一层 hidden states 是一组 contextualized token representations，不是单个 feature vector。

第 `i` 个 token 在第 `l` 层的表示是：

$$
x_i^l
\in
\mathbb{R}^{D}
$$

整层 hidden states 是：

$$
X^l =
\left[
x_1^l,
x_2^l,
\ldots,
x_N^l
\right]
$$

bridge-attn 需要 action tokens attend 到这些 token hidden states，因此这里使用 token sequence，不使用 pooled vector。

## 2. VLM Layer Schedule

当前 InternVL3 language model 被截断为前 14 个 transformer blocks：

```python
layers = layers[:14]
```

`output_hidden_states=True` 返回：

$$
\left[
X^0,
X^1,
X^2,
\ldots,
X^{14}
\right]
$$

其中：

- `X^0` 是输入 embedding states。
- `X^l` 是第 `l` 个 transformer block 的输出。

当前设计不使用最后一层 `X^14`。VLM hidden states 固定抽取：

$$
\mathcal{L}_{vlm}
=
\left\{
3,6,9,12
\right\}
$$

当前 action head 使用 8 个 action blocks，对应 schedule：

$$
r =
\left[
3,3,6,6,9,9,12,12
\right]
$$

第 `l` 个 action block 使用：

$$
F_{l,k}
=
F_k^{r_l}
$$

其中：

$$
F_{l,k}
\in
\mathbb{R}^{B \times N_{vlm} \times D}
$$

`F_{l,k}` 是 current VLM image-context token hidden states。

## 3. Source Grouping

最终只保留两个 context branches。

visual evidence branch：

$$
C_{l,k}^{vis}
=
\left[
F_{l,k},
S_k
\right]
$$

action-condition branch：

$$
C_{k,l}^{act}
=
\left[
P_k,
z_k^s
\right]
$$

其中：

- `F_lk` 是 current VLM hidden states。
- `S_k` 是 short memory tokens。
- `P_k` 是 plan tokens / plan slots。
- `z_k_s` 是 state token。

这里的拼接都是 token-level concat，不是 feature 维度拼接。

## 4. Short Memory Adapter

short memory 不裸拼到 VLM feature 后面，使用一个轻量 adapter。

adapter 使用 residual MLP：

$$
\hat{S}_k
=
S_k
+
\gamma_{mem}
A_{mem}
\left(
LN_{mem}(S_k)
\right)
$$

其中：

$$
\gamma_{mem}
\leftarrow
0.1
$$

`A_mem` 固定使用：

```text
LayerNorm -> Linear -> GELU -> Linear
```

这个 adapter 的目的不是强语义对齐，而是 feature-space adaptation / distribution calibration，让 short memory 成为 bridge-attn 可以稳定消费的 historical visual evidence tokens。

## 5. Source Embedding 与 Time Embedding

为了区分 token 来源，使用 learnable source embeddings。符号使用 `b`，避免和 state token 混淆。

$$
b_{vlm},
b_{mem},
b_{plan},
b_{state}
\in
\mathbb{R}^{1 \times 1 \times D}
$$

short memory 还加入 time-offset embedding：

$$
R_{\Delta t}
\in
\mathbb{R}^{B \times N_s \times D}
$$

current VLM tokens：

$$
\tilde{F}_{l,k}
=
LN_{vlm}(F_{l,k})
+
b_{vlm}
$$

short memory tokens：

$$
\tilde{S}_k
=
\hat{S}_k
+
b_{mem}
+
R_{\Delta t}
$$

visual context：

$$
C_{l,k}^{vis}
=
\left[
\tilde{F}_{l,k},
\tilde{S}_k
\right]
$$

short memory 固定来自两个历史时刻：`t_k - R` 和 `t_k - R/2`。因此 `R_dt` 使用两个 time bins，并按 token 所属历史时刻重复。

## 6. State Token

原始 state / proprio 是：

$$
s_k
\in
\mathbb{R}^{B \times d_s}
$$

通过 MLP 编码成一个 token：

$$
z_k^s
=
MLP_{state}(s_k)
$$

其中：

$$
z_k^s
\in
\mathbb{R}^{B \times 1 \times D}
$$

加入 state source embedding：

$$
\tilde{z}_k^s
=
z_k^s
+
b_{state}
$$

state 作为 action prediction 的硬条件直接进入 action-condition branch。

## 7. Plan Adapter 与 Plan Gate

progress-state planner 输出一个 base plan token：

$$
P_k^{base}
\in
\mathbb{R}^{B \times 1 \times D}
$$

action head 在内部把它扩展成 8 个 virtual plan slots：

$$
N_p = 8
$$

$$
P_k
\in
\mathbb{R}^{B \times N_p \times D}
$$

如果外部直接传入多个 plan tokens，action head 会先按 mask 做平均 pooling，再加 slot embeddings 得到固定数量的 plan slots。因此当前 policy-side contract 仍然是固定 8 个 action-condition plan slots。

plan slots 是：

$$
P_k
\in
\mathbb{R}^{B \times N_p \times D}
$$

plan adapter 固定使用：

```text
LayerNorm -> Linear
```

action-condition branch 由 gated plan tokens 和直接注入的 state token 组成。

原因是：

$$
C_{k,l}^{act}
=
\left[
P_k,
z_k^s
\right]
$$

gate 整个 action branch 会把 state token 一起缩放，因此这里只 gate plan tokens。

plan gate 定义为：

$$
\beta_l^{plan}
=
1
+
\lambda_{plan}
\tanh
\left(
g_l^{plan}
\right)
$$

其中：

$$
\lambda_{plan}
=
0.25
$$

$$
g_l^{plan}
\leftarrow
0
$$

初始化时：

$$
\beta_l^{plan}
=
1
$$

取值范围：

$$
\beta_l^{plan}
\in
\left(
1-\lambda_{plan},
1+\lambda_{plan}
\right)
$$

当 `lambda_plan = 0.25` 时：

$$
\beta_l^{plan}
\in
\left(
0.75,
1.25
\right)
$$

构造 plan tokens：

$$
\tilde{P}_{k,l}
=
\beta_l^{plan}
A_{plan}(P_k)
+
b_{plan}
$$

构造 action-condition context：

$$
C_{k,l}^{act}
=
\left[
\tilde{P}_{k,l},
\tilde{z}_k^s
\right]
$$

## 8. Visual Branch Gate

visual branch 包含 current VLM hidden states 和 short memory tokens：

$$
C_{l,k}^{vis}
=
\left[
\tilde{F}_{l,k},
\tilde{S}_k
\right]
$$

它们都是 visual evidence，因此作为整体 branch 调节。

visual cross-attn 输出：

$$
O_l^{vis}
=
CA_l^{vis}
\left(
LN(X_l'),
C_{l,k}^{vis}
\right)
$$

visual gate 使用 output-level gate：

$$
\alpha_l^{vis}
=
1
+
\lambda_{vis}
\tanh
\left(
g_l^{vis}
\right)
$$

其中：

$$
\lambda_{vis}
=
0.5
$$

$$
g_l^{vis}
\leftarrow
0
$$

初始化时：

$$
\alpha_l^{vis}
=
1
$$

取值范围：

$$
\alpha_l^{vis}
\in
\left(
1-\lambda_{vis},
1+\lambda_{vis}
\right)
$$

当 `lambda_vis = 0.5` 时：

$$
\alpha_l^{vis}
\in
\left(
0.5,
1.5
\right)
$$

这里采用 output-level gate。相比 VLA-Adapter 的 logit-level gate，这种写法直接调节 visual branch residual，行为更清楚。

## 9. 为什么 Visual Gate 范围更大

当前固定：

```text
lambda_vis = 0.5
lambda_plan = 0.25
```

原因：

visual branch 是主要感知证据来源，包含当前场景、物体位置和近期历史视觉变化，信息量和动态变化都更大，因此允许更大的 adaptive range。

plan tokens 是 high-level intent / task plan。它应该 bias action generation，而不是 dominate action generation，所以调节范围小一点更稳。

重点不是说 VLM 一定比 plan 重要，而是 visual evidence 应该有更大的动态调节空间，plan 应该稳定提供意图偏置。

## 10. Direct Bridge-Attn Block

设第 `l` 个 action block 的输入为：

$$
X_l
\in
\mathbb{R}^{B \times N_a \times D}
$$

其中 `N_a = 32`。

先做 action self-attn：

$$
X_l'
=
X_l
+
SA_l
\left(
LN(X_l)
\right)
$$

两个 cross-attn：

$$
O_l^{vis}
=
CA_l^{vis}
\left(
LN(X_l'),
C_{l,k}^{vis}
\right)
$$

$$
O_l^{act}
=
CA_l^{act}
\left(
LN(X_l'),
C_{k,l}^{act}
\right)
$$

residual update：

$$
\bar{X}_l
=
X_l'
+
\alpha_l^{vis}
O_l^{vis}
+
O_l^{act}
$$

最后 FFN：

$$
X_{l+1}
=
\bar{X}_l
+
FFN_l
\left(
LN(\bar{X}_l)
\right)
$$

注意：action-condition branch 没有整体 gate。plan token 使用自己的轻量 gate，state token 作为硬条件直接进入 cross-attn context。

## 11. 最终结构

最终每个 block 是：

```text
DirectBridgeActionBlock
  action self-attn
  visual cross-attn over [current VLM hidden states, short memory]
  action-condition cross-attn over [plan tokens, state token]
  FFN
```

最终上下文划分是：

$$
C_{l,k}^{vis}
=
\left[
\tilde{F}_{l,k},
\tilde{S}_k
\right]
$$

$$
C_{k,l}^{act}
=
\left[
\tilde{P}_{k,l},
\tilde{z}_k^s
\right]
$$

排除结构：

```text
4 source -> 4 cross-attn -> 4 gate
```

排除结构：

```text
context -> bridge tokens -> action tokens
```

最终目标是让 32 个 noisy action tokens 直接从 visual evidence branch 和 action-condition branch 中读取信息。
