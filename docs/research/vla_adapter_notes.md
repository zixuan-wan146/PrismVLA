# VLA-Adapter 的 Bridge-Adapter 与 Bridge-Attn 设计说明

本文基于本仓库中的 VLA-Adapter 参考代码：

- `reference-repo/vla_adapter`
- source commit: `23fa0c9c159e2aa04341cdd3e924f44061311060`

重点解释三个问题：

1. VLA-Adapter 里的 bridge-adapter 到底是什么。
2. bridge-attn 在 action head 中具体怎么做。
3. 它的 context 和 gating 设计对 PrismVLA 有什么启发。

## 1. 总体结论

VLA-Adapter 并不是先生成一组独立的中间 bridge tokens，再让 action head 去读这些 bridge tokens。

它更接近下面这个流程：

1. 在 VLM/LLM 输入序列里插入 `NUM_TOKENS = 64` 个 learned action query embeddings。
2. 这些 action query tokens 和图像 patch tokens、语言 tokens 一起经过 LLM。
3. 从多个 LLM hidden layers 中取出两类 context：
   - visual/task hidden states
   - action-query hidden states
4. action head 内部用 action decoder tokens 作为 query，通过 gated cross-attention 读取这些 context。

所以 VLA-Adapter 的核心 bridge 并不是一个单独模块名，而是让 action decoder tokens 通过 gated cross-attention 读取 VLM hidden states，最后预测 continuous actions。

源码主要位置：

- action query 插入：`reference-repo/vla_adapter/prismatic/extern/hf/modeling_prismatic.py`
- action head 和 bridge-attn：`reference-repo/vla_adapter/prismatic/models/action_heads.py`
- proprio/noisy action projector：`reference-repo/vla_adapter/prismatic/models/projectors.py`
- token/action constants：`reference-repo/vla_adapter/prismatic/vla/constants.py`

## 2. Token 流程

VLA-Adapter 使用 Qwen2.5-0.5B 作为 LLM backbone。它定义：

$$
N_a = 64
$$

其中 `N_a` 是 action query token 数量，对应源码里的 `NUM_TOKENS = 64`。

模型里有一组 learned action query embeddings：

$$
A^q \in \mathbb{R}^{N_a \times D}
$$

源码中初始化位置是：

```python
self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim)
self.action_queries.weight.data.zero_()
```

训练或推理时，原本 action token 位置上的 embedding 会被替换成这组 learned action queries：

$$
E_{input}[m_{action}]
\leftarrow
A^q
$$

然后构造 multimodal sequence：

$$
Z_0 =
\left[
e_{bos},
V,
E_{text/action}
\right]
$$

其中 `V` 是 projected visual patch embeddings：

$$
V \in \mathbb{R}^{N_v \times D}
$$

`E_text/action` 是语言 token embedding 加上被替换后的 action query embeddings。`D` 是 LLM hidden dimension。

这整个序列进入 LLM：

$$
Z^\ell = \operatorname{LLM}_\ell(Z^{\ell-1})
$$

## 3. Context 是怎么取出来的

VLA-Adapter 会从 LLM 多层 hidden states 中抽取两类 tokens。

第一类是 visual/task tokens：

$$
T^\ell =
Z^\ell_{1:N_v}
$$

第二类是 action-query hidden states：

$$
A^\ell =
Z^\ell_{N_v+N_p:N_v+N_p+N_a}
$$

其中 `N_p` 是 prompt tokens 的数量。

源码中的对应逻辑是：

```python
actions_hidden_states = text_hidden_states[
    :, NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + NUM_TOKENS
]

task_latten_states = item[:, :NUM_PATCHES]

all_hidden_states = torch.cat((task_latten_states, actions_hidden_states), 2)
```

因此 action head 接收到的 `actions_hidden_states` 不是只有 action tokens，而是：

$$
H^\ell =
\left[
T^\ell,
A^\ell
\right]
$$

多层堆叠后：

$$
H =
\left\{
H^0, H^1, \ldots, H^L
\right\}
$$

这点很关键：VLA-Adapter 的 context 是多层 VLM hidden states，而不是一个已经压缩好的单一 context token。

## 4. Proprio Context

机器人状态通过一个小 MLP 投影到 LLM hidden dimension：

$$
p =
W_2 \,
\sigma(W_1 s)
$$

其中 `s` 是 proprio state：

$$
s \in \mathbb{R}^{d_s}
$$

`p` 是状态 token：

$$
p \in \mathbb{R}^{D}
$$

源码对应 `ProprioProjector`：

```python
projected_features = self.fc1(proprio)
projected_features = self.act_fn1(projected_features)
projected_features = self.fc2(projected_features)
```

## 5. Action Head 的输入 tokens

VLA-Adapter 的 continuous action head 是 `L1RegressionActionHead`。

它先把 hidden states 切成：

$$
T =
H_{:, :, 0:N_v, :}
$$

$$
A =
H_{:, :, N_v:N_v+N_a, :}
$$

其中 `T` 是 visual/task hidden states，`A` 是 action-query hidden states。

然后它创建一组 action decoder tokens。对于 LIBERO：

$$
C = 8
$$

其中 `C` 是 action chunk length，对应源码里的 `NUM_ACTIONS_CHUNK = 8`。

初始化时，action decoder tokens 来自一个零张量：

$$
X_0 \in \mathbb{R}^{C \times d_a D}
$$

经过第一层 MLP 投影后变成：

$$
X_0 \in \mathbb{R}^{C \times D}
$$

这里的 `X_0` 才是 action head 中真正作为 query 的 action decoder tokens。

也就是说，VLA-Adapter 里有两类容易混淆的 action tokens：

- `action_queries`: 64 个 learned embeddings，插入 LLM 序列，经过 LLM 后形成 `A^ell`。
- `action decoder tokens`: action head 内部的 `C` 个 tokens，用来预测 continuous action chunk。

这两者不是同一个东西。

## 6. Bridge-Attn 的基本形式

在 `MLPResNet` 中，每个 block 都会做一次带 context 的 attention：

$$
X_\ell =
\operatorname{BridgeAttn}_\ell
\left(
X_{\ell-1},
A^\ell,
T^\ell,
p
\right)
$$

其中：

- `X_(ell-1)` 是 action decoder tokens。
- `A^ell` 是 action-query hidden states。
- `T^ell` 是 visual/task hidden states。
- `p` 是 proprio token。

源码中是：

```python
x = block(
    x,
    h_t=task_hidden_states[:, i + 1, :],
    h_a=actions_hidden_states[:, i + 1, :],
    p=p,
)
```

注意源码变量名有些混乱：

- `h_t` 实际表示 visual/task tokens。
- `h_a` 实际表示 action-query hidden states。
- `p` 是 proprio token。

## 7. 原版 MLPResNetBlock 的 Attention

原版 `MLPResNetBlock` 把 context 分成三组：

1. self branch: 当前 action decoder tokens `X`
2. action/proprio branch: `A^ell` 和 `p`
3. visual/task branch: `T^ell`

定义：

$$
C_a =
\left[
A^\ell,
p
\right]
$$

query 来自 action decoder tokens：

$$
Q = XW_Q
$$

self branch:

$$
K_x = XW_K,
\quad
V_x = XW_V
$$

action/proprio branch:

$$
K_a = C_aW_K,
\quad
V_a = C_aW_V
$$

visual/task branch:

$$
K_t = T^\ell W_K,
\quad
V_t = T^\ell W_V
$$

VLA-Adapter 用一个 zero-initialized scalar gate：

$$
\alpha = \tanh(g)
$$

$$
g \leftarrow 0
$$

attention logits 是：

$$
L =
\frac{
\left[
QK_x^\top,
QK_a^\top,
\alpha QK_t^\top
\right]
}{
\sqrt{d_h}
}
$$

然后：

$$
W =
\operatorname{softmax}(L)
$$

value 拼接：

$$
V =
\left[
V_x,
V_a,
V_t
\right]
$$

attention 输出：

$$
O = W V
$$

最后经过输出投影、残差和 FFN：

$$
X' =
\operatorname{FFN}
\left(
X + OW_O
\right)
$$

这个就是 VLA-Adapter 原版 bridge-attn 的核心。

## 8. Pro 版本的 Bridge-Attn

`MLPResNetBlock_Pro` 做了几处增强：

1. self branch、adapter branch、task branch 使用独立的 K/V projection。
2. 对不同 branch 的 key 加 RoPE。
3. 保留 zero-initialized tanh gate。

Pro 版本中：

$$
C_{adapter} =
\left[
A^\ell,
p
\right]
$$

$$
C_{task} =
T^\ell
$$

attention logits 是：

$$
L =
\frac{
\left[
QK_x^\top,
QK_{adapter}^\top,
\alpha QK_{task}^\top
\right]
}{
\sqrt{d_h}
}
$$

其中：

$$
\alpha = \tanh(g)
$$

这说明 Pro 版本仍然是在 action decoder tokens 上做 bridge-attn，只是把 source-specific projection 写得更干净。

## 9. Gating 的真实含义

VLA-Adapter 的 gate 是 logit-level gate：

$$
L_t =
\alpha QK_t^\top
$$

其中：

$$
\alpha = \tanh(g)
$$

初始化时：

$$
g = 0,
\quad
\alpha = 0
$$

直观含义是：模型一开始不会强依赖 visual/task branch，训练过程中再逐渐打开这条分支。

但是要注意一个细节：这个 gate 不是 hard mask。因为当 gate 为 0 时，visual/task branch 的 logits 是 0，而不是负无穷。

因此：

$$
\alpha = 0
\nRightarrow
\operatorname{softmax}(L_t) = 0
$$

它只是把这一组 logits 压到零附近，降低该 branch 的竞争强度，但 value branch 仍然在 softmax 里参与竞争。

如果我们想让某个 source 在初始化时真正接近关闭，更清楚的设计是 output-level gate：

$$
X' =
X
+ O_{self}
+ \beta_t O_t
$$

其中：

$$
\beta_t = \tanh(g_t),
\quad
g_t \leftarrow 0
$$

这样当 gate 为 0 时，整个 source 的输出贡献就是 0。

## 10. Context 的具体语义

VLA-Adapter 的 bridge-attn 里，context 不是一个普通拼接后的大 token 序列，而是有明确来源的分支。

### 10.1 Action-Query Hidden States

$$
A^\ell \in \mathbb{R}^{N_a \times D}
$$

这 64 个 tokens 来自 LLM 里的 action query positions。它们已经通过 LLM self-attention 看过 visual patches 和 language prompt。

它们的作用更像：

`LLM-conditioned action intent carriers`

也就是带有任务语义和动作位置语义的 latent tokens。

### 10.2 Visual/Task Hidden States

$$
T^\ell \in \mathbb{R}^{N_v \times D}
$$

这部分来自 LLM hidden states 的 image patch positions。它保留了更细粒度的视觉 grounding 信息。

### 10.3 Proprio Token

$$
p \in \mathbb{R}^{1 \times D}
$$

它提供当前机器人状态，例如夹爪、末端位置、关节状态等。

### 10.4 Action Decoder Tokens

$$
X \in \mathbb{R}^{C \times D}
$$

这是 action head 里真正被更新并最终预测 continuous actions 的 tokens。

在 LIBERO 中：

$$
C = 8
$$

在我们的 PrismVLA 设计中，如果 horizon 为 32，则更自然的是：

$$
C = H = 32
$$

也就是每个 noisy action token 对应一个未来 action step。

## 11. 和我们当前 BridgeAdapter 的差异

我们当前代码里的 `BridgeAdapter` 是另一种结构：先用 learned bridge tokens 读取 raw VLM features、action queries、state、plan、short memory。

然后 action head 再读这些 bridge tokens：

noisy action tokens 再通过 cross-attention 读取 bridge tokens。

也就是一个两跳结构：

$$
C_{ctx}
\rightarrow
B_{bridge}
\rightarrow
X_{action}
$$

当前默认配置里：

$$
N_{bridge} = 16
$$

$$
N_{action\_query} = 64
$$

这和 VLA-Adapter 不一样。VLA-Adapter 没有这 16 个中间 bridge action tokens。它是让 action decoder tokens 直接读取 VLM hidden states 后预测 actions。

所以如果我们说“借鉴 VLA-Adapter 的 bridge-attn”，核心不应该是照搬 16 个 bridge tokens，而应该是借鉴：

1. action/noisy action tokens 作为 query。
2. context 按来源拆分。
3. 对高风险 context source 使用 zero-init gate。
4. action decoder 内部直接读取 VLM/memory/plan/state context。

## 12. 对 PrismVLA 的建议结构

结合我们当前设计，推荐把 bridge-attn 改成直接作用在 32 个 noisy action tokens 上。

设：

$$
H = 32
$$

$$
X_\tau \in \mathbb{R}^{H \times D}
$$

其中 `X_tau` 是 flow matching 当前噪声时刻的 action tokens。

context sources 定义为：

$$
F_k \in \mathbb{R}^{N_f \times D}
$$

$$
S_k \in \mathbb{R}^{N_s \times D}
$$

$$
P_k^{slot} \in \mathbb{R}^{N_p \times D}
$$

$$
e^s_k \in \mathbb{R}^{1 \times D}
$$

其中：

- `F_k` 是当前 VL visual tokens 或 pooled/current VL features。
- `S_k` 是 short memory visual tokens。
- `P_k_slot` 是 planner token 经过 virtual slot expansion 后的 plan slots。
- `e_s_k` 是 robot state embedding。

推荐 action block 写成 source-wise bridge-attn：

$$
O_F =
\operatorname{Attn}
\left(
X_\tau,
F_k,
F_k
\right)
$$

$$
O_S =
\operatorname{Attn}
\left(
X_\tau,
S_k,
S_k
\right)
$$

$$
O_P =
\operatorname{Attn}
\left(
X_\tau,
P_k^{slot},
P_k^{slot}
\right)
$$

$$
O_s =
\operatorname{Attn}
\left(
X_\tau,
e^s_k,
e^s_k
\right)
$$

然后用 source-wise output gates：

$$
X' =
X_\tau
+ O_F
+ \beta_S O_S
+ \beta_P O_P
+ \beta_s O_s
$$

其中：

$$
\beta_S = \tanh(g_S),
\quad
\beta_P = \tanh(g_P),
\quad
\beta_s = \tanh(g_s)
$$

建议初始化：

$$
g_S \leftarrow 0,
\quad
g_P \leftarrow 0,
\quad
g_s \leftarrow 0
$$

这样当前 VL branch 可以作为主干直接使用，short memory、plan slots、state branch 通过 zero-init gates 渐进进入。

如果想让 state 从一开始就稳定参与，也可以把 state branch 不设 gate：

$$
X' =
X_\tau
+ O_F
+ \beta_S O_S
+ \beta_P O_P
+ O_s
$$

## 13. Plan Token Expansion 的位置

你现在提出的方案是先输出一个 plan token，再扩展成 4 到 16 个 virtual slots。这个和 VLA-Adapter 的 64 个 action query 有相似动机：单 token 信息密度太高，action decoder tokens 读取时容易瓶颈化。

设 planner 输出：

$$
p_k \in \mathbb{R}^{D}
$$

用 learned slot queries 扩展：

$$
Q^{slot} \in \mathbb{R}^{N_p \times D}
$$

可以用一个小 cross-attn 或 FiLM-MLP 得到：

$$
P_k^{slot}
=
\operatorname{SlotExpand}
\left(
p_k,
Q^{slot}
\right)
$$

最简单稳定的形式是：

$$
P_k^{slot}[i]
=
\operatorname{MLP}
\left(
\left[
p_k,
q_i^{slot}
\right]
\right)
$$

其中：

$$
i = 1, \ldots, N_p
$$

推荐先设：

$$
N_p = 8
$$

原因是：

- 4 个 slots 可能仍然偏窄。
- 16 个 slots 会增加 action block 的 context 长度。
- 8 个 slots 对 32-step horizon 是一个比较均衡的容量。

## 14. Short Memory Token 数量的建议

当前 short memory 如果来自两个历史时刻：

$$
t_k - R
$$

$$
t_k - \frac{R}{2}
$$

并且每个时刻压缩到 `N_m` 个 tokens，则：

$$
N_s = 2N_m
$$

如果每帧压缩到 16 个 tokens：

$$
N_s = 32
$$

如果每帧压缩到 8 个 tokens：

$$
N_s = 16
$$

对于 action horizon:

$$
H = 32
$$

我建议 short memory context 先控制在：

$$
N_s \in [16, 32]
$$

再加上 plan slots：

$$
N_p = 8
$$

整体 context 长度会比较可控。

## 15. 最终理解

VLA-Adapter 的 bridge-adapter 可以理解成：将 VLM hidden states 以 adapter context 的形式接入 action head。

它的 bridge-attn 可以理解成：

action decoder tokens 对 VLM-derived context 做 gated cross-attention。

对我们最重要的借鉴是：

1. 不要把 context 简单 concat 后无差别喂给 action head。
2. context 应该按来源拆分。
3. action/noisy action tokens 应该直接作为 query。
4. plan、short memory 这类新增 source 应该用 zero-init gate 渐进打开。
5. 16 个中间 bridge tokens 不是 VLA-Adapter 的必要结构，反而可能变成不必要的信息瓶颈。

因此，PrismVLA 后续 bridge-attn 更推荐的主线是让 32 个 noisy action tokens 直接读取 current VL、short memory、expanded plan slots 和 state。

而不是：

$$
C_{ctx}
\rightarrow
B_{bridge}
\rightarrow
X_{action}
$$
