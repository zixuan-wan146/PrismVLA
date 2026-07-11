# StarVLA LIBERO Training Recipe 调研

调研对象：`reference-repo/starVLA`  
来源仓库：`https://github.com/starVLA/starVLA.git`  
本地快照：`starVLA_dev`, commit `6dc01d0`  
调研日期：2026-06-28

本文只记录 StarVLA 当前仓库里可追溯到源码、配置和脚本的 LIBERO 训练流程。需要特别注意：README 的公开结果、YAML 默认值、训练脚本里的 CLI 覆写并不完全一致，复现时应优先以实际启动命令为准。

## 1. 结论概览

StarVLA 的 LIBERO 训练不是每个 task 单独训练一个 policy，而是把 LIBERO 的四个 suite 先转换成 LeRobot 格式，再通过 `data_mix` 组成 mixture 训练。

官方 README 描述的主要结果是一个 policy 覆盖四个 suite：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

当前仓库中主 LIBERO recipe 的数据混合名是：

```yaml
data_mix: libero_all
```

`libero_all` 等权混合四个 LeRobot 数据集：

```yaml
libero_all:
  - libero_object_no_noops_1.0.0_lerobot
  - libero_goal_no_noops_1.0.0_lerobot
  - libero_spatial_no_noops_1.0.0_lerobot
  - libero_10_no_noops_1.0.0_lerobot
```

单 suite 训练在设计上就是把 `data_mix` 换成只包含一个数据集的 mixture。仓库当前只显式注册了 `libero_goal` 单 suite；Gemma4/MiniCPM 示例文档提到 `libero_spatial/libero_object/libero_10`，但当前 `DATASET_NAMED_MIXTURES` 没有看到这三个单 suite 名称。因此如果要严格单独训练 spatial/object/10，需要先补齐 mixture 注册。

## 2. 关键入口文件

| 类型 | 路径 | 作用 |
|---|---|---|
| README | `reference-repo/starVLA/examples/LIBERO/README.md` | LIBERO 数据下载、训练和评估说明 |
| 训练脚本 | `reference-repo/starVLA/examples/LIBERO/train_files/run_libero_train.sh` | 官方 LIBERO 训练启动模板 |
| 训练 YAML | `reference-repo/starVLA/examples/LIBERO/train_files/starvla_cotrain_libero.yaml` | 默认模型、数据、优化器参数 |
| 数据 registry | `reference-repo/starVLA/examples/LIBERO/train_files/data_registry/data_config.py` | LIBERO modality、normalization、mixture |
| 数据准备 | `reference-repo/starVLA/examples/LIBERO/data_preparation.sh` | 下载四个 suite，并写入 `modality.json` |
| trainer | `reference-repo/starVLA/starVLA/training/train_starvla.py` | VLA action loss 训练循环 |
| dataloader | `reference-repo/starVLA/starVLA/dataloader/lerobot_datasets.py` | 根据 `data_mix` 构建 LeRobot mixture dataset |
| DeepSpeed | `reference-repo/starVLA/starVLA/config/deepseeds/ds_config.yaml` | bf16、ZeRO-2、真实 gradient accumulation |

## 3. 数据准备流程

### 3.1 下载的数据集

`examples/LIBERO/data_preparation.sh` 下载四个 Hugging Face dataset：

```text
IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
```

默认组织方式：

```text
playground/Datasets/LEROBOT_LIBERO_DATA/
  libero_spatial_no_noops_1.0.0_lerobot/
  libero_object_no_noops_1.0.0_lerobot/
  libero_goal_no_noops_1.0.0_lerobot/
  libero_10_no_noops_1.0.0_lerobot/
```

脚本还会下载 VLM cotrain 数据：

```text
StarVLA/LLaVA-OneVision-COCO
```

但主训练脚本调用的是 `train_starvla.py`，只走 VLA dataloader，不走 `train_starvla_cotrain.py`。因此当前 LIBERO 主 recipe 中，`datasets.vlm_data` 存在于 YAML 里，但训练 loop 实际只消费 `datasets.vla_data`。

### 3.2 modality.json

数据准备脚本会把 `examples/LIBERO/train_files/modality.json` 复制到每个 suite 的 `meta/` 目录。映射关系如下：

```yaml
video:
  primary_image: observation.images.image
  wrist_image: observation.images.wrist_image

annotation:
  human.action.task_description: task_index

state:
  x: 0
  y: 1
  z: 2
  roll: 3
  pitch: 4
  yaw: 5
  pad: 6
  gripper: 7

action:
  x: 0
  y: 1
  z: 2
  roll: 3
  pitch: 4
  yaw: 5
  gripper: 6
```

这里 action 是 7 维，state 是 8 维。训练模型配置里 `state_dim` 常设为 7，因为主路径默认没有 `include_state: true`，state 不一定被打包进模型输入。

## 4. LIBERO 数据配置

`Libero4in1DataConfig` 定义四套 suite 共用的数据结构：

```yaml
robot_type: libero_franka
embodiment_tag: FRANKA

video_keys:
  - video.primary_image
  - video.wrist_image

state_keys:
  - state.x
  - state.y
  - state.z
  - state.roll
  - state.pitch
  - state.yaw
  - state.pad
  - state.gripper

action_keys:
  - action.x
  - action.y
  - action.z
  - action.roll
  - action.pitch
  - action.yaw
  - action.gripper

language_keys:
  - annotation.human.action.task_description

observation_indices: [0]
state_indices: [0]
action_indices: [0, 1, 2, 3, 4, 5, 6, 7]
```

训练样本最终被打包成：

```python
{
    "image": [primary_pil, wrist_pil],
    "lang": instruction,
    "action": np.ndarray(shape=[8, 7], dtype=np.float16),
    "robot_tag": "franka",
    # "state": optional, only when include_state is true
}
```

图像在 `_pack_sample()` 中统一 resize 到 `224x224`。

### 4.1 action normalization

LIBERO recipe 对 action 的连续维度做 `min_max` normalization：

```yaml
normalization_modes:
  action.x: min_max
  action.y: min_max
  action.z: min_max
  action.roll: min_max
  action.pitch: min_max
  action.yaw: min_max
```

`action.gripper` 没有在这个 LIBERO transform 中显式设置为 `binary`。保存 merged statistics 时，mask 会根据 normalization mode 判断；没有 binary 标注的维度会被视作连续维度。评估脚本里又会对 gripper 做二值化：

```text
normalized_actions[:, 6] < 0.5 -> 0
else -> 1
```

如果我们迁移这个 recipe，gripper normalization 和 unnormalization 要单独核对，不能默认认为它在训练侧已经 binary-normalized。

## 5. Suite Mixture 设置

当前可追溯到 registry 的 LIBERO mixture：

| data_mix | 数据集 | 用途 |
|---|---|---|
| `libero_all` | object + goal + spatial + libero_10，权重均为 1.0 | 官方四套件联合训练 |
| `libero_goal` | goal，权重 1.0 | 单 suite goal 训练 |
| `multi_robot` | 指向 `LEROBOT_LIBERO_DATA/libero_10...` | 看起来是旧/临时配置，不建议直接复用 |

Gemma4/MiniCPM 脚本注释里写了：

```text
DATA_MIX = libero_all / libero_spatial / libero_object / libero_goal / libero_10
```

但在当前快照中，`libero_spatial`、`libero_object`、`libero_10` 这三个单 suite mixture 没有在 `DATASET_NAMED_MIXTURES` 中注册。要按 suite 单独训练，建议补成：

```python
DATASET_NAMED_MIXTURES.update({
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
})
```

## 6. 主训练 YAML 参数

`starvla_cotrain_libero.yaml` 的默认参数如下：

```yaml
run_id: starvla
run_root_dir: playground/Checkpoints
seed: 42
is_debug: false
version_id: "0.21"

framework:
  name: QwenGR00T
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 8

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA
    data_mix: libero_all
    action_type: delta_qpos
    sequential_step_sampling: false
    per_device_batch_size: 16
    load_all_data_for_training: true
    video_backend: torchvision_av

trainer:
  max_train_steps: 100000
  num_warmup_steps: 5000
  save_interval: 5000
  eval_interval: 100
  learning_rate:
    base: 2.5e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: qwen_vl_interface
  gradient_clipping: 1.0
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08
```

注意：`trainer.gradient_accumulation_steps` 在当前 `train_starvla.py` 中不是实际 DeepSpeed accumulation 的来源。真实 accumulation 来自 Accelerate/DeepSpeed 配置。

## 7. 主训练脚本实际覆写

`run_libero_train.sh` 会覆盖 YAML 的关键字段：

```bash
Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=1229_libero4in1_qwen3oft
```

实际启动命令的主要覆写：

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
```

因此主脚本模板的实际 recipe 是：

```yaml
framework.name: QwenPI
framework.qwenvl.base_vlm: playground/Pretrained_models/Qwen3.5-0.8B
datasets.vla_data.data_mix: libero_all
datasets.vla_data.per_device_batch_size: 16
trainer.freeze_modules: ""
trainer.max_train_steps: 80000
trainer.save_interval: 10000
trainer.logging_frequency: 100
trainer.eval_interval: 100
```

脚本里还有一处疑似错误：

```bash
--trainer.vla_data.video_backend torchvision_av
```

dataloader 读取的是：

```yaml
datasets.vla_data.video_backend
```

所以这条 CLI 覆写大概率不会生效。当前 YAML 已经有 `datasets.vla_data.video_backend: torchvision_av`，因此通常不影响主 recipe。

## 8. Trainer 训练流程

`train_starvla.py` 的训练流程：

1. 读取 YAML。
2. 把 CLI dotlist merge 到 YAML。
3. `build_framework(cfg)` 根据 `framework.name` 构建模型。
4. `build_dataloader(cfg, dataset_py=cfg.datasets.vla_data.dataset_py)` 构建 VLA dataloader。
5. 按 `trainer.learning_rate` 分组构建 AdamW。
6. 根据 `trainer.freeze_modules` 冻结模块。
7. 用 Accelerate/DeepSpeed prepare model、optimizer、dataloader。
8. 每 step 取一个 VLA batch，调用：

```python
output_dict = self.model.forward(batch_vla)
action_loss = output_dict["action_loss"]
```

9. 只优化 action loss。
10. 每 `eval_interval` 调一次 `eval_action_model()`。
11. 每 `save_interval` 保存 checkpoint。

这里的 `eval_action_model()` 不是 LIBERO rollout 成功率评估，而是在当前训练 batch 上调用 `predict_action()` 后计算 normalized action 的欧氏距离/MSE。真正 LIBERO success rate 评估在 `examples/LIBERO/eval_files/` 里单独跑。

## 9. 分布式和优化器参数

主训练使用：

```yaml
distributed_type: DEEPSPEED
zero_stage: 2
bf16: true
fp16: false
gradient_accumulation_steps: 1
gradient_clipping: 1.0
```

对应文件：

```text
starVLA/config/deepseeds/deepspeed_zero2.yaml
starVLA/config/deepseeds/ds_config.yaml
```

关键点：

```yaml
train_micro_batch_size_per_gpu: auto
train_batch_size: auto
gradient_accumulation_steps: 1
zero_optimization:
  stage: 2
  cpu_offload: false
```

如果用 8 张 GPU 且 `per_device_batch_size=16`：

```text
effective batch size = 16 * 8 * 1 = 128
```

这也解释了 MiniCPM 脚本注释中的说法：主 upstream 设置实际 GA 是 1，YAML 里的 `trainer.gradient_accumulation_steps: 4` 只是配置字段，不会自动改变 DeepSpeed accumulation。

Gemma4 示例用 `_make_accelerate_config.py` 动态生成 DeepSpeed config，显式把 `gradient_accumulation_steps` 写进 DeepSpeed JSON。这个做法比直接改 `trainer.gradient_accumulation_steps` 更可靠。

## 10. 动作头/框架变体

StarVLA 的 LIBERO README 表格报告了多种动作头：

```text
StarVLA-FAST
StarVLA-OFT
StarVLA-π
StarVLA-GR00T
```

当前主脚本覆写为 `QwenPI`，YAML 默认是 `QwenGR00T`。几个常见框架的差异如下。

| framework.name | 训练方式 | action horizon 默认 | 主要特点 |
|---|---:|---:|---|
| `QwenGR00T` | Flow-matching DiT action head | 8 | 使用最后一层 VLM hidden state，经 DiT action head 预测连续动作 |
| `QwenPI` | Layer-wise flow-matching / cross-DiT | 16 | 使用多层 VLM hidden states；但 LIBERO YAML 会覆写为 8 |
| `QwenOFT` | Action token + MLP regression | 8 | 在 prompt 里插入 action special token，用 MLP 做连续动作回归 |
| `QwenFast` | FAST tokenizer 离散动作 token | 15 | 把连续动作编码成 FAST token，走 autoregressive token prediction |

### 10.1 QwenGR00T

默认参数：

```yaml
framework:
  name: QwenGR00T
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
  action_model:
    action_model_type: DiT-B
    action_hidden_dim: 1024
    hidden_size: 1024
    action_dim: 7
    state_dim: 7
    action_horizon: 8
    repeated_diffusion_steps: 8
    noise_beta_alpha: 1.5
    noise_beta_beta: 1.0
    noise_s: 0.999
    num_timestep_buckets: 1000
    num_inference_timesteps: 4
    num_target_vision_tokens: 32
    diffusion_model_cfg:
      cross_attention_dim: 2048
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: ada_norm
      num_layers: 16
      output_dim: 1024
```

构建时会把 `diffusion_model_cfg.cross_attention_dim` 对齐到实际 VLM hidden size。

### 10.2 QwenPI

默认参数：

```yaml
framework:
  name: QwenPI
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
    num_vl_layers: 36
  action_model:
    action_model_type: LayerwiseFM
    action_dim: 7
    state_dim: 7
    action_horizon: 16
    repeated_diffusion_steps: 2
    num_inference_timesteps: 4
    num_target_vision_tokens: 32
    noise_beta_alpha: 1.5
    noise_beta_beta: 1.0
    noise_s: 0.999
    num_timestep_buckets: 1000
    diffusion_model_cfg:
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: ada_norm
      attention_head_dim: 64
```

虽然 QwenPI 默认 `action_horizon=16`，LIBERO YAML 中 `framework.action_model.action_horizon=8` 会覆盖它。主脚本使用 QwenPI 时，最终 horizon 应按 YAML 的 8 处理。

QwenPI forward 中还把 `repeated_diffusion_steps` 强制设为 2：

```python
repeated_diffusion_steps = 2
```

因此即使配置里改这个字段，也要检查源码是否仍有硬覆写。

### 10.3 QwenOFT

默认参数：

```yaml
framework:
  name: QwenOFT
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: MLP
    action_dim: 7
    action_hidden_dim: 2560
    future_action_window_size: 8
    past_action_window_size: 0
```

OFT 依赖带 action special tokens 的 VLM checkpoint。训练时在 instruction 后拼接 action token prompt，然后对 action token hidden states 做 MLP 回归，loss 是 L1。

### 10.4 QwenFast

默认参数：

```yaml
framework:
  name: QwenFast
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: FAST
    action_dim: 7
    future_action_window_size: 15
    past_action_window_size: 0
```

FAST 把 raw action 映射成 FAST tokens，再映射到 VLM action tokens，训练目标是 action token 的自回归预测。

## 11. 各 LIBERO suite 的训练设置

四个 suite 共享同一套 modality、action space、image size、horizon 和 optimizer 参数。差别主要是 `data_mix` 指向的数据集不同。

| suite | LeRobot 数据集名 | 当前是否注册单 suite mix | 推荐 data_mix |
|---|---|---:|---|
| LIBERO-Spatial | `libero_spatial_no_noops_1.0.0_lerobot` | 否 | `libero_spatial`，需补注册 |
| LIBERO-Object | `libero_object_no_noops_1.0.0_lerobot` | 否 | `libero_object`，需补注册 |
| LIBERO-Goal | `libero_goal_no_noops_1.0.0_lerobot` | 是 | `libero_goal` |
| LIBERO-10 | `libero_10_no_noops_1.0.0_lerobot` | 否 | `libero_10`，需补注册 |
| 四套件联合 | 上述四个 | 是 | `libero_all` |

### 11.1 四套件联合训练

这是 README 结果表对应的主路线。

```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```

实际关键参数：

```yaml
data_mix: libero_all
per_device_batch_size: 16
num_processes: nvidia-smi 检测到的 GPU 数
deepspeed_zero_stage: 2
gradient_accumulation_steps: 1
effective_batch_size_on_8gpu: 128
max_train_steps: 80000
save_interval: 10000
eval_interval: 100
```

README 结果表里 StarVLA 系列报告的是 30K steps、约 9.54 epochs。当前脚本模板是 80K，YAML 是 100K。复现实验时需要先确认要复现的是 README 表格结果还是当前工程模板。

### 11.2 LIBERO-Goal 单 suite 训练

当前仓库可直接用：

```bash
data_mix=libero_goal
```

示例：

```bash
DATA_MIX=libero_goal bash examples/LIBERO/train_files/run_libero_train.sh
```

但 `run_libero_train.sh` 现在没有读取环境变量 `DATA_MIX`，而是写死：

```bash
data_mix=libero_all
```

所以需要手动改脚本变量，或新建一个外部 wrapper 覆写 `--datasets.vla_data.data_mix libero_goal`。

建议命令模板：

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3.5-0.8B \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir playground/Checkpoints \
  --run_id libero_goal_qwenpi
```

### 11.3 LIBERO-Spatial / Object / 10 单 suite 训练

当前需要先补 registry：

```python
"libero_spatial": [
    ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
],
"libero_object": [
    ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
],
"libero_10": [
    ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
],
```

然后分别把 CLI 的 `--datasets.vla_data.data_mix` 改成：

```text
libero_spatial
libero_object
libero_10
```

其它训练参数和 `libero_goal` 相同，除非想按 suite 大小调整 steps。

StarVLA 没有在当前主 LIBERO recipe 中给出每个单 suite 专属训练 step 数。Gemma4 README 的示例只写了 quick ablation：

```bash
DATA_MIX=libero_spatial MAX_STEPS=50000 sbatch examples/Gemma4/submit_hpc3_libero.sh
```

这可以作为单 suite ablation 的参考，不应直接当作官方四套件结果的标准 recipe。

## 12. Gemma4 和 MiniCPM 的 LIBERO 变体

StarVLA 还提供了两个 LIBERO 训练变体，主要用于换 VLM backbone。

### 12.1 Gemma4

入口：

```text
examples/Gemma4/submit_hpc3_libero.sh
```

默认参数：

```yaml
framework: Gemma4PI
base_vlm: google/gemma-4-E2B-it
data_mix: libero_all
max_steps: 100000
per_device_bs: 2
grad_accum: 8
num_gpus: 8
effective_batch_size: 128
attn_impl: sdpa
zero_stage: 2
```

这个脚本会调用 `_make_accelerate_config.py` 动态生成 DeepSpeed 配置，把真实 `gradient_accumulation_steps` 写进 DeepSpeed JSON。

README 报告：

```text
Gemma4-E2B + PI head
40K optimizer steps
effective batch size = 128
8 x H100
average success rate = 96.0%
```

### 12.2 MiniCPM

入口：

```text
examples/MiniCPM/submit_hpc3_libero.sh
```

默认参数：

```yaml
framework: MiniCPMPI
base_vlm: openbmb/MiniCPM-V-4.6
data_mix: libero_all
max_steps: 80000
per_device_bs: 16
grad_accum: 1
num_gpus: 8
effective_batch_size: 128
attn_impl: sdpa
freeze_modules: ""
```

MiniCPM 脚本明确说明它对齐 upstream 主 LIBERO 脚本：per-device batch 16，GA=1，8 GPU 总 batch 128。

## 13. 评估流程和各 suite 参数

训练过程中的 `eval_interval` 不是仿真评估。真正 LIBERO evaluation 分两端：

1. StarVLA 环境启动 policy server。
2. LIBERO 环境启动 simulation client。

Server：

```bash
bash examples/LIBERO/eval_files/run_policy_server.sh
```

Client：

```bash
LIBERO_HOME=/path/to/LIBERO \
TASK_SUITE_NAME=libero_goal \
NUM_TRIALS_PER_TASK=50 \
bash examples/LIBERO/eval_files/eval_libero.sh
```

各 suite 的 evaluation max steps：

| suite | max_steps | 注释 |
|---|---:|---|
| `libero_spatial` | 220 | longest training demo 193 |
| `libero_object` | 280 | longest training demo 254 |
| `libero_goal` | 300 | longest training demo 270 |
| `libero_10` | 520 | longest training demo 505 |
| `libero_90` | 400 | longest training demo 373 |

README 的标准评估是每个 suite 10 tasks，每个 task 50 episodes，即每个 suite 500 trials。

## 14. 迁移到我们项目时的注意点

1. StarVLA 的四套件联合训练 horizon 是 8，而我们当前 LIBERO direct bridge Stage 1 是 H=32/R=16；不能直接照搬 action horizon。
2. StarVLA 使用 frame-level random mixture sampling；我们当前 Stage 1 使用 episode-level fixed replan nodes，progress state 需要按 episode 顺序递推。
3. StarVLA 的训练 action target 是 normalized action chunk；我们要确认自己 cache/loader 中 action normalization 和 gripper 处理是否一致。
4. StarVLA 训练时图像 resize 到 224x224，VLM 现跑；我们 Stage 1 主要走 token cache，不加载 VLM。
5. StarVLA 的 `trainer.gradient_accumulation_steps` 容易误导，真实 GA 要看 DeepSpeed config。
6. 如果要做 per-suite ablation，先补齐 `libero_spatial/libero_object/libero_10` mixture 注册，不要只改 CLI。
7. README 结果表、YAML 和脚本存在 step/backbone/head 不一致，需要在实验记录里明确写出最终 merged config。

## 15. 推荐复现实验矩阵

如果我们只是研究 StarVLA recipe，不建议先跑四套件大训练。更稳的顺序：

| 阶段 | data_mix | max_steps | 目的 |
|---|---|---:|---|
| smoke | `libero_goal` | 10-100 | 验证 dataloader、VLM、action head、checkpoint 保存 |
| 单 suite ablation | `libero_goal` | 50000 或 80000 | 对齐可注册单 suite |
| 补 registry 后单 suite | `libero_spatial/object/10` | 50000 | 检查 suite 差异 |
| 四套件联合 | `libero_all` | 30000 | 对齐 README 表格步数 |
| 四套件工程模板 | `libero_all` | 80000 | 对齐当前主训练脚本 |

每次实验必须保存：

```text
config.full.yaml
config.yaml
dataset_statistics.json
run_libero_train.sh snapshot
checkpoint path
eval result json/video path
```

StarVLA trainer 会在 run dir 下保存 full/accessed config 和 dataset statistics，这是后续复盘 recipe 的关键文件。
