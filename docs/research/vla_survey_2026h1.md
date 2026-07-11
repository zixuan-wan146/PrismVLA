# 2026 H1 VLA / Embodied Training Recipe Survey

查询日期：2026-06-28  
时间窗口：2025-12-28 到 2026-06-28

本文只把近半年内仍能找到公开仓库、训练入口或配置接口的 VLA / 具身智能项目放进主样本。近半年论文如果没有确认到公开训练 recipe，只放在“相关但不纳入主表”里。OpenVLA-OFT、openpi 等更早项目只作为背景对照，不作为近半年样本。

## 1. 结论先行

近半年的开源 VLA / 具身仓库在 train recipe 上有几个共同趋势：

1. Recipe 不再只是一个 shell 命令，而是把模型、数据、训练器、输出和评测拆成显式配置。
2. 数据 recipe 会描述 modality、action/state 维度、normalization、dataset mixture，而不是只写一个 data path。
3. 模型 recipe 会明确 base model / checkpoint、LoRA 或可训练模块、是否冻结视觉/语言主干、action head 或 diffusion head 的训练开关。
4. Trainer recipe 会外置 batch size、gradient accumulation、learning rate、warmup、max steps、save/eval interval、precision、resume/checkpoint 策略。
5. Eval/serve recipe 也开始配置化，包括 checkpoint path、inference server、denoising steps、并发 client、episode 数、task offset、视频输出。
6. 对我们当前 Stage1 来说，最容易误导人的不是 learning rate 本身，而是 batch 的语义和 cache 训练边界。如果 `batch_size=1` 实际是一条 episode，就必须在文档和 config 中直接写明。

对 PrismVLA 的直接建议：

1. 保留当前 `episode-level fixed-replan-node training` 口径，并把它写进训练 recipe 的结构化字段。
2. 把 train/eval/server 共同需要的路径、VLM 本地加载、parallel clients、episode count 全部放进 config/profile，不放进脚本逻辑。
3. 下一轮训练不应只按 train loss 选模型，至少要同时保留固定小评测的 success、视频、action diagnostics 和 best loss。
4. 如果后续进入 Stage2 或 unfreeze VLM，需要新增独立 recipe，不能复用 Stage1 cache recipe 的文档口径。

## 2. 纳入标准

主样本必须同时满足：

1. 时间相关性：项目、论文、版本或主要更新发生在 2025-12-28 到 2026-06-28。
2. 开源可查：存在公开 repo、官方文档或训练入口。
3. Recipe 可见：能看到训练脚本、配置字段、CLI 参数、数据配置或 fine-tuning 文档。

不满足第 3 点的近半年论文不作为 recipe 样本。原因很简单：没有公开训练入口时，无法判断它们是怎么指定 batch、scheduler、dataset mixture、action normalization 和 checkpoint policy 的。

## 3. 主样本对比

| 项目 | 近半年依据 | Recipe 入口 | 配置风格 | 对我们的启发 |
| --- | --- | --- | --- | --- |
| NVIDIA Isaac-GR00T N1.7 / N1.5 | README 标注 2026-06-13 N1.7、2026-04-29 N1.5 | `scripts/gr00t_finetune.py`、`gr00t/configs/finetune_config.py` | dataclass config + CLI | 模型、数据、trainer 分层很清楚，适合我们学习 |
| VLA-GSE | arXiv 2026，仓库提供 pretrain / finetune | `vla-scripts/pretrain.py`、`vla-scripts/finetune.py` | OpenVLA/OXE 风格脚本 + Python dataset mixture | 显式写 dataset mixture、LoRA、batch、grad accum、lr |
| FineVLA | 2026-05 项目，FineVLA-Data + FineVLA-Policy | `FineVLA-Policy/scripts/pretrain.sh` | stage config name + shell 参数 | 多阶段 recipe 命名清楚，但脚本中硬编码路径的风险要避免 |
| LeRobot current docs | 2026 仍活跃维护，ICLR 2026 页面和当前文档包含训练入口 | `lerobot-train`、policy registry、dataset repo id | registry + CLI override | 适合参考 dataset / policy / output / wandb 的统一入口 |

## 4. Isaac-GR00T

Source:

- https://github.com/NVIDIA/Isaac-GR00T
- https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/gr00t/configs/finetune_config.py

GR00T 的 recipe 结构很清楚，官方仓库把 fine-tuning 作为一条标准路径：用户给出 dataset path、GPU 数、output dir、max steps、data config 和 video backend，然后进入训练脚本。

更重要的是它的配置文件不是把所有参数塞进脚本，而是拆成几个 dataclass：

1. `GR00TModelConfig`
2. `GR00TDataConfig`
3. `GR00TTrainerConfig`

其中 data config 会管理：

1. `data_config`
2. `embodiment_tag`
3. `video_backend`
4. `modality_config`
5. `modality_transform`
6. `max_seq_length`
7. `max_state_dim`
8. `max_action_dim`

trainer config 会管理：

1. `batch_size`
2. `num_gpus`
3. `max_steps`
4. `learning_rate`
5. `weight_decay`
6. `warmup_ratio`
7. `lora_rank`
8. `save_steps`
9. `dataloader_num_workers`
10. `resume`
11. `tune_llm`
12. `tune_visual`
13. `tune_projector`
14. `tune_diffusion_model`

这个设计的关键点是：数据形态、模型可训练部分、训练器超参是三个独立层次。训练命令只是选择这些配置，不承担业务逻辑。

对我们的影响：

1. `Stage1` 应该有独立的 model/data/trainer/eval config 分层。
2. `load_vlm=false`、`use_token_cache=true`、`vlm_local_files_only=true` 这类边界条件必须是 config 字段。
3. `action_dim`、`state_dim`、`horizon`、`replan_interval`、`progress_state` 的语义应该属于 data/model contract，不应该散落在脚本文档里。
4. 后续如果 unfreeze visual / projector / diffusion/action head，需要像 GR00T 一样显式声明 trainable modules。

## 5. VLA-GSE

Source:

- https://github.com/YuhuaJiang2002/VLA-GSE
- https://raw.githubusercontent.com/YuhuaJiang2002/VLA-GSE/main/README.md

VLA-GSE 的训练入口延续 OpenVLA 系列风格，README 直接给出 pretrain 和 finetune 的启动方式。它的 recipe 通过 CLI 参数加 Python dataset mixture 来指定。

可见的 recipe 字段包括：

1. `vla.type`
2. `data_root_dir`
3. `run_root_dir`
4. `run_id`
5. `image_aug`
6. `vla_path`
7. `dataset_name`
8. `adapter_tmp_dir`
9. `lora_rank`
10. `batch_size`
11. `grad_accumulation_steps`
12. `learning_rate`
13. `save_steps`
14. W&B project/entity

它还在代码里注册数据 mixture，例如为某个 dataset 名称指定权重。这个做法有一个优点：训练命令里的 `dataset_name` 不是裸字符串，而是能落到明确的 mixture 定义。

风险是，脚本入口参数很多，如果没有对应的 YAML 或 manifest，复现实验时很容易丢掉某个关键参数。

对我们的影响：

1. 可以保留 CLI override，但必须在 run manifest 中落盘最终展开后的配置。
2. cache manifest 和 dataset mixture 应该是结构化字段，不要只依赖命令行字符串。
3. `batch_size=1` 必须额外声明 batch unit 是 episode 还是 transition。VLA-GSE 这种命令行字段无法自动表达这层语义。
4. `save_steps` 和 checkpoint policy 要分开。保存频率不等于 best checkpoint 选择逻辑。

## 6. FineVLA

Source:

- https://finevla.xlang.ai/
- https://github.com/xlang-ai/FineVLA
- https://raw.githubusercontent.com/xlang-ai/FineVLA/main/FineVLA-Policy/README.md

FineVLA 把数据构建和策略训练拆成 FineVLA-Data 与 FineVLA-Policy。Policy 侧使用脚本选择不同 stage 的训练配置，例如 OpenVLA SFT、DiT pretrain、latent action model pretrain、OpenX 相关 pretrain。

它的 recipe 思路是：用 stage config name 表达训练阶段，而不是只靠一个通用 train.py。这个对多阶段 VLA 很重要，因为 SFT、latent action、diffusion/action head、OpenX 混合数据的训练目标不一样。

可见的 stage 维度包括：

1. OpenVLA SFT
2. DiT pretrain
3. latent action model pretrain
4. OpenX latent action pretrain
5. OpenX latent DiT pretrain

需要注意的是，公开 README 里的部分示例更像开发脚本，存在环境变量、机器路径和第三方服务配置混在一起的问题。我们不能照抄这种写法，尤其不能把个人路径、token 或平台相关变量写进 repo 默认配置。

对我们的影响：

1. Stage1 cache training、Stage2 raw image training、eval server 需要拆成不同 recipe 名称。
2. recipe 名称应该描述训练目标，例如 `libero_stage1_episode_cache_w4`，而不是只描述数据集。
3. shell 脚本只能 parse profile 并调用包内逻辑，不能把核心训练决策写死在脚本里。
4. 文档需要明确“这个 recipe 训练什么、不训练什么”。例如当前 Stage1 不训练 VLM，不使用 raw image forward。

## 7. LeRobot Current Docs

Source:

- https://github.com/huggingface/lerobot
- https://huggingface.co/docs/lerobot/bring_your_own_policies
- https://huggingface.co/docs/lerobot/pi0fast

LeRobot 不是单一近半年 VLA 论文，而是当前开源机器人训练框架。它对 recipe 的价值在于入口统一：dataset、policy、output、job name、device、wandb 等都通过 `lerobot-train` 和 policy registry 管理。

可见 recipe 维度包括：

1. dataset repo id
2. policy path 或 policy type
3. output dir
4. job name
5. device
6. wandb enable
7. policy-specific config
8. evaluation / record workflow

它的优点是：训练、录制、评测、policy 加载走统一命令族，用户不需要每次猜哪个脚本是权威入口。

对我们的影响：

1. `scripts/train.py`、`scripts/eval.py`、`scripts/serve.py` 是权威入口，参数应该来自 config/profile。
2. eval profile 应该和 training profile 一样可复现，例如 `total_episodes`、`parallel_clients`、`episode_offset`、`task_offset`、`max_steps`、`checkpoint`。
3. 本地模型权重路径和 `local_files_only` 应该是 server recipe 的一部分，不能每次默认访问 Hugging Face。
4. 输出目录、视频目录、run manifest 应该有统一命名规则，避免 eval 结果散落。

## 8. 近半年相关但不纳入主表的工作

这些工作在时间上相关，但截至本次查询没有确认到足够公开的训练 recipe。它们可以作为方向参考，不能作为我们制定 recipe 的主要依据。

| 工作 | 时间相关性 | 当前处理 |
| --- | --- | --- |
| StaKe | arXiv 2026-06，关注 keyframe / key-state supervision | 有项目页和论文，但未确认公开训练 repo，因此不纳入主表 |
| PokeVLA | arXiv 2026-04，探索交互式 poking 对 VLA 的帮助 | 项目页显示 code 将公开，当前不等于可复现 recipe |
| EXPO-FT | arXiv 2026-05，关注 RL fine-tuning / exploration policy optimization | 论文提到 codebase，但未找到可引用训练仓库 |
| ReFineVLA / fewer layers fine-tuning | arXiv 2026-06，研究 VLA finetune 哪些层更有效 | 目前未确认官方训练 repo |

Source:

- https://arxiv.org/abs/2606.26801
- https://getterupper.github.io/PokeVLA
- https://arxiv.org/abs/2604.20834
- https://arxiv.org/abs/2605.25477
- https://arxiv.org/abs/2606.20246

## 9. 背景对照，不作为近半年主样本

OpenVLA-OFT 和 openpi 对 recipe 设计仍有参考价值，但它们的主要开源时间早于本次窗口，所以不能混进“近半年调研”的主结论。

OpenVLA-OFT 的典型风格：

1. 训练入口是 `vla-scripts/finetune.py`。
2. recipe 通过 CLI 指定 base VLA、dataset、LoRA rank、batch size、gradient accumulation、learning rate、image augmentation 和保存间隔。
3. 优点是直观，缺点是复杂实验容易变成长命令，不利于保留完整语义。

openpi 的典型风格：

1. 训练入口通过 config name 选择。
2. Python config registry 负责保存 policy、data、normalization 和 training 参数。
3. 优点是 recipe 可组合，缺点是用户必须明确知道最终展开配置。

对我们来说，这两个背景样本支持同一个结论：命令可以保留，但最终 recipe 必须能落到可审计的 config 和 manifest。

## 10. 对 PrismVLA 的具体建议

### 10.1 Stage1 训练 recipe 应该固定的字段

建议 Stage1 YAML 至少显式包含这些字段：

```yaml
recipe:
  name: libero_stage1_episode_cache_w4
  stage: stage1
  objective: flow_matching_action_head
  batch_unit: episode
  training_mode: episode_level_fixed_replan_node

model:
  load_vlm: false
  use_token_cache: true
  use_raw_images: false
  trainable_modules:
    - bridge
    - action_head
    - progress_conditioning
  frozen_modules:
    - vlm
  progress_state:
    source: frozen_cache_recurrent_state
    recurrence: sequential_within_episode

data:
  suite: libero_10
  cache_manifest: local_data/cache/libero_10/manifest.json
  horizon: 32
  stride: 16
  replan_interval: 16
  action_dim: 7
  state_dim: 8
  drop_tail_shorter_than_horizon: true
  loss_nodes: full_horizon_nodes

trainer:
  optimizer: adamw
  batch_size: 1
  batch_size_meaning: one_episode_per_optimizer_step
  max_steps: 5000
  warmup_steps: 500
  scheduler: cosine_decay
  min_lr_ratio: 0.1
  checkpoint_policy:
    save_latest: false
    save_best_by: train_loss
    keep_top_k: 1
```

这里最重要的是 `batch_size_meaning`。如果只写 `batch_size: 1`，读者会自然理解成一条 transition 或一个 frame，这和当前训练实际不一致。

### 10.2 Eval / serve recipe 应该固定的字段

建议 eval profile 至少显式包含：

```yaml
eval:
  suite: libero_spatial
  task_offset: 0
  task_limit: 1
  total_episodes: 20
  parallel_clients: 4
  episodes_per_client: 5
  episode_offset: 0
  max_steps: 25
  action_horizon: 16
  save_video: true

server:
  checkpoint: step_best
  vlm_name: local_data/models/InternVL3-1B
  vlm_local_files_only: true
  bf16: true
  per_client_progress_state: true
```

这能避免两个问题：

1. eval 次数和并发数写死在脚本里。
2. server 每次默认去 Hugging Face 拉 VLM，而不是优先使用本地权重。

### 10.3 Run manifest 应该记录最终展开值

建议每次训练和评测都落盘一个 manifest，包含：

1. git commit 和 dirty diff 状态。
2. config 文件路径和最终展开后的 config。
3. checkpoint 输入和输出路径。
4. cache manifest checksum 或样本数。
5. episode 数、pass 数估计、batch unit。
6. optimizer、scheduler、lr groups、warmup、max steps。
7. precision、device、显存 floor 或资源占用约束。
8. eval task、episode offset、parallel clients、视频路径。
9. best checkpoint 选择依据。

这样后续讨论“5k step 是否正常”“best loss 出现在第几步”“eval 用的是否是本地模型”时，不需要重新翻日志猜。

## 11. 和当前训练状态的对应关系

结合当前 PrismVLA 的实际训练口径，文档应统一为：

1. Stage1 当前是 `episode-level fixed-replan-node training`。
2. `batch_size=1` 表示每个 optimizer step 处理一条 episode。
3. 每条 episode 内顺序递推 frozen progress state `M`。
4. 所有 full-horizon nodes 计算 FM loss。
5. 5000 optimizer steps 大约等于 500 episodes × 10 passes。
6. 当前是 action-side / bridge / action head 热身，不能把它解释成完整 VLA 端到端训练。
7. 当前 eval 能靠近目标物体但抓取和放置不稳定，这更像 action head 和阶段策略还不够，而不是训练推理链路整体错位。

后续训练建议：

1. 从 `step_best` 或已确认健康的 checkpoint 继续，而不是只看最后一步。
2. 降低 continuation LR 或缩小高 LR group，避免继续用热身阶段过激设置。
3. 每隔固定步数跑小规模固定 eval profile，视频和 manifest 一起保存。
4. 增加 action diagnostics：per-dim loss、gripper sign、chunk 内时间位置 loss、open-loop reconstruction。
5. 等 Stage1 行为稳定后，再设计 Stage2 raw image / VLM unfreeze recipe。

## 12. 最终建议的 recipe 组织方式

建议仓库中形成三类文件：

1. `configs/experiment/libero_stage1.yaml`  
   只描述训练 recipe，包括 data/model/trainer/checkpoint。

2. `configs/runtime/libero_profiles/*.env` 或后续 YAML 化 profile  
   只描述 eval / serve / rollout profile，包括 task、episode、parallel clients、video、checkpoint。

3. `run_outputs/**/manifest.yaml`  
   每次运行自动生成，保存最终展开配置和结果索引。

脚本的责任应该只剩三件事：

1. 读取 profile。
2. 调用包内训练或评测入口。
3. 把命令和最终配置写入 manifest。

这比把参数写在脚本里更接近 GR00T / LeRobot 这类近期仓库的 recipe 方向，也能避免后续再出现“batch size、eval 并发、本地 VLM、checkpoint policy 到底在哪里定义”的混乱。
