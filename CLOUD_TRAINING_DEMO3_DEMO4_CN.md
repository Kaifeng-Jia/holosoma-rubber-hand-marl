# Demo 3 / Demo 4 云端训练清单

## 1. 当前结论

- 分支：`rubber_hand_marl_baseline`
- Demo 3 里程碑：`0d864f713156f07eb72253483c2d0553e3072978`
- Demo 4 里程碑：`e8c9db4a4d7ba744ffb121c078a6d1b5f1dd3c12`
- 本清单基础里程碑：`2266fbcc5284768577488db1c0b42d83644ed275`；Demo 3 正式管线位于
  **包含本清单最新版本的部署 HEAD**，部署时必须记录该 HEAD，不能只停在前三个历史里程碑。
- Demo 3 与 Demo 4 均已具备正式 train、真实 PPO-update smoke、actor-only evaluate/record 和
  resume 入口；两者都**尚未开始正式训练**。
- 本文件只负责云端执行和资产交付，不替代
  `MULTI_AGENT_EMERGENCE_ROADMAP.md` 的研究决策地位。
- 本轮 Demo 3/4 里程碑当前首先是本地提交；在云端 clone 前，必须把里程碑和本清单后续提交显式推送到
  `private` remote。没有推送成功前不得假定云端能取得这些文件。

## 2. 运行方式

Demo 3 与 Demo 4 是两个完全独立的单 GPU 任务，不使用 DDP 或 `torchrun`。2026-08-29 用户确认
租用一张 RTX 4090，并以 `4096` environments 顺序训练：

```text
阶段 A：GPU 0 训练 Demo 3，CUDA_VISIBLE_DEVICES=0，WORLD_SIZE=1
阶段 B：Demo 3 完成并回收后，GPU 0 训练 Demo 4，CUDA_VISIBLE_DEVICES=0，WORLD_SIZE=1
```

建议在云端使用两个独立 clone，而不是让两个 Isaac Sim 进程共用一个 checkout。这样不仅隔离
日志，也避免两个进程首次启动时同时写 `converted_rank0` USD cache。

```bash
git clone <private-repository-url> holosoma-demo3
git clone <private-repository-url> holosoma-demo4
git -C holosoma-demo3 checkout rubber_hand_marl_baseline
git -C holosoma-demo4 checkout rubber_hand_marl_baseline
git -C holosoma-demo3 merge-base --is-ancestor 2266fbcc HEAD
git -C holosoma-demo4 merge-base --is-ancestor 2266fbcc HEAD
test "$(git -C holosoma-demo3 rev-parse HEAD)" = \
  "$(git -C holosoma-demo4 rev-parse HEAD)"
test -f holosoma-demo3/scripts/train_demo3_tug.py
test -f holosoma-demo3/scripts/evaluate_demo3_tug.py
test -f holosoma-demo4/scripts/train_demo4_rotate.py
git -C holosoma-demo3 rev-parse HEAD
```

最后一条输出就是本次部署 commit，必须抄入两个 run 的记录中。祖先检查只保证云端清单基础存在；
入口文件检查和两个 clone 的 HEAD 相等检查共同保证它们使用同一个完整部署版本。

在两个 clone 中分别安装/激活相同的项目环境：

```bash
bash scripts/setup_isaacsim.sh
source scripts/source_isaacsim_setup.sh
```

若两个 clone 共用同一套全局 conda/Isaac 安装，首次安装必须串行执行；资产转换和 smoke
也依次完成。正式训练先启动 Demo 3，完成并回收后再启动 Demo 4，不同时运行两个训练进程。

不要上传或共享以下运行时缓存：

```text
src/holosoma/holosoma/data/robots/converted_rank*/
__pycache__/
.pytest_cache/
```

## 3. 必需资产

完整的路径、字节数和 SHA256 见 `CLOUD_ASSETS_DEMO3_DEMO4.tsv`。

Git 会携带 runtime reference、URDF 和 mesh，但整个 `logs/` 被 `.gitignore` 排除。云端必须
额外复制以下 checkpoint 到**完全相同的仓库相对路径**：

```text
Demo 3:
logs/Demo3Tug/checkpoints/model_07999_actor164_table_neutral.pt
SHA256 048f952cad01d5fda42851502347ee626751dccab923905af303eb44807ef01b

Demo 4:
logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt
SHA256 727630e9cec654d88bfb454d5eaabd70d7db629478598a2039140653a57cbf43
```

上传后，在 Demo 3 clone 中校验所有必需的 tracked 与 external 资产：

```bash
awk -F '\t' 'NR>1 && ($1=="shared" || $1=="demo3") && $3!~/optional/ \
  {print $6 "  " $7}' CLOUD_ASSETS_DEMO3_DEMO4.tsv | sha256sum -c -
```

在 Demo 4 clone 中执行同样的完整校验：

```bash
awk -F '\t' 'NR>1 && ($1=="shared" || $1=="demo4") && $3!~/optional/ \
  {print $6 "  " $7}' CLOUD_ASSETS_DEMO3_DEMO4.tsv | sha256sum -c -
```

这两个命令同时校验各自 clone 的 robot URDF、table URDF、runtime reference 和外部 checkpoint；
Demo 3 JSON manifest 与 Demo 4 ViSER reference 属于可选 provenance/preview，不参与强制校验。

Demo 3 的 `actor164` 文件是环境 smoke、正式训练和 actor-only 评估的冻结 warm-start。当前契约
明确使用 WBT Pull07999 的 164-D lossless 扩展，不再把 Plan5 Pull08050 视为本轮候选。

只有重新合成 Demo 4 reference 时才需要
`logs/Plan5Pull/eval_full8000_seed721/model_08050_object_centric.npz`；正式训练和评估不需要它。

## 4. Demo 4：云端启动流程

### 4.1 资产与环境 smoke

以下命令均在 `holosoma-demo4` 中运行。console log 应写在训练 output directory 外部；正式
训练入口会拒绝复用一个预先非空的输出目录。

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/smoke_demo4_rotate_environment.py \
  --steps 8 \
  --seed 721 \
  --output /tmp/demo4_environment_smoke.json

env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/smoke_demo4_rotate_ppo_update.py
```

第二条只运行 `1 env × 4 steps × 1 epoch × 1 minibatch`，验证真实 rollout、Actor/Critic 更新及
checkpoint round-trip，不属于正式训练。

### 4.2 环境数选择

代码 preset 与本轮正式云端实验均已冻结为 `4096`。相同 15000 iterations 下，它相对 2048
会把总采样量翻倍，因此必须先在目标 RTX 4090 上运行一次独立容量测试：

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/train_demo4_rotate.py \
  --iterations 1 \
  --num-envs 4096 \
  --steps-per-env 24 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/capacity_only_seed721_env4096
```

通过后从 iteration 0 启动正式 run，不把 capacity checkpoint 当作正式初始化。

### 4.3 当前已确认的正式 4096-env 命令

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/train_demo4_rotate.py \
  --iterations 15000 \
  --num-envs 4096 \
  --steps-per-env 24 \
  --checkpoint-interval 150 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/rectangular_pull_pull_rotate90_full15000_seed721_env4096
```

固定设置：

```text
shared Actor      164 → 512 → 256 → 128 → 29
team Critic       527 → 512 → 256 → 128 → 1
PPO               5 epochs，4 mini-batches，clip 0.2
gamma/lambda      0.99 / 0.95
entropy/LR        0.005 / actor 1e-3 / critic 1e-3
physics/control   200 Hz / 50 Hz
table             20 kg，friction 0.5/0.5，restitution 0
episode           316 frames，6.32 s
target            actual table +90° unwrapped yaw progress
checkpoint        每 150 iterations，且总会保存最终 iteration
```

新 run 的编号从 `model_00000.pt` 到 `model_15000.pt`；Pull 的 `08050` 只是 Actor warm-start 来源，
不会让 Demo 4 从 iteration 8050 编号。正式 run 共保存 100 个周期 checkpoint，加上初始
`model_00000.pt`，合计 101 个模型文件。

### 4.4 监控

每轮查看 `metrics.jsonl` 中至少这些字段：

```text
reward_mean
completed_episodes
success_count / success_rate
robot_fall_count
table_safety_count
reference_horizon_count
yaw_sample_mean_rad / min / max
terminal_yaw_mean_rad / min / max
policy_drift_mean_abs / max_abs
surrogate_loss / value_loss / entropy / kl
actor_grad_norm / critic_grad_norm
```

不能仅凭训练 reward 或 iteration 编号选 checkpoint；训练后必须做 actor-only 物理回放。

### 4.5 Resume

以下示例从 iteration 7500 再训练 7500 轮：

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/train_demo4_rotate.py \
  --resume logs/Demo4Rotate/rectangular_pull_pull_rotate90_full15000_seed721_env4096/model_07500.pt \
  --iterations 7500 \
  --num-envs 4096 \
  --steps-per-env 24 \
  --checkpoint-interval 150 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/rectangular_pull_pull_rotate90_full15000_seed721_env4096
```

`--iterations` 表示额外轮数。Resume 恢复 Actor、Critic、两个 optimizer、两个 normalizer、iteration
和 optimizer 中的实际学习率，但不恢复物理瞬时状态、rollout buffer 或完整 RNG 连续状态。
若从旧 checkpoint 分叉新实验，应使用新的 output directory，避免覆盖原目录中编号更大的模型。

### 4.6 Actor-only 评估与 ViSER

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_demo4_rotate.py \
  --checkpoint logs/Demo4Rotate/rectangular_pull_pull_rotate90_full15000_seed721_env4096/model_15000.pt \
  --episodes 5 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/eval_model15000_seed721
```

评估输出 `evaluation.json` 和 `representative_episode.npz`。多 seed 应使用独立目录，例如
`seed721/722/723`；同一 seed 下重复 deterministic episode 不等于随机鲁棒性验证。

```bash
PYTHONPATH=src/holosoma_retargeting \
python -m holosoma_retargeting.viser_dual_a1_player \
  --rollout-npz logs/Demo4Rotate/eval_model15000_seed721/representative_episode.npz \
  --robot-urdf src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf \
  --object-urdf src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_widetable_plan5_pull_training.urdf \
  --port 8098
```

## 5. Demo 3：正式云端启动流程

### 5.1 资产、环境与 PPO smoke

以下命令均在 `holosoma-demo3` 中运行：

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/smoke_demo3_tug_environment.py \
  --steps 8 \
  --seed 721 \
  --output /tmp/demo3_environment_smoke.json

env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/smoke_demo3_tug_ppo_update.py
```

第一条确认两台 rubber-hand G1、方桌物理常量、相反 Pull 轴和真实观测/奖励/value shape；第二条只
运行 `1 env × 4 steps × 1 epoch × 1 minibatch`，确认 Actor/Critic 确实更新、Actor normalizer
保持冻结、per-agent GAE 和 v2 checkpoint round-trip。两条都不是正式训练。

### 5.2 环境数选择

代码 preset 与本轮正式云端实验均已冻结为 `4096`。相同 iterations 下总采样量是 2048
配置的两倍，因此先在目标 RTX 4090 上做独立容量测试；若 OOM，只报告并讨论，不自动降档。

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/train_demo3_tug.py \
  --critic-only-iterations 0 \
  --full-actor-iterations 1 \
  --num-envs 4096 \
  --steps-per-env 24 \
  --checkpoint-interval 1 \
  --seed 721 \
  --output-dir logs/Demo3Tug/capacity_only_seed721_env4096
```

容量测试通过后，正式 run 仍从冻结 warm-start 重新开始，不继承 capacity checkpoint。

### 5.3 已确认的正式 4096-env 命令

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/train_demo3_tug.py \
  --critic-only-iterations 50 \
  --full-actor-iterations 15000 \
  --num-envs 4096 \
  --steps-per-env 24 \
  --checkpoint-interval 150 \
  --seed 721 \
  --output-dir logs/Demo3Tug/square_table_diagonal_tug_full15000_seed721_env4096
```

固定设置：

```text
warm-start        WBT Pull07999 的冻结 164-D lossless 扩展，只加载 Actor/normalizer
shared Actor      164 → 512 → 256 → 128 → 29；同一组实时参数分别控制 A/B
ego-first Critic  527 → 512 → 256 → 128 → 1；每个 agent 各有 value/return/GAE
PPO               5 epochs，4 mini-batches，clip 0.2
gamma/lambda      0.99 / 0.95
entropy/LR        0.005 / actor 1e-3 / critic 1e-3，adaptive KL 0.01
value/grad         value-loss coef 1.0 / max grad norm 1.0
optimizer          Actor/Critic 均为 AdamW，weight decay 0
physics/control   200 Hz / 50 Hz
table             20 kg，friction 0.5/0.5，restitution 0
episode           317-frame reference horizon；机器人明确摔倒也会终止
initialization    fixed frame-0；固定质量/材料；首版不做对手初态随机化
task reward       A/B 各自相反 Pull 轴上的 signed table progress velocity，权重 10
motion prior      六项 Pull/WBT 权重 0.5/0.5/1/1/1/1；action-rate -0.1；joint-limit -10
checkpoint        critic 边界和此后每 150 个 full-actor iterations；始终保存最终模型
```

方桌的 reference 通道只用于 reset/schema，不进行逐帧桌子 pose tracking；不限制必须用手，不惩罚
incidental contact，也不把桌子偏离演示轨迹作为 termination。首个 `50` iterations 只训练新 Critic，
之后 `15000` iterations 同步更新一个共享 Actor 和 Critic。编号为：

```text
model_00000.pt  初始 Actor + 新 Critic
model_00050.pt  critic-only 边界
model_00200.pt  150 full-actor iterations
...
model_15050.pt  15000 full-actor iterations（最终）
```

正式 run 保存 100 个 full-actor 周期 checkpoint，另有 `model_00000.pt` 和 critic 边界
`model_00050.pt`，合计 102 个模型文件。

### 5.4 监控

每轮至少查看：

```text
reward_agent_a_mean / reward_agent_b_mean
progress_agent_a_sample_mean_m / progress_agent_b_sample_mean_m
table_displacement_sample_mean_m
completed_episodes / clear_robot_fall_count / reference_horizon_count
advantage_agent_a_mean/std / advantage_agent_b_mean/std
policy_drift_mean_abs / policy_drift_max_abs
surrogate_loss / value_loss / entropy / kl
actor_grad_norm / critic_grad_norm
actor_learning_rate / critic_learning_rate
```

竞争任务中 A/B reward 不需要同时单调上升，也不能用单一总 reward 选模型。训练 checkpoint 的
最终判断必须来自独立 actor-only 物理回放。

### 5.5 Resume

以下示例从 `model_07550.pt` 继续完成原定 `50 + 15000` schedule：

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/train_demo3_tug.py \
  --resume logs/Demo3Tug/square_table_diagonal_tug_full15000_seed721_env4096/model_07550.pt \
  --critic-only-iterations 50 \
  --full-actor-iterations 15000 \
  --num-envs 4096 \
  --steps-per-env 24 \
  --checkpoint-interval 150 \
  --seed 721 \
  --output-dir logs/Demo3Tug/square_table_diagonal_tug_full15000_seed721_env4096
```

Demo 3 的两个 iteration 参数描述**完整固定 schedule**，不是 resume 后额外增加的轮数。Resume
严格恢复 Actor、Critic、两个 optimizer、两个 normalizer、iteration 和 checkpoint 内实际学习率；
只有显式传入 `--actor-learning-rate` 或 `--critic-learning-rate` 才覆盖恢复值。它不恢复物理瞬时
状态、rollout buffer 或完整 RNG 连续状态。恢复 checkpoint 必须位于本次 `--output-dir` 内，且其 metadata、
原始 seed/env/steps/schedule、完整非 LR PPO 更新契约和全部冻结资产契约必须完全一致。这里的
“同一目录”是指 checkpoint、原 `run_config.json` 和原账本必须作为一个完整目录共同存在；如果把
这个完整目录原样迁移到另一台机器或另一个绝对路径，允许继续训练，但 resume config 会明确记录
`ledger_origin_output_dir` 和 `ledger_relocated=true`，不会把迁移伪装成原路径续训。

Resume 不覆盖原 `run_config.json` 或 `metrics.jsonl`，而是生成例如：

```text
run_config_resume_from_07550.json
metrics_resume_from_07550.jsonl
status_resume_from_model_07550.json
```

若目录中已经存在比所选 resume checkpoint 更新的 `model_*.pt`，入口会拒绝倒退覆盖；若同名
resume 的 run-config、metrics 或 status 任一分段已经存在，也会要求先人工核对，不自动删除、
截断或覆盖任何历史数据。

### 5.6 Actor-only 评估与 ViSER

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_demo3_tug.py \
  --checkpoint logs/Demo3Tug/square_table_diagonal_tug_full15000_seed721_env4096/model_15050.pt \
  --episodes 5 \
  --seed 721 \
  --output-dir logs/Demo3Tug/eval_model15050_seed721
```

评估只调用共享 Actor，不调用 Critic。每个 episode 都显式执行相同的 `reset_all + zero-action
settling`，断言从相同 reference phase 开始；reset 后的实际状态差异按物理量级做 sanity check 并
完整写入 JSON。horizon episode 必须精确执行剩余 reference 步数，真实摔倒则允许提前终止并单独计数。
胜负按 episode 末桌子沿 Agent A 初始 Pull 轴的净位移：

```text
> +0.05 m  Agent A 胜
< -0.05 m  Agent B 胜
其余        平局
```

该 `0.05 m` 只属于 evaluation protocol，不进入 reward 或 termination。输出包括
`evaluation.json`，以及实际存在的 A 胜/B 胜/最大绝对位移代表 episode（重复选择只保存一个）对应的
`representative_episode_XXX.npz`。从 `evaluation.json` 选定一个路径后回放：

```bash
PYTHONPATH=src/holosoma_retargeting \
python -m holosoma_retargeting.viser_dual_a1_player \
  --rollout-npz <representative_episode_XXX.npz> \
  --robot-urdf src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf \
  --object-urdf src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_squaretable_demo3_training.urdf \
  --port 8097
```

首版 fixed frame-0 且无初态随机化。`deterministic_inference` 只表示 Actor 输出使用 mean action，
不承诺 GPU 接触求解逐 bit 相同；相同 seed 的重复 episode 可能因接触动力学出现不同胜负或提前
摔倒，它们可用于重复性/敏感性检查，但不应包装成经过初态随机化的鲁棒性证据。正式结果仍应使用
独立目录记录 seed 721/722/723，并如实说明当前随机化边界。

## 6. 云端训练结束后必须回收

每个正式 run 至少保存并下载：

```text
run_config.json
metrics.jsonl
status.json
如发生 resume：run_config_resume_from_*.json / metrics_resume_from_*.jsonl / status_resume_from_*.json
model_00000.pt
按各 Demo 已确认间隔保存的 model_*.pt（Demo 3/4 均为每 150 iterations；Demo 3 另存 00050）
最终 model_*.pt
所有 evaluation.json
每个候选 checkpoint 的 representative_episode.npz
外部 scheduler/stdout/stderr log
```

不要回收：

```text
converted_rank*/
Isaac/Kit cache
__pycache__/
.pytest_cache/
临时 capacity-only checkpoint（确认无保留价值后）
```

训练开始时和结束时都记录：Git commit、GPU 型号、驱动/CUDA、Python/torch/Isaac Sim/Isaac Lab
版本、环境数、seed、reference/checkpoint SHA256 与 wall-clock 时间。
