# Demo 3 / Demo 4 云端训练清单

## 1. 当前结论

- 分支：`rubber_hand_marl_baseline`
- Demo 3 里程碑：`0d864f713156f07eb72253483c2d0553e3072978`
- Demo 4 里程碑：`e8c9db4a4d7ba744ffb121c078a6d1b5f1dd3c12`
- Demo 4 已具备正式 train、真实 PPO-update smoke、actor-only evaluate/record 和 resume 入口。
- Demo 3 已具备 reference、环境、共享 Actor、ego-first Critic 和底层 PPO，但还缺正式 train、
  真实 PPO-update smoke、evaluate/record 三个入口。因此 **Demo 3 暂时不能启动正式云端训练**。
- 本文件只负责云端执行和资产交付，不替代
  `MULTI_AGENT_EMERGENCE_ROADMAP.md` 的研究决策地位。
- Demo 4 里程碑当前首先是本地提交；在云端 clone 前，必须把里程碑和本清单后续提交显式推送到
  `private` remote。没有推送成功前不得假定云端能取得这些文件。

## 2. 并行方式

Demo 3 与 Demo 4 是两个完全独立的单 GPU 任务，不使用 DDP 或 `torchrun`：

```text
GPU 0：Demo 3，CUDA_VISIBLE_DEVICES=0，WORLD_SIZE=1
GPU 1：Demo 4，CUDA_VISIBLE_DEVICES=1，WORLD_SIZE=1
```

建议在云端使用两个独立 clone，而不是让两个 Isaac Sim 进程共用一个 checkout。这样不仅隔离
日志，也避免两个进程首次启动时同时写 `converted_rank0` USD cache。

```bash
git clone <private-repository-url> holosoma-demo3
git clone <private-repository-url> holosoma-demo4
git -C holosoma-demo3 checkout rubber_hand_marl_baseline
git -C holosoma-demo4 checkout rubber_hand_marl_baseline
git -C holosoma-demo3 merge-base --is-ancestor e8c9db4a HEAD
git -C holosoma-demo4 merge-base --is-ancestor e8c9db4a HEAD
test "$(git -C holosoma-demo3 rev-parse HEAD)" = \
  "$(git -C holosoma-demo4 rev-parse HEAD)"
git -C holosoma-demo3 rev-parse HEAD
```

最后一条输出就是本次部署 commit，必须抄入两个 run 的记录中。祖先检查只保证 Demo 4 代码存在；
两个 clone 的 HEAD 相等检查才保证它们使用同一个部署版本。

在两个 clone 中分别安装/激活相同的项目环境：

```bash
bash scripts/setup_isaacsim.sh
source scripts/source_isaacsim_setup.sh
```

若两个 clone 共用同一套全局 conda/Isaac 安装，首次安装必须串行执行；只有资产转换和 smoke
分别通过后，才并行启动两个训练进程。

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

Demo 3 的 `actor164` 文件是当前环境 smoke 的强制依赖，也是当前正式 warm-start 候选；如果
第 5.3 节最终决定改用 Plan5 Pull08050，必须同步更新 checkpoint、初始化代码和资产清单。

只有重新合成 Demo 4 reference 时才需要
`logs/Plan5Pull/eval_full8000_seed721/model_08050_object_centric.npz`；正式训练和评估不需要它。

## 4. Demo 4：云端启动流程

### 4.1 资产与环境 smoke

以下命令均在 `holosoma-demo4` 中运行。console log 应写在训练 output directory 外部；正式
训练入口会拒绝复用一个预先非空的输出目录。

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 \
python scripts/smoke_demo4_rotate_environment.py \
  --steps 8 \
  --seed 721 \
  --output /tmp/demo4_environment_smoke.json

env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 \
python scripts/smoke_demo4_rotate_ppo_update.py
```

第二条只运行 `1 env × 4 steps × 1 epoch × 1 minibatch`，验证真实 rollout、Actor/Critic 更新及
checkpoint round-trip，不属于正式训练。

### 4.2 环境数选择

`2048` 是当前代码 preset，也是既有 Plan5 双机器人正式训练采用的规模；Demo 4 本地真实
environment/PPO smoke 使用的是 `1 env`，并未做 2048-env 容量测试。`2048` 与 `4096` 都必须在
目标云端 GPU 验证容量；`4096` 在接口上受支持，但相同 8000 iterations 会把总样本量翻倍。

若希望使用 4096，先运行一次独立容量测试：

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 \
python scripts/train_demo4_rotate.py \
  --iterations 1 \
  --num-envs 4096 \
  --steps-per-env 24 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/capacity_only_seed721_env4096
```

通过后从 iteration 0 启动正式 run，不把 capacity checkpoint 当作正式初始化。

### 4.3 当前已确认的正式 2048-env 命令

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 \
python scripts/train_demo4_rotate.py \
  --iterations 8000 \
  --num-envs 2048 \
  --steps-per-env 24 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/rectangular_pull_pull_rotate90_seed721_env2048
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
checkpoint        每 1000 iterations，且总会保存最终 iteration
```

新 run 的编号从 `model_00000.pt` 到 `model_08000.pt`；Pull 的 `08050` 只是 Actor warm-start 来源，
不会让 Demo 4 从 iteration 8050 编号。

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

以下示例从 iteration 4000 再训练 4000 轮：

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 \
python scripts/train_demo4_rotate.py \
  --resume logs/Demo4Rotate/rectangular_pull_pull_rotate90_seed721_env2048/model_04000.pt \
  --iterations 4000 \
  --num-envs 2048 \
  --steps-per-env 24 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/rectangular_pull_pull_rotate90_seed721_env2048
```

`--iterations` 表示额外轮数。Resume 恢复 Actor、Critic、两个 optimizer、两个 normalizer、iteration
和 optimizer 中的实际学习率，但不恢复物理瞬时状态、rollout buffer 或完整 RNG 连续状态。
若从旧 checkpoint 分叉新实验，应使用新的 output directory，避免覆盖原目录中编号更大的模型。

### 4.6 Actor-only 评估与 ViSER

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 \
python scripts/evaluate_demo4_rotate.py \
  --checkpoint logs/Demo4Rotate/rectangular_pull_pull_rotate90_seed721_env2048/model_08000.pt \
  --episodes 5 \
  --seed 721 \
  --output-dir logs/Demo4Rotate/eval_model08000_seed721
```

评估输出 `evaluation.json` 和 `representative_episode.npz`。多 seed 应使用独立目录，例如
`seed721/722/723`；同一 seed 下重复 deterministic episode 不等于随机鲁棒性验证。

```bash
PYTHONPATH=src/holosoma_retargeting \
python -m holosoma_retargeting.viser_dual_a1_player \
  --rollout-npz logs/Demo4Rotate/eval_model08000_seed721/representative_episode.npz \
  --robot-urdf src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf \
  --object-urdf src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_widetable_plan5_pull_training.urdf \
  --port 8098
```

## 5. Demo 3：当前可执行部分与正式训练缺口

### 5.1 已可执行的环境 smoke

以下命令在 `holosoma-demo3` 中运行：

```bash
env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/smoke_demo3_tug_environment.py \
  --steps 8 \
  --seed 721 \
  --output /tmp/demo3_environment_smoke.json
```

当前代码应确认：两台 rubber-hand G1、方桌 20 kg、材料 `0.5/0.5/0`、Actor groups
`154+4+6=164`、critic `[E,2,527]`、per-agent reward/value 与 shared physical done。

### 5.2 为什么目前不能直接正式训练

仓库中尚不存在：

```text
scripts/train_demo3_tug.py
scripts/smoke_demo3_tug_ppo_update.py
scripts/evaluate_demo3_tug.py
```

底层 `Demo3PPO` 虽已实现，但直接临时拼接 Python 命令会绕过 output-directory 防覆盖、配置记录、
checkpoint 恢复契约、真实 rollout 指标和 terminal-state 录制，因此禁止把这种临时方式用于云端
正式 run。

### 5.3 补入口前需确认的 Demo 3 契约

1. Warm-start：保持当前 WBT Pull07999 的 164-D lossless 扩展，还是改为 Plan5 Pull08050。
2. 对手更新：首版是否采用当前设计，即双方共享同一个、同步更新的 Actor，而不是 frozen opponent
   或 opponent pool。
3. 任务奖励：当前 `signed_table_progress_velocity` 权重 `10.0` 是否转为正式值。
4. 训练节奏：是否采用 `50 critic-only + 8000 full-actor`、环境数 2048，以及每 1000 轮保存。
5. 胜负协议：以 episode 末桌子沿两条相反 pull axis 的净位移判定胜负时，平局阈值是多少。
6. 随机化：首版是否保持固定 frame-0、固定质量/摩擦和无对手初态随机化，先建立最小 baseline。

在上述六项确认前，继承配置中的 `30000 iterations` 以及 provisional reward `10.0` 都不能视为
正式训练决定。

## 6. 云端训练结束后必须回收

每个正式 run 至少保存并下载：

```text
run_config.json
metrics.jsonl
status.json
model_00000.pt
按该 Demo 最终确认间隔保存的 model_*.pt（Demo 4 为每 1000；Demo 3 待确认）
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
