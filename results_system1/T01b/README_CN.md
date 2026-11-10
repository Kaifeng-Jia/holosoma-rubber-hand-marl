# T01b 返包（第一阶段）：F1–F6、T03 第 0 轮核对、云端准备

本文件只写事实。漂移分析、方案 A 正式评测、T04 方案 A 质量档位正在本机 GPU 上按顺序运行，完成后在本目录追加第二阶段返包。

## Receipt

- 任务书：`docs/system1/T01b_fixes_drift_schemeA_zh.md`，sha256 `c17e49b21dff595357d02ca44aa4ceb26ed9a57604ea57f3a24fd3b473e28527`
  - 同时读了新版 `T03_schemeC_marl_sonic_zh.md`（`5b726758…adae7e0`）、`T04_mass_sweep_zh.md`（`3d4f2787…28f9f4`）、`README_zh.md`（`e4c641a0…5ab314`）。
- 开始：2026-11-10 约 01:50 EST（git pull 之后；第一条产出是 01:55 写的 T04-D 停止记录）。F1–F6 全部完成：03:09 EST。
- 机器：hostname `kevin-Legion-R9000P-ADR10`，GPU NVIDIA GeForce RTX 5070 Laptop（8151 MiB）。
- 本目录 git HEAD：开工时 `b8a81c3`；本返包基于 `ae5fa59` 加本次提交。
- Python：`hssim_s1`（由 `hssim` 克隆，Python 3.11.15，torch 2.7.0+cu128，Isaac Sim 5.1 / Isaac Lab 2.3.0）；SONIC 原生评测用 `sonic_s1`。

## 0. 本机 T04-D 队列

- 开工时队列已经停了，不是我停的。
  - 停止时间：2026-11-09 22:14 EST，系统关机（`last -x`：shutdown 22:14，reboot 2026-11-10 00:31）。
  - 停止时正在跑 `pull_m5kg_seed721_joint8000`：最后一行 metrics 是第 4680 轮（22:14:43）；最后一个 checkpoint 是 `model_04050.pt`（21:31:08）。stdout 最后是正常的 DEBUG 行，没有 traceback。没有 exit 记录，因为队列进程随系统一起结束了。
  - 前一段 `pull_m5kg_seed721_critic50` 已完成。
- 处理：日志全部保留在 `logs_system1/T04/D/`，停止记录在 `logs_system1/T04/D/ops/pull_m5kg_seed721_joint8000.stopped.json`。
  - 结论：**资源调整中止，未评测，不使用。**
- T01 在 CPU PhysX 上跑的方案 A 临时结果（`logs_system1/T01/schemeA_cpu/`）：标为**试跑**，不交付。
  - 目录里加了 `TRIAL_ONLY_试跑.txt`。
  - 12 组里有 11 组有 summary。缺的 push_pm2 是 T01 时一次失败的尝试，已移到 `_to_delete/T01_schemeA_cpu_push_pm2_failed_attempt1`，没有重跑。

## 1. F1–F6

总门文件：`F1_F6_gate.json`（`all_passed: true`）。本机流水线读到它之后才开始方案 A。

| 项 | 结果 | 文件 |
|---|---|---|
| F1 离线（门槛 < 1e-6，TF32 关） | 六个任务各 120 步。tokenizer、actor 观测、动作、q*、参考关节（与 npz 和原始文件比）、`[E, A×29]` 排布的最大差全部为 0。**通过** | `F1_offline_dual_vs_single.json` |
| F1 在线（只记录） | 12 段（6 任务 × 2 台）已完成。双机器人环境中物体移到 (50, 50) m，与单机器人结果并列，见下表 | `F1_online_dual_vs_single.{csv,json}` |
| F2 冻结动作缩放 | 方案 A manifest：hip_pitch Kp 99.098，动作缩放 0.54755，PD 目标 − q* 最大差 0（3 回合）。三个 C run 的 run_config 中值相同。开关关闭时 D 回归：Pull 721 ±2 共 51 回合逐项相同，轨迹最大差 0。**通过** | `F2_frozen_action_scales.json`、`F2_regression_D_pull721_pm2.{csv,json}` |
| F3 TF32 关 | 与原生导出比动作最大差 6.68e-6（预期约 1e-5）。batch 1/2/128：原样批量推理最大差 6.7e-6，不一致；改为固定 128 行分块后差为 0。验证 a（GPU）：两段平均 MPJPE 58.16 mm，门槛 1.2 × 72.65 = 87.19 mm（原生 TF32 关），没有摔倒。验证 b 已记录。**通过** | `F3_tf32_off.json` |
| F4 body 名对应 | pelvis、14 个跟踪 body、4 个末端 body，六个任务全部对应。断言写在 `SonicDualDriver.__init__`。**通过** | `F4_body_correspondence.json` |
| F5 按环境重置 | 8 个环境，第 3 步重置其中 2 个：其余 6 个与不重置时差 0；被重置的 2 个与新开始时差 0。**通过** | `F5_per_env_reset.json` |
| F6 run_config diff | 9 个预检段（云端命令原样运行 10 轮）没有意外字段，见第 3 节。**通过** | `F6_summary.json`、`F6_run_config_diffs/` |

F1 在线（双机器人 vs 单机器人，GPU PhysX，TF32 关；MPJPE 和漂移单位 mm）：

| 任务/机器人 | 双：MPJPE | 单：MPJPE | 双：相对骨盆 | 单：相对骨盆 | 双：末帧漂移 | 单：末帧漂移 | 摔倒（双/单） |
|---|---|---|---|---|---|---|---|
| push/0 | 364.3 | 503.1 | 40.6 | 39.5 | 551 | 743 | 否 / 否 |
| push/1 | 524.0 | 522.1 | 39.4 | 39.4 | 887 | 882 | 否 / 否 |
| pull/0 | 281.5 | 281.5 | 32.8 | 32.8 | 443 | 443 | 否 / 否 |
| pull/1 | 269.5 | 269.6 | 33.4 | 33.4 | 449 | 450 | 否 / 否 |
| kick/0 | 266.6 | 269.3 | 34.5 | 34.2 | 239 | 232 | 是(297) / 是(297) |
| kick/1 | 259.4 | 265.5 | 35.9 | 36.0 | 217 | 223 | 是(297) / 是(297) |
| chair/0 | 196.6 | 173.9 | 47.8 | 46.6 | 409 | 338 | 否 / 否 |
| chair/1 | 207.1 | 264.5 | 46.3 | 46.0 | 646 | 942 | 否 / 否 |
| bucket/0 | 80.7 | 82.1 | 43.8 | 43.7 | 150 | 150 | 否 / 否 |
| bucket/1 | 382.2 | 378.7 | 35.8 | 35.8 | 1184 | 1167 | 否 / 否 |
| table/0 | 714.1 | 714.9 | 28.6 | 28.8 | 1435 | 1436 | 否 / 否 |
| table/1 | 460.5 | 447.7 | 42.3 | 42.5 | 887 | 858 | 否 / 否 |

- 两种环境中，Kick 两台机器人都在第 297 步（最后一步）被判摔倒，原因见第 4 节。
- CORE4D 三个任务：环境要求两个终止项都在。因此保留 `reference_horizon`，`joint_bad_tracking` 的三个阈值设为 1e9，只用于这个诊断。
  - 第一次运行因终止项不全报错，日志改名为 `*.attempt1.log` 保留。

## 2. T03 第 0 轮核对

- 内容：方案 C 的 `model_00000.pt`（输出层权重和偏置为零）与方案 A 在同一组冻结初始状态上比较。每个任务 3 个回合（nominal + 2 个 ±2），TF32 关。
- 结果：Pull、Table、Bucket 三个任务的全部轨迹数组最大差为 0，回合字段没有差异。**通过**
- 文件：`T03_iter0_schemeC_vs_A.json`。
- checkpoint 来自第一次 C 预检（见下）；保存这个 checkpoint 时还没有执行第一步。

## 3. 云端准备

### 第一次 C 预检失败与修复

- 现象：三个 C 预检全部在第一步无限递归。
- 原因：训练脚本和 telemetry 会把 `env.step` 换成自己的观测函数，而 `SonicResidualEnv.__setattr__` 把这个赋值转发给了内层环境，形成循环。
- 修复：`step` 留在 wrapper 上（`system1/scheme_c.py`，`_OWN` 加入 `"step"`）。
- 重跑：日志在 `logs_system1/T01b/preflight_c2/`，三个都跑满 10 轮。

### 预检结果

全部 10 轮，没有 NaN/Inf。本机 RTX 5070 Laptop 单独运行，取第 4–10 轮的中位数：

| run | 迭代数 | 本机 s/it | diff 对照 | 意外字段 |
|---|---|---|---|---|
| d_pull:40（critic50 + joint8000） | 50 + 8000 | 4.84（critic50 段） | V3-PullFull-rerun seed721 | 0 |
| d_table:10 | 12000 | 5.19 | T-07 seed721 | 0 |
| d_table:20 | 12000 | 5.19 | T-07 seed721 | 0 |
| d_bucket:2 | 12000 | 5.43 | B-RZ0 seed721 | 0 |
| d_bucket:4 | 12000 | 5.43 | B-RZ0 seed721 | 0 |
| d_bucket:8 | 12000 | 5.15 | B-RZ0 seed721 | 0 |
| c_pull:20（critic50 + joint8000） | 50 + 8000 | 4.10（critic50 段） | V3-25 scratch seed721 | 0 |
| c_table:5 | 12000 | 4.67 | T-07 seed721 | 0 |
| c_bucket:1 | 12000 | 5.30 | B-RZ0 seed721 | 0 |

- **允许不同的字段：**
  - D：质量相关字段、目录、元数据。
  - C：`system1_scheme_c`（PD 增益、冻结缩放、SONIC checkpoint 和 spec 的 sha256、TF32 设置等），以及 Table/Bin 的 `save_interval` 2000 → 1000（T03 要求）。
  - `bucket_reference`：只是 checkout 根目录不同。旧目录和本目录中该文件的 sha256 相同（table `d0bd50f2…`，bucket `3de13d54…`），归为目录类。
- **没有预检的段：** 两个 Pull 的 joint8000 段。原因是 10 轮预检不会产生它们的父 checkpoint `model_00050.pt`。
  - 处理：`run_train_queue.py` 改为每段启动、写出 run_config 后，立即对照论文 run 做 diff。出现意外字段就停止该段和整个队列，diff 存为 `ops/<label>.run_config_diff.json`。
  - joint8000 的对照是 V3 对应的 joint8000 run_config。
  - 已用本机 7 个预检的 run_config 测过这个检查，全部判为通过。

### 云端脚本（`system1/cloud/`）

| 脚本 | 作用 |
|---|---|
| `setup_cloud.sh` | 新克隆 `system1-sonic`；用 holosoma 自带的 `setup_isaacsim.sh` 建 `hssim_s1`；SONIC 代码固定在 `1e1ec4f5`，权重 HF rev `6733128a`，并校验 sha256；解包输入；记录版本 |
| `make_cloud_inputs.py` | 打包仓库里没有的只读输入，不进 git。新版 `exports_system1/cloud_inputs_v2/cloud_inputs.tar`（27.6 MB，sha256 `964f9a9621d7c6cc47ff54cc1754073ba36fbf41c65134dd9059b3cb77bcbb1d`）比旧版多了 6 份论文 run_config，供启动时 diff |
| `repro_pull.sh` | Table Pulling 论文 seed721：±2（51 回合）和 ±8 并行在两张卡上评测，逐回合对照 V3 `episodes.csv` |
| `run_train_queue.py` | 一张 GPU 一个队列；每段记录 launch/exit；启动时做 F6 diff；跳过已完成段；遇到失败就停 |
| `start_queues.py` | 按 GPU 数分配 9 个 run（最长的先排，每次给负载最小的卡），每个队列分配独立 CPU 段和 LOCAL_RANK，后台启动；`--dry-run` 只打印方案 |

## 4. 其他发现（只记录）

- **Kick 参考的最后一帧不连续。**
  - 文件：论文用的 `rubber_hand_largetable_v1/kick/plan5_attempt09_dual_kick_mirrored_runtime.npz`（sha256 `8eba6106…`）。
  - 第 296 → 297 帧：两台机器人的 body 位置最大跳 0.815 m，其他相邻帧最大 0.059 m；根部朝向一帧内变化约 154°；关节最大跳 0.527 rad。物体轨迹在这一帧没有跳。
  - 其他五个任务没有这种情况。最大单帧 body 位移：
    - Bucket 0.10 m（机器人 0 在 377 帧、机器人 1 在 317 帧）；
    - Push、Pull、Chair、Table ≤ 0.051 m；
    - Kick 除最后一帧外最大 0.056 m。
  - 影响一：两种环境里 Kick 都在第 297 步判为摔倒。这一步骨盆姿态误差平方从 0.04 跳到 7.0。
  - 影响二：SONIC 的未来窗口在末尾钳到最后一帧，所以从第 252 步起，窗口里有这一帧。双机器人环境中，相对骨盆 MPJPE 在第 201–251 步约 35/37 mm，第 276–296 步约 56/68 mm。
  - 漂移分析按预登记的定义照算，不剔除。方案 A 也照原设置跑，没有改参考或窗口。
- **F1 在线，全局 MPJPE 的双减单：**
  - Push 机器人 0：−138.8 mm；Chair 机器人 1：−57.4 mm；Chair 机器人 0：+22.7 mm；Table 机器人 1：+12.8 mm。
  - 其余 8 段的绝对差 ≤ 6.1 mm。
  - 相对骨盆 MPJPE 在 12 段里的差都 ≤ 1.2 mm。
- **控制组 (iii)：** 单人先验在同一设置下（单机器人、无物体、第 0 帧开始）跟踪角色参考。
  - Push 两段跑满 308 步，没有摔倒。
  - Pull 两段分别在第 82、51 步摔倒；Kick 两段分别在第 42、21 步摔倒。
  - 三个先验的训练动作都是带物体的单人示范（`*_w_obj.npz`）：Push 为 `a1/sub6_largetable_033`，Pull 为 `pull/sub3_largetable_010`，Kick 为 `kick/sub16_largetable_028`。
  - 摔倒原因没有查实，没有改设置。

## 5. 与任务书的偏离

1. F1 在线（CORE4D）：保留了 `joint_bad_tracking`，阈值设为 1e9，只用于诊断。Plan5 任务只保留 timeout。最后一步的重置检查跳过，因为 timeout 恰好发生在最后一步。
2. F6：两个 Pull joint8000 段改为在云端启动时自动 diff（见第 3 节）。
3. 本机流水线 `system1/t01b_pipeline.sh` 加了一道门：方案 A 正式评测要等 `F1_F6_gate.json` 写出且 `all_passed` 为真才开始。漂移分析和控制组不等这道门。
4. 训练相关代码改动，开关关闭时行为不变：`scheme_c.py` 的 step 修复，`run_train_queue.py` 的 SIGKILL 兜底和启动时 diff。

## 6. 进行中（本机 GPU，按顺序）

- 03:08–03:09 EST：控制组 6 段完成。
- 03:09:42：漂移分析完成，输出在 `drift/`；(v) 检查 `integration_problem_found: false`。第二阶段返包会补写这部分说明。
- 03:09:42：门文件通过，方案 A 正式评测开始。


- 漂移分析控制组 (iii) → 漂移分析 (i)(ii)(iv)(v) → (v) 检查 → 方案 A 正式评测（六任务 ±2、±8，含 nominal）→ hip_pitch 敏感性（Push/Pull/Kick 两档）→ T04 方案 A（Pull 5/10/40 kg，Table 2.5/10/20 kg，Bin 2/4/8 kg）。
- 日志在 `logs_system1/T01b/pipeline.log`。
