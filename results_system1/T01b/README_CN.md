# T01b 返包：F1–F6、T03 第 0 轮核对、云端准备（第一阶段）；漂移分析、方案 A 正式评测（第二阶段）

本文件只写事实。第 0–5 节是第一阶段（2026-11-10 03:12 EST 提交 `fbb39f6`），第 6 节起是第二阶段。T04 方案 A 质量档位的返包在 `results_system1/T04/README_CN.md`。

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


---

# 第二阶段：漂移分析、方案 A 正式评测

## 6. 时间线（本机 RTX 5070 Laptop，2026-11-10，America/New_York）

| 内容 | 开始 | 结束 |
|---|---|---|
| 漂移 (i) SONIC 单机器人 14 段（预登记 `4f31670` 于 02:08:54 提交之后） | 02:09:12 | 02:20:54 |
| 漂移 (iii) 对照组 6 段 | 03:08:02 | 03:09:42 |
| 漂移汇总 (i)(ii)(iv)(v)，门文件检查通过 | 03:09:42 | 03:09:42 |
| 方案 A 主设置：6 任务 × ±2/±8，共 12 组 | 03:09:42 | 05:12:32 |
| hip_pitch 敏感性：Push/Pull/Kick × ±2/±8，共 6 组 | 05:12:32 | 05:56:25 |
| T04 方案 A：9 个非原质量档 × ±2/±8，共 18 组（见 `results_system1/T04/README_CN.md`） | 05:56:25 | 09:08:37 |
| 本机 GPU 空闲（流水线结束后我没有立即接着处理） | 09:08:37 | 15:11 |
| nominal 视频、汇总表、T04 质量读回、评测脚本回归检查 | 15:13 | 15:26 |

36 组评测的退出码都是 0，`summary.json` 都是 `complete: true`，每组 51 回合（nominal + 50 个扰动状态）。

## 7. 漂移分析

定义和判定规则见 `preregistration_drift.md`，计算时没有改。结果文件在 `drift/`。

### (i)(ii) 逐段结果

单机器人，无物体，从第 0 帧开始，GPU PhysX，TF32 关。参考脚滑只用参考数据计算，对照组不适用。

| 组 | 段 | 步数/参考 | 摔倒（步） | 全局 MPJPE (mm) | 相对骨盆 (mm) | 末帧漂移 (cm) | 最大漂移 (cm) | 漂移速率 (cm/s) | 参考行程 (m) | 每米行程漂移 (cm/m) | 参考脚滑 S (m) | 每米行程脚滑 (cm/m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SONIC | sonic_sample_A001 | 2001/2001 | 否 | 52.8 | 17.8 | 11.7 | 13.9 | 0.29 | 16.84 | 0.7 | 8.50 | 50.5 |
| SONIC | sonic_sample_A001_M | 2001/2001 | 否 | 63.5 | 17.9 | 18.3 | 19.3 | 0.46 | 16.84 | 1.1 | 8.49 | 50.5 |
| SONIC | push_agent0 | 308/308 | 否 | 503.1 | 39.5 | 74.3 | 78.1 | 12.07 | 1.81 | 41.0 | 2.61 | 143.9 |
| SONIC | push_agent1 | 308/308 | 否 | 522.1 | 39.4 | 88.2 | 92.3 | 14.33 | 1.96 | 45.0 | 2.64 | 134.5 |
| SONIC | pull_agent0 | 315/315 | 否 | 281.5 | 32.8 | 44.3 | 50.5 | 7.04 | 1.35 | 32.9 | 1.76 | 130.4 |
| SONIC | pull_agent1 | 315/315 | 否 | 269.6 | 33.4 | 45.0 | 51.5 | 7.14 | 1.35 | 33.4 | 1.68 | 124.7 |
| SONIC | kick_agent0 | 297/297 | 是（297） | 269.3 | 34.2 | 23.2 | 50.9 | 3.91 | 2.17 | 10.7 | 3.77 | 173.9 |
| SONIC | kick_agent1 | 297/297 | 是（297） | 265.5 | 36.0 | 22.3 | 49.1 | 3.76 | 2.17 | 10.3 | 3.78 | 174.4 |
| SONIC | chair_agent0 | 390/390 | 否 | 173.9 | 46.6 | 33.8 | 43.3 | 4.34 | 1.70 | 19.9 | 0.37 | 21.7 |
| SONIC | chair_agent1 | 390/390 | 否 | 264.5 | 46.0 | 94.2 | 96.7 | 12.08 | 1.12 | 84.1 | 1.63 | 145.5 |
| SONIC | bucket_agent0 | 496/496 | 否 | 82.1 | 43.7 | 15.0 | 17.6 | 1.51 | 1.55 | 9.7 | 0.66 | 42.5 |
| SONIC | bucket_agent1 | 496/496 | 否 | 378.7 | 35.8 | 116.7 | 120.5 | 11.76 | 2.55 | 45.8 | 4.01 | 157.6 |
| SONIC | table_agent0 | 686/686 | 否 | 714.9 | 28.8 | 143.6 | 144.4 | 10.47 | 2.55 | 56.4 | 3.92 | 154.0 |
| SONIC | table_agent1 | 686/686 | 否 | 447.7 | 42.5 | 85.8 | 89.5 | 6.26 | 2.32 | 37.1 | 4.14 | 178.7 |
| 对照 | push_agent0 | 308/308 | 否 | 353.0 | 55.9 | 51.2 | 59.4 | 8.31 | 1.81 | 28.3 | | |
| 对照 | push_agent1 | 308/308 | 否 | 356.6 | 55.0 | 46.9 | 52.5 | 7.61 | 1.96 | 23.9 | | |
| 对照 | pull_agent0 | 82/315 | 是（82） | 141.9 | 87.5 | 14.4 | 20.2 | 8.80 | 0.29 | 49.2 | | |
| 对照 | pull_agent1 | 51/315 | 是（51） | 241.2 | 90.9 | 73.7 | 73.7 | 72.30 | 0.18 | 407.6 | | |
| 对照 | kick_agent0 | 42/297 | 是（42） | 299.7 | 105.3 | 72.4 | 72.4 | 86.14 | 0.28 | 259.8 | | |
| 对照 | kick_agent1 | 21/297 | 是（21） | 156.5 | 94.8 | 38.6 | 38.6 | 91.82 | 0.15 | 265.2 | | |

- 对照组摔倒的段，参考行程只算到摔倒那一步。
- 逐秒分解见 `drift/drift_per_second.csv`。12 段角色参考中，参考位移 ≥ 5 cm 的秒共 66 个，其中沿行进方向分量 along < 0（落后于参考）的有 59 个。对照组 6 段同口径为 5/15。
- Kick 两段在第 297 步按 SONIC 判据判为摔倒，这一步就是第 4 节所述的参考末帧跳变。

### (iii) 对照组

- Push、Pull、Kick 三个单人先验，结果见上表"对照"行。
- Chair、Bin、Table 是两人任务，没有单人先验：不适用。

### (iv) 漂移与脚滑

- 主口径：横轴为每米行程脚滑（cm/m），纵轴为每米行程漂移（cm/m）。n = 14，Pearson r = 0.496（p = 0.071），Spearman ρ = 0.437（p = 0.118）。
- 补充口径：横轴为 S（m），纵轴为 cm/s。n = 14，Pearson r = −0.407（p = 0.149），Spearman ρ = −0.305（p = 0.288）。
- 没有段因参考行程 < 0.10 m 被排除。所有点都画了：`drift/scatter_drift_vs_slip_per_m.png`、`drift/scatter_drift_rate_vs_slip_total.png`；数值在 `drift/correlations.json`。

### (v) 接入问题排查

`drift/checks_v.json`：`integration_problem_found: false`。

- **方向：** 14 段都没有被标记。参考位移 ≥ 5 cm 的秒里，cos < 0 的秒数最多是 kick_agent1 的 2/5。
- **帧率：** 参考 50 fps，控制 dt 0.02 s，一致。
- **坐标系：** 第 0 帧 pelvis 水平差 0.0 m，yaw 差 ≤ 1.4e-5°。

### (vi) 为新版方案 B 摸底（只调查，未实现）

**运动学规划器（planner ONNX）**

- **是否已发布：** `planner_sonic.onnx` 在 HF `nvidia/GEAR-SONIC` 里。
  - revision `6733128a`，与 T01 所用 SONIC 权重是同一个 revision；查询时 HF main 也还是这个 revision。
  - 773,952,989 字节，sha256 `39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea`，与 HF 的 LFS 记录一致。
  - GitHub 仓库 `1e1ec4f5` 里没有这个文件，也没有它的 LFS 指针；部署文档用 HF 下载脚本取。
  - 已下载到 `/home/kevin/third_party/GR00T-WholeBodyControl/sonic_hf_extra/`，不进 git。许可与 SONIC 权重相同（NVIDIA Open Model License），按 CLAUDE.md 不提交、不转发。
- **Python 调用：** `sonic_s1` 环境里的 onnxruntime 1.27.0 只有 CPU 和 Azure 后端，没有 CUDA。
  - 单条调用成功：首次调用 0.03 s；输出 `mujoco_qpos` 形状 [1, 64, 36]，`num_pred_frames` 为 44。
- **批量调用：** 11 个输入的第 0 维全部固定为 1。batch = 2 和 8 都被拒绝（`Got: 2 Expected: 1`）。
  - 结论：批量只能逐环境循环调用。改模型的 batch 维没有试。
- **接口（官方文档 `docs/source/references/planner_onnx.md`）：**
  - 输入：
    - 4 帧 MuJoCo qpos 上下文：根位置、wxyz 四元数、29 个关节（MuJoCo 顺序），世界坐标，Z 向上；
    - mode、移动方向、朝向、目标速度、高度；
    - 可选 4 个根部路点 `specific_target_positions` / `specific_target_headings`。`has_specific_target = 1` 时，用它们覆盖弹簧模型的目标根位置和朝向。
  - 输出：世界坐标的 qpos 序列，30 Hz，部署端重采样到 50 Hz。预测长度 24–64 帧，由 `allowed_pred_num_tokens` 控制。
- **探测记录：** `planner_probe.json`；脚本在 `logs_system1/T01b/planner_probe/probe.py`。探测只用了合成的站立上下文，只检查接口，没有检查输出质量。

**teleop 编码器**

- **配置：** `gear_sonic/config/actor_critic/encoders/teleop_mlp.yaml`，四个输入：
  - `command_multi_future_lower_body`：参考动作中下半身 12 个关节（MuJoCo 顺序的前 12 个，即两条腿）在多个未来帧的位置和速度，拼接成一个向量。关节空间，不涉及坐标系。
  - `vr_3point_local_target`：3 个点的位置。3 个点是 `left_wrist_yaw_link`、`right_wrist_yaw_link`、`torso_link`（见 `motion.yaml` 的 `vr_3point_body`）。位置是相对参考动作根锚点的位移，表达在参考锚点的坐标系里；代码注释说明这里的 local 指参考动作的根，不是机器人的根。只用当前一帧。
  - `vr_3point_local_orn_target`：同样 3 个点的姿态四元数，相对参考锚点姿态。
  - `motion_anchor_ori_b`：参考锚点相对机器人锚点的姿态，在机器人坐标系里，用 6D 表示。
- **训练时：** 四项都由参考动作（motion_lib）和机器人状态算出。
- **部署时（`docs/source/tutorials/vr_wholebody_teleop.md`）：**
  - 在 VR_3PT 模式下，移动由 planner 生成；上半身跟 PICO 头显和两只手柄的 3 点，需要先标定。
  - 文档写的是"head + 2 hands"，训练配置里第三个点是 `torso_link`。两者怎么对应，没有查到明确说明。

## 8. 方案 A 正式评测

### 设置

- **协议：** 与 T01 第 4 步相同：每个任务 50 个冻结初始状态，±2 和 ±8 两档，另加 nominal 一个回合；确定性动作，无观测噪声。
- **运行环境：** GPU PhysX，TF32 关。
- **SONIC：** release checkpoint，sha256 `e6bdab3f…ded8909`；spec sha256 `6c7d13e3…20640a`。
- **低层控制：**
  - hip_pitch 的 Kp 为 99.098、Kd 为 6.309；动作缩放冻结为论文值，hip_pitch 是 0.54755。
  - PD 直接接收 q*：1836 个回合里，PD 目标与 q* 的最大差全部为 0。
- **初始状态：** 与论文 D 评测用的是同一组文件，sha256 记在 `episodes.csv` 的 `state_file_sha256` 列。
- **终止条件：** 与 V3 统一评测相同（`JointBadTrackingZOnly`）。满足下列任一项即终止：
  - 机器人参考 body 的重力 z 分量误差 > 0.8；
  - 物体位置偏离参考 > 0.25 m；
  - 物体姿态偏离参考 > 0.8 rad。
- **指标口径：**
  - 物体误差、手部误差的口径与 V3 收尾包的 `scripts/finalize_v3_results_20261104.py` 相同。手部误差用同一套 URDF FK。Push 的参考与 V3 的 `push_reference_for_fk.npz` 逐位相同。
  - 掌面误差直接用 V3 的 `v3_palm_postprocess.py`，只改了输入和输出目录。

### 主设置（逐状态与 D 配对）

D 是 V3 收尾包 `episodes.csv` 的 `Full` 组：论文主结果的三个种子，同一组初始状态文件，sha256 已逐项核对一致。按 README_zh 预登记的口径，每个种子做一次精确双侧 McNemar 检验。

| 任务 | 档 | A 完成 | A 平均进度 | 失败回合终止帧：物体位置 > 0.25 m / 物体姿态 > 0.8 rad / 两项都未超 | SONIC 判摔倒 | SONIC 全局 MPJPE (mm) | 相对骨盆 (mm) | D 三种子完成率 均值 ± SD (%) | McNemar p（721 / 722 / 723） |
|---|---|---|---|---|---|---|---|---|---|
| push | ±2 | 0/50 | 0.336 | 50 / 0 / 0 | 0 | 147.5 | 45.8 | 78.0 ± 16.4 | 1.5e-11 / 7.1e-15 / 4.7e-10 |
| push | ±8 | 0/50 | 0.321 | 49 / 0 / 1 | 1 | 160.3 | 48.2 | 77.3 ± 14.5 | 1.2e-10 / 1.4e-14 / 5.8e-11 |
| pull | ±2 | 0/50 | 0.460 | 50 / 0 / 0 | 5 | 123.5 | 39.5 | 99.3 ± 1.2 | 1.8e-15 / 1.8e-15 / 3.6e-15 |
| pull | ±8 | 0/50 | 0.477 | 50 / 0 / 0 | 3 | 145.7 | 39.2 | 76.0 ± 11.1 | 2.3e-13 / 4.7e-10 / 3.6e-12 |
| kick | ±2 | 0/50 | 0.721 | 50 / 0 / 0 | 0 | 201.4 | 32.8 | 100.0 ± 0.0 | 1.8e-15 / 1.8e-15 / 1.8e-15 |
| kick | ±8 | 1/50 | 0.726 | 49 / 0 / 0 | 1 | 212.0 | 33.3 | 98.0 ± 2.0 | 7.1e-15 / 3.6e-15 / 1.4e-14 |
| chair | ±2 | 0/50 | 0.647 | 0 / 50 / 0 | 0 | 67.7 | 44.7 | 95.3 ± 6.4 | 1.1e-13 / 3.6e-15 / 1.8e-15 |
| chair | ±8 | 0/50 | 0.646 | 7 / 46 / 0 | 0 | 91.1 | 45.2 | 82.0 ± 18.0 | 4.7e-10 / 9.1e-13 / 1.8e-15 |
| bucket | ±2 | 0/50 | 0.480 | 23 / 28 / 0 | 0 | 89.9 | 40.9 | 84.7 ± 18.6 | 1.8e-15 / 4.7e-10 / 5.7e-14 |
| bucket | ±8 | 0/50 | 0.436 | 25 / 25 / 0 | 0 | 86.3 | 39.1 | 61.3 ± 12.9 | 7.3e-12 / 3.0e-08 / 7.5e-09 |
| table | ±2 | 0/50 | 0.386 | 50 / 0 / 0 | 0 | 99.6 | 36.2 | 86.7 ± 14.7 | 5.8e-11 / 3.6e-15 / 2.8e-14 |
| table | ±8 | 0/50 | 0.382 | 50 / 0 / 0 | 0 | 114.1 | 36.2 | 84.0 ± 6.9 | 2.8e-14 / 1.8e-12 / 1.8e-12 |

- **完成情况：** 600 个扰动回合里只有 1 个完成，是 Kick ±8 的 `perturbed_048`。
- **两项物体误差：** 一个回合可能同时超两项（Chair ±8、Bin），所以这两列加起来可以超过失败数。按定义，"两项都未超"的回合是由机器人姿态项触发的终止。
- **SONIC 摔倒列与平均值：**
  - "SONIC 判摔倒"按 SONIC 自己的评测判据统计，与回合终止是两回事。
  - SONIC MPJPE 是 50 个回合各自的均值再取平均（两台机器人都算），每个回合只算到它终止为止。
- **手部和掌面误差：** 只对完成的扰动回合计算（V3 口径）。
  - Kick ±8 那 1 个完成回合的手部误差是 28.25 cm。
  - Chair、Bin、Table 没有完成回合，所以没有掌面误差。
  - Push、Pull 的掌面目标在 V3 中就缺失（missing），Kick 不适用。
- **Kick 末帧：** Kick 的失败回合最晚在第 218 步终止，都没有到窗口开始包含末帧的第 252 步。完成的 Kick 回合（主设置 1 个、敏感性组 2 个）都在第 297 步被 SONIC 判为摔倒，这一步就是参考末帧跳变；V3 终止条件没有被触发。

### nominal（单独列出）

| 任务 | 完成 | 进度 | 执行步数 / 参考步数 | SONIC 判摔倒 |
|---|---|---|---|---|
| push | 否 | 0.282 | 87/309 | 否 |
| pull | 否 | 0.494 | 156/316 | 否 |
| kick | 否 | 0.718 | 214/298 | 否 |
| chair | 否 | 0.647 | 253/391 | 否 |
| bucket | 否 | 0.439 | 218/497 | 否 |
| table | 否 | 0.386 | 265/687 | 否 |

- ±2 和 ±8 两个目录里的 nominal 是同一个初始状态，各跑了一次。轨迹逐位相同，见 `schemeA/nominal_pm2_vs_pm8.csv`（18 对全部相同，含 T04）。
- 视频：每个任务导出一段 nominal，没有挑回合。存放在 `exports_system1/T01b/schemeA_nominal/`，不进 git；sha256 见 `schemeA/videos_sha256.json`。
  - 视频是 MuJoCo 运动学回放，不是物理仿真：蓝色和橙色是两台机器人，灰色是物体。

### hip_pitch 敏感性（预登记）

- **改动：** 只改两侧 hip_pitch（下标 0、6）的执行器参数：力矩上限 88 → 139，速度上限 32 → 20，armature 0.01018 → 0.02510。
- **不变的部分：** Kp、Kd 和动作缩放与主设置相同。逐字段 diff 见 `schemeA/manifest_diffs_vs_main.json`。
- **配对：** 下表最后一列按初始状态与主设置配对，用精确双侧检验。

| 任务 | 档 | 主设置：完成 / 平均进度 / SONIC 判摔倒 | SONIC hip_pitch：完成 / 平均进度 / SONIC 判摔倒 | 配对：仅主设置完成 / 仅敏感性完成 / 精确 p |
|---|---|---|---|---|
| push | ±2 | 0/50 / 0.336 / 0 | 0/50 / 0.332 / 0 | 0 / 0 / 1.00 |
| push | ±8 | 0/50 / 0.321 / 1 | 0/50 / 0.318 / 0 | 0 / 0 / 1.00 |
| pull | ±2 | 0/50 / 0.460 / 5 | 0/50 / 0.428 / 8 | 0 / 0 / 1.00 |
| pull | ±8 | 0/50 / 0.477 / 3 | 0/50 / 0.467 / 5 | 0 / 0 / 1.00 |
| kick | ±2 | 0/50 / 0.721 / 0 | 2/50 / 0.732 / 2 | 0 / 2 / 0.50 |
| kick | ±8 | 1/50 / 0.726 / 1 | 0/50 / 0.721 / 0 | 1 / 0 / 1.00 |

## 9. 第二阶段文件

| 文件 | 内容 |
|---|---|
| `schemeA/episodes.csv` | 主设置 + 敏感性 18 组 × 51 回合 = 918 行。前 36 列与 V3 `episodes.csv` 同名同序，其后 22 列为新增，见下 |
| `schemeA/per_task_summary.csv` | 每组 50 个扰动回合的汇总，以及 nominal 的各项 |
| `schemeA/schemeA_vs_D_mcnemar.csv` | A 与 D（V3 Full，三个种子）逐状态配对 |
| `schemeA/hip_pitch_sensitivity_paired.csv` | 敏感性组与主设置逐状态配对 |
| `schemeA/nominal_pm2_vs_pm8.csv` | 两个目录中 nominal 的一致性 |
| `schemeA/run_configs.json` | 36 组评测的完整命令、起止时间、git HEAD 与 status、初始状态文件及 sha256、PD/动作缩放/力矩/速度/armature、质量读回 |
| `schemeA/manifest_diffs_vs_main.json` | 敏感性组和 T04 各组的 run_manifest 与同任务同档主设置逐字段 diff |
| `schemeA/provenance.json` | 机器人 URDF、SONIC checkpoint/spec、初始状态、参考文件的 sha256；手部 FK 自检；评测代码 sha256；V3 表的 sha256 |
| `schemeA/videos_sha256.json` | 6 段 nominal 视频 |
| `planner_probe.json` | (vi) planner ONNX 接口探测 |
| `regression_evaluator_readback_flag.json` | 评测脚本加读回开关后的 D 回归检查（见第 10 节） |

`episodes.csv` 新增列：

- 方案与档位：`scheme`；`mass_kg`（从 PhysX 读回）；`original_mass_kg`；`mass_ratio`；`retrained`（A 不训练，填 `not_applicable_no_training`）；`hip_pitch_actuator`；`tf32`。
- 每台机器人的 SONIC 跟踪：`sonic_mpjpe_mm_r0/r1`、`sonic_mpjpe_pelvis_rel_mm_r0/r1`、`sonic_fall_step_r0/r1`、`sonic_fell_any`、`sonic_pelvis_dxy_final_m_r0/r1`。
- 其他：`height_bias_cm`；`pd_target_minus_qstar_max_abs`；`launch_git_head`；`run_manifest_sha256`；`terminal_object_pos_error_m` 和 `terminal_object_ori_error_rad`（最后保存的状态，即重置前的终止状态，与参考的差）。
- 沿用列中改了填法的：
  - `training_seed` 和 `checkpoint_iteration` 为空，因为 A 不训练；
  - `checkpoint_sha256` 填 SONIC checkpoint；
  - `config_sha256` 填该组 `run_manifest.json` 的 sha256（V3 这一列是空的）。

## 10. 第二阶段与任务书的偏离和补充

1. **主设置 push ±2 启动时有未提交改动，diff 没有保存。**
   - 这一组 03:09:42 启动，HEAD 是 `ae5fa59`。其余 35 组启动时 HEAD 是 `fbb39f6`，没有未提交改动。
   - 当时的 git status 只列出 7 个修改过的文件：`system1/cloud/` 下 3 个、`scheme_c.py`、`f1_dual_online.py`、`t01b_pipeline.sh`、`t04/diff_run_config.py`。
   - 这些文件都不在方案 A 评测路径上：`scheme_c.py` 只在 `--controller sonic_residual` 时导入。
   - 评测路径上的 `scripts/evaluate_paper_supplement.py`、`system1/scheme_a.py`、`system1/sonic_controller.py` 在 `ae5fa59` 和 `fbb39f6` 中的 sha256 相同。
   - 已改为在启动记录里保存 `git diff HEAD`：`scheme_a_queue.py` 和云端 `run_train_queue.py` 都改了。
2. **T04 要求读回惯量，评测时只读回了质量。**
   - 评测脚本原来只读回质量和摩擦。现在补上了惯量和质心读回，并加了 `--readback-only` 开关，默认关闭。
   - 用同一评测脚本、同一控制器、同一资产路径，把 12 个质量档从 PhysX 读回了一遍，结果在 `results_system1/T04/A/object_physics_readback.csv`。
   - 评测没有重跑。
   - **回归检查：** 开关关闭时，D Pull 721 ±2 前 3 个回合与 F2 回归的结果比较，回合字段 0 处不同，轨迹最大差 0（`regression_evaluator_readback_flag.json`）。
3. **渲染脚本第一次跑时 Chair、Bin 失败：** 物体网格的相对路径解析错了。`render_traj.py` 改成先取绝对路径，重跑成功。失败日志保存为 `logs_system1/T01b/render/*.attempt1.log`。
4. **(vi) 下载了 planner ONNX**，见第 7 节 (vi)。
