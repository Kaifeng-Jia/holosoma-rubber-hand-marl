# T03 C 第二版：停止第一版、实现、开训前检查、开训

本文件只写事实。时间一律为 America/New_York。

- **依据：**
  - 任务书 `docs/system1/T03_schemeC_marl_sonic_zh.md` 的"C 第二版"一节，以 `f32cded` 为准，文件 sha256 `43083cbe2a15911c36e3561411b2400f43b22e682b2fce0dfab045196839917d`；
  - Kevin 转交的执行消息，2026-11-11 14:41 收到。
- **机器：**
  - 云端：5 × RTX 4090，跑训练、冒烟测试、第 0 轮对比和本轮评测；
  - 本机：kevin-Legion-R9000P-ADR10，RTX 5070 Laptop，跑复现核对。

## 1. 停止 C 第一版（14:43:35）

- **范围：** 只停进程，没有删任何文件。
  - 停了 8 个还在训练的 run；`c_pull_m20kg_seed721_joint8000` 已在 14:39 前训完到第 8050 轮，不用停。
  - D 的 6 个 run 没有动：5 个在训练，`d_pull_m40kg` 已训完。
- **做法：** 脚本 `system1/cloud/stop_runs.py`。
  - 对训练子进程发 SIGTERM，60 秒内没退出的再发 SIGKILL；
  - 队列进程随后自己写出 `exit.json`，completed = false；
  - 每个 run 的记录在 `v1_stopped/<label>.stopped.json`。

| run | 最后记录的轮数 | 停止时最新的 checkpoint | sha256（前 16 位） | 结束方式 |
|---|---|---|---|---|
| c_table_m5kg（主实验） | 8102 | model_08000.pt | 183e76560473e6f9 | SIGKILL |
| c_bucket_m1kg（主实验） | 7743 | model_07000.pt | 71daa4b8e210a9f6 | SIGKILL |
| c_pull_m40kg（质量扫描） | 6864 | model_06050.pt | 0f1e9ae63de16a02 | SIGTERM |
| c_table_m10kg | 7101 | model_07000.pt | bdcb7c48279afc6f | SIGKILL |
| c_table_m20kg | 6849 | model_06000.pt | 0fe4b8df0d656f92 | SIGKILL |
| c_bucket_m2kg | 6376 | model_06000.pt | 742f403b1547de66 | SIGKILL |
| c_bucket_m4kg | 6446 | model_06000.pt | e9a54171b0173a01 | SIGKILL |
| c_bucket_m8kg | 6405 | model_06000.pt | de78485800a41562 | SIGKILL |

## 2. 实现（提交 `15826ca`、`3e41687`、`9be580e`）

- **开关：** 训练脚本新增两个开关，评测脚本按 checkpoint 里记的种类构造 C。两个开关都默认关闭，关闭时第一版的命令和行为不变。
  - `--sonic-residual-kind {v1,in,out}`，默认 v1；
  - `--c-train-diag`。
- **C-in：**
  - 环境把 a 截断到 [−1, 1]，Δq = 论文 D 的动作缩放 × 截断后的 a；
  - 参考加 Δq 后加到 SONIC 窗口的每一帧，再截断到 URDF 硬限位（`dof_pos_lower/upper_limit_list`）；
  - "上一步动作"观测用截断后的 a；learner 存的、算对数概率用的，是未截断的 a；
  - 初始标准差 0.3，熵系数 0。
- **C-out：**
  - SONIC 收到未修改的参考；
  - q* = SONIC 输出 + 论文 D 的动作缩放 × a_c，a_c 是按 D 的动作项截断后的 a：`clip_actions = True`，`action_clip_value = 100`；
  - "上一步动作"观测用 C-out 自己的 a，与 D 相同；SONIC 自己的"上一步动作"仍是它自己的输出；
  - 动作变化惩罚按 q* 换成 D 的动作单位计算；
  - 初始标准差和熵系数与 D 相同，三个任务都是 1.0 和 0.005。
- **两种 C 共同：**
  - 输出层初始化为 0；
  - checkpoint 里记 `system1_sonic_residual_kind`，续训和评测时都核对。
- **训练日志诊断：** 由 `--c-train-diag` 打开，`system1/c_train_diag.py`，每 100 轮写一条 `system1_c_train_diag`。内容包括：
  - 修正量的均值、95 分位、最大值，以及最后一轮的逐关节值；
  - |a| > 1 的比例和实际截断比例；
  - 动作标准差；
  - SONIC 摔倒判定按三条分别的触发率；
  - 结束回合中超时和终止各多少，终止的再按物体位置、物体姿态、机器人躯干分开；
  - C-in 的 Δq/scale 上界、参考越限量，C-out 的 q* 核对。
- **每个关节的上限 scale_i：** 见 `per_joint_scale_and_limits.csv`，同时列出关节限位。

  | 关节 | scale_i（rad） | 度 |
  |---|---|---|
  | hip_pitch、hip_yaw、waist_yaw | 0.5475 | 31.37 |
  | hip_roll、knee | 0.3507 | 20.09 |
  | ankle_pitch、ankle_roll、waist_roll、waist_pitch、shoulder ×3、elbow、wrist_roll | 0.4386 | 25.13 |
  | wrist_pitch、wrist_yaw | 0.0745 | 4.27 |

  - D 对照组的动作截断：holosoma 机器人配置 `clip_actions = True`、`action_clip_value = 100.0`，在 `joint_control` 动作项里截断，然后乘缩放。
- **CPU 冒烟测试：** `system1/tests/test_scheme_c_v2.py`，7 项全部通过。

## 3. 开训前检查（全部通过）

| 检查 | 结果 | 文件 |
|---|---|---|
| 冒烟训练：正式命令（`run_train_queue.py --preflight 10`），两种 C × 三个任务，10 轮 | 6/6 不报错。run_config 与 D 对照组逐字段比较，意外字段都是 0；C-in 的 Table/Bin 只有熵系数一项不同，归为"C-in 预登记" | `smoke_run_config_diffs/`、`smoke_metrics/` |
| Pull 会更新 actor 的训练段：从冒烟 critic 段的 checkpoint 续训 10 轮 | 两种 C 都不报错。动作标准差 C-in 0.299 → 0.295，C-out 0.990 → 0.928 | `smoke_metrics/smoke1_*_pull_joint_smoke` |
| 第 0 轮确定性 rollout 与方案 A 一致：TF32 关，同一组初始状态的前 3 个，云端 GPU 0 | C-in、C-out 在 Pull、Table、Bin 上逐数组最大差都是 0.0，回合字段没有差别 | `iter0/iter0_{cin,cout}_vs_A.json` |
| C-in：本机复现脚本核对 \|Δq_i\| ≤ scale_i、交给 SONIC 的参考在限位内 | 三个任务的 max \|Δq_i\|/scale_i 都是 1.0，越限量 0；SONIC 收到的窗口等于截断后的"参考 + Δq"，最大差 0 | `replay/` |
| C-out：q* 与"SONIC 输出 + 缩放 × a"的最大差 | 三个任务都是 0；SONIC 收到的窗口等于未修改的参考，差 0 | `replay/` |
| 两种 C：PD 目标 − q* | 0 | `replay/` |

- **复现核对的设置：**
  - 本机 RTX 5070 Laptop，每次 256 个环境 × 240 步，随机采样；
  - checkpoint：Pull 用训练段冒烟的 `model_00010.pt`，Table、Bin 用冒烟的 `model_00000.pt`；
  - 6 次运行在 15:15–15:20。
- **参考是否在限位内：** 另外离线检查过，Pull、Table、Bin 的参考关节角，在全部帧上都在 URDF 限位以内。所以第 0 轮 C-in 的截断不起作用。

## 4. 偏离和补充

1. **诊断计数的修正：** 第一次冒烟时，诊断里"终止"的计数一直是 0。
   - 原因：终止管理器的标记在同一步的重置中被清零了。
   - 改为和训练脚本同一口径：超时取 `extras["time_outs"]`，其余结束的回合算终止（`3e41687`）。
   - 第二次冒烟中，每个窗口的终止数、超时数都和训练脚本自己记的失败数、超时数相等。
   - 第 0 轮对比和复现核对用的是第一次冒烟的 checkpoint；诊断代码不影响 checkpoint。
2. **第 0 轮对比在云端 RTX 4090 上做，** 和这次 C 的训练同一类卡；T01b 第一版的同一项检查是在本机做的。
3. **常数 Δq 检查的取法：** 两侧 `hip_pitch` 取负（屈髋），`waist_pitch` 取正（躯干前倾），三个关节同时取各自上限的 0.5 倍或 1.0 倍，其余关节为 0。
   - 也就是 hip −0.274 / −0.5475 rad，waist +0.219 / +0.4386 rad；
   - 另跑一组 Δq = 0 作为同卡对照。
   - "各取 0.5 倍和 1.0 倍"按"同时取"理解；
   - 评测脚本的开关是 `--sonic-const-dq`，只用于方案 A 路径，默认关闭。
4. **第一版停止时 checkpoint 的 ±2 评测：** 9 个都在云端 4090 上做，包括已训完的 C Pull 第 8050 轮。
   - 本机流水线也会按 T03 的协议，对 C Pull 第 8050 轮做 ±2 和 ±8 评测，GPU 是 RTX 5070 Laptop；
   - 其中 ±2 那一次 15:21 已经开始。

## 5. 开训（15:21，云端 git `0e0f164`，`system1/cloud/c2_launch.sh`）

- **6 个 run：** 种子 721，每卡最多 3 个计算进程。

  | run | GPU |
  |---|---|
  | cin_pull:20 | 0 |
  | cout_pull:20 | 0 |
  | cin_table:5 | 0 |
  | cout_table:5 | 1 |
  | cin_bucket:1 | 3 |
  | cout_bucket:1 | 4 |

- **空出的位置先做评测：** GPU 1、3、4 各一个位置，作业清单 `logs/evals/jobs.json` 共 12 个：
  - 第一版停止时的 checkpoint ±2，9 个；
  - 常数 Δq 检查，3 个。
- **调度器：** `system1/cloud/c2_scheduler.py`，自动执行任务书的两条规则，决定和数字写进云端 `logs/train/c2_decisions.json`：
  - 第 1000 轮提前停止；
  - 第 2000 轮选择 C 的种类，并开 C 质量扫描，耗时长的先开。
