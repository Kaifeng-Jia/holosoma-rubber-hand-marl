# T14 一前一后训练任务：参考截图（开训前给 Kevin 看）

- **依据：**Kevin 2026-11-26 决定（"T14 第 9g2 步之后"）第二项第 1、2、6 条。本文件 2026-11-27 00:25（纽约）写入，后续阶段会在这里补。
- **参考：**sub8_suitcase_020，1.0 倍。直接取 9g2 已有的求解结果，没有重解：`logs_system1/t14/step9g2/sub8_suitcase_020/m1p00`，判定 C 适用。
  - 构造代码冻结在 `b170599`，即 9g2 运行时的 commit。`frontback.py`、`role_solve.py`、`partner.py`、`step9_eval.py`、`step9_gates.py` 从那以后没有改动，sha256 记在 `source_manifest.json`。
  - 30 Hz 双机参考有 60 帧（OMOMO 序列帧 103–162），原样复制。
  - 用 Table 参考当时用的转换脚本 `scripts/synthesize_core4d_smalltable_runtime_reference.py` 转成 50 Hz，得到 99 帧（1.96 s）。在时间对齐的帧上，关节、根、物体和 30 Hz 原值的差都是 0。
- **箱子：**构造里 1.0 倍的加长箱子，0.192 × 0.562 × 0.376 m，单个 box 碰撞体，原点在箱子中心。
  - 质量 5 kg，惯量按均匀密度箱子算。
  - 摩擦和恢复系数取 Table Transport 的（0.5, 0.5, 0），和桌子一样通过实验配置设置，不写在 URDF 里。
- **资产：**`src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14fb/table/`（清单 `t14_reference.json`），生成脚本 `system1/t14/frontback_train/prepare_reference.py`。
- **截图：**`reference/frontback_reference_sidebyside.jpg`，脚本 `system1/t14/frontback_train/render_reference.py`。
  - 画的是 50 Hz 训练参考本身。
  - 三个时刻按事先写好的规则取：
    - 抱起 = 箱子高度第一次比第 0 帧高 2 cm，参考帧 2；
    - 走路中段 = 两台骨盆的平均水平位移第一次达到全程一半，参考帧 55；
    - 放下 = 箱子高度最后一次比末帧高 2 cm，参考帧 96。
- **视频：**Drive `DuetMimic_transfer/T14_frontback/frontback_reference_sub8_suitcase_020_m1p00.mp4`，左边是人，右边是两台机器人，50 Hz 全程。清单见 `drive_manifest.csv`。
- **参考本身的一个事实：**箱子在第 0 帧离地 4.5 cm，最后一帧离地 6.2 cm（箱底最低点，箱子水平没有倾斜），整段参考里箱子都没有着地。
  - 原因是 OMOMO 的这个片段从"离地"开始、到"落地"之前结束：前后紧挨着的 96–102 帧、163–170 帧是 floor 段。
  - 这一点没有改。

## 本机冒烟：通过，可以开卡（2026-11-27 01:58）

- **结论：**4 项都通过：三组仿真内检查、200 轮训练加验证、T10 门槛、MuJoCo 闭环跑通。云端 d_frontback 可以开训。
- **时间（纽约，2026-11-27）：**

  | 时间 | 事项 |
  |---|---|
  | 00:35:42–00:36:56 | 三组仿真内检查（8 个环境）：噪声关两次、噪声开一次 |
  | 00:36:56–00:55:20 | 200 轮训练，命令和云端 d_frontback 相同，只有环境数是 2048（5070 只有 8 GB，放不下 4096）；之后在第 200 轮上跑云端的验证命令 |
  | 00:55–01:54 | 等内存：脚本给 Isaac ±2/±8 两档同时跑设的门槛是可用内存 ≥ 22 GB，本机一直只有 21 GB。01:54 按记下的 PID 停掉这个脚本，改成两档依次跑、每档门槛 17 GB（`run_smoke_part2.sh`），命令不变 |
  | 01:54:59–01:57:41 | Isaac ±2、±8 finals → Isaac 场景导出 → actor ONNX → T10 门槛 → MuJoCo ±2 |

- **1. 仿真内检查**（`smoke/isaac_checks.csv`，每项一行）：
  - 观测 v1 和定义逐项比较：参考朝向最大差 3.5e-7（噪声开 2.8e-7）；
  - 接触三项回放：奖励最大差 1.8e-6，d 最大差 5.1e-6，计数、失接触、终止都没有不一致；
  - 噪声关的两次结果完全相同。
- **2. 200 轮训练**（`smoke/smoke_train.json`）：
  - passed，rc 0，没有非有限值；
  - 每轮 4.95 s（2048 个环境，5070）；
  - 云端验证命令跑完 51 个回合，非标称回合完成 0 个（200 轮的策略）。
- **3. T10 门槛**（`smoke/t10_gate.json`，用 200 轮 checkpoint）：
  - 初始状态：102 个（±2、±8 各 51），MuJoCo 和 Isaac 比，关节最大差 0，根位置 3.0e-8 m，物体 0，都 ≤ 1e-4；
  - 观测 v1：Isaac 标称回合（18 步后因关节跟踪差终止）的每一步逐维比较，最大差 7.3e-7 ≤ 1e-4；
  - 正运动学 4.0e-7 m；ONNX 和 Isaac 动作最大差 1.9e-6。
- **4. Isaac finals 和 MuJoCo ±2**（`smoke/isaac_finals.csv`、`smoke/mujoco_pm2.csv`）：只看流水线能不能跑通，200 轮策略的成绩没有意义。
  - Isaac ±2、±8 各 51 个回合，complete；
  - MuJoCo ±2 51 个回合，complete；
  - 所有回合都在 9–30 步因关节跟踪差终止，没有完成的回合。
- **和 d_table 的不同：**
  - 训练命令只有两个参数不同：参考清单（箱子 URDF 随清单换）和接触标签；
  - run 计划另有两处不同：固定预算（不用停止规则），以及验证 / 测试状态文件的路径（数值和 Table 相同）；
  - 逐项对比见 `48bd064` 的 dry-run。
- **脚本和记录：**
  - 脚本：`system1/t14/frontback_train/run_smoke.sh`、`run_smoke_part2.sh`；打包脚本 `package_smoke.py`；
  - 记录：`smoke/records/`。

## 测试状态第 0 帧穿透（Kevin 2026-11-27 第 5 条，只报告）

- **训练开局不读 ±2 / ±8 状态文件：**
  - `train_core4d_smalltable.py` 没有 `--states` 参数；开局取参考里随机一帧，加 T10b 随机化。
  - 训练中每 1000 轮的验证读 `states/table_validation_states.json`（另一个种子的 ±2 状态），只画曲线。
  - ±2 / ±8 只在训练结束后的评测里用。
- **量法：**和 `make_states_t14.py` 的第 0 帧检查相同：仿真碰撞形状，MuJoCo `mj_geomDistance`，所有连杆（含手）。这次多记了最深的连杆和状态。
- **结果**（`states_frame0/frame0_links.csv`，每个文件一行；数和 `make_states_t14.py` 当时的输出相同）：

  | 任务 | 文件 | 最大深度 | 哪台、哪个连杆、哪个状态 | 有连杆在物体里的状态数 / 51 | 深于 1 cm 的状态数 |
  |---|---|---|---|---|---|
  | T13 Table | `system1/t13v4/states_t14r6C/table_pm2_states.json` | 0 | — | 0 | 0 |
  | T13 Table | `system1/t13v4/states_t14r6C/table_pm8_states.json` | 4.51 cm | r1 right_hip_yaw_link，perturbed_010 | 14 | 12 |
  | T13 Push | `system1/t13v4/states_t14r6C/push_pm2_states.json` | 0.15 cm | r1 right_rubber_hand_link，perturbed_013 | 51 | 0 |
  | T13 Push | `system1/t13v4/states_t14r6C/push_pm8_states.json` | 0.30 cm | r1 right_rubber_hand_link，perturbed_013 | 50 | 0 |
  | 一前一后 | `system1/t14/frontback_train/states/table_pm2_states.json` | 3.43 cm | r1 right_rubber_hand_link，perturbed_010 | 51 | 38 |
  | 一前一后 | `system1/t14/frontback_train/states/table_pm8_states.json` | 13.29 cm | r1 right_rubber_hand_link，perturbed_010 | 51 | 48 |

- **说明：**
  - Push 和一前一后的标称状态（不扰动）手本来就在物体里 0.1 cm，所以几乎每个状态都算"在物体里"；
  - Table ±8 最深的是髋，不是手。
- **脚本：**`system1/t14/frontback_train/states_frame0_links.py`（sonic_s1），另有每个状态每台机器人的明细表 `frame0_links_per_state.csv`。
