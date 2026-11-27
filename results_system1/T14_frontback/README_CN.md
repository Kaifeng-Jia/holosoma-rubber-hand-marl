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

## 掌心离箱面 4–6 cm 的来源（Kevin 2026-11-27 第 4 条，只报告，参考不改）

- **结论：**
  - 机器人 0 左手的 4.7 cm 一部分来自示范：人的左掌心本来就在箱外 1.4–2.5 cm，按哪个箱子量而不同（见下表）。其余是求解时拉开的：机器人比交互网格看到的人多 1.75 cm（逐帧差的中位数），比真实箱子上的人多约 3 cm。
  - 机器人 1 左手的 6.3 cm 全部来自求解：人的那只手在 +x 侧面上或略在面内，机器人的手却到了箱子 +y 端面之外。
- **怎么量的：**`palm_gap/`，脚本 `system1/t14/frontback_train/palm_gap_check.py`（hsretargeting_s1，只用 CPU，只读）。
  - **坐标和尺度：**求解器的世界和尺度（人按 smpl_scale 0.7155 缩放），30 Hz 双机参考，60 帧。
  - **正负号：**正 = 在箱外。
  - **人的掌心：**SMPL-X 掌侧 67 个顶点的均值，即求解器的 `L_Palm` / `R_Palm`，和交互网格、9g2 的 prediag 用的是同一个点。
  - **机器人的掌心：**和接触标签相同的橡胶手掌心点。这个点在手外壳掌侧面外 1.1 cm，所以掌心点在箱里时，手外壳不一定在箱里。
  - **人这边量了四个箱子：**
    - 交互网格看到的"抬平后的自己的箱子"；
    - 机器人的共享箱子 P（prediag 的定义）；
    - 人实际抱的箱子；
    - 采集到的 suitcase 网格。
    - 抬平的箱子和真实箱子最多差 25°。
  - **复现核对：**重建出来的人、P、双机参考和求解文件的差 ≤ 5e-9；prediag 和 eval.json 完全相同；50 Hz 下复现出原来的 4.70 / −0.35 / 6.33 / 4.60 cm。
- **中位数**（cm，60 帧；p95 和最值、分阶段的值在 `palm_gap_summary.json`）：

  | 手 | 人：抬平的箱子 | 人：真实箱子 | 人：suitcase 网格 | 机器人 | 机器人碰箱子的位置 |
  |---|---|---|---|---|---|
  | r0 左（示范者左） | 2.46 | 1.43 | 1.71 | 4.69 | 手外壳碰 −x 面（−0.10 cm），接触点在掌心点往指尖方向 6.7 cm |
  | r0 右（示范者右） | −3.15 | −0.72 | 0.73 | −0.35 | 碰 +x 面，掌面和面法线差 11.7° |
  | r1 左（镜像的示范者右） | −3.15 | −0.72 | 0.73 | 6.34 | 碰 **+y 端面**（60 帧都是），离 +x 面上的锚点 14.1 cm |
  | r1 右（镜像的示范者左） | 2.46 | 1.43 | 1.71 | 4.61 | 同 r0 左 |

- **r0 左手在求解里的情况：**
  - 交互网格给的目标点离箱面 2.46 cm，机器人掌心离这个点 2.36 cm；
  - 接触锚点在 −x 面上（权重 10，整段有效），机器人掌心离锚点 5.95 cm；
  - 掌面朝向跟住了人：人的掌面法线和箱面法线差 50.5°，机器人差 50.8°，和人的目标差 1.2°；
  - 所以手是用指尖那一端碰着箱子，手掌斜着，掌心点就离开箱面 4.7 cm。
  - 没有改求解项重跑，所以分不出这多出来的 1.75 cm 里锚点、交互网格、朝向、防穿透各占多少。
- **人的手本身：**人的手部顶点每一帧都在 suitcase 网格里 2.4–6.0 cm，所以接触标签的 should_contact 全程为真，但人的掌心点在箱外。

## 状态改为"未准备好"；各片段抓法（Kevin 2026-11-27 第二条决定第 2 条，只报告）

- **队列：**d_frontback 先不开训，状态是"未准备好"，排在队尾。顺序是 d_push_to10k → d_table_to10k → c_out_push_to10k → d_frontback（`system1/t13v4/queue_s9.py` 的 `KEVIN`，提交 `d4b54b9`，13:23 重启队列）。
  - 原因（Kevin）：机器人 1 的左手 60 帧都在箱子 +y 端面上，离 +x 侧面上的锚点 14.1 cm，抓法和人不一样。
  - 由 Kevin 定用哪条片段训练，再放回队列。
- **各片段抓法：**`grasp_faces/README_CN.md`，13 条片段 × 4 只手，量法和 `palm_gap_check.py` 相同。
  - sub8_suitcase_020 1.00 的逐帧结果和 `palm_gap/palm_gap.csv` 完全相同。
