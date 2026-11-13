# T06 第 1 步（CPU 部分）：手和身体的初始穿插（2026-11-13，CPU）

依据：`docs/system1/T06_dexterous_hand_zh.md` 第 1 步的"初始无穿插"。GPU 部分（在 Isaac 里加载、站立 5 秒、读整机质量和关节限位）还没做：按 Kevin 的指令，要等本机最终评测全部结束。

## 做了什么

- 模型：合成的 G1 + Sharpa URDF（`system1/hand/assets/g1_29dof_sharpa.urdf`），只用碰撞网格。
- 方法：用 yourdfpy 摆姿态，用 FCL（trimesh CollisionManager）检测每个 Sharpa 碰撞网格和每个 G1 碰撞网格之间的接触，记录接触对和最深穿插。
- 不算的接触对：
  - flange 和 `<side>_wrist_yaw_link` 是设计上的安装面，单独列出；
  - 同一只手内部的接触对不测，官方 USD 里也过滤掉了。
- 姿态：
  - 身体关节全为 0；
  - Bin 参考第 0 帧（每个人一次），身体关节用第 2 步 Sharpa 腕部重解之后的值，手指张开，即 locked 参考第 0 帧。
- 用时：几秒。

命令（只用 CPU 0–15 号核）：

    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/self_collision_check.py --out results_system1/T06/step1/self_collision.json

## 结果

| 姿态 | 手–身体接触对 | 最深 | 最深的一对 |
|---|---|---|---|
| 身体关节全为 0 | 0 | 0 | — |
| Bin 参考第 0 帧，第 1 个人 | 16 | 1.52 cm | 左手中指近节（left_middle_PP）对左髋 yaw 连杆 |
| Bin 参考第 0 帧，第 2 个人 | 4 | 1.80 cm | 左手腕座（left_hand_wrist）对左髋 roll 连杆 |

- 第 1 个人的其他较深接触对：左手掌（left_hand_C_MC）对左髋 roll 连杆 1.06 cm，左手食指近节对左髋 yaw 连杆 1.01 cm。
- 安装面（flange 对 wrist_yaw_link）：所有姿态下穿插都是 0。
- 现在 D 的配置关闭了机器人自碰撞（`enable_self_collisions=False`），所以仿真里这些重叠不会产生力。
- 任务书门槛是"初始无穿插"：零姿态满足；参考初始姿态下手和大腿重叠超过 1 cm，这一项按事实报告，不算通过。

## 更正

11/13 早上回报里，第 1 个人"最深 1.06 cm，左手手掌对左大腿"写错了。1.06 cm 只是手掌那一对；所有接触对里最深的是 1.52 cm，左手中指近节对左髋 yaw 连杆。

## 文件

- `self_collision.json`：每个姿态的全部接触对和深度。
- `SHA256SUMS`。
