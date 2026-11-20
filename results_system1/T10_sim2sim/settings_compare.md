# Isaac 与官方 SONIC 部署的设置对照（T10 第 0 步）

所有数字由 `system1/sim2sim/settings_compare.py` 从文件读出，没有手填。逐关节的全部数值见 `settings_compare_joints.csv`；关键差值见 `settings_compare_checks.json`。

## 来源

| 列 | 来源 |
|---|---|
| Isaac D / Isaac C-in | 本机 5070 最终评测时 PhysX 实际使用的值：评测脚本加 `--sim2sim-dump` 导出（`logs_system1/t10/isaac/{d,cin}_table/dump/scene.json`），sha256 见 SHA256SUMS |
| SONIC 训练环境 | `system1/sonic/sonic_release_spec.json`（T01 从 SONIC 自己的 Isaac Lab 环境导出，checkpoint `sonic_release/last.pt` e6bdab3f…） |
| 官方部署 | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`（按头文件里的公式计算；GR00T-WholeBodyControl 1e1ec4f） |
| 官方 MuJoCo sim2sim | `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`、`gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml` |

## 逐项对照

| 项目 | Isaac（我们训练和评测） | 官方 SONIC 部署 / sim2sim | 是否一致 |
|---|---|---|---|
| 策略频率 | 50 Hz（dt 0.02） | 控制线程 50 Hz（`control_dt_ = 0.02`）；输入线程 100 Hz，planner 10 Hz，电机指令 500 Hz | 一致 |
| 仿真频率 | 200 Hz（sim dt 0.005，每个策略步 4 个子步；PhysX TGS） | MuJoCo `SIMULATE_DT: 0.005`（200 Hz）；真机电机驱动器自己的伺服频率 | 仿真频率一致，求解器不同 |
| PD 的计算位置 | holosoma 每个 200 Hz 子步算 τ = kp(q* − q) − kd·q̇ 并按力矩上限截断，作为力矩施加；PhysX 关节驱动刚度、阻尼为 0 | 部署发 LowCmd（每个电机 q、dq=0、kp、kd、τff）；sim2sim 每个 200 Hz 步算 τ = τff + kp(q − q_now) + kd(dq − dq_now)，按电机力矩上限截断；真机由电机驱动器执行 | 公式相同 |
| C-in 的 PD 增益 | SONIC 的增益（hip_pitch 用 SONIC 的值）；与部署端 kp 最大差 1.11e-06，kd 最大差 2.02e-07 | 部署：kp = 电枢 × (2π·10 Hz)²，kd = 2·2·电枢·ω；踝、腰 roll/pitch 加倍；hip_pitch 按 7520_22 | 一致 |
| C-in 的力矩上限 | 我们 URDF 的力矩上限（hip_pitch_actuator = ours）；与官方 sim2sim 电机上限最大差 51.0 Nm | sim2sim `motor_effort_limit_list` | 不一致的关节：left_hip_roll_joint, right_hip_roll_joint |
| D 的 PD 增益 | 论文 D 的增益（见 csv `isaac_D_kp/kd`），动作缩放 0.25 × 力矩上限 / kp（冻结） | 不适用（D 不经过 SONIC） | — |
| 关节电枢 | PhysX 逐关节（例如 hip_pitch 0.0102，knee 0.0251） | 官方 MJCF 统一 armature 0.01、damping 0.05、frictionloss 0.2（腕 0.1） | 不一致（本次 MuJoCo 按 Isaac 的值，见 README） |
| 关节阻尼、摩擦 | 0、0 | 见上一行 | 不一致 |
| 关节最大速度 | PhysX 每关节硬限制（例如 32 / 20 rad/s） | MuJoCo 没有关节速度限制 | MuJoCo 无对应项 |
| 关节顺序 | holosoma 顺序 = URDF 顺序（左腿、右腿、腰、左臂、右臂，即 MuJoCo 顺序） | 部署内部：电机按 MuJoCo 顺序；SONIC 观测和 ZMQ 流按 IsaacLab 顺序（`isaaclab_to_mujoco` 映射） | 我们给 SONIC 的输入按 IsaacLab 顺序（`sonic_controller.to_sonic`），与部署一致 |
| SONIC 的默认姿态 | SONIC 训练环境的默认姿态（我们 Isaac 里 SONIC 的目标 = 默认 + 缩放 × 动作）；与部署 `default_angles` 最大差 0.010 rad | 部署 `default_angles`（站立：hip_pitch −0.312、knee 0.669、ankle_pitch −0.363、肩 0.2 等） | 不一致 |
| SONIC 的动作缩放 | SONIC 训练环境 `action_scale`；与部署 `g1_action_scale` 最大差 1.89e-08 | 0.25 × 力矩上限 / kp | 一致 |
| 手 | 橡胶手（`main_mesh_collision_rubberhand.urdf`，固定在腕上，无自由度） | sim2sim 场景 43 自由度，带 Dex3 手（每只 7 自由度，`NUM_HAND_MOTORS: 7`）；部署可选 Dex3 手控制 | 不一致：真机手的配置要和博士生确认 |
| 地面、摩擦 | 地面材质 1.0/1.0，组合方式 multiply；机器人 1.0；物体 0.5/0.5，恢复系数 0 | 官方 sim2sim：geom 摩擦 1.0（MuJoCo 取最大值） | 本次 MuJoCo 按 Isaac 的组合值（见 README） |
| 碰撞 | 机器人：URDF 圆柱改胶囊、球、网格的 PhysX 凸包；Table 5 个盒子；Bin 凸分解 16 块；自碰撞开启 | 官方 MJCF 的碰撞几何 | 本次 MuJoCo 用 Isaac 的碰撞几何 |
| 编码器输入：参考关节位置、速度 | 未来 10 帧、每隔 5 帧（0.1 s），关节顺序 IsaacLab；先 290 个位置再 290 个速度（`command_multi_future_nonflat`） | `motion_joint_positions_10frame_step5` + `motion_joint_velocities_10frame_step5`（同样 10 帧 × 5） | 一致 |
| 编码器输入：参考朝向 | 参考骨盆朝向在机器人骨盆系下，旋转矩阵前两列（10 帧 × 5） | `motion_anchor_orientation_10frame_step5`；另加航向补偿：参考先乘以「机器人初始航向 × 参考初始航向的逆」再算 | 结构一致；部署多一个航向对齐（起始时把参考转到机器人航向） |
| 编码器模式字段 | `encoder_index` one-hot 3 维（g1 = [1,0,0]） | `encoder_mode_4` 4 维 = [模式号, 0, 0, 0]（g1 = 0） | 表示不同，官方 ONNX 与 last.pt 的 token 逐元素相同（见 consistency） |
| 解码器输入 | token 64 + 10 帧历史（角速度 3、关节位置−默认 29、关节速度 29、上一步动作 29、重力方向 3；最旧在前） | `token_state` + `his_*_10frame_step1`（同样的项和顺序） | 一致 |
| 我们的 158 维观测（每台机器人） | 按 holosoma 观测管理器的字母顺序拼接：上一步动作 29、骨盆角速度（骨盆系）3、关节位置−默认 29、关节速度 29、参考关节位置和速度 58（当前帧）、参考躯干朝向在实际躯干系下 6；同伴项：同伴根的平面相对位置和速度（观察者航向系，截断到 ±1）4 | 部署端没有这一层 | 上真机需要额外提供（见 deploy_interface_zh.md） |
| 输入接口 | Isaac 里直接把修正后的参考窗口交给 SONIC | 运动文件、键盘、手柄、ZMQ（v1：逐帧 29 关节位置和速度 + 根四元数 + 帧号；v3：再加 SMPL；v4：直接发 64 维 token，绕过编码器）、ROS2 | 结构上可接（见 README 第 0 步结论） |

## 逐关节摘要

| 关节 | Isaac D kp/kd | Isaac C-in kp/kd | 部署 kp/kd | Isaac C-in 力矩上限 | 官方 sim 力矩上限 | Isaac 电枢 | SONIC 默认 / 部署默认 |
|---|---|---|---|---|---|---|---|
| left_hip_pitch_joint | 40.18 / 2.558 | 99.10 / 6.309 | 99.10 / 6.309 | 88 | 88.00 | 0.0102 | -0.309 / -0.312 |
| left_hip_roll_joint | 99.10 / 6.309 | 99.10 / 6.309 | 99.10 / 6.309 | 139 | 88.00 | 0.0251 | -0.001 / 0.000 |
| left_hip_yaw_joint | 40.18 / 2.558 | 40.18 / 2.558 | 40.18 / 2.558 | 88 | 88.00 | 0.0102 | 0.003 / 0.000 |
| left_knee_joint | 99.10 / 6.309 | 99.10 / 6.309 | 99.10 / 6.309 | 139 | 139.00 | 0.0251 | 0.677 / 0.669 |
| left_ankle_pitch_joint | 28.50 / 1.814 | 28.50 / 1.814 | 28.50 / 1.814 | 50 | 50.00 | 0.0072 | -0.371 / -0.363 |
| left_ankle_roll_joint | 28.50 / 1.814 | 28.50 / 1.814 | 28.50 / 1.814 | 50 | 50.00 | 0.0072 | -0.007 / 0.000 |
| right_hip_pitch_joint | 40.18 / 2.558 | 99.10 / 6.309 | 99.10 / 6.309 | 88 | 88.00 | 0.0102 | -0.322 / -0.312 |
| right_hip_roll_joint | 99.10 / 6.309 | 99.10 / 6.309 | 99.10 / 6.309 | 139 | 88.00 | 0.0251 | 0.006 / 0.000 |
| right_hip_yaw_joint | 40.18 / 2.558 | 40.18 / 2.558 | 40.18 / 2.558 | 88 | 88.00 | 0.0102 | -0.001 / 0.000 |
| right_knee_joint | 99.10 / 6.309 | 99.10 / 6.309 | 99.10 / 6.309 | 139 | 139.00 | 0.0251 | 0.663 / 0.669 |
| right_ankle_pitch_joint | 28.50 / 1.814 | 28.50 / 1.814 | 28.50 / 1.814 | 50 | 50.00 | 0.0072 | -0.353 / -0.363 |
| right_ankle_roll_joint | 28.50 / 1.814 | 28.50 / 1.814 | 28.50 / 1.814 | 50 | 50.00 | 0.0072 | 0.004 / 0.000 |
| waist_yaw_joint | 40.18 / 2.558 | 40.18 / 2.558 | 40.18 / 2.558 | 88 | 88.00 | 0.0102 | -0.001 / 0.000 |
| waist_roll_joint | 28.50 / 1.814 | 28.50 / 1.814 | 28.50 / 1.814 | 50 | 50.00 | 0.0072 | -0.001 / 0.000 |
| waist_pitch_joint | 28.50 / 1.814 | 28.50 / 1.814 | 28.50 / 1.814 | 50 | 50.00 | 0.0072 | 0.003 / 0.000 |
| left_shoulder_pitch_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | 0.210 / 0.200 |
| left_shoulder_roll_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | 0.190 / 0.200 |
| left_shoulder_yaw_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | -0.000 / 0.000 |
| left_elbow_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | 0.594 / 0.600 |
| left_wrist_roll_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | -0.009 / 0.000 |
| left_wrist_pitch_joint | 16.78 / 1.068 | 16.78 / 1.068 | 16.78 / 1.068 | 5 | 5.00 | 0.0043 | -0.008 / 0.000 |
| left_wrist_yaw_joint | 16.78 / 1.068 | 16.78 / 1.068 | 16.78 / 1.068 | 5 | 5.00 | 0.0043 | 0.001 / 0.000 |
| right_shoulder_pitch_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | 0.197 / 0.200 |
| right_shoulder_roll_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | -0.205 / -0.200 |
| right_shoulder_yaw_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | 0.009 / 0.000 |
| right_elbow_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | 0.595 / 0.600 |
| right_wrist_roll_joint | 14.25 / 0.907 | 14.25 / 0.907 | 14.25 / 0.907 | 25 | 25.00 | 0.0036 | 0.005 / 0.000 |
| right_wrist_pitch_joint | 16.78 / 1.068 | 16.78 / 1.068 | 16.78 / 1.068 | 5 | 5.00 | 0.0043 | -0.005 / 0.000 |
| right_wrist_yaw_joint | 16.78 / 1.068 | 16.78 / 1.068 | 16.78 / 1.068 | 5 | 5.00 | 0.0043 | -0.000 / 0.000 |
