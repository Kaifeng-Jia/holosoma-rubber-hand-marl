# Retarget Rubber Hand A.1

## 1. 目标

A.1 将包含完整手部信息的 InterMimic/OMOMO 演示动作转换为 G1 橡胶手动作。它的首要目标是保留演示中的手掌朝向和动作语义，而不是要求尺寸不同、无手指自由度的橡胶手网格与人手表面完全一致。

A.1 的最终阶段只修改以下六个关节：

- `left_wrist_roll_joint`
- `left_wrist_pitch_joint`
- `left_wrist_yaw_joint`
- `right_wrist_roll_joint`
- `right_wrist_pitch_joint`
- `right_wrist_yaw_joint`

机器人浮动基座、腿、腰、肩、肘以及物体位姿全部继承固定尺寸物体 baseline，不在 A.1 最终阶段改变。

## 2. 使用的橡胶手资产

Retargeting 使用：

```text
src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf
src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largetable.xml
```

训练侧使用：

```text
src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf
```

训练 URDF 中左右橡胶手的 visual 和 collision 都使用真实橡胶手网格：

```text
meshes/left_rubber_hand.STL
meshes/right_rubber_hand.STL
```

橡胶手通过 fixed joint 安装在 G1 的 `wrist_yaw_link` 上。它自身没有手指自由度；手部整体 roll、pitch 和 yaw 由 G1 的三个腕关节提供。

## 3. 原始 PT 数据中的手部信息

InterMimic PT 每帧至少包含 591 个数值。A.1 使用其中两类数据：

```text
[162:318]  52 个 SMPL-H 关节的世界坐标，reshape 为 (52, 3)
[383:591]  52 个 SMPL-H 关节的全局四元数，reshape 为 (52, 4)
```

全局四元数使用 `xyzw` 顺序。reshape 后：

```text
左腕：joint index 17
右腕：joint index 36
```

因此 A.1 不是只读取 `[383:591]` 中的两个标量，而是读取左右腕各自完整的四元数。

载入后首先执行：

1. 四元数单位化；
2. 沿时间轴修正 `q` 与 `-q` 的符号跳变；
3. 拒绝零范数四元数。

## 4. 从腕四元数恢复解剖掌面

PT 腕关节坐标系不应直接等同于机器人橡胶手坐标系。A.1 使用手部关键点构造解剖掌面，并标定一个整段动作恒定的 wrist-to-palm 旋转。

每只手使用：

```text
Wrist
Index1
Middle1
Pinky1
```

首先构造掌面正交基：

```text
forward = normalize(Middle1 - Wrist)
across  = normalize(Index1 - Pinky1)
across  = normalize(across - dot(across, forward) * forward)
normal  = normalize(cross(forward, across))
```

右手的 `across` 方向做镜像处理，使左右手的掌面法向具有一致的解剖语义。关键点掌面旋转记为：

```text
R_landmark = [forward, across, normal]
```

对整段序列估计恒定旋转：

```text
R_wrist_to_palm = mean(R_wrist^-1 * R_landmark)
```

最终逐帧目标掌面为：

```text
R_palm_target(t) = R_wrist(t) * R_wrist_to_palm
```

如果 wrist-to-palm 标定误差的 90 分位数超过配置阈值，pipeline 会停止，而不是静默生成错误动作。

## 5. 从人手掌面映射到机器人橡胶手

左右橡胶手 mesh 在各自 link 坐标系中的朝向不同。A.1 为每只手保存一个固定的 robot palm basis：

```text
R_robot_link_target(t) = R_palm_target(t) * R_robot_palm_basis^-1
```

这样比较和求解的目标是机器人 `left_rubber_hand_link` 与 `right_rubber_hand_link` 的世界旋转，而不是直接复制人手四元数数值。

## 6. A.1 三阶段流程

### Stage 1：scaled nominal object

使用演示人物缩放比例生成 nominal table，在 nominal scene 中完成常规全身 retarget。

这一阶段：

- 使用 interaction mesh；
- 保留足部约束和关节限位；
- 保留物体非穿透约束；
- 不应用 PT 腕姿态。

输出：

```text
<task>_nominal_scaled.npz
```

### Stage 2：fixed physical object

将 Stage 1 动作适配到真实尺寸的桌子，得到固定尺寸物体 baseline。

这一阶段同样不应用 PT 腕姿态，因此不会让掌面目标与物体非穿透约束互相竞争。

输出：

```text
<task>_fixed_object_base.npz
```

### Stage 3：A.1 wrist-only post-process

以 Stage 2 的每一帧为输入，冻结除左右腕以外的所有 qpos。每只手独立求解：

```text
min || Log(R_robot_link_target * R_robot_link_current^-1) ||
```

优化变量只有腕部：

```text
[wrist_roll, wrist_pitch, wrist_yaw]
```

求解遵守腕关节上下限，并使用上一帧解作为下一帧初值以保持时间连续性。若任意帧的最终姿态误差超过 `max_solver_error_deg`，pipeline 会报错。

输出：

```text
<task>_fixed_object.npz
```

该文件就是 A.1 训练参考候选。

## 7. 为什么 A.1 最终阶段允许视觉穿透

人手演示包含手指关节和人手几何，而橡胶手是尺寸不同的单一刚体。同一个腕部枢轴位置和掌面朝向，不可能同时保证两种手部网格具有相同的接触表面。

A.1 选择优先保留：

- baseline 的全身动作；
- baseline 的肩肘和腕部枢轴位置；
- PT 演示的完整掌面朝向；
- 机器人腕关节的物理限位和时间连续性。

A.1 不在 Stage 3 重新执行物体非穿透求解。这样做是有意的，因为重新消除穿透会移动肩肘或躯干，使结果偏离演示动作。视觉穿透表示人手和橡胶手几何之间的 embodiment mismatch，并不表示腕部 retarget 失败。

训练或仿真时仍应启用橡胶手碰撞。A.1 提供运动学 reference，物理引擎和控制策略决定实际可达到的接触状态与接触力。

## 8. 运行命令

从 retargeting package 目录运行：

```bash
cd /home/kevin/holosoma-rubber-hand-largetable/src/holosoma_retargeting/holosoma_retargeting

/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  examples/robot_retarget.py \
  --data-path /home/kevin/holosoma/OMOMO_new \
  --save-dir /tmp/holosoma_rubber_plan_a1_pipeline \
  --task-type object_interaction \
  --task-name sub6_largetable_033 \
  --data-format smplh \
  --task-config.object-name largetable \
  --fixed-object-size-adaptation \
  --retargeter.pt-wrist-orientation.enable \
  --retargeter.foot-sticking-tolerance 0.05
```

最终目录包含：

```text
sub6_largetable_033_nominal_scaled.npz
sub6_largetable_033_fixed_object_base.npz
sub6_largetable_033_fixed_object.npz
```

## 9. 输出不变量与验证要求

对 `fixed_object_base.npz` 和最终 `fixed_object.npz` 应验证：

1. shape、帧数、FPS 和物体轨迹一致；
2. 只有六个腕关节 qpos 可以变化；
3. 其他机器人关节、floating base 和物体 qpos 必须逐元素一致；
4. 左右橡胶手世界姿态与 PT 掌面目标一致；
5. 腕关节不得越限；
6. 腕关节速度不得出现不连续跳变。

当前验证序列：

```text
sub6_largetable_033
frames = 186
fps = 30

PT wrist-to-palm calibration p90:
  left  = 0.000069 deg
  right = 0.000065 deg

A.1 robot hand-link orientation error p90/max:
  left  = 0.0 / 0.0 deg
  right = 0.0 / 0.0 deg

changed qpos indices:
  [26, 27, 28, 33, 34, 35]

maximum absolute change outside the six wrist qpos:
  0.0

wrist speed at 30 FPS:
  p90 = 0.847484 rad/s
  p99 = 2.663014 rad/s
  max = 5.112121 rad/s

minimum wrist joint-limit margin:
  0.515911 rad

rubber-hand/table signed distance:
  left minimum  = -0.085606 m
  right minimum = -0.085362 m
```

负的 signed distance 是第 7 节所述的预期 embodiment mismatch，不是隐藏的验证失败。完整 pipeline 的数值结果仍应以运行日志和输出 NPZ 的验证脚本为准。

## 10. 主要实现文件

```text
src/holosoma_retargeting/holosoma_retargeting/src/utils.py
  - 读取 PT 左右腕全局四元数
  - 四元数归一化与时间连续化
  - wrist-to-palm 标定和掌面目标构造

src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py
  - robot palm basis
  - 人手掌面到 robot hand-link 的映射
  - A.1 三腕关节逐帧求解

src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py
  - 三阶段 orchestration
  - baseline 与 A.1 NPZ 保存

src/holosoma_retargeting/holosoma_retargeting/config_types/retargeter.py
  - A.1 开关和误差阈值
```

## 11. 当前边界

- A.1 当前只支持 `smplh` InterMimic PT 格式；
- 当前只接入 G1 橡胶手；
- 不与 motion augmentation 同时使用；
- 不恢复人手手指自由度或接触压力分布；
- 不在 retarget 阶段估计训练中的实际接触力；
- A.1e 等接触投影实验不属于本 pipeline。
