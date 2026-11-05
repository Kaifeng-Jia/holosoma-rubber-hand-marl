# Retarget Rubber Hand Plan B

## 1. 目标与分支关系

Plan B 为 G1 橡胶手直接设计一个适合推桌子的机器人姿态，而不是复制演示中的人手腕四元数：

- 两个掌面朝向桌板靠近机器人的竖直侧面；
- 两手沿桌边分开 `0.24 m`；
- 两只橡胶手的长轴向下，即手指向下、手腕位于桌板上方；
- 使用真实 MuJoCo collision geom 将 palm/table 有符号距离优化到接触边界；
- 腰、下肢、floating base、肩、肘和腕都可以参与全身求解；
- 物体 pose 始终锁定，不由 Plan B 推动或改写。

Plan B 与 A.1 平行，二者都从 fixed-object baseline 分支：

```text
Stage 1: nominal scaled object
                 |
Stage 2: fixed physical object baseline
                 |
        +--------+--------+
        |                 |
     A.1 wrist-only    Plan B full-body
     PT orientation   palm contact SQP
```

Plan B 不读取 PT 腕四元数，也不使用 A.1 的六腕关节结果。A.1、Plan B 和 legacy hand-orientation 模式互斥。

## 2. 三阶段 pipeline

### Stage 1：scaled nominal object

在随演示人物统一缩放的桌子上运行常规 interaction-mesh retargeting，输出：

```text
<task>_nominal_scaled.npz
```

### Stage 2：fixed physical object baseline

将 Stage 1 动作适配到真实尺寸桌子。Plan B 和 A.1 都在本阶段关闭，输出：

```text
<task>_fixed_object_base.npz
```

### Stage 3：独立 Plan B

Plan B 直接以 Stage 2 的 `q_fixed_base` 为 nominal reference，重新运行 full-body SQP，输出：

```text
<task>_fixed_object_plan_b.npz
```

Stage 3 的优化变量从 floating base 开始，要求：

```text
q_a_init_idx = -7
```

因此 Plan B 可以让腰和腿配合上身前倾，不继承 A.1 “只改六个腕关节”的限制。

## 3. 桌侧目标的定义

目标基于 MuJoCo 中的真实 tabletop collision box：

```text
geom = largetable_tabletop
face_axis = 2
face_sign = -1
vertical_axis = 1
lateral_axis = 0
```

所有轴都在 tabletop geom 局部坐标中定义，不依赖世界坐标中的固定朝向。

左右目标点位于竖直侧面中心，两手间距为 `0.24 m`。桌子每帧的 object rotation 会把目标点、向内法向和向下方向变换到世界坐标。

## 4. 橡胶手坐标与接触点

左右橡胶手使用各自的：

```text
left_rubber_hand_link
right_rubber_hand_link
```

代码从 collision mesh 顶点和 geom 的局部 `pos/quat` 推导掌面支撑点。左右 mesh 必须分别计算；不能共享由循环变量遗留的 `side`。

支撑点主要用于：

- 安全位和接近阶段的任务空间位置；
- 左右手在桌侧的切向分布；
- palm normal 与 robot link orientation 的线性化。

弯曲橡胶手的支撑平面中心可能不是一个真实 mesh 顶点，因此最终接触不以该虚拟点到桌面的距离为准。

## 5. 真实几何接触

当手掌接近桌子后，Plan B 直接使用：

```text
mujoco.mj_geomDistance(
    palm_collision_geom,
    largetable_tabletop,
)
```

其线性化有符号距离进入两个位置：

1. 硬约束要求新距离不小于 `0`；
2. 软目标要求真实 palm/table 距离趋近 `0`。

因此目标是从桌外贴到碰撞边界，而不是把虚拟目标放进桌内。固定 preload 偏移实验已被删除，因为有限高度的桌侧会让最近接触顶点随帧变化，单一毫米偏移不能代表整段动作。

每帧求解后还会重新调用真实几何距离做严格验收。若最小 robot/object 距离低于：

```text
-penetration_tolerance - collision_validation_tolerance
```

则继续 SQP；达到最大迭代数仍不满足时直接报错，并输出碰撞 geom、距离、位置误差、姿态误差及 tabletop 局部 mesh bounds。

## 6. 姿态与时间相位

Plan B 同时使用三类任务：

- `palm normal`：掌面朝向桌侧；
- `twist`：两手长轴都沿 tabletop 局部竖直方向向下；
- `tangent position`：维持左右手间距和竖直位置。

“手指向下”用于让腕部壳体位于桌板上方。沿桌边左右展开的早期实验会在稳定接触段造成 `wrist_pitch_link` 与 tabletop 相交，已被删除。

时间相位为：

```text
retract/orient -> delayed approach -> stable contact -> release
```

默认参数：

```text
contact_distance = 0.10 m
fade_frames = 24
release_frames = 36
approach_clearance = 0.05 m
```

`fade_frames` 控制进入和 approach 延迟；`release_frames` 只控制接触结束后的退出，使动作在序列结束前回到 baseline，同时不推迟稳定接触的开始。

## 7. 保留的约束

Plan B 没有删除 baseline 约束，继续保留：

- robot/object 非穿透；
- robot/ground 非穿透；
- 关节上下限；
- 足部 sticking 和 nominal foot height；
- 可选 self-collision；
- SQP trust region；
- baseline posture regularization；
- 跨帧 smoothness。

Plan B 使用 G1 XML 中腕 roll、pitch、yaw 的真实物理范围，覆盖 baseline 中针对旧腕部处理的手工窄限位。

## 8. 运行命令

从 retargeting package 目录运行：

```bash
cd /home/kevin/holosoma-rubber-hand-largetable/src/holosoma_retargeting/holosoma_retargeting

/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  examples/robot_retarget.py \
  --data-path /home/kevin/holosoma/OMOMO_new \
  --save-dir /tmp/holosoma_rubber_plan_b_pipeline \
  --task-type object_interaction \
  --task-name sub6_largetable_033 \
  --data-format smplh \
  --task-config.object-name largetable \
  --fixed-object-size-adaptation \
  --retargeter.plan-b-palm-contact.enable \
  --retargeter.foot-sticking-tolerance 0.05
```

最终目录包含：

```text
sub6_largetable_033_nominal_scaled.npz
sub6_largetable_033_fixed_object_base.npz
sub6_largetable_033_fixed_object_plan_b.npz
```

## 9. Viser 查看

当前最终验证资产：

```text
/tmp/holosoma_rubber_plan_b_final/sub6_largetable_033_fixed_object_plan_b.npz
```

查看命令：

```bash
cd /home/kevin/holosoma-rubber-hand-largetable/src/holosoma_retargeting/holosoma_retargeting

/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  viser_player.py \
  --robot-urdf models/g1/g1_29dof.urdf \
  --object-urdf models/largetable/largetable.urdf \
  --qpos-npz /tmp/holosoma_rubber_plan_b_final/sub6_largetable_033_fixed_object_plan_b.npz
```

## 10. 当前定量验证

验证序列：

```text
task = sub6_largetable_033
frames = 186
fps = 30
qpos shape = (186, 43)
stable contact frames = 56..147
```

不变量：

```text
table pose vs physical input max error = 0
table pose vs fixed-object baseline max error = 0
joint-limit max violation = 0
```

全帧碰撞：

```text
minimum robot/object signed distance = -0.040406 mm
validation tolerance = 0.100000 mm
```

稳定接触段：

```text
left palm normal error:
  p50 = 0.046849 deg
  p90 = 0.049241 deg
  max = 0.049520 deg

right palm normal error:
  p50 = 0.042404 deg
  p90 = 0.046101 deg
  max = 0.047060 deg

left downward-twist error:
  p50 = 0.088677 deg
  p90 = 0.130406 deg
  max = 0.148367 deg

right downward-twist error:
  p50 = 0.081459 deg
  p90 = 0.153024 deg
  max = 0.174827 deg

left real palm/table distance:
  p50 = 0.000000 mm
  p90 = 0.004783 mm
  max = 0.441985 mm

right real palm/table distance:
  p50 = 0.002273 mm
  p90 = 0.003213 mm
  max = 0.004562 mm
```

Plan B 是 full-body 分支，因此相对 fixed-object baseline，机器人 `qpos[0:36]` 都允许变化；物体 `qpos[-7:]` 完全不变。

## 11. 当前限制

- 当前只接入 G1、`largetable_tabletop` box 和固定尺寸物体 pipeline；
- 只针对该序列验证了默认轴、手距和权重；
- Plan B 生成的是运动学训练 reference，不直接计算接触力；
- 实际训练必须启用 rubber-hand collision；
- 当前最大机器人关节速度为 `11.2928 rad/s`，发生在第 `8 -> 9` 帧的 `left_wrist_roll_joint`；
- 硬逐帧限速和过强 temporal smoothness 都会破坏早期碰撞可行性，相关实验代码已删除；
- 在批量用于训练前，应继续解决进入阶段腕速峰值，并在更多人物、桌子尺度和初始站位上运行同样的碰撞/接触验收。

## 12. 相对 fixed-object baseline 的代码模块

新增：

```text
PlanBPalmContactConfig
  Plan B 几何轴、手距、相位、权重和严格验收参数

Plan B target geometry
  tabletop box 解析
  左右 rubber-hand collision mesh 支撑点
  palm normal / downward twist / tangent targets

Plan B phase logic
  demo wrist/object proximity
  delayed approach
  独立 release

Plan B SQP objectives
  full-body palm position
  palm normal
  downward twist
  real geom signed-distance contact
  baseline posture regularization

Plan B strict validation
  每帧真实 collision distance 复验
  失败 geom 和局部边界诊断

Plan B Stage 3 orchestration
  从 fixed-object baseline 独立分支
  输出 *_fixed_object_plan_b.npz
```

未删除或替换：

```text
Stage 1 nominal retargeting
Stage 2 fixed-object adaptation
interaction mesh / Laplacian objective
foot constraints
joint limits
object non-penetration
A.1 wrist-only pipeline
legacy hand-orientation pipeline
```

删除或拒绝纳入最终实现的实验：

```text
固定 contact preload depth
沿桌边展开的左右 twist
逐帧硬 joint-delta 限速
过强 temporal smoothness
远距离阶段的真实最近点后撤目标
```
