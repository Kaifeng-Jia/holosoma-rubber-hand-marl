# HoloSOMA `push_largetable` 手掌朝向实现报告

> 日期：2026-07-16  
> 分支：`push_largetable`  
> 工作区：`/home/kevin/holosoma-push-largetable`  
> Git 基线：`main` / `7e100d5c1fbaca2169339911a737552020dfdeb7`  
> 主要验证序列：`sub6_largetable_033`（186 帧）  
> 文档范围：说明第一、二阶段 fixed-object retarget 完成以后，为解决 G1 手掌朝向问题所做的全部实现与实验

## 1. 报告范围与最终结论

此前的 `PUSH_LARGETABLE_FROM_MAIN_REPORT_2026-07-12.md` 已记录从 `main` 开始的资产处理、OBJ 原点校准、nominal scale 以及两阶段 fixed-object retarget。本报告不重复替代那份报告，而是以它的结果为起点，专门记录后续的手腕/手掌处理。

最终保留的方法是：

1. 第一阶段继续在“人体与物体同比缩小”的 nominal 场景中生成参考动作。
2. 第二阶段继续生成不包含手掌朝向目标的真实尺寸物体动作，作为 baseline 和对照结果。
3. 当显式启用 hand orientation 时，再增加一次真实尺寸适配；它从第一阶段的 `q_nominal` 出发，在接触期间加入“手掌朝向物体”的软目标。
4. 最终只约束手掌法向，不约束手指长轴，因此 `finger_direction_weight = 0.0`。
5. 不使用硬编码关节角、不把手腕锁死、不加入食指/小指的虚拟接触点，也不要求不存在的灵巧手指关节完成抓边动作。

因此，最终方法的准确名称应是“接触阶段的 palm-direction objective”，而不是完整的固定手姿态，也不是 fingers-up hard constraint。

## 2. 第一、二阶段基线

手掌处理开始前，两阶段 fixed-object retarget 已经完成。下面只列出与本报告衔接所需的代码基线。

### 2.1 第一阶段：nominal scaled retarget

第一阶段保持 OmniRetarget 原有的尺度一致性：

- 人体关节按 `smpl_scale` 缩小到 G1 尺度。
- demo object mesh 和物体原点高度也按相同比例缩小。
- 机器人、地面不缩放。
- 在临时的 scaled-object MuJoCo XML 中求解。
- 输出 `*_nominal_scaled.npz`。

这部分基线涉及：

- `config_types/retargeting.py`
  - 新增 `fixed_object_size_adaptation: bool = False` 开关。
- `src/utils.py`
  - 新增 `create_uniformly_scaled_object_scene_xml()`，只缩放目标物体子树及其引用的 mesh asset。
- `examples/robot_retarget.py`
  - 新增 `create_grounded_nominal_object_poses()`，使落地物体绕地面缩放，而不是保留真实物体的初始原点高度。
  - 创建临时 nominal scene，执行第一阶段求解。
- `src/interaction_mesh_retargeter.py`
  - 新增可选 `scene_xml_path`，允许第一阶段载入临时 scaled-object XML。

### 2.2 第二阶段：real-size fixed-object adaptation

第二阶段把 target object 换成真实尺寸，同时保留第一阶段 interaction mesh 作为 nominal correspondence：

- `object_poses` 使用 nominal scaled object。
- `object_poses_augmented` 使用真实尺寸物体位姿。
- `object_points_local_demo` 使用 nominal object points。
- `object_points_local` 使用真实物体 points。
- `q_nominal_list = q_nominal`，让真实尺寸适配尽量保持第一阶段动作结构。
- `anchor_nominal_foot_height = True`，防止优化器通过整体抬高机器人来够到桌面。
- 输出原来的 `*_fixed_object.npz`。

这就是后续手掌实验的 baseline。它已经解决了“真实桌子尺寸”和“机器人双脚落地”的主要矛盾，但没有规定 G1 rubber hand 的哪一面朝向桌子。

## 3. 为什么还需要手掌朝向处理

原始 interaction-mesh retarget 主要匹配人体关节、机器人 link 和物体点之间的空间关系。它能把手部 link 移到桌边附近，但没有充分确定 rubber hand 绕自身轴的旋转。

对于 G1，这会产生下列现象：

- 手的位置接近桌沿，但手背朝向桌面或桌沿。
- 手腕向下折，视觉上像用手指端部或手背推桌子。
- 由于 G1 没有可独立优化的灵巧手指，自由度不足以复现“拇指在桌下、其余四指在桌上”的人类抓边姿态。

所以需要补充的是手掌法向，而不是重新设计整套 retarget，也不是为 `largetable` 写一组固定关节角。

## 4. 最终求解流程

启用 `--retargeter.hand-orientation.enable` 后，当前流程会产生三个结果：

| 阶段 | 起点与参考 | 物体尺寸 | 手掌目标 | 输出 |
|---|---|---|---|---|
| Stage 1 | 原始 nominal retarget | nominal scaled | 关闭 | `*_nominal_scaled.npz` |
| Stage 2 | 从 `q_nominal` 适配 | 真实尺寸 | 关闭 | `*_fixed_object_base.npz` |
| Stage 3 | 再次从 `q_nominal` 适配 | 真实尺寸 | 接触时开启 | `*_fixed_object.npz` |

这里最重要的实现细节是：

```text
                    ┌─ Stage 2: real-size, no palm target → fixed_object_base
Stage 1 q_nominal ──┤
                    └─ Stage 3: real-size + palm target   → fixed_object
```

Stage 3 不是从 Stage 2 的 `q_fixed_base` 继续做后处理。它和 Stage 2 是从同一个 `q_nominal` 出发的两个真实尺寸适配分支。

实验中，如果先完成 Stage 2，再强行扭转手腕，优化器容易陷入局部不可行状态，或者为了转动手掌而明显移动手、脚和 root。让手掌目标在“nominal → real-size”适配过程中同时生效，能够找到更稳定、可达的整臂姿态。

未启用 hand orientation 时，流程仍保持原来的两阶段行为，最终直接输出 `*_fixed_object.npz`。

## 5. 最终保留的代码修改

### 5.1 `config_types/retargeter.py`

新增 `HandOrientationConfig`：

```text
enable                   = False
contact_distance         = 0.10 m
fade_frames              = 8
palm_direction_weight    = 5.0
finger_direction_weight  = 0.0
```

并在 `RetargeterConfig` 中新增：

```text
hand_orientation: HandOrientationConfig
```

各参数含义：

- `enable`：显式开启第三阶段；默认关闭，避免改变其他任务的标准行为。
- `contact_distance`：在人类 demo 中，wrist 到 demo object sampled surface 的最近距离阈值。
- `fade_frames`：每个连续接触区间首尾的渐入/渐出帧数。
- `palm_direction_weight`：手掌朝向目标的软代价权重。
- `finger_direction_weight`：可选的 hand roll/长轴方向权重；最终设为 `0.0`，因此当前不生效。

当前文件中 `enable` 的字段注释仍写有 `fingers-up`，这是实验阶段遗留的描述；实际默认配置和最终求解只保留 palm direction。该注释应在后续清理时改名，但本报告没有额外修改代码。

### 5.2 `examples/robot_retarget.py`

#### 配置验证

`validate_config()` 增加：

- hand orientation 只能与 `fixed_object_size_adaptation` 一起使用。
- 当前只支持 `robot == "g1"` 的 rubber-hand model。
- fixed-object adaptation 只适用于 `object_interaction`。
- fixed-object adaptation 不与 pose augmentation 同时启用。

#### 参数传递

`build_retargeter_kwargs_from_config()` 把 `retargeter_config.hand_orientation` 传给 `InteractionMeshRetargeter`。

#### 三阶段分支

在 fixed-object 路径中：

1. 计算 `refine_hand_orientation = cfg.retargeter.hand_orientation.enable`。
2. Stage 1 通过 `dataclasses.replace()` 强制关闭 hand orientation。
3. 启用手掌目标时，Stage 2 也使用关闭 hand orientation 的独立 retargeter，并保存 `*_fixed_object_base.npz`。
4. Stage 3 使用启用 hand orientation 的 retargeter。
5. Stage 3 的参数明确为：

```text
q_a_init       = q_nominal[0]
q_nominal_list = q_nominal
original       = False
demo object    = nominal scaled object
target object  = real-size object
```

因此 Stage 3 不是从 `q_fixed_base[0]` 开始，也没有把 `q_fixed_base` 作为 nominal reference。

### 5.3 `src/interaction_mesh_retargeter.py`

#### 构造参数与初始化

`InteractionMeshRetargeter.__init__()` 新增：

```text
hand_orientation: HandOrientationConfig | None
```

随后调用 `_init_hand_orientation()`。该函数为左右手解析：

- demo wrist 名称：`L_Wrist` / `LeftHand`、`R_Wrist` / `RightHand`。
- 机器人 hand link/body id。
- G1 rubber hand 的局部 palm normal。
- 同侧 7 个允许影响手掌朝向的 arm joints：
  - shoulder pitch
  - shoulder roll
  - shoulder yaw
  - elbow
  - wrist roll
  - wrist pitch
  - wrist yaw

当前 G1 rubber-hand 局部 palm normal 为：

```text
left  = [-0.07513681, -0.99540367, -0.05938011]
right = [-0.07514936,  0.99540878, -0.05927846]
```

限制 Jacobian 只作用于同侧 7 个上肢关节，可以避免手掌朝向目标直接驱动 floating base、腰部或腿部。

#### 接触区间检测

`_compute_hand_orientation_weights()` 完成以下步骤：

1. 把每帧 human wrist 从世界坐标变换到 demo object 局部坐标。
2. 计算 wrist 到 `object_points_local_demo` sampled surface 的最近距离。
3. 距离不超过 `0.10 m` 时认为该手处于接触阶段。
4. 把离散接触帧拆成连续区间。
5. 每个区间首尾用 8 帧做线性 fade，避免目标突然打开或关闭。

在 `sub6_largetable_033` 上，检测到的范围为：

```text
left:  frames 8..148（包含 fade）
right: frames 7..147（包含 fade）
```

#### 手掌目标生成

`_compute_fixed_push_hand_orientation_targets()` 不硬编码世界坐标中的“向前”。它先根据 demo 判断每只手从物体哪一侧施力：

```text
inward_world = object_center - demo_wrist
inward_world.z = 0
```

然后：

1. 把每个 active frame 的水平 inward direction 转换到 demo object 局部坐标。
2. 对 active frames 求平均，得到稳定的 object-local pushing side。
3. 每帧使用真实 target object rotation 把它转换回世界坐标。
4. 把该方向作为 palm target。

这样，目标会随物体朝向变化，而不是针对 `largetable` 写死某个世界轴。

函数仍会计算一个沿物体边缘的 lateral direction：

```text
lateral = world_up × inward
```

它是可选 finger/hand-long-axis target，但最终 `finger_direction_weight = 0.0`，所以不会影响当前解。

#### SQP 目标项

新增 `_calc_body_orientation_linearization()`，通过 MuJoCo `mj_jac()` 取得 hand body 的角速度 Jacobian。新增 `_skew()` 用于一阶方向线性化。

对当前 hand rotation `R`、局部 palm normal `n_local`：

```text
n_world = R · n_local
```

其一阶近似为：

```text
n_linear = n_world + [-skew(n_world) · J_angular] · Δq_arm
```

接触时加入软目标：

```text
weight_palm × weight_fade × ||n_linear - n_target||²
```

其中：

```text
weight_palm = 5.0
weight_fade ∈ [0, 1]
```

这个目标加入原有 objective list；原有 Laplacian matching、object non-penetration、joint limits、self-collision、nominal tracking、smoothness 和 foot constraints 没有被删除。

finger direction 的同类目标代码仍保留为可选实验入口，但由于权重为零，最终结果是 palm-only。

### 5.4 `src/utils.py` 没有保留手部专用修改

实验中曾加入 `compute_smplh_hand_frames()`，从 SMPL-H 手指 joint 推算 palm/finger frame；最终已删除。

因此当前 `src/utils.py` 相对 `main` 的改动属于第一阶段 nominal scene 的 `create_uniformly_scaled_object_scene_xml()`，不是最终手掌方案的一部分。报告特别注明这一点，以免把已撤销的 demo-hand frame 实验误认为当前依赖。

## 6. 实验过程与被撤销的方案

以下内容曾实际尝试，但没有保留为最终行为。

### 6.1 直接调整 wrist pitch

最初尝试通过单一 wrist pitch 改善“手腕完全向下”。这种方法只能控制一个关节角，不能稳定定义 palm normal；同一个角度在左右手、不同肩肘姿态和不同接触侧上含义不同，因此没有作为通用方案保留。

### 6.2 食指/小指接触点或 fingers-up 目标

曾尝试通过 index/pinky 或手长轴定义 fingers-up/full hand frame。撤销原因：

- G1 当前模型是 rubber hand，没有与 SMPL-H 对应的独立灵巧手指自由度。
- 这些接触点会引入人为假设，不是标准 retarget 的直接观测。
- raw `sub6_largetable_033.pt` 的 SMPL-H 手 frame 并不支持“全程世界 +Z fingers-up”；该序列的 demo finger direction 主要朝世界 `-Z`。
- 强行要求 palm inward 与 fingers-up 会显著增加整臂和身体补偿。

### 6.3 数据驱动的 SMPL-H palm/finger frame

曾从 `.pt` 的 52 个 SMPL-H joints 推算 demo hand frame，并把它映射到机器人。

- 较高权重在约第 15 帧出现 infeasible。
- 降低权重后可以完成，但 hand position 平均改变约 62–72 mm，root 约 6–8 mm，feet 约 12–13 mm。
- palm 与目标同半球比例提高，但平均方向误差仍较大，Viser 中的手姿态也不符合预期。

因此相关 helper、target pipeline 和配置字段均被删除。

### 6.4 硬约束 palm hemisphere 或完整固定姿态

曾测试 palm 必须与 inward direction 同半球的硬约束，以及 palm + finger 两轴的完整固定 orientation。

- hard hemisphere constraint 在约第 15 帧导致求解 infeasible。
- 从已完成的 Stage 2 结果进行 exact full-orientation post-processing，容易陷入局部不可行或只发生部分旋转。
- 增加 object-motion phase gating 没有从根本上解决可达性问题。

最终改为软目标，并把它放入 nominal → real-size 的同一次适配中。

### 6.5 强制手指水平

后续又测试了让 rigid hand 的长轴水平：

| 方案 | 结果 | 结论 |
|---|---|---|
| roll soft weight = 0.1 | palm 仍较好，但手指与水平目标仍约差 75°；root/feet/hand 位移增加 | 收益太小 |
| roll weight = 1.0 | frame 75 palm error 约 20.7°/16.1°，root 平均约 42.9 mm，hand 位移约 147–149 mm | 明显破坏动作 |
| 仅 wrist 关节 | 可把 palm 控制到约 4–8°，但手的位置移动约 10.7–12.8 cm | 接触位置代价过大 |
| 同侧 7-DOF arm | frame 75 仍约 34 mm position error，palm 约 28°，水平误差约 23–30° | 无法同时满足 |

这些结果表明，对当前 G1 rubber-hand 几何和已有接触位置而言，“手掌稳定朝向桌子”和“手指严格水平”不能低代价同时满足。最终接受 palm-only，手指方向由原有 retarget、关节限制和 smoothness 自然决定。

## 7. 最终验证结果

最终 palm-only 配置在 `sub6_largetable_033` 的 186 帧上完成求解。

### 7.1 手掌朝向

```text
frame 75 palm error:
left  ≈ 1.2°
right ≈ 2.8°

pushing interval palm-toward-object positive ratio:
left  = 100%
right = 100%

mean palm dot target:
left  ≈ 0.999
right ≈ 0.999
```

### 7.2 相对无手掌目标的 Stage 2 baseline

```text
root position delta: mean ≈ 7.0 mm, max ≈ 17.2 mm
feet position delta: mean ≈ 3.3 mm, max ≈ 15.0 mm
left hand position delta:  mean ≈ 31.9 mm, max ≈ 46.8 mm
right hand position delta: mean ≈ 42.7 mm, max ≈ 61.4 mm
object pose max delta: 0
```

这说明手掌朝向主要由上肢调整完成；root 和 feet 有小幅变化，物体轨迹没有被手掌目标修改。

### 7.3 已知妥协

最终 rigid-hand 长轴仍主要向下，平均世界 Z 分量约为：

```text
left  ≈ -0.984
right ≈ -0.973
```

这是明确接受的妥协：保证 palm 朝向正确、接触位置合理、脚和 root 基本稳定，放弃无法由非灵巧手模型低代价实现的手指朝向。

## 8. 运行命令与输出

在当前 worktree 中运行：

```bash
cd /home/kevin/holosoma-push-largetable/src/holosoma_retargeting/holosoma_retargeting

/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  examples/robot_retarget.py \
  --data-path /home/kevin/holosoma/OMOMO_new \
  --save-dir /tmp/holosoma-final-palm-viser \
  --task-type object_interaction \
  --task-name sub6_largetable_033 \
  --data-format smplh \
  --task-config.object-name largetable \
  --fixed-object-size-adaptation \
  --retargeter.hand-orientation.enable \
  --retargeter.debug \
  --retargeter.visualize \
  --retargeter.foot-sticking-tolerance 0.05
```

输出：

```text
sub6_largetable_033_nominal_scaled.npz
sub6_largetable_033_fixed_object_base.npz
sub6_largetable_033_fixed_object.npz
```

用于后续训练或最终检查的是 `*_fixed_object.npz`。`*_fixed_object_base.npz` 用于比较启用手掌目标前后的差异。

如果不加：

```text
--retargeter.hand-orientation.enable
```

程序保持原来的两阶段 fixed-object retarget，不会启用手掌朝向目标。

## 9. 当前工作区状态与边界

截至本报告生成时：

- 当前分支仍是 `push_largetable`。
- HEAD 与创建 worktree 时的 `main` 基线相同：`7e100d5c`。
- 第一、二阶段 fixed-object 修改和本报告记录的手掌修改都仍是未提交变更。
- 本报告仅新增文档，没有再修改 retargeting 代码或参数。

当前实现的适用边界：

- 仅在显式启用时生效。
- 当前 hand link geometry/palm normal 只针对 G1 rubber hands 做过验证。
- pushing side 从 demo wrist 与 object pose 自动计算，不检查 `object_name == "largetable"`。
- 它可以扩展到其他固定尺寸、落地、推压类物体，但应逐序列检查接触阈值、手掌朝向、脚底稳定和手-物体穿透。
- 它不是 dexterous grasp retarget，不能生成独立拇指/四指包住桌边的动作。

## 10. 最终保留项与撤销项清单

### 保留

- 可选 `HandOrientationConfig`。
- 基于 demo wrist-to-object distance 的接触检测。
- 接触区间 8 帧 fade。
- 基于 demo pushing side、随目标物体旋转的 palm target。
- G1 rubber-hand local palm normal。
- 同侧 7-DOF arm angular Jacobian。
- contact-phase palm soft objective。
- 从 Stage 1 分叉的 Stage 2 baseline 和 Stage 3 palm result。
- `nominal_scaled`、`fixed_object_base`、`fixed_object` 三个独立输出。

### 未保留为最终行为

- 固定 wrist joint angle。
- 手腕 hard lock。
- palm hemisphere hard constraint。
- index/pinky 虚拟接触约束。
- SMPL-H demo hand frame helper。
- fingers-up hard target。
- finger horizontal/roll target（代码入口保留，但默认权重为零）。
- 从 Stage 2 结果继续做 hand post-processing。

## 11. 对后续工作的建议

在继续批量 retarget 前，建议先冻结当前 palm-only 版本，并用已选的多个 largetable 序列做一致性验证。每条序列至少检查：

1. `fixed_object_base` 与 `fixed_object` 的手掌朝向差异。
2. 双脚最低高度和 foot drift。
3. root 是否因上肢目标发生异常移动。
4. hand link 是否进入桌面或桌腿。
5. 接触区间是否由 `0.10 m` 阈值正确识别。
6. 训练侧使用的 table USD/URDF/XML 尺寸是否与 retargeting 的真实尺寸资产一致。

如果后续确实要求“手指水平”或“拇指在桌下”，应把它视为新的机器人末端执行器设计/接触建模任务，而不是继续提高当前 hand-roll 权重。
