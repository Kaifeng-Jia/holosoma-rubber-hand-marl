# HoloSOMA `push_largetable` 阶段报告

> 日期：2026-07-12  
> 目标：从原始 `main` 基线出发，为 G1 生成与真实尺寸落地物体交互的 retargeting 动作  
> 当前示例：`sub3_largetable_020`  
> 当前分支：`push_largetable`  
> 当前工作区：`/home/kevin/holosoma-push-largetable`

## 1. 当前结论

当前实现没有把 demo object 简单改成真实尺寸，而是采用两阶段方法：

1. **Nominal 阶段**：人体和物体按照同一个人体到机器人身高比例缩放，先得到几何一致、脚在地面的机器人参考动作。
2. **Fixed-object 阶段**：保持 demo interaction mesh 的 nominal 尺度，将 target 换成真实尺寸物体，并参考 nominal motion 重新优化机器人姿态。
3. 第二阶段额外锚定每一帧 nominal motion 的双脚高度，防止优化器通过整体抬高机器人来够到桌面。
4. Viser 最终显示的是第二阶段的真实尺寸物体和 `fixed_object` 动作；第一阶段不打开 Viser。

最终方法没有加入 `largetable` 名称判断、手指映射、手掌方向或“推桌子”专用约束，因而可以继续用于其他**固定尺寸、落地、不被抬起**的交互物体。

## 2. Git 基线与工作区隔离

### 2.1 基线

本任务没有继续沿用之前的 `push-bigbox-experiment` 或 `fixed-object-contact-retargeting` 实验分支，而是重新从原始 `main` 建立独立 worktree：

```text
main commit: 7e100d5c1fbaca2169339911a737552020dfdeb7
branch:      push_largetable
worktree:    /home/kevin/holosoma-push-largetable
```

这样做的原因是旧实验分支已经包含多轮物体缩放、grounding 和接触约束试验，不能作为干净基线。当前分支与 `/home/kevin/holosoma` 并行，旧工作区的数据和用户改动没有被 reset、checkout 或删除。

### 2.2 当前未提交内容

当前 Python 改动集中在：

```text
src/holosoma_retargeting/holosoma_retargeting/config_types/retargeting.py
src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py
src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py
src/holosoma_retargeting/holosoma_retargeting/src/utils.py
```

新增/恢复的资产为：

```text
src/holosoma_retargeting/holosoma_retargeting/models/largetable/largetable.obj
src/holosoma_retargeting/holosoma_retargeting/models/largetable/largetable.urdf
src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largetable.xml
```

这些内容目前还没有 commit。

## 3. 数据集下载与当前数据状态

### 3.1 官方下载流程

HoloSOMA 的 `holosoma_retargeting/README.md` 指定使用 **InterMimic 处理后的 OMOMO 数据**，而不是直接把原始 OMOMO `.npy` 输入 retargeter。官方步骤是：

1. 从项目 README 中给出的 [Google Drive 链接](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view) 下载 processed OMOMO。
2. 解压到 `demo_data/OMOMO_new`。
3. 目录中应包含逐序列的 `.pt` 文件。

既有项目记录还表明曾下载约 325 MB 的 OmniRetarget 数据。当前 largetable 工作实际使用的是 InterMimic 处理后的 OMOMO `.pt`。

### 3.2 当前可验证数据

数据没有复制进新 worktree，而是继续复用：

```text
/home/kevin/holosoma/OMOMO_new
```

当前实测：

```text
全部 .pt：       4421 条
largetable：      334 条
当前序列：        sub3_largetable_020.pt
当前序列形状：    (157, 591)
```

因此新工作区运行时使用绝对 `--data-path /home/kevin/holosoma/OMOMO_new`，不需要移动或重复下载数据。

### 3.3 `.pt` 数据解析

`load_intermimic_data()` 从每帧 591 维数据中读取：

```text
[162:318]  → 52 × 3 人体关节世界坐标
[318:325]  → 物体位置和四元数
```

InterMimic 中的物体字段会重排成：

```text
[qw, qx, qy, qz, x, y, z]
```

之后进入 MuJoCo 前再转换成：

```text
[x, y, z, qw, qx, qy, qz]
```

人物身高来自 `demo_data/height_dict.pkl`。`sub3` 的记录身高约为 `1.941 m`，G1 高度为 `1.32 m`，所以：

```text
smpl_scale = 1.32 / 1.941 ≈ 0.68006
```

## 4. largetable OBJ 处理流程

### 4.1 原始模型

离线脚本位于旧工作区：

```text
/home/kevin/holosoma/process_largetable.py
```

输入模型：

```text
data/captured_objects/largetable_cleaned_simplified.obj
```

原始包围盒约为：

```text
X: -10.8268 ～ 10.8268
Y: -17.7559 ～ 0
Z: -10.8268 ～ 10.8268
```

局部 `Y` 是桌子的高度轴；桌腿沿局部负 `Y` 延伸。`.pt` 中的物体四元数已经把局部 `+Y` 旋转到接近世界 `+Z`，因此 OBJ 不应再做额外轴旋转。

### 4.2 尺度计算

根据 `.pt` 数据和手部交互高度，目标桌高定为：

```text
TARGET_HEIGHT = 0.428 m
SOURCE_HEIGHT = 17.755909
SCALE = 0.428 / 17.755909
      ≈ 0.02410465
```

所有 OBJ 顶点统一乘以该比例，不做非均匀缩放。

### 4.3 原点处理

脚本不使用包围盒中心作为新原点，也不把模型强制移动到几何中心。它保留源 OBJ 原点，只添加：

```text
LOCAL_Y_OFFSET = +0.06 m
```

最终逻辑为：

```text
v_scaled = v_source × 0.02410465
v_output.y = v_scaled.y + 0.06
```

处理后 OBJ 包围盒约为：

```text
min:     [-0.260976, -0.368000, -0.260976]
max:     [ 0.260976,  0.060000,  0.260976]
extent:  [ 0.521953,  0.428000,  0.521953]
```

在 `sub3_largetable_020` 第一帧物体姿态下，世界包围盒约为：

```text
bottom z ≈ -0.0063 m
top z    ≈  0.4310 m
```

四个桌腿底部中心的平均高度约为 `-0.0016 m`，最大高差约 `8.6 mm`。这说明 `+0.06 m` 的局部原点校准基本合理，最低顶点的轻微负值主要来自物体小幅倾斜。

### 4.4 “起始点/原点”的两种含义

本项目里容易混淆两个概念：

1. **OBJ 局部原点**：由原始网格定义；当前方案保留它，不改成包围盒中心。
2. **每帧物体世界位姿**：来自 `.pt`，局部点通过下式进入世界坐标：

```text
p_world = R(q_object) · p_local + t_object
```

因此不能只看 OBJ 顶点坐标判断桌子是否落地，必须同时应用 `.pt` 的旋转和平移。

### 4.5 机器人初始配置与 Interaction Mesh 起始点

如果“起始点”指机器人优化的 `q_init`，当前 object-interaction 路径按以下方式建立：

1. 找到整段动作中左右脚趾关节的最低 Z，并从所有人体关节 Z 中减去该值。
2. 将贴地后的人体关节整体乘 `smpl_scale`。
3. G1 floating base 的初始位置使用第一帧缩放后的人体 `Pelvis` 坐标。
4. 初始朝向不是直接复制人体四元数，而是在水平面计算“人体 pelvis 指向物体原点”的方向作为机器人局部 X 轴，世界 `+Z` 作为局部 Z 轴，再转换成 MuJoCo 四元数。
5. 29 个可动关节从零开始，由第一帧最多 50 次 SQP iteration 求出可行姿态；后续帧以上一帧结果 warm start。

可写成：

```text
x_axis = normalize((object_xy - pelvis_xy), z=0)
z_axis = [0, 0, 1]
y_axis = z_axis × x_axis
q_root = quaternion([x_axis, y_axis, z_axis])

q_init = [pelvis_xyz, q_root, zeros(29)]
```

如果“起始点”指 Interaction Mesh 的物体点，则 `load_object_data()` 使用固定随机种子 `42`，从 OBJ 表面进行目标数量为 100 的 even sampling。当前 largetable 网格因为断开/薄片结构，日志会提示实际得到约 `93/100` 个点。真实点记为 `P_object`，nominal demo 点为：

```text
P_demo = smpl_scale × P_object
```

每帧先把映射后的人体关节转换进物体局部坐标系，再与这些物体采样点拼接，进行 Delaunay tetrahedralization；得到的邻接关系和 Laplacian coordinates 就是该帧优化的 source interaction mesh 目标。

## 5. URDF/XML 与碰撞几何

### 5.1 初始问题

原始 largetable OBJ 不是 watertight mesh，并且包含许多小的断开部分。若在 MuJoCo 中把整个桌子作为单一 mesh collision geom，MuJoCo 会接近按整体 convex hull 处理，使桌子下方本应为空的区域变成“实心”。这会让机器人手、腿和桌下空间产生错误碰撞。

### 5.2 当前结构

URDF 保留一份完整视觉 mesh，但碰撞改为五个 box primitive：

```text
1 × tabletop
4 × legs
```

对应尺寸为：

```text
tabletop center:   (0, 0.036275, 0)
tabletop full size:(0.5219528, 0.04745, 0.5219528)

leg centers:       (±0.23525, -0.17835, ±0.23525)
leg full size:     (0.05, 0.3793, 0.05)
```

组合 XML 同样使用：

- 一个不参与碰撞的视觉 mesh；
- 五个透明 box collision geoms；
- 真实尺寸视觉和碰撞外包围盒误差小于 `1e-7 m`。

验证结果表明桌下中心恢复为空，primitive 总体积约为错误全局凸包体积的 `14.3%`。

## 6. `main` 的标准 retargeting 行为

原始 `main` 的 object-interaction 流程为：

1. 人体关节先贴地，再整体乘 `smpl_scale`。
2. 物体 X/Y 平移随人体缩放。
3. 物体初始 Z 高度被保留，只缩放后续 Z 位移。
4. `load_object_data()` 返回：
   - `object_local_pts`：真实尺寸 target points；
   - `object_local_pts_demo`：乘 `smpl_scale` 的 demo points。
5. source interaction mesh 由缩放人体关节和缩放 demo object points 构建。
6. target interaction mesh 由 G1 link points 和真实 target object points 构建。
7. SQP 优化最小化 interaction-mesh Laplacian 变形，同时施加地面/物体非穿透、脚 XY sticking、关节限制、平滑和 trust-region 约束。

人体腕关节映射保持原始配置：

```text
L_Wrist → left_rubber_hand_link
R_Wrist → right_rubber_hand_link
```

没有额外语义告诉优化器“手必须压在桌面边缘”或“手掌必须朝前”。

## 7. 问题诊断与方案演进

### 7.1 直接使用 `main`

真实桌高为 `0.428 m`，但人体缩放比例为 `0.68006`。若把人和交互动作理解成机器人尺度，等效桌高只有：

```text
0.428 × 0.68006 ≈ 0.291 m
```

因此机器人会继承“推约 0.291 m 高物体”的深弯腰姿态。Viser 中虽然 target mesh 是真实尺寸，机器人姿态仍像在推小桌子。

在第 86 帧，缩放人体双腕约为 `0.287/0.291 m`，真实桌面约为 `0.432 m`。单纯把 demo geometry 改成真实尺寸会让手低于桌面约 `14 cm`，并不能解决问题。

### 7.2 被放弃的方向

过程中考虑或试验过以下方向，但没有合入当前分支：

- 从包含旧缩放实验的分支继续开发；
- 直接把 demo object 改成真实尺寸；
- 整体抬高人体/机器人以对齐真实桌面；
- 添加 index、pinky、thumb 或手掌方向等推桌专用约束；
- 使用大规模 contact-aware 特殊优化器。

放弃原因分别是：基线不干净、源几何不一致、脚会悬空、会对单个物体过拟合，或者改动范围明显超过标准 retargeting。

### 7.3 最终两阶段方案

新增 CLI 开关：

```text
--fixed-object-size-adaptation
```

只在 `object_interaction` 且未启用原有 pose augmentation 时使用。

#### 阶段 1：scaled nominal scene

- 人体关节：乘 `smpl_scale`。
- demo object points：乘 `smpl_scale`。
- nominal target points：同样使用缩放 object points。
- MuJoCo 视觉 mesh、primitive collision size、局部位置和惯量坐标：通过临时 XML 统一乘 `smpl_scale`。
- 临时 XML 写在系统临时目录，不污染模型目录。
- 第一帧物体原点高度也按地面基准缩放：

```text
z_nominal(t) = z_physical(t) - (1 - smpl_scale) × z_physical(0)
```

因为原 preprocessing 已经缩放了 Z 方向运动量，上式只补上此前被保留的初始高度，使缩小物体仍落在地面附近。

本阶段输出：

```text
sub3_largetable_020_nominal_scaled.npz
```

#### 阶段 2：real fixed object adaptation

- source interaction mesh：继续使用阶段 1 的缩放人体、缩放 demo object 和 nominal object pose。
- target object：切换为真实 OBJ 采样点、真实碰撞几何和真实物体 pose。
- 机器人初始化与姿态参考：使用阶段 1 的 `q_nominal`。
- 原有 nominal tracking cost 约束 G1 root 和下肢不要偏离 nominal motion 太远。
- 最终 Viser 只由第二阶段 retargeter 打开，因此显示真实尺寸物体。

本阶段输出：

```text
sub3_largetable_020_fixed_object.npz
```

### 7.4 双脚高度锚定

第一版两阶段实现虽然让手和躯干明显抬高，但优化器选择整体抬升机器人，脚部比 nominal 高约 `6–8 cm`。

最终版本在第二阶段增加通用 lower-body anchoring：

1. 对每一帧读取 nominal motion 中左右脚 link 的世界 Z。
2. 在线性化 SQP 中，用当前脚部 Jacobian 约束最终脚高度接近对应 nominal 高度。
3. 默认容差为 `5 mm`。
4. 不强制脚永远为世界 `z=0`；如果 nominal 动作中某只脚正常抬起，最终动作仍允许保留该抬脚高度。

它只在 `--fixed-object-size-adaptation` 路径启用，不改变原始 `main` 路径。

## 8. 代码修改摘要

### `config_types/retargeting.py`

- 新增 `fixed_object_size_adaptation: bool = False`。
- 默认关闭，因此旧命令行为不变。

### `examples/robot_retarget.py`

- 校验新模式只用于固定物体 object interaction。
- 计算落地的 nominal object poses。
- 自动运行 nominal 和 fixed-object 两次 retarget。
- nominal 阶段关闭 debug/Viser；fixed 阶段沿用用户的可视化设置。
- 分开保存 nominal 与最终结果，避免覆盖原 baseline。

### `src/utils.py`

- 新增通用 MuJoCo scene XML 缩放函数。
- 只缩放指定 object body subtree 和它引用的 mesh assets。
- 不缩放机器人和地面。

### `src/interaction_mesh_retargeter.py`

- 允许 nominal 阶段覆盖 MuJoCo scene XML 路径。
- 新增 nominal foot-height anchoring hard constraint。
- 原 Laplacian、joint limits、non-penetration、smoothness 和 wrist mapping 保持不变。

## 9. 验证结果

### 9.1 工程验证

- 新 CLI 开关已由 Tyro 正确解析。
- 临时缩放 XML 可被 MuJoCo 加载：`nq=43, nv=41, ngeom=108`。
- largetable mesh scale 为 `[0.68006182, 0.68006182, 0.68006182]`。
- tabletop collision half-size同步缩放为约 `[0.17748, 0.01613, 0.17748]`。
- 两帧 smoke test 的两个阶段均成功求解。
- 完整 157 帧两个阶段均成功求解。
- 最终输出形状为 `(157, 43)`，全部数值有限。

### 9.2 第 86 帧对比

以下高度是 MuJoCo link origin 高度，主要用于同模型版本之间的相对比较：

| 指标 | 原 `main` baseline | nominal | 当前 fixed-object |
|---|---:|---:|---:|
| 左手 link Z | 0.264 m | 0.283 m | 0.365 m |
| 右手 link Z | 0.283 m | 0.282 m | 0.377 m |
| pelvis Z | 0.646 m | 0.628 m | 0.635 m |
| torso Z | 0.674 m | 0.659 m | 0.663 m |
| 左脚踝 link Z | 0.067 m | 0.079 m | 0.083 m |
| 右脚踝 link Z | 0.034 m | 0.041 m | 0.043 m |
| torso 倾角 | 93.3° | 89.1° | 69.3° |

用 `mujoco.mj_geomDistance()` 测量手部和桌子碰撞几何的表面距离：

```text
baseline: 左手 0.1523 m，右手 0.1104 m
fixed:    左手 0.0589 m，右手 0.0455 m
```

结论：当前方案显著降低深弯腰程度、提高双手并保持脚部接近 nominal 高度，但仍有约 `4.5–5.9 cm` 的表面间隙。由于当前目标是保持标准 retargeting，本轮没有继续加入手部接触或手掌方向约束。

## 10. 当前运行命令

```bash
cd /home/kevin/holosoma-push-largetable/src/holosoma_retargeting/holosoma_retargeting

/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  examples/robot_retarget.py \
  --data-path /home/kevin/holosoma/OMOMO_new \
  --task-type object_interaction \
  --task-name sub3_largetable_020 \
  --data-format smplh \
  --task-config.object-name largetable \
  --fixed-object-size-adaptation \
  --retargeter.debug \
  --retargeter.visualize
```

默认输出目录下会生成：

```text
demo_results/g1/object_interaction/omomo/sub3_largetable_020_nominal_scaled.npz
demo_results/g1/object_interaction/omomo/sub3_largetable_020_fixed_object.npz
```

Viser 播放的是第二个结果对应的真实尺寸 target。

## 11. 当前限制与下一步决策

1. 当前方法假设物体固定尺寸、在地面附近运动且不被抬起；举起物体任务不应直接使用 nominal-grounded pose 公式。
2. 临时 XML 缩放函数要求组合 XML 中 object body 名称包含 `object_name`，并能从该 body 的 geoms 找到所引用 mesh。
3. 当前只验证了 `sub3_largetable_020` 完整序列；推广到其他物体前应至少回归 `largebox` 和另一个镂空/细腿物体。
4. 剩余手部间隙是真实存在的几何表面间隙。是否加入通用 contact term 应单独讨论，不能根据一张截图直接加入手指或手掌专用规则。
5. 在进入训练前，应使用 `fixed_object.npz` 做格式转换，并确认训练侧 USD/碰撞资产与 retargeting 中的真实 largetable 尺寸一致。

## 12. 建议的提交前检查

```bash
git -C /home/kevin/holosoma-push-largetable status --short
git -C /home/kevin/holosoma-push-largetable diff --check
git -C /home/kevin/holosoma-push-largetable diff --stat main
```

在用户确认 Viser 的完整动作可接受之前，不建议立即 commit 或扩展到 parallel retargeting。
