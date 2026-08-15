# 双机器人宽桌几何设计

## 1. 文档状态

- 状态：几何预检设计，坐标轴审计进行中
- 建立日期：2026-08-14
- 所属主线：`rubber_hand_marl_baseline`
- 上位路线：`MULTI_AGENT_EMERGENCE_ROADMAP.md`
- 原始单机器人资产：`objects_largetable.urdf`
- 计划新增资产：`objects_widetable.urdf`

本文只管理双机器人同侧 A1 Push 所需的桌子几何、坐标、站位和
验证记录，不定义新的研究路线。正式尺寸发生变化时更新本文，不再
创建平行的宽桌设计文档。

## 2. 设计目标

第一版宽桌必须允许两台同质 G1 在桌子同一侧执行相同的 A1 Push，
同时尽量保持冻结单机器人 A1 所依赖的接触几何。

必须保持：

- 固定橡胶手资产和真实碰撞；
- A1 的推动方向和物体参考轨迹；
- 推动方向上的桌面深度；
- 桌面高度和厚度；
- 推动侧接触边相对物体原点的位置；
- 桌子 reference root origin；
- 桌腿厚度和高度。

允许改变：

- 与推动方向垂直的桌面横向宽度；
- 左右桌腿的横向位置；
- 两个机器人 reference 的对称横向平移；
- 与最终布局对应的 ghost teammate 分布。

## 3. 当前桌子几何

当前 `objects_largetable.urdf` 使用一个 link，主要碰撞尺寸为：

| 部件 | 局部中心 | 尺寸 |
|---|---|---|
| 桌面 | `(0, 0.036275, 0)` | `(0.5219528, 0.04745, 0.5219528)` |
| 四条桌腿 | `x=±0.23525, y=-0.17835, z=±0.23525` | `(0.05, 0.3793, 0.05)` |

由此可得：

- 桌面顶部在局部竖直方向约为 `0.060000 m`；
- 桌面底部约为 `0.012550 m`；
- 桌腿底部约为 `-0.368000 m`；
- 当前横向和推动方向尺寸都约为 `0.522 m`；
- link origin 不等同于 COM；当前 inertial origin 恰好写在 link origin，
  但不应把这一旧设置当作新桌的物理依据。

当前坐标解释是假设：local Y 为资产竖直轴，local X 和 local Z 为桌面
平面。Isaac Sim 中经过 reference quaternion 后的实际世界映射必须在
创建新资产前完成只读确认。

## 4. 加宽原则

只沿与 A1 推动方向垂直的局部横向轴加宽。若坐标审计确认该轴为
local X，则候选桌面为：

```text
tabletop size = (W, 0.04745, 0.5219528)
tabletop center = (0, 0.036275, 0)
```

若审计表明横向轴不是 local X，则必须先更新本文和主路线，再修改
资产。禁止通过猜测直接缩放世界坐标轴。

桌腿保持 `0.05 m × 0.3793 m × 0.05 m`，只对称向外移动。保留当前
桌腿外边缘距离桌面边缘约 `0.000726 m` 的几何关系时，local X 桌腿
中心可写为：

```text
x_leg = ±(W / 2 - 0.0257264)
```

前后桌腿的 local Z 中心继续保持 `±0.23525 m`。

## 5. 宽度候选

| 候选 | 横向宽度 | 相对当前宽度 | 用途 |
|---|---:|---:|---|
| A | 1.2 m | 约 2.30 倍 | 较紧凑备选 |
| B | 1.4 m | 约 2.68 倍 | 首个实现候选 |
| C | 1.6 m | 约 3.07 倍 | 机器人仍冲突时的上界备选 |

优先实现 1.4 m。只有在可视化或碰撞预检失败时，才在同一个候选文件
中试验 1.2 m 或 1.6 m。最终仓库只保留选中的正式宽桌资产，不保留
三个近似重复的训练资产。

## 6. Visual 与 Collision

第一版不对旧 `largetable.obj` 进行非均匀整体缩放。整体缩放会同时
拉粗桌腿，并造成 visual、collision 和惯量不一致。

计划使用五组匹配的 box primitive：

```text
1 个桌面 box
4 个桌腿 box
```

每一组 visual 和 collision 使用相同中心与尺寸。这样可以直接核查：

- 桌面宽度；
- 桌腿厚度；
- 桌腿位置；
- 接触边位置；
- Viser/Isaac 显示与实际碰撞是否一致。

如果现有 Viser 只接受 OBJ 而不能显示 URDF primitive，必须先记录加载
限制，再选择以下最小方案之一：

1. 为宽桌增加一个与 collision 完全一致的简单 OBJ；
2. 对 viewer 增加 URDF primitive 支持。

在加载路径审计完成前不选择方案。

## 7. 原点、COM 与惯量

宽桌围绕 link origin 对称加宽。横向边界由原来的约
`[-0.261, +0.261] m` 变成 1.4 m 候选的 `[-0.700, +0.700] m`。

这保证：

- object reference pose 不因加宽平移；
- 推动方向接触边不移动；
- 横向几何仍关于原点对称；
- COM 仍位于横向中央对称平面。

最终 inertial origin 由桌面、桌腿和可选中央载荷的质量分布计算，
不强制等于 link origin。最终惯量必须使用各 box 自身惯量和
parallel-axis theorem 重新计算，不能沿用当前 `0.002` 的旧值。

几何预检允许使用明确标注的临时惯性参数，但这些参数：

- 不用于训练；
- 不进入论文指标；
- 不用于判断单机器人或双机器人能力；
- 在正式质量/摩擦能力扫描前必须替换。

## 8. 双机器人 reference 布局

不重新 retarget A1，也不修改 A1 关节角。两个机器人使用同一条关节
时间序列，只对机器人 root/body reference 施加刚性横向平移：

```text
agent_0_reference = A1 robot reference - d/2 lateral offset
agent_1_reference = A1 robot reference + d/2 lateral offset
shared_table_reference = original A1 object reference
```

其中 `d` 是几何预检后冻结的机器人中心间距。桌子 reference 只有
一条，两台机器人不能分别控制两条互相冲突的桌子轨迹。

## 9. Ghost teammate 分布

Ghost 分布只能在 `d` 冻结后确定。两台机器人朝向大致相同时，左右
机器人在自己的 heading frame 中应看到符号相反的横向 teammate
位置。

单机器人兼容训练按 50% 概率采样左右两种情况：

```text
relative position mean ≈ [0, +d] or [0, -d]
relative position noise = geometry preflight 后冻结
relative velocity mean = [0, 0]
relative velocity range = geometry preflight 后冻结
```

实际坐标分量和符号以 heading-frame 审计为准。不得把假设的 world
axis 直接写入 Actor 观测。

## 10. 几何验收门槛

### Viser

- 宽桌外观和尺寸正确；
- 两台 G1 可在同一侧同时显示；
- 两个 A1 手部区域都落在桌面或预期接触边范围；
- 桌腿位置自然，没有明显 visual mismatch。

### Isaac Sim

- reset 时无机器人-机器人 interpenetration；
- reset 时无机器人-桌子 interpenetration；
- 两台机器人脚部和下肢不被桌腿阻挡；
- primitive collision 与 visual 一致；
- 临时静态或零动作 rollout 中桌子无数值爆炸；
- 使用的机器人仍为 `main_mesh_collision_rubberhand.urdf`；
- 无 spherehand、halfspherehand 或 hemisphere-hand 活跃引用。

### Reference compatibility

- 桌子 root pose 与原 A1 object reference 一致；
- 桌面高度和推动侧边缘位置与原 A1 一致；
- 机器人横向平移不改变关节时间序列；
- 两个机器人共享同一条桌子 reference。

## 11. 验证记录

| 项目 | 状态 | 结果 |
|---|---|---|
| A1 object local/world 轴审计 | pending | - |
| Viser 对 URDF primitive 的支持 | pending | - |
| Isaac Sim 对宽桌 primitive 的加载 | pending | - |
| 1.4 m 候选视觉检查 | pending | - |
| 1.4 m 候选 collision 检查 | pending | - |
| 最终桌宽 | pending | - |
| 最终机器人中心间距 | pending | - |
| Ghost 位置/速度范围 | pending | - |
| 正式固定质量、COM、惯量和摩擦 | deferred | Stage 4 capacity sweep |

## 12. 下一步

1. 只读解析 A1 NPZ 的首帧 object quaternion 和轨迹方向。
2. 核对 Isaac Sim URDF 导入和坐标变换。
3. 核对 Viser 的对象 visual 加载路径。
4. 在新增资产前报告确认后的横向局部轴。
5. 经确认后创建首个 1.4 m primitive 宽桌候选。
