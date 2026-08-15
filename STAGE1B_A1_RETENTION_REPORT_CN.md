# Stage 1B-1：宽桌单实体 A1 保持性验证

日期：2026-08-15

分支：`rubber_hand_marl_baseline`
结论：**名义左右侧 gate 未通过；几何与镜像排查已完成。2026-08-15 执行的
50-iteration 单左侧 A1 适应 smoke 同样未通过，并严重损害右侧保持性。不得继续
500 iterations，也不得进入双实体 Stage 1B-2/Stage 2。**

## 1. 本阶段要回答的问题

Stage 1B-1 不是协作训练。环境中只有一台具有实体、碰撞和控制的
rubber-hand G1；另一台机器人仅由 ghost 观测表示。本阶段只回答：

> 将冻结的 A1 actor 放到未来双机器人所需的 1.4 m 宽桌布局，并加入
> 队友观测接口后，左右两侧是否都能保持原 A1 能力？

本阶段不修改：

- A1 checkpoint；
- PPO；
- WBT reward；
- WBT termination 阈值；
- 原始 `main` 或其他 worktree；
- rubber hand 的碰撞资产。

## 2. 变量隔离

### 2.1 原桌基准

原 A1 使用：

- `objects_largetable.urdf`；
- 桌面约 `0.5219528 m x 0.5219528 m`；
- 质量 `0.1 kg`；
- 原 A1 单机器人参考轨迹。

### 2.2 Stage 1B retention 宽桌

新增 `objects_widetable_a1_retention.urdf`，它不是最终训练桌。它只用于
隔离“横向加宽”这一变量：

- 桌面宽度：`1.4 m`；
- 桌面深度、高度、推动侧边缘和 object origin 与原 A1 保持一致；
- 总质量继续保持 `0.1 kg`；
- 原 A1 的 contact/dynamics 参数原样保留；
- COM：`(0, 0.015111745244133, 0)`；
- 惯量根据 1.4 m 五箱体模型计算，并由 1 kg 预检值按质量线性缩放：
  - `ixx = 0.0031387487608351`；
  - `iyy = 0.0218020759603284`；
  - `izz = 0.0197524457212314`。

原 `objects_widetable.urdf` 仍是 1 kg 几何预检资产，没有被覆盖，也没有
用于本次保持性结论。

## 3. 左右 A1 参考的生成

两侧机器人中心间距固定为 `0.8 m`。对于每一帧，先由桌子 quaternion
求出 table-local-X 在世界坐标中的单位向量 `e_x(t)`，然后生成：

```text
left_offset(t)  = -0.4 * e_x(t)
right_offset(t) = +0.4 * e_x(t)
```

平移被一致地加入所有机器人平移通道：

- root position；
- 所有 body world positions；
- root linear velocity；
- 所有 body world linear velocities。

关节角、关节角速度、body orientation 以及所有 object 通道保持不变。
速度偏移由位置偏移按 50 Hz 做有限差分得到；多轨迹情况下不得跨 clip
边界做差分。

生成脚本：

```text
scripts/prepare_stage1b_a1_retention.py
```

运行时生成的左右 NPZ 和 SHA-256 记录在：

```text
logs/WholeBodyTracking/stage1b_a1_retention_v1/generated_references/manifest.json
```

## 4. Ghost 观测

trajectory ghost 表示对侧虚拟机器人沿另一条移位 A1 reference 运动。对
当前实体机器人，观测为：

```text
relative_position_b = yaw_inverse(self_orientation)
                      * (ghost_position_w - self_position_w)

relative_velocity_b = yaw_inverse(self_orientation)
                      * (ghost_velocity_w - self_velocity_w)
```

actor 新增四维的顺序固定为：

```text
[relative_position_b_x,
 relative_position_b_y,
 relative_velocity_b_x,
 relative_velocity_b_y]
```

ghost 不具有实体，不碰撞、不施力、不执行策略。冻结 checkpoint 的新四列
输入权重全部为零，因此理论上无论 ghost 数值如何，actor 输出都应与零
ghost 完全一致。

## 5. 测试协议

### 5.1 650-step 配对检查

固定 seed 42，分别在左右侧比较：

- 宽桌 + zero ghost；
- 宽桌 + trajectory ghost。

结果：

| 侧别 | action 最大绝对差 | joint state 最大绝对差 | object position 最大绝对差 |
|---|---:|---:|---:|
| left | 0.0 | 0.0 | 0.0 |
| right | 0.0 | 0.0 | 0.0 |

这证明 ghost 数值没有改变冻结 A1 的动作或物理轨迹。

### 5.2 正式保持性检查

固定条件：

- seed：42；
- 单环境、Isaac Sim、CUDA；
- 每组 6500 个 policy steps；
- 分析前 20 个 closed attempts；
- checkpoint：`model_07999_actor158.pt`；
- 不训练；
- 不启用额外随机 ghost 扰动。

Stage 1B gate 要求宽桌完成率至少保留原桌完成率的 90%。原桌为 80%，
因此连续比例门槛为 72%；20 次离散样本下至少需要 `15/20 = 75%`。

## 6. 正式结果

| 指标 | 原桌 + zero ghost | 宽桌 left + trajectory ghost | 宽桌 right + trajectory ghost |
|---|---:|---:|---:|
| closed attempts | 20 | 20 | 20 |
| completed | 16 | 1 | 17 |
| completion rate | 80% | 5% | 85% |
| early termination rate | 20% | 95% | 15% |
| joint position RMSE | 0.1732 rad | 0.1833 rad | 0.1757 rad |
| key-body position RMSE | 0.0330 m | 0.0362 m | 0.0329 m |
| object position RMSE | 0.0526 m | 0.0649 m | 0.0637 m |
| object orientation mean error | 8.39 deg | 12.74 deg | 11.41 deg |
| interaction proxy success | 100% | 100% | 100% |
| fall proxy rate | 0% | 0% | 0% |

相对原桌完成率的保持比例：

```text
left  = 0.05 / 0.80 = 6.25%
right = 0.85 / 0.80 = 106.25%
```

因此：

- 右侧通过；
- 左侧严重失败；
- Stage 1B-1 的双侧整体 gate 未通过。

## 7. 当前能够排除什么

### 7.1 不是 ghost 输入直接改变了动作

zero/trajectory ghost 的配对轨迹逐元素一致，最大差为 0.0。冻结 actor 的
新增权重确实为零，因此左侧失败不是 ghost velocity 非零造成的。

### 7.2 不是典型的整机动作崩溃

左侧的 joint/body tracking RMSE 只比原桌轻微增加，fall proxy 仍为 0%，
并且每次 attempt 都出现手部接触和正确方向的物体运动代理。机器人仍在
执行 A1 并与桌子交互，但物体轨迹更容易越过 WBT 的终止边界。

### 7.3 不是“宽桌必然无法使用”

同一宽桌、同一 checkpoint、同一 seed 的右侧达到 85%，高于原桌 80%。
问题具有明确的一侧性。

## 8. 左侧失败的精确诊断

当前左右 reference 只做平移，不做镜像。A1 原动作本身并非左右镜像对称；
两只手的接触、发力时序、身体重心和脚步也可能存在偏置。把同一个非镜像
动作平移到桌子中心两侧后，接触点相对桌子 COM 改变，产生的平面力矩为：

```text
tau = r_contact x F_contact
```

左右侧的 `r_contact` 符号相反，但 `F_contact` 和整机姿态没有镜像，所以
桌子的 yaw/横向响应并不保证对称，也不保证都与原 object reference 一致。

此外，现行布局保持的是“两台机器人 root 的中点等于原 A1 root”，不是
“两台机器人相对桌子中心严格位于 +/-0.4 m”。原 A1 root 相对 table-local-X
本身具有偏置：

```text
original mean local-X = +0.0402 m
left mean local-X     = -0.3598 m
right mean local-X    = +0.4402 m
```

这会进一步放大左右接触几何不对称。

### 8.1 exact termination 与接触通道

录制器已经扩展为在 physics step 后、环境 reset 前保存与
`BadTrackingZOnly` 完全相同的阈值操作数和布尔原因，包括：

- root-z 与 projected-gravity error；
- 四个关键 body 的 z error；
- object position/orientation error；
- rubber-hand body origin、net contact force，以及相对桌子 COM 的 yaw-moment proxy。

在 650-step 精确诊断中，原 translation-only 左侧只出现一次 reset，原因仅为
`object_pos > 0.25 m`；root、orientation、关键 body 均未触发。右侧为零次
reset。这确认主要失配发生在机器人对桌子的闭环作用，而不是机器人自身先
摔倒或不再跟踪 A1。

### 8.2 已排除的几何候选

以下结果均使用同一 checkpoint、同一宽桌与 exact termination 通道；数值为
650 policy steps 内 reset 次数：

| 候选 | left resets | right resets | left 主要原因 |
|---|---:|---:|---|
| 原 translation-only 布局 | 1 | 0 | object position |
| 桌心严格 `±0.4 m` | 3 | 0 | object position/orientation |
| 桌心严格 `±0.35 m` | 4 | 0 | object position/orientation |
| 仅镜像左侧 reference | 13 | 0 | object position |
| 镜像 reference + policy I/O，确定性 | 4 | 0 | object position |
| 固定镜像平面 + policy I/O，确定性 | 7 | 0 | object position，另有少量 robot tracking |

由此可以排除：

- ghost 数值；
- 原轨迹相对桌心 `+0.0402 m` 的均值偏置；
- 把间距从 `0.8 m` 缩到 `0.7 m`；
- 单纯镜像 reference；
- 在冻结单侧 actor 外增加一个纯几何 policy-I/O 镜像层。

### 8.3 为什么镜像层仍不能解决

冻结 A1 actor 的 154 维原始观测由 actions、base angular velocity、DOF
position/velocity、reference joint position/velocity 和 6D reference orientation
组成；新增四维 teammate 输入的权重为零。修正 6D rotation 的列布局后，离线
checkpoint 推理能够以不超过 `7.6e-6` 的误差复现录制动作。

第 0 步敏感性分解显示：左/右动作最大差为 `0.987`；只把三维
`base_ang_vel` 换成右侧值，动作差降到 `5.9e-5`，再换 6D orientation 后约为
`1e-6`。但让初始角速度更接近右侧的映射并未改善完整 rollout：动态镜像从
4 次 reset 变为 6 次，固定镜像平面为 7 次。这说明冻结 actor 只在原单侧
训练分布上可靠；瞬时观测等价不等于接触动力学和整段闭环等价。

因此当前缺口不是某一个还没找到的平移量或符号，而是**单侧 WBT 策略没有
学到跨侧分布**。继续叠加几何补丁的收益已经低于进行有界适应训练。

## 9. 队友观测范围

正式 rollout 的原始 buffer 范围：

| 侧别 | position abs max | velocity abs max |
|---|---|---|
| left | `[0.5382, 0.9361] m` | `[0.6757, 0.9814] m/s` |
| right | `[0.5453, 1.0902] m` | `[0.6918, 1.0600] m/s` |

当前 observation term 采用 `[-1,1]` 裁剪。右侧正式 rollout 中：

- position 坐标样本约 1.79% 超过 1 m；
- velocity 坐标样本约 0.023% 超过 1 m/s。

这不影响当前冻结 actor，但正式 teammate-aware training 前，应明确选择：

- 保留米制输入并把裁剪扩大到约 `[-1.5,1.5]`；或
- 使用固定物理尺度，例如 position 除以 1.5–2.0 m，再裁剪到 `[-1,1]`。

本阶段不擅自冻结该选择。

## 10. 50-iteration 单左侧适应 smoke

### 10.1 冻结配置

- 初始化：`model_07999_actor158.pt`；
- 布局：桌心严格 `±0.4 m`，训练使用 centered left reference；
- 环境：4096 个并行环境、24 steps/environment/update、seed 42；
- 预算：50 PPO iterations，共 `4,915,200` transitions；
- 宽桌、WBT reward、termination 与原 WBT randomization 保持不变；
- teammate position/velocity observation scale 都设为 0；
- checkpoint 不加载旧 optimizer，使用新 optimizer；
- actor 新增四列在训练后仍严格为 0，最大绝对值为 `0.0`。

训练产物仅作为失败诊断保留：

```text
logs/WholeBodyTracking/20260815_191438-stage1b_a1_left_adapt50_seed42-locomotion/model_08048.pt
SHA-256: 444fba042c31d2478cb531e5d4216632c7bd0bc98b7eab108302e85611c5d615
```

### 10.2 在线信号与正式 gate

在线训练的 mean reward 从 `1.05` 增加到 `10.04`，末段 mean episode length
达到 `174.93`。这些信号看似改善，但不能替代冻结 checkpoint 的完整轨迹测试。
首次 PPO update 的 KL 为 `10.321`，随后 adaptive schedule 才把 KL 拉回约
`0.01`；这表明首次更新已经发生过大的 policy drift。

同一 `model_08048.pt`、seed 42、单环境、6500 steps、前 20 个 closed attempts：

| 指标 | centered left | centered right |
|---|---:|---:|
| completed | 2/20 | 2/20 |
| completion rate | 10% | 10% |
| early termination rate | 90% | 90% |
| fall proxy rate | 85% | 90% |
| joint position RMSE | 0.2743 rad | 0.2737 rad |
| key-body position RMSE | 0.0731 m | 0.0758 m |
| object position RMSE | 0.0756 m | 0.0706 m |
| object orientation mean error | 12.31 deg | 11.63 deg |
| hand-contact attempt rate | 90% | 95% |
| interaction proxy success | 80% | 80% |

前 20 attempts 的精确终止原因计数如下；同一 attempt 可以同时触发多个原因：

| 原因 | left | right |
|---|---:|---:|
| reference root position | 2 | 4 |
| reference root orientation | 0 | 0 |
| key-body position | 1 | 0 |
| object position | 12 | 14 |
| object orientation | 3 | 0 |

相同 centered 布局的 650-step 回归更直接：冻结 checkpoint 为 left 3 resets、
right 0 resets；适配 checkpoint 变为 left 4 resets、right 7 resets。单左侧适配
没有改善同协议左侧短 rollout，并明确破坏了右侧能力。

### 10.3 决策

- `model_08048.pt` 不升级为 baseline；
- 不从它继续 500 iterations；
- 冻结基线仍是 `model_07999_actor158.pt`；
- 不加入实体队友、ghost 随机化或正式 MARL；
- 下一次适配必须从 `07999` 重新开始，同时保留 centered left/right 训练分布，
  并在实施前冻结更严格的首次更新约束；
- 下一轮采用逐级微型 gate，而不是把“50 iterations”继续当成小预算：先验证
  单次更新的 KL、左右 650-step 回归，再决定是否增加更新次数。

具体采用左右 reference 混合、原侧 replay 比例、actor anchor 或更低且有上限的
学习率，属于下一轮实现决策；在讨论并冻结配置前不擅自选择。

## 11. 方案 2：双侧平衡、低更新幅度适配

### 11.1 配置与训练稳定性

本轮从冻结的 `model_07999_actor158.pt` 重新开始，不继承失败的
`model_08048.pt`。配置为：

- centered left/right reference 按 `0.5/0.5` 采样；
- 4096 environments，24 steps/environment/update；
- actor/critic learning rate 及上下限都固定为 `1e-5`；
- 1 learning epoch、4 mini-batches、PPO clip `0.05`；
- teammate position/velocity observation scale 仍为 0；
- rubber-hand G1、1.4 m retention table、原 WBT reward/termination/randomization。

第一次更新的左右 occupancy 为 `50.79%/49.21%`，KL 为 `0.0003`；
扫描到第 5 次更新时 KL 一直约为 `0.0001--0.0003`。这说明方案 2
解决了方案 1 首次更新 KL `10.321` 的 policy drift，也没有再出现
右侧灾难性遗忘。

### 11.2 0--5 次更新的 650-step 左侧扫描

| PPO updates | resets | joint RMSE (rad) | key-body RMSE (m) | object RMSE (m) |
|---:|---:|---:|---:|---:|
| 0（冻结 `07999`） | 3 | 0.186177 | 0.064986 | 0.099534 |
| 1 | 4 | 0.185756 | 0.065193 | 0.093381 |
| 2 | 4 | 0.193578 | 0.094371 | 0.111405 |
| 3 | 3 | 0.183393 | 0.063461 | 0.108613 |
| 4 | 3 | 0.186553 | 0.064230 | 0.094722 |
| 5 | 4 | 0.186891 | 0.063498 | 0.112345 |

第 4 次更新 `model_08002.pt` 是短 gate 的最佳折中点：左侧 reset 数和
类型与冻结基线一致，key-body 与 object RMSE 改善；右侧仍为 0 resets，
object RMSE 从 `0.070519 m` 改善为 `0.067419 m`。其 SHA-256 为：

```text
cce6aad3a3906ed5d28e3610592ee284109f739f2fb141b42679c0d3e6c8cd69
```

### 11.3 `model_08002.pt` 的 6500-step 正式 gate

同一 checkpoint、seed 42、单环境、无评估扰动，分别取前 20 个 closed
attempts：

| 指标 | centered left | centered right |
|---|---:|---:|
| completed | 0/20 | 18/20 |
| completion rate | 0% | 90% |
| early termination rate | 100% | 10% |
| fall proxy rate | 5% | 10% |
| joint position RMSE | 0.1900 rad | 0.1741 rad |
| key-body position RMSE | 0.0427 m | 0.0318 m |
| object position RMSE | 0.0612 m | 0.0644 m |
| hand-contact attempt rate | 100% | 100% |
| interaction proxy success | 85% | 100% |
| correct displacement direction | 18/18 evaluable | 20/20 |
| mean actual table displacement | 0.826 m | 1.516 m |

前 20 attempts 的精确终止原因如下；同一 attempt 可以同时触发多个原因：

| 原因 | left | right |
|---|---:|---:|
| reference root position | 1 | 0 |
| reference root orientation | 0 | 0 |
| key-body position | 1 | 0 |
| object position | 7 | 2 |
| object orientation | 11 | 0 |

### 11.4 决策

- 方案 2 在优化稳定性和右侧保留上成功，但双侧正式 gate 失败；
- `model_08002.pt` 仅作为诊断候选，不升级为 baseline；
- 冻结 baseline 仍是 `model_07999_actor158.pt`；
- 不继续通过单纯增加 PPO updates 推进：0--5 次扫描已经证明指标非单调，
  且第 4 次的正式左侧完成率为 0%；
- 左侧失败不等于“不会推”：手接触、交互和位移方向都大部分正确，
  主要是 object position/orientation WBT gate 无法跟随左侧 reference 到轨迹末端；
- 下一轮应先重新审视左侧 reference/object trajectory 与 Stage 1B 成功定义的兼容性，
  而不是继续扩大训练量。

## 12. 持久化产物

原始录制、JSON 指标和生成 reference 放在 gitignored 目录：

```text
logs/WholeBodyTracking/stage1b_a1_retention_v1/
```

该目录不是 Git commit 的一部分，但应随 worktree 保留。代码、资产契约、
测试和本报告进入 Git；`main` 与其他 worktree 不受影响。
