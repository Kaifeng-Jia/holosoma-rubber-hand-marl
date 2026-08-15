# Stage 1B-1：宽桌单实体 A1 保持性验证

日期：2026-08-15

分支：`rubber_hand_marl_baseline`
结论：**名义左右侧 gate 未通过；停止在 Stage 1B-1，不训练，不进入双实体 Stage 1B-2/Stage 2。**

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

## 8. 左侧失败的初步解释

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

录制是 action 前状态，而 termination 在随后 physics step 后计算，因此现有
NPZ 不能对每次越界作严格的 post-physics 归因。不过失败前状态已经显示：

- 左侧 32 个记录到的 done transition 中，14 个在 action 前的 object position
  error 已超过 0.25 m；另有 1 个 object orientation error 已超过 0.8 rad；
- 脚踝/手腕 z tracking 没有超过 0.25 m；
- torso-z proxy 没有超过 0.5 m。

这支持“主要是物体轨迹/力矩失配”的判断，但 exact termination attribution
仍需在 post-physics、reset 前记录各 termination term。

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

## 10. Gate 后的下一步

当前不应：

- 运行 50/500 次适应训练；
- 用训练掩盖 reference/layout 错误；
- 加入 ghost 随机化；
- 进入两个实体机器人或正式 MARL。

下一轮应依次执行：

1. 在 reset 前记录每个 termination term 的 post-physics 判定；
2. 审计左右手接触位置、接触力与相对桌子 COM 的 yaw moment；
3. 只读比较两个候选，而不是立即修改正式布局：
   - table-centerline 对称布局；
   - 对一侧使用经过验证的镜像 reference；
4. 与用户确认采用哪一个最小修正；
5. 重新执行同一 20-attempt 左右 nominal gate；
6. 只有左右都通过，才决定观测缩放、reset variation 和 Stage 1B-2。

## 11. 持久化产物

原始录制、JSON 指标和生成 reference 放在 gitignored 目录：

```text
logs/WholeBodyTracking/stage1b_a1_retention_v1/
```

该目录不是 Git commit 的一部分，但应随 worktree 保留。代码、资产契约、
测试和本报告进入 Git；`main` 与其他 worktree 不受影响。
