# 多智能体涌现主路线图

## 1. 文档地位与当前状态

- 状态：唯一有效执行指南
- 最近更新：2026-08-28
- 分支：`rubber_hand_marl_baseline`
- 基线提交：`8038c092`（`wbt-four-action-priors-v1`）
- 当前唯一方案：**Plan 5——按动作分网的 reference-guided 多智能体强化学习**
- 当前阶段：Push 与 Pull 两个独立协作 baseline 已完成；Push 后续训练暂缓，Pull 已通过用户
  人工动作质量验收。Demo 3 对角桌腿竞争式拉拽与 Demo 4 协作旋转长桌均已完成相互隔离的
  reference、环境、MAPPO、正式 train/resume、actor-only evaluate/record 和真实 Isaac/CUDA
  验证，**两者都尚未开始正式训练**。Demo 3 正式管线已提交并推送为 `ca893dba`。用户确认租用
  一张 RTX 4090，Demo 3 与 Demo 4 均使用 `4096 env × 24 steps`，先做容量验证，再在同一张 GPU
  上顺序训练；不把两个 Demo 混成一个 policy。
- 机器人：Unitree G1 29-DoF，固定 rubber hand
- 活动实验：Push、Pull、Demo 3 与 Demo 4 使用相互隔离的网络、reference、checkpoint 和日志

本文件取代此前所有总路线文档。技术细节可以保留在专项文档中，但不得
建立与本文件并行的“另一份总路线图”。

执行过程中，如下事项存在不确定性时必须先与用户讨论：

- actor/critic observation、参数共享方式或网络维度；
- reference、reward、termination、reset 或 success 定义；
- 桌子几何、质量、惯量、摩擦或碰撞语义；
- checkpoint 初始化、训练预算或 checkpoint 晋升；
- cooperative/competitive 场景的任务定义；
- 任何可能把当前问题重新变成混合动作或 task-oriented 单智能体研究的改动。

只读检查、测试和已明确批准的小型 smoke 可以直接进行。重大实现前必须
说明修改文件、文件职责、修改内容和回退条件。

## 2. 研究目标

核心问题是：

> 能否把独立训练的单智能体 WBT 动作先验迁移到多智能体强化学习中，
> 使同质 G1 在局部观测下，通过共享策略和全局 critic，对同一物体产生
> 可量化的合作或竞争涌现行为？

第一目标是建立强、清晰、可复现的 baseline，而不是立即叠加创新模块。

主要假设：

1. WBT 初始化比相同架构从零训练收敛更快或最终表现更好。
2. 联合多智能体训练优于把同一个单智能体 actor 简单复制给两台机器人。
3. 合作场景中，两台机器人都对物体运动产生可测量的因果贡献。
4. 竞争场景中，在不预设角色的情况下会出现争抢、阻挡、让位或接触点切换等行为。

## 3. 老师确认的范围

### 3.1 每种动作使用独立网络

Push A1、Push Plan B、Kick、Pull 等动作不混入同一个 actor。每种动作保留
自己的 reference、checkpoint 和后续多智能体网络。

“每种动作独立网络”与“同一动作内参数共享”并不矛盾：

```text
Push 网络：多个 push agent 共享一套 Push actor 参数
Pull 网络：多个 pull agent 共享另一套 Pull actor 参数
```

Push A1 是第一个已建立的合作 baseline。Pull 现在作为第二个独立合作动作推进，
可以在独立配置、独立日志和独立 GPU 上与 Push 并行训练，但两者不得混为一个
actor、一个 rollout buffer 或一个 checkpoint。竞争性物体争抢在合作环境和评测
工具稳定后加入。Plan B 与 Kick 保留为后续动作实验，不是当前 Push/Pull baseline
的依赖。

### 3.2 停止的方向

以下内容不属于当前主线：

- push/kick/pull 混合动作训练；
- skill-conditioned actor、CVAE、动作蒸馏或混合专家；
- 单智能体任意目标点搬运；
- 单智能体 target-oriented reward、方向泛化或 reference annealing；
- 左右单侧 checkpoint 分工；
- 继续为简单横移的单侧 A1 reference 增加 PPO iterations；
- 通用物体结构理解、视觉感知或 sim-to-real。

这些方向以后可以作为独立研究问题重新提出，但不得悄悄进入 Plan 5 baseline。

### 3.3 实验场景

Plan 5 包含两类实验：

1. **合作场景**：两台或多台同质机器人共同推动或拉动物体。
2. **竞争场景**：多个机器人围绕同一物体进行争抢、占位或控制权竞争。

主线顺序与并行边界：

```text
双机器人 Push A1 合作 baseline ─┐
                                 ├─ 各自动作内训练与评测成熟后 -> 竞争性物体争抢
双机器人 Pull 合作 baseline ─────┘
```

Push 与 Pull 可以同时占用两张 GPU，但它们仍是两个独立实验，不是 mixed-action
training，也不是 task-oriented 单智能体目标点搬运。

## 4. 已冻结资产与不变量

### 4.1 四个单动作 WBT 先验

四个独立 8,000-iteration WBT checkpoint 已在
`STEP1_BASELINE.md` 和 `STEP1_MOTION_MANIFEST.yaml` 中冻结：

| 动作 | 冻结 checkpoint | 用途 |
|---|---|---|
| Push A1 | `20260728_051638.../model_07999.pt` | 首个 Plan 5 baseline |
| Push Plan B | `20260728_132654.../model_07999.pt` | 后续 push 对照 |
| Kick | `20260806_055921.../model_07999.pt` | 后续独立动作实验 |
| Pull | `20260807_071217.../model_07999.pt` | 第二个合作动作 |

这些 checkpoint 是 reference-guided PPO/WBT 先验，不是 behavioral cloning，
也不是最终多智能体模型。

### 4.2 Rubber-hand 不变量

所有活动配置、reference、训练、评估和可视化只允许使用 fixed rubber hand。
Hemisphere、half-sphere 和 sphere-hand 资产不得进入本分支的活动命令或配置。

### 4.3 Push 宽桌与站位

首个双机器人布局冻结为：

- 桌面横向宽度：`1.4 m`；
- 两台机器人中心间距：`0.8 m`；
- 相对中央 A1 reference 的横向偏移：`-0.4 m / +0.4 m`；
- 两台机器人位于同一推动侧；
- 一条共享桌子 reference 和一条共享 motion phase。

宽桌只沿与推动方向垂直的 local X 轴加宽，保持桌面高度、推动方向深度、
接触边和 object origin。详细几何契约保留在
`WIDETABLE_GEOMETRY_DESIGN_CN.md`。

正式训练宽桌的物理参数冻结为：

- 总质量：`20 kg`；
- COM：object frame 中 `[0, 0.015111745244133, 0] m`；
- 惯量对角：`[0.62774975216702, 4.36041519206568, 3.95048914424628] kg·m²`，
  非对角项为零；
- 所有五个碰撞形状的 static/dynamic friction：`0.5 / 0.5`；
- restitution：`0.0`；
- 首轮正式 baseline 不做质量、惯量、COM 或材料 domain randomization。

`0.1 kg`、运行时 friction `1.0 / 1.0` 的 preflight 资产只用于一步环境 smoke，禁止进入
正式训练。Stage 4 的 capacity calibration 用于测量上述固定物理设置下的能力与因果贡献，
不得在未确认的情况下改写这组 baseline 常量。

### 4.4 Pull paired reference、z-wide 桌与物理契约

Pull 不沿用 Push 的 local-X 横向平移规则，也不使用 Push 固定的 `0.8 m`
agent spacing。冻结的单机器人 Pull 在桌子 local-X 方向拉动；其双机器人合作
横向轴是 **table-local-Z**，镜像面是 **table-local-XY**。

冻结的 Pull paired reference 由已接受的单机器人物理 rollout attempt 8 离线生成：

1. 在初始桌坐标系中，把实测桌轨迹与其 table-local-Z 镜像轨迹取对称平均；
   平移使用算术平均，姿态使用两条相对旋转的 SO(3) 测地线中点。因此运行时只有
   一条共享桌轨迹，不为两台机器人维护两条冲突的 object reference。
2. 把原机器人姿态运输到该共享桌轨迹，并在 table-local-XY 平面上做完整 G1
   左右镜像，包含 root、左右关节交换和符号变换。
3. 两台机器人分别沿 table-local-Z 向外偏移 `0.4390236 m`。该值严格来自
   `0.6742736 - 0.23525`，用于把原小桌腿位置对齐到 z-wide 桌腿位置；不再叠加
   Push 的 `lateral_spacing_m`。

reference 契约冻结为：

- `316` 帧、`50 Hz`；两台机器人共享同一个 phase；
- robot joint/body/velocity 通道均为 `[frame, 2 agents, ...]`，桌子通道只保留一份；
- 共享桌完整 reference 的平面净位移为 `0.584278 m`；对称化后的初始桌坐标系
  local-Z 漂移数值上为零（最大绝对值约 `1.23e-16 m`）；
- agent 的 table-local-Z 间距不是固定常量，参考内 min/mean/max 为
  `1.0520 / 1.4215 / 1.6259 m`；
- 活动机器人资产只能是 `main_mesh_collision_rubberhand.urdf`，不得出现
  hemisphere、half-sphere 或 sphere hand。

Pull 专用正式桌资产为
`objects_widetable_plan5_pull_training.urdf`：

- tabletop 尺寸：`0.5219528 × 0.04745 × 1.4 m`；
- 桌腿坐标：local-X `±0.23525 m`，local-Z `±0.6742736 m`；
- 总质量：`20 kg`；COM：`[0, 0.015111745244133, 0] m`；
- 惯量对角：`[3.95048914424628, 4.36041519206568, 0.62774975216702] kg·m²`；
- 五个碰撞形状的 static/dynamic friction：`0.5 / 0.5`；restitution：`0.0`；
- 首轮不进行质量、惯量、COM 或材料 domain randomization。

冻结产物及完整 SHA256：

| 产物 | 路径 | SHA256 |
|---|---|---|
| 单机物理来源 attempt 8 | `.../pull_7999_seed42_attempt08_success_viser_qpos.npz` | `d5c0705cfc3e643e75c9a9b1b28a21473b8b042134bd37ae935c7192c179bcdc` |
| ViSER paired reference | `logs/Plan5Pull/reference_preview/pull_attempt08_mirrored_pair_widetable_viser.npz` | `adf41371577bf32c606f3ab6caa07b2ac1826bf8e6dce8f524467aa6674812d6` |
| 显式 A/B/shared-table qpos | `logs/Plan5Pull/reference_preview/plan5_attempt08_mirrored_pair_qpos.npz` | `fab0d3ab51ab83495abfce39acf9c6bbc15caebd32f4b0db12acd66e538c9c65` |
| 完整 runtime reference | `holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/pull/plan5_attempt08_mirrored_pair_runtime.npz` | `571f23055f06648fb30b7abe5d90746a80c8555461ef91b6a23bccd9ad8404d4` |
| Pull z-wide URDF | `holosoma/data/motions/g1_29dof/whole_body_tracking/objects_widetable_plan5_pull_training.urdf` | `c260652d23e31dc0978a167137c45727cddb156eebedab77c9a40aa30079a8c2` |
| 原 Pull WBT checkpoint | `.../20260807_071217-rubberhand_pull_sub3_010_8000_seed42-locomotion/model_07999.pt` | `fc5a5d3b66bc076598b28f6c1e8ad822798ce6ce44c9f0370056dc8ad44495dd` |
| Pull `158-D` checkpoint | `logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt` | `f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364` |
| `158-D` 转换 manifest | `logs/WholeBodyTracking/marl_compat_pull_v1/conversion_manifest.json` | `ab30286729948e3c102479dab2199e5fbc99674fe5757bd6add14ea2c3fa2f3e` |

可复现生成链固定为：用 `synthesize_plan5_pull_reference.py` 的 `--output` 与
`--qpos-output` 从同一 attempt 8 同时写出 ViSER 和显式 qpos，再把该 qpos 交给
`synthesize_plan5_pull_runtime_reference.py` 扩展为正式 runtime reference。正式 NPZ 的
provenance 只记录仓库相对路径与源/模型 SHA，不写入本机绝对路径。

### 4.5 158 维 actor 兼容接口

Stage 1A 已把冻结 A1 actor 从 154 维无损扩为 158 维：

```text
原始 WBT actor observation      154
队友/对手相对平面位置             2
队友/对手相对平面速度             2
总 actor observation             158
```

冻结转换 checkpoint：

`logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt`

转换保持原 154 列、原 normalizer 和原 deterministic actor 输出不变；新增
四列初始权重为零。实现与验证提交：`eec8fa19`。

Pull 使用同一接口规则。其 `154→158` 转换相对原 Pull `model_07999.pt` 的
deterministic actor 输出最大绝对误差为 `0.0`；原 154 列与 normalizer 保持不变，
新增 teammate 四列首层权重严格为零，旧 optimizer state 被移除。该转换只保证
初始化策略等价，不表示 frozen Pull actor 已适应双机器人 z-wide `20 kg` 物理环境。

## 5. Plan 5 架构契约

### 5.1 Actor

- 保留现有单机器人 actor 架构和 29 维连续关节位置 action。
- 同一动作内所有 agent 共享一套 actor 参数。
- 每台机器人分别调用 actor：

```text
actor(obs_agent_0[158]) -> action_agent_0[29]
actor(obs_agent_1[158]) -> action_agent_1[29]
```

- actor 不一次输出 58 维 action。
- actor 不接收 agent ID、预设角色或动作类别标签。
- 执行时 actor 只能使用自身局部信息和队友/对手的相对位置、速度。

### 5.2 队友/对手 observation

四个新增通道必须在单智能体兼容阶段与多智能体阶段保持相同顺序、单位和维度：

```text
relative_position_b_x
relative_position_b_y
relative_velocity_b_x
relative_velocity_b_y
```

位置单位为米，速度单位为米每秒，均表达在观察者自身 heading frame。

单智能体兼容阶段使用有界随机化的虚拟队友/对手状态；多智能体阶段替换为
另一台真实机器人的相对状态。具体随机化范围、clip 和 normalizer 处理在
Plan 5 设计讨论中冻结，不能沿用旧左右适应实验的范围而不重新审查。

初始 baseline 不向 actor 提供：

- 队友完整关节角；
- 队友 contact force；
- 全局桌子真值；
- 显式意图或角色；
- 对手未来动作。

### 5.3 Centralized critic

训练采用 centralized training、decentralized execution。

- actor 保持现有单机器人结构；
- critic 为多智能体环境重新建立，不复用旧 298 维单机器人 critic 权重；
- critic 可以观察两台机器人、共享桌子、reference phase 和必要的全局状态；
- critic 的 privileged information 不得进入 actor；
- actor 与 critic observation 必须分别记录维度和字段顺序。

是否把上一时刻 action、接触状态或对手状态加入 critic，需要在实现前逐项确认，
不默认堆叠所有可用 simulator truth。

### 5.4 Paired reference

每个动作任务各自使用一份同步的双机器人 reference：

```text
agent_0 reference
agent_1 reference
one shared table reference
one shared motion phase
```

不能让两台机器人独立采样两条 motion，也不能为同一张桌子维护两条相互冲突的
object reference。

reference 的作用是提供 WBT 先验和姿态约束，不是硬编码实际物理轨迹。真实桌子
只由两台机器人接触产生的物理力推动。

Push A1 继续使用原中央 A1 在 table-local-X 上左右平移得到的 paired reference；
Pull 使用 4.4 节冻结的显式 table-local-Z 镜像 runtime reference。二者只共享
Plan 5 的接口和算法结构，不共享 motion 数据、phase、rollout buffer 或 checkpoint。

### 5.5 Reward 与 termination

第一版只做多智能体化所必需的变化：

```text
team reward
  = shared table/reference tracking
  + mean(agent motion tracking and stability)
  - shared invalid/fall penalties
```

约束：

- 两个 agent 获得同一个 team reward；
- 不增加 hand-only reward；
- 不禁止腿、髋、膝或躯干的偶发接触；
- 不加入目标点、动作选择或角色奖励；
- reference 相似度、物理成功和接触语义分别评估；
- 任一机器人严重失稳或共享桌子进入无效状态时，环境联合 reset。

最终公式和权重必须从原 WBT reward 逐项审计后再确认。

### 5.6 Checkpoint 晋升标准

对应动作的 robot reference 始终保留在 WBT reward 中，但它是动作先验和软约束，
不是要求训练后逐关节完全复制的硬标准。Push 与 Pull 各自的 checkpoint 按以下
优先级独立晋升：

1. 共享桌子连续完成 reference 的比例、沿轨迹进度和位置误差；
2. 两台机器人在整个连续 rollout 中保持物理有效，没有失稳或非法状态；
3. 桌子运动来自真实接触，后续需证明双方具有因果贡献；
4. paired body/joint tracking、yaw 和非手部接触作为重要辅助诊断。

合理的双机器人动力学适应可以偏离 A1；但若偏离没有带来桌子任务收益，或导致机器人失稳，
仍然不能晋升。不得只用总 WBT reward、单步 KL 或 ViSER 几何外观代替任务完成度。

checkpoint 的正式晋升评测使用 deterministic actor mean，但 GPU 物理 rollout 不保证位级
复现。因此单次 seed 只用于 smoke 和定位；正式比较必须使用多个 seed，并对每个 seed 做少量
重复，报告完成率与误差分布，不能把单次终止帧当作稳定排序。

## 6. 已完成证据与关闭结论

### 6.1 已完成

- 四个独立动作 WBT 先验已冻结。
- A1 与 Pull 的 154→158 维 actor 转换均已通过严格等价验证。
- 1.4 m 宽桌和 0.8 m 双机器人布局已通过 Viser 与 Isaac reset 几何检查。
- Pull table-local-Z 镜像 paired reference、完整 runtime reference 和 z-wide 20 kg
  桌资产已冻结，并通过 ViSER 人工验收和一步 CUDA 环境 smoke。
- Rubber-hand 碰撞资产已验证。
- 双实体无训练 action-trace mechanics preflight 已完成。

### 6.2 已关闭的单侧路线

旧实验尝试让一台横移到桌子侧端的机器人复现中央 A1 桌子轨迹，包括单侧适应、
左右混合 reference、mirror/symmetry 诊断和 first-layer adapter。结论是：

- 简单平移后的单侧 reference 在几何上可显示，但在动力学上脆弱；
- 单机器人从侧端推动会产生明显额外偏航力矩；
- 增加 PPO iterations 不能解决 reference 与物理任务之间的不一致；
- 该路线不再是 Plan 5 的前置条件。

关键证据保留在 Git 历史：
`75cc65de`、`59471d90`、`6e7a54c1`、`e6e0949d`、`29a88833`。
活动代码和路线图不再维护这些实验分支。

### 6.3 双实体 mechanics preflight

`bb315ef1` 的无训练双实体回放得到：

| 指标 | 结果 |
|---|---:|
| reference 平面位移 | 1.506 m |
| 实际平面位移 | 0.818 m |
| 运动方向余弦 | 0.9997 |
| 最大桌面偏航 | 10.37° |
| 终点桌面偏航 | -0.20° |

桌子方向正确且终点偏航接近零，支持“两侧偏航力矩可相互抵消”的 Plan 5
力学假设。但两台机器人均失稳，开环偏离后出现大量非手部接触，因此直接复制
两条单机器人 action trace 被否决。下一步必须使用在线 actor 和联合训练。

该结果支持 Plan 5，但不是可部署策略，也不是 MARL 训练结果。

## 7. 执行阶段

### Stage 0——隔离主线

状态：完成。

- 独立 worktree 与分支已建立；
- 原 `main` 和其他 worktree 未被修改；
- 四动作 checkpoint 已冻结；
- 本文件是唯一总路线。

### Stage 1——兼容接口与几何

状态：完成。

- 154→158 维 actor 无损转换；
- teammate observation 字段固定；
- 宽桌与双机器人站位固定；
- 双实体 mechanics preflight 支持 Plan 5。

### Stage 2——最小双机器人在线环境

状态：完成；9.1 清单与退出条件均已通过。

工作：

1. 在一个 Isaac Sim environment 中创建两台 physical rubber-hand G1 和一张共享桌子。
2. 两次调用同一个 Push A1 actor，并把两组 29 维 action 分发给各自机器人。
3. 使用一份 paired reference 和共享 phase。
4. 用真实相对位置/速度替换 ghost buffers。
5. 建立全局 critic observation。
6. 实现 shared reward、joint reset、碰撞语义和双机器人 recorder。
7. 完成确定性 reset、channel、维度、reward-sign 和短 rollout 测试。

退出条件：

- 两台机器人与桌子无初始穿模；
- 两个 actor 输入均为 158 维且字段顺序一致；
- 两组 action 正确路由到对应机器人；
- critic 只能在训练端读取全局状态；
- 两台机器人和共享桌子的轨迹、接触与 termination 可被正确记录；
- 活动资产中没有 hemisphere/sphere hand。

### Stage 3——Push A1 MARL baseline

必要实验组：

1. 单机器人 frozen A1；
2. 两台机器人复制 frozen 158-D A1 actor，不训练；
3. 相同双机器人架构从零训练；
4. A1 初始化的 shared-actor MARL。

训练阶梯：

```text
50 iterations      环境、维度和 reward smoke
500 iterations     学习方向与稳定性检查
2,000 iterations   初步合作行为
8,000 iterations   第一版完整 baseline
```

上述节点是观测与记录里程碑，不是用少量 iterations 提前否决学习方法的硬 gate。
短程阶段只检查数据、资产、维度、梯度、数值和日志是否有效；只要这些实现有效性
条件成立，就按已确认预算收集足量学习曲线，再用固定物理回放判断策略质量。

任何正式训练启动前，必须先向用户明确报告并确认：

1. actor/critic 网络结构、输入输出维度和字段；
2. 初始化 checkpoint、训练参数与冻结参数；
3. environment 数、rollout 长度、seed、learning rate、schedule 和 iteration 预算；
4. reward 各项权重、termination/reset 阈值和 checkpoint gate；
5. 桌子质量、惯量、COM、table-ground friction、hand-table friction 和碰撞配置；
6. reset/domain randomization 范围及正式多 seed 重复评测协议。

Push 和 Pull 分别从各自严格等价的 `158-D` WBT actor 初始化，并分别建立全新的
centralized critic、optimizer、rollout storage、日志目录和 checkpoint 序列。用户于
2026-08-22 确认本地单 GPU 正式设置：每项 `2,048` environments、每 iteration 每环境
`24` steps、先 `50` iterations critic-only bootstrap，再 `8,000` iterations full-actor，
每 `1,000` iterations 保存。先训练 Pull，再干净重训 Push；Push 不加载旧 MARL
`model_13050.pt`。两项不得交叉加载 checkpoint 或混合数据。

### Stage 4——合作真实性与物理校准

冻结或扫描：

- 桌子质量、惯量与 COM；
- table-ground 和 hand-table friction；
- 初始间距与小幅 reset randomization；
- reference phase offset；
- 外部扰动和单 agent 失效条件。

核心指标：

- 完成率、位移、方向和偏航；
- 两台机器人稳定性；
- 每台机器人的接触 impulse 和桌子功率贡献；
- 去掉任一 agent 后的性能下降；
- 同步性、相位差和恢复能力；
- 非手部接触比例，仅作为语义报告，不作为硬 reward。

必须证明第二台机器人具有因果贡献，而不只是同场出现。

### Stage 5——Pull 与竞争场景

状态：Pull 首个完整 PPO baseline 已训练完成，seed 721 连续物理回放和用户人工动作质量
验收均通过；固定多 seed 统计仍待完成。

1. Pull 的独立 paired reference、runtime loader、z-wide 物理资产、`158-D` actor、
   配置和 CUDA smoke 已完成；
2. 正式设置已经确认：先在本地单 GPU 上完成 Pull，再干净重训 Push；两个动作必须使用
   独立 actor/critic、optimizer、rollout storage、日志和 checkpoint；
3. Push/Pull 各自形成稳定合作 baseline 后，再建立 competitive object-grabbing 环境；
4. 分动作比较 WBT 初始化与从零训练；
5. 量化角色分化、阻挡、争抢、让位和接触点切换等涌现行为。

创新模块只在 baseline 暴露明确限制后选择。

## 8. 已冻结的实施决策

2026-08-18 已确认以下八项：

- [x] 单智能体随机 teammate 输入只做短程接口与鲁棒性检查，不做长期适应训练。
- [x] 同一动作内使用 shared actor 和 decentralized execution；centralized critic 从零训练。
- [x] 首版使用左右平移的 paired A1 robot reference 和一条共享桌子 reference。
- [x] Pull 使用 table-local-Z 镜像的显式 paired runtime reference 和一条对称平均的共享桌轨迹；
  不套用 Push 的固定横向间距。
- [x] 使用简单 shared WBT reward，不加入显式分工、hand-only 或合作塑形奖励。
- [x] 任一机器人失效时 joint reset；允许并记录偶发非手部接触。
- [x] 冻结已有 actor normalizer，完整 actor 参与 MARL 更新。
- [x] 训练采用 `50 -> 500 -> 2,000 -> 8,000 iterations` 观测里程碑；短程结果不作为
  学习方法的提前否决 gate，只有实现有效性错误可中止运行。
- [x] `0.1 kg` 只用于环境 smoke；正式宽桌冻结为 `20 kg`、friction `0.5 / 0.5`、
  restitution `0.0`，首轮不做物理随机化。

## 9. 执行清单与结果记录

状态含义：`[ ]` 未开始，`[~]` 进行中，`[x]` 完成并通过 gate，`[!]` 完成但未通过。
每个完成项必须在本节记录日期、commit、测试或产物、定量结果和结论。

### 9.1 Stage 2——环境与数据流

- [x] 只读审计 simulator、environment、PPO、rollout storage 和 recorder 的现有结构。
- [x] 冻结精确的修改文件、模块边界和回退方案。
- [x] 建立 opt-in 双 articulation Isaac Sim 状态、接触、reset 与 torque 适配层。
- [x] 用真实 CUDA reset 验证两台 physical rubber-hand G1、共享宽桌和 collision graph。
- [x] 定义 shared actor batch 与两组 `29-D` action 的纯张量 shape/routing 契约。
- [x] 建立 per-agent actor 数据与 per-environment team 数据分离的 rollout storage。
- [x] 建立 opt-in 双机器人 ActionManager 控制项，把两组 `29-D` action 独立转换为 torque。
- [x] 建立最小 Plan 5 environment shell，接通 paired command、dual action 和双实体 simulator。
- [x] 把 shared actor batch 和 action routing 接入在线环境。
- [x] 实现每台机器人 `158-D` actor observation 和真实 teammate 相对状态。
- [x] 建立最小 centralized critic observation。
- [x] 建立全新 centralized critic 网络、normalizer 和 optimizer。
- [x] 建立 paired A1 reference 的内存张量契约和共享 object reference。
- [x] 建立 paired command，共享环境级 phase，并联合 reset 两台机器人和一张桌子。
- [x] 把 paired A1 reference 接入每台机器人的在线 observation。
- [x] 实现 shared reward、joint reset、termination 和碰撞语义。
- [x] 实现 actor checkpoint、冻结 normalizer、全新 critic/optimizer 的加载契约。
- [x] 扩展 recorder，区分两台机器人、共享桌子、接触和终止原因。
- [x] 完成确定性 reset、维度、坐标、action routing、reward sign 和短 rollout 测试：
  reset、基础 shape、间距、一步物理、heading-frame、agent-swap 和 critic shape 已通过；
  在线 actor/critic、11 项 reward sign/有限性和 20 步 CUDA rollout 已通过。
- [x] 完成单智能体随机 teammate observation 的短程鲁棒性检查。

### 9.2 Stage 3——Push A1 训练 gate

- [x] 打通 stochastic rollout、log-prob、team GAE、单次 PPO update 与 checkpoint round-trip。
- [!] `1 iteration` 训练入口 preflight：管线通过，但原 WBT `1e-3` actor LR 的稳定性未通过。
- [x] tuned preflight：`32 envs`、actor LR `1e-5`、fresh critic LR `1e-3` 通过 KL/drift gate。
- [x] `50 iterations`：环境与数值 smoke。
- [!] `50 iterations` 后三 seed 行为方向 gate：数值稳定，但 A1 姿态/总 reward 平均退化。
- [x] centralized critic-only `50 iterations` warm-up：actor/normalizer 逐张量不变，critic
  value loss 改善且数值稳定。
- [!] 解耦 actor/critic LR 并完成 `10 iterations` actor 解冻 gate：优化器机制正确，
  但三 seed A1 行为方向仍退化。
- [!] 仅训练 teammate 四列的 `10 iterations` gate：A1 基础参数严格不变，但 adaptive-KL
  将 actor LR 放大至 `6.57e-3`，adapter 过强且三 seed 行为退化。
- [!] 固定 actor LR `1e-5` 的 teammate 四列 gate：参数与步长安全，但三 seed reset 增加且
  global body orientation 退化，尚无合作改善证据。
- [x] 建立连续 object-reference evaluator：从 frame 0 运行至首次 termination 或 309 帧结束，
  不跨 reset，报告 completion、progress、planar/along/lateral error、yaw 与精确失败子原因。
- [!] 用新标准重评现有 checkpoint：full actor 10 iterations 当前领先，但无一 checkpoint
  完成 reference，暂不晋升。
- [x] 增加连续终止误差和脚踝/手腕逐 link Z 诊断；确认存在真实整体高度塌陷，也存在刚越过
  tracking threshold 的案例；deterministic actor mean 下 GPU rollout 仍非位级复现，正式
  gate 改用多 seed × 少量重复统计。
- [!] 正式 `20 kg` Frozen A1 基准：三 seed × 三次均未完成 reference，但 9/9 初始推动方向
  正确且均由机器人失稳终止；作为 formal full-actor 的固定对照，不视为配置错误。
- [x] 正式 `20 kg` centralized critic-only `50 iterations` warm-up：actor/normalizer 逐张量
  不变，critic value-loss 趋势改善，数值、checkpoint 和实际物理记录均通过 gate。
- [x] 正式 `20 kg` full-actor `50 iterations`：完成实现有效性和短程诊断；不再用作
  方法淘汰 gate。
- [x] `500/1,000 iterations`：完成第一段足量观测窗口，确认学习仍在变化且未收敛。
- [x] 下一轮独立 Push/Pull PPO 设置已确认：本地单 GPU 顺序运行、`2,048` environments、
  `50 critic-only + 8,000 full-actor`、每 `1,000` iterations 保存；Push 不加载旧 `13050`。
- [x] Pull `2,048 env × 1 iteration` 完整 actor+critic 容量验证通过；无 OOM/NaN，资产、
  维度、运行时物理与日志有效；该独立 checkpoint 禁止用于正式训练。
- [x] Pull `50 iterations` critic-only bootstrap 完成；actor 与 actor normalizer 逐张量不变，
  50 轮 policy drift、actor gradient 与 KL 均为零。
- [x] Pull `8,000 iterations` full-actor 正式训练完成；8 个千轮 checkpoint 齐全，最终
  `model_08050.pt` 完成 seed 721 的 316/316 帧连续物理回放。
- [x] Push 按相同规模从 A1 `158-D` WBT actor 干净重训：完成 `50 critic-only + 8,000
  full-actor iterations`，随后从 `model_08050.pt` 连续训练 7,000 iterations 至
  `model_15050.pt`；训练日志、checkpoint 和状态账本完整。
- [!] Push 后期 checkpoint 固定重复评测：`13050/14050/15050` 各做
  `3 seeds × 3 independent launches`。`13050` 为当前最佳但仅 `1/9` 完整成功，后两者
  均为 `0/9`；这说明策略仍需继续训练或后续受控诊断，不等于 Plan 5 路线失败。用户决定
  **需要续训，但当前暂停**，不得在未再次确认设置前自动启动。

训练中只有 NaN/Inf、错误资产或物理常量、维度/数据错接、checkpoint/日志损坏等
实现有效性问题可以提前停止；正常的低 reward、低完成率或动作偏差应记录为学习曲线，
不能在训练量不足时改写主线。

### 9.3 Stage 4——物理与合作真实性

- [ ] 在冻结的正式物理设置下完成单机器人/双机器人 capacity calibration。
- [x] 审计 preflight 宽桌的 PhysX 运行时质量、COM、惯量和材料参数。
- [x] 冻结正式桌子质量、惯量、COM 和摩擦，并完成 CUDA/PhysX 运行时校验。
- [ ] 完成 frozen-copy、scratch、WBT-initialized 三组双机器人对照。
- [~] 完成多 seed 正式评测：Push `13050/14050/15050` 已完成固定 frame-0 的
  `3 seeds × 3 repeats` 首轮重复性检查；稳健 Push 候选和 Pull 多 seed 统计仍待完成。
- [ ] 完成单 agent removal、接触 impulse、桌子功率贡献和非手接触报告。
- [ ] 证明第二台机器人具有可量化的因果贡献。

### 9.4 Stage 5——动作与场景扩展

- [x] 冻结 Pull attempt 8 来源、table-local-Z 镜像规则和 `0.4390236 m` outward offset。
- [x] 生成 316 帧、50 Hz 的 ViSER paired reference，并于 2026-08-22 完成人工批准。
- [x] 生成完整 Pull runtime reference，并通过 channel、name reorder、finite 和 quaternion 检查。
- [x] 建立 Pull z-wide 20 kg 桌资产，冻结 rubber-hand、`0.5/0.5` friction 和 `0` restitution。
- [x] 完成 Pull actor `154→158` 严格等价转换；新增 teammate 四列权重为零。
- [x] 完成 Pull 独立 command/smoke/baseline 配置与一步 CUDA smoke。
- [!] Frozen Pull actor 连续回放在 316 帧 reference 的 frame 156 触发
  `joint_bad_tracking`；该结果证明需要训练，不否决 reference 或 Plan 5。
- [x] 已向用户报告并确认独立 Push/Pull PPO 的网络、优化器、环境数、预算、reward、
  termination、物理和保存/评测协议。
- [x] 使用独立 Pull checkpoint 建立 Pull 合作 baseline；不与 Push 混合数据或 checkpoint。
- [ ] 建立 competitive object-grabbing 环境。
- [ ] 比较 WBT 初始化与从零训练。
- [ ] 量化争抢、阻挡、让位、接触点切换和控制权变化。

### 9.5 已完成记录

#### 2026-08-18：实施决策冻结

- commit：`bd4f1af0`；
- 结果：八项技术决策全部确认，Plan 5 从概念讨论进入执行；
- gate：通过；
- 下一项：只读代码审计。

#### 2026-08-18：Stage 2 只读代码审计

- commit：`bd4f1af0`；
- 检查范围：`BaseTask`、`WholeBodyTrackingManager`、Isaac Sim、`MotionCommand`、
  action/observation/reward/termination managers、PPO、rollout storage、recorder 和训练入口；
- 结果：现有框架以“每个环境一台 robot”为硬假设，不存在可直接启用的 MARL 路径；
- 证据：simulator 只维护一个 `_robot` articulation，WBT 与 PPO 张量均以
  `[num_envs, ...]` 表示一台机器人；已有双实体 mechanics preflight 证明两台
  rubber-hand G1 与共享桌的 Isaac Lab 物理创建可行；
- 决策：新增 Plan 5 专用双机器人 simulator/environment/MAPPO 适配层，保持原
  单机器人 WBT、标准 PPO 和 `main` 路径不变；actor 使用 per-agent 样本，team
  return/value 每个物理环境只计算一份；
- 测试：只读审计，无运行时修改；
- gate：通过；
- 下一项：冻结精确文件边界并实现无 Isaac 依赖的 shape/action-routing 基础层。

#### 2026-08-18：Shared actor batch/action shape 基础层

- commit：`1c72ff25`；
- 文件：`agents/mappo/batch_layout.py` 及其单元测试；
- 结果：固定 `2 agents x 158-D observation x 29-D action` 契约；
  `[env, agent, feature]` 与 shared-actor batch 可逆转换，agent 顺序不串线；
- 测试：`7 passed`，并通过 `py_compile` 与 `git diff --check`；
- gate：通过；
- 下一项：实现区分 per-agent actor 数据和 per-environment team 数据的 rollout storage。

#### 2026-08-18：Multi-agent rollout storage

- commit：`6d82d121`；
- 文件：`agents/mappo/storage.py` 及其单元测试；
- 结果：agent 字段使用 `[time, env, agent, ...]`，team/critic 字段使用
  `[time, env, ...]`；minibatch 先采样完整 environment transition，再展开 agent；
- 测试：MAPPO 基础层合计 `12 passed`，并通过 `py_compile` 与 `git diff --check`；
  当前环境未安装 `ruff`，未新增依赖；
- gate：通过；
- 下一项：实现 Plan 5 专用 Isaac Sim 双 articulation 状态与控制适配层。

#### 2026-08-18：双 articulation Isaac Sim 适配层（静态 gate）

- commit：`ed9e4f17`；
- 文件：`simulator/isaacsim/dual_robot_isaacsim.py`、Isaac Sim 扩展钩子、
  独立 simulator 配置和配置测试；
- 结果：普通 `isaacsim` 行为保持 opt-out；Plan 5 配置增加第二 articulation，
  并公开 `[env, agent, ...]` root/DOF/body/contact 张量及双机器人写入接口；
- 测试：MAPPO、配置和 rubber-hand 资产回归合计 `20 passed`，并通过
  `py_compile` 与 `git diff --check`；
- gate：静态 gate 通过，物理 gate 尚未执行；
- 下一项：建立最小 Plan 5 environment，使 paired reset 与 action routing 可驱动
  双 articulation，然后运行真实 CUDA reset smoke。

#### 2026-08-18：Paired A1 reference 张量契约

- commit：`c95780c6`；
- 文件：`envs/marl/paired_a1_reference.py` 及其单元测试；
- 结果：直接从冻结 A1 `MotionLoader` 在内存中生成左右 robot reference；
  每帧沿共享桌子 local X 平移 `-0.4/+0.4 m`，关节角与姿态不变，object
  reference 只保留一份，不生成冗余 NPZ；
- 测试：paired reference 与 MAPPO 基础层合计 `17 passed`，并通过
  `py_compile` 与 `git diff --check`；
- gate：纯张量 gate 通过，在线 reset/phase 尚未接入；
- 下一项：实现最小 Plan 5 environment orchestration。

#### 2026-08-18：双机器人 ActionManager 控制项

- commit：`18f4fe4c`；
- 文件：`managers/action/terms/marl.py`、独立 action preset 及单元测试；
- 结果：ActionManager 显式接收 agent-major `58-D` action，并在控制器内部保持
  `[env, 2 agents, 29 dofs]`；两台机器人分别使用自己的 DOF position/velocity
  计算 PD torque，再经 `apply_agent_torques()` 写入对应 articulation；普通单机器人
  action preset 未改变；
- 测试：本项与 MAPPO、paired reference、双机器人配置及 rubber-hand 资产回归合计
  `29 passed`；首次测试因遗漏项目 `PYTHONPATH` 未进入收集，补齐既定源码路径后全部通过；
- gate：控制项及纯张量路由通过，shared actor 与在线 environment 尚未接入；
- 下一项：实现 paired phase/reset 的在线 command，使两台机器人和共享桌子可联合 reset。

#### 2026-08-18：Paired A1 command（共享 phase 与联合 reset）

- commit：`59f248c2`；
- 文件：`managers/command/terms/marl.py`、独立 command preset 及单元测试；
- 结果：每个物理 environment 只维护一个 frame index；同一 frame 同时生成两份
  robot state，并仅生成和写入一份 object state；partial reset 只影响被选中的 environment；
  首个 physics gate 使用精确 reference reset，不提前混入 pose noise 或 adaptive sampler；
- 测试：paired command、原 WBT motion sampling、双机器人 action、paired reference 和
  MAPPO 基础层合计 `35 passed`，并通过 `py_compile` 与 `git diff --check`；
- gate：command 数据流通过；真实 Isaac Sim/CUDA joint reset 尚未执行，不计为 physical gate；
- 下一项：建立最小 Plan 5 environment shell，把 `[env, 2, 29]` actor action、paired
  command 和 DualRobotIsaacSim 接在一起，然后运行单环境 CUDA reset smoke。

#### 2026-08-18：最小双实体在线环境与真实 CUDA smoke

- commit：`070a0f60`；
- 文件：`envs/marl/plan5_push_manager.py`、Plan 5 smoke experiment、
  `scripts/smoke_plan5_environment.py`、simulator 初始化时序修正及对应测试；
- 环境：一个 Isaac Sim environment、两台 physical rubber-hand G1、一张共享 `0.1 kg`
  宽桌；`0.1 kg` 仍只用于环境 smoke，不代表正式训练物理参数；
- 结果：paired command 完成联合 reset；root shape 为 `[1, 2, 13]`，DOF shape 为
  `[1, 2, 29]`，两机器人间距为 `0.80000001 m`；输入一组 `[1, 2, 29]` 零动作后
  状态保持有限且未立即 reset；
- 资产：实际配置加载 `main_mesh_collision_rubberhand.urdf`，左右 rubber-hand link、
  STL 与 collision prim 存在，未发现 hemisphere/hemispherical 资产 token；
- 修正：最初主 contact sensor 在第二 articulation 引起的 stage recomposition 后未初始化；
  将主 contact sensor 延后到所有机器人 articulation 建立完成后创建，两套 sensor 均正常；
- 测试：相关 MAPPO、environment、dual action、paired command 和 simulator config 回归
  `34 passed`；`py_compile` 与 `git diff --check` 通过；真实 CUDA smoke `passed: true`；
- 边界：当前 smoke reward 刻意为空，因此一步 reward 为 `0.0`；这只证明物理环境和
  数据路由可运行，不表示 actor observation、centralized critic、shared reward 或 MAPPO 已完成；
- gate：最小双实体物理启动 gate 通过；
- 下一项：实现每台机器人 `158-D` actor observation，以真实 teammate 相对平面位置和
  速度替换 ghost 字段，并验证 heading-frame 坐标和 agent-swap 一致性。

#### 2026-08-18：每台机器人 `158-D` actor observation

- commit：`557e331b`；
- 文件：`managers/observation/terms/marl.py`、Plan 5 observation preset、smoke experiment
  与 observation/环境测试；
- 数据契约：每个 environment 输出 `actor_obs [E, 2, 154]` 和
  `teammate_obs [E, 2, 4]`；shared actor 边界拼接为 `[E, 2, 158]`，未把两台机器人
  合并成一个 316 维 actor 输入；
- 154 维兼容性：term 名称、顺序、scale、noise、clip 与原 Push A1 完全一致，只把
  单机器人数据 provider 替换为 per-agent provider；paired motion command 和 orientation
  分别读取同一共享 phase 下的两份 robot reference；
- teammate 语义：另一台真实机器人相对平面位置和相对平面速度，使用观察者自身 yaw
  转换到 heading frame；不向 actor 提供队友关节、contact force、角色或桌子全局真值；
- 测试：CPU 测试覆盖 90° yaw、agent-swap、154+4 维度和原配置继承，相关回归
  `46 passed`；`py_compile` 与 `git diff --check` 通过；
- 真实 CUDA：一步 smoke 输出 `[1, 2, 154] + [1, 2, 4] = [1, 2, 158]`，全部有限，
  `passed: true`；
- 边界：teammate term 当前继承兼容 checkpoint 的 `clip=(-1, 1)`，足以完成接口 gate，
  但正式训练前仍须依据 rollout 分布确认是否保留；critic、shared reward 和在线 actor
  checkpoint 尚未接入；
- gate：per-agent actor observation 与真实 teammate 坐标 gate 通过；
- 下一项：先确认最小 centralized critic 的字段集合，再实现全新 critic observation；
  不默认加入上一时刻 action、contact force 或其他冗余 simulator truth。

#### 2026-08-18：`527-D` centralized critic observation

- commit：`63a43b00`；
- 文件：Plan 5 observation preset、`managers/observation/terms/marl.py`、CUDA smoke 与测试；
- 数据契约：每个物理 environment 只输出一个 `critic_obs [E, 527]`，没有 agent 轴；
- 组成：一份共享 joint reference `58-D`、一份 normalized phase `1-D`、两台机器人各自
  physical/tracking state `2 × 228-D`，以及一份共享桌子 tracking state `12-D`；
- 去重：没有把两份原 `298-D` object-WBT critic 直接拼接；共享 motion command、phase
  和桌子只出现一次；桌子使用 actual-vs-reference position/orientation/velocity error；
- 权限边界：critic 没有 raw contact force、角色 ID 或额外动作历史；actor observation
  没有因此获得任何新的 privileged information；
- 测试：逐 term 维度相加为 `527`，并检查不存在 contact/role term；相关回归
  `47 passed`，`py_compile`、旧符号搜索和 `git diff --check` 通过；
- 真实 CUDA：一步 smoke 同时输出 actor `[1, 2, 158]` 和 centralized critic
  `[1, 527]`，全部有限，`passed: true`；
- 边界：本项只建立 critic 输入，尚未实例化新 critic 网络、critic normalizer、optimizer
  或 MAPPO 更新循环；
- gate：centralized critic observation gate 通过；
- 下一项：接通 shared actor batch、冻结 A1 actor normalizer、加载 `158-D` actor，
  并为 `527-D` 输入创建全新的 critic/normalizer/optimizer。

#### 2026-08-18：A1 actor warm-start 与全新 critic 初始化

- commit：`d5eb339f`；
- 文件：`agents/mappo/initialization.py` 及真实 checkpoint 回归测试；
- source：`model_07999_actor158.pt`，SHA256
  `11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224`，iteration `7999`；
- actor：严格加载完整 `158-D -> 29-D` state dict，新增四列保持零权重；所有 actor
  参数可训练，不加载旧 optimizer；
- actor normalizer：加载原 `158-D` mean/variance/count，强制 eval 且忽略 update 请求；
- critic：创建全新 `527-D -> 1-D` MLP，不读取 checkpoint 中的旧 `298-D` critic；
- critic normalizer/optimizer：均从零创建；actor optimizer 也从空 state 创建；
- fail-closed：checkpoint SHA256 或 compatibility version 不符时拒绝加载；
- 测试：真实 checkpoint、冻结统计量、全局 WBT config 不变、模型前向和空 optimizer
  state 共 `50 passed`，`py_compile` 与 `git diff --check` 通过；
- CUDA：actor `[4, 158] -> [4, 29]`、critic `[2, 527] -> [2, 1]` 前向均有限；
  actor normalizer 保持冻结，critic normalizer 从 count `0` 开始更新；
- 边界：模型 bundle 已正确建立，但尚未接入 environment rollout、team return/GAE 或 PPO update；
- gate：模型初始化与加载边界通过；
- 下一项：实现最小 MAPPO rollout orchestration，使同一个 actor 对 `[E,2,158]` 逐 agent
  执行，并把两组 action 送入现有 dual action routing；critic 每个 environment 只评估一次。

#### 2026-08-18：Shared actor 与 team critic 在线一步路由

- commit：`0da986f8`；
- 文件：`agents/mappo/runner.py`、runner 测试和升级后的 CUDA environment smoke；
- actor 数据流：`actor_obs [E,2,154] + teammate_obs [E,2,4] -> [E,2,158] ->
  [E×2,158] -> shared actor -> [E×2,29] -> [E,2,29]`；
- critic 数据流：`critic_obs [E,527] -> team critic -> value [E,1]`，每个物理 environment
  只评估一次，没有按 agent 复制 team value；
- 环境边界：runner 通过 `env.step({"actions": [E,2,29]})` 进入既有 dual action term，
  两台 articulation 仍分别计算和接收自己的 torque；
- 测试：runner shape、fail-closed 和 fake environment action 捕获纳入相关回归，
  `52 passed`，`py_compile` 与 `git diff --check` 通过；
- 真实 CUDA/Isaac：冻结 A1 actor 输出 `[1,2,29]`，新 critic 输出 `[1,1]`，实际推进
  一个 physics/control step 后状态全部有限且没有立即 reset，`passed: true`；
- 边界：当前 runner 是确定性在线数据流 gate；尚未实现 stochastic rollout、log-prob、
  shared reward、team return/GAE 或 PPO update；
- gate：shared actor 和 centralized critic 在线路由通过；
- 下一项：逐项映射原 WBT reward/termination 到 team 语义，先实现简单 shared reward 与
  joint termination，再进行多步无更新 rollout。

#### 2026-08-18：Shared WBT reward、joint termination 与短 rollout

- commit：`2c9efe2b`；
- reward：逐项保留原 Push A1 的 11 个 term、sigma 和权重；9 个机器人项先按 agent
  独立计算再取均值，2 个共享桌子项每个 environment 只计算一次；
- tracking：paired command 增加与原 `MotionCommand` 同义的 heading-aligned 相对身体
  position/orientation reference，没有改变 A1 reference、Actor 或 Critic；
- termination：保留原 `BadTrackingZOnly` 阈值；任一机器人 tracking 失败，或共享桌子
  position/orientation 越界，均对整个双机器人 environment joint reset；
- 碰撞语义：保留原 `undesired_contacts` 的 `-0.1` 轻量惩罚并在两台机器人之间取均值；
  偶发非手部接触不是 termination，也没有加入 hand-only、角色或合作 shaping；
- 配置：原一步物理 smoke 继续使用空 reward；新增 `g1_29dof_plan5_push_baseline`
  才启用正式 shared reward 和 joint termination，防止两个 gate 混淆；
- 测试：reward 权重/公式、agent 平均、共享 object、接触语义、任一机器人失效 joint reset、
  object reset 和无接触 termination 均纳入回归；Plan 5/MAPPO 相关测试 `73 passed`，
  `compileall` 与 `git diff --check` 通过；
- 真实 CUDA/Isaac：冻结 `model_07999_actor158.pt`，连续 20 个 control steps，无梯度、
  无 optimizer/normalizer 更新；Actor `[1,2,158] -> [1,2,29]`、Critic `[1,527] -> [1,1]`，
  所有状态和 reward term 有限，`reset_count=0`；
- reward 观测：总 reward 范围 `[-0.0583, 0.0720]`；启动负值来自 reset 后 previous action
  为零产生的原 A1 action-rate 瞬态；object position reward 为 `[0.7925, 0.9996]`，object
  orientation reward 为 `[0.8795, 0.9997]`；
- 资产：两台实体均为 `main_mesh_collision_rubberhand.urdf`，hemisphere token 为空；
- gate：shared reward、joint termination、collision semantics 与确定性短 rollout 通过；
- 边界：尚未扩展双机器人 recorder，也未实现 stochastic action、log-prob、team GAE 或
  PPO update；因此此结果不是训练成功或合作成功证明；
- 下一项：扩展 recorder，使其区分 robot 0、robot 1、共享桌子、双方接触与具体 termination
  原因，并用短 rollout 验证记录内容。

#### 2026-08-18：Paired recorder 与 termination 原因快照

- commit：`5c1827f1`；
- recorder：沿用并扩展现有 `EvalRecordingCallback`，检测 `paired_motion_command` 后记录
  agent-major 通道，不建立第二套重复 recorder；
- 双机器人通道：actor observation `[T,2,158]`，joint/action/torque `[T,2,29]`，root
  `[T,2,*]`，body/contact `[T,2,...]`；substep 明确为 `[T,2,decimation,29]`；
- 共享通道：桌子 reference/actual pose 与 velocity 只记录一次，shape 为 `[T,3/4]`，
  不按 agent 复制；shared reward、done、timeout 同样每个 environment 一份；
- termination：通用 `TerminationManager` 保存每个 term 本步结果，`BaseTask` 在 reset 前
  clone 到 `extras["termination_terms"]`；NPZ 分别记录 `termination_term_timeout` 和
  `termination_term_joint_bad_tracking`，避免 reset 后原因丢失；
- contact：双方 body-level net force/history 保留显式 agent 维；recording smoke 临时开启
  object-filter diagnostics，记录全身 table filter `[T,1,78,3]` 与 rubber-hand table filter
  `[T,1,4,3]`，metadata 中保存每个 filter 的 prim path；
- 传感器边界：额外 object-filter diagnostics 只在 `--record-output` smoke/eval 时启用，
  正式 baseline 配置仍默认关闭，不改变训练物理、reward、observation 或吞吐；
- 测试：paired NPZ shape、agent 顺序、共享 object 单份、termination pre-reset 快照和旧接口
  回归均通过；worktree 全部 24 个测试文件共 `158 passed`，`compileall` 与
  `git diff --check` 通过；
- 真实 CUDA/Isaac：冻结 A1 Actor 连续 5 步，成功写入并重新读取
  `/tmp/plan5_recorder_smoke.npz`；关键通道 shape 与双侧 `/Robot/`、`/Robot_1/` filter
  覆盖均通过脚本 fail-closed 自检，`reset_count=0`；
- gate：双机器人、共享桌子、双方接触与具体 termination 原因均可记录，recorder gate 通过；
- 边界：该 5 步文件是接口 smoke，不是训练结果或合作行为证据；临时 NPZ 不作为正式数据集；
- 下一项：完成单智能体随机 teammate observation 的短程鲁棒性检查；随后复核 Stage 2
  全部退出条件，再进入 stochastic rollout、team GAE 与 PPO update 实现。

#### 2026-08-18：单智能体随机 teammate 输入鲁棒性

- commit：`d076a641`；
- 验证对象：冻结的 `model_07999_actor158.pt`、原 Push A1 motion、`objects_largetable.urdf`
  和 `main_mesh_collision_rubberhand.urdf`；未使用 generic largebox 或 hemisphere hand；
- checkpoint 边界：154→158 转换后的四个 teammate 输入列保持严格零权重，actor normalizer
  对新增四列保持 identity，因此这是旧 A1 行为的向后兼容检查；
- CPU：对同一批 257 组 154-D A1 observation，分别拼接全零 teammate 和从
  `[-1,1]` 均匀采样的 teammate；真实 checkpoint 的 29-D action 逐元素严格相同，
  `rtol=0`、`atol=0`；
- 真实 CUDA/Isaac：单机器人连续执行随机 teammate 路径 20 个 control steps，最大采样
  绝对值 `0.9805`，相对全零路径的 `max_abs_action_difference=0.0`，`reset_count=0`，
  reward 范围 `[0.0196, 0.1148]`，状态、动作和 reward 全部有限；
- 测试：worktree 全套 `159 passed`，`compileall` 与 `git diff --check` 通过；
- gate：随机 teammate 输入不会污染冻结 A1 先验，Stage 2 全部 checklist 与退出条件通过；
- 边界：该结果不表示 actor 已学会使用 teammate 信息；MARL 更新后新增四列将变为非零，
  届时必须通过消融与因果评测确认其用途；
- 下一项：进入 Stage 3，先实现 stochastic action sampling、log-prob、team rollout storage、
  team return/GAE 与单次 PPO update 的闭环；通过无 NaN、shape、参数更新和 checkpoint
  round-trip gate 后，才启动 `50 iterations` 训练 smoke。

#### 2026-08-18：MAPPO 随机 rollout 与单次更新闭环

- commit：`44bbebb7`；
- 随机策略：shared Gaussian actor 对 `[E,2,158]` 分别采样，每个 agent 独立记录
  `action/log-prob/mean/sigma`，两组 29-D action 仍由 dual action routing 分发；
- team 数据：central critic、shared reward、joint done、value、return 和 advantage 均只保存
  一份 `[T,E,...]`，没有为两个 agent 复制 527-D privileged observation；
- GAE：沿用原 WBT PPO 的 `gamma`、`lambda` 和 timeout bootstrap；joint done 截断整个 team
  的递推，rollout 后才写入 return/advantage，未写入时 storage fail-closed；
- PPO：每个 agent 单独计算 probability ratio；同一 team advantage 按 agent 顺序复制两次；
  critic 每个 environment transition 只计算一次 value loss；沿用原 clipping、entropy、value
  coefficient、gradient clipping 和 adaptive-KL 学习率规则；
- checkpoint：保存 shared actor、central critic、两套 optimizer、冻结 actor normalizer、critic
  normalizer 和 Plan 5 维度/来源哈希；metadata 不匹配时拒绝恢复；
- 真实 CUDA/Isaac：`1 env × 2 agents × 4 steps`，无 reset；reward 范围
  `[-0.0371,0.0154]`，log-prob 范围 `[-21.3032,-18.2103]`；actor/critic 最大参数变化
  均为 `0.0010000`，actor normalizer 不变，checkpoint round-trip 通过；
- 首次单 mini-batch 在更新前计算得到 `KL=0`、平均 surrogate 数值为 `0`，属于新旧策略
  初始一致且标准化 advantage 均值为零的预期结果；actor gradient norm `87.45`，参数实际更新；
- 测试：worktree 全套 `165 passed`，`compileall` 与 `git diff --check` 通过；Ruff 未安装，
  未为此 gate 改动依赖；
- gate：训练数据流、team GAE、actor/critic 更新和可恢复 checkpoint 均通过；这仍不是合作行为
  或最终训练质量证据；
- 下一项：建立有界的 `50 iterations` 训练入口、日志和 checkpoint 保存，使用当前已冻结的
  paired A1 reference、shared WBT reward、joint reset、rubber-hand 双机器人和宽桌；训练后按
  数值稳定性、reference tracking、桌面运动、双方稳定性与接触记录决定是否进入 500 iterations。

#### 2026-08-18：有界训练入口与 1-iteration 稳定性诊断

- commit：`ac6d68e6`；
- 入口：`scripts/train_plan5_push.py` 支持 iterations、environment 数、rollout 长度、seed、
  output directory 和 resume；目录非空时 fail-closed，不覆盖既有结果；
- 产物：每轮写 `metrics.jsonl`，启动时写 `run_config.json`，关闭前写 `status.json`，保存
  初始和最终 resumable checkpoint；status 文件用于规避 Isaac 关闭流程可能吞掉异常退出码；
- 默认 smoke：seed `721`、`8 envs × 2 agents × 24 steps`；实际约 `2.0 s/iteration`，8 GB
  RTX 5070 Laptop GPU 可以运行；当前宽桌仍为仅限 smoke 的 `0.1 kg` 资产；
- 诊断增强：rollout 分开记录 timeout 与非-timeout reset；每轮在相同固定 observation 上
  比较更新前后 deterministic action，报告 mean/max absolute policy drift；
- 可重复结果：`reset_count=10`，其中 `timeout_count=0`、`tracking_failure_count=10`；reward
  mean `-0.0431`，范围 `[-0.3975,0.0844]`；
- 更新稳定性：平均 KL `12.3562`，远高于目标 `0.01`；adaptive schedule 在第一轮内把
  actor/critic LR 从 `1e-3` 降到下限 `1e-5`；固定 observation 的 deterministic action
  漂移 mean `0.3739`、max `2.5100`；
- 排除项：source actor noise std mean `0.50816`，更新后 `0.50857`，因此失败不是探索噪声
  突然坍缩；所有张量与 loss 有限，checkpoint 和日志完整，因此也不是 CUDA/数据流故障；
- gate 结论：训练入口和持久化通过，但原单智能体 WBT 的 `1e-3` actor learning rate 不能
  直接晋升到 50-iteration warm-start MARL；不得以增加 iterations 代替诊断；
- 测试：全套 `165 passed`，`compileall` 与 `git diff --check` 通过；
- 待确认的最小对照：优先保持 fresh critic LR `1e-3`，只把 pretrained actor LR 降为
  `1e-4` 并做同一 seed 的 1-iteration 对照；若 KL/漂移仍过大，再试 `1e-5`。另一条对照是
  增加并行环境数后复测原 `1e-3`，但 1e-3 已直接造成大幅 mean shift，当前优先级较低。

#### 2026-08-18：Warm-start learning-rate 与 batch 对照

- commit：`693e2a64`；训练入口增加独立的 actor/critic 初始 LR 参数，未改变 adaptive-KL
  规则；fresh critic 可以从较高 LR 开始，但仍随 actor KL 一起自适应调整；
- 所有对照使用同一 seed `721`、相同 paired A1 reference、reward、termination、24 steps/env、
  rubber-hand 双机器人与 `0.1 kg` smoke 宽桌；
- `8 envs, actor=1e-4, critic=1e-3`：KL `0.11737`，mean/max drift
  `0.03038/0.19434`；比原 `1e-3` 明显改善，但 KL 仍约为目标的 11.7 倍；
- `8 envs, actor=1e-5, critic=1e-3`：KL `0.02536`，mean/max drift
  `0.01645/0.07943`；接近但仍高于 adaptive 上边界 `0.02`；
- `32 envs, actor=1e-5, critic=1e-3`：每轮 768 个 team transitions、1536 个 actor samples；
  KL `0.01536`，mean/max drift `0.01560/0.07762`；actor LR 保持 `1e-5`，critic LR 自适应到
  `2.96e-4`，没有降到下限；约 `2.37 s/iteration`，8 GB GPU 可容纳；
- tracking failure：32-env 更新前 rollout 为 `35/768=4.56%`；第一轮 rollout 发生在任何参数
  更新前，因此这里只作为 frozen prior 的随机采样基线，不用于评价新 LR 的学习成效；
- gate：冻结首个 50-iteration smoke 为 `seed=721, num_envs=32, steps_per_env=24,
  actor_lr=1e-5, critic_lr=1e-3`；当前结论仅适用于数值 smoke，不冻结正式物理训练参数。

#### 2026-08-18：50-iteration 数值 smoke 与行为方向评测

- 训练产物：`logs/Plan5Push/a1_mappo_smoke50_seed721_lr1e5_env32/`；50 行 JSONL、
  `run_config.json`、`status.json`、初始与 iteration-50 resumable checkpoint 均完整；
- checkpoint SHA256：初始 `02e3f8be6e7be57e7164823ceeedbd4273a7567b1c427412d474f372c29bccfa`；
  iteration 50 `93d9cb36c69ad97ed213a71015b7e41c0094bbfde6081e44a906744159e297d0`；
- 数值稳定性：所有指标有限；前 10→后 10 的 reward mean `-0.03918→-0.03538`，value
  loss `0.10184→0.08507`，KL mean `0.01841→0.01855`；KL 全程范围
  `[0.01536,0.02257]`；
- 稳定性边界：tracking failures 前 10/后 10 分别为每轮 `43.8/44.1`，没有爆炸也没有
  改善；actor LR 保持 `1e-5`，critic LR 最终被 adaptive-KL 联动降至 `1e-5`；
- 评测支持 commit：`2b868082`；现有 CUDA smoke 可加载 MAPPO training checkpoint，报告
  reward mean、各 term mean 和具体 termination count，并支持显式 seed；
- deterministic 行为评测：seeds `721/722/723`，source 与 iteration 50 各运行 100 steps；
  两者全部状态有限且仅发生 `joint_bad_tracking` reset；
- 三 seed 平均 final−source：总 reward mean `-0.00424`，global body orientation
  `-0.10098`，relative body orientation `-0.07360`，relative body position `-0.04950`，
  object position `+0.01210`，object orientation `-0.01744`，undesired contacts `+0.10667`，
  reset count `-0.33`；
- 权重证据：teammate 四列从 L2 `0` 增至 `0.05505`，说明策略开始使用新输入；旧 154 列
  的累计 delta L2 为 `0.27081`，整个 actor delta L2 为 `0.48854`，说明 A1 先验也发生了
  不可忽略的累计漂移；
- 当时 gate 结论：50-iteration “环境与数值 smoke”通过，但“行为学习方向”未通过，并曾据此
  暂停进入 500 iterations。该短程淘汰规则已被 2026-08-19 的“学习实验判定方式修正”取代，
  当前只把本结果保留为早期诊断证据；
- 待讨论：下一步必须优先限制 actor 对旧 A1 prior 的累计漂移，同时让 fresh central critic
  获得足够学习速率；任何 actor/critic 分离调度、critic warm-up 或 KL early-stop 改动都需先确认。

#### 2026-08-18：Centralized critic-only warm-up

- 实现 commit：`242dc5e8`；`Plan5PPO.update(update_actor=False)` 在不执行 actor forward、
  backward、optimizer step 或 adaptive-KL 调度的情况下，只训练 centralized critic；训练入口
  通过 `--critic-only` 显式启用，默认联合训练路径保持不变；
- 测试：单元测试证明 actor 参数逐张量不变、critic 参数发生更新、actor gradient 与 KL 为零，
  actor/critic LR 不被联动调整；worktree 全套 `166 passed`；
- 训练配置：从原 A1 warm-start 开始，seed `721`、`32 envs × 24 steps`、actor LR 字段
  `1e-5`（冻结、无 optimizer state）、critic LR `1e-3`；先跑 10 iterations 验证，再从
  `model_00010.pt` 原位恢复到 iteration 50；使用的 `0.1 kg` 宽桌仍只属于 smoke；
- 产物：`logs/Plan5Push/a1_critic_warmup10_seed721_env32/`；共 50 行 JSONL，status
  `passed: true`；iteration-50 checkpoint SHA256
  `7615e80148ab136cfd3e569b60570c1ea75d797ec064f39e7c852fb0f7b83ffc`；
- 参数证据：iteration 0→50 的 actor 和冻结 actor normalizer 逐张量严格相同；actor optimizer
  state 始终为空；critic 参数 delta L2 `15.6118`、最大绝对变化 `0.26423`，critic optimizer
  state 从 `0` 增至 `8`；
- 数值证据：所有指标有限；value loss 前 10/后 10 均值 `0.07492→0.05631`，下降约
  `24.8%`；critic grad norm 前 10/后 10 均值 `0.3122→0.4407`，全程有限；actor grad、
  KL、mean/max policy drift 全程严格为 `0`，critic LR 保持 `1e-3`；
- 行为边界：reward mean 前 10/后 10 为 `-0.03954→-0.03977`，tracking failure 每轮均值
  `41.9→44.6`。这是预期的：actor 完全冻结，因此本阶段只验证 critic 学习，不宣称行为改善；
- gate：critic-only warm-up 通过。下一步不是直接训练 500 iterations，而是先确认短程 actor
  解冻方案，尤其要解决原 adaptive-KL 同时缩放 actor/critic LR、导致 critic LR 被 actor KL
  拖到 `1e-5` 的耦合问题；确认后再从 iteration-50 critic-warmed checkpoint 做小型对照。

#### 2026-08-18：学习率解耦与 10-iteration actor 解冻 gate

- 实现 commit：`424c64b4`；Plan 5 MAPPO 的 adaptive policy-KL 只调节 actor optimizer，
  不再改变独立 centralized critic optimizer 的 LR；原单智能体 PPO 路径未修改；
- 测试：高 KL 与低 KL 两个方向均验证 actor LR 按规则变化而 critic LR 逐值不变；worktree
  全套 `168 passed`，并通过 `compileall` 与 `git diff --check`；
- 训练配置：从 critic-only `model_00050.pt` 恢复，seed `721`、`32 envs × 24 steps`、
  actor LR `1e-5`、critic LR `1e-3`，联合更新 10 iterations 至 iteration 60；
- 产物：`logs/Plan5Push/a1_actor_unfreeze10_from_critic50_seed721_env32/`；status
  `passed: true`；iteration-60 checkpoint SHA256
  `859cc9e887460b665977a1d9116c8580ef42f0d261efc7c2f7cad2886c135e4f`；
- 数值结果：所有指标有限；KL mean `0.01794`、范围 `[0.01624,0.02160]`；actor LR
  全程 `1e-5`，critic LR 全程 `1e-3`；value loss 首次/末次 `0.08147→0.05543`；
  每轮 deterministic policy drift mean 平均 `0.01575`；
- 参数结果：iteration 50→60 actor delta L2 `0.21379`、最大绝对变化 `0.00211`；第一层旧
  154 列 delta L2 `0.12250`，teammate 四列从零增长至 L2 `0.02264`；critic delta L2
  `7.98588`；
- 行为评测：冻结 A1 与 iteration 60 分别在 seeds `721/722/723` 做 100-step 确定性
  rollout，六次均有限；final−source 总 reward 分别为 `-0.01743/-0.00542/+0.01242`，
  三 seed 平均 `-0.00348`；reset 平均 `2.33→2.00`；
- 三 seed raw-term 平均 final−source：global body orientation `-0.05257`、relative body
  orientation `-0.02833`、relative body position `-0.02624`、object orientation `+0.03291`、
  object position `-0.00010`、undesired contacts `+0.06667`；
- 当时 gate：学习率解耦在机制上成功，但行为方向仍未通过。critic 得以继续学习，并没有自动阻止
  actor 改坏 A1 tracking；当时暂停到 50 或 500 iterations。该暂停规则后来已被足量训练原则
  取代；本项仍用于说明 fresh critic 初期可能造成的 actor drift。下一步曾讨论更直接的 A1 prior
  保护方式，而不是继续堆叠 PPO 更新量。

#### 2026-08-18：Teammate-input-only adapter gate

- 实现 commit：`b52e5463`；网络结构不变，仅允许第一层新增 teammate 输入列 `154:158`
  接收 actor 梯度；旧 154 列、其余层、bias 与 action noise 在 optimizer step 后强制逐张量恢复；
- 测试：四列发生更新、全部冻结参数严格不变、teammate 为零时 deterministic action 与原 A1
  完全一致；训练入口记录 `actor_update_mode=teammate_input_only`；全套 `170 passed`；
- 训练配置：从 critic-only `model_00050.pt` 恢复，seed `721`、`32 envs × 24 steps`、
  初始 actor LR `1e-5`、critic LR `1e-3`，训练 10 iterations 至 iteration 60；
- 产物：`logs/Plan5Push/a1_teammate_adapter10_from_critic50_seed721_env32/`；status
  `passed: true`；iteration-60 checkpoint SHA256
  `b648d895019cc002c7bb056bd8088c837c6088d2d6fc0353f518aec5267550ed`；
- 硬不变量：iteration 50→60 的旧 154 列和其余 actor state 逐张量严格相同；唯一变化键为
  `actor_module.module.0.weight`，四列最终 L2 `3.11210`、最大绝对值 `0.24611`；
- 调度诊断：受限 adapter 在首轮产生低 KL，原 adaptive-KL 在每个 minibatch 连续把 LR 乘
  `1.5`；actor LR 首轮已达到 `6.568e-3`，十轮范围 `[5.138e-5,6.568e-3]`，最终
  `4.444e-3`；KL mean 仍为 `0.01123`。因此本轮没有实际维持名义上的 `1e-5` 小步更新；
- 行为评测：复用完全相同的 frozen A1 seeds `721/722/723` 基准，adapter checkpoint 分别做
  100-step 确定性 rollout；final−source 总 reward 为 `-0.00422/-0.01162/-0.00320`，三 seed
  平均 `-0.00634`；reset 平均保持 `2.33`；
- 三 seed raw-term 平均 final−source：global body orientation `-0.00880`、relative body
  orientation `-0.02208`、relative body position `-0.03243`、object orientation `+0.01858`、
  object position `+0.01135`、undesired contacts `+0.16000`；
- gate：未通过。该结果否定的是“受限 adapter 继续沿用无上限 adaptive-KL”，尚未否定四列
  adapter 本身；最小后续对照是把 actor LR 固定在 `1e-5` 后重跑相同 10 iterations，但此项
  需先确认，不能把本次失败直接归因于 adapter 容量不足。

#### 2026-08-18：Fixed-LR teammate adapter gate

- 实现 commit：`7b8b36de`；`--teammate-input-only` 要求显式 actor LR，并把 Plan 5 PPO
  schedule 固定为 `fixed`；run config 记录实际 update mode 与 schedule；普通 full-actor 和
  critic-only 模式不变；
- 测试：真实 teammate-only PPO update 验证 fixed schedule 下 actor/critic LR 均保持不变；
  全套 `171 passed`，并通过 `compileall` 与 `git diff --check`；
- 训练配置：从 critic-only `model_00050.pt` 恢复，seed `721`、`32 envs × 24 steps`、
  actor LR 固定 `1e-5`、critic LR 固定 `1e-3`，训练 10 iterations 至 iteration 60；
- 产物：`logs/Plan5Push/a1_teammate_adapter_fixed1e5_10_from_critic50_seed721_env32/`；
  iteration-60 checkpoint SHA256
  `08a777540fba427c98396485c26cc13ca54290fca13cd7d9bcaad7e072dbe0c5`；
- 数值结果：所有指标有限；actor LR 和 critic LR 全程分别严格为 `1e-5/1e-3`；KL mean
  `1.83e-7`；每轮 deterministic drift mean 平均 `9.10e-5`、max 平均 `8.69e-4`；value
  loss 首次/末次 `0.08147→0.05589`；
- 参数不变量：旧 154 列和其余 actor state 逐张量严格相同；四列最终 L2 `0.01342`、最大
  绝对值 `9.46e-4`，相较 adaptive 版本的 L2 `3.11210` 已消除步长膨胀；
- 行为评测：seeds `721/722/723` 的 100-step deterministic final−source reward 分别为
  `-0.00259/+0.00419/-0.00329`，三 seed 平均 `-0.00056`；reset 平均 `2.33→3.00`；
- 三 seed raw-term 平均 final−source：global body orientation `-0.04191`、relative body
  orientation `-0.00020`、relative body position `-0.01095`、object orientation `+0.01750`、
  object position `+0.00062`、undesired contacts `+0.03333`；
- 当时 gate：参数安全 gate 通过，但行为方向 gate 未通过。结果比 full actor 和 adaptive adapter
  更接近 frozen A1，却没有稳定提升 team reward，且 reset 与 global body orientation 变差；
  当时暂停进入 50 或 500 iterations。该短程暂停规则现已废止，但仍不能把“几乎保住 A1”
  误写成“已学会合作”。

#### 2026-08-18：Object-centric evaluator 与现有 checkpoint 重评

- 实现 commit：`7b86e3b2`；新增纯张量 object trajectory metrics，并扩展 CUDA evaluator：
  每次从 frame 0 连续运行，首次 termination 立即停止，绝不把 reset 后的新 episode 混入；
- 指标：309 帧 reference completion、平面/along/lateral error、实际路径和净位移、沿轨迹进度、
  方向余弦与 yaw；termination 在 reset 前区分 robot ref height、robot orientation、tracked-body
  height、object position 和 object orientation；
- 正确性：reference frame 必须严格为 `0,1,2,...`；纯指标与 termination 测试、全套回归
  `174 passed`；真实 CUDA frozen-A1 preflight 正确识别 seed 721 在 frame 16 因 object position
  error 超过 `0.25 m` 终止，而不是机器人跌倒；
- 评测协议：seeds `721/722/723`，完整 reference 上限 309 帧；critic-warmed actor 与 frozen
  A1 逐张量相同，因此不重复；adaptive adapter 因 LR 膨胀属于无效配置，不参与候选比较；

| checkpoint | 平均完成帧 | completion | 沿轨迹进度 | 方向余弦 | planar RMSE | 首次失败主因 |
|---|---:|---:|---:|---:|---:|---|
| Frozen A1 | 20.33 | 6.58% | 0.164 m | 0.862 | 0.115 m | 2 object-pos / 1 robot-height |
| Full actor 10 | 20.67 | 6.69% | **0.217 m** | **0.947** | 0.145 m | 2 robot-height / 1 object-pos |
| Full actor 50 | **32.67** | **10.57%** | 0.091 m | 0.446 | 0.112 m | 3 robot-height |
| Fixed adapter 10 | 25.00 | 8.09% | 0.126 m | **0.947** | **0.082 m** | 3 robot-height |

- 解释：full actor 10 的 A1 姿态退化并非全无任务收益；它把平均沿轨迹进度提高约 `32%`，
  因而按新标准是当前最有希望的候选。full actor 50 虽存活更久，但 seed 721 出现负进度，
  说明增加更新量没有形成一致的正确方向；adapter 的低 RMSE部分来自移动较少，不能单独判优；
- termination 子因：所有 robot failure 均来自 reference-body height 或 tracked-body Z threshold；
  没有 robot orientation、object orientation 或 timeout failure。它们可能包含真实下沉/失稳，
  也可能包含仍然稳定但偏离 A1 的姿态，下一步需用连续物理高度和倾角数值区分；
- gate：四组均为 `0/3` 完成，距 309 帧目标很远，均不通过。下一步不是立即加 A1 anchor，
  而是先量化 robot-height failure 的物理严重程度，并审计 object-position termination 与当前
  reference/reward 是否给出了可学习的连续信号；之后再决定继续 full actor 还是调整 termination。

#### 2026-08-18：连续终止误差与重复性诊断

- 实现：`JointBadTrackingZOnly` 只增加 reset 前连续诊断，不改变任何 reward、termination 条件
  或阈值；evaluator 现在报告 reference/actual root height、gravity-Z、四个受监控 link 的逐项
  Z 误差、桌子位置/姿态误差及对应阈值；
- 受监控 link：`left/right_ankle_roll_link` 与 `left/right_wrist_yaw_link`。因此原先笼统的
  `tracked-body Z` 可能表示脚部失稳，也可能只是手腕偏离 A1，必须逐项解释；
- 真实失稳证据：seed 721 中 Frozen A1 和 full actor 10 的 0 号机器人参考 root height 均约
  `0.73 m`，实际分别约 `0.22/0.18 m`，误差 `0.513/0.546 m`，超过 `0.50 m` 阈值；这不是
  轻微 A1 姿态差，而是整体高度塌陷；
- 边界案例：首轮 seed 722/723 的 tracked-body 最大 Z 误差分别出现 `0.2504/0.2949 m`
  （Frozen）和 `0.2674 m`（full actor 10），相对 `0.25 m` 阈值从仅超 `0.4 mm` 到超
  `44.9 mm` 不等；是否允许这类偏离必须结合具体 link 和机器人稳定性判断，不能统一放宽；
- 重复性：相同 checkpoint 与 seed 再次启动 Isaac/PhysX 后，若干案例的终止帧和主因发生变化；
  runner 已确认使用 deterministic `act_inference` 均值动作，不是策略采样噪声。因此单次
  GPU rollout 只作为诊断样本，不能独立决定 checkpoint 晋升；
- 测试：termination 定向测试 `4 passed`；worktree 全套回归 `174 passed`，并通过
  `compileall` 与 `git diff --check`；
- gate：诊断工具通过。现阶段没有证据支持简单删除 robot tracking termination，也没有许可
  继续 500 iterations；下一步先冻结正式桌子物理参数，再用多 seed × 少量重复协议比较候选。

#### 2026-08-18：Preflight 宽桌运行时物理审计

- evaluator 直接读取 PhysX view 的 mass、COM pose、3×3 inertia 和每个 collision shape 的
  material，不再仅依赖 URDF 文本推断运行时参数；
- 1-step CUDA preflight 通过；实际质量为 `0.1 kg`，COM 为
  `[0, 0.015111745, 0] m`，惯量对角为
  `[0.003138749, 0.021802075, 0.019752447] kg·m²`，均与 preflight URDF 一致；
- 五个桌子 collision shape 的 PhysX material 均为 static friction `1.0`、dynamic friction
  `1.0`、restitution `0.0`。URDF 中的 `0.9` contact 值没有成为运行时材料；此前不能把
  Plan 5 smoke 的实际摩擦报告为 `0.9`；
- 当前 Plan 5 的 randomization manager 为空，因此上述质量和材料没有被 domain randomization
  改写；
- gate：运行时审计工具通过，但正式物理参数仍未冻结。下一步与用户确认 nominal 总质量和
  固定摩擦，再建立独立正式资产；`0.1 kg / 1.0 friction` 不进入正式训练。

#### 2026-08-18：正式宽桌物理参数冻结

- 用户确认正式 nominal 总质量 `20 kg`、static/dynamic friction `0.5 / 0.5`、restitution
  `0.0`；COM 和惯量按与宽桌五盒几何一致的均匀密度模型计算；
- 新建独立正式资产 `objects_widetable_plan5_training.urdf`，不修改 `0.1 kg` preflight 资产；
- 正式 Plan 5 baseline 使用固定 startup material 配置，首轮不进行质量、惯量、COM、材料或
  其他 physics domain randomization；
- 训练入口在创建环境后从 PhysX 读回质量、COM、惯量和全部 collision-shape material；任一项
  不匹配即拒绝训练，并把完整读回值写入 `run_config.json`；
- 真实 CUDA 一步检查通过：质量 `20.0 kg`，COM `[0, 0.015111745335, 0] m`，惯量对角
  `[0.6277497411, 4.3604149818, 3.9504892826] kg·m²`，五个 collision shape 均为
  `[0.5, 0.5, 0.0]`；
- 正式训练入口最小合法门禁通过：`4 envs × 1 step × 1 critic-only iteration`，无 reset、
  actor drift 严格为零，checkpoint/status/run config 均成功写出，且 `run_config.json` 包含四个
  环境的完整 PhysX 物理读回值；
- gate：正式物理配置通过。下一步先完成回归测试与提交，然后在启动任何正式训练前向用户
  报告并确认完整网络、PPO、reward、termination、物理、随机化及评测设置。

#### 2026-08-19：正式 20 kg Frozen A1 基准

- 协议：冻结 `model_07999_actor158.pt`，deterministic actor mean，seeds `721/722/723` 各
  3 次，每次从 frame 0 连续运行至首次 termination 或完整 309 帧；
- 结果：`0/9` 完成；平均执行 `27.78/309` 帧，范围 `22–41`；平均沿轨迹进度 `0.0403 m`，
  平均方向余弦 `0.9504`，平均 planar RMSE `0.0347 m`，最终 yaw 误差均值 `1.873°`；
- 失败分层：`9/9` 为 robot tracking failure，`0/9` 为 object-position，`0/9` 为
  object-orientation；多数案例是 0 号机器人 root/reference 高度误差超过 `0.5 m`；
- 解释：Frozen A1 在正式负载下能产生正确初始推动方向，reference 和桌面偏航没有首先失效，
  但无法维持机器人稳定；该结果是后续 full-actor 学习的正式对照，而不是增加硬约束的依据。

#### 2026-08-19：正式 20 kg critic-only 50-iteration warm-up

- 配置：seed `721`、`32 envs × 24 steps`、actor LR `1e-5` 但关闭 actor update、fresh
  centralized critic LR `1e-3`，共 50 iterations；
- 产物：`logs/Plan5Push/a1_formal20kg_critic50_seed721_env32/model_00050.pt`；checkpoint
  SHA256 `08ff9b3404b2bdf387f85f85fe1537757a7301f6d8c9961fefe415ffaab9302a`；
- 不变量：Actor state 和 Actor normalizer 相对 frozen source 逐张量完全一致，50 轮 actor
  grad、KL 和 deterministic policy drift 均严格为零；
- 数值：全部有限；value loss 前/后 25 轮均值 `0.12405→0.11909`，下降约 `4.0%`，线性趋势
  为负；reward 前/后 10 轮均值 `-0.05103→-0.05014`，符合 Actor 未更新的预期；
- 物理：`run_config.json` 记录 32 个环境质量均为 `20.0 kg`，每个环境五个 collision shape
  均为 `[0.5, 0.5, 0.0]`；
- gate：通过。当时下一步是从该 checkpoint 恢复进行 50-iteration full-actor 短程检查，
  并做三 seed × 三次 object-centric 评测；“短程不通过即阻止 500 iterations”的规则后来
  已被足量训练原则取代。

#### 2026-08-19：正式 20 kg full-actor 50-iteration gate

- 配置：从正式 critic-only `model_00050.pt` 恢复，seed `721`、`32 envs × 24 steps`，
  actor LR `1e-5`、centralized critic LR `1e-3`；联合更新 50 iterations 至 iteration 100；
  reward、termination、reference、20 kg 桌子物理和首轮无 physics randomization 契约均未修改；
- 产物：
  `logs/Plan5Push/a1_formal20kg_fullactor50_from_critic50_seed721_env32/model_00100.pt`；
  checkpoint SHA256
  `1021e007631203307a53a1ff5fc3f4d971852fb9bda39753c6bf728f6e1f73f9`；
- 数值审计：全部有限；actor/critic LR 全程分别保持 `1e-5/1e-3`；actor 参数相对
  iteration 50 的 delta L2 为 `0.48341`，最大绝对变化 `0.00495`；KL mean `0.01803`、
  max `0.02269`；value loss 前/后 10 轮均值 `0.11061→0.09098`；训练 reward
  `-0.05147→-0.04557`，但每轮 reset 前/后 10 轮均值 `32.4→34.6`，没有显示稳定性改善；
- 正式行为协议：deterministic actor mean，seeds `721/722/723` 各 3 次，每次从 frame 0
  运行至首次 termination 或完整 309 帧；与 Frozen A1 使用完全相同的正式物理和评测协议；

| checkpoint | 完成 | 平均帧数（范围） | 平均沿轨迹进度 | 平均方向余弦 | planar RMSE | yaw 绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen A1 | 0/9 | 27.78（22–41） | **0.0403 m** | 0.9504 | 0.0347 m | 1.873° |
| Full actor iteration 100 | 0/9 | 27.89（23–35） | 0.0267 m | **0.9561** | 0.0242 m | **0.793°** |

- 失败分层：iteration 100 的 `9/9` 均由 robot tracking termination 触发，`0/9` 为
  object-position，`0/9` 为 object-orientation；多数是 0 号机器人 root/reference 高度误差
  超过 `0.5 m`，与 Frozen A1 的主要失败模式相同；
- 解释：较低的 planar RMSE 和 yaw error 伴随更小的桌子位移，不能独立证明行为改善；平均
  存活帧数几乎不变，而推进量下降约 `33.8%`。训练 reward 和 value loss 的改善没有转化为
  完整物理 rollout 的稳定性或任务完成率；
- 当时 gate：**未通过**，曾暂停扩训到 `500 iterations`，但不因失败而偏离 Plan 5 主线；
  该短程暂停规则后来已被足量训练原则取代。当时下一步只做最小诊断：定位 0 号机器人高度
  失稳首先发生在哪个 root/body/link、对应 motion
  phase、接触状态及左右差异；先判断是 reference/初始化的非对称问题，还是当前 reward 下的
  可学习稳定性问题。任何 reward、termination、reference 或布局修改必须在证据形成后另行确认。

#### 2026-08-19：A1 软先验与物理安全 termination 对齐

- 用户确认：手掌精确接触不是当前阶段目标；腿、髋或其他身体部位参与施力可以接受，不增加
  手部专用 reward，也不惩罚 incidental contact；A1 是动作先验和软引导，而非必须逐帧复刻的
  硬约束；核心仍是机器人保持可用状态并让桌子跟随共享 reference；
- termination 语义：原 `0.5 m` reference-body 高度误差和 `0.25 m` 手腕/脚踝 Z 误差继续
  完整记录，但不再触发 reset；保留宽松的 reference orientation 安全门槛 `0.8`、桌子位置
  `0.25 m` 和姿态 `0.8 rad`；新增 tracked `torso_link` 绝对最低高度 `0.40 m` 作为物理
  低高度安全门槛；
- `0.40 m` 依据：A1 全轨迹 pelvis 为 `0.668–0.768 m`，tracked torso 为
  `0.705–0.822 m`；门槛允许相对最低 A1 torso 约 `0.30 m` 的下蹲偏离，但不把 torso
  落到桌面附近或以下的持续塌陷直接当作有效策略；
- 测试：termination、paired command、recording、reward 和 Plan 5 manager 定向测试
  `21 passed`；HoloSoma 核心套件 `143 passed`；仓库级自动发现的可选 inference/retargeting
  测试因环境未安装 `sshkeyboard`、`netifaces` 和 `mujoco` 在收集期停止，与本次变更无关；
- CUDA 预检：Frozen A1 在 frame 26、iteration 100 在 frame 19 首次触发新低高度门槛；
  两者均为 0 号机器人 torso 约 `0.37/0.36 m`，旧 reference-height、手腕/脚踝、机器人姿态、
  object-position 和 object-orientation 条件均未触发；证明新实现已把失败原因从“偏离 A1”
  分离为“绝对低高度”；
- gate：实现与语义检查通过，但旧 iteration 100 不因 termination 改写而自动晋升。下一步从
  actor 未变的正式 critic-only iteration 50 checkpoint 重新训练一个短程 full-actor 候选；
  训练前再次确认完整设置，完成后仍按三 seed × 三次、309 帧协议评估，不直接扩到 500。

#### 2026-08-19：软 termination 下的 full-actor 复测

- 配置：从正式 critic-only `model_00050.pt` 恢复，seed `721`、`32 envs × 24 steps`，
  actor LR `1e-5`、centralized critic LR `1e-3`，联合更新 50 iterations 至 iteration 100；
  除上一节确认的 termination 语义外，网络、reward、reference、20 kg 桌子和随机化均不变；
- 产物：
  `logs/Plan5Push/a1_formal20kg_softtermination_fullactor50_from_critic50_seed721_env32/model_00100.pt`；
  checkpoint SHA256
  `7cf1a08a660fd3a32c4b46d51fb2a4c7b0004455bc0623b8dfb9b64b03f2c577`；
- 数值审计：全部有限；actor/critic LR 全程保持 `1e-5/1e-3`；actor delta L2
  `0.46754`、最大绝对变化 `0.00479`；KL 前/后 10 轮均值 `0.01895→0.01824`；
  reward `-0.07356→-0.06853`，value loss `0.29644→0.27200`；reset
  `22.3→22.7`，未形成稳定性改善趋势；运行时 32 张桌子均为 `20 kg`、五个碰撞形状均为
  `[0.5,0.5,0.0]`；
- 同口径评测：新 termination 下，Frozen A1 与候选均使用 deterministic actor mean，
  seeds `721/722/723` 各 3 次、最多 309 帧；

| checkpoint | 完成 | 平均帧数（范围） | 平均沿轨迹进度 | 平均方向余弦 | planar RMSE | yaw 绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen A1 | 0/9 | **30.67（21–50）** | 0.0424 m | 0.8887 | **0.0381 m** | **1.923°** |
| Soft-termination iteration 100 | 0/9 | 24.89（21–32） | **0.0644 m** | **0.9533** | 0.0528 m | 3.325° |

- 失败分层：两组均为 `9/9` 的 0 号机器人 `torso_link < 0.40 m`；没有 object-position、
  object-orientation 或机器人姿态首先失败。候选推动更激进、方向更一致，但更早进入低高度，
  且桌面位置与偏航误差变大；
- reward 审计：六个机器人 motion term、action-rate、joint-limit 和 undesired-contact 当前都先对
  两台机器人取平均，再形成一份 team reward；object term 也是共享 reward。这种 mean pooling
  允许“一台机器人退化、另一台维持表现”被平均值部分掩盖，是 0 号机器人持续被牺牲的一个
  可检验假设，不应直接当作已证实根因；
- 当时结论：该 50-iteration 候选没有显示短程行为改善。此结果仅保留为诊断记录，不再用作
  否决 Plan 5 或阻止足量训练的 gate；下节给出取代该早期判定的正式执行规则。

#### 2026-08-19：学习实验判定方式修正与 1,000-iteration 观测窗口

- 方法论修正：此前把 50-iteration 行为结果用作方法晋升 gate 过于严格。短程运行只能验证
  数据、资产、维度、checkpoint、梯度、数值和日志是否有效，不能判断 learning-based 方法
  是否成立；Plan 5 尚未经过足量训练，因此不得写成方法失败；
- 撤销未经充分验证的硬条件：`torso_link < 0.40 m`、reference-body 高度误差和手腕/脚踝
  Z 误差全部只作诊断，不参与 reset，也不新增 torso reward；保留原有宽松 orientation、
  object-position、object-orientation 和 timeout termination；
- 冻结范围：共享 actor、centralized critic、A1 reference、现有 reward 及跨 agent mean、
  20 kg 桌子、`0.5/0.5` 摩擦、reset、observation 和首轮无 physics randomization 均不改；
- 训练计划：从 actor 未更新的正式 critic-only `model_00050.pt` 开始，训练 1,000 个
  full-actor iterations；`32 envs × 24 steps`、seed `721`、actor LR `1e-5`、critic LR
  `1e-3`；每 50 iterations 保存 checkpoint；
- 观测指标：reward、reset/episode、KL、value loss、policy drift、各 raw reward term；训练完成后
  再对代表性 checkpoint 做固定 frame-0 的 object-centric 回放，形成学习曲线；torso 高度、
  A1 相似度和接触方式是诊断指标，不是提前停止条件；
- 唯一提前停止理由：NaN/Inf、错误资产或物理常量、维度/数据错接、checkpoint/日志损坏等
  实现有效性问题。正常的低 reward、低完成率或非预期动作不构成中止理由；
- 本 1,000-iteration 窗口结束后才评估趋势；若仍在改善，可继续扩大训练预算；若足量数据
  显示无学习趋势，再讨论 reward、网络或 reference，不在训练途中做临时设计改动。

#### 2026-08-19：1,000-iteration 观测窗口结果

- 完整性：训练从 iteration 50 连续运行到 1050，共 1,000 行指标，status `passed: true`；
  训练用时约 `1880.9 s`（31.3 分钟）；每 50 iterations 的 checkpoint 均完整保存；
- 最终产物：
  `logs/Plan5Push/a1_formal20kg_heightdiag_fullactor1000_from_critic50_seed721_env32/model_01050.pt`；
  SHA256 `5085511de1d82b720ab6365112066d6ff5d6386f679a6df6a0e7740222441489`；
- 数值与物理：全程有限；actor LR 约 `1e-5`、critic LR 固定 `1e-3`；运行时桌子质量
  `20 kg`、五个 collision shape 材料均为 `[0.5,0.5,0.0]`；
- 每 100 updates 的训练趋势：reward mean 从首段 `-0.0791` 持续提高到末段 `+0.00791`；
  value loss 从 `0.4308` 降到 `0.02286`；KL 从 `0.01896` 缓慢降到 `0.01640`；
  policy drift mean 从 `0.01600` 降到 `0.01446`；
- raw reward 末段相对首段：global ref position `+0.1091`、relative body position
  `+0.1448`、relative body orientation `+0.1043`、global linear velocity `+0.0695`、
  object position `+0.0237`、object orientation `+0.0344`；action-rate raw penalty
  `65.74→27.52`，undesired contact `0.406→0.246`；
- 需要结合回放解释的训练指标：reset count 从首段 `21.2` 增至末段 `34.5`。训练使用随机
  motion phase，频繁 reset 可能让 batch 含有更多高相似初始状态，因此 reward 上升不能单独
  证明完整轨迹改善；这也是固定 frame-0 评测不可省略的原因；
- 固定评测协议：当前相同 termination（所有高度项仅诊断），deterministic actor mean，
  seeds `721/722/723` 各 3 次、最多 309 帧；500 updates 对应 `model_00550.pt`，
  1,000 updates 对应 `model_01050.pt`；

| checkpoint | 完成 | 平均帧数（范围） | 平均沿轨迹进度 | 平均方向余弦 | planar RMSE | yaw 绝对误差 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen A1 | 0/9 | **72.78（40–92）** | **0.0604 m** | 0.9269 | 0.0747 m | 11.905° |
| 500 actor updates | 0/9 | 45.89（30–95） | 0.0594 m | 0.7997 | 0.0719 m | **7.418°** |
| 1,000 actor updates | 0/9 | 58.78（29–92） | 0.0575 m | **0.9616** | **0.0696 m** | 9.235° |

- 终止分层：Frozen 为 6 robot-orientation / 3 object-position；500 updates 为
  7 robot-orientation / 2 object-position；1,000 updates 为 6 robot-orientation /
  3 object-position；三者都没有完成整条 reference；高度诊断可超阈值但不触发 reset；
- 解释：500 updates 相对 Frozen 明显退化，1,000 updates 又恢复平均帧数，并在方向一致性和
  planar RMSE 上超过 Frozen，说明策略仍在学习且尚未收敛。推进量没有增加，完整率仍为零，
  因此也不能把训练 reward 上升写成任务已经学会；
- 当前结论：本结果支持继续给当前冻结设计更多训练预算，不支持因最初 50/500 updates 的表现
  临时修改 reward、网络或 reference。下一次训练预算和 checkpoint 评测间隔需与用户确认；
  方法判断应依据更长学习曲线，而不是重新引入短程淘汰 gate。

#### 2026-08-22：Pull paired reference、正式物理资产与 CUDA 训练前验证

- 来源：冻结 Pull WBT `model_07999.pt` 的成功物理 rollout attempt 8。双机器人 reference
  以 table-local-Z 为横向轴、table-local-XY 为镜像面；共享桌轨迹是原实测相对轨迹与其
  local-Z 镜像的对称平均，机器人再分别向外偏移 `0.4390236 m` 对齐 z-wide 桌腿；
- reference：`316` 帧、`50 Hz`，一条共享 object trajectory 和一个共享 phase；完整 runtime
  文件 SHA256 为 `571f23055f06648fb30b7abe5d90746a80c8555461ef91b6a23bccd9ad8404d4`；
- 人工验收：用户于 2026-08-22 在 ViSER 中批准该双机器人 Pull 布局与动作语义；两台
  rubber-hand G1 位于同一拉动侧、沿 table-local-Z 对称并对应两侧桌腿区域，未引入
  hemisphere/sphere hand。该批准只覆盖几何与运动学 reference，不是动力学可行性或训练
  成功证明；ViSER 文件 SHA256 为
  `adf41371577bf32c606f3ab6caa07b2ac1826bf8e6dce8f524467aa6674812d6`；
- actor 兼容：Pull `154→158` checkpoint SHA256 为
  `f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364`；原 154 维
  deterministic 输出最大绝对误差 `0.0`，新增 teammate 四列首层权重严格为零；
- 物理：Pull z-wide URDF SHA256 为
  `c260652d23e31dc0978a167137c45727cddb156eebedab77c9a40aa30079a8c2`。一步真实 CUDA
  smoke 通过，实际加载两台 rubber-hand G1、单张 `20 kg` 桌；PhysX 读回 COM/惯量与
  资产一致，五个碰撞形状材料均为 `[0.5, 0.5, 0.0]`；
- frozen 连续回放：使用未经 Pull MARL 适应的 `158-D` actor，从 reference frame 0 开始，
  在总长 316 帧中的 frame 156 触发 `joint_bad_tracking` 联合 reset（recorder 保存
  frame index `0..155` 共 156 条）。记录文件
  `logs/Plan5Pull/frozen_smoke_20260822/pull_frozen_316_object_centric.npz`，SHA256
  `e27710ad400ea35f7cf47918fe1f0a2b22b96c032e0638a39e8d0a0d4cb14b6a`；
- object 结果：smoke 内置 object-reference metric 报告 actual net displacement
  `0.001603 m`、actual path `0.001913 m`、along progress `-0.001417 m`、终止时 final
  position error `0.25177 m`，而完整 reference 平面净位移为 `0.584278 m`。recorder 的
  world-XY 首末样本直接相减约为 `0.00404 m`，两者测量时序/口径不同；正式比较固定使用
  命名的 object-reference metric，不把未限定的首末差混作同一指标；
- 结论：loader、维度、reference、rubber-hand、z-wide 资产和真实物理环境已接通；frozen
  单机先验不能直接完成新的双机器人 Pull 动力学任务是预期训练前结果，说明需要独立 Pull
  PPO 适应，不构成对 paired reference、Plan 5 或 Pull 路线的否决；
- 后续决定：用户确认改为本地单张 RTX 5070 顺序训练，每项 `2,048` environments；先 Pull、
  后 Push，均从各自 WBT `158-D` actor 建立全新 MARL critic/optimizer。每项先做
  `50 critic-only` bootstrap，再做 `8,000 full-actor`，每 `1,000` iterations 保存；Push
  不加载旧 `model_13050.pt`。禁止 mixed-action 数据、跨动作 checkpoint 或 task-oriented
  单智能体目标点奖励进入当前主线；
- 训练前里程碑提交为 `3cedc74e`。独立容量目录
  `logs/Plan5Pull/pull_capacity_full1_seed721_env2048` 完成一次 `2,048 env × 24 steps` 的
  完整 actor+critic PPO 更新，最终 `status.passed=true`，iteration 用时 `5.655 s`；运行时
  读回为 `20 kg`、COM `[0, 0.0151117453, 0] m`、Pull 惯量对角
  `[3.9504893, 4.3604150, 0.6277497] kg·m²`、五个碰撞形状材料均
  `[0.5, 0.5, 0.0]`，网络维度为 actor `158→512→256→128→29`、critic
  `527→512→256→128→1`；
- 容量测试显式设置 actor 初始 LR `1e-5`，adaptive-KL schedule 在第 1 iteration 后将其
  调整为 `3.375e-5`。因此 `1e-5` 不能描述为固定 LR。用户确认正式训练沿用该既有 Push
  adaptive schedule，并要求每 `1,000` iterations 保存；容量 checkpoint 只用于验证显存
  峰值，不参与任一正式阶段的 resume；
- 正式 critic-only 输出目录为 `logs/Plan5Pull/pull_20kg_critic50_seed721_env2048`，50/50
  metrics 完整且 `status.passed=true`。最终 `model_00050.pt` SHA256 为
  `f211fd0b6603c5f28e52d62443fd03ccec138765dbc190e6b02e01187e2b8ab7`；相对
  `model_00000.pt`，actor 与 actor normalizer 最大绝对差均为 `0.0`，critic 最大参数变化
  为 `0.43130`。正式 full-actor 必须从该 checkpoint resume，不得从容量测试 resume。

#### 2026-08-22：Pull 2,048-env 完整训练与首轮物理验收

- 正式目录：`logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048`；训练从
  iteration 50 连续运行至 8050，共 8,000 行有限 metrics，`status.passed=true`，耗时约
  `10.33 h`。每千轮保存一次，`model_01050.pt` 至 `model_08050.pt` 共 8 个 checkpoint；
- 最终 `model_08050.pt` SHA256 为
  `727630e9cec654d88bfb454d5eaabd70d7db629478598a2039140653a57cbf43`。最后 100 轮相较
  最初 100 轮，平均 reward 从 `0.07109` 升至 `0.15086`，tracking failure 从
  `419.28/iteration` 降至 `0.56/iteration`，value loss 从 `0.10996` 降至 `0.00423`；
- seed 721、frame-0、actor mean-action、单环境连续 PhysX 回放中，`model_08050.pt` 完成
  316/316 帧，无 reset 或 tracking failure。桌子实际/参考净位移为
  `0.58082/0.58428 m`，沿轨进度比 `99.40%`，最终平面误差 `0.00799 m`，最终 yaw
  误差 `0.262°`，平面 RMSE `0.01467 m`。录制文件
  `logs/Plan5Pull/eval_full8000_seed721/model_08050_object_centric.npz` SHA256 为
  `b6c6c056a609849b3c19cb80516bd687a7d5ddacd5f1d5ebe5d06ed82e6d3e0f`；
- `model_07050.pt` 在相同协议下也完成 316/316 帧，但沿轨进度比 `89.88%`、最终平面
  误差 `0.06377 m`、最终 yaw 误差 `4.05°`，因此数值主候选为 `08050`。保留 `07050`
  仅用于比较动作自然度、抖动与角色分工；
- 本结果只覆盖一个 seed 的确定性固定初始化回放，不能解释为统计稳健性证明；用户随后已在
  ViSER 中确认 Pull 动作质量良好。Pull 固定多 seed 统计仍待完成，但它不再阻塞独立 Push
  的干净训练。

#### 2026-08-27：Push live-state 完整训练、重复评测与暂停决定

- 实时状态修正：双机器人执行时，每台 actor 在每个控制步读取对应实体机器人的实时关节位置
  和速度；这些量属于本机 proprioception，不是 centralized critic 的特权信息。修正后用户确认
  `model_08050.pt` 的手臂高频振荡明显消失；网络仍为 shared actor
  `158→512→256→128→29`、team critic `527→512→256→128→1`，没有把桌子状态加入 actor；
- 干净训练链：critic-only 目录
  `logs/Plan5Push/a1_livepd_20kg_critic50_seed721_env2048/`，完整 8,000 轮目录
  `logs/Plan5Push/a1_livepd_20kg_full8000_from_critic50_seed721_env2048/`，续训目录
  `logs/Plan5Push/a1_livepd_20kg_continue7000_from_08050_seed721_env2048/`。续训从 iteration
  `8050` 连续到 `15050`，共 7,000 条有限 metrics，`status.passed=true`；环境数 `2,048`、
  每环境每轮 `24` steps、seed `721`、20 kg 宽桌、材料 `[0.5, 0.5, 0.0]`、原 11 项
  WBT/object reward、adaptive actor LR 和 critic LR `1e-3` 均保持不变；
- 保留的当前关键 checkpoint SHA256：`08050` 为
  `58265a7d200695793b85b0821aea927fa90129f916eb30b438b1f621354e8266`；`13050` 为
  `0ba8ae21239c7b6c0198f07086abbb0cfc3875dc908626df9e9fc0891692aede`；`14050` 为
  `8fb2e29cd2af7b9cd05655b13ddfdd0b9d00de9d93028139890f11e8d99f3333`；`15050` 为
  `b6be5f9b385122e8398ecc0494bc8fad9fdd13e72cd8766130f501f84f4b75e9`；
- 固定评测协议：`13050/14050/15050` 分别使用 seeds `721/722/723`，每个 seed 独立启动
  3 次，共 27 次；均从 frame 0、deterministic actor mean、相同 reference、初态和 PhysX
  常量开始，最多 309 帧。这里的 seed 不随机化 reference、初态、摩擦或质量，因此该实验
  衡量的是相同固定场景下跨 Isaac/PhysX 启动的重复稳定性，不是环境泛化；

| checkpoint | 完整成功 | 平均帧数 | 中位帧数 | 平均沿轨进度 | 首个终止原因 |
|---|---:|---:|---:|---:|---|
| `13050` | **1/9** | **182.00** | **172** | **76.31%** | 8 次 object-position，1 次完成 |
| `14050` | 0/9 | 83.78 | 90 | 2.86% | 3 次 robot、6 次 object-position |
| `15050` | 0/9 | 87.78 | 91 | 2.94% | 3 次 robot、6 次 object-position |

- 解释：PPO 优化的是 2,048 个随机 phase 环境中的期望回报，不保证固定 frame-0 物理回放随
  iteration 单调改善。`14050/15050` 的训练平均 reward 与 object reward 没有显示实现错误，
  但策略在接触闭环中的修正更激进；代表性 seed 721 回放中，0 号机器人相邻 action 的平均
  变化由 `13050` 的 `1.031` 增至 `14050/15050` 的 `3.363/3.592`。接触、平衡和共享桌子的
  耦合会放大小的策略参数变化，所以后两个 checkpoint 比 `13050` 差并不矛盾，也不能仅凭
  iteration 编号选择最终模型；
- 当前判定：`13050` 是保留的当前最佳 Push 候选，但 `1/9` 不足以宣称稳健。用户明确判断
  **该方向仍需继续训练，但不是现在**。本轮不启动新训练；未来恢复前只需重新确认 resume
  checkpoint、训练预算和相同评测间隔，不因当前回归偏离 Plan 5 主线；
- 存储清理：按
  `logs/Plan5Push/CLEANUP_20260827_MANIFEST.tsv` 永久删除 203 个已批准文件，共
  `740,530,526` bytes（`706.225 MiB`），包括旧平滑目录
  `a1_8050_jointacc_smooth2000_seed721_env2048/model_08050.pt`。manifest SHA256 为
  `5701468c10019182bd0813a44cc65d5e582e10c45e828d7ead1628e0cd5c61c5`；删除后清单内
  路径剩余 `0` 项，`logs/Plan5Push` 约为 `169 MiB`。当前恢复链、上述后期 checkpoint、
  27 份重复评测、run config/metrics/status 账本和最小历史对照链均已保留。

#### 2026-08-28：Demo 3 对角桌腿竞争式拉拽——独立环境与训练接口就绪

- 范围：Demo 3 是 Push/Pull 两个协作 Demo 之后的独立竞争实验；两台实体 rubber-hand G1
  位于方桌对角桌腿处，各自尝试把桌子拉向自己。它不读取、覆盖或续训 `Plan5Push`、
  `Plan5Pull` 的环境配置、critic、optimizer、checkpoint 或日志；所有运行产物限定在
  `logs/Demo3Tug/`；
- reference：以原始单机器人 Pull 为动作先验，只移除桌子的平面 XY/heading 运动，并把第二台
  机器人绕世界 Z 轴旋转 180°。保留机器人 Z、roll/pitch、29 关节轨迹，以及桌子的 Z/tilt；
  runtime reference 为 `317` 帧、`50 Hz`、每台 `29` joints/`51` bodies，SHA256
  `5baedb3f109402c521590701263facfa6b42649f4f18bdcefb5933d9b4a7caf5`。文件中的桌子通道只用于
  reset 与 schema，不是训练跟踪目标；command 到末帧后 clamp，绝不循环回写机器人或桌子；
- 资产：沿用原方桌尺寸，质量固定 `20 kg`，COM `[0,-0.0124134734,0] m`，惯量对角
  `[0.8219720,1.2061834,0.8219720] kg·m²`；静/动摩擦 `0.5/0.5`，restitution `0`；URDF
  SHA256 `386da8a4201a9365f7d4eef9e6ae2fb326c2777fac4998b72c4af6de693f9d1b`；
- Actor：共享参数，输入为原 Pull `154` + 对手相对位置/速度 `4` + 实际桌子相对 XY、平面
  线速度、相对 yaw sin/cos `6`，合计 `164`；不含 yaw rate。原 Pull `158-D` actor 的新增
  六列首层权重为零，转换前后 deterministic 输出误差 `0.0`；转换 checkpoint SHA256
  `048f952cad01d5fda42851502347ee626751dccab923905af303eb44807ef01b`；
- MAPPO：同一个 Actor 与同一个 Critic 分别作用于 A/B。Critic 输入保持 `527` 维，但为每个
  agent 构造 `[ego, opponent, ego Pull reference, actual table, phase]` 的 ego-first 全局状态；
  reward/value/return/advantage 均为 `[T,E,2,1]`，共享物理 done/timeout 扩展到两条 GAE；
  不再复制一份 team advantage 给双方；
- reward/termination：每台机器人分别获得 Pull 动作、稳定和平滑先验；只有实际桌子沿两条相反
  拉拽轴的速度项符号相反。不限制必须用手，不惩罚 incidental contact，不跟踪演示桌子位姿，
  也不因桌子偏离 reference 终止。episode 只在 317 帧 horizon 或机器人明确摔倒时结束；
- 正式训练契约：冻结 Pull07999 的 164-D lossless 扩展作为 Actor/normalizer warm-start；Critic
  和 optimizer 新建；两名竞争者始终调用同一个同步更新的共享 Actor。正式 baseline 为
  `50 critic-only + 8000 full-actor`、`4096 env × 24 steps`、seed `721`、每 `1000` 个
  full-actor iterations 保存；对应 `model_00050.pt`、`model_01050.pt` … `model_08050.pt`。
  `signed_table_progress_velocity` 权重正式冻结为 `10.0`；首版保持 fixed frame-0、固定质量/材料、
  无对手初态随机化。完整非 LR PPO 更新契约写入 run config 和 checkpoint；resume 只允许显式
  覆盖 Actor/Critic learning rate。4096 是本轮用户确认的正式规模，但必须先在目标 RTX 4090
  单独做容量测试；若 OOM，不自动降档，先回报并讨论；
- 评估协议：部署时只调用共享 Actor，不调用 Critic。episode 末桌子沿 Agent A 初始 Pull 轴净位移
  `>+0.05 m` 判 A 胜、`<-0.05 m` 判 B 胜，其余为平局；该阈值只用于 evaluation，不属于 reward
  或 termination。每回合显式走相同 reset/settling 路径并从同一 reference phase 开始；reset 后
  的微小 PhysX 数值差以物理量级 sanity check 和实际误差记录处理，不要求 bitwise 相同。horizon
  episode 的 Actor 步数必须精确，真实摔倒允许提前终止并单独计数。评估在 auto-reset 前捕获真实
  terminal state，输出 `evaluation.json` 与去重的五通道 ViSER NPZ；
- 验证：当前最终 CPU 回归分为共享 MAPPO/Demo 3/4 的 `90 passed`，以及环境、Plan 5、
  rubber-hand 与跨 Demo 隔离边界的 `104 passed`。真实 Isaac/PhysX 环境 smoke 使用 `1 env × 8 steps`，
  确认 actor groups `[1,2,154]/[1,2,4]/[1,2,6]`、critic `[1,2,527]`、reward `[1,2]`、
  value `[1,2,1]`、物理 `200 Hz`、控制 `50 Hz` 及上述资产常量。真实 CUDA PPO smoke 使用
  `1 env × 4 steps × 1 epoch × 1 minibatch`，Actor/Critic 都发生参数更新、Actor normalizer
  冻结、v2 checkpoint 往返恢复通过；正式 train 入口另以隔离的 1-iteration `/tmp` run 验证完整
  run ledger/checkpoint/退出流程，并用其 checkpoint 跑完 316-step actor-only 评估，终态捕获与
  ViSER NPZ 输出闭环通过。两回合复核进一步确认相同 start phase、horizon 精确步数和合法提前摔倒
  能同时记录；确定性 Actor 不被误写成 GPU 接触动力学逐 bit 确定。Resume smoke 确认原始
  `run_config.json` SHA 保持不变，恢复段使用独立且不可覆盖的 run-config/metrics/status，且拒绝
  倒退覆盖更新模型；完整账本目录允许跨机器迁移，但会显式记录原路径和 relocation lineage；
- 当前状态：正式管线已就绪，**尚未开始正式 50+8000 训练**。临时 1-iteration 结果只证明程序
  闭环，不作为竞争效果或 checkpoint 质量证据。

#### 2026-08-28：Demo 4 协作旋转长桌——静态桌子与团队 yaw 任务管线就绪

- 任务定义：两台实体 rubber-hand G1 位于长方桌斜对角，各自跟踪独立的 Pull 动作先验，依靠
  真实接触共同把桌子沿世界 Z 轴正向旋转 `+90°`。训练不指定逐帧桌子轨迹，不要求桌子在机器人
  动作开始前旋转，也不限定必须由手完成接触；桌子只在 reset 时采用静态初始位姿，随后完全由
  PhysX 动力学决定；
- 隔离边界：Demo 4 使用独立的 command、observation、reward、termination、environment、MAPPO、
  train/evaluate/smoke 入口和 `logs/Demo4Rotate/` 输出目录；不覆盖、不续训 Push、Pull 或 Demo 3
  的环境、optimizer 和日志；
- reference：A 使用 Pull `model_08050.pt` 的物理回放，B 是绕世界 Z 轴严格旋转 `180°` 的副本；
  两者均保留完整机器人姿态、29 关节和速度信息。reference 为 `316` 帧、`50 Hz`；桌子每一帧
  的位置/姿态恒定、线速度/角速度严格为零，桌子通道仅负责 reset/schema。ViSER 文件 SHA256
  为 `706143a04c8f04c5165470b8d661ba5a9fb41b80ab7103913d4a5e968606369b`，runtime 文件
  SHA256 为 `3eb482e1bdb9072b50054c5501cda1d4646bef39ef754df55c03c281626199f3`；
- Actor/critic：两台机器人共享同一个 Actor，分别执行 `164→512→256→128→29`；每台输入为
  原 Pull ego/reference `154` 维、teammate 相对平面位置/速度 `4` 维、实际桌子相对 XY、平面
  线速度和相对 yaw 的 sin/cos `6` 维，明确不含桌子 yaw rate。Actor 从 Pull
  `model_08050.pt` 的 actor 与 normalizer 初始化，`158→164` 的新增列置零；critic 与两个
  optimizer 从新状态开始。共享 team critic 为 `527→512→256→128→1`，每个环境只有一套
  team reward/value/return/advantage；
- reward：六项机器人 Pull/WBT 动作先验权重为 `[0.5,0.5,1,1,1,1]`，action-rate 为 `-0.1`，
  joint-limit 为 `-10`；团队任务采用相邻物理步累计的无缠绕 yaw progress，整段 `+90°` 的归一化
  progress 总贡献为 `10`，首次安全达到目标再奖励 `5`。错误方向跨越 `±π` 不会被误判为成功；
  机器人摔倒或桌子安全终止的同一步不发放 yaw task reward；
- episode/安全：首次达到 `+90°` 即成功，不额外要求保持；reference horizon 为 `316` 帧
  （`6.32 s`）。机器人明确摔倒、桌子倾斜超过 `60°`、平面漂移超过 `3 m`，或桌高离开
  `[0.05,1.5] m` 时终止；这些是物理失控边界，不是动作形态约束；
- 正式训练预设：单 GPU、`4,096` environments、每轮每环境 `24` control steps、seed `721`、
  `8,000` iterations、每 `1,000` iterations 保存一次；物理频率 `200 Hz`、控制频率 `50 Hz`；
  长方桌质量 `20 kg`，静/动摩擦 `0.5/0.5`，restitution `0`。PPO 沿用 WBT baseline：
  `5` epochs、`4` mini-batches、clip `0.2`、gamma `0.99`、GAE lambda `0.95`、entropy `0.005`，
  actor/critic 初始学习率均为 `1e-3`、adaptive KL `0.01`；
- 可观测性：训练日志直接从真实 rollout 记录四类终止数量、完成 episode、成功率、连续 yaw
  progress 与 terminal yaw 的 mean/min/max；不通过再次调用有状态 reward 函数来伪造统计。
  独立 deterministic actor-only 评估入口会保留 reset 前的 terminal state，并输出代表性五通道
  ViSER NPZ；checkpoint 同时锁定 reference SHA、网络维度、桌子常量、reward 和无 yaw-rate 契约；
- 验证：Demo 4 定向 CPU 测试 `58 passed`；包含 Demo 3、Plan 5 与 MAPPO 隔离回归
  `129 passed`。真实 Isaac/CUDA PPO smoke 以 `1 env × 4 steps` 完成 rollout、Actor/Critic 更新和
  checkpoint round-trip；Actor/Critic 最大参数变化均约 `0.001`，所有张量与指标有限，Actor
  normalizer 保持冻结。静态物理 smoke 另已确认 reset 后桌子不会自行旋转；
- 当前状态：静态布局、训练管线、恢复契约、评估入口和 smoke 均已完成，**尚未启动正式训练**。
  下一 gate 只包含用户对静态 ViSER 的最终视觉确认，以及正式训练参数的确认；不再人为设计桌子
  逐帧 reference。
