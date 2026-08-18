# 多智能体涌现主路线图

## 1. 文档地位与当前状态

- 状态：唯一有效执行指南
- 最近更新：2026-08-18
- 分支：`rubber_hand_marl_baseline`
- 基线提交：`8038c092`（`wbt-four-action-priors-v1`）
- 当前唯一方案：**Plan 5——按动作分网的 reference-guided 多智能体强化学习**
- 当前阶段：Plan 5 技术设计讨论；尚未开始双机器人训练环境实现
- 机器人：Unitree G1 29-DoF，固定 rubber hand
- 第一动作：Push A1

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

首个 baseline 只使用 Push A1。Pull 是下一项合作扩展；竞争性物体争抢在
合作环境和评测工具稳定后加入。Plan B 与 Kick 保留为后续动作实验，不是
首个 baseline 的依赖。

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

首个实现顺序：

```text
双机器人 Push A1 合作
    -> Pull 合作扩展
    -> 竞争性物体争抢
```

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

### 4.3 宽桌与站位

首个双机器人布局冻结为：

- 桌面横向宽度：`1.4 m`；
- 两台机器人中心间距：`0.8 m`；
- 相对中央 A1 reference 的横向偏移：`-0.4 m / +0.4 m`；
- 两台机器人位于同一推动侧；
- 一条共享桌子 reference 和一条共享 motion phase。

宽桌只沿与推动方向垂直的 local X 轴加宽，保持桌面高度、推动方向深度、
接触边和 object origin。详细几何契约保留在
`WIDETABLE_GEOMETRY_DESIGN_CN.md`。

几何预检不负责冻结最终质量、惯量和摩擦。正式物理参数必须在 Plan 5
capacity calibration 中单独确认。

### 4.4 158 维 actor 兼容接口

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

首个任务使用一份同步的双机器人 reference：

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

## 6. 已完成证据与关闭结论

### 6.1 已完成

- 四个独立动作 WBT 先验已冻结。
- A1 154→158 维 actor 转换已通过无损等价验证。
- 1.4 m 宽桌和 0.8 m 双机器人布局已通过 Viser 与 Isaac reset 几何检查。
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

状态：下一阶段，等待技术设计确认。

工作：

1. 在一个 Isaac Sim environment 中创建两台 physical rubber-hand G1 和一张共享桌子。
2. 两次调用同一个 Push A1 actor，并把两组 29 维 action 分发给各自机器人。
3. 使用一份 paired reference 和共享 phase。
4. 用真实相对位置/速度替换 ghost buffers。
5. 建立全局 critic observation。
6. 实现 shared reward、joint reset、collision filtering 和双机器人 recorder。
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

任何阶段未通过 gate 时先诊断，不自动增加训练量。

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

Push baseline 稳定后：

1. 为 Pull 独立建立相同的多智能体接口和训练组；
2. 保留 Pull 独立 actor/checkpoint，不与 Push 混合；
3. 建立 competitive object-grabbing 环境；
4. 比较 WBT 初始化与从零训练；
5. 量化角色分化、阻挡、争抢、让位和接触点切换等涌现行为。

创新模块只在 baseline 暴露明确限制后选择。

## 8. 已冻结的实施决策

2026-08-18 已确认以下八项：

- [x] 单智能体随机 teammate 输入只做短程接口与鲁棒性检查，不做长期适应训练。
- [x] 同一动作内使用 shared actor 和 decentralized execution；centralized critic 从零训练。
- [x] 首版使用左右平移的 paired A1 robot reference 和一条共享桌子 reference。
- [x] 使用简单 shared WBT reward，不加入显式分工、hand-only 或合作塑形奖励。
- [x] 任一机器人失效时 joint reset；允许并记录偶发非手部接触。
- [x] 冻结已有 actor normalizer，完整 actor 参与 MARL 更新。
- [x] 训练采用 `50 -> 500 -> 2,000 -> 8,000 iterations` gate。
- [x] `0.1 kg` 只用于环境 smoke；正式物理参数通过单/双机器人 capacity calibration 冻结。

## 9. 执行清单与结果记录

状态含义：`[ ]` 未开始，`[~]` 进行中，`[x]` 完成并通过 gate，`[!]` 完成但未通过。
每个完成项必须在本节记录日期、commit、测试或产物、定量结果和结论。

### 9.1 Stage 2——环境与数据流

- [x] 只读审计 simulator、environment、PPO、rollout storage 和 recorder 的现有结构。
- [x] 冻结精确的修改文件、模块边界和回退方案。
- [ ] 建立两台 physical rubber-hand G1 和一张共享宽桌的单环境实体结构。
- [x] 定义 shared actor batch 与两组 `29-D` action 的纯张量 shape/routing 契约。
- [x] 建立 per-agent actor 数据与 per-environment team 数据分离的 rollout storage。
- [ ] 把 shared actor batch 和 action routing 接入在线环境。
- [ ] 实现每台机器人 `158-D` actor observation 和真实 teammate 相对状态。
- [ ] 建立最小 centralized critic observation 和全新 critic。
- [ ] 建立 paired A1 reference、共享 phase 和共享 object reference。
- [ ] 实现 shared reward、joint reset、termination 和碰撞语义。
- [ ] 实现 actor checkpoint、冻结 normalizer、全新 critic/optimizer 的加载契约。
- [ ] 扩展 recorder，区分两台机器人、共享桌子、接触和终止原因。
- [ ] 完成确定性 reset、维度、坐标、action routing、reward sign 和短 rollout 测试。
- [ ] 完成单智能体随机 teammate observation 的短程鲁棒性检查。

### 9.2 Stage 3——Push A1 训练 gate

- [ ] `50 iterations`：环境与数值 smoke。
- [ ] `500 iterations`：学习方向与稳定性检查。
- [ ] `2,000 iterations`：初步合作与搭便车诊断。
- [ ] `8,000 iterations`：第一版完整 Push A1 baseline。

每一级未通过时先记录失败层级和证据，不自动进入下一级或增加训练量。

### 9.3 Stage 4——物理与合作真实性

- [ ] 完成单机器人/双机器人 capacity calibration。
- [ ] 冻结正式桌子质量、惯量、COM 和摩擦。
- [ ] 完成 frozen-copy、scratch、WBT-initialized 三组双机器人对照。
- [ ] 完成多 seed 正式评测。
- [ ] 完成单 agent removal、接触 impulse、桌子功率贡献和非手接触报告。
- [ ] 证明第二台机器人具有可量化的因果贡献。

### 9.4 Stage 5——动作与场景扩展

- [ ] 使用独立 Pull checkpoint 建立 Pull 合作 baseline。
- [ ] 建立 competitive object-grabbing 环境。
- [ ] 比较 WBT 初始化与从零训练。
- [ ] 量化争抢、阻挡、让位、接触点切换和控制权变化。

### 9.5 已完成记录

#### 2026-08-18：实施决策冻结

- commit：待与 Stage 2 只读审计结果一并提交；
- 结果：八项技术决策全部确认，Plan 5 从概念讨论进入执行；
- gate：通过；
- 下一项：只读代码审计。

#### 2026-08-18：Stage 2 只读代码审计

- commit：待本次清单提交；
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

- commit：待本次代码提交；
- 文件：`agents/mappo/batch_layout.py` 及其单元测试；
- 结果：固定 `2 agents x 158-D observation x 29-D action` 契约；
  `[env, agent, feature]` 与 shared-actor batch 可逆转换，agent 顺序不串线；
- 测试：`7 passed`，并通过 `py_compile` 与 `git diff --check`；
- gate：通过；
- 下一项：实现区分 per-agent actor 数据和 per-environment team 数据的 rollout storage。

#### 2026-08-18：Multi-agent rollout storage

- commit：待本次代码提交；
- 文件：`agents/mappo/storage.py` 及其单元测试；
- 结果：agent 字段使用 `[time, env, agent, ...]`，team/critic 字段使用
  `[time, env, ...]`；minibatch 先采样完整 environment transition，再展开 agent；
- 测试：MAPPO 基础层合计 `12 passed`，并通过 `py_compile` 与 `git diff --check`；
  当前环境未安装 `ruff`，未新增依赖；
- gate：通过；
- 下一项：实现 Plan 5 专用 Isaac Sim 双 articulation 状态与控制适配层。
