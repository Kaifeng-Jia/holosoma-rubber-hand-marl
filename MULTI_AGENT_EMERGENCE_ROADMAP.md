# 多智能体涌现主路线图

## 1. 文档地位与当前状态

- 状态：唯一有效执行指南
- 最近更新：2026-08-18
- 分支：`rubber_hand_marl_baseline`
- 基线提交：`8038c092`（`wbt-four-action-priors-v1`）
- 当前唯一方案：**Plan 5——按动作分网的 reference-guided 多智能体强化学习**
- 当前阶段：Stage 3 学习闭环已通过；下一项为 `50 iterations` 环境与数值 smoke
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
