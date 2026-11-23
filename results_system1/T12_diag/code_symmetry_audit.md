# T12-diag 第 4 步：代码里所有和机器人编号有关的地方

只读审查，由一个子代理完成。标"已复核"的条目由我本人另行核对过。

- **路径约定：** `B/` 指 `/home/kevin/holosoma-core4d-base/src/holosoma/holosoma/`。行号按旧工作树 HEAD `7df11c92`，下面说明为什么是这一版。
- **身份对应：** agent 0 = 仿真里的 `Robot`，agent 1 = `Robot_1`。

## 代码版本

- **训练清单：** D Table 种子 721 的训练目录（`logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920`）里，`cloud_provenance/package_manifest.json` 列出了云端训练用的 453 个源文件和各自的 sha256。
- **旧工作树：** 449 个和清单一致。另外 4 个有未提交改动，取 HEAD 里的版本后，sha256 和清单一致。
  - 这 4 个文件是 `scripts/train_core4d_smalltable.py`、`scripts/smoke_core4d_smalltable.py`、`agents/mappo/core4d_smalltable_evaluation.py`、`agents/mappo/initialization.py`。
  - 它们的未提交改动都是默认关闭的开关，这次训练没有用到。
  - 所以训练代码等于旧工作树 HEAD 的这一版，下面审查的就是它。
- **本分支：** 439 个一致，14 个不同。这 14 个全部是 System1 加的开关，默认关闭，没有按编号区分的改动。评测脚本 `scripts/evaluate_paper_supplement.py` 跑的是本分支的代码。

## 逐项结果

| 类别 | 位置 | 内容 | 判定 |
|---|---|---|---|
| actor 观测（154） | `B/managers/observation/terms/marl.py:24-77` | 依次为 actions、base_ang_vel、dof_pos、dof_vel、motion_command、motion_ref_ori_b，都由各自的状态和各自的参考算出；噪声 σ 两边相同；没有 agent ID 或角色输入 | 对称（各跟各的参考，属设计内） |
| teammate 观测（4） | `marl.py:195-210`；`B/config_values/wbt/g1/observation.py:150,156` | `teammate_ids = [1, 0]`，在自己的坐标系里看对方的平面相对位置和速度，clip ±1 | 对称 |
| 共享 actor | `B/agents/mappo/core4d_smalltable_runner.py:70-78` | 把 [E,2,158] 摊平成 [E·2,158]；共用一个观测归一化器和同一个动作 std | 对称 |
| critic 观测（527） | `B/config_values/marl/g1/observation.py:48-90` | agent_* 各项按 agent 0、agent 1 的固定位置排列 | 中心化 critic 的正常设计 |
| **critic 的参考指令** | `B/managers/observation/terms/marl.py:80-83`（**已复核**） | `centralized_shared_motion_command` 返回 `command[:, 0]`，即只有 agent 0 的 58 维参考关节位置和速度（critic 第 456–514 维），没有 agent 1 的 | **不对称，不属于设计内**（沿用 A1"两人参考相同"的设定） |
| 奖励 | `B/managers/reward/terms/marl.py:24-74, 129-138, 248-263`；`B/managers/reward/manager.py:171-174` | 各跟踪项和正则项先逐 agent 计算，再在 agent 维取平均（各 0.5）；奖励是整队一个值 [N] | 对称 |
| advantage | `B/agents/mappo/ppo.py:90-96, 195-205, 255-256` | 整队 GAE，用 `repeat_interleave` 复制给两个 agent，两人的 advantage 相同 | 对称 |
| 桌子交互奖励 | `B/managers/reward/terms/core4d_bucket.py:248-279`（**已复核**） | 桌子位置和朝向两人共享；接触门控只在变体 B 生效，本次是 A_no_rel_no_height，门控恒为 1；训练日志 12000 轮 `Bucket/gate_mean` 最小、最大都是 1.0（**已复核**） | 对称；接触对奖励没有任何影响 |
| 终止 | `B/managers/termination/terms/marl.py:49-109`；`.../core4d_smalltable.py:14-22`；`B/envs/marl/core4d_smalltable_manager.py:11-24` | 任一 agent 躯干倾斜、桌子位置 > 0.25 m 或朝向 > 0.8 rad，按整个环境终止 | 对称 |
| 训练重置 | `B/managers/command/terms/marl.py:85-99`；`.../core4d_smalltable.py:85-127` | 每个环境抽一帧，两人用同一帧；各写自己的参考状态；初始噪声为 0 | 对称 |
| 评测初始状态 | `table_states.json`（sha256 389e8da5…）；`scripts/paper_evaluation_utils.py:29-51` | 两人用同一规则独立抽样：xy ±2 cm、yaw ±2°；实测均值 agent 0 \|xy\| 1.02/1.03 cm、\|yaw\| 1.06°，agent 1 0.92/1.10 cm、1.04° | 对称 |
| 动作和 PD | `B/managers/action/terms/marl.py:44-49, 110-115, 190-191, 263-285` | kp、kd、action_scale、裁剪、扭矩限幅都是两人共用的同一组；没有延迟和执行器随机化 | 对称 |
| 机器人资产 | `isaacsim.py:303-320`；`dual_robot_isaacsim.py:34-38, 64-67` | 同一个 ArticulationCfg 和 URDF；关节和刚体顺序有代码校验 | 对称 |
| 域随机化 | `B/config_values/marl/g1/randomization.py:6-20` | 只固定了桌子材质，没有推搡，没有机器人摩擦或质量随机化 | 对称 |
| 手–桌接触传感器 | `isaacsim.py:509-522`；`core4d_bucket.py:217-218`（**已复核**） | 4 个过滤路径：Robot 左、右手，然后 Robot_1 左、右手，有代码校验 | 对称 |
| 参考的构建 | `holosoma_retargeting/dual_pull_runtime_reference.py:338-358` | 两人用同一套正向运动学和差分，横向偏移为 0；robot_a 对应人 0 | 对称（参考内容本身不同，见 README 第 5 节） |

## 不能用"各自跟踪自己的参考"解释的不对称

1. **生效中：** critic 只看 agent 0 的参考关节指令（见上表）。
   - critic 输入里有相位，参考又是一段固定片段，理论上 agent 1 的参考可以从相位推出来。
   - 两人拿到的 advantage 本来就相同。
   - 它是否导致了接触上的分化，**未验证**。
2. **未生效：** 推搡代码只推主机器人（`B/envs/wbt/wbt_manager.py:126-128`）。本次没有配置推搡。
3. **未验证，预计一致：** Robot_1 的身体接触传感器复用了主机器人的身体编号映射（`dual_robot_isaacsim.py:88-90, 103-106`）。两者资产相同，只影响 undesired_contacts 惩罚，权重 −0.1。
4. **未生效：** Robot_1 生成时 x 方向偏 0.8 m（`dual_robot_isaacsim.py:29-33`），重置时被参考状态覆盖。
5. **未查，不属于代码层面：** PhysX 在同一环境里求解两个 articulation 的先后顺序。

## 训练日志里两台机器人各自的接触率（已复核）

数据来自 `metrics.jsonl` 的 `Bucket/contact_agent{0,1}_mean`：每台机器人任一只手法向力 > 1 N 的比例，只记录、不进奖励。

| 训练轮次 | agent 0 | agent 1 |
|---|---|---|
| 1–10 | 0.214 | 0.225 |
| 101–500 | 0.158 | 0.285 |
| 2001–4000 | 0.039 | 0.200 |
| 8001–12000 | 0.018 | 0.640 |
