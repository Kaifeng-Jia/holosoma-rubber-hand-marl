# 写作代码导航：从源动作到双人策略与评测

核对日期：2026-09-25。用途是回答“这个描述在代码哪里成立”，不是安装、训练教程或论文正文。所有链接相对本文件，Mac clone 后可直接阅读；查看函数名即可定位，不依赖服务器绝对路径或 `logs/`。

阅读时区分三件事：源码支持什么、某次运行启用了什么、评测实际证明了什么。源码中的候选分支（包括默认关闭或失败的候选）不能自动当作训练方法；实际运行以保存的 `run_config.json`、checkpoint metadata、reference/asset manifest 及其 SHA256 契约为准。

## 1. 先认清主线与复用边界

```text
LAFAN / AMASS-SMPLX / InterMimic-OMOMO ── 通用格式入口 ─┐
CORE4D 两人+一物体 ── canonical adapter ──────────────┤
                                                     v
                Omni 单阶段 / nominal→physical two-stage
                                      ↓（可选，按资产契约）
                           wrist-only A1 后处理
                                      ↓
                paired compact → runtime FK/速度/统一时钟
                                      ↓
                 双机器人环境 → 共享 actor / team critic
                                      ↓
                    PD 子步控制 → actor-only 评测
```

这不是“所有数据、所有实验都经过同一条 A1 链”。尤其小桌存在从 Stage1 nominal 预览经审核晋升的资产路径；CORE4D fresh 初始化与 OMOMO/WBT warmstart 也必须分开写。

| 想回答的问题 | 先读的代码与关键符号 | 写作边界 |
| --- | --- | --- |
| 哪些属于复用的运动重定向基础？ | [InteractionMeshRetargeter](../../src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py)：`retarget_motion`、`iterate`、`solve_single_iteration` | Omni 的交互网格/逐帧约束优化是基础；该文件也含本项目扩展，不能把整个当前文件称为未修改上游。 |
| 哪些属于复用的 WBT/PPO 基础？ | [WBT 配置](../../src/holosoma/holosoma/config_values/wbt/g1/experiment.py)、[PPO](../../src/holosoma/holosoma/agents/ppo/ppo.py)：`PPO`、`EmpiricalNormalization`；[网络构造](../../src/holosoma/holosoma/agents/modules/module_utils.py)：`setup_ppo_actor_module`、`setup_ppo_critic_module` | 身体跟踪配置、策略网络、归一化及 PPO 目标不是本项目重新发明的算法。 |
| 本项目需要重点说明的接口是什么？ | 以下第 2–8 节：CORE4D adapter、双人配对与共享物体、A1、runtime、MAPPO batch/契约、reward 变体及 control state 刷新 | 将“适配、配对、训练接口和经验性探索”与“新的优化算法”分开。 |

## 2. 多源动作怎样进入共同表示？

| 阅读问题 | 文件 → 关键函数/类 | 应检查的契约 |
| --- | --- | --- |
| 通用入口究竟支持哪些源格式？ | [robot_retarget.py](../../src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py) → `load_motion_data`、`create_task_constants`、`main`；[data_type.py](../../src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py) | `robot_only` 下有 lafan、smplh、mocap、smplx；`object_interaction` 从 InterMimic `.pt` 读取。支持接口不等于已对所有数据集训练验证。 |
| LAFAN/AMASS 的前处理在哪里？ | [extract_global_positions.py](../../src/holosoma_retargeting/holosoma_retargeting/data_utils/extract_global_positions.py) → `extract_global_positions`；[prep_amass_smplx_for_rt.py](../../src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py) → `load_ori_npz_file`、`run_smplx_model`、`compute_height`、`main` | BVH 全局关节位置，或 SMPL-X 全局关节位置与身高，进入通用 loader。 |
| OMOMO/InterMimic 的位置与手腕姿态来自哪里？ | [src/utils.py](../../src/holosoma_retargeting/holosoma_retargeting/src/utils.py) → `load_intermimic_data`、`load_intermimic_wrist_quaternions`、`build_pt_wrist_palm_orientation_targets`；[OMOMO 示例](../../demo_scripts/demo_omomo_wb_tracking.sh) | 位置输入与腕部四元数是不同通道；示例脚本是用法，不是全部实际训练的记录。 |
| CORE4D 原始两人数据如何规范化？ | [prepare_core4d_sequence.py](../../scripts/prepare_core4d_sequence.py) → `main`；[core4d_adapter.py](../../src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_adapter.py) → `load_core4d_sequence`、`Core4DPairSequence`、`resample_core4d_pair_sequence`、`load_canonical_core4d_sequence` | 同步两人和物体帧；Y-up→Z-up；保存完整 127 关节、身体子集、身高和腕部通道。canonical 明确 `object=wxyz,wrist=xyzw`，不能泛称所有文件都同一四元数顺序。 |
| 物体网格和源坐标如何保留？ | [core4d_object_assets.py](../../src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_object_assets.py)、[core4d_adapter.py](../../src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_adapter.py) → `_convert_object_poses` | 不将源网格局部坐标、世界变换、后续物理 collider 混为一谈。 |

## 3. Omni、two-stage 与 wrist-only A1 如何衔接？

1. 从 [retarget_core4d_pair.py](../../scripts/retarget_core4d_pair.py) 的 `parse_args`、`_retarget_person`、`run` 开始。`--method omni` 与 `--method two-stage` 显式区分；该 CLI 当前默认是 `two-stage`。两个人独立调用求解器，最后组合为一份共享物体参考，不是双机器人联合优化。
2. 读 [core4d_retarget.py](../../src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_retarget.py) 的 `prepare_core4d_person_retarget_input`、`require_shared_physical_object_qpos`、`save_core4d_pair_reference`：每人可有不同人体尺度，但最终物理物体轨迹必须一致。
3. 读 [robot_retarget.py](../../src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py) 的 `run_fixed_object_size_adaptation`：Stage1 使用人各自缩放的 nominal 场景；Stage2 适配到显式传入的 physical object，后续可选 Stage3。CORE4D 的足部锚定、弹性约束等具体覆写在 `_retarget_person`，不能只看通用配置默认值。
4. A1 入口是 [refine_core4d_pair_a1.py](../../scripts/refine_core4d_pair_a1.py) 的 `run`、`_allowed_qpos_indices`、`_require_only_allowed_change`、`_apply_refinement_solver`。默认 `wrist-only`；目标构造见 [core4d_retarget.py](../../src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_retarget.py) 的 `build_core4d_a1_palm_orientation_targets`。
5. 腕部求解本体是 [InteractionMeshRetargeter.apply_pt_wrist_orientation_postprocess](../../src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py)：每只手只求 roll/pitch/yaw，共六个腕关节；浮动基座、腿、腰、肩、肘和物体从 baseline 精确复制。它不重新求解物体碰撞约束，因此姿态误差小不自动意味着无穿透或物理执行成功。

`full-arm`、`wrist-dominant`、Plan B、palm-collision 等候选仍在当前源码中。检查 [retargeter.py](../../src/holosoma_retargeting/holosoma_retargeting/config_types/retargeter.py) 的各 `enable=False`、入口参数及输出 manifest 的 `wrist_mode`/solver snapshot；不要把它们写成默认 A1，也不要因“有实现”推断其已被采纳。

小桌例外： [export_core4d_small_table_pair.py](../../scripts/export_core4d_small_table_pair.py) 从两人的 Stage1 `nominal_scaled` qpos 导出共享缩放物体预览，最初标记 `diagnostic_preview_not_training_asset`、`training_ready=False`。是否可训练还需后续 promotion/asset manifest；此脚本本身不是训练批准。对应样例可读 [smalltable runtime manifest](../../src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/core4d_pair_runtime_fps50.manifest.json)。

## 4. paired compact 怎样变成 runtime？

| 阅读问题 | 文件 → 关键符号 | 核心区别 |
| --- | --- | --- |
| compact 数据存什么？ | [core4d_pair_runtime_reference.py](../../src/holosoma_retargeting/holosoma_retargeting/core4d_pair_runtime_reference.py) → `Core4DCompactPairReference`、`load_core4d_compact_pair_reference` | 两机器人 `robot_qpos[T,2,36]` 和一个 `object_qpos[T,7]`；36=浮动基座7+关节29。 |
| 30 Hz 如何成为均匀 50 Hz？ | 同文件 → `resample_core4d_pair_reference`、`build_core4d_pair_runtime_reference`、`build_core4d_pair_runtime_reference_file`；[CLI](../../scripts/synthesize_core4d_smalltable_runtime_reference.py) → `main` | 位置/关节线性插值，四元数 SLERP；不循环，末帧不超出源时长。FK 和流形差分复用 [dual_pull_runtime_reference.py](../../src/holosoma_retargeting/holosoma_retargeting/dual_pull_runtime_reference.py) 的 `build_dual_pull_runtime_reference`。 |
| 环境为何不能只读播放文件？ | [paired_motion_reference.py](../../src/holosoma/holosoma/envs/marl/paired_motion_reference.py) → `PairedMotionReference`、`_validate_arrays`、`sample`；[core4d_smalltable_reference.py](../../src/holosoma/holosoma/envs/marl/core4d_smalltable_reference.py) → `Core4DSmallTableReference` | runtime 包含关节速度、body 位姿/线角速度及物体速度；文件边界 WXYZ，载入转 XYZW；CORE4D 扩展物体角速度。ViSER rollout 不等于完整训练参考。 |
| 旧 A1 复制布局与真实双人参考如何共存？ | [command/terms/marl.py](../../src/holosoma/holosoma/managers/command/terms/marl.py) → `PairedA1MotionCommand.setup`；[paired_a1_reference.py](../../src/holosoma/holosoma/envs/marl/paired_a1_reference.py) → `PairedA1Reference.from_motion_loader` | 显式 paired 文件已有两人位置，不再二次 lateral offset；未提供 paired 文件才走单人参考派生的旧路径。 |
| 参考重置、质心与物体原点速度谁负责？ | [command/terms/core4d_smalltable.py](../../src/holosoma/holosoma/managers/command/terms/core4d_smalltable.py) → `Core4DSmallTableMotionCommand`、`_write_reference_state`、`object_origin_velocity_to_com_velocity`、`object_com_velocity_to_origin_velocity` | runtime 的物体原点速度与 simulator COM 速度不能直接混用。 |

## 5. CORE4D MAPPO 与 OMOMO/WBT warmstart 应怎样分开写？

| 路径 | 实际入口和关键符号 | 初始化契约 |
| --- | --- | --- |
| CORE4D paired MAPPO | [train_core4d_smalltable.py](../../scripts/train_core4d_smalltable.py) → `main`；[core4d_smalltable_initialization.py](../../src/holosoma/holosoma/agents/mappo/core4d_smalltable_initialization.py) → `initialize_core4d_smalltable_model_bundle` | 新 actor、新 critic、新 normalizer、新 optimizer；`--resume` 才恢复训练状态。不是从 OMOMO actor warmstart。 |
| 单人 A1/WBT actor 接口扩展 | [convert_a1_checkpoint_for_marl.py](../../scripts/convert_a1_checkpoint_for_marl.py) → `main`、`_output_equivalence`；[checkpoint_compat.py](../../src/holosoma/holosoma/agents/ppo/checkpoint_compat.py) → `expand_ppo_checkpoint_for_teammate_obs`、`validate_lossless_expansion` | 154→158：新增四列权重为零；新增 normalizer mean=0、var/std=1；旧输出保持兼容，optimizer 状态移除。扩展不是已经学会合作。 |
| Plan5 warmstart 双人训练 | [train_plan5_push.py](../../scripts/train_plan5_push.py)；[mappo/initialization.py](../../src/holosoma/holosoma/agents/mappo/initialization.py) → `initialize_plan5_model_bundle`、`FrozenEmpiricalNormalization` | 载入经 SHA256/版本验证的 actor prior 与冻结 actor normalizer；team critic、critic normalizer 和 optimizer 新建。源 checkpoint 的数据来源仍需证据包中的运行记录确认。 |

MAPPO 的执行主干： [Core4DSmallTablePPO](../../src/holosoma/holosoma/agents/mappo/core4d_smalltable_ppo.py) 继承 [Plan5PPO](../../src/holosoma/holosoma/agents/mappo/ppo.py)。按 `collect_rollout` → `compute_team_returns_and_advantages` → `_update_minibatch` 阅读：每个物理环境一条 team reward/GAE，team advantage 分配到两个 actor 样本，仍使用 PPO clipped surrogate 与 value loss。[MultiAgentRolloutStorage](../../src/holosoma/holosoma/agents/mappo/storage.py) 和 [HomogeneousAgentBatchLayout](../../src/holosoma/holosoma/agents/mappo/batch_layout.py) 将 agent batch 与 team batch 分开。

不同物体场景由 [core4d_pair_experiments.py](../../src/holosoma/holosoma/config_values/marl/g1/core4d_pair_experiments.py) 的 `Core4DPairExperiment`、`get_core4d_pair_experiment` 选择资产与物理契约；[with_pair_experiment](../../src/holosoma/holosoma/config_values/marl/g1/core4d_smalltable_experiment.py) 接入配置。这不是每个物体一套新学习算法。

## 6. 158 / 527 / 29 分别是什么？

先看 [core4d_smalltable_observation.py](../../src/holosoma/holosoma/config_values/marl/g1/core4d_smalltable_observation.py)，再看 [marl/g1/observation.py](../../src/holosoma/holosoma/config_values/marl/g1/observation.py) 的 `plan5_actor_obs`、`plan5_teammate_obs`、`plan5_centralized_critic_obs` 和 [observation/terms/marl.py](../../src/holosoma/holosoma/managers/observation/terms/marl.py)。

| 接口 | 内容与尺寸 | 关键实现 |
| --- | --- | --- |
| 每机器人 actor 输入 158 | 本体/参考 `154 = 58关节位置速度命令 + 6参考朝向 + 3角速度 + 29关节位置 + 29关节速度 + 29前动作`；另加队友平面相对位置/速度4 | `paired_*`、`_real_teammate_planar_state_b`。队友状态在观察者 heading 坐标系，组配置 clip 为 [-1,1]；不是对方完整状态。 |
| 每环境 critic 输入 527 | 两机器人各228维的参考误差、14个跟踪 body 位姿、本体状态与动作，共456；加关节命令58、物体跟踪12、phase1 | `centralized_agent_*`、`centralized_shared_object_tracking`、`centralized_shared_phase`。注意当前 `centralized_shared_motion_command` 取 `command[:,0]`，不是两份命令拼接。 |
| 每机器人动作 29 | 同一个 actor 参数集合对两个机器人各产生29个关节动作；环境布局 `[N,2,29]` | [core4d_smalltable_runner.py](../../src/holosoma/holosoma/agents/mappo/core4d_smalltable_runner.py) → `Core4DSmallTablePolicyRunner._prepare`、`sample`、`decide`。 |

actor 输入没有独立的当前物体位姿/接触力块；集中式 critic 有额外的双人和物体信息。actor-only 推理不意味着“无参考动作输入”。不要把这套158维契约套到其他扩展 demo；各自 checkpoint 需单独检查。

## 7. 共享基础11项奖励与变体去哪里看？

基础配置在 [core4d_smalltable_reward.py](../../src/holosoma/holosoma/config_values/marl/g1/core4d_smalltable_reward.py)，公式实现主要在 [reward/terms/marl.py](../../src/holosoma/holosoma/managers/reward/terms/marl.py)。以下是配置权重，不是已经乘过 timestep 的 episode reward。

| 基础项 | 权重 |
| --- | ---: |
| `motion_global_ref_position_error_exp` | 0.5 |
| `motion_global_ref_orientation_error_exp` | 0.5 |
| `motion_relative_body_position_error_exp` | 1.0 |
| `motion_relative_body_orientation_error_exp` | 1.0 |
| `motion_global_body_lin_vel` | 1.0 |
| `motion_global_body_ang_vel` | 1.0 |
| `action_rate_l2`（函数 `penalty_action_rate`） | -0.1 |
| `limits_dof_pos` | -10.0 |
| `undesired_contacts` | -0.1 |
| `object_global_ref_position_error_exp` | 1.0 |
| `object_global_ref_orientation_error_exp` | 1.0 |

[RewardManager.compute](../../src/holosoma/holosoma/managers/reward/manager.py) 将原始项乘权重和 `dt` 一次。身体项按两机器人聚合到 team reward，物体项对应一个共享物体。

| 变体问题 | 入口 → 实现 | 与基础的关系 |
| --- | --- | --- |
| 原始 interaction mesh 探索 | `with_interaction_reward_term` → [InteractionMeshReward](../../src/holosoma/holosoma/managers/reward/terms/interaction_mesh.py) 的 `grouped_quadratic_error`、`__call__`；预计算见 [compile_training_reference](../../src/holosoma_retargeting/holosoma_retargeting/interaction_mesh_training_reference.py) | 显式选择后在11项之上追加 `interaction_mesh`；与重定向器同名不代表优化问题相同。 |
| 物体 z 误差加权 | `with_object_z_error_weight` → `object_global_ref_position_error_exp` | 改现有物体位置误差里的 z² 系数，不新增一项。 |
| 高度惩罚探索 | `with_object_height_penalty` → `ObjectHeightErrorPenalty` | 仅非零权重时新增负向平方高度误差；默认0时该项不存在。 |
| 桶 A/B 与 ablation、小桌5kg A | [core4d_bucket_reward.py](../../src/holosoma/holosoma/config_values/marl/g1/core4d_bucket_reward.py) → `with_bucket_interaction_reward`；[core4d_bucket_contract.py](../../src/holosoma/holosoma/config_values/marl/g1/core4d_bucket_contract.py) → `bucket_block_weights`、`bucket_reward_contract`；[BucketInteractionReward](../../src/holosoma/holosoma/managers/reward/terms/core4d_bucket.py) → `contact_gate`、`__call__` | 保留6身体项+3正则，用一个位置/朝向/高度/关系与接触门控 block 替换2个物体项。具体 variant 会去掉关系或高度贡献；不能统一称为“基础11项+新奖励”。 |

判定一次实验实际用了哪种奖励，请沿 [train_core4d_smalltable.py](../../scripts/train_core4d_smalltable.py) 的配置分支到 `run_config` 构造处：核对 `reward_terms`、`reward_variant`、z/height 参数、`interaction_contract`、`bucket_reward_contract`、实验与资产 SHA256。checkpoint 侧用 [expected_core4d_smalltable_checkpoint_metadata / validate_core4d_smalltable_checkpoint](../../src/holosoma/holosoma/agents/mappo/core4d_smalltable_ppo.py) 防止不兼容恢复。训练日志中的 `reward_mean` 还可能含 timeout value bootstrap；不要直接当成纯环境奖励。

## 8. action 怎样到 PD？为什么要子步刷新？

按这条调用链阅读即可：

1. [BaseTask._physics_step](../../src/holosoma/holosoma/envs/base_task/base_task.py)：在 `control_decimation` 循环内，每个 physics substep 都调用 action manager 再推进模拟。
2. [DualJointPositionActionTerm](../../src/holosoma/holosoma/managers/action/terms/marl.py)：`process_actions` 处理动作裁剪/延迟；`apply_actions` 取最新状态；`_compute_torques` 应用动作缩放、计算位置目标 PD 并限幅，动作不是直接输出关节力矩。
3. [dual_robot_isaacsim.py](../../src/holosoma/holosoma/simulator/isaacsim/dual_robot_isaacsim.py)：`get_agent_dof_control_state` 每次读取两机器人最新 articulation joint buffers；`apply_agent_torques` 分别施加力矩。

关键区分：`agent_dof_pos/vel` 是供 observation/reward/critic 使用的 control-step 快照；PD 必须读取 physics-substep 的新状态，不能在整个 decimation 期间复用旧快照。实际 physics/control 频率仍以该运行契约为准。回归检查见 [test_position_controller_refreshes_live_state_each_physics_substep](../../src/holosoma/tests/managers/action/test_dual_joint_control.py)。

## 9. 评测在测什么，不能从什么推断成功？

- 入口：[evaluate_core4d_smalltable.py](../../scripts/evaluate_core4d_smalltable.py) 的 `main`、`_snapshot`、`_episode_height_metrics`。核对 checkpoint/reference/物理契约、actor-only、关闭 actor observation noise、终止前物理状态抓取与代表 episode 的选择方式。
- 推理与输出：[core4d_smalltable_evaluation.py](../../src/holosoma/holosoma/agents/mappo/core4d_smalltable_evaluation.py) 的 `deterministic_core4d_smalltable_actions`、`save_core4d_smalltable_viser`、`core4d_smalltable_evaluation_reward_metadata`。`reward_sum` 使用共同原始 tracking score；interaction/z加权/高度惩罚/bucket 训练项不一定进入评测总分，metadata 会明确两种 regime。
- 终止定义：[core4d_smalltable_termination.py](../../src/holosoma/holosoma/config_values/marl/g1/core4d_smalltable_termination.py)、[reference_horizon_reached](../../src/holosoma/holosoma/managers/termination/terms/core4d_smalltable.py)、[Core4DSmallTableManager._check_termination](../../src/holosoma/holosoma/envs/marl/core4d_smalltable_manager.py)。报告中分开 `completed_reference`、`bad_tracking`、物体位置 RMSE、位移和高度指标；不能仅凭一项 reward 或播放画面认定协作成功。

这里只导航评测实现。具体某 checkpoint 的成功率、RMSE、候选失败结论和演示素材应引用交接证据包，不从当前源码默认参数倒推，也不只引用服务器 `logs/` 路径。

## 10. 想快速核查叙述时，读哪些测试？

| 叙述 | 现有回归测试（阅读即可；本次交接未运行仿真/训练） |
| --- | --- |
| CORE4D 源转换与双人尺度/物体一致性 | [test_core4d_adapter.py](../../tests/test_core4d_adapter.py)、[test_core4d_retarget.py](../../tests/test_core4d_retarget.py)、[test_retarget_core4d_pair_cli.py](../../tests/test_retarget_core4d_pair_cli.py) |
| A1 只改六个腕关节；候选模式边界 | [test_pt_wrist_orientation.py](../../tests/retargeting/test_pt_wrist_orientation.py)、[test_refine_core4d_pair_a1.py](../../tests/test_refine_core4d_pair_a1.py) |
| runtime 采样与 explicit paired 不重复位移 | [test_core4d_pair_runtime_reference.py](../../src/holosoma_retargeting/tests/test_core4d_pair_runtime_reference.py)、[test_paired_motion_reference.py](../../src/holosoma/tests/envs/marl/test_paired_motion_reference.py)、[test_paired_a1_command.py](../../src/holosoma/tests/managers/command/test_paired_a1_command.py) |
| 158/527/29、fresh 初始化和 actor-only 推理 | [test_core4d_smalltable_observation.py](../../src/holosoma/tests/managers/observation/test_core4d_smalltable_observation.py)、[test_core4d_smalltable_mappo.py](../../src/holosoma/tests/agents/mappo/test_core4d_smalltable_mappo.py) |
| 154→158 无损扩展与原训练不被隐式替换 | [test_checkpoint_compat.py](../../src/holosoma/tests/agents/ppo/test_checkpoint_compat.py)、[test_marl_compat_config.py](../../src/holosoma/tests/agents/ppo/test_marl_compat_config.py) |
| reward/恢复契约/PD 刷新 | [test_core4d_bucket_reward.py](../../src/holosoma/tests/managers/reward/test_core4d_bucket_reward.py)、[test_core4d_smalltable_interaction_checkpoint.py](../../src/holosoma/tests/agents/mappo/test_core4d_smalltable_interaction_checkpoint.py)、[test_core4d_height_penalty_checkpoint.py](../../src/holosoma/tests/agents/mappo/test_core4d_height_penalty_checkpoint.py)、[test_dual_joint_control.py](../../src/holosoma/tests/managers/action/test_dual_joint_control.py) |
