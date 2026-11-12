# T03 C 第二版：四项核对（Kevin 2026-11-12 第 3 条）

本文只写事实，不做解释。时间均为 America/New_York。

- 第 a、c 项只用已有的评测输出，在 CPU 上计算；第 a 项另在本机加跑了两个评测。
- 第 b 项离线重建配置，不启动仿真。
- 第 d 项在本机 GPU 上做训练回放，回放期间不跑别的任务。

## 用时与运行记录

- **00:34** 暂停本机流水线。
- **00:57** 按记录的进程号停掉旧流水线（PID 345176），启动新版（PID 422797，git 03081e7）。
- **00:57–01:04** 第 d 项回放，共 9 次。
- **01:05–01:55** 第 a 项本机复核，2 个评测。
- 第 b、c 项的脚本在此期间在 CPU 上运行。
- 暂停用的 PAUSE 标志文件已移到 `_to_delete/post_train_PAUSE_flag_20261112T0104`，没有删除。

| 项 | 脚本 | 输出 |
|---|---|---|
| a | `system1/t03/c2_recheck_compare.py`（评测由流水线按 `recheck.json` 启动） | `a_same_gpu/` |
| b | `system1/t03/c2_eval_path_diff.py` | `b_eval_path/` |
| c | `system1/t03/c2_object_check.py`、`system1/render/u5/render_keyframes.py` | `c_object/` |
| d | `system1/t03/c2_bound_fraction.py` | `d_bound_fraction/` |

每个脚本的定义都写在其 docstring 里，写于计算之前。

## a 同一种 GPU 复核

- **做法：** checkpoint 从云端拷到本机，sha256 一致。在本机 RTX 5070 上用同一评测脚本、同一份初始状态文件，重跑 ±2，共 50 个扰动状态加 nominal。
- **对照：** 云端 RTX 4090 上的同一评测。

| | Pull C-in 第 1050 轮 | Bin C-in 第 1000 轮 |
|---|---|---|
| 完成：本机 / 云端（/50） | 43 / 41 | 34 / 39 |
| 平均进度：本机 / 云端 | 0.937 / 0.928 | 0.882 / 0.917 |
| 两边都完成 | 34 | 26 |
| 只在本机完成 | 9 | 8 |
| 只在云端完成 | 7 | 13 |
| 两边都未完成 | 0 | 3 |
| 精确 McNemar 双侧 p | 0.804 | 0.383 |
| 执行步数相同的状态（/50） | 34 | 26 |
| SONIC 摔倒：本机 / 云端 | 0 / 0 | 0 / 0 |
| nominal：本机 / 云端 | 完成 / 完成（都是 316 步） | 未完成，393 步 / 完成，497 步 |

- 逐状态结果见 `a_same_gpu/per_state.csv`，包括完成与否、执行步数、进度、物体 RMSE 和终止时物体位置误差。
- 来源与 sha256 见 `a_same_gpu/provenance.csv`。
- **参考：** 同一份 Pull 初始状态、方案 A（Δq = 0），两台机器的对照在 `results_system1/T03/c2_first_evals/dq0_cloud_vs_A_laptop.csv`。

## b 评测路径：C 与 D 同任务 ±2

**比较的组合**
- C 一侧：C-in 的云端评测，以及第 a 项的本机复核。
- D 一侧：同一评测脚本，种子 721，都在本机 5070 上评测：
  - Pull：论文主结果第 1050、8050 轮，以及和 C Pull 一样从零训练的 V3-25 第 1050、8050 轮；
  - Bin：论文主结果第 2000、12000 轮；
  - Table：论文主结果第 12000 轮。

**输出文件**
- `manifest_diff.csv`：run_manifest 逐项比较。
- `checks.csv`：下表各指定项。
- `termination_config.json`：重建出的终止配置。

| 项 | 结果 |
|---|---|
| residual kind | C 为 `sonic_residual`，`system1_scheme_c_kind = in`；D 为 policy，没有 residual |
| checkpoint sha256 | C：manifest 值等于本机拷贝的 sha256，拷贝时已与云端文件核对。D：最终 checkpoint 与论文登记前缀一致（Pull 727630e9、Table f163d11a、Bin 7ba039fb）；V3-25 没有登记值 |
| 初始状态文件 sha256 | 所有组合 C = D = 冻结文件（Pull 473c1fc0…、Bin 631026d3…、Table 389e8da5…） |
| 终止阈值 | manifest 没有记录。按评测脚本的配置构建调用离线重建：只取决于 `--case`；控制方式只通过 `with_sonic_pd` 改 PD 增益。`config.termination` 在 C 和 D 之间逐项相同，都是 `timeout` 加 `JointBadTrackingZOnly`：bad_ref_pos 0.5、bad_ref_ori 0.8、bad_motion_body_pos 0.25、minimum_ref_body_height 0.4、bad_object_pos 0.25、bad_object_ori 0.8，身体列表也相同 |
| 确定性动作 | 两边都调用 `deterministic_core4d_smalltable_actions`，取 actor 均值。C 再交给 `SonicResidualEnv.step`。manifest：C `scheme_c_actor_mean_then_sonic`，D `actor_mean_only_critic_forbidden` |
| TF32 | C 的 manifest 记为 `tf32 = false`。D 的 manifest 没有这个字段。全仓库搜索设置 TF32 的位置共 4 处：<br>• `system1/sonic_controller.py`：`allow_tf32 = self.tf32`，评测时为 False；<br>• `system1/sonic/native_eval_dump.py` 和 `check_f3_batch.py`：独立的 SONIC 工具，评测脚本不导入；<br>• `src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py`：`set_float32_matmul_precision("high")`。<br>在 hssim_s1 中导入评测脚本用到的 holosoma 模块后核对：matmul 精度仍是 "highest"，`allow_tf32` 为 False，`fast_sac` 的模块没有被导入。仿真启动（`init_sim_imports`）之后的导入不在这次核对范围内 |
| 回合时长、观测噪声、终点规则 | 相同 |
| 机器人和物体 URDF、读回的物体质量、惯量、材质 | 相同 |
| 评测代码 | 各评测启动时的 git HEAD 之间，评测脚本、`scheme_c`、`scheme_a`、`sonic_controller` 没有差异。唯一例外是 D Bin 第 2000 轮：它在较早的版本 bcf6185 上跑，差异只在 C 的方案类型开关和常数 Δq 两处，D 的 policy 路径没有变（`manifest_diff.csv`） |
| GPU | 云端 C 评测是 RTX 4090；本机复核和 D 都是 RTX 5070 Laptop |
| 其他不同的 manifest 字段 | Pull 对论文 D：`v3_ablation`。C Pull 的 checkpoint 带 `from_scratch: true`，论文 D Pull 为 null。Pull 对 V3-25：没有其他不同。Table、Bin：没有其他不同 |

## c 物体是否到位

**定义**（写在脚本 docstring 里）
- **推动比例：** 沿用 T02 预注册 9.1。
- **物体 RMSE：** 评测输出 summary 里的 `object_rmse_3d_m`。
- **终止时物体误差：** 取最后一个保存的状态，位置误差为 |物体 − 参考|，朝向误差为 2·acos|⟨q, q_ref⟩|。与方案 A 返包的定义相同。

下表只统计判为完成的回合（50 个扰动状态中完成的那些）。最后一列统计全部 50 个扰动回合中，终止位置误差超过 0.25 m 的个数。

| 任务 | 组 | 完成数 | 推动比例中位数 | 物体 RMSE 中位数 (m) | 终止位置误差 中位数 / 最大 (m) | 终止朝向误差 中位数 / 最大 (rad) | 全部 50 回合中位置误差 > 0.25 m |
|---|---|---|---|---|---|---|---|
| Pull | C-in 第 1050 轮（云端） | 41 | 0.886 | 0.090 | 0.123 / 0.216 | 0.140 / 0.287 | 7 |
| Pull | D 论文 第 1050 轮 | 35 | 0.778 | 0.102 | 0.152 / 0.232 | 0.078 / 0.265 | 14 |
| Pull | D 论文 第 8050 轮（最终） | 50 | 0.990 | 0.017 | 0.011 / 0.128 | 0.020 / 0.043 | 0 |
| Pull | D 从零训练 V3-25 第 1050 轮 | 8 | 0.898 | 0.100 | 0.130 / 0.223 | 0.195 / 0.341 | 41 |
| Pull | D 从零训练 V3-25 第 8050 轮（最终） | 49 | 0.979 | 0.037 | 0.036 / 0.153 | 0.029 / 0.107 | 1 |
| Table | C-in 第 1000 轮（云端） | 43 | 1.013 | 0.061 | 0.067 / 0.226 | 0.079 / 0.488 | 6 |
| Table | D 论文 第 2000 轮（云端） | 24 | 0.963 | 0.108 | 0.155 / 0.246 | 0.067 / 0.237 | 19 |
| Table | D 论文 第 12000 轮（最终） | 35 | 1.006 | 0.091 | 0.123 / 0.191 | 0.051 / 0.129 | 15 |
| Bin | C-in 第 1000 轮（云端） | 39 | 1.029 | 0.100 | 0.127 / 0.200 | 0.238 / 0.396 | 5 |
| Bin | D 论文 第 2000 轮 | 27 | 1.048 | 0.103 | 0.152 / 0.183 | 0.036 / 0.159 | 6 |
| Bin | D 论文 第 12000 轮（最终） | 50 | 1.032 | 0.068 | 0.084 / 0.121 | 0.016 / 0.129 | 0 |

**其他输出**
- 逐回合数据见 `c_object/episodes.csv`；均值、90 分位、全部回合口径见 `c_object/summary.csv`。
- **俯视图：** `c_object/topview_nominal_{pull,table,bucket}.png`。画的是 nominal 回合，左为 C-in，右为 D 论文最终 checkpoint。内容包括物体中心和两台机器人骨盆的 x-y 轨迹，灰色虚线为参考。方案颜色按 `system1/render/scheme_colors.py`，C 绿、D 橙。
- **关键帧拼图：** `c_object/keyframes_nominal_{pull,table,bucket}_cin.png`，C-in nominal 回合的 6 帧，等间隔取，含首尾。渲染沿用 U5 流水线：`g1_native_execution` 预设，Unitree 原色，棋盘地面，论文 F3 同任务相机。每张拼图的来源和 sha256 记录在同名 `.json` 里。

**物体颜色说明：** U5 流水线用物体 URDF 自带的颜色，所以 Pull 和 Table 的桌子是浅蓝色，Bin 是灰色（rgba 0.4）。论文 U5 F3 的 Bin 原图，manifest 里也记为 rgba 0.4，即 `role_geom_mapping` 中 `bucket003_visual` 那一项。我没有改色。

## d C-in 动作顶在上限的比例（|a| ≥ 0.99）

**做法**
- 和开训前检查同一套回放：用该 run 的正式训练命令（取自云端启动记录），加上 `--resume <checkpoint> --num-envs 256 --steps-per-env 240 --iterations 1`，输出到临时目录。
- 打开 `SYSTEM1_C_DIAG`，记录 learner 存下的采样动作 a，即截断到 [−1, 1] 之前的值。
- 统计每个关节在 240 × 256 × 2 = 122,880 个样本中的比例。
- 本机 RTX 5070，TF32 与训练相同，为开。
- Pull 用保存的第 1050、2050、3050 轮 checkpoint。

**说明**
- a 以 float16 记录。0.99 附近的取值间隔是 0.00049，所以离 0.99 不到 0.0005 的值，可能落在阈值的任一侧。
- 9 次回放的检查项：|Δq|/上限 的最大值都是 1.0；交给 SONIC 的参考越过关节限位的量为 0；参考核对误差为 0。

**全关节（第 1000 / 2000 / 3000 轮）**

| 任务 | \|a\| ≥ 0.99 | \|a\| > 1 | 训练日志 frac_clipped（第 1000 / 2000 / 3000 轮） |
|---|---|---|---|
| Pull | 0.9% / 2.8% / 5.0% | 0.8% / 2.7% / 4.8% | 0.6% / 2.4% / 4.3% |
| Table | 2.5% / 6.5% / 10.0% | 2.3% / 6.3% / 9.7% | 2.7% / 7.4% / 11.8% |
| Bin | 1.8% / 4.1% / 6.8% | 1.7% / 4.0% / 6.6% | 1.9% / 4.2% / 6.5% |

**按第 3000 轮排序，各任务比例最高的 5 个关节**（上限一列为 scale，单位 rad）

| 任务 | 关节 | 上限 (rad) | 第 1000 轮 | 第 2000 轮 | 第 3000 轮 |
|---|---|---|---|---|---|
| Pull | right_ankle_roll | 0.4386 | 1.1% | 15.7% | 27.1% |
| Pull | left_wrist_roll | 0.4386 | 5.9% | 13.9% | 18.5% |
| Pull | right_wrist_yaw | 0.0745 | 5.0% | 10.6% | 16.7% |
| Pull | left_ankle_pitch | 0.4386 | 1.6% | 3.3% | 13.8% |
| Pull | right_knee | 0.3507 | 0.8% | 6.3% | 11.7% |
| Table | left_wrist_yaw | 0.0745 | 4.5% | 17.3% | 35.4% |
| Table | right_shoulder_yaw | 0.4386 | 4.5% | 23.9% | 29.3% |
| Table | right_wrist_roll | 0.4386 | 23.6% | 28.6% | 27.7% |
| Table | right_wrist_pitch | 0.0745 | 1.8% | 13.9% | 26.2% |
| Table | left_ankle_roll | 0.4386 | 4.3% | 15.7% | 21.8% |
| Bin | right_wrist_yaw | 0.0745 | 2.2% | 11.5% | 27.2% |
| Bin | left_wrist_yaw | 0.0745 | 7.6% | 16.5% | 21.3% |
| Bin | right_ankle_roll | 0.4386 | 1.7% | 7.4% | 19.6% |
| Bin | left_ankle_roll | 0.4386 | 3.7% | 11.0% | 18.1% |
| Bin | left_shoulder_yaw | 0.4386 | 3.1% | 9.6% | 14.3% |

- 29 个关节的完整数据见 `d_bound_fraction/per_joint.csv`，包括 |a| ≥ 0.99、|a| > 1 和平均 |a|。
- 每次回放的命令、checkpoint sha256、git HEAD 和诊断摘要见 `d_bound_fraction/replays.json`。

## 与任务书的偏离和其他记录

- 第 b 项的终止阈值不在 manifest 里，是离线重建配置得到的，不是从评测时的实际运行状态读出来的。
- 第 c 项的 C 评测在云端 4090 上跑，D 评测在本机 5070 上跑（D Table 第 2000 轮在云端）。同一 checkpoint 在两台机器上的差异见第 a 项。
- 第 d 项的回放在本机 5070 上跑；训练本身在云端 4090 上。
- 第 d 项最后一次回放启动时，git HEAD 是 afdc9e8，前 8 次是 03081e7。两者之间只改了核对脚本和 watcher，没有改训练和环境代码。
