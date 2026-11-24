# T13 第四版准备：第 2 节新代码和检查、第 4 节 A 本机提速

- **依据：**`docs/system1/T13_parallel_runs_zh.md`（sha256 前 16 位 `043ac2905badc51a`），Kevin 2026-11-23：第 2 节的新代码和检查、第 4 节 A 现在就做；测试和验证状态等训练参考定了再生成；不开云。
- **第二轮：**Kevin 2026-11-24 的八条决定（第 7 节）。按决定改了物体点噪声，重跑了受影响的检查，接上了验证脚本，备好了按台的手部力诊断。
- **第三轮（2026-11-24 11:04 起）：**换上 T14 第 6 轮 C 参考（Table k = 1.0、Push m = 1.0）后的训练入口、测试和验证状态、回放检查、冒烟、验证测试、云上运行器，见 `t14ref/README_CN.md`。
- **时间：**第一轮 2026-11-23 23:50 – 2026-11-24 01:00；第二轮 2026-11-24 01:00 – 02:20（纽约时间）。本机 RTX 5070 Laptop，环境 `hssim_s1`。没有开云。
- **训练参考还没定：**下面的检查用的是现有参考（见第 3 节），不是 T14 的新参考。
- **开关：**新功能全部加了开关，默认关闭；关闭时和原来的实现相同。唯一例外是提速改动，见第 5 节。

## 1. 实现了什么（开关和文件）

| 内容 | 开关 | 文件 |
|---|---|---|
| 观测 v1（actor 154 + 5 = 159） | `--obs-v1` | `holosoma/managers/observation/terms/marl_v1.py`、`config_values/marl/g1/observation_v1.py` |
| 噪声 v1（训练开；验证、评测和检查用 `--obs-v1-noise off`） | 同上，加 `--obs-v1-noise on/off` | 同上；数值都放在 `marl_v1.py` 的 `NOISE_V1`、`NECK_POINT_TORSO_M` 和截断常数这一处 |
| 训练中的 Isaac 验证和评测用观测 v1（噪声关） | `scripts/evaluate_paper_supplement.py --obs-v1 --critic-v1` | 评测脚本里观测强制噪声关；只读 actor 和归一化器，不加载 critic；summary 记 `system1_t13v4` |
| critic 修正：两台的参考关节指令（critic 527 → 585） | `--critic-v1` | `marl_v1.centralized_both_motion_commands` |
| 接触三项：HDMI `eef_contact_exp` 奖励、HDMI `cum_lost_contact_steps` 终止、InterMimic 不该碰部位惩罚 | `--contact-v1 <标签文件>` | `holosoma/managers/reward/terms/contact_v1.py`（数值都在 `CONTACT_V1`）；仿真开关 `enable_object_link_contact`，是"物体 – 每个机器人连杆"的轻量接触传感器 |
| 逐帧接触标签（应接触、接触点、人体部位远近） | — | `system1/t13v4/contact_labels.py` |
| 每台各算各的奖励（D-Table-PA） | `--per-agent-reward` | `holosoma/managers/reward/terms/marl_per_agent.py`、`managers/reward/per_agent_team_manager.py`、`envs/marl/core4d_bucket_per_agent_manager.py`、`system1/t13v4/per_agent_ppo.py` |
| D-Push 初始化：158 维先验 → 159 维，新增输入权重置零 | Push 加 `--obs-v1` | `system1/t13v4/mappo_v1.py` |
| T10b 随机化接到 Push（Plan5）训练脚本 | `--system1-randomization` | `scripts/train_plan5_push.py` |
| MAPPO 尺寸改为模块常数（默认值不变），v1 时打补丁 | 随 `--obs-v1` / `--critic-v1` | `agents/mappo/runner.py`、`ppo.py`、`system1/t13v4/mappo_v1.py` |
| 检查、计时、同步扫描、同种子轨迹 | `--t13v4-check`、`--t13v4-profile`、`--t13v4-sync-scan`、`--t13v4-trace` | `system1/t13v4/isaac_checks.py`、`profiling.py`、`compare_traces.py`；单元测试 `system1/tests/test_t13v4.py`、`test_t13v4_per_agent.py` |
| 提速（默认开，结果逐位不变） | 环境变量 `HOLOSOMA_S1_SPEEDUP=0` 关闭 | `holosoma/utils/system1_speedup.py`，改动见第 5 节 |

### 颈后点

- 取值：torso_link 坐标系里 **(0.003, −0.0005, 0.2898) m**。
- 取法：`torso_link_rev_1_0.STL` 顶部有两个原厂吊环（两个独立网格分量），中心在 y = +0.0722 和 −0.0732，x = 0.003，z = 0.2898。颈后点取这两个吊环中心的中点。
- 网格里颈后正中没有单独的"把手"分量，所以这一点是我的解读。
- Kevin 2026-11-24：先用这个点，以后可能再改。改这个点只需要改 `NECK_POINT_TORSO_M` 这一处。

### 人体部位对应表（不该碰部位惩罚用）

- 这张表和 T14 Push 腿部检查共用同一张：`system1/t14/body_parts.py` 的 `LINK_TO_SMPLX`。
- 规则：按重定向的关键点对应关系（`data_type.py` 的 `JOINTS_MAPPING`）。人体部位 = SMPL-X 顶点里，蒙皮权重最大的关节属于所列关节的那些顶点。

| 机器人连杆 | 人体部位（SMPL-X 关节） |
|---|---|
| pelvis | 骨盆（0） |
| left/right_hip_roll_link、left/right_hip_yaw_link | 大腿（1 / 2） |
| left/right_knee_link | 小腿（4 / 5） |
| left/right_ankle_roll_link | 脚（7、10 / 8、11） |
| torso_link | 躯干（3、6、9、13、14） |
| head_link | 头颈（12、15、22–24） |
| left/right_shoulder_yaw_link | 上臂（16 / 17） |
| left/right_elbow_link、left/right_wrist_{roll,pitch,yaw}_link | 前臂（18 / 19） |
| 两只橡胶手 | 不计（手的接触是要的） |

## 2. 第 2 节第 4 条的检查

### 2.1 单元检查（不用仿真）

`system1/tests/test_t13v4.py`、`test_t13v4_per_agent.py`，全部通过（`records/test_t13v4*.log`）。

**观测 v1**
- 和按定义用 numpy 手算的结果对比：噪声关时误差 ≤ 1e-5（实际约 1e-7）；噪声开时，把状态里的偏移计入后同样 ≤ 1e-5。
- 噪声关时，参考朝向这一项和旧项逐值相同。
- 噪声分布（4096 个环境）：每回合偏移和每步噪声都在设定范围内，标准差等于均匀分布的理论值。偏移只在回合重置时重抽。
- 颈后点和航向偏差两台各自独立抽（两台之间的相关系数：颈后点偏移 −0.034，航向偏差 0.006）。
- 物体点的偏移和每步噪声每个环境只抽一份，两台用同一份（按 Kevin 2026-11-24 第 3 条；测试检查形状是 [环境, 1, 3]，两台得到的物体点噪声相同）。

**接触三项**
- 随机状态下和定义对比，误差 7.6e-9。
- 三种构造情况，结果都和定义一致：
  - **手悬空**（离接触点 1 m、无力）：奖励 0.1389 = 5·e^(−1/0.3)·e^(−10/40)；第 25 步终止。
  - **手贴上但不用力**：奖励 3.894 = 5·e^(−10/40)，记丢接触；20 N 时奖励 5.0、不丢接触。
  - **用腿顶**：左小腿 50 N，而人的小腿离物体超过 0.1 m：该机器人奖励 5·e^(−5) = 0.0337，另一台 5.0；人的对应部位在附近时不罚。
- 不该接触的帧，奖励为 1。

**每台各算的奖励（PA）**
- 两台给同一份团队奖励和价值时，逐台 GAE 的回报和团队实现逐值相同（差 0.0）。
- 优势的归一化照 demo3 的做法：两台合在一起、用总体标准差；和团队流用同一估计量相比，差 3e-8。

### 2.2 仿真内检查

8 个环境；结果在 `isaac_checks.csv` 和 `records/isaac_checks_*.json`。

| 检查 | Table | Push | Table-PA |
|---|---|---|---|
| 维数 | actor [·,2,154] + teammate [·,2,5] = 159；critic 585 | 同 | 同 |
| teammate 顺序 | neck_partner_rel_pos_b（2）、object_rel_pos_b（3） | 同 | 同 |
| critic 含两台的参考关节指令（116 维） | 是 | 是 | 是 |
| 观测 v1 和手算，噪声关 | 3.3e-7 | 2.6e-7 | — |
| 观测 v1 和手算，噪声开（物体点噪声两台共用，第二轮重跑） | 6.8e-7；噪声都在范围内；物体点噪声形状 [8, 1, 3] | 6.8e-7；同 | — |
| 参考回放 | 687 帧：奖励误差 2.6e-6；丢接触、终止逐帧一致 | 309 帧：奖励误差 2.7e-6；逐帧一致 | 687 帧：奖励误差 4.1e-6；逐帧一致 |
| 两个传感器测到的手部力 | 差 0.0 N | 差 0.0 N | — |
| 逐台项的均值和团队项 | — | — | 9 项都 0.0；物体项共享 |

**回放里的数据（只记录；噪声关）**
- 现有 Table 参考：应接触的帧里，手部力 > 1 N 的只占 15%，丢接触 85%，有 309 帧（按环境计）会触发终止。原因是旧参考的手不压桌面。
- 现有 Push 参考：手部力 > 1 N 占 87%，丢接触 13%，没有触发终止。
- 两个任务都没有触发不该碰部位的惩罚。
- 这些比例和回放前的仿真状态有关，见第 2.4 节。

**每步的 GPU→CPU 同步**
- 新代码本身：用 CUDA 假环境逐行扫描，0 处。扫描中修掉了两处：计数器 `tensor += int`，以及用 Python 列表做索引。
- 在仿真里调用新项时，仍有来自现有访问函数的同步：
  - 读物体位姿要经过 `ObjectRegistry.resolve_indices`，里面有 `max`、`unique`、`.item()`；
  - IsaacLab 传感器的惰性更新（`sensor_base.py:355`）。
- 旧的物体项、bucket 项和 critic 也会经过这两处。这次没有改它们。

**200 轮冒烟**
- 共同配置：2048 个环境，`--system1-randomization --obs-v1 --critic-v1 --contact-v1`。
- 记录在 `smoke_200.json`：

| run | 结果 | 末轮 s/轮 | 平均奖励（首轮 → 末轮） | 每轮重置数（首轮 → 末轮） |
|---|---|---|---|---|
| D-Table | passed，无非有限值（指标和 checkpoint 参数） | 4.79（中位 4.89） | 0.0015 → 0.097 | 281 → 647 |
| D-Table-PA | passed，无非有限值 | 5.25（中位 5.04） | 0.0015 → 0.120 | 281 → 595 |
| D-Push（158 维先验 → 159 维初始化，加随机化） | passed，无非有限值 | 4.17（中位 4.19） | 0.124 → 0.081 | 365 → 591（无超时） |

- 冒烟只看能不能稳定跑、有没有非有限值，不评价学习效果。
- D-Push 的平均奖励在 200 轮里从 0.124 降到 0.081，如实记下。

### 2.3 训练中 Isaac 验证用观测 v1（Kevin 2026-11-24 第 5 条）

- 接到 `scripts/evaluate_paper_supplement.py`，命令加 `--obs-v1 --critic-v1`（只对 `--controller policy` 有效）。
  - 观测配置用 `observation_v1(obs_v1, critic_v1, noise=False)`：噪声在评测里强制关；环境里噪声若是开的，脚本直接报错。
  - 读 checkpoint 前先把 MAPPO 尺寸改成 159 / 585，再检查形状：Table 是 `[2, 159, 585, 29]`，Push 是 `[2, 154 + 5, 585, 29]`。
  - 只加载 actor 和观测归一化器，critic 不加载（`inference = actor_mean_only_critic_forbidden`）。
  - PA 的 checkpoint 带有回报模式的元数据；脚本先把它取出来记进 summary，再做原有的元数据检查。
  - summary 里新增 `system1_t13v4`：开关、teammate 和 critic 维数、MAPPO 尺寸、环境里噪声是否开（必须是 false）；PA 另记 `return_model`。
- 测试：用第 2.2 节的 200 轮冒烟 checkpoint，确定性动作，随机化关，噪声关，各 5 个状态。结果在 `validation_obs_v1.csv`、`records/validation_summary_*.json`。

| run | checkpoint | 状态集 | 结果 |
|---|---|---|---|
| table_a | D-Table 冒烟 `model_00200.pt`（d3711471…） | T10b Table 验证状态（87a71656…），前 5 个 | complete；5 回合，步数 170 / 144 / 158 / 146 / 152，都没有走完参考 |
| table_b | 同 table_a，再跑一次 | 同 | complete；和 table_a 的轨迹逐位相同（`records/validation_determinism.json`：65 个数组，0 个不同） |
| table_pa | D-Table-PA 冒烟（062d66b0…） | 同 | complete；5 回合，步数 266–269，都没有走完参考；`return_model = per_agent_reward_value_return_advantage` |
| push | D-Push 冒烟（8bb894b6…） | 论文 Push 评测状态（只读目录 `holosoma-core4d-base/.../push_states.json`，1799ea1e…），前 5 个 | complete；5 回合都走完 309 步 |

- 这些 checkpoint 只训了 200 轮，这里只看验证能不能按观测 v1 跑通、是否确定，不看成绩。
- 现在还没有 T13 v4 的验证状态（第 8 条），所以 Table 用了 T10b 的验证状态，Push 用了论文的评测状态。
- MuJoCo 端的观测 v1 还没做（第 5 条：以后再做）。

### 2.4 按台的诊断：应接触帧里手部力 > 1 N 的比例（Kevin 2026-11-24 第 6 条）

- 代码路径已备好：参考回放检查（`isaac_checks.replay_contact`）按台输出 `hand_force_over_1n_in_should_r0/r1` 和样本数 `should_samples_r0/r1`。
- 按第 6 条，正式结果等新参考定了再跑。下面是用现有参考跑通代码路径时的数字，见 `isaac_checks.csv` 里 `diag_` 开头的列。

| 任务 | 条件 | 机器人 0 | 机器人 1 | 两台合计 | 样本数（0 / 1） |
|---|---|---|---|---|---|
| Table | 噪声关（r2、r3 两次，相同） | 7.7% | 21.0% | 15.0% | 779 / 958 |
| Table | 噪声开（两次，相同） | 7.2% | 19.4% | 13.9% | 779 / 958 |
| Push | 噪声关（r2、r3 两次，相同） | 87.3% | 86.5% | 86.9% | 489 / 489 |
| Push | 噪声开（两次，相同） | 78.9% | 80.6% | 79.8% | 489 / 489 |

- **同一条件重跑，数字逐位相同**：Table、Push 噪声关各跑了 3 次（第一轮 00:04 一次，加 r2、r3），合计比例都是 15.0% / 86.9%；噪声开各 2 次，也相同。
- **噪声开和噪声关之间的数字不同**（Push 86.9% 对 79.8%）。回放每帧都把两台机器人和物体写成参考状态，再走一个控制步；观测噪声本身不进物理。
- **我查到的：**
  - 默认配置里没有用随机数的物理随机化（只有固定的物体材质）；PD 是纯 P 控制，`kp·(目标 − 位置) − kd·速度`，没有跨帧的缓存。
  - 回放前环境的参考起始帧是重置时随机抽的。噪声开时，重置里多抽了噪声，随机数序列后移，起始帧就不同（`isaac_checks_*_h2.json` 的 `initial_time_steps`；例如 Push 噪声关 [287, 56, 219, …]，噪声开 [244, 150, 158, …]）。
  - 同一进程里把整段参考再回放一遍（`S1_T13V4_REPLAY_PASSES=2`），第二遍的数字和第一遍不同：Push 噪声关 86.9% → 85.6%，Table 噪声关 15.0% → 14.7%。
  - 也就是说，每帧写回状态之后，仿真里还有一部分状态会从前面的帧带过来，回放结果和之前的历史有关。具体是哪部分状态（例如 PhysX 的接触缓存）没有查清。
- **不受影响的：**所有这些 run 里，奖励、丢接触和终止都和 numpy 按定义的计算一致（奖励误差 ≤ 2.7e-6，丢接触、终止不一致 0）。受历史影响的只是仿真给出的力和距离本身。
- 现有参考的这几个数字只用来跑通代码；不当作对参考的评价。

## 3. 检查用的参考和标签

所有标签文件的 json 记录都在 `records/labels_*.json`。

| 用途 | 参考 | 标签 |
|---|---|---|
| Table 检查和冒烟 | 现有训练参考 `core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz`（582e7669…） | `table_current_5kgA_labels.npz` |
| Push 检查和冒烟 | 现有单人参考 `rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz`（b669468a…）；PairedA1Reference，间距 0.8780472 m | `push_current_a1_labels.npz` |
| 另外生成，备用 | T14 第 3 轮的 Table 新参考 `exports_system1/T14_references/r3/table_k1p0_a10` | `table_r3_k1p0_a10_labels.npz` |

- T14 新参考那份标签还没在仿真里用过：新参考还没有训练入口。
- **标签规则：**
  - **应接触：**原始人体数据里，手部任一顶点离物体表面 1.5 cm 以内；参考帧按最近的源帧对应。
  - **接触点：**参考里机器人的手掌点（仿真手凸包上的掌心点），投影到仿真碰撞表面，存成物体坐标系里的点。
  - **Push 的同伴：**继承源角色的标签，接触点按角色等距平移，左右手不对调。
  - **人体部位远：**离物体超过 0.1 m。
- **手掌点所在的连杆：**仿真跟踪的刚体列表里没有橡胶手（它通过固定关节挂在 wrist_yaw_link 下面），所以运行时用 wrist_yaw_link 坐标系里的手掌点；这个点由 URDF 固定关节换算得到。

## 4. 第 4 节 A 的计时

条件：D-Table 新配置，2048 个环境，本机 5070。

**基线分解**
- 50 轮，计时器在每段边界做同步，所以总时间略偏大；见 `timing_breakdown.csv`。
- 每轮 5.00 s，其中：

  | 环节 | 时间 |
  |---|---|
  | 物理（24 步 × 4 个 PD 和 PhysX 子步） | 3.31 s |
  | 奖励 | 0.31 s |
  | 重置 | 0.30 s |
  | 观测 | 0.14 s |
  | 终止 | 0.13 s |
  | 刷新张量 | 0.04 s |
  | PPO 更新 | 0.49 s |
  | 策略推理和存储 | 约 0.07 s |

- GPU 利用率 74%，进程 CPU 约 1.5 核。
- 同步扫描（一个 24 步回合）：共 11099 次同步。最多的几处：
  - `paired_motion_reference.py:236`：2400 次（每次参考采样都要在 Python 里判断一次 `torch.any`）；
  - `object_registry.py`：约 2100 次；
  - IsaacLab 传感器：约 1000 次；
  - `state_utils.py:27`：600 次；
  - 重置循环 `base_task.py:238/271`：各 281 次。
- **去掉接触三项时的对照：**20 轮，每轮 5.31 s、物理 3.57 s，没有变快。新传感器和接触项的开销在本机的波动范围以内。

## 5. 提速改动（第 4 节 A 第 3、4 条）

只改不影响结果的地方，都受 `HOLOSOMA_S1_SPEEDUP` 控制，默认开：

1. **参考采样的越界检查**（`paired_motion_reference.py`）：Python 里的 `torch.any` 换成 `torch._assert_async`，在 GPU 上断言，不同步；越界仍然报错。
2. **重置里的逐环境循环**（`base_task.py`）：先判断仿真器有没有 `on_episode_end` / `on_episode_start` 钩子，没有就不进循环。
3. **四元数换序**（`state_utils.py` 两个函数）：列表索引改成切片加 `cat`，数值相同。

**同种子 10 轮对比**（`records/compare_*.json`）
- 改动关闭两次之间：逐位相同。也就是说，本机 GPU PhysX 在同种子下是确定的。
- 关闭和开启之间：动作、奖励、done、价值、log-prob、各项损失逐位相同。

**秒 / 轮**（`timing.csv`，中位数）

| 对比 | 关闭 | 开启 |
|---|---|---|
| 第一组，50 轮（先关后开） | 4.71 | 5.04 |
| 交替运行，30 轮：第 1 对 | 5.01 | 4.70 |
| 交替运行，30 轮：第 2 对 | 4.54 | 4.53 |

- 各组之间差 −6% 到 +7%，符号不一致。本机的波动（笔记本 GPU 的温度和频率、同时在跑的 T14 CPU 任务）大于这三处改动的效果，看不出可测的提速。
- 这三处改动共去掉一个回合约 2900 次同步；物理步进占约 2/3，没有动。
- 结果逐位不变，所以保留默认开；需要时设 `HOLOSOMA_S1_SPEEDUP=0` 关闭。
- **没做的：**
  - 线程数（`OMP_NUM_THREADS` 等）没有单独测试。本机进程只用约 1.5 核，所以我判断它不是本机瓶颈，这只是判断，没有测量。按 Kevin 2026-11-24 第 7 条，放到云端测速里做（`records/cloud_speed_test_plan.md`）。
  - `object_registry` 和 IsaacLab 传感器那几处同步还在。前者改动面较大，后者在 IsaacLab 安装目录里。
- **禁止项**（环境数、步数、频率、学习率、PPO 参数、网络、混合精度）一项都没有动。

## 6. 偏离和未完成

1. **颈后点**取的是两个吊环的中点（第 1 节）。Kevin 2026-11-24：先用这个点。
2. **"自己的航向系"**用 torso_link 的偏航加航向偏差，三项统一。原来的队友项用的是 pelvis 偏航。Kevin 2026-11-24 同意。
3. **物体点的噪声**：第一轮按机器人各自独立抽。Kevin 2026-11-24 定为每个环境一份、两台共用，已改，受影响的检查已重跑（第 2.1、2.2 节）。
4. **Push 的随机化**：这是 Plan5 训练脚本第一次加 T10b 随机化。原来的"物体物理量等于冻结值"检查在随机化下必然不过，改成检查质量比在 [0.8, 1.2] 内；其余各项由随机化自带的读回检查逐项核对范围。Kevin 2026-11-24 同意。
5. **D-Push 的先验**：训练脚本读 `logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt`，本分支没有这个文件。我从只读目录 `holosoma-rubber-hand-marl/logs/` 复制了一份（sha256 11ef1fe7…，和脚本要求一致），放在本分支的 `logs/`，不进 git。第一次 Push 冒烟就是因为缺这个文件停的，第二次停在质量检查上（见第 4 条）。两次的输出都移到了 `_to_delete/`。Kevin 2026-11-24 同意复制。
6. **接触权重 w_c**：Table 是 bucket 物体块，(1 + 1) / 2 = 1.0；Push 是 (1 + 1) / 2 = 1.0。
7. **未完成：**
   - MuJoCo 端的观测 v1（任务书写明最终评测前补；Kevin 2026-11-24：以后做）；
   - 测试状态和验证状态等 T14 第 5 轮的参考定了再生成；
   - 按台的手部力诊断，正式结果等新参考定了再跑。
   - 回放诊断和之前的历史有关（第 2.4 节）。原因没有查清；参考定了以后怎么跑这一项（例如固定噪声关和种子，或换几组起始状态报范围），需要 Kevin 定。
8. **Pull** 的镜像标签（左右手对调）没有实现：本轮的 run 只有 Table 和 Push。

## 7. Kevin 2026-11-24 的八条决定和对应的处理

| # | 决定 | 处理 |
|---|---|---|
| 1 | 颈后点先用 torso_link (0.003, −0.0005, 0.290) m | 不改；只在 `NECK_POINT_TORSO_M` 一处 |
| 2 | 三项都用躯干偏航加航向偏差 | 不改 |
| 3 | 物体点噪声每个环境一份，两台相同；颈后点和航向偏差仍各自独立 | 已改 `ObsV1State`（物体点偏移和每步噪声形状 [环境, 1, 3]）；单元测试加了形状和两台相同的断言；重跑仿真内噪声开的检查（Table、Push） |
| 4 | Push 随机化检查的改法、复制先验到 `logs/` | 不改 |
| 5 | 上云前，训练中每 2000 轮的 Isaac 验证必须用观测 v1（噪声关） | 接进评测脚本并测试（第 2.3 节） |
| 6 | 新参考定了以后，回放检查加按台的"应接触帧里手部力 > 1 N 的比例" | 代码路径已备好，用现有参考跑通（第 2.4 节）；正式结果等参考定了。跑通时发现这一比例和回放前的仿真状态有关（同条件重跑相同，换条件或再回放一遍会变几个百分点） |
| 7 | 提速不再做本机的工作；云端提速测试计划里加"线程数默认 vs 实例 vCPU 数"各一次 | 只写了计划：`system1/t13v4/cloud_speed_test_plan.md`（副本 `records/cloud_speed_test_plan.md`）；没有开云。两个训练脚本加了 `--torch-threads N`（不加时不变），run_config 记 `torch_threads`（请求值、实际值、三个线程环境变量）。开关测试（2 轮、256 个环境，`records/torch_threads_smoke.json`）：Table 加 `--torch-threads 4`，实际 4；Push 不加，实际 16（PyTorch 默认；本机 nproc 32，环境里 `OPENBLAS_NUM_THREADS=32`）。两个都跑完 |
| 8 | 测试和验证状态等 T14 第 5 轮 | 没有生成 |
