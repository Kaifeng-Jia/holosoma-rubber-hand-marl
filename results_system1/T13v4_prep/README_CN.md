# T13 第四版准备：第 2 节新代码和检查、第 4 节 A 本机提速

- **依据：**`docs/system1/T13_parallel_runs_zh.md`（sha256 前 16 位 `043ac2905badc51a`），Kevin 2026-11-23：第 2 节的新代码和检查、第 4 节 A 现在就做；测试和验证状态等训练参考定了再生成；不开云。
- **时间：**2026-11-23 23:50 – 2026-11-24 01:50（纽约时间）。本机 RTX 5070 Laptop，环境 `hssim_s1`。没有开云。
- **训练参考还没定：**下面的检查用的是现有参考（见第 3 节），不是 T14 的新参考。
- **开关：**新功能全部加了开关，默认关闭；关闭时和原来的实现相同。唯一例外是提速改动，见第 5 节。

## 1. 实现了什么（开关和文件）

| 内容 | 开关 | 文件 |
|---|---|---|
| 观测 v1（actor 154 + 5 = 159） | `--obs-v1` | `holosoma/managers/observation/terms/marl_v1.py`、`config_values/marl/g1/observation_v1.py` |
| 噪声 v1（训练开；验证、评测和检查用 `--obs-v1-noise off`） | 同上，加 `--obs-v1-noise on/off` | 同上；数值都放在 `marl_v1.py` 的 `NOISE_V1`、`NECK_POINT_TORSO_M` 和截断常数这一处 |
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
- 网格里颈后正中没有单独的"把手"分量，所以这一点是我的解读，请 Kevin 确认。
- 改这个点只需要改 `NECK_POINT_TORSO_M` 这一处。

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
- 噪声分布（4096 个环境）：每回合偏移和每步噪声都在设定范围内，标准差等于均匀分布的理论值。偏移只在回合重置时重抽，两台之间独立。

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
| 观测 v1 和手算，噪声开 | 6.8e-7，噪声都在范围内 | 4.7e-7，噪声都在范围内 | — |
| 参考回放 | 687 帧：奖励误差 2.6e-6；丢接触、终止逐帧一致 | 309 帧：奖励误差 2.7e-6；逐帧一致 | 687 帧：奖励误差 4.1e-6；逐帧一致 |
| 两个传感器测到的手部力 | 差 0.0 N | 差 0.0 N | — |
| 逐台项的均值和团队项 | — | — | 9 项都 0.0；物体项共享 |

**回放里的数据（只记录）**
- 现有 Table 参考：应接触的帧里，手部力 > 1 N 的只占 15%，丢接触 85%，有 309 帧（按环境计）会触发终止。原因是旧参考的手不压桌面。
- 现有 Push 参考：手部力 > 1 N 占 87%，丢接触 13%，没有触发终止。
- 两个任务都没有触发不该碰部位的惩罚。

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
  - 线程数（`OMP_NUM_THREADS` 等）没有单独测试。本机进程只用约 1.5 核，所以我判断它不是本机瓶颈，这只是判断，没有测量。
  - `object_registry` 和 IsaacLab 传感器那几处同步还在。前者改动面较大，后者在 IsaacLab 安装目录里。
- **禁止项**（环境数、步数、频率、学习率、PPO 参数、网络、混合精度）一项都没有动。

## 6. 偏离、未完成和需要 Kevin 定的

1. **颈后点**取的是两个吊环的中点（第 1 节），请确认。
2. **"自己的航向系"**用 torso_link 的偏航加航向偏差。原来的队友项用的是 pelvis 偏航；参考朝向一项按任务书用躯干偏航，三项统一成躯干。
3. **物体点的噪声**按机器人各自独立抽。任务书只对颈后点和航向偏差写了"两台各自独立"，物体点没写，这是我的选择。
4. **Push 的随机化**：这是 Plan5 训练脚本第一次加 T10b 随机化。原来的"物体物理量等于冻结值"检查在随机化下必然不过，改成检查质量比在 [0.8, 1.2] 内；其余各项由随机化自带的读回检查逐项核对范围。
5. **D-Push 的先验**：训练脚本读 `logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt`，本分支没有这个文件。我从只读目录 `holosoma-rubber-hand-marl/logs/` 复制了一份（sha256 11ef1fe7…，和脚本要求一致），放在本分支的 `logs/`，不进 git。第一次 Push 冒烟就是因为缺这个文件停的，第二次停在质量检查上（见第 4 条）。两次的输出都移到了 `_to_delete/`。
6. **接触权重 w_c**：Table 是 bucket 物体块，(1 + 1) / 2 = 1.0；Push 是 (1 + 1) / 2 = 1.0。
7. **未完成：**
   - MuJoCo 端的观测 v1（任务书写明最终评测前补）；
   - 训练中每 2000 轮的 Isaac 验证要用观测 v1、噪声关，评测脚本还没接上；
   - 测试状态和验证状态等参考定了再生成。
8. **Pull** 的镜像标签（左右手对调）没有实现：本轮的 run 只有 Table 和 Push。
