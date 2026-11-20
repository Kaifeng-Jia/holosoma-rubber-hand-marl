# T10：MuJoCo sim2sim（C-in 和 D，Table 和 Bin）

依据：`docs/system1/T10_sim2sim_mujoco_zh.md`（sha256 `211bf1f7…4fe4c71955`），`docs/system1/plan_rss2027_zh.md`。2026-11-20 02:00 开始，约 03:10 结束，用时约 1 小时 10 分钟（纽约时间）。只用已有的最终 checkpoint，不训练，不改训练和评测设置。只用本机：MuJoCo 和 ONNX 推理在 CPU 上，Isaac 重跑和视频渲染用 5070。没有开云端。

## 结果（第 3 步，闭环评测）

- **规模：** 36 组 × 51 回合，共 1,836 回合，全部 rc 0、complete。
  - 2026-11-20 02:37–03:03 在 CPU 上跑，16 个单线程进程并行，每组 52–1,141 s。
- **协议：** 和 `results_system1/T03/d_final_5070/` 相同，具体为：
  - 冻结状态文件和 sha256 相同；
  - 动作取确定性均值；
  - 时序相同：第 k 步的观测 → 动作 → 4 个子步 → 用参考第 k 帧做终止判定 → 记录 → 参考推进到 k+1；
  - 终止条件相同（躯干重力 z 差 > 0.8、物体位置误差 > 0.25 m、物体朝向误差 > 0.8 rad）；
  - 完成 = 走满全部参考帧且没有触发 `joint_bad_tracking`；
  - 指标函数相同；
  - 只统计 50 个扰动回合，nominal 单独列。
- **checkpoint：** 和 Isaac（5070）最终评测相同，sha256 逐组核对，见 `isaac_vs_mujoco.csv` 的 `same_checkpoint` 列。
- **不调参：** 没有改任何参数去提高 MuJoCo 的结果。
- **Pull 不做**（任务书）。

| 任务 | 方案 | 低层设置 | 档位 | Isaac（5070）完成数，种子 721 / 722 / 723 | MuJoCo 完成数，种子 721 / 722 / 723 | 合计 Isaac → MuJoCo（150 回合） |
|---|---|---|---|---|---|---|
| Bin | C-in | (i) Isaac 训练时的 PD | ±2 | 49 / 43 / 43 | 0 / 0 / 0 | 135 → 0 |
| Bin | C-in | (i) Isaac 训练时的 PD | ±8 | 30 / 28 / 29 | 0 / 0 / 0 | 87 → 0 |
| Bin | C-in | (ii) 官方部署 PD + 官方力矩上限 | ±2 | 49 / 43 / 43 | 0 / 0 / 0 | 135 → 0 |
| Bin | C-in | (ii) 官方部署 PD + 官方力矩上限 | ±8 | 30 / 28 / 29 | 0 / 0 / 0 | 87 → 0 |
| Bin | D | 论文设置 | ±2 | 50 / 32 / 45 | 47 / 24 / 47 | 127 → 118 |
| Bin | D | 论文设置 | ±8 | 39 / 23 / 31 | 24 / 19 / 29 | 93 → 72 |
| Table | C-in | (i) Isaac 训练时的 PD | ±2 | 31 / 48 / 33 | 8 / 18 / 15 | 112 → 41 |
| Table | C-in | (i) Isaac 训练时的 PD | ±8 | 24 / 41 / 38 | 3 / 20 / 18 | 103 → 41 |
| Table | C-in | (ii) 官方部署 PD + 官方力矩上限 | ±2 | 31 / 48 / 33 | 10 / 17 / 14 | 112 → 41 |
| Table | C-in | (ii) 官方部署 PD + 官方力矩上限 | ±8 | 24 / 41 / 38 | 4 / 19 / 17 | 103 → 40 |
| Table | D | 论文设置 | ±2 | 35 / 49 / 46 | 0 / 0 / 0 | 130 → 0 |
| Table | D | 论文设置 | ±8 | 45 / 37 / 40 | 0 / 3 / 0 | 122 → 3 |

**nominal 回合（单独列，不计入完成率）：**

| 任务 | 方案 | 低层设置 | Isaac nominal 完成（种子 721 / 722 / 723） | MuJoCo nominal 完成 |
|---|---|---|---|---|
| Bin | C-in | (i) Isaac 训练时的 PD | True / True / True | False / False / False |
| Bin | C-in | (ii) 官方部署 PD + 官方力矩上限 | True / True / True | False / False / False |
| Bin | D | 论文设置 | True / False / False | True / True / True |
| Table | C-in | (i) Isaac 训练时的 PD | True / False / False | False / True / True |
| Table | C-in | (ii) 官方部署 PD + 官方力矩上限 | True / False / False | False / True / True |
| Table | D | 论文设置 | True / True / True | False / False / False |

**失败的情况（`per_seed_summary.csv`、`episodes.csv`）：**

- 1,466 个没完成的扰动回合里，绝大多数在终止时物体位置误差 > 0.25 m。
- 躯干倾斜（重力 z 差 > 0.8）终止的只有 8 个，最早在第 79 步。没有出现"一开始就摔倒"，第 3 步的停止条件没有触发。
- **早期失败（< 50 步）只出现在 Bin：**

  | 方案 | 档位 | MuJoCo | Isaac |
  |---|---|---|---|
  | C-in（设置 i） | ±8 | 45 / 150 | 23 / 150 |
  | D | ±8 | 23 / 150 | 8 / 150 |
  | C-in | ±2 | 2 / 150 | 0 / 150 |
  | D | ±2 | 0 / 150 | 0 / 150 |

- **C-in Bin**（两种低层设置都是 0 完成）：±2 平均执行 298–321 步，参考共 497 步。
  - 以种子 721 nominal 为例：MuJoCo 中物体相对参考的误差和 Isaac 一直相近，到第 275 步两边都已把桶抬到约 0.65 m。
  - 第 298 步 MuJoCo 里桶的高度降到 0.487 m、朝向误差 0.85 rad，回合终止；同一步 Isaac 里桶高 0.663 m。
- **D Table**（论文设置，0 完成，只有 ±8 种子 722 完成 3 个）：平均执行 352–437 步，参考共 687 步。
  - 以种子 721 nominal 为例：到第 200 步，MuJoCo 和 Isaac 的物体误差都是 0.028 m，机器人 0 的根高度相差 0.005 m。
  - 第 250 步起分开：参考把桌子抬到 0.334 m，MuJoCo 里桌子高 0.272 m，Isaac 里 0.282 m。
  - 第 434 步 MuJoCo 物体误差 0.252 m，回合终止；Isaac 在第 430 步是 0.130 m，走完了全程。
- **C-in 的两种低层设置**（(i) Isaac 训练时的 PD；(ii) 官方部署 PD 加官方力矩上限）：结果几乎相同，Table 合计 41 → 41 和 41 → 40，Bin 都是 0。两者的差别只在左右髋 roll 的力矩上限（139 / 88 Nm），PD 增益相同（见 `settings_compare.md`）。
- **视频只留本地：**
  - `exports_system1/t10/videos/mujoco_cin_bucket_seed721_nominal.mp4`（MuJoCo，298 步终止）；
  - `isaac_cin_bucket_seed721_nominal.mp4`（Isaac，完成）；
  - `mujoco_cin_bucket_seed721_nominal_keyframes.png`；
  - 均用 U5 渲染流水线。


## 第 0 步：调研

- **官方 sim2sim 的进程结构：**
  - `gear_sonic/scripts/run_sim_loop.py` 起一个 MuJoCo 仿真进程，经 Unitree SDK 的 DDS 通道模拟真机（LowState / LowCmd）。仿真 200 Hz，每步算 PD 力矩并按电机上限截断。
  - `gear_sonic_deploy/deploy.sh sim|real` 起 C++ 部署 `g1_deploy_onnx_ref`，有 4 个线程：输入 100 Hz、控制 50 Hz（编码器和解码器 ONNX）、planner 10 Hz、电机指令 500 Hz。sim 和 real 只差网卡。
- **输入接口：** 运动文件、键盘、手柄、ZMQ、ROS2。
  - ZMQ 协议 v1 可以逐帧接收整身参考：29 关节位置和速度（IsaacLab 顺序）、根四元数、全局帧号。部署端的 `StreamedMotionMerger` 从新一段的第一帧起覆盖旧数据，再从缓冲区里按"10 帧、每隔 5 帧"组成编码器窗口，这和我们 Isaac 给 SONIC 的窗口相同。
  - 协议 v4 可以直接发 64 维 token，绕过编码器。
  - **结论：** 结构上一致，C-in 不需要改训练就能接上。第 0 步的停止条件没有触发。
- **PD、力矩上限、频率、关节顺序、默认姿态、手、动作缩放、观测坐标系：** 逐项对照见 `settings_compare.md`，逐关节数值见 `settings_compare_joints.csv`，均由脚本从文件读出。要点：
  - C-in 在 Isaac 里用的 PD 增益与官方部署的公式值相同（最大差 1e-6），也与 SONIC 训练环境相同。
  - 力矩上限只有左右髋 roll 不同：我们 139 Nm，官方 sim2sim 电机上限 88 Nm。
  - 部署端的默认姿态是取到三位小数的值，与 SONIC 训练环境最大差 0.01 rad。
  - 官方 MJCF 统一设关节 armature 0.01、damping 0.05、frictionloss 0.2，Isaac 是逐关节电枢、阻尼 0、摩擦 0。
  - 官方 sim2sim 用 Dex3 手（43 自由度场景），我们用橡胶手。
- **holosoma 自带的 MuJoCo 后端**（`src/holosoma/holosoma/simulator/mujoco/`）只支持一台机器人：`prepare_sim` 里 `robot_count=1`，物体是 TODO，访问时报"Only 'robot' is available"。承载不了两台 G1 加一个物体，所以按任务书单独写了最小的 MuJoCo 场景和评测循环（`system1/sim2sim/`）。
- **SONIC 权重：**
  - 我们 Isaac 里用的是 PyTorch checkpoint `sonic_release/last.pt`（e6bdab3f…），不是 ONNX。
  - 官方部署的 `model_encoder.onnx` / `model_decoder.onnx` 本机原来没有。这次从同一个 HF 仓库 `nvidia/GEAR-SONIC` 下载：不是门控仓库，无需新的同意；与 last.pt 同一份 NVIDIA Open Model License；放在本机 `third_party/GR00T-WholeBodyControl/sonic_hf_release/`，不进 git。
  - sha256：编码器 013ab028…，解码器 c7241a12…。HF 的 `config.json` 写明这两个文件的训练文件就是 `sonic_release/last.pt`。
  - 由于 Isaac 用的不是 ONNX，"按 sha256 核对同一文件"做不到，改为数值核对（见第 2 步）。

## 第 1 步：MuJoCo 场景

- **Isaac 导出。** 为了让 MuJoCo 和 Isaac 用同一套物理参数，在评测脚本里加了开关 `--sim2sim-dump`，关闭时评测不变。打开时，从和最终评测完全相同的场景里导出 PhysX 实际用的：
  - 碰撞形状，以及 PhysX cook 出来的凸体：机器人网格连杆各 1 个凸包、约 34 个顶点；Table 5 个盒子；Bin 凸分解 16 块；
  - 每个刚体的质量、质心、惯量；
  - 关节限位、电枢、摩擦、最大速度；
  - 材质和组合方式；
  - 求解器设置；
  - PD 常数。
  
  用最终评测的原始 argv 加 `--limit 1` 重跑了 D 和 C-in、Table 和 Bin、种子 721 的 nominal 回合。重跑轨迹和最终评测保存的轨迹逐元素相同（最大差 0.0）。
- **场景（`system1/sim2sim/mj_env.py`）。**
  - 两台 G1 来自训练用的 URDF（`main_mesh_collision_rubberhand.urdf`）。每个 Isaac 刚体的质量、质心、完整惯量取 PhysX 的值；Isaac 合并掉的固定连杆质量为 0（已含在父刚体里）。整机质量和 Isaac 相等（33.3431 kg）。
  - 关节的限位和电枢取 PhysX 值，阻尼和摩擦为 0（与 Isaac 相同）。每个关节一个力矩电机。
  - 碰撞几何就是 Isaac 的碰撞形状：胶囊、球、cook 出来的凸体。
  - 物体：Table 5 个盒子、5 kg；Bin 16 个凸块、1 kg；质心和惯量取 PhysX 值。
  - 摩擦按 PhysX 的组合结果设置：机器人对地面 1.0，物体对地面 0.5，机器人对物体 0.75（PhysX 组合方式：地面 multiply，其余 average）。恢复系数 0。
  - 仿真 200 Hz（0.005 s），策略 50 Hz，每个控制步 4 个子步，每个子步按 holosoma 的公式算 PD 力矩并截断。
  - Isaac 没有对应项的设置（积分器、摩擦锥、求解器迭代）用 MuJoCo 默认值，和官方 sim2sim 场景一样。
- **初始状态检查（`consistency/initial_states.csv`、`initial_states_summary.json`）。**
  - 范围：Table、Bin 的 ±2、±8 冻结状态文件，与 T03 相同，sha256 已核对，共 204 个状态。
  - 做法：每个状态按评测脚本的方式重置（参考第 0 帧，再对每台机器人的根做 `yaw_transform`），然后和 Isaac 评测时重置后的实际状态逐项比较。
  - 结果：关节角最大差 0.0，根位置 1.2e-7 m，根朝向 7.9e-8 rad，物体位姿 0.0，各项速度 ≤ 6e-8。全部在阈值（1e-4）内。

## 第 2 步：观测和动作的一致性

### 1. ONNX（`consistency/onnx_check.csv`）

- **我们的策略：** 12 个最终 checkpoint（D 和 C-in × Table、Bin × 3 个种子），连同观测归一化导出为一个 ONNX。在 1,000 个随机观测上和 PyTorch 逐元素比较：
  - 一半输入取 N(0, 1)，另一半按各自归一化的均值和标准差采样。
  - C-in 的 6 个：最大差 1.2e-6 – 1.9e-6，**通过**。
  - D 的 6 个：最大差 7.2e-6 – 1.7e-5。**按字面阈值 1e-5，有 4 个没过**（D Table 721、723，D Bin 721、723）。超出的部分全部来自 N(0, 1) 那一半，那一半里 D 的动作幅值最大到 42。按归一化分布采样的那一半全部 < 1e-5（最大 4.8e-6）。
  - **float64 对照：** 同一计算图转成 float64 导出，和 PyTorch 的 float64 比较，12 个最大差都 ≤ 4.3e-14，转换本身完全一致。另外，PyTorch 自己 float32 和 float64 之间的差，D 最大 1.8e-5，与上面 ONNX 的差同一量级。
  - **在真实观测上：** Isaac 录到的逐步观测送进 ONNX，和 Isaac 在 GPU 上算出的动作比较（`observations_summary.json`）：D Table 3.3e-6，C-in Table 1.4e-6，D Bin 1.7e-5，C-in Bin 1.7e-6。
- **SONIC：** 官方 ONNX（编码器 + 解码器）对 `last.pt`。编码器输入按部署端的布局：4 维模式字段 [0, 0, 0, 0]，加我们的 tokenizer 去掉 3 维 one-hot 后的部分。
  - 1,000 个随机输入：token 逐元素相同（最大差 0.0）；动作最大差 1.1e-5，**按字面阈值没过**。PyTorch float32 对 float64 是 8.6e-6，ONNX 对 PyTorch float64 是 8.1e-6。
  - Isaac 录到的真实输入（Table 1,374 个、Bin 994 个）：token 全部相同；ONNX 对 Isaac GPU 的动作最大差 7e-6（Table）和 1.1e-5（Bin）。
- **我的判断：** 没过的部分是 float32 舍入，转换没有问题，所以没有停下，继续往下做。这一条和任务书的字面阈值不一致，你如果不认可，可以叫停。

### 2. 观测构造（`consistency/observations_per_dim.csv`、`observations_summary.json`）

- **做法：** 对 Isaac 录到的 4 个 nominal 回合（D、C-in × Table、Bin，种子 721），每一步把 Isaac 的状态写进 MuJoCo，用 MuJoCo 端的代码构造 158 维观测，和 Isaac 记录的观测逐维比较。
- **结果：** 4 个回合 158 维全部通过，最大差 1.35e-6，主要是参考朝向和角速度的 float32 舍入。阈值 1e-4。
  - MuJoCo 正运动学算出的 32 个刚体位置和 Isaac 差 ≤ 9.2e-7 m。朝向差 ≤ 1.5e-3 rad，这是 float32 四元数经 arccos 放大后的舍入。
- **C-in 另外核对 SONIC 的两路输入：**
  - 930 维本体感觉历史最大差 7.2e-7；
  - 1761 维 tokenizer 最大差 2.7e-7；
  - 通过。
- **过程中发现并改正的两处（都在 MuJoCo 端或核对脚本，Isaac 未改）：**
  - holosoma 观测管理器按项名的**字母顺序**拼接（`managers/observation/manager.py:141`），不是配置里写的顺序。MuJoCo 端已按字母顺序拼接。
  - 回合结束后环境重置会把最后一步的 `policy_action` 清零，所以录到的值是 0。核对时 C-in 的 a_used 改为直接取录到的动作截断后的值，这和 `scheme_c` 的定义相同。

### 3. 开环回放（`consistency/openloop_1s.csv`、`openloop_1s.png`、`openloop_1s_summary.json`）

- **做法：** 从同一个 nominal 初始状态出发，MuJoCo 逐步执行 Isaac 录到的 PD 目标，跑前 1 秒（50 步）。只报告，没有调任何参数。
- **1 秒时的偏差：**

| 回合 | 根位置（两台机器人） | 关节角最大 / 平均 | 物体位置 |
|---|---|---|---|
| D Table | 5.2 / 2.8 cm | 0.093 / 0.011 rad | 0.05 cm |
| C-in Table | 7.0 / 11.6 cm | 0.148 / 0.022 rad | 1.8 cm |
| D Bin | 14.7 / 18.0 cm | 0.50 / 0.073 rad | 5.1 cm |
| C-in Bin | 9.8 / 19.9 cm | 0.53 / 0.069 rad | 15.1 cm |

- 第 1 个控制步的第 0 个子步，两边力矩完全相同；分歧从第一个物理子步之后开始。
- MuJoCo 在初始状态就有自碰撞穿透：手臂凸包和躯干最深 3.5 cm，手和髋 2–3 cm。Isaac 第 0 步里同一个关节有一个大幅修正，说明 Isaac 里也存在这些穿透。两个仿真器消除穿透的方式不同：PhysX 最大推开速度 1 m/s，MuJoCo 是软接触。

## 第 4 步

`deploy_interface_zh.md`：每台机器人的进程和接口、需要的外部信息、C-in 接官方部署的两条路线（ZMQ v1 / v4）和还差的地方、要和博士生确认的问题清单。

## 文件

| 文件 | 内容 |
|---|---|
| `settings_compare.md`、`settings_compare_joints.csv`、`settings_compare_checks.json` | 第 0 步对照 |
| `deploy_interface_zh.md` | 第 4 步 |
| `consistency/` | 第 1、2 步的全部核对（CSV、JSON、PNG） |
| `episodes.csv` | MuJoCo 每回合一行（36 组 × 51 回合） |
| `per_seed_summary.csv` | 每组的完成数和终止原因计数 |
| `isaac_vs_mujoco.csv` | 每组 Isaac（5070）和 MuJoCo 的完成数并排，以及差几个回合 |
| `configs/` | MuJoCo 每次评测的 run_manifest 和启动、退出记录；Isaac 导出的启动、退出记录和 `scene.json`（gzip）；编译后的两个场景 MJCF（`mjcf_*.xml`） |
| `provenance.json` | checkpoint、ONNX、URDF、MJCF、状态文件、Isaac 导出的 sha256 |
| `SHA256SUMS` | |

- 只留本地、不进 git 的：
  - Isaac 导出的 `collision_convexes.npz`（5.3 MB，超过单文件上限）和逐步录制，在 `logs_system1/t10/isaac/`；
  - MuJoCo 每个回合的轨迹，在 `logs_system1/t10/mj_eval/`；
  - 策略 ONNX，在 `exports_system1/t10/onnx/`；
  - 视频，在 `exports_system1/t10/videos/`。
- 上面这些文件的 sha256 都写在 `provenance.json` 里。

## 命令和代码版本

- **Isaac 导出**（hssim_s1，5070）：`python3 system1/sim2sim/run_isaac_dumps.py`。
  - 运行时 HEAD 1541db2 加未提交的改动，改动全文存在各 `launch.json` 的 `git_diff` 里；这些改动随后原样提交为 43217ef。
- **SONIC 参考输出**（hssim_s1，CPU）：
  - `system1/sim2sim/sonic_torch_ref.py --out logs_system1/t10/onnx/sonic_torch_ref_random.npz`
  - `system1/sim2sim/sonic_torch_ref.py --n 1 --recorded logs_system1/t10/isaac/cin_table/dump/nominal.sim2sim.npz,logs_system1/t10/isaac/cin_bucket/dump/nominal.sim2sim.npz --out logs_system1/t10/onnx/sonic_torch_ref_recorded.npz`
- **核对**（sim2sim_s1，CPU，代码 43217ef）：
  - `system1/sim2sim/onnx_check.py --out results_system1/T10_sim2sim/consistency --onnx-dir exports_system1/t10/onnx_final --sonic-ref logs_system1/t10/onnx/sonic_torch_ref_random.npz,logs_system1/t10/onnx/sonic_torch_ref_recorded.npz`
    - 重新导出的 24 个 ONNX 与评测用的 `exports_system1/t10/onnx/` 逐字节相同。
  - `system1/sim2sim/check_initial_states.py --out results_system1/T10_sim2sim/consistency`
  - `system1/sim2sim/check_observations.py --out results_system1/T10_sim2sim/consistency`
  - `system1/sim2sim/check_openloop.py --out results_system1/T10_sim2sim/consistency`
- **闭环：** `python3 system1/sim2sim/run_mj_evals.py --out logs_system1/t10/mj_eval --jobs 16`
  - 代码 43217ef，工作区干净（各 `launch.json` 的 `git_status` 为空）。
- **返包：** `system1/sim2sim/package_t10.py --evals logs_system1/t10/mj_eval --out results_system1/T10_sim2sim` 和 `system1/sim2sim/settings_compare.py --out results_system1/T10_sim2sim`，代码 e20b771。
- **冒烟测试：** `PYTHONPATH=src/holosoma sim2sim_s1/bin/python -B -m pytest -q system1/tests/test_sim2sim.py`，4 项通过。
- **环境：**
  - 新建 conda 环境 `sim2sim_s1`：python 3.11，mujoco 3.3.7，onnxruntime 1.23.2，onnx 1.19.1，torch 2.7.0 CPU，以及 holosoma 需要的包；版本见 `configs/pip_freeze_sim2sim_s1.txt`。
  - hssim、hsretargeting 没有改动。


## 和任务书的偏离

1. **第 2 步第 1 项，ONNX 阈值。** 4 个 D 策略和 SONIC 解码器在 1,000 个随机输入上，按字面阈值 1e-5 没过（最大 1.7e-5）。float64 对照证明转换完全一致；真实观测上 C-in 全部 < 1e-5，D Bin 1.7e-5。我判断这是 float32 舍入，没有停，继续往下做。
2. **SONIC 的"同一文件、按 sha256 核对"。** Isaac 用的是 `last.pt`，不是 ONNX，做不到。改为数值核对：官方 ONNX 的 token 与 `last.pt` 逐元素相同，动作差在 float32 舍入范围。官方 ONNX 是这次从 HF 下载的。
3. **场景没有用 holosoma 的 MuJoCo 后端**，因为它不支持两台机器人加物体。另写了最小场景，这是任务书允许的。
4. **碰撞几何没有直接用 URDF 网格**，用的是 PhysX cook 出来的凸体，这样和 Isaac 一致。
5. **以下 Isaac 设置在 MuJoCo 里没有对应项，未建模：**
   - 物体刚体的线性和角速度阻尼 0.01；
   - PhysX 的关节最大速度（32 / 20 / 37 rad/s）；
   - TGS 求解器的迭代次数；
   - 最大穿透恢复速度 1 m/s。
6. **评测脚本 `scripts/evaluate_paper_supplement.py` 加了开关 `--sim2sim-dump`**，关闭时不变。第 2 步用的是带这个开关重跑的 Isaac 回合，重跑与已保存的轨迹逐元素相同。
