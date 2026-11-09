# T01 返包（一）：隔离、复现验证、SONIC 接入验证

截至 2026-11-09 17:05 EST。方案 A 的正式评测尚未完成，原因见第 6 节。

## 0. receipt

| 项 | 值 |
|---|---|
| 任务书 | `docs/system1/T01_isolation_sonic_spike_schemeA_zh.md`，sha256 `226e1fe572a90fdb377cec9f15b7f009e0456d51af718b66640307635df8ea7c` |
| 开始 | 2026-11-09 15:52:11 EST |
| 本目录 | `/home/kevin/duet-system1`，分支 `system1-sonic`；开工时 HEAD `40bab6d`，工作区干净（只有被忽略的 `logs_system1/`） |
| 机器 | `kevin-Legion-R9000P-ADR10`；1 × RTX 5070 Laptop 8151 MiB；驱动 580.178.04 |
| 旧目录 | 开工时和交付时的 HEAD、`git status --porcelain` 的 sha256、`git diff HEAD` 的 sha256 完全相同，见 `old_dirs_status_start.txt` / `old_dirs_status_end.txt` |
| 环境 | `hssim_s1` 和 `sonic_s1`，都由 `hssim` 克隆而来；版本和相对 `hssim` 的包差异见 `versions.json`。`hssim` 本身未改动，numpy 仍为 1.26.0，holosoma 仍指向 `/home/kevin/holosoma` |

## 1. 第 0 步：隔离与快照

- **快照：** commit `931217a`，`snapshot: holosoma-core4d-base working tree 2026-11-09T15:53:26-0500`。
  - 旧目录 HEAD `7df11c92` 与本分支起点 `6acb096` 只差一个 md 文件，代码没有差异，所以直接在本分支应用了补丁。
  - 内容：应用 `git diff HEAD`（8 个文件），再复制 70 个未跟踪代码文件（`scripts/` 59 个、`src/` 不含 `data/` 的 3 个、`tests/` 8 个）。
  - 逐文件的原路径和 sha256 见 `system1/snapshot_manifest.json`；该清单也列出了被排除的 4622 个未跟踪条目，按目录分类计数。
- **`hssim_s1`：** holosoma 改为从本目录 editable 安装（`--no-deps`）。之后为加载 SONIC 网络，又以 `--no-deps` 加装了 `gear_sonic`（editable，指向 `/home/kevin/third_party`）、`einx 0.4.3`、`frozendict 2.5.1`、`torch-einops-utils 0.2.6`、`vector_quantize_pytorch 1.32.2`。没有改动已有的包。
- **`sonic_s1`（SONIC 官方评测）：**
  - 克隆 `hssim` 后执行 `pip install -e gear_sonic[training]`。这一步把 numpy 升到 1.26.4，并装入 pyarrow 26；pyarrow 26 要求 numpy ≥ 2、无法导入，所以又改装 `pyarrow==19.0.1`。
  - Isaac Sim 5.1.0 / Isaac Lab v2.3.0（ext 0.47.2），与论文主结果的后端版本相同。SONIC 文档要求 Isaac Lab 2.3+。

## 2. 复现验证（门槛）：通过

- **评测对象：** V3 登记的 Table Pulling seed 721 主模型（`model_08050.pt`，sha256 `727630e9…`），±2 档，冻结状态 `pull_states.json`（`473c1fc0…`）。
- **命令：** 与原评测相同（`repro_cmd.sh`），只是代码改用本目录、解释器改用 `hssim_s1`。
- **结果：** 51/51 个回合与 V3 `episodes.csv` 逐项一致，包括 completed_reference、termination_reason、executed_steps、progress、object_pos/xy/height RMSE、raw_reward_sum_common、hand_pos_error_cm（按 V3 同一 FK 流程复算）。所有差值都是 0.0；轨迹数组与原评测输出逐位相同。见 `repro_pull_seed721_pm2_compare.csv/json`。
- **时间顺序：** 复现进程 15:55 启动，所有模块在启动时载入。之后在 16:05–16:10 修改了训练和评测脚本（T04 质量开关），不影响这次运行。

## 3. 第 1 步：SONIC 原生验证

- **仓库：** `/home/kevin/third_party/GR00T-WholeBodyControl`，commit `1e1ec4f5f27e6c3e03048f693bd7c60cce04afda`。只拉取了 `gear_sonic/data/assets/**` 的 LFS 文件。
- **权重：** HF `nvidia/GEAR-SONIC`，revision `6733128a`。API 显示 `gated: False`，没有需要点击同意的页面；license 为 nvidia-open-model-license。
  - `sonic_release/last.pt` 的 sha256 是 `e6bdab3f…`，与 HF LFS 记录一致。全部文件哈希见 `sonic_assets_sha256.txt`。
  - 权重不在 git 里。
- **评测：** 官方 `eval_agent_trl.py` 的 Mode 2 命令，示例动作 `walk_forward_amateur_001__A001` 及其镜像，各 2002 帧，128 env。
  - 我用 `system1/sonic/native_eval_dump.py` 以脚本方式调用官方入口，只加了只读钩子，用于导出观测、动作和状态。
  - 额外加了 `++eval_output_dir` 用于保存指标文件。命令见 `sonic_native_cmd.sh`。
- **编码器：** 官方 release config 的 `encoder_sample_probs` 对 g1/teleop/smpl 都是 1.0，每个 env 随机选编码器。所以另跑了一次只改为 g1=1、其余为 0 的版本。

| 运行 | mpjpe_g (mm) | mpjpe_l (mm) | mpjpe_pa (mm) | success_rate |
|---|---|---|---|---|
| 官方命令（混合编码器） | 86.88 | 17.06 | 10.63 | 1.0 |
| G1-only | **71.10** | 15.62 | 10.21 | 1.0 |

- **验证 a 基准：** 按 `preregistration_validation_a.md`（写于任何 SONIC 运行之前），基准取 G1-only 的 71.10 mm，门槛为 1.2 × 71.10 = 85.31 mm 且不摔倒。
- **SONIC 评测环境自带的启动随机化：** 原生评测里仍开着 startup 随机化，包括每个 env 的默认关节角加 U(−0.01, 0.01)、body 质量和 COM 随机化；训练专用的 push 事件是关闭的。

## 4. 第 2 步：模型对照（`robot_model_diff.csv`）

SONIC 一侧的数值从它的仿真运行时读回，我们一侧取 holosoma 配置和 URDF，按关节名对齐。

- **相同：**
  - 29 个关节名称集合和位置限位；
  - 标称默认姿态；
  - 除双侧 `hip_pitch` 外，其余 27 个关节的 Kp/Kd、力矩限制、速度限制、armature、动作缩放；
  - 所有非手部 link 的质量；
  - 200 Hz 物理、50 Hz 策略。
- **不同：**
  - **hip_pitch（左右）：** SONIC Kp 99.10 / Kd 6.309 / 力矩 139 / 速度 20 / armature 0.0251 / 动作缩放 0.3507；我们 40.18 / 2.558 / 88 / 32 / 0.0102 / 0.5475。
  - **关节顺序：** SONIC 用 Isaac Lab 的 BFS 交错顺序，我们用 MuJoCo 顺序。按名字映射。
  - **手：** SONIC 是 Dex3 手（palm 加 7 个手指 link，全部固定关节），每侧 0.697 kg；我们是 rubber hand，每侧 0.17 kg。URDF 总质量 34.39 vs 33.34 kg。
  - **碰撞体：** SONIC 多处用圆柱或没有碰撞体；我们用 mesh。共 32 个 link 不同。
  - **PD 实现：** SONIC 用 `ImplicitActuatorCfg`（PhysX 关节驱动）；我们用 Python 在每个 200 Hz 子步算显式 PD，再以力矩施加。

## 5. 第 3 步：接入路线 1

### 5.1 实现

- **`system1/sonic_controller.py`：** 用 gear_sonic 自己的 `custom_instantiate` 构建 release 策略并加载权重，推理时与 SONIC 评测一样开启 TF32。
  - G1 编码器的输入（未来 10 帧、间隔 5 帧的关节位置和速度，以及 pelvis 相对朝向）和 930 维本体观测（10 帧历史）由 holosoma 的状态和参考构造；
  - 支持 E × 2 批量推理；
  - 默认姿态用 SONIC 标称值。
- **离线等价性检查（`sonic_obs_equivalence_g1only.json`）：** 用 SONIC 原生运行导出的状态，逐步重建观测，共 2 个 env × 2000 步。
  - tokenizer 的 G1 部分：最大差 4.8e-7；
  - 930 维本体观测：最大差 1.0e-7。比对时用了原生 env 各自随机化后的默认角。
  - 把 SONIC 自己的观测喂给我们加载的网络，动作最大差 0.057；用重建的观测，动作最大差 0.081。原生运行是 128 env 一批，这里是 2 env 一批，差异来自 TF32 推理。
- **本体观测的实际排布：** base_ang_vel(3)、joint_pos_rel(29)、joint_vel(29)、last_action(29)、gravity_dir(3)，每项 10 帧，按时间从旧到新排列。由导出数据逐值确认（误差 0）。
- **低层控制：**
  - SONIC 的关节目标 q* = 默认角 + scale × a，通过现有动作项换算成 a_h = (q* − default)/scale_h 传入，PD 收到的是同一个 q*；
  - PD 增益用 SONIC 的：只有 hip_pitch 改为 99.10 / 6.309，经配置开关替换；
  - 机器人模型（力矩上限 88 等）保持我们的。

### 5.2 验证 a 与 b

- **设置：** holosoma 单机器人 WBT 环境、rubber-hand URDF、无物体、无随机化、从第 0 帧开始。
- **后端：** 都在 **Isaac Sim 的 CPU PhysX** 上运行，原因见第 6 节。
- **"摔倒"判据：** SONIC 评测的终止条件，即 pelvis 高度差 > 0.25 m，或 pelvis 姿态误差平方 > 1.0，或双踝/双腕任一高度差 > 0.25 m。
- **MPJPE：** 与 SONIC 定义相同：14 个 body，世界坐标，每个 post-step k 对比参考第 k 帧，取平均。

| 验证 | 参考 | 执行步数 | 摔倒 | MPJPE (mm) | pelvis 相对 MPJPE (mm) | pelvis 水平漂移，末帧/最大 (mm) | SONIC 原生 G1-only (mm) |
|---|---|---|---|---|---|---|---|
| a | A001 | 2001/2001 | 否 | 54.37 | 16.1 | 83 / 137 | 53.30 |
| a | A001_M | 2001/2001 | 否 | 89.56 | 16.2 | 144 / 197 | 88.83 |
| a | 两段平均 | | | **71.97** | | | **71.10** |
| b | Table Pulling 角色 1（agent 0） | 315/315 | 否 | 282.88 | 32.8 | 442 / 506 | — |
| b | Table Transport 角色 1（agent 0） | 686/686 | 否 | 721.05 | 28.5 | 1445 / 1450 | — |

- **验证 a：** 71.97 / 71.10 = 1.012，不超过 1.2，且两段都没有摔倒。按预登记标准判为**通过**，但这是 CPU PhysX 后端的结果（见第 6 节）。
- **验证 b 的参考：** 就是论文评测用的 runtime 参考（Pull `571f2305…`，Table `582e7669…`）。"角色 1"按 agent 索引 0 取。
- **视频：** 4 段，在 `exports_system1/T01/`，不进 git，sha256 见 `videos_sha256.json`。
  - 由记录的状态用 MuJoCo 离线回放渲染（`system1/render/render_traj.py`），不是仿真器直接录屏。
  - 蓝色是机器人，红点是机器人的跟踪 body，绿点是参考 body。

## 6. 与任务书的偏离、需要 Kevin 决定的事项

1. **GPU 冲突。**
   - 本机只有一块 8 GB 笔记本 GPU。T04-D 训练（2048 env）启动后占约 6.5 GB；一个 holosoma Isaac 评测实例约需 2.7 GB。所以训练期间无法在 GPU 上跑 SONIC 验证或方案 A。
   - T03（2048 env × 2 × SONIC 推理）和 T04-D 也不能同时跑。
   - T04-D 六个 run 串行，按实测 4.4–5.7 s/迭代估算，需要约 80 小时。
2. **验证 a/b 改在 CPU PhysX 上做。** 同一个 Isaac Sim 5.1 场景，只把 `--device` 改为 cpu。为了评估后端差异，我用 CPU 把论文 D（Pull 721 ±2）重评了一遍，与 V3 的 GPU 结果对照（`backend_check_pull_seed721_pm2_cpu_vs_gpu.*`）：
   - 完成数 50/50 对 50/50，逐回合步数相同；
   - 平均物体 RMSE 1.744 对 2.042 cm，平均手误差 3.652 对 3.780 cm；
   - 逐回合指标都有差异，物体 RMSE 最大差 5.6 cm。
3. **方案 A 的六任务评测：** 为了不在等待期间空转，已在 CPU PhysX 上运行，绑定 8–31 号核，**标记为临时**，输出到 `logs_system1/T01/schemeA_cpu/`，不作为交付结果。
   - 选项 (a)：Kevin 另提供一块 GPU（或云端）给 A/B/C 评测和 T03；
   - 选项 (b)：在 T04-D 各段之间暂停队列，在 GPU 上跑方案 A，按 GPU 评测估计约数小时；
   - 选项 (c)：接受 CPU PhysX 作为 A/B/C 的后端，同时在 CPU 上重评 D，使对比在同一后端上进行。
   - 选哪一项请 Kevin 决定。在决定之前，不把 CPU 上的方案 A 数字当作正式结果交付。
4. **验证 a 的基准** 用 G1-only 原生运行，不用官方混合编码器的数值。理由和选择已在运行前写入 `preregistration_validation_a.md`。
5. **SONIC 原生评测：** 使用官方命令的 128 env，额外加了 `eval_output_dir` 和只读导出钩子。G1-only 那次只改了 `encoder_sample_probs` 一项。

## 7. T04-D（已开训）

- **队列：** `system1/t04/run_d_queue.py`，16:33 EST 启动。顺序为 Pull 5 → Table 2.5 → Pull 10 → Table 10 → Pull 40 → Table 20 kg，seed 721。
- **配置：**
  - Pull 按 V3 Full-rerun：50 次 critic-only + 8000 次联合，episode telemetry；
  - Table 按论文 T-07：fresh 12000。
- **质量变体：** URDF 的质量和惯量同乘 m/m0。PhysX 读回值已写入 run_config。
- **Pull 5 kg 两段的 run_config 对比：** 与 V3 reference 的差异只有质量、目录和运行元数据（commit、时间、进程信息），**0 个意外差异**（`pull_m5kg_seed721_*.run_config_diff.json`）。
- **进度：** critic50 段已完成；joint8000 段在运行，17:05 时到第 299 次迭代。

## 8. 代码提交

| commit | 内容 |
|---|---|
| `931217a` | 快照 |
| `11fa126` | gitignore |
| `ce5ad9e` | 复现检查 |
| `393879a` | 质量开关和 URDF |
| `e925f01` | SONIC 控制器、验证工具、方案 A 开关 |
| `d58eca4` | 方案 A 队列修正和 Push 证据文件 |

- 所有新功能默认关闭。默认 `policy` 路径的评测脚本在 CPU 上已跑通 51 个回合；GPU 上的逐位复现复查要等 GPU 空出后再做。
- smoke test：`system1/tests/test_mass_variants.py`、`system1/tests/test_sonic_controller.py`，以及两个质量开关的 8-env 训练 smoke（`logs_system1/T04/smoke/`）。

### 用时

约 1 小时 15 分钟：15:52–17:05 EST。
