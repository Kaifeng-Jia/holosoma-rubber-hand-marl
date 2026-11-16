# T08 幻灯片备用素材 D7、D8

依据：`docs/system1/T08_slides_assets_zh.md`（sha256 `2c562371…5ada5610`）。Kevin 2026-11-15 转交时同意：本目录的 GIF 作为例外提交进仓库，每个不超过 5 MB。

只渲染已有的物理仿真轨迹，没有训练，没有重跑仿真。只用了本机 RTX 5070（EGL 渲染）和 CPU。2026-11-15 20:46（纽约时间）生成。

## 文件一览

| 文件 | 内容 | 大小 |
|---|---|---|
| `d7_right_end_attempt00.gif` / `_mid.png` / `.json` | D7 右端，回合 0 | GIF 2.45 MB |
| `d7_left_end_attempt00.gif` / `_mid.png` / `.json` | D7 左端，回合 0 | GIF 1.36 MB |
| `d8_pull_schemeA_perturbed_000.gif` / `_mid.png` / `.json` | D8 方案 A，Pull ±2 `perturbed_000` | GIF 1.08 MB |
| `d8_pull_A_pelvis_vs_reference.csv` / `.json` | D8 逐步骨盆与参考，A 和 D（种子 721） | 820 行 |
| `d7_sources.json` | D7 原始记录的出处和 sha256 | |
| `d7_attempts.csv` | D7 两端各 20 个回合的原始逐回合结果 | 40 行 |
| `d7_initial_states.json` | D7 两端各 20 个回合开始时保存的状态 | |
| `SHA256SUMS` | 本目录全部文件 | |

- 每个 GIF 和 PNG 旁边的 `.json` 是完整记录，包括：
  - 所用的轨迹行号、相机参数、编码命令；
  - 输入文件的 sha256；
  - 渲染助手和代码的 sha256，以及完整命令。

## 通用规格（两组相同）

- **渲染：** 沿用 U5_media 的流水线，复用本分支 `system1/render/u5/` 里已有的拷贝（同一个 `figround_common.py` 助手和 `g1_native_execution` 场景）。
  - Unitree 原色，浅蓝物体（物体 URDF 自带的材质 0.7 0.8 0.9），棋盘地面；
  - 方位角 135°，俯角 −22°，视野 45°，1920×1080。
  - 只按保存的状态摆姿态：不走物理步，不跑策略，不插值。
- **GIF：**
  - 1920×1080 渲染后缩到 960×540（Lanczos），20 帧/秒，循环播放；
  - 第 k 帧显示保存的第 ⌊k × 倍速 × 50 / 20⌋ 行，最后一行一定在；
  - ffmpeg 生成调色板（256 色，bayer 抖动）。
- **播放倍速：** 三段回合都在 8 秒以内，全部是 1×，没有"2× speed"。
  - 不到 4 秒的回合（D7 左端、D8）把最后一帧停留到满 4 秒，停留的帧右上角标"episode ended"。
- **静帧：** 回合正中间那一行 ⌊(n − 1) / 2⌋，1920×1080，和 GIF 同一机位。

## D7：单机器人推桌策略在宽桌两端

### 原始记录（只读，全部 sha256 见 `d7_sources.json`）

这次检查是 holosoma-rubber-hand-marl 里的 Stage 1B-1 宽桌保持性检查（2026-08-15）。

**报告：**
- `STAGE1B_A1_RETENTION_REPORT_CN.md`，提交 d04b312a，之后已从工作区删除，从 git 历史读取，sha256 `53a2da15…`。
- 报告里的结果：右端 17/20，左端 1/20；左端失败时桌子被推偏，机器人没有摔倒。

**checkpoint：**
- `logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt`，sha256 `11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224`。
- 确认依据：
  - 报告写了这个文件名；
  - 2026-08-17 的提交 ad8a80cf 写明它就是当时的冻结策略；
  - sha256 与转换记录 `conversion_manifest.json` 的输出一致。
- 它由 `rubberhand_a1_8000_seed42/model_07999.pt`（sha256 `54fbfa4b…`，训练种子 42，单人示范 A1 推桌）把观测补成 158 维得到，新增 4 列的权重为 0。

**宽桌：**
- `objects_widetable_a1_retention.urdf`，取 d04b312a 的版本，sha256 `3eba7861…`，拷贝在本分支 `system1/t08/assets/`。
- 1.4 m 宽的五箱体桌，0.1 kg，沿用原 A1 桌的质量和接触参数。
- 后来改名为 `objects_widetable_plan5_preflight.urdf`，只改了名字和注释。

**机器人：** `main_mesh_collision_rubberhand.urdf`，当时和本分支的 sha256 相同（`7ed217f2…`）。

**放置方式：**
- 把 A1 参考整体沿桌子局部 X 轴平移：左端 −0.4 m，右端 +0.4 m，两机器人间距 0.8 m。
- 关节角、姿态和物体通道都不变。
- 对侧只有一个按另一条平移参考运动的 ghost（无实体、不碰撞）。
- 平移脚本 `scripts/prepare_stage1b_a1_retention.py`，取 d04b312a 版本；源动作 `sub6_largetable_033_a1_mj_fps50_w_obj.npz`，sha256 `b669468a…`。

**运行：** seed 42，单环境，Isaac Sim，每端 6500 个策略步，统计前 20 个结束的回合。完整命令行没有记录。

**录制和逐回合结果：**
- 每端的轨迹：`logs/WholeBodyTracking/plan5_preflight_v1/formal_wide_{right,left}_trajectory.npz`。
  - 原在 `stage1b_a1_retention_v1/`，提交 ae7ea70b 改了目录名；
  - 文件写于 08-15 04:00 和 04:06，在报告提交之前。
- 每端的逐回合结果：同目录下的 `.json`。
- 各项指标与报告第 6 节逐项一致，包括完成数、关节 RMSE、关键体 RMSE、物体误差、队友观测范围。
- 逐回合结果整理在 `d7_attempts.csv`。

**初始状态：**
- 两端回合 0 都从 seed 42 下的第一次环境重置开始：桌子位置相同，机器人沿桌子 X 方向相差 0.8 m。
- 其他回合从上一回合结束处接着开始：提前终止后会重置；跑完参考则从参考开头重新开始，不重置环境。
- 每个回合开始时保存的状态见 `d7_initial_states.json`。

**轨迹已有，按任务书第 3 条直接渲染，没有重跑。**

### 选回合（任务书第 2 条）

| 端 | 20 个回合完成数 | 多数结果 | 选中的回合 |
|---|---|---|---|
| 右端 | 17/20 | 完成 | 编号最小的完成回合：回合 0 |
| 左端 | 1/20（只有回合 1 完成） | 失败 | 编号最小的失败回合：回合 0 |

- 两段 GIF 和 PNG 左上角都写了这一端的总数："right end: 17/20 completed"、"left end: 1/20 completed"。

### 相机

两段完全相同：lookat (−0.0588, −0.2713, 0.6485)，距离 3.449 m，方位角 135°，俯角 −22°。

- lookat 的水平位置：两段回合里桌子包围盒的中心。
- lookat 的高度：取两段回合全部机器人和桌子网格包围盒高度范围的中点，不是桌子包围盒中心的高度（约 0.2 m）。
  - 第一次试渲染用桌子中心的高度，画面下方空出一大片地面，机器人只占很小一块。按常驻规则"机器人和物体要占画面的主要部分"，改成了现在的高度。
- 距离：用 U5 的 `camera_for` 规则（四周留 7%）装下两段回合所有帧的全部包围盒。
  - 因为要同时装下两段回合里桌子约 1.4 m 的移动和两侧不同的机器人位置，单帧画面里的机器人比 D8 小。

### 逐个文件记录

| 文件 | 任务 | 方法 | 训练种子 | checkpoint sha256 | 初始状态 | 是否完成 | 终止原因 | 执行步数 | 播放 | 物理仿真 |
|---|---|---|---|---|---|---|---|---|---|---|
| `d7_right_end_attempt00.gif`、`_mid.png` | 单机器人推宽桌，右端 | 冻结的 A1 单机器人推桌策略（单人示范，未训练过宽桌） | 42 | `11ef1fe7…3e224` | 右端回合 0（seed 42 第一次重置，录制第 0 行） | 是 | 跑完参考（reached_end），没有提前终止 | 308（6.16 s） | 1×，GIF 6.2 s，124 帧；PNG 第 153 行 | 是（Isaac Sim 录制） |
| `d7_left_end_attempt00.gif`、`_mid.png` | 单机器人推宽桌，左端 | 同上 | 42 | 同上 | 左端回合 0（seed 42 第一次重置，录制第 0 行） | 否 | 提前终止，见下 | 184（3.68 s） | 1×，最后一帧停留 5 帧到 4.0 s（80 帧）；PNG 第 91 行 | 是（Isaac Sim 录制） |

- **左端回合 0 的终止原因：** `terminated=True`，不是超时，也没有摔倒（fall proxy False）。
  - 具体是哪一项终止条件触发的，录制里没有记：录制保存的是动作前的状态，终止判定在之后的物理步，报告第 8 节也这样写明。
  - 最后一行的物体位置误差是 0.240 m，WBT 的物体位置终止阈值是 0.25 m。
- **右端回合 0：** 物体位置一直在阈值内，所以算完成；但最后一行桌子偏航比参考多转了 28°（实际 −36.7°，参考 −8.3°）。GIF 如实回放这一点。

### 和报告不一致的地方（照实记录）

- 报告第 6 节表里，两端的 fall proxy 都写 0%。录制的 JSON 里，左端是 5%（回合 4），右端是 10%（回合 5 和 10）。
- 这次渲染的两个回合（回合 0）fall proxy 都是 False。

## D8：方案 A 在 Pull 上

| 文件 | 任务 | 方法 | 训练种子 | checkpoint sha256 | 初始状态 | 是否完成 | 终止原因 | 执行步数 | 播放 | 物理仿真 |
|---|---|---|---|---|---|---|---|---|---|---|
| `d8_pull_schemeA_perturbed_000.gif`、`_mid.png` | Pull，±2 档 | 方案 A：静态目标 + SONIC，不训练，开环 | 无（确定性） | SONIC release `e6bdab3f…ded8909` | `perturbed_000`（状态文件 `473c1fc0…`，与幻灯片里 D 的 Pull 动图同一个初始状态） | 否 | `joint_bad_tracking` | 92（1.84 s） | 1×，最后一帧停留 42 帧到 4.0 s（80 帧）；PNG 第 46 行 | 是（holosoma Isaac Sim，本机 5070，T01b 主设置） |

- **轨迹：** `logs_system1/T01b/schemeA_main/pull_pm2/perturbed_000.trajectory.npz`，共 93 行（初始状态加 92 步），全部渲染。
- **相机：** 与 U5_media 里 D 的 Pull 视频（`videos/pull.json`）逐位相同：lookat (0.1004, 0.3059, 0.6344)，距离 3.477 m，方位角 135°，俯角 −22°。已用程序核对。

### CSV：`d8_pull_A_pelvis_vs_reference.csv`

- **列：** 按任务书，`scheme, task, state_id, step, time_s, reference_frame, robot, ref_pelvis_x, ref_pelvis_y, pelvis_x, pelvis_y, planar_error_m, ref_object_x, ref_object_y, object_x, object_y`。
  - 每个保存的行、每台机器人各一行；
  - `step` 是行号 j：第 0 行是初始状态，第 j 行是第 j 步之后的状态；
  - `time_s = j / 50`。
- **坐标对齐和 d：** 与 `results_system1/T01b/preregistration_drift_in_task.md` 相同，直接调用它的代码 `system1/sonic/drift_in_task.episode_metrics`。
  - 这次给它加了一个默认关闭的开关 `series`，打开时额外返回逐行坐标；关闭时输出不变。
- **A：** 93 行 × 2 台机器人。
- **D：** 种子 721，本机 5070 最终评测保存的轨迹 `logs_system1/post_train/d_curve/D_pull_paper_seed721/it08050_pm2/perturbed_000.trajectory.npz`。
  - checkpoint `727630e9…`，与 U5 Pull 视频的 checkpoint 相同；
  - 完成，316 步，保存 317 行 × 2 台机器人。
- 两种方案各自的行数照实保留，不对齐、不截断。
- **检查（任务书第 4 条）：** A 这一回合的末值和最大值，与 `results_system1/T01b/drift_in_task/episodes.csv` 同一行逐位相同，通过：

  | 机器人 | 末值（cm） | 最大值（cm） |
  |---|---|---|
  | robot 0 | 21.70777 | 21.70777 |
  | robot 1 | 10.69465 | 10.69465 |

- 完整检查记录在 `d8_pull_A_pelvis_vs_reference.json`。

## 命令和代码版本

渲染和 CSV 在提交 `6b0b9d027d46fa55d4a3b8ddbf99a878816ca8cb` 上运行，D7 来源记录在提交 `2324f656afda1fff1e5bf727629d54576a4e2e19` 上运行，运行时工作区都没有未提交的改动。

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl sonic_s1/bin/python -B system1/render/u5/render_t08.py --d7 --d8 --out results_system1/T08_slides
    hssim_s1/bin/python -B system1/t08/d8_pull_csv.py --out results_system1/T08_slides
    hssim_s1/bin/python -B system1/t08/d7_sources.py --out results_system1/T08_slides

- **冒烟测试：** `system1/tests/test_render_t08.py`，5 项全部通过，覆盖播放规则、相机公式和选回合规则。

## 和任务书的偏离

- **D7 相机高度：** 任务书写"对准宽桌中心"。水平方向对准了宽桌中心，高度改为全部包围盒高度范围的中点，理由见上。如果要严格对准桌子中心的高度，改一个参数重渲即可。
- 没有其他偏离。
