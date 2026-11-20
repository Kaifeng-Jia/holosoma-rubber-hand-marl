# T10s：其余任务的 sim2sim（Isaac → MuJoCo）返包

任务书：`docs/system1/T10s_sim2sim_other_tasks_zh.md`。本返包只写事实，不解释结论。

- **范围：**
  - D：Push、Pull、Kick、Chair，种子 721–723，论文设置；
  - C-in：Pull，种子 721–723，PD 两种设置：(i) Isaac，(ii) official。
  - 每组 ±2、±8 两档，每档 50 个扰动状态，nominal 单独记。
- **设置：** 与 T10 相同，没有调参。
- **机器：** CPU，MuJoCo 3.3.7，环境 `sim2sim_s1`。Isaac 的 dump 在本机 RTX 5070 上跑。

## 用时（America/New_York，2026-11-20）

| 步骤 | 时间 |
|---|---|
| ONNX 导出和检查 | 约 04:09 完成 |
| Isaac dump 首次运行 | 04:01–04:05 |
| Isaac dump 重跑，加 `runtime_reference.npz` | 11:10–11:16 |
| 初始状态、观测、开环检查 | 11:17–11:37 |
| MuJoCo 闭环（36 组，16 个并行） | 11:38–11:44 |
| 视频 | 12:39–12:41 |

首次的 dump 早于参考导出功能，缺少 `runtime_reference.npz`，所以重跑了一次。旧 dump 移到了 `_to_delete/t10s_dumps_without_runtime_reference_20261120/`，所有检查都用重跑后的 dump。

## 1. 一致性检查（`consistency/`）

| 检查 | 结果 | 阈值 / 规则 |
|---|---|---|
| 初始状态（4 个任务 × ±2 / ±8 冻结状态文件，共 408 个状态） | 全部通过；关节 0，根位置最大 5.95e-8 m，物体位置 0 | 1e-4 |
| 观测（D 721 nominal 每任务一个；C-in Pull 721） | 全部通过；最大差：Push 2.25e-6，Pull 9.9e-7，Kick 1.56e-6，Chair 2.43e-6，C-in Pull 9.0e-7 | 1e-4 |
| C-in Pull 的 SONIC 输入 | 通过；actor 观测 4.2e-7，tokenizer 2.4e-7；SONIC 动作 ONNX 和 Isaac 的最大差 8.1e-6 | 1e-4 |
| ONNX float64 图和 torch float64 比较（15 个 checkpoint） | 全部通过，最大 1.99e-13 | ≤ 1e-10 |
| ONNX float32 | 只报数值。Push 三个种子的 actor 输出最大差 4.6e-5–6.1e-5，超过 T10 原来的字面阈值 1e-5（`onnx_check.csv` 里 `passed=False` 指这一列）；其余 ≤ 9.5e-6 | 按 T10s 规则只报告 |

**开环 1 s**：MuJoCo 回放 Isaac 记录的动作 50 步，和 Isaac 轨迹比较，只报数值。

| 运行 | 根部位置误差 r0 / r1 (m) | 关节误差 最大 / 平均 (rad) | 物体位置误差 (m) |
|---|---|---|---|
| D Push 721 | 1.173 / 0.153 | 0.940 / 0.143 | 0.075 |
| D Pull 721 | 0.074 / 0.008 | 0.121 / 0.023 | 0.001 |
| D Kick 721 | 0.267 / 0.416 | 0.909 / 0.117 | 0.007 |
| D Chair 721 | 0.034 / 0.081 | 0.101 / 0.017 | 0.002 |
| C-in Pull 721 | 0.189 / 0.131 | 0.259 / 0.045 | 0.006 |

D Push 721 的 nominal 在 Isaac 里第 87 步就终止了。笔记本 5070 的最终评测 `d_final_5070` 里也一样：三个种子的 Push nominal 在 Isaac 里都没完成。所以 Push 的 Isaac dump 只有 87 步。

## 2. 闭环结果：完成数，三个种子合计

每档每种子 50 个扰动状态，nominal 不计入。

| 任务 | 方案 | PD | 档位 | Isaac 5070（721 / 722 / 723） | MuJoCo（721 / 722 / 723） |
|---|---|---|---|---|---|
| Push | D | 论文 | ±2 | 126/150（39 / 50 / 37） | 150/150（50 / 50 / 50） |
| Push | D | 论文 | ±8 | 118/150（31 / 49 / 38） | 150/150（50 / 50 / 50） |
| Pull | D | 论文 | ±2 | 149/150（50 / 50 / 49） | 150/150（50 / 50 / 50） |
| Pull | D | 论文 | ±8 | 112/150（43 / 28 / 41） | 101/150（39 / 29 / 33） |
| Kick | D | 论文 | ±2 | 150/150（50 / 50 / 50） | 0/150 |
| Kick | D | 论文 | ±8 | 142/150（47 / 49 / 46） | 0/150 |
| Chair | D | 论文 | ±2 | 143/150（44 / 49 / 50） | 150/150（50 / 50 / 50） |
| Chair | D | 论文 | ±8 | 129/150（33 / 46 / 50） | 147/150（48 / 50 / 49） |
| Pull | C-in | Isaac | ±2 | 118/150（50 / 50 / 18） | 0/150 |
| Pull | C-in | Isaac | ±8 | 74/150（22 / 38 / 14） | 0/150 |
| Pull | C-in | official | ±2 | 118/150（同上） | 0/150 |
| Pull | C-in | official | ±8 | 74/150（同上） | 0/150 |

- **nominal：**
  - MuJoCo 里 D Push、Pull、Chair 三个种子都完成；D Kick、C-in Pull 都没完成。
  - Isaac 里 Push 三个种子都没完成，C-in Pull 723 没完成，其余都完成。
- **Isaac 一侧的数字来源：** `results_system1/T03/d_final_5070`（D Push、Kick、Chair）和 `results_system1/T03/c2_main`（D Pull、C-in Pull）。checkpoint 和状态文件的 sha256 与 MuJoCo 一侧逐行核对，结果相同，见 `isaac_vs_mujoco.csv` 的 `same_checkpoint`。
- **合表：** T10 和 T10s 合在 `../isaac_vs_mujoco_all.csv`，共 72 行，`source` 列注明来自 T10 还是 T10s。

## 3. 失败分析（`failure_summary.csv`、`failure_episodes.csv`）

**统计口径：**
- 终止原因：取每个失败回合最后一条终止记录。
  - 第 r 台机器人躯干重力 z 误差 > 0.8，记为 `robot r tilt`；
  - 物体位置误差 > 0.25 m，记为 `object_position`；
  - 物体姿态误差 > 0.8 rad，记为 `object_orientation`。
  - 同一回合可以同时满足几条。
- 物体误差开始分开的步数：物体位置误差第一次超过 0.1 m 的那一步。
- 阶段：参考物体在那一步的运动，按 `system1/t10b/phases.py` 的规则标注：
  - 转动：角速度 > 0.5 rad/s；
  - 抬起、放下：竖直速度超过 ±0.05 m/s；
  - 举起中：高于初始高度 5 cm；
  - 贴地推行 / 拉行：平面速度 > 0.05 m/s；
  - 静止：以上都不是。

| 组 | 失败回合（含 nominal） | 终止原因 | 终止步数 | 物体误差超过 0.1 m 的步数 | 当时参考阶段 |
|---|---|---|---|---|---|
| D Kick 721 / 723（±2、±8） | 51 / 51 | 机器人 1 倾倒 45–51 次，机器人 0 倾倒 0–7 次，物体位置 ≤ 1 次 | 中位 56–60 | 大多数回合物体误差没到 0.1 m；有的组中位 58–99 | 终止时：静止或贴地推行 |
| D Kick 722（±2、±8） | 51 / 51 | 机器人 1 倾倒 39–44 次，机器人 0 倾倒 8–12 次 | 中位 115–118 | 26–32 个回合，中位 110–115 | 静止 |
| C-in Pull（三个种子，两档，两种 PD） | 每组 51 / 51 | 全部是物体位置 | 中位 152–155 | 全部回合，中位 123–125 | 抬起 33–51 次，贴地推行 1–17 次；终止时全部是贴地推行 |
| D Pull ±8 | 11 / 21 / 17 | 物体位置（721 有 1 次机器人 0 倾倒） | 中位 190–196 | 中位 136–138 | 大多是贴地推行 |
| D Chair ±8 721 / 723 | 2 / 1 | 物体姿态 | 253–359 | 229–251 | 转动 |

- **D Kick 的参考：** 物体在第 40–57 帧第一次移动，在第 180–218 帧第二次移动。种子 721、723 的机器人 1 在第 55–67 步倾倒，也就是第一次移动刚结束时。开环 1 s 时，机器人 1 的根部已经偏了 0.42 m。
- C-in Pull 的两种 PD 设置下，终止步数和终止原因几乎相同。

## 4. 视频（只在本地，不进 git）

`exports_system1/t10/videos/{isaac,mujoco}_d_{push,pull,kick,chair}_seed721_nominal.mp4`，每个视频旁边有一个 `.json`（轨迹 sha256、相机、帧数）和第一帧 `.png`。

- 风格：论文 Fig. 4 / U5（`system1/render/u5/render_nominal_video.py`），只按保存的状态做运动学回放，1920×1080，50 fps。
- 帧数：

| 任务 | Isaac | MuJoCo |
|---|---|---|
| Push | 87（Isaac 里第 87 步终止） | 309 |
| Pull | 316 | 316 |
| Kick | 298 | 63 |
| Chair | 391 | 391 |

- **Kick 的相机：** 论文的 F3 相机没有 Kick，用了同一张桌子场景的 Push 相机（新开关 `--camera-task`，不加时行为不变）。Kick 的起始站位和 Push 不同，近处那台机器人的脚在画面下边缘被切掉一点。
- **MuJoCo 视频的物体 URDF：** 取自同一 checkpoint 的 Isaac dump 运行的 summary（`exports_system1/t10/videos/shims/`）。

## 5. 和任务书的偏离

1. 5 个 Isaac dump 重跑了一次，原因见用时一节。重跑用的命令、checkpoint 和状态文件都和第一次相同；初始状态和观测检查都用重跑后的 dump。
2. ONNX float32：Push 的误差超过 T10 的字面阈值 1e-5。按 T10s 的规则（float64 ≤ 1e-10 为通过，float32 只报告），只报数值，不判失败。
3. Kick 的视频用了 Push 的相机，见第 4 节。
4. D Push 721 nominal 在 Isaac 里第 87 步终止。观测和开环检查都用这条 87 步的轨迹，没有另选回合。

## 文件

| 文件 | 内容 |
|---|---|
| `episodes.csv` | MuJoCo 每回合一行，共 1836 行 |
| `per_seed_summary.csv` | 每组一行，共 36 行 |
| `isaac_vs_mujoco.csv` | Isaac 和 MuJoCo 的对比 |
| `failure_summary.csv`、`failure_episodes.csv`、`phase_labels_zh.json` | 失败分析 |
| `consistency/` | 初始状态、观测、开环、ONNX 检查 |
| `configs/` | MuJoCo 运行清单、启动和退出记录，Isaac 导出记录，gzip |
| `provenance.json` | checkpoint、ONNX、URDF、MJCF、状态文件、Isaac 导出的 sha256 |
| `SHA256SUMS` | 本目录所有文件的校验和 |

生成脚本：`system1/sim2sim/package_t10.py`、`system1/sim2sim/failure_t10s.py`。
