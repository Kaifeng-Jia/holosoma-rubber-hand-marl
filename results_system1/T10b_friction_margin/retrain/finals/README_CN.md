# T10b B 第 5 步：试点最终评测

本文件只写事实。

- **依据：** 任务书 `docs/system1/T10b_friction_margin_zh.md` B 部分第 5 步：
  - Isaac 里按 `d_final_5070` 的协议评测，±2 和 ±8；
  - MuJoCo 里按 T10 的设置和协议评测，±2 和 ±8；
  - 和原策略并排比较。
  - Kevin 2026-11-21："C-in 按计划跑满 12000 轮，之后照做 Isaac 和 MuJoCo 的最终评测。D 继续按停止规则跑，不改设置。"
- **本版内容：** C-in Bin 种子 721（`cin_bucket_seed721_rand/`）。
  - D Table 种子 721 还在按停止规则训练，第 12000 轮验证是 36。停下后用同样的脚本补进 `d_table_seed721_rand/`。

## C-in Bin 种子 721

### 0. checkpoint 和运行

- **checkpoint：** 试点训满 12000 轮，用最后一个：`model_12000.pt`。
  - 2026-11-21 14:56 从云端复制到本机 `exports_system1/t10b_pilot/cin_bucket_seed721_rand/`，不进 git。
  - sha256 `be438b4a06356520d5b16646a6cda6ac22c299bf242cf1ec938b339cae735a9d`，和云端文件、第 12000 轮验证记录里的值相同。
- **原策略：** T03 的 C-in Bin 种子 721 最终 checkpoint，即 T10 / T10b A 用的那个。
  - 原策略的数取自已有记录：Isaac 取 `logs_system1/post_train/final/` 的本机 5070 最终评测，MuJoCo 取 `logs_system1/t10/mj_eval/`。没有重跑。

**Isaac（`system1/t10b/run_finals.py`）**

- 命令是原策略本机 5070 最终评测的命令，只换了 `--checkpoint`、`--model-id`、`--output-dir` 三项：
  - ±2：`cin_bucket_m1kg_seed721_fresh12000_pm2.launch.json`；
  - ±8：`cin_bucket_m1kg_seed721_fresh12000_pm8.launch.json`。
- 状态文件和原评测相同，sha256 逐个核对过：±2 `631026d3…d71f`，±8 `3bae05a1…6ff0`。
- 在本机 RTX 5070 Laptop 上两路并发，CPU 槽位和原最终评测一样，是 16-23 / 24-31。
  - 14:57 开始，±8 15:30 结束，±2 15:31 结束，退出码都是 0。
- 启动时 HEAD 是 `2767ad9`。`run_finals.py` 当时还没提交，之后没改，原样提交在 `03d9658`。git 状态记在各自的 launch 记录里。

**场景核对（不在任务书里）**

- 用试点 checkpoint 跑了一次 nominal，带 `--sim2sim-dump`，把导出的场景和 T10 导出的原策略场景 `logs_system1/t10/isaac/cin_bucket/dump` 逐项比较。结果在 `scene_check.json`。
- 不同的只有两处：
  - checkpoint 路径；
  - 三个只在试点导出里有的字段：`physics_materials`、`scene_default_material`、`runtime_reference`。这三项是 T10 之后的代码才开始导出的。
- 碰撞体凸包逐个相等。其余各项，包括机器人、物体、PD、物理设置，都相同。

**MuJoCo（`system1/t10b/run_mj_finals.py`，提交 `03d9658`）**

- `system1/sim2sim/mj_eval.py` 加了 `--onnx`、`--checkpoint` 两个开关，只换策略。其余照 T10：
  - 场景用 T10 的导出 `logs_system1/t10/isaac/cin_bucket/dump`；
  - 状态文件和 T10 相同；
  - PD 分两种：(i) isaac，(ii) official。
- 两个开关不打开时，行为不变。
- **ONNX：** 用 `system1/sim2sim/onnx_check.py --extra-checkpoint` 导出，检查结果在 `configs/onnx_check.csv`。
  - 和 PyTorch 比：float32 最大差 1.9e-6；float64 计算图最大差 4.9e-15，T10s 的通过标准是 ≤ 1e-10，通过。
  - 同一开关把原 checkpoint 再导出一次，得到的 ONNX 和 T10 的 `cin_bucket_seed721.onnx` sha256 相同。
- 每组单线程，跑在各自的一个核上（0–4 号核），14:58 开始，15:04 全部结束。
- **复现（不在任务书里）：** 用现在的代码，把原策略 ±2、PD isaac 那组重跑了一次。
  - 51 个回合和 T10 的结果逐个相同：完成与否、步数都相同，物体 RMSE 差 0.0。
  - ONNX、checkpoint、状态、场景的 sha256 也都相同。

### 1. 完成数（50 个扰动状态，nominal 另记）

| 仿真器 | 档 | PD | 原策略 | 试点 |
|---|---|---|---|---|
| Isaac | ±2 | — | 49 | 0 |
| Isaac | ±8 | — | 30 | 0 |
| MuJoCo | ±2 | isaac | 0 | 0 |
| MuJoCo | ±2 | official | 0 | 0 |
| MuJoCo | ±8 | isaac | 0 | 0 |
| MuJoCo | ±8 | official | 0 | 0 |

- nominal：原策略在 Isaac 的 ±2、±8 两组里都完成了；其余各组，包括试点的所有组，都没完成。
- 试点的验证曲线（Isaac 验证集，每 2000 轮）：0、0、0、0、0、0。见 `../pilot_progress/`。

### 2. 失败回合

**判定规则：**

- **终止原因：**
  - Isaac：用最后一行的物体，和终止判定用的那一帧参考比，规则同 T10b A。
  - MuJoCo：用评测器自己记录的终止误差。
  - 两边按同样的先后判：物体位置误差 > 0.25 m，记物体位置；否则物体姿态误差 > 0.8 rad，记物体姿态；否则记机器人姿态。
- **阶段和谁拿着物体：** 按参考，取终止判定用的那一帧。阶段规则同 `pilot_progress`。
- **"> 0.1 m"：** 物体偏离参考第一次超过 0.1 m 的那一步。

| 仿真器 | 档 | PD | 策略 | 失败 | 终止步，中位（范围） | 物体位置 / 物体姿态 / 机器人姿态 | 终止时的参考阶段 | 终止时谁拿着物体 | > 0.1 m 的步，中位；当时的阶段 |
|---|---|---|---|---|---|---|---|---|---|
| Isaac | ±2 | — | 试点 | 50 | 359（38–370） | 45 / 5 / 0 | 转动 47，抬起 2，静止 1 | 机器人 1 47，机器人 0 2，没人 1 | 204；抬起 42，静止 8 |
| Isaac | ±2 | — | 原策略 | 1 | 314 | 1 / 0 / 0 | 转动 1 | 机器人 1 1 | 213；抬起 1 |
| MuJoCo | ±2 | isaac | 试点 | 50 | 359（36–364） | 49 / 1 / 0 | 转动 48，静止 2 | 机器人 1 48，没人 2 | 204；抬起 43，静止 7 |
| MuJoCo | ±2 | isaac | 原策略 | 50 | 298（212–366） | 36 / 14 / 0 | 转动 48，抬起 2 | 机器人 1 48，机器人 0 2 | 204；抬起 41，静止 5，转动 2，举起中 2 |
| MuJoCo | ±2 | official | 试点 | 50 | 359（36–364） | 48 / 2 / 0 | 转动 48，静止 2 | 机器人 1 48，没人 2 | 204；抬起 43，静止 7 |
| MuJoCo | ±2 | official | 原策略 | 50 | 299（212–362） | 39 / 11 / 0 | 转动 47，抬起 3 | 机器人 1 47，机器人 0 3 | 204；抬起 39，静止 5，转动 4，举起中 2 |
| Isaac | ±8 | — | 试点 | 50 | 358（24–366） | 43 / 7 / 0 | 转动 46，静止 4 | 机器人 1 45，没人 4，两台 1 | 204；抬起 32，静止 18 |
| Isaac | ±8 | — | 原策略 | 20 | 119.5（4–314） | 19 / 1 / 0 | 静止 10，转动 6，抬起 4 | 没人 10，两台 5，机器人 1 3，机器人 0 2 | 6；静止 15，抬起 5 |
| MuJoCo | ±8 | isaac | 试点 | 50 | 357（8–362） | 47 / 3 / 0 | 转动 37，静止 12，抬起 1 | 机器人 1 36，没人 12，两台 2 | 200；抬起 29，静止 21 |
| MuJoCo | ±8 | isaac | 原策略 | 50 | 293（7–339） | 45 / 5 / 0 | 转动 29，静止 16，抬起 5 | 机器人 1 28，没人 16，机器人 0 5，两台 1 | 199；抬起 26，静止 21，举起中 3 |
| MuJoCo | ±8 | official | 试点 | 50 | 357（8–362） | 47 / 3 / 0 | 转动 38，静止 11，抬起 1 | 机器人 1 36，没人 11，两台 2，机器人 0 1 | 200；抬起 29，静止 21 |
| MuJoCo | ±8 | official | 原策略 | 50 | 291.5（7–315） | 45 / 5 / 0 | 转动 29，静止 16，抬起 5 | 机器人 1 28，没人 16，机器人 0 5，两台 1 | 199；抬起 26，静止 21，举起中 3 |

- **T10 的计法：** 三种原因各自计数，一个回合可以同时算进几种。按这种计法的结果在 `comparison.csv` 的 `t10_terminated_*` 列。
  - 原策略 MuJoCo ±2、PD isaac：36 / 18 / 0，和 T10 的计法一致。
- **参考的时间线：** 497 帧，见 `../pilot_progress/README_CN.md` 第 4 节：
  - 169–222 帧只有机器人 0 拿着物体，223–277 帧交接，278–465 帧只有机器人 1 拿着；
  - 261–414 帧是转动段。

### 3. 同一个状态在两个仿真器里的终止步

每个扰动状态取"MuJoCo 终止步 − Isaac 终止步"。MuJoCo 用 PD isaac 那组；Isaac 完成的回合按 497 步算。

| 档 | 策略 | 差的中位 | 四分位 | 差在 5 步以内的状态数 |
|---|---|---|---|---|
| ±2 | 试点 | −1 | −5.75 到 4.5 | 25 / 50 |
| ±2 | 原策略 | −199 | −199.75 到 −198 | 0 / 50 |
| ±8 | 试点 | −1 | −54 到 2 | 23 / 50 |
| ±8 | 原策略 | −199 | −202.5 到 −32 | 5 / 50 |

### 4. 和任务书的偏离

- **任务书之外多做了两项核对：** 第 0 节的场景核对，以及 MuJoCo 复现。
- **改了两个 T10 工具：**
  - `onnx_check.py` 加了 `--extra-checkpoint`；
  - `mj_eval.py` 加了 `--onnx`、`--checkpoint`。
  - 两处改动都是开关，不打开时行为不变。同一开关重新导出原 checkpoint，得到的 ONNX 和 T10 的逐字节相同。
- 评测协议、状态、场景都没有改。

### 文件（`cin_bucket_seed721_rand/`）

| 文件 | 内容 |
|---|---|
| `comparison.csv` | 每个仿真器 × 档 × PD × 策略一行：完成数、nominal、终止原因计数（本文件的计法和 T10 的计法）、失败回合终止步的中位和范围、> 0.1 m 的步中位、终止阶段、物体 RMSE 中位、checkpoint 和状态的 sha256 |
| `episodes.csv` | 每个回合一行，试点和原策略、两个仿真器都在里面。字段同 T10b A 的 `episodes.csv`，另加 `simulator`、`policy`、`level_cm`、`pd`、`divergence_step_0p1m`、`termination_reference_frame`、`phase_at_*`、`holder_at_*`、`last_torso_gravity_z_err_max`（只有 MuJoCo 有） |
| `scene_check.json` | 第 0 节的场景核对 |
| `provenance.json` | 打包脚本和 `references.json` 的 sha256，MuJoCo 复现的结果 |
| `configs/` | Isaac 和 MuJoCo 每组的 launch / exit / `run_manifest.json`（gzip），MuJoCo 复现组的记录，`onnx_check.csv` |

- **生成命令：** 只读，只用 CPU。
  ```
  hssim_s1/bin/python -B system1/t10b/package_finals.py --run cin_bucket_seed721_rand
  ```
