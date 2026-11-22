# T10b B 第 5 步：试点最终评测

本文件只写事实。

- **依据：** 任务书 `docs/system1/T10b_friction_margin_zh.md` B 部分第 5 步：
  - Isaac 里按 `d_final_5070` 的协议评测，±2 和 ±8；
  - MuJoCo 里按 T10 的设置和协议评测，±2 和 ±8；
  - 和原策略并排比较。
  - Kevin 2026-11-21："C-in 按计划跑满 12000 轮，之后照做 Isaac 和 MuJoCo 的最终评测。D 继续按停止规则跑，不改设置。"
- **本版内容：** 两个试点都已评完。
  - D Table 种子 721：`d_table_seed721_rand/`，见下面"D Table 种子 721"一节。
  - C-in Bin 种子 721：`cin_bucket_seed721_rand/`。
- **两个试点的完成数（50 个扰动状态，括号里是原策略）：**

| 试点 | Isaac ±2 | Isaac ±8 | MuJoCo ±2 | MuJoCo ±8 |
|---|---|---|---|---|
| D Table 721（`model_14000.pt`） | 31（35） | 24（45） | 34（0） | 38（0） |
| C-in Bin 721（`model_12000.pt`） | 0（49） | 0（30） | 0（0） | 0（0） |

- C-in 的 MuJoCo 两种 PD 结果相同，都是 0；表里列的是 PD isaac。

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

## D Table 种子 721

### 0. checkpoint 和运行

- **checkpoint：**
  - 2026-11-21 18:51，第 14000 轮验证完成数是 31。第 12000、14000 两次验证都没有超过第 10000 轮的 45，停止规则触发，调度器按记下的 PID 停了训练。训练日志记到第 14699 轮。
  - 按任务书，用停下时的最后一个 checkpoint，`model_14000.pt`，sha256 `2f343369af2865c448dcc0406a9e6c788ea67675a93312dca76f58bf8ded1806`。
  - 它来自核对过 sha256 的云端完整拷贝（`../cloud_copy_20261121/`），放在本机 `exports_system1/t10b_pilot/d_table_seed721_rand/`，不进 git。
- **原策略：** 论文 D Table 种子 721 的 `model_12000.pt`（`f163d11a…`），用的是已有的本机 5070 评测，没有重跑：
  - ±2：`logs_system1/post_train/d_curve/D_table_paper_T-07_seed721/it12000_pm2`，完成 35；
  - ±8：`logs_system1/post_train/d_paper_final/D_table_paper_T-07_seed721/it12000_pm8`，完成 45。
  - 这两份和 T03 `c2_main` 的 D Table 721 那两行是同一份评测。
- **开始时间：** Kevin 2026-11-22 说"我用完了，cpu和gpu现在都能用"，之后才开始。

**Isaac（`run_finals.py`，HEAD `0625d8e`，没有未提交改动）**

- 命令是上面两份原评测的命令，只换了 `--checkpoint`、`--model-id`、`--output-dir`。
- 状态文件 sha256 和原评测相同：±2 `389e8da5…dac1`，±8 `6a626f48…fcad`。
- 本机 5070 两路并发，01:47 开始，±8 02:50 结束，±2 02:52 结束，退出码都是 0。

**场景核对（不在任务书里）**

- 方法同 C-in：试点 checkpoint 的 nominal 场景导出，和 T10 的 `logs_system1/t10/isaac/d_table/dump` 比。
- 不同的只有 checkpoint 路径，以及 T10 之后新增的三个字段（同 C-in）。
- 碰撞体凸包逐个相等。

**MuJoCo（`run_mj_finals.py`，HEAD `0625d8e`）**

- 场景用 T10 的导出 `logs_system1/t10/isaac/d_table/dump`。状态、PD 增益、力矩上限和 T10 的 D Table 721 逐项相同，只换了 ONNX。
- D 只有论文 PD 一种设置，和 T10 相同。
- 01:47 开始，01:48 结束。
- **ONNX 检查（`configs/onnx_check.csv`）：**
  - float32 和 PyTorch 最大差 1.55e-5，按字面阈值 1e-5 没过。T10 里原 D Table 721 的导出也是 1.53e-5，同样没过，T10 当时判为 float32 舍入。
  - float64 计算图最大差 3.4e-14，通过 T10s 的标准（≤ 1e-10）。
  - 同一开关重新导出原 checkpoint，sha256 和 T10 的 `d_table_seed721.onnx` 相同。
- **复现：** 原策略 ±2 在现在的代码上重跑，51 个回合和 T10 逐个相同，物体 RMSE 差 0.0。

### 1. 完成数（50 个扰动状态，nominal 另记）

| 仿真器 | 档 | 原策略 | 试点 |
|---|---|---|---|
| Isaac | ±2 | 35 | 31 |
| Isaac | ±8 | 45 | 24 |
| MuJoCo | ±2 | 0 | 34 |
| MuJoCo | ±8 | 0 | 38 |

- **nominal：** 试点四组都完成了；原策略 Isaac 两组完成，MuJoCo 两组没完成。
- **试点完成回合的物体 RMSE 中位：** Isaac ±2 9.03 cm、±8 9.56 cm；MuJoCo ±2 11.40 cm、±8 10.48 cm。
  - 原策略 Isaac 完成回合是 9.07、8.62 cm。原策略在 MuJoCo 没有完成的回合。
- **试点的验证曲线（Isaac 验证集）：** 4、30、26、32、45、36、31，见 `../pilot_progress/`。

### 2. 失败回合

判定规则同 C-in 一节。D 的参考里接触权重全为 0，所以没有"谁拿着物体"这一列。

| 仿真器 | 档 | 策略 | 失败 | 终止步，中位（范围） | 物体位置 / 物体姿态 / 机器人姿态 | 终止时的参考阶段 | > 0.1 m 的步，中位；当时的阶段 |
|---|---|---|---|---|---|---|---|
| Isaac | ±2 | 试点 | 19 | 410（273–500） | 17 / 1 / 1 | 举起中 10，贴地推行 9 | 232；举起中 19 |
| Isaac | ±2 | 原策略 | 15 | 339（255–419） | 15 / 0 / 0 | 举起中 14，贴地推行 1 | 259；举起中 15 |
| MuJoCo | ±2 | 试点 | 16 | 441（390–473） | 16 / 0 / 0 | 贴地推行 11，放下 3，举起中 2 | 247；举起中 16 |
| MuJoCo | ±2 | 原策略 | 50 | 356（339–432） | 50 / 0 / 0 | 举起中 49，贴地推行 1 | 291；举起中 50 |
| Isaac | ±8 | 试点 | 26 | 313.5（258–484） | 25 / 1 / 0 | 举起中 20，贴地推行 5，放下 1 | 231；举起中 23，贴地推行 2，静止 1 |
| Isaac | ±8 | 原策略 | 5 | 333（70–348） | 4 / 1 / 0 | 举起中 4，静止 1 | 257；举起中 3，静止 2 |
| MuJoCo | ±8 | 试点 | 12 | 391（282–482） | 11 / 1 / 0 | 举起中 7，贴地推行 3，放下 2 | 231；举起中 11，静止 1 |
| MuJoCo | ±8 | 原策略 | 50 | 358（76–430） | 48 / 0 / 2 | 举起中 45，静止 3，贴地推行 2 | 286.5（48 个回合超过）；举起中 46，静止 2 |

- **T10 的计法：** 原策略 MuJoCo ±2 是 50 / 0 / 0，±8 是 48 / 0 / 2，和 T10 的 `per_seed_summary.csv` 相同。
- **参考的时间线：** 687 帧，阶段见 `../pilot_progress/references.json`。

### 3. 同一个状态在两个仿真器里是否完成

| 档 | 策略 | 两边都完成 | 只在 Isaac 完成 | 只在 MuJoCo 完成 | 两边都没完成 |
|---|---|---|---|---|---|
| ±2 | 试点 | 21 | 10 | 13 | 6 |
| ±2 | 原策略 | 0 | 35 | 0 | 15 |
| ±8 | 试点 | 22 | 2 | 16 | 10 |
| ±8 | 原策略 | 0 | 45 | 0 | 5 |

### 4. 和任务书的偏离

- 同 C-in 一节：多做了场景核对和 MuJoCo 复现；评测协议、状态、场景都没有改。
- `system1/t10b/pilot_report.py` 改了一处：原训练没有的轮数（D 的第 14000 轮）不再报错，原训练那一列留空。
  - 第 2000–12000 轮的输出和改之前逐项相同。

### 文件（`d_table_seed721_rand/`）

文件和 C-in 一节相同：`comparison.csv`、`episodes.csv`、`scene_check.json`、`provenance.json`、`configs/`。D 只有一种 PD，所以 `pd` 列都是 `isaac`。

```
hssim_s1/bin/python -B system1/t10b/package_finals.py --run d_table_seed721_rand
```
