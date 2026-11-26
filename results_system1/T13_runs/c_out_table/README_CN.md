# T13 第二轮 c_out_table：结束、拷回、自动关机、本机评测

- **依据：**
  - 任务书 `docs/system1/T13_parallel_runs_zh.md` 第 5、6、8 节；
  - 第 8 节 2026-11-26 03:20 补充（收尾后自动关机）；
  - 第 8 节 2026-11-26 10:54 补充（C-out 的 q* 对照改判法，"c_out_table 用同一判法"，见第 4 节）。
- **配置：**
  - C-out-Table：SONIC 输出 + 残差，同 T03。
  - Table 5 kg，种子 721。
  - 其余和第一轮 d_table 完全相同：观测 v1、噪声 v1、T10b 随机化、critic 修正、接触三项、4096 个环境、停止规则、验证。
- **时间（纽约，2026-11-26）：**

  | 时间 | 事项 |
  |---|---|
  | 12:46:37 | 云上训练到上限 10000 轮结束，停止规则没有触发（`records/decision.json`） |
  | 12:50:06 | 本机流水线由定时任务启动（`system1/t13v4/post_run.py`，状态记录 `post_run_status.json`） |
  | 12:50:06–12:52:25 | 拷回 |
  | 12:52:25–12:53:39 | 云机 199 自动关机（只关机，不释放），关机命令 12:52:39 发出，60 s 后连不上 |
  | 12:53:42–14:00:50 | Isaac 噪声关：±2 到 13:59:11，±8 12:55:12 启动、到 14:00:50 |
  | 14:00:50–14:01:47 | Isaac 场景导出 |
  | 14:01:47–14:56:57 | Isaac 噪声开：±2 到 14:56:56，±8 14:03:17 启动、到 14:56:57 |
  | 14:01:47–14:01:51 | ONNX 导出和核对 |
  | 14:01:51–14:02:46 | MuJoCo 检查（新判法），第 1 次就通过 |
  | 14:02:46–14:06:46 | MuJoCo 闭环 ±2 / ±8 |
  | 14:06:46–14:14:34 | 视频 |
  | 14:56:57–15:01:42 | 接触统计 |
  | 15:01:42–15:02:27 | 打包、上传 Drive |

- **机器：**kevin-Legion-R9000P-ADR10，NVIDIA GeForce RTX 5070 Laptop GPU。
  - Isaac 同时最多 2 个进程。
  - 同时在跑的有 T14 第 9g2 步的 CPU 出图：
    - 12:41:56–12:54:12，2 个并行，和 Isaac 噪声关只重叠了开头约 30 s；
    - 13:00:00–13:03:04，一次一个，每个限 4 GB 内存。
- **代码：**
  - Isaac 噪声关两组在 `feadaa7` 上启动，噪声开两组在 `581ce34` 上启动。两者之间只改了 T14 的 3 个文件（出图、打包），评测代码没有变。
  - 评测脚本和 MuJoCo 端的开关从 `6b07dcb` 起没改过（`records/code_switches.diff`）。
  - C-out 的 q* 新判法在 `c91b819` 里，c_out_push 续跑时已经用过。
- **输入 sha256：**见 `inputs_sha256.csv`。
  - checkpoint `model_10000.pt`：`46e2a623…`；
  - 测试状态 ±2：`74a55320…`，±8：`c1b78801…`，和 d_table 相同；
  - T14 第 6 轮 C 参考；物体 `desk001_5kg_training.urdf`。

## 结论（只列事实）

| 组（各 50 个扰动状态；nominal 另报） | ±2 | ±8 | 走完回合的物体位置 RMSE | nominal |
|---|---|---|---|---|
| Isaac，噪声关 | **44 / 50** | **43 / 50** | 9.49 / 10.06 cm | 走完 / 走完 |
| Isaac，噪声开 | **38 / 50** | **35 / 50** | 9.53 / 10.36 cm | 走完 / 走完 |
| MuJoCo（T10 设置，Isaac PD），噪声关 | **6 / 50** | **7 / 50** | 10.63 / 9.92 cm | 第 336 步失败 / 同左 |

- **MuJoCo 的完成数和 Isaac 差很多，**也远低于第一轮 d_table 的 MuJoCo（34 / 31，同一组测试状态）。
  - MuJoCo 的 nominal 也没走完：第 336 步（共 687 步，6.72 s）物体位置偏差超过 0.25 m。
  - 第 5 节是只读的对比；没有改任何设置去重跑。
- **失败：**
  - Isaac 4 组合计 40 个：39 个是"物体位置偏差 > 0.25 m"，1 个是"物体姿态"（噪声开 ±2）。失败发生在第 292–547 步。
  - MuJoCo 2 组合计 87 个，全部是"物体位置偏差 > 0.25 m"，发生在第 329–458 步。
- **MuJoCo 端检查：**按新判法全部通过（第 4 节）。用 MuJoCo 算出的 SONIC 输入得到的 q*（旧判法，只报告）也没有超门槛的步。
- **训练没有按停止规则停：**跑到上限 10000 轮，最后一次验证 40 / 50（第 1 节）。

## 1. 停止、拷回、关机

- **停止规则（4096 个环境，轮数减半）：**4000 轮以后，如果验证完成数连续 2000 轮没有超过此前的最好值，就停；最早在 6000 那次验证之后停；否则训到 10000。
- **验证完成数依次为：**
  - 1000: 0、2000: 1、3000: 7、4000: 8、5000: 9、6000: 1、7000: 19、8000: 24、9000: 17、10000: 40。
  - 7000、8000、10000 都是新的最好值，没有出现连续 2000 轮不涨，所以训到上限 10000 轮。
  - 用最后一个 checkpoint：`model_10000.pt`。`decision.json` 记录预热 0 轮，和 d_table 相同。
- **拷回：**`rsync` 到 `logs_system1/t13v4/runs/c_out_table/`，共 1620 个文件，逐个和云上核对 sha256，全部一致（`records/copyback_*`）。
- **关机：**先确认本机已有 checkpoint、`decision.json`、`launch.json`、`state.json`、日志和 `run_config.json`，再确认云机上没有别的进程、没有 GPU 计算进程，然后执行 `shutdown`。
  - 关机时间 12:52:39，写在 `post_run_status.json` 的 `cloud_shutdown` 里。
- 云上除了关机没有改动过任何东西。

## 2. 评测设置

- **和 d_table 相同，**只换了结构：`--controller sonic_residual`。
- **Isaac：**`d_final_5070` 协议。
  - 命令取这个 run 训练中最后一次验证的命令，只换了这几项：解释器（本机 `hssim_s1`）、checkpoint、状态文件、model-id、输出目录。
  - 确定性动作，随机化关；噪声开那一组只多加 `--obs-v1-noise on`。
- **MuJoCo：**T10 设置。
  - 场景是这个 run 自己的 Isaac 场景导出（nominal 回合）。
  - PD 用 Isaac PD；SONIC 用 ONNX（`model_encoder.onnx`、`model_decoder.onnx`）；actor 用 ONNX，观测 v1、噪声关。
  - 状态文件和 Isaac 相同。
- **完成数：**只数 50 个扰动状态，nominal 单独报，不计入。

## 3. 对照（`comparison.csv`）

| 来源 | 种子 | Isaac ±2 | Isaac ±8 | MuJoCo ±2 | MuJoCo ±8 |
|---|---|---|---|---|---|
| 论文 D Table（T10；原参考、原观测、无随机化） | 721 | 35 | 45 | 0 | 0 |
| 论文 D Table | 722 | 49 | 37 | 0 | 3 |
| 论文 D Table | 723 | 46 | 40 | 0 | 0 |
| T10b 试训 D Table + 随机化（14000 轮） | 721 | 31 | 24 | 34 | 38 |
| T03 C-out（c2_main，12000 轮） | 721 | 41 | 27 | — | — |
| T13 第一轮 d_table，噪声关（同配置，结构 D） | 721 | 44 | 46 | 34 | 31 |
| T13 第一轮 d_table，噪声开 | 721 | 44 | 46 | — | — |
| **T13 第二轮 c_out_table，噪声关** | 721 | **44** | **43** | **6** | **7** |
| **T13 第二轮 c_out_table，噪声开** | 721 | **38** | **35** | — | — |

- d_table 和 c_out_table 用的是同一组测试状态（sha256 相同）。
- 论文 D、T10b、T03 的测试状态不同：T13 用新参考，状态重新生成。

## 4. MuJoCo 端检查（T10 门槛；新的 q* 判法；`records/sim2sim_check_*`、`records/onnx_check.csv`）

| 项 | 结果 | 门槛 |
|---|---|---|
| 初始状态：102 个（±2、±8 各 51 个），对比 Isaac 复位后的状态 | 关节 0 rad，根位置最大 3.0e-8 m，物体位置 0 m | 1e-4 |
| 观测 159 维（nominal 687 步，噪声关），逐维对比 Isaac 录下的观测 | 最大 1.4e-6；队友 5 维最大 8.0e-7 | 1e-4 |
| SONIC 输入（actor 观测、tokenizer），MuJoCo 算出的对比 Isaac 录下的 | 8.3e-7 / 2.4e-7 | 1e-4 |
| **q*：把 Isaac 录下的 SONIC 输入喂给 SONIC ONNX，加论文 D 缩放 × clip(a)，对比 Isaac 每步的 PD 目标（新判法）** | **2.9e-6** | **1e-4** |
| q*：用 MuJoCo 算出的 SONIC 输入（旧判法，只报告） | 3.4e-6，没有超门槛的步 | — |
| ONNX（float64 图）对 PyTorch | 6.8e-14 | 1e-10 |
| ONNX（float32）对 PyTorch | 2.1e-5（只记录，按 T10s 规则不判定） | — |
| actor ONNX 在 Isaac 录下的观测上算出的动作，对比 Isaac 的动作 | 1.7e-5 | — |
| MuJoCo 正运动学，Isaac 的全部刚体 | 位置 8.2e-7 m，姿态 1.4e-3 rad | — |

**改判法的原因（同 c_out_push，详见 `../c_out_push/README_CN.md` 第 4 节、`../c_out_push/checks_qstar_diag.json`）：**

- **SONIC 编码器的输出是量化的：**64 个分量，每格 0.0625。
- c_out_push 第 294 步（robot 0），MuJoCo 算出的 tokenizer 输入和 Isaac 录下的只差约 2e-7，但有 1 个分量正好在取整边界上，两边取到相邻的两格。q* 因此差 0.0100 rad，超过 1e-4，其余步约 1e-6。
- **Kevin 2026-11-26 选方案 1：**
  - q* 用 Isaac 录下的 SONIC 输入算，< 1e-4；
  - SONIC 输入本身单独核对，< 1e-4；
  - 两项分开查，都要过。
  - c_out_table 用同一判法（`post_run.py` 对 C-out 自动加 `--qstar-inputs isaac`）。
- **c_out_table 这次没有跳格：**两种 q* 都在 1e-5 以下，旧判法下也会通过。

## 5. MuJoCo 和 Isaac 的 nominal 对比（只读，`nominal_diverge_pm2.json`）

- **方法：**
  - 同一个 nominal 状态，两边的 nominal 轨迹逐帧对比，只用两边都有的前 337 帧；
  - 脚本 `system1/t13v4/nominal_diverge.py`，不跑仿真；
  - ±8 的 nominal 状态和 ±2 相同，`nominal_diverge_pm8.json` 逐行一致。
- **结果：**

  | 步（50 Hz） | 物体偏差：Isaac | 物体偏差：MuJoCo | 物体：MuJoCo 对 Isaac | 机器人根：MuJoCo 对 Isaac（两台取大） |
  |---|---|---|---|---|
  | 80（1.6 s） | 0.5 cm | 0.5 cm | 0.1 cm | 5.9 cm |
  | 160（3.2 s） | 0.6 cm | 0.6 cm | 0.2 cm | 7.3 cm |
  | 200（4.0 s） | 4.4 cm | 2.7 cm | 3.4 cm | 2.9 cm |
  | 240（4.8 s） | 5.9 cm | 6.8 cm | 5.3 cm | 4.2 cm |
  | 280（5.6 s） | 10.0 cm | 8.0 cm | 12.3 cm | 18.3 cm |
  | 320（6.4 s） | 14.2 cm | 18.5 cm | 30.2 cm | 34.0 cm |
  | 336（6.72 s） | 13.6 cm | 24.1 cm | 33.2 cm | 40.4 cm |

  - 机器人根的位置差第 18 步超过 1 cm、第 68 步超过 5 cm、第 247 步超过 10 cm。
  - 物体位置差第 173 步超过 1 cm、第 214 步超过 5 cm、第 276 步超过 10 cm。
  - 两边物体高度都比参考低约 6 cm（第 240 步以后）。
  - Isaac 的 nominal 后来走完了，物体位置 RMSE 8.79 cm。
- **对照，第一轮 d_table 同一状态**（`nominal_diverge_d_table_pm2.json`）：
  - 根位置差第 21 步超过 1 cm、第 148 步超过 5 cm；物体位置差第 228 步超过 1 cm、第 393 步超过 10 cm；
  - MuJoCo 的 nominal 走完了。

## 6. 视频（Drive `DuetMimic_transfer/T13_runs/c_out_table/`，`drive_manifest.csv`）

- **6 段：**±2 的 nominal 和 perturbed_000，各配全景、近景、双手近景。
- **布局：**左 Isaac（噪声关），右 MuJoCo，并排，做法同 d_table。
- **内容：**
  - nominal：Isaac 走完；MuJoCo 第 336 步失败。
  - perturbed_000：Isaac 第 363 步失败（物体位置）；MuJoCo 走完。
- **关键帧检查**（`videos/*.keyframes.jpg`，看了 nominal 的全景、近景和 perturbed_000 的双手近景）：
  - 两边都还在跑时，两台机器人和桌子都在画面里。
  - 一边结束、停在最后一帧以后，画面跟着另一边走，停住那一边的机器人移到画面边上，有一部分出了画面：
    - nominal 近景右边（MuJoCo），7.84 s 以后；
    - perturbed_000 双手近景左边（Isaac），7.84 s 以后。
  - Isaac nominal 全景 11.76 s、13.72 s 两张，左边那台的一只手离开了桌子。
    - 参考里 r0 的手接触比例本来也只有 0.585（第 7 节）。

## 7. 内部诊断（不进论文）

**训练日志（`training_diag.csv/json`；文件编号第 1–10000 轮；"近 50 轮"是最后 50 行）：**

| 指标 | r0 近 50 轮 / 全程 | r1 近 50 轮 / 全程 |
|---|---|---|
| 应接触时手力 > 1 N 的比例 | 0.926 / 0.790 | 0.933 / 0.788 |
| 应接触时丢接触的比例 | 0.074 / 0.210 | 0.067 / 0.213 |
| 第 2 条丢接触终止信号（每轮次数） | 13.5 / 54.7 | 27.1 / 71.7 |
| 第 3 条惩罚：有惩罚的步数比例 | 0.007 / 0.011 | 0.011 / 0.028 |

**评测状态的几何接触比例（`contact_summary.csv`、`contact_episodes.csv`）：**

- 口径同 d_table：URDF 碰撞几何离桌子 ≤ 1.5 cm 算接触；窗口从参考第一次手接触算起，到最后一个执行行。
- 下表是 50 个扰动回合的平均；走完和失败的回合分开的数在 `contact_summary.csv`。
- MuJoCo 回合多数在第 329–458 步结束，窗口比走完的回合短。

| 组 | 手 r0 / r1 | 两台同时有手接触 | 手以外 r0 / r1 | 其中腿 r0 / r1 |
|---|---|---|---|---|
| 参考 | 0.585 / 0.765 | 0.574 | 0 / 0 | 0 / 0 |
| Isaac 噪声关 ±2 | 0.571 / 0.877 | 0.569 | 0.100 / 0.069 | 0.066 / 0.046 |
| Isaac 噪声关 ±8 | 0.584 / 0.885 | 0.582 | 0.079 / 0.048 | 0.054 / 0.036 |
| Isaac 噪声开 ±2 | 0.590 / 0.871 | 0.584 | 0.082 / 0.079 | 0.056 / 0.060 |
| Isaac 噪声开 ±8 | 0.592 / 0.865 | 0.585 | 0.075 / 0.077 | 0.056 / 0.058 |
| MuJoCo ±2 | 0.595 / 0.843 | 0.594 | 0.093 / 0.067 | 0.087 / 0.037 |
| MuJoCo ±8 | 0.594 / 0.833 | 0.593 | 0.104 / 0.071 | 0.097 / 0.036 |
| 对照：d_table Isaac 噪声关 ±2 | 0.569 / 0.725 | 0.565 | 0.020 / 0.028 | 0.016 / 0.028 |
| 对照：d_table MuJoCo ±2 | 0.581 / 0.755 | 0.577 | 0.005 / 0.076 | 0.003 / 0.076 |

## 8. 和任务书的偏离、说明

- **MuJoCo 检查用的是新 q* 判法，**这是 Kevin 定的（第 4 节）。这次第 1 次就通过，没有续跑。
- **MuJoCo 完成数远低于 Isaac 和 d_table：**按常驻规则"结果和预期差很多：不要改设置去修结果，先如实回报"，没有改任何设置、没有重跑。第 5 节只是只读的对比。
- **评测里的手受力比例不做：**按 Kevin 2026-11-25 的决定（任务书第 8 节第 4 条），同 d_table。
- **`inputs_sha256.csv`、`records/code_switches.diff`、`records/copyback_sha256_*.txt`：**流水线打包时没有生成，事后用 `system1/t13v4/inputs_run.py --run c_out_table` 补上，只读。
- **`nominal_diverge_*.json`：**流水线之外加的只读诊断。

## 9. 文件

- `summary.csv`：每组的完成数、RMSE、失败分类。
- `episodes.csv`：每组每个回合一行，列沿用 T10b 返包，另加组、模拟器、噪声。
- `comparison.csv`、`training_diag.csv/json`、`contact_summary.csv`、`contact_episodes.csv`。
- `nominal_diverge_pm2.json`、`nominal_diverge_pm8.json`、`nominal_diverge_d_table_pm2.json`（第 5 节）。
- `inputs_sha256.csv`、`drive_manifest.csv`、`post_run_status.json`（流水线每一步的命令、起止时间、检查结果，包括关机）。
- `videos/`：关键帧图和渲染记录。
- `records/`：
  - 每组的 launch / exit 记录和 run_manifest；
  - `decision.json`、`validation_curve.json`；
  - 拷回核对（`copyback_copyback.txt`、云上和本机的 sha256 清单）；
  - MuJoCo 检查、ONNX 检查；
  - `code_switches.diff`。
- `SHA256SUMS`。
