# T13 第二轮 c_out_push：结束、拷回、自动关机、本机评测

- **依据：**
  - 任务书 `docs/system1/T13_parallel_runs_zh.md` 第 5、6、8 节；
  - 第 8 节 2026-11-26 03:20 补充（收尾后自动关机）；
  - 第 8 节 2026-11-26 10:54 补充（C-out 的 q* 对照改判法，见第 4 节）。
- **配置：**
  - C-out-Push：SONIC 输出 + 残差，同 T03。
  - Push 20 kg，种子 721。
  - 其余和第一轮 d_push 完全相同：观测 v1、噪声 v1、T10b 随机化、critic 修正、接触三项、4096 个环境、停止规则、验证。
- **时间（纽约，2026-11-26）：**

  | 时间 | 事项 |
  |---|---|
  | 09:21:39 | 云上训练按停止规则停（`records/decision.json`），09:23:39 结束 |
  | 09:31:42 | 本机流水线由定时任务启动（`system1/t13v4/post_run.py`，状态记录 `post_run_status.json`） |
  | 09:31:42–10:00:50 | 拷回 |
  | 10:00:50–10:02:01 | 云机 196 自动关机（只关机，不释放），关机命令 10:01:00 发出，60 s 后连不上 |
  | 10:02:03–10:29:04 | Isaac 噪声关：±2 到 10:27:37，±8 10:03:33 启动、到 10:29:04 |
  | 10:29:04–10:29:43 | Isaac 场景导出 |
  | 10:29:43–11:09:44 | Isaac 噪声开：±2 到 11:08:04，±8 10:31:13 启动、到 11:09:44 |
  | 10:29:43–10:29:47 | ONNX 导出和核对 |
  | 10:29:47–10:29:56 | MuJoCo 检查，第 1 次：q* 逐步对照不过，流水线按规则停下（第 4 节） |
  | 11:09:44 | Kevin 改判法后，从检查这一步续跑（`--resume-from checks`；前 5 步不重跑，只重新核对它们的输出） |
  | 11:09:44–11:10:00 | MuJoCo 检查，第 2 次（新判法）：通过 |
  | 11:10:00–11:12:51 | MuJoCo 闭环 ±2 / ±8 |
  | 11:12:51–11:16:11 | 视频 |
  | 11:16:11–11:17:51 | 接触统计 |
  | 11:17:51–11:18:37 | 打包、上传 Drive |

- **机器：**kevin-Legion-R9000P-ADR10，NVIDIA GeForce RTX 5070 Laptop GPU。
  - Isaac 同时最多 2 个进程，每个约 5.4 GB。
  - 同时在跑的有 T14 的 CPU 求解（16–31 号核；按内存门和"T13 评测在跑时最多 2 个"让路）。
- **代码：**
  - Isaac 四组都在 `225894d` 上启动。
  - 续跑在 `259254d` 上，它包含 `c91b819`（新判法和续跑选项）。
  - 评测脚本和 MuJoCo 端的开关从 `6b07dcb` 起没改过（`records/code_switches.diff`）。
- **输入 sha256：**见 `inputs_sha256.csv`。
  - checkpoint `model_09025.pt`：`c3293ee4…`；
  - 测试状态 ±2：`a6933d54…`，±8：`2edd4351…`，和 d_push 相同；
  - T14 第 6 轮 C 参考。

## 结论（只列事实）

| 组（各 50 个扰动状态；nominal 另报） | ±2 | ±8 | 走完回合的物体位置 RMSE | nominal |
|---|---|---|---|---|
| Isaac，噪声关 | **46 / 50** | **46 / 50** | 8.98 / 8.73 cm | 走完 / 走完 |
| Isaac，噪声开 | **45 / 50** | **47 / 50** | 7.92 / 8.54 cm | 走完 / 走完 |
| MuJoCo（T10 设置，Isaac PD），噪声关 | **50 / 50** | **50 / 50** | 6.96 / 8.43 cm | 走完 / 走完 |

- **失败：**Isaac 4 组合计 16 个。
  - 15 个是"物体位置偏差 > 0.25 m"；
  - 1 个是"机器人姿态"（噪声关 ±2）。
- MuJoCo 没有失败。
- **MuJoCo 端检查：**按新判法全部通过（第 4 节）。第 1 次按 T10c 的旧判法，q* 有 1 步不过。

## 1. 停止、拷回、关机

- **停止规则（4096 个环境，轮数减半）：**验证完成数依次为
  - 1000: 22、2000: 37、3000: 36、4000: 37、5000: 48、6000: 44、7000: 49、8000: 47、9000: 49。
  - 7000 轮的最好值 49 之后，连续 2000 轮没有超过，在 9000 那次验证后停。
  - 用停下时最后一个 checkpoint：`model_09025.pt`（预热 25 轮后计 9000 轮）。
- **拷回：**`rsync` 到 `logs_system1/t13v4/runs/c_out_push/`，共 1473 个文件，逐个和云上核对 sha256，全部一致（`records/copyback_*`）。
- **关机：**先确认本机已有 checkpoint、`decision.json`、`launch.json`、`state.json`、日志和 `run_config.json`，再确认云机上没有别的进程、没有 GPU 计算进程（10:00:55 查过一次），然后执行 `shutdown`。
  - 关机时间 10:01:00，写在 `post_run_status.json` 的 `cloud_shutdown` 里。
- 云上除了关机没有改动过任何东西。

## 2. 评测设置

- **和 d_push 相同，**只换了结构：`--controller sonic_residual`。
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
| 论文 D Push（T03 `d_final_5070`；原参考、原观测、无随机化） | 721 | 39 | 31 | — | — |
| 论文 D Push | 722 | 50 | 49 | — | — |
| 论文 D Push | 723 | 37 | 38 | — | — |
| T13 第一轮 d_push，噪声关（同配置，结构 D） | 721 | 43 | 44 | 50 | 50 |
| T13 第一轮 d_push，噪声开 | 721 | 36 | 42 | — | — |
| T13 第二轮 c_out_push，噪声关 | 721 | 46 | 46 | 50 | 50 |
| T13 第二轮 c_out_push，噪声开 | 721 | 45 | 47 | — | — |

- d_push 和 c_out_push 用的是同一组测试状态（sha256 相同）。
- 论文 D 的测试状态不同：T13 用新参考，状态重新生成。

## 4. MuJoCo 端检查和 q* 判法的改动（T10 门槛；`records/sim2sim_check_*`、`records/onnx_check.csv`）

| 项 | 结果 | 门槛 |
|---|---|---|
| 初始状态：102 个（±2、±8 各 51 个），对比 Isaac 复位后的状态 | 关节 0 rad，根位置最大 6.0e-8 m，物体位置 0 m | 1e-4 |
| 观测 159 维（nominal 309 步，噪声关），逐维对比 Isaac 录下的观测 | 最大 1.4e-6；队友 5 维最大 1.4e-6 | 1e-4 |
| SONIC 输入（actor 观测、tokenizer），MuJoCo 算出的对比 Isaac 录下的 | 1.4e-6 / 2.1e-7 | 1e-4 |
| **q*：把 Isaac 录下的 SONIC 输入喂给 SONIC ONNX，加论文 D 缩放 × clip(a)，对比 Isaac 每步的 PD 目标（新判法）** | **2.7e-6** | **1e-4** |
| q*：用 MuJoCo 算出的 SONIC 输入（旧判法，现在只报告） | 1.0e-2（只有第 294 步超过门槛） | — |
| ONNX（float64 图）对 PyTorch | 4.5e-13 | 1e-10 |
| ONNX（float32）对 PyTorch | 2.1e-4（只记录，按 T10s 规则不判定） | — |
| actor ONNX 在 Isaac 录下的观测上算出的动作，对比 Isaac 的动作 | 8.6e-6 | — |
| MuJoCo 正运动学，Isaac 的全部刚体 | 位置 5.8e-7 m，姿态 1.3e-3 rad | — |

**为什么改判法（Kevin 2026-11-26 选方案 1；第 1 次的检查结果见 `checks_first_try_summary.json`，诊断见 `checks_qstar_diag.json`，脚本 `system1/t13v4/cout_qstar_diag.py`）：**

- **第 1 次按 T10c 的判法：**用 MuJoCo 算出的 SONIC 输入跑 SONIC ONNX，逐步对比 q*。309 步里只有第 294 步（robot 0）超过门槛，q* 差 0.0100 rad（动作差 0.029）；其余各步约 1e-6。流水线按规则停在这一步。
- **原因是 SONIC 编码器的输出是量化的：**64 个分量，每格 0.0625，这一步只出现 15 种取值。
  - MuJoCo 算出的 tokenizer 输入和 Isaac 录下的只差 2.1e-7。但第 294 步 robot 0 有 1 个分量正好在取整边界上，两边取到了相邻的两格，差 0.0625。
  - 309 步 × 2 台里，编码器输出不同的只有这一处。
  - 在这一步 Isaac 录下的输入上加 N(0, 1e-6) 扰动，200 次里 45.5% 动作会变超过 1e-3，最大变 0.029。
- **用 Isaac 录下的 SONIC 输入：**309 步的 q* 最大差 2.7e-6，动作最大差 1.6e-5，全部过门槛。
- **新判法（`check_sim2sim_v1.py --qstar-inputs isaac`，`post_run.py` 对 C-out 自动加；不加时仍是旧判法）：**
  - q* 用 Isaac 录下的 SONIC 输入算，< 1e-4；
  - SONIC 输入本身单独核对，< 1e-4；
  - 两项分开查，都要过。
  - 用 MuJoCo 算出的 SONIC 输入得到的 q*，以及超门槛的步，继续写在 summary 里，只报告。
  - c_out_table 用同一判法。

## 5. 视频（Drive `DuetMimic_transfer/T13_runs/c_out_push/`，`drive_manifest.csv`）

- **6 段：**±2 的 nominal 和 perturbed_000，各配全景、近景、双手近景。
- **布局：**左 Isaac（噪声关），右 MuJoCo，并排，做法同 d_push。
- **内容：**nominal 和 perturbed_000 两边都走完（309 步）。
- **关键帧检查**（`videos/*.keyframes.jpg`）：
  - 近景和双手近景里，两台机器人的手和桌沿在每张关键帧里都在画面内。
  - 最后两张关键帧（5.28 s、6.16 s）机器人站直后，左边那台的头被左上角的说明框挡住一部分，双手近景里头顶出了画面。
  - 全景按两段回合的包围盒取景，距离 4.2–4.3 m，机器人在画面里比较小，和 d_push 的全景同一规则。

## 6. 和任务书的偏离、说明

- **MuJoCo 检查的 q* 判法改了，**见第 4 节；这是 Kevin 定的。流水线第 1 次停下以后，从检查这一步续跑：
  - 拷回、关机、Isaac 噪声关、场景导出、ONNX 都没有重跑，续跑时重新核对了它们的输出（`post_run_status.json` 的 `resume_verify`）；
  - Isaac 噪声开由第 1 次的流水线跑完，续跑时核对记录完整，没有重跑。
- **评测里的手受力比例不做：**按 Kevin 2026-11-25 的决定（任务书第 8 节第 4 条），同 d_push。
- **`inputs_sha256.csv`、`records/code_switches.diff`、`records/copyback_sha256_*.txt`：**流水线打包时没有生成，事后用 `system1/t13v4/inputs_run.py --run c_out_push` 补上，只读。
- **训练诊断只算到所用 checkpoint 为止：**训练日志记到第 9382 轮（文件编号），只用 ≤ 9025 轮的 9000 行。

## 7. 内部诊断（不进论文）

**训练日志（`training_diag.csv/json`；文件编号第 26–9025 轮；"近 50 轮"是最后 50 行）：**

| 指标 | r0 近 50 轮 / 全程 | r1 近 50 轮 / 全程 |
|---|---|---|
| 应接触时手力 > 1 N 的比例 | 0.742 / 0.661 | 0.661 / 0.601 |
| 应接触时丢接触的比例 | 0.254 / 0.335 | 0.331 / 0.393 |
| 第 2 条丢接触终止信号（每轮次数） | 48.6 / 85.4 | 52.3 / 98.9 |
| 第 3 条惩罚：有惩罚的步数比例 | 0.032 / 0.033 | 0.040 / 0.041 |

**评测状态的几何接触比例（`contact_summary.csv`、`contact_episodes.csv`）：**
- 口径同 d_push：URDF 碰撞几何离桌子 ≤ 1.5 cm 算接触；窗口从参考第一次手接触算起，到最后一个执行行。
- 下表是 50 个扰动回合的平均；走完和失败的回合分开的数在 `contact_summary.csv`。

| 组 | 手 r0 / r1 | 两台同时有手接触 | 手以外 r0 / r1 | 其中腿 r0 / r1 |
|---|---|---|---|---|
| 参考 | 0.822 / 0.822 | 0.822 | 0 / 0 | 0 / 0 |
| Isaac 噪声关 ±2 | 0.764 / 0.746 | 0.731 | 0.360 / 0.240 | 0.349 / 0.233 |
| Isaac 噪声关 ±8 | 0.767 / 0.747 | 0.734 | 0.374 / 0.246 | 0.364 / 0.235 |
| Isaac 噪声开 ±2 | 0.770 / 0.745 | 0.732 | 0.379 / 0.246 | 0.367 / 0.239 |
| Isaac 噪声开 ±8 | 0.761 / 0.743 | 0.727 | 0.362 / 0.245 | 0.350 / 0.235 |
| MuJoCo ±2 | 0.753 / 0.724 | 0.714 | 0.324 / 0.183 | 0.323 / 0.167 |
| MuJoCo ±8 | 0.757 / 0.723 | 0.714 | 0.324 / 0.190 | 0.320 / 0.180 |
| 对照：d_push Isaac 噪声关 ±2 | 0.826 / 0.830 | 0.821 | 0.226 / 0.337 | 0.205 / 0.314 |

## 8. 文件

- `summary.csv`：每组的完成数、RMSE、失败分类。
- `episodes.csv`：每组每个回合一行，列沿用 T10b 返包，另加组、模拟器、噪声。
- `comparison.csv`、`training_diag.csv/json`、`contact_summary.csv`、`contact_episodes.csv`。
- `inputs_sha256.csv`、`drive_manifest.csv`、`post_run_status.json`（流水线每一步的命令、起止时间、检查结果，包括关机和续跑）。
- `checks_first_try_summary.json`（第 1 次 MuJoCo 检查的 summary）、`checks_qstar_diag.json`（第 294 步诊断）。
- `videos/`：关键帧图和渲染记录。
- `records/`：
  - 每组的 launch / exit 记录和 run_manifest；
  - `decision.json`、`validation_curve.json`；
  - 拷回核对（`copyback_copyback.txt`、云上和本机的 sha256 清单）；
  - MuJoCo 检查（第 2 次）、ONNX 检查；
  - `code_switches.diff`。
- `SHA256SUMS`。
