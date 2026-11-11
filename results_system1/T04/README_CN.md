# T04 返包（第一部分）：方案 A 全部质量档位评测

本文件只写事实。

- **D 的直接评测**（原质量策略放到其他档，不重训）：见 `D/README_CN.md`（2026-11-10 第二部分）。
- **D、C 的重训 run：** 在云端训练中。
- **B：** 等 T02 规划器验证的决定。

## Receipt

- 任务书：`docs/system1/T04_mass_sweep_zh.md`，sha256 `3d4f2787d6b82f34fae0ca1405bd4bd87bc489ccfccfcbe3a642c0419f9859f4`（2026-11-10 第三版）。
- 机器：hostname `kevin-Legion-R9000P-ADR10`，GPU NVIDIA GeForce RTX 5070 Laptop（8151 MiB）。评测开始时 GPU 上没有其他任务。
- git HEAD：`fbb39f6`，18 组启动时都没有未提交改动。Python 环境：`hssim_s1`。
- 时间（America/New_York，2026-11-10）：
  - 评测 05:56:25 – 09:08:37（3 h 12 min），由 `system1/t01b_pipeline.sh` 在方案 A 主设置和敏感性组之后自动运行；
  - 质量读回 15:20 – 15:23。

## 做了什么

- **方案 A，不训练。** 每个任务 4 个质量档 × ±2/±8 两档 × 51 回合（nominal + 50 个扰动状态）。
  - 原质量档（Pull 20 kg、Table 5 kg、Bin 1 kg）直接用 T01b 方案 A 主设置的结果，没有重跑。
  - 其余 9 个档位是这次新跑的 18 组。
- **评测协议：** 与方案 A 主设置相同：GPU PhysX，TF32 关，冻结初始状态（sha256 在 `mass_sweep_episodes.csv` 的 `state_file_sha256` 列），终止条件和指标与 V3 统一评测相同。
- **只改质量：** 用 `system1/mass_variants.py` 生成质量变体 URDF：质量和 6 个惯量分量按同一比例缩放，其他内容逐字不变。
  - Bin 的 URDF 只有一个 `<mass>` 和一个 `<inertia>`，没有报错。
  - 9 个变体 URDF 都已在 git 中。
- **质量和惯量读回：** 用评测脚本的 `--readback-only` 从 PhysX 读回。评测脚本、控制器和资产路径都与评测时相同，没有运行回合。
  - 12 个档位里，读回质量与 URDF 完全相同。
  - 惯量 9 个分量的最大相对差：Bin 1.3e-5，Pull 4.8e-8，Table 6.7e-8（float32 精度）。
  - 读回时所用 URDF 的 sha256，与评测 1224 行记录的逐行一致。
  - 详见 `A/object_physics_readback.csv`，其中还有质心和摩擦。

## 结果

| 任务 | 质量 (kg) | 是否原质量 | ±2 完成 / 平均进度 | ±8 完成 / 平均进度 | ±2 / ±8 SONIC 判摔倒 | nominal 进度 |
|---|---|---|---|---|---|---|
| pull | 5 | 否 | 0/50 / 0.462 | 0/50 / 0.451 | 3 / 8 | 0.494 |
| pull | 10 | 否 | 0/50 / 0.462 | 0/50 / 0.467 | 6 / 5 | 0.494 |
| pull | 20 | 是 | 0/50 / 0.460 | 0/50 / 0.477 | 5 / 3 | 0.494 |
| pull | 40 | 否 | 0/50 / 0.440 | 0/50 / 0.465 | 13 / 5 | 0.494 |
| table | 2.5 | 否 | 0/50 / 0.386 | 0/50 / 0.388 | 0 / 0 | 0.386 |
| table | 5 | 是 | 0/50 / 0.386 | 0/50 / 0.382 | 0 / 0 | 0.386 |
| table | 10 | 否 | 0/50 / 0.386 | 0/50 / 0.387 | 0 / 0 | 0.386 |
| table | 20 | 否 | 0/50 / 0.385 | 0/50 / 0.386 | 0 / 0 | 0.384 |
| bucket | 1 | 是 | 0/50 / 0.480 | 0/50 / 0.436 | 0 / 0 | 0.439 |
| bucket | 2 | 否 | 0/50 / 0.497 | 0/50 / 0.481 | 0 / 0 | 0.557 |
| bucket | 4 | 否 | 0/50 / 0.511 | 0/50 / 0.458 | 0 / 0 | 0.431 |
| bucket | 8 | 否 | 0/50 / 0.456 | 0/50 / 0.454 | 0 / 0 | 0.433 |

- **完成情况：** 24 组、1200 个扰动回合全部未完成。
- **失败回合在终止帧上的物体误差（与 T01b 主设置同口径）：**
  - Pull、Table：所有档位都是物体位置偏离参考 > 0.25 m。
  - Bin 1、2、4 kg：位置超限和姿态超限两类都有。4 kg ±8 有 1 个回合两项都未超。
  - Bin 8 kg：±2 有 49 个位置超限、1 个姿态超限；±8 有 50 个位置超限，其中 1 个同时姿态超限。
  - 逐组计数见 `A/mass_sweep_summary.csv` 的 `n_fail_terminal_*` 三列。
  - > **更正（2026-11-11）：** 上面关于 Bin 4 kg ±8 和 Bin 8 kg ±8 的两句不成立，原因是参考帧错位一帧，见文末"更正"一节；以 T02 返包为准。
- **nominal：** 每个档位 ±2 和 ±8 两个目录里的 nominal 轨迹逐位相同（`results_system1/T01b/schemeA/nominal_pm2_vs_pm8.csv`）。

## 文件

| 文件 | 内容 |
|---|---|
| `A/mass_sweep_episodes.csv` | 12 档 × 2 档扰动 × 51 回合 = 1224 行。列与 `results_system1/T01b/schemeA/episodes.csv` 相同，另加 `delta_cm` 和 `seed`。`delta_cm` 是 T02 方案 B1 的接触偏移 δ，A 不适用，为空；2026-11-10 更正：之前填的是扰动档，与 `perturbation_level_cm` 重复。`seed`：A 不训练，为空。`retrained` 列填 `not_applicable_no_training` |
| `A/mass_sweep_summary.csv` | 每组 50 个扰动回合的汇总和 nominal |
| `A/object_physics_readback.csv` | 12 档 PhysX 读回值与 URDF 值 |

- 配置和 diff：每组评测的完整命令、起止时间、git 信息，在 `results_system1/T01b/schemeA/run_configs.json`。
- 与同任务原质量主设置的 run_manifest 逐字段 diff，在 `results_system1/T01b/schemeA/manifest_diffs_vs_main.json`。只有物体 URDF 路径、读回质量和 `system1_object_mass_variant` 下的字段不同。
- 生成脚本：`system1/package_scheme_a.py`、`system1/t04/readback_a.sh`、`system1/t04/readback_table.py`。

## 与任务书的偏离

1. **惯量读回是事后补的。** 任务书要求"每个档位开训或评测前读回实际质量和惯量，写进 manifest"。评测时 manifest 里只有读回的质量，惯量是 URDF 写入值。
   - 惯量读回是在评测结束后，用同一评测路径单独做的；评测没有重跑。
   - 现在评测脚本的 manifest 已同时记录惯量和质心。
   - 训练脚本本来就把读回的质量、惯量和质心写进 run_config，这一点 T04-D 和 T03 的 F6 预检已经核对过。
2. **"额外评测"没有做。** 即把原质量的 D 和 C 直接放到其他档位上评测，不重训；轻档位用原质量 D 评测，标"未重训"。
   - 原因：T01b 规定本机 GPU 只做 SONIC 相关的评测。
   - 这部分还没有做。

## 更正（2026-11-11，Kevin 决定）

- **错在哪里：** "结果"一节里"失败回合在终止帧上的物体误差"的分类，以及 `A/mass_sweep_summary.csv` 的 `n_fail_terminal_*` 三列，都用错了参考帧。
  - 用的是轨迹同一行的参考帧，即 `reference_frame`，等于执行步数。
  - 终止判定 `JointBadTrackingZOnly` 用的是 `reward_reference_frame`，即 `command.time_steps`，比同一行早一帧。
  - 所以这个分类错位一帧。这与 T01b 第 15 节的更正是同一个问题。
- **按终止判定的对齐方式重算：** 用已有轨迹，不重跑仿真。
  - 重算后，A 质量扫描 18 组非原质量档的每个失败回合，都能由物体位置、物体姿态、机器人躯干三项中的一项解释。

  | 任务 | 质量 | 档 | 原表（物体位置 / 物体姿态 / 两项都未超） | 重算 |
  |---|---|---|---|---|
  | Bin | 4 kg | ±8 | 35 / 15 / 1 | 36 / 15 / 0 |
  | Bin | 8 kg | ±8 | 50 / 1 / 0 | 50 / 0 / 0 |

  - 其余 16 组的三个数不变。
- **不成立的两句：**
  1. "4 kg ±8 有 1 个回合两项都未超"：重算后这个回合是物体位置超限，两项都未超的回合为 0。
  2. "±8 有 50 个位置超限，其中 1 个同时姿态超限"（Bin 8 kg）：重算后 50 个都是位置超限，没有同时姿态超限的回合。
- **不受影响的部分：** 完成数、进度、终止步和其他各列。
- **原数据文件不改：** `A/mass_sweep_episodes.csv`、`A/mass_sweep_summary.csv`、`mass_sweep_episodes.csv`、`mass_sweep_summary.csv` 保持原样。
- **以 T02 返包为准：** `results_system1/T02/object_error_check/` 里的终止原因表。
  - 这些表统一用终止判定的对齐方式，A 质量扫描各档会随 B 质量扫描一起写入。
