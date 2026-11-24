# T10c：C-out 的 MuJoCo sim2sim（T13 第 5 节 C1）

- **依据：**`docs/system1/T13_parallel_runs_zh.md` 第 5 节；Kevin 2026-11-23 同意用 5070 做 3 个 Isaac 场景导出（T14 第 8 节第 4 条）。
- **时间：**2026-11-23 约 21:25–21:55（纽约时间），只评测，不训练。
- **对象：**C-out 三个最终 checkpoint（T03 c2_main，种子 721），与 Isaac 5070 最终评测逐个核对 sha256：
  - Table 5 kg：`cout_table_m5kg_seed721_fresh12000/model_12000.pt`，`6b86e1d5…`
  - Bin 1 kg：`cout_bucket_m1kg_seed721_fresh12000/model_12000.pt`，`66b52523…`
  - Pull 20 kg：`cout_pull_m20kg_seed721_joint8000/model_08050.pt`，`64dc72ba…`
- **硬件：**
  - MuJoCo 和 ONNX 都在 CPU 上：环境 `sim2sim_s1`，**MuJoCo 3.3.7**（和 T10 / T10s 同一个环境）。
  - Isaac 导出在本机 RTX 5070 上，环境 `hssim_s1`。
  - 没有开云端。

## 结果：闭环完成数（每档 50 个扰动状态，nominal 单独列）

| 任务 | 档位 | Isaac 5070 最终评测 | MuJoCo | MuJoCo − Isaac | nominal 完成 Isaac / MuJoCo（步数） |
|---|---|---|---|---|---|
| Pull 20 kg | ±2 | 50 | 50 | 0 | True / True（316 / 316） |
| Pull 20 kg | ±8 | 38 | 33 | −5 | True / True（316 / 316） |
| Table 5 kg | ±2 | 41 | 11 | −30 | True / False（687 / 408） |
| Table 5 kg | ±8 | 27 | 10 | −17 | True / False（687 / 408） |
| Bin 1 kg | ±2 | 1 | 0 | −1 | False / False（399 / 381） |
| Bin 1 kg | ±8 | 2 | 2 | 0 | False / False（399 / 381） |

- **Isaac 列：**从 `results_system1/T03/c2_main/per_seed_summary.csv` 读出。另外用 Isaac 每回合的 `summary.json` 重数了一遍，结果一致（打包脚本里做了断言）。
- **和 Isaac 共用：**checkpoint、冻结状态文件（sha256 见下）、协议（确定性动作、时序、终止条件、"完成"的定义），全部和 T10 相同。
- **没有调参：**没有改任何设置去提高 MuJoCo 的结果。

**终止步和终止原因（扰动回合里没完成的，`isaac_vs_mujoco.csv`、`episodes.csv`）：**

| 任务 | 档位 | 模拟器 | 终止步 中位数（最小–最大） | < 50 步终止 | 终止原因（一个回合可同时满足多项） | 失败分类（原因@终止时的参考阶段） | 物体误差首次 > 0.1 m 的步（中位数） |
|---|---|---|---|---|---|---|---|
| Table | ±2 | Isaac | 338（304–474） | 0 | 物体位置 5，物体朝向 4 | 位置@举起中 5，朝向@举起中 4 | 241 |
| Table | ±2 | MuJoCo | 444（402–498） | 0 | 物体位置 39 | 位置@举起中 39 | 308 |
| Table | ±8 | Isaac | 309（250–459） | 0 | 物体位置 14，物体朝向 9 | 位置@举起中 14，朝向@举起中 9 | 240.5 |
| Table | ±8 | MuJoCo | 410（296–469） | 0 | 物体位置 39，物体朝向 1 | 位置@举起中 39，朝向@举起中 1 | 296.5 |
| Bin | ±2 | Isaac | 401（366–473） | 0 | 物体朝向 34，物体位置 16 | 朝向@翻转 33，位置@翻转 16 | 202 |
| Bin | ±2 | MuJoCo | 396（366–458） | 0 | 物体位置 49，物体朝向 2 | 位置@翻转 49，朝向@翻转 1 | 203 |
| Bin | ±8 | Isaac | 402（21–482） | 3 | 物体朝向 28，物体位置 20 | 朝向@翻转 25，位置@翻转 19，其余 4 | 202 |
| Bin | ±8 | MuJoCo | 378.5（8–476） | 11 | 物体位置 37，物体朝向 12 | 位置@翻转 28，位置@静止 9，朝向@翻转 7，其余 4 | 202 |
| Pull | ±8 | Isaac | 192（156–197） | 0 | 物体位置 12 | 位置@抬起 8，位置@贴地拉行 4 | 137 |
| Pull | ±8 | MuJoCo | 194（155–218） | 0 | 物体位置 17 | 位置@抬起 11，位置@贴地拉行 5，位置@放下 1 | 140 |

- **两边都没有出现躯干倾斜**（重力 z 差 > 0.8）导致的终止。
- **Isaac 侧的终止原因：**按保存的轨迹重算，用最后一行的物体状态，对照终止判定时用的那一帧参考（即上一行的参考）。
  - 每个 Isaac 失败回合都至少满足一项物体条件，所以没有用到"无物体条件、推定为躯干倾斜"这一类。
- **MuJoCo 侧的终止原因：**取评测器记录的最后一次终止判定。
- **参考阶段：**按参考物体的运动标注（`system1/t10b/phases.py`，中文名见 `phase_labels_zh.json`）。
- **参考长度：**Table 687 步，Bin 497 步，Pull 316 步。
- **MuJoCo 每组用时：**Table 311 s / 294 s，Bin 329 s / 266 s，Pull 191 s / 138 s（±2 / ±8，单线程）。

## 一致性检查（`consistency/`，阈值同 T10 / T10s）

| 检查 | 范围 | 结果 | 阈值 / 规则 | 判定 |
|---|---|---|---|---|
| 初始状态 | 3 任务 × ±2 / ±8 冻结状态文件，306 个状态，对照 C-out Isaac 最终评测重置后的实际状态 | 关节 0；根位置最大 1.18e-7 m；根朝向 7.9e-8 rad；物体位置 0 | 1e-4 | 通过 |
| 158 维观测 | 每个任务 1 个 Isaac nominal 回合（Table 687 步、Bin 399 步、Pull 316 步） | 最大差：Table 8.8e-7，Bin 1.47e-6，Pull 8.1e-7 | 1e-4 | 通过 |
| SONIC 输入：930 维本体感觉历史 / 1761 维 tokenizer | 同上；C-out 用未修改的参考，SONIC 的上一动作是它自己的输出 | 最大 6.0e-7 / 2.5e-7 | 1e-4 | 通过 |
| q* 逐步对照 | 同上；q* = SONIC ONNX 目标 + 论文 D 缩放 × clip(a)，对照 Isaac 每步记录的 PD 目标 | a 用 Isaac 记录值：Table 2.7e-6、Bin 3.0e-6、Pull 2.2e-6 rad；a 用 MuJoCo 观测 + actor ONNX：2.8e-6 / 3.4e-6 / 2.5e-6 rad | 1e-4 | 通过 |
| SONIC 动作 ONNX 对 Isaac GPU | 同上 | 最大 6.3e-6 / 9.5e-6 / 8.6e-6 | 只报告（同 T10） | — |
| actor ONNX 在 Isaac 录到的观测上对 Isaac 动作 | 同上 | 3.1e-6 / 5.7e-6 / 3.8e-6 | 只报告（同 T10） | — |
| ONNX float64 图对 PyTorch float64 | 3 个 actor，1,000 个随机观测 | 最大 7.1e-14 | ≤ 1e-10（T10s 规则） | 通过 |
| ONNX float32 | 同上 | Table 1.22e-5，Bin 5.7e-6，Pull 3.05e-5。超出 1e-5 的部分都来自 N(0, 1) 那一半输入，按归一化采样的那一半 ≤ 4.8e-6。`onnx_check.csv` 里 Table、Pull 的 `passed=False` 指这一列 | 只报告（T10s 规则） | — |
| 开环 1 s | MuJoCo 逐步执行 Isaac 记录的 q*，50 步 | 见下表 | 只报告（同 T10） | — |

**开环 1 s 时的偏差：**

| 回合 | 根位置 r0 / r1 | 关节角 最大 / 平均 | 物体位置 |
|---|---|---|---|
| C-out Table | 14.7 / 13.2 cm | 0.216 / 0.050 rad | 5.9 cm |
| C-out Bin | 10.5 / 21.3 cm | 0.214 / 0.045 rad | 10.8 cm |
| C-out Pull | 5.9 / 13.5 cm | 0.323 / 0.054 rad | 0.16 cm |

## 做了什么

1. **代码**（新功能都加了开关，默认行为不变）：
   - `sonic_mj.COutController`：新类。
     - SONIC 收到未修改的参考；
     - q* = SONIC 输出（holosoma 顺序）+ 论文 D 动作缩放 × clip(a, ±action_clip_value)；
     - actor 的上一动作是原始 a；SONIC 的上一动作保持它自己的输出。
     - 和 `system1/scheme_c.py` 的 kind "out" 一致。
   - `mj_eval.py`：加 `--scheme C-out`。
     - 场景用 `logs_system1/t10/isaac/cout_<task>` 的导出；
     - PD 只允许 isaac；
     - 状态文件按 C-out 最终评测取，核对 sha256。
   - `run_isaac_dumps.py`：加 `COUT_LAUNCH`，只在 `--only` 里点名时运行，默认列表不变。
   - `check_observations.py`：`--runs cout_*` 时，核对 SONIC 输入，并逐步对照 q*。
   - `check_openloop.py`：`--runs cout_*`。
   - `check_initial_states.py`：`--scheme C-out`。
   - `run_mj_evals.py`：`--cout-tasks`，默认为空。默认任务列表仍是 36 组，已核对。
   - 新增打包脚本 `package_t10c.py`。
   - **改动以后仍和原来一致：**
     - T10 原有 smoke test 加 C-out 新测试，`system1/tests/test_sim2sim.py` 5 个全部通过；
     - 用改动后的脚本重跑 T10 C-in Table 的观测核对，`observations_summary.json` 和 T10 返包逐字相同。
2. **ONNX 导出：**21:30，`exports_system1/t10/onnx/cout_{table,bucket,pull}_seed721.onnx`，用已有的 `onnx_check.py --extra-checkpoint`。
3. **Isaac 导出：**
   - 3 个 nominal 回合，argv 和 C-out 最终评测完全相同，只加了 `--limit 1 --sim2sim-dump`。
   - 在 tmux 里一次一个，21:36:52–21:40:42，三个都 rc 0。
   - 单个进程 RSS 约 5.5 GB，GPU 显存约 3.2 GB。
   - 开跑前 `nvidia-smi` 显示没有其他计算进程。
4. **一致性检查：**21:45:28–21:46:40。
5. **闭环评测：**
   - 6 组，每组 51 个回合（nominal 加 50 个扰动），21:39:41–21:50:54。
   - 在 tmux 里跑，同时最多 3 个单线程进程，每个都 rc 0、complete。
6. **打包：**`package_t10c.py`。

## 和 T13 第 5 节、T10 的偏离和需要说明的事

- **Isaac 导出的时机：**T13 写的是"等 T14 第一个停点的统计跑完、内存有余量后再开"，并且列为 CPU 任务。Isaac 导出用了 5070，这一点 Kevin 已在 T14 第 8 节第 4 条同意。
- **C-out 导出比 T10 的 C-in 导出多两样东西，都是 T10s 加的记录项：**
  - 多存了 `runtime_reference.npz`，即 Isaac 实际采样的参考。所以 Table、Bin 的 MuJoCo 端按 T10s 的方式直接用它。T10 的 C-in Table、Bin 当时用的是 holosoma 的参考加载器。观测核对里参考相关的维度和 Isaac 差 ≤ 1e-6。
  - `scene.json` 里多了 `physics_materials`、`scene_default_material`、`runtime_reference` 三个记录字段。
  - PD、力矩上限、动作缩放、控制配置这几项和 C-in 导出逐项相同。用两份导出编译出的 MJCF 场景逐字相同。
  - 和 T10 C-in 的 run manifest 逐项比较见 `configs/run_manifest_vs_t10_cin.json`：kp、kd、力矩上限、子步、凸包、状态文件、SONIC ONNX 都相同，只有 checkpoint、ONNX、方案名、导出路径不同。
- **MuJoCo 版本：**闭环和检查都用 `sim2sim_s1` 的 MuJoCo 3.3.7，物理接触走 `mj_step` 的碰撞流程。没有用到 `mj_geomDistance`，主线程在 hsretargeting 的 MuJoCo 3.10.0 里发现的问题出在这个函数上。
- **ONNX float32：**Table、Pull 的 actor 在随机输入的 N(0, 1) 那一半上超过 1e-5（最大 3.05e-5）。按 T13 "ONNX 用 float64 判定"，以 float64 的结果为准。
- **结果和 Isaac 差得多的组：**Table ±2（41 → 11）和 ±8（27 → 10）。按 CLAUDE.md 如实报告，没有改设置。

## 文件

| 文件 | 内容 |
|---|---|
| `isaac_vs_mujoco.csv` | 每个任务 × 档位一行：两边的完成数、nominal、终止步统计、终止原因、失败分类、物体发散步、sha256 |
| `episodes.csv` | 每回合一行，Isaac 和 MuJoCo 各 306 行 |
| `consistency/` | 初始状态、观测 / SONIC 输入 / q*、ONNX、开环 1 s（CSV、JSON、PNG） |
| `configs/` | MuJoCo run manifest、launch / exit 记录（gzip，含当时的 git HEAD / status / diff）；Isaac 导出的 launch / exit / scene.json；编译后的 MJCF；`run_manifest_vs_t10_cin.json`；`code_diff_vs_HEAD.patch`；`git_head.txt`；`commands.txt`；runner 和检查日志 |
| `provenance.json` | checkpoint、ONNX、状态文件、Isaac 导出、MJCF、URDF 的 sha256 |
| `phase_labels_zh.json` | 参考阶段名 |

**冻结状态文件 sha256（前 12 位）：**

| 任务 | ±2 | ±8 |
|---|---|---|
| Table | `389e8da5fb81` | `6a626f48c145` |
| Bin | `631026d39f43` | `3bae05a1b1d7` |
| Pull | `473c1fc05632` | `3c2a60c7982d` |

**MuJoCo 输出：**`logs_system1/t10c/mj_eval/`（不进 git）；Isaac 导出：`logs_system1/t10/isaac/cout_*`；ONNX：`exports_system1/t10/onnx/cout_*`。
