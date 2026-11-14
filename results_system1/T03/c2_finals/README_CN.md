# T03 最终 checkpoint 对比表：本机 RTX 5070

依据：Kevin 2026-11-12 第 4 项（C 主实验 Pull 两个 run 的最终 ±2、±8 完成率，与 D 同任务最终结果并排），以及"论文最终对比表全部用本机 5070 的结果"。

## 做了什么

- 本机流水线（`system1/t03/post_train.py`）在本机 RTX 5070 Laptop 上评每个 run 的最终 checkpoint，±2、±8 各一次。
- 本表由 `system1/t03/finals_table.py` 从已有输出汇总，只读、CPU 运行，没有新增计算。
- **C**：C v2 主实验（C-in、C-out 种子 721）和 C-in 补种子（722、723）。
  - Pull 取第 8050 轮，Table、Bin 取第 12000 轮。
- **D**：论文主结果的最终 checkpoint，种子 721–723。
  - ±2 来自流水线 d_curve 对最终 checkpoint 的评测，±8 来自 d_paper_final。
- 两边的冻结状态文件 sha256 相同，每行都记了 sha256。

## 当前结果（2026-11-13 21:2x，C 全部 run 的最终评测已完成）

原质量的 C 和 D，每格是 50 个扰动状态里的完成数。各重训质量档的行也在 `finals.csv` 里（`mass_kg`、`retrained` 两列），汇总见 `results_system1/T04/retrained/`。

**Pull（20 kg，第 8050 轮）**

| 方案 | 种子 | ±2 完成 | ±8 完成 | ±8 未完成的终止项 |
|---|---|---|---|---|
| C-in | 721 | 50 | 22 | joint_bad_tracking 28 |
| C-in | 722 | 50 | 38 | joint_bad_tracking 12 |
| C-in | 723 | 18 | 14 | joint_bad_tracking 36 |
| C-out | 721 | 50 | 38 | joint_bad_tracking 12 |
| D | 721 | 50 | 43 | joint_bad_tracking 7 |
| D | 722 | 50 | 28 | joint_bad_tracking 22 |
| D | 723 | 49 | 41 | joint_bad_tracking 9 |

**Table（5 kg，第 12000 轮）**

| 方案 | 种子 | ±2 完成 | ±8 完成 | ±8 未完成的终止项 |
|---|---|---|---|---|
| C-in | 721 | 31 | 24 | joint_bad_tracking 26 |
| C-in | 722 | 48 | 41 | joint_bad_tracking 9 |
| C-in | 723 | 33 | 38 | joint_bad_tracking 12 |
| C-out | 721 | 41 | 27 | joint_bad_tracking 23 |
| D | 721 | 35 | 45 | joint_bad_tracking 5 |
| D | 722 | 49 | 37 | joint_bad_tracking 13 |
| D | 723 | 46 | 40 | joint_bad_tracking 10 |

**Bin（1 kg，第 12000 轮）**

| 方案 | 种子 | ±2 完成 | ±8 完成 | ±8 未完成的终止项 |
|---|---|---|---|---|
| C-in | 721 | 49 | 30 | joint_bad_tracking 20 |
| C-in | 722 | 43 | 28 | joint_bad_tracking 22 |
| C-in | 723 | 43 | 29 | joint_bad_tracking 21 |
| C-out | 721 | 1 | 2 | joint_bad_tracking 48 |
| D | 721 | 50 | 39 | joint_bad_tracking 11 |
| D | 722 | 32 | 23 | joint_bad_tracking 27 |
| D | 723 | 45 | 31 | joint_bad_tracking 19 |

**没有完成的 nominal 回合**（同一 run 的 ±2、±8 用同一个 nominal 状态，两次结果相同）：

| run | nominal 步数 |
|---|---|
| C-in Pull 723 | 161 / 316 |
| C-in Table 722 | 336 / 687 |
| C-in Table 723 | 398 / 687 |
| C-out Bin 721 | 399 / 497 |
| D Bin 722 | 313 / 497 |
| D Bin 723 | 324 / 497 |

其余 run 的 nominal 都完成。

C-in Table 722 的最终 checkpoint（`model_12000.pt`，sha256 b1fad2fa…ed5d）2026-11-13 17:29 从云端取回，和云端训练目录里的文件 sha256 一致；本机 ±2、±8 在 17:29–18:45 评完。

## 文件

- `finals.csv`：每行一个评测，字段如下。
  - 结果：完成数、平均进度、nominal 回合、未完成回合的终止项和步数范围、未完成回合里 SONIC 摔倒的个数；
  - 溯源：GPU、checkpoint 的 sha256、状态文件的 sha256、summary.json 的 sha256、输出目录。
- `SHA256SUMS`：本目录各文件的 sha256。

命令：

    hssim_s1/bin/python -B system1/t03/finals_table.py --out results_system1/T03/c2_finals

## 和任务书的偏离

没有。
