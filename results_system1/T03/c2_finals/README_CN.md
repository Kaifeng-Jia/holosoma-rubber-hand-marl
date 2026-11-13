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

## 当前结果（2026-11-12 19:4x）

**Pull**（50 个扰动状态）

| 方案 | 种子 | ±2 完成 | ±8 完成 | ±8 未完成的终止原因 |
|---|---|---|---|---|
| C-in | 721 | 50 | 22 | joint_bad_tracking 28 个，第 150–172 步 |
| C-out | 721 | 50 | 38 | joint_bad_tracking 12 个，第 156–197 步 |
| D | 721 | 50 | 43 | joint_bad_tracking 7 个 |
| D | 722 | 50 | 28 | joint_bad_tracking 22 个 |
| D | 723 | 49 | 41 | joint_bad_tracking 9 个 |

- 所有 nominal 回合都完成，各 316 步。
- C 两行未完成的回合里，SONIC 都没有摔倒。

**Table、Bin**
- C 的这两个任务还在训练，表里记为"未完成，没有 summary.json"。
- D 的数已在表里。

## 文件

- `finals.csv`：每行一个评测，字段如下。
  - 结果：完成数、平均进度、nominal 回合、未完成回合的终止项和步数范围、未完成回合里 SONIC 摔倒的个数；
  - 溯源：GPU、checkpoint 的 sha256、状态文件的 sha256、summary.json 的 sha256、输出目录。
- `SHA256SUMS`：本目录各文件的 sha256。

命令：

    hssim_s1/bin/python -B system1/t03/finals_table.py --out results_system1/T03/c2_finals

## 和任务书的偏离

没有。C 其余 run 训完、本机评完后，重跑上面的命令更新本表。
