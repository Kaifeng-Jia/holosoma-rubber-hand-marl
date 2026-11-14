# 论文图表草图（橡胶手版本）

依据：`docs/system1/paper_results_plan_zh.md` 第三节，第三次修订（不加灵巧手；正文放 Push、Pull、Table、Bin，Kick、Chair 放附录）。

## 做了什么

- `system1/render/paper_figs.py` 从已提交的 CSV 生成全部表和图，CPU 运行，没有手填的数字。缺的数据留空，并在表里写明原因，或在图上标出。
- **配色**（`system1/render/scheme_colors.py` 的 `PAPER_COLOR`）：A 蓝、B 黄（虚线）、C-in 绿、C-out 深绿（与 C-in 同一色系，明度不同）、D 橙；图例写全称。
  - 用 `system1/render/palette_check.py` 在白底上逐对检查（OKLab ΔE×100，含 Machado 2009 三种色盲模拟）：除 B–D 外都达标（正常视觉 ≥ 15，色盲 ≥ 8）。B–D 正常视觉 13.7，所以 B 画虚线。
  - C-in 在白底上的对比度只有 2.8:1，所以图都配了图例，每张图旁边都有 CSV 可以查原始数值。

## 文件

| 文件 | 内容 | 数据来源 | 状态 |
|---|---|---|---|
| `table1_main.csv` / `.md` | 表 1：方案 × 任务，±2 / ±8 完成率（多种子的给均值 ± 样本标准差，另列每个种子）、完成回合的物体误差、评测 GPU | A：T01b `schemeA/episodes.csv`；B：T02 `episodes.csv`（B0）；C-in、C-out、D：T03 `c2_main/per_seed_summary.csv` | 草图，有缺格，见下 |
| `table1_b_variants.csv` | B 的全部变体（B0、B1-0、B1-2、B1-4、B-open） | T02 | 完整：所有变体、所有任务都是 0 |
| `fig2_training_efficiency.png` / `.csv` | 图 2：完成率（±2）对环境步数，Pull、Table、Bin 三栏 | T03 `train_efficiency/curve_points.csv`（全部是云端 4090 的点） | 草图，缺 34 个点，见下 |
| `fig3_load_capacity.png` / `.csv` | 图 3：完成率（±2）对质量倍数，D 和 C-in（每档重训，种子 721）；标出物理上限和触发了规则但没训练的档位 | T04 `mass_sweep_summary.csv`、`physics_check/physics_check.csv` | 完整（单种子） |
| `fig5_summary_raw.csv` | 图 5 的原始数值：每个方案在五个维度上的值 | 表 1、T03 训练效率、T04 | 训练效率一列暂缺 |
| `fig5_summary_radar.png` | 图 5 雷达图草图 | 同上 | 草图 |

图 4（为什么不学习就不行：Pull 关键帧，A 与 D 并排，加俯视轨迹）沿用已有的 `results_system1/T01b/figures/keyframes_pull.png`、`topdown_pull.png`，这次没有重画。图 1 是示意图，不来自数据，没做。

## 缺的数据

- **表 1：**
  - D 在 Push 上没有本机 5070 的评测，Kick、Chair（附录）也没有，那几格留空。V3 收尾时在云端 4090 上评过，按"主对比全部在 5070 上评"的规定，不混进来。
  - C 只做了 Pull、Table、Bin（方案规定）。
  - B 不适用于 Kick：规划器会去掉踢腿动作（T02）。
- **图 2：**
  - C 在 Table、Bin 上第 9000–12000 轮的 28 个点，正在云端补评（2026-11-13 22:15 开始）。
  - D Bin 种子 721 第 0–10000 轮的 6 个点从没在 4090 上评过，要不要补等 Kevin 决定。
  - 图里没画的点，在对应 checkpoint 处用竖向细虚线标出，均值和范围只在种子齐全的点上画。
- **图 5：** 训练效率（AUC）要等图 2 的点补齐。C-out 没有承重实验，所以承重边界这一维没有值。

## 图 5 的归一化（写进图注）

| 维度 | 原始值 | 归一化 |
|---|---|---|
| 完成率 ±2 | Pull、Table、Bin 三个任务的平均完成率（多种子先取种子平均） | ÷ 100 |
| 完成率 ±8 | 同上，±8 档 | ÷ 100 |
| 训练效率 | Pull、Table、Bin 各 run 的曲线下面积（迭代跨度内的平均完成率）取平均 | ÷ 100 |
| 承重边界 | 每个任务 ±2 完成率 ≥ 50% 的最大质量倍数，三个任务取平均 | ÷ 8（训练过的最大倍数） |
| 不需要训练 | A、B = 1，C、D = 0 | — |

- 雷达图会让人去比面积，这容易误导，所以 `fig5_summary_raw.csv` 同时给出原始数字（计划放附录）。
- 某个方案在某一维上没有值时（缺数据或不适用），这一维在图上是断开的。

生成命令：

    hssim_s1/bin/python -B system1/render/paper_figs.py
