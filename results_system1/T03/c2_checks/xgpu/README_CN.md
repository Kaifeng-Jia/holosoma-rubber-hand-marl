# T03 跨 GPU 对照：本机 RTX 5070 Laptop 与云端 RTX 4090

依据：Kevin 2026-11-12 第 1 项。D 论文主结果种子 721 的 Pull 第 1050 轮、Bin 最终 checkpoint（第 12000 轮）在云端 4090 上各跑一次 ±2，与本机 5070 已有结果逐状态对比，统计翻转数和 McNemar 检验，并与 C-in 的跨 GPU 结果（核对 3a）放在同一张表。

## 做了什么

- **云端 D 评测**：2 个，排在云端评测队列最前面（`system1/cloud/c2_scheduler.py`，`xgpu/` 任务），依次在 GPU 2 上跑。两次启动前，这张卡上都在跑 2 个训练：C-in Bin 2 kg 种子 721，C-in Bin 1 kg 种子 723。

  | 任务 | 开始 | 结束 | 用时 |
  |---|---|---|---|
  | D Pull 第 1050 轮 | 2026-11-12 03:46 | 04:38 | 52 分钟 |
  | D Bin 第 12000 轮 | 2026-11-12 04:38 | 06:09 | 91 分钟 |

- **本机对照**：
  - D 两个用流水线 d_curve 已有的结果，2026-11-11 17:07–17:56 跑的，每个 checkpoint 只评过一次。
  - C-in 两个用核对 3a 的本机复核结果。
- **对比脚本** `system1/t03/xgpu_compare.py`，CPU 运行，只读已有评测输出：
  - 输入每个 checkpoint 两边的 `summary.json`；
  - 50 个扰动状态逐个对比，nominal 单列。

## 口径

- 评测器 `scripts/evaluate_paper_supplement.py`，两边相同，±2 冻结状态集。
  - 四对各自的 checkpoint sha256 两边相同，初始状态文件 sha256 两边相同，脚本里有断言。
  - 四对两边用的 git 提交里，评测器的 sha256 都是 `7c4e2714dec4…`（见 `provenance.csv`）。
- **完成**：评测器的 `completed_reference`，即所有参考帧都执行完，且 joint_bad_tracking 没有触发。
- **翻转数**：只在本机完成的状态数，加只在云端完成的状态数。
- **McNemar 检验**：精确双侧，在翻转上做二项检验，p = 0.5。
- **相同执行步数**：两边执行步数完全相同的状态数。

## 结果（50 个扰动状态）

| 方案 | 任务 | 轮次 | 5070 完成 | 4090 完成 | 两边都完成 | 只本机 | 只云端 | 都没完成 | 翻转数 | McNemar p | 相同执行步数 | nominal（5070 / 4090） |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-in | Pull | 1050 | 43 | 41 | 34 | 9 | 7 | 0 | 16 | 0.804 | 34 | 完成 / 完成 |
| C-in | Bin | 1000 | 34 | 39 | 26 | 8 | 13 | 3 | 21 | 0.383 | 26 | 第 393 步终止 / 完成 |
| D | Pull | 1050 | 35 | 44 | 32 | 3 | 12 | 3 | 15 | 0.035 | 32 | 完成 / 完成 |
| D | Bin | 12000 | 50 | 50 | 50 | 0 | 0 | 0 | 0 | 1.000 | 50 | 完成 / 完成 |

- 前三行里，两边执行步数相同的状态数，正好等于两边都完成的状态数。也就是说，凡是至少一边提前终止的状态，两边的执行步数都不一样。
- D Pull 第 1050 轮的 15 个翻转，终止的那一边都是物体位置误差超过 0.25 m（终止时误差 0.250–0.270 m）。
- 平均进度：

  | 方案 / 任务 | 5070 | 4090 |
  |---|---|---|
  | C-in Pull | 0.937 | 0.928 |
  | C-in Bin | 0.882 | 0.917 |
  | D Pull | 0.862 | 0.936 |
  | D Bin | 1.000 | 1.000 |

## 配置核对（`manifest_diff.csv`）

- 两边 `run_manifest.json` 逐项 diff 后，不同的只有：
  - checkpoint 路径；
  - GPU 名；
  - model_id；
  - SONIC spec 路径（只有 C-in 有这一项；spec 的 sha256 两边相同）。
- 命令行（`provenance.csv`）：本机多了 `--device cuda:0` 和 taskset；云端用 `CUDA_VISIBLE_DEVICES` 指定卡。
- 启动时的工作区：
  - 云端两边干净；
  - 本机 D 两个干净；
  - 本机 C-in 两个只有一个未跟踪目录 `results_system1/T03/c2_checks/`，不涉及代码。

## 和任务书的偏离

- 没有偏离。
- 本机的 D 结果用的是已有结果，没有为这张表重跑。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.csv` | 上表 |
| `per_state.csv` | 每个状态在两边的完成情况、执行步数、进度、物体 RMSE、终止时物体位置误差 |
| `provenance.csv` | checkpoint 和初始状态文件的 sha256、两边 GPU、git HEAD、评测器 sha256、完整命令、summary.json 的 sha256、目录 |
| `manifest_diff.csv` | 两边 run_manifest 的逐项差异 |
| `SHA256SUMS` | 本目录各文件的 sha256 |

命令：

    hssim_s1/bin/python -B system1/t03/xgpu_compare.py --out results_system1/T03/c2_checks/xgpu
