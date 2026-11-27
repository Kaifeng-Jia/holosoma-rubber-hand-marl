# T13 §9 续训到计数 10000：准备和本机续训冒烟

- **依据：**
  - T13 任务书第 9 节（Kevin 2026-11-26，固定训练预算）："续训要恢复迭代计数、优化器、观测归一化和随机化进度；续训后前 10 轮的 loss、奖励和续训前最后 10 轮量级对得上，对不上先停"；
  - Kevin 2026-11-27："现在就接"，199 / 196 / 210 共用一个队列。
- **协议改为固定训练预算：**之前按停止规则停下的 checkpoint 和评测留作记录，不作正式结果。正式结果只用计数第 10000 轮。
- **三个续训 run**（`cloud_runs.py` 的 `CONT`；c_out_table 已在 10000，不动）：

  | run | 起点 checkpoint（本机） | 计数起点 | 还要训的轮数 | 结束文件 |
  |---|---|---|---|---|
  | d_push_to10k | `runs/d_push/train/model_06025.pt` | 6000（预热 25） | 4000 | model_10025.pt |
  | d_table_to10k | `runs/d_table/train/model_07500.pt` | 7500 | 2500 | model_10000.pt |
  | c_out_push_to10k | `runs/c_out_push/train/model_09025.pt` | 9000（预热 25） | 1000 | model_10025.pt |

## 1. 续训恢复什么

- **训练脚本的 `--resume` 严格加载：**
  - actor、critic、两个优化器（含学习率）、actor / critic 观测归一化、迭代计数（`load_training_state_dict`，strict）；
  - checkpoint 文件编号接着原来的编号。
- **"随机化进度"：**T10b 随机化里没有随迭代变化的量。
  - 开局按环境抽一次（材料、质量、重心、关节偏置），种子同为 721；
  - 每次重置抽 PD 增益；
  - 推力按每个环境自己的时间表（1–3 s）。
  - 所以续训开局重新抽的分布和原训练相同，不需要另外恢复。
- **Push 路径的 joint-acceleration ramp：**这三个 run 的命令里都没有 `--joint-acceleration-weight`，不启用。
- **配置不变：**`plan_diff.json` 里，4096 个环境下原 run 和续训 run 的训练命令逐项比较，只差 `--iterations`、`--output-dir`、`--resume` 三项。续训没有预热阶段。
- **验证：**照常每 1000 轮（计数）一次，只画曲线，不用来停。

## 2. 续训核对的判法（第一次续训之前定下）

- **续训前窗口：**到起点为止的 10 轮，取自本机拷回的 `metrics.jsonl`，存在 `<run>/continue_ref.json`。
- **续训后窗口：**续训的前 10 轮。
- **必须全部满足，否则按 PID 停训练（SIGINT），不评测，等 Kevin：**
  - 续训第一轮的编号 = 起点 + 1，run_config 里的 start_iteration = 起点；
  - 续训用的 checkpoint 的 sha256 = 本机这份的 sha256。
    - Table 路径取 run_config 里记的值；Push 路径（`train_plan5_push.py`）的 run_config 不记这个值，所以由 `cloud_runs.py` 在开训前算传给 `--resume` 的文件；
    - sha256 不对时 `cloud_runs.py` 直接拒绝开训。
  - critic 学习率相等（三个 run 都恒为 0.001）；
  - 以下 6 个量的"续训后 10 轮均值 / 续训前 10 轮均值"在 [1/3, 3] 之间：reward_mean、value_loss、|surrogate_loss|、entropy、每个环境的重置次数、actor 学习率（随 KL 自适应，每轮都变）。
- **门槛 [1/3, 3] 的依据：**原三个 run 最后 1000 轮里相邻 10 轮窗口之比，这些量最大在 0.37–2.7 倍之间（c_out_push 的 value_loss 和 actor 学习率波动最大）。
- **判法定下之后改过两处，都在任何云端续训之前：**
  1. 原来要求 actor 学习率逐轮相等。后来发现 actor 学习率随 KL 每轮变化，这一项一定不会相等，改成上面 10 轮均值之比。这次改动在三个本机冒烟开始之前。
  2. Push 路径的 sha256 来源，见上。原判法在 Push 路径上取不到这个值，冒烟第一次核对因此判为不过（d_push、c_out_push）。修正后用现有输出重新核对，第一次的记录留在 `smoke_first_check.json`。门槛没有改。

## 3. 本机续训冒烟（2026-11-27 02:08–02:12，5070）

- **做法：**云端续训命令原样，只改三处：环境数（D 2048；C-out 512，和第二轮 C-out 本机冒烟相同）、12 轮、输出目录。然后用第 2 节的判法核对。
- **结果**（比值 = 续训后 / 续训前）：

  | run | rc | 判定 | reward | value_loss | \|surrogate\| | entropy | 重置 / 环境 | actor 学习率 |
  |---|---|---|---|---|---|---|---|---|
  | d_push_to10k | 0 | 通过 | 1.114 | 0.902 | 1.249 | 0.994 | 1.052 | 0.537 |
  | d_table_to10k | 0 | 通过 | 1.025 | 0.833 | 1.192 | 0.996 | 0.686 | 0.560 |
  | c_out_push_to10k | 0 | **不通过** | 1.064 | 1.423 | 1.484 | 0.998 | 0.948 | **0.254** |

  三个 run 的编号、start_iteration、sha256、critic 学习率四项都满足。
- **c_out_push_to10k 只差 actor 学习率一项：**
  - 12 轮里 actor 学习率从 2.6e-4 一路降下来，10 轮均值是续训前的 0.254 倍，低于 1/3；
  - 原 run 最后 1000 轮里，这个量相邻窗口之比最小是 0.368，单轮最低到过 1e-5；
  - 本机冒烟只用 512 个环境，云端是 4096 个。
- **处理：**没有改判法，也没有重跑。c_out_push_to10k 在队列里标为"未准备好"，等 Kevin 决定。
- **文件：**每个 run 一个目录：`continue_ref.json`、`plan_diff.json`、`smoke.json`、`smoke_first_check.json`、`smoke_run_config.json`、`smoke_metrics.jsonl`（前 12 轮）。
- **脚本：**
  - `system1/t13v4/continue_prep.py`：参考窗口、plan 对比、本机冒烟、重新核对；
  - `cloud_runs.py`：`CONT`、`continue_check`；
  - `test_cloud_runs_continue.py`：用假的训练程序测续训控制逻辑，4 个场景都通过：Table 续训、Push 续训、reward ×10 时自动停、sha256 不对时拒绝开训。

## 4. 机器队列（`system1/t13v4/queue_s9.py`，tmux queue_s9，02:15:52 启动）

- **顺序：**d_frontback → d_push_to10k → d_table_to10k → c_out_push_to10k。每连上一台，取最前面、已经准备好的任务。
- **开训前：**
  - 云端代码 = 本机 HEAD；
  - C-out 任务核对 SONIC 权重的 sha256，不拷权重；
  - GPU 空闲；
  - 续训任务上传 checkpoint 和 `continue_ref.json`，核对 sha256；
  - d_frontback 先在云端跑 10 轮冒烟。
- **训练结束后：**
  - post_run 拷回、核对通过后写 `queue_release.json`，不关机；
  - 队列里还有任务，就在同一台接着开下一个；
  - 队列空了，按 T13 原规则关机：除平台进程外没有别的进程、没有 GPU 进程，查 3 次，间隔 60 s。
- **空等上限：**机器空着、队列里只剩没准备好的任务时，最多等 1 小时，之后按同样的规则关机。
- **状态：**哪个任务在哪台、几点开始、几点拷回，写在 `results_system1/T13_runs/queue_status.json`，有变化就提交。
- **不自己开机。**每 10 分钟试连 SSH 一次。
