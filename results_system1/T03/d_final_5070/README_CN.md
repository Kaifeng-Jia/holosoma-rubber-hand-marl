# T03 D 论文主结果在本机 5070 上的最终评测（Push、Kick、Chair）

依据：Kevin 2026-11-13 23:0x："本机 5070 补评 D：先评 Push（3 个种子 × ±2/±8），再评 Kick、Chair，都按最终评测的协议。"目的是让表 1 的 D 格全部来自本机 5070，和 A、B、C 同一张卡。

## 做了什么

- **启动器：** `system1/t03/run_d_finals_local.py`（提交 988d566）。
  - 18 个评测：Push、Kick、Chair × 种子 721、722、723 × ±2、±8。
  - 两个评测同时跑，CPU 分别绑 16–23 和 24–31。
  - 2026-11-13 23:15:58 开始，2026-11-14 03:30:59 全部结束，共约 4 小时 15 分钟。
  - 每对（±2 和 ±8 同时）用时：Push 约 22–28 分钟，Kick 约 26–32 分钟，Chair 约 33–34 分钟（见 `runner.log`）。
- **评测协议：** 和本机其他 D 最终评测相同。
  - `scripts/evaluate_paper_supplement.py --controller policy`，最终 checkpoint，`--device cuda:0`；
  - 确定性，TF32 关闭（评测程序默认）；
  - Push 另加 `--push-placement-proof system1/evidence/preflight_push_registered.json`（sha256 f28a5c09…），和所有 Push 评测一样。
- **checkpoint 和状态文件：** 和 V3 收尾包（`paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104/results/episodes.csv`，变体 Full）用的是同一批。
  - 启动前，每个 checkpoint 在本机按 V3 记录的 sha256 找到并核对，每个状态文件也按 V3 的 `state_file_sha256` 核对，全部一致。
  - 迭代数：Push 13050，Kick 8000，Chair 12000。

| 任务 | ±2 状态文件 | ±8 状态文件 |
|---|---|---|
| Push | U5 `push_states.json`（1799ea1e…） | V3 扩展评测 `push_pm8.json`（a58b2fae…） |
| Kick | U6 `kick_states.json`（c1ac57c6…） | `kick_pm8.json`（dba247ca…） |
| Chair | unified_evaluation `chair_states.json`（5bb8ff9f…） | `chair_pm8.json`（f9a144ed…） |

- **打包：** `system1/t03/package_d_final_5070.py`，只用 CPU、只读，列和 `results_system1/T03/c2_main/` 相同。
  - 手部位置误差：用 URDF 正运动学算，每个完成的回合都有。
  - 掌心误差：Chair 用 V3 掌心后处理；Kick 不适用；Push 没算，和 V3 收尾包、方案 A 返包一样。
- **评测程序：** 18 个评测启动时的 git HEAD 不全相同（988d566、604170a、f575e20），因为评测期间我有提交，但这几次提交都没改评测程序。
  - 评测程序在 18 个评测里的 sha256 都是 b7b38c5a…，按启动时的 HEAD 取。
  - 启动记录里另存了 `git_status` 和 `git_diff`。

## 结果（`scheme_summary.csv`、`per_seed_summary.csv`）

18 个评测都是 rc 0、51 回合、complete。完成率只算 50 个扰动状态，名义回合不计入（V3 规则）。

| 任务 | 档位 | 各种子完成率 %（721 / 722 / 723） | 均值 ± 样本标准差 |
|---|---|---|---|
| Push | ±2 | 78 / 100 / 74 | 84.0 ± 14.0 |
| Push | ±8 | 62 / 98 / 76 | 78.7 ± 18.1 |
| Kick | ±2 | 100 / 100 / 100 | 100.0 ± 0.0 |
| Kick | ±8 | 94 / 98 / 92 | 94.7 ± 3.1 |
| Chair | ±2 | 88 / 98 / 100 | 95.3 ± 6.4 |
| Chair | ±8 | 66 / 92 / 100 | 86.0 ± 17.8 |

- **名义回合：** Kick、Chair 的名义回合都完成了。
- Push 的名义回合 3 个种子都没完成：
  - 都是 `joint_bad_tracking` 终止，分别在第 87、89、93 步（参考共 309 步）；
  - V3 收尾包里同样是这 3 个步数、同样的终止原因。
- 每个评测的物体误差、手部误差、掌心误差（Chair），以及各自统计了多少回合，见 `per_seed_summary.csv`。

## 和 V3 收尾包的对照（`v3_vs_5070.csv`）

同一个 checkpoint、同一组冻结状态，V3 收尾包在云端评过一次（它的 episodes.csv 没有记 GPU 型号），这次在本机 5070 又评了一次。按 checkpoint 和状态文件的 sha256 配对，18 组全部配上。

| 任务 | 种子 | 档位 | V3 完成数 | 5070 完成数 | 结果相同的状态 | 只有 V3 完成 | 只有 5070 完成 |
|---|---|---|---|---|---|---|---|
| Push | 721 | ±2 | 37 | 39 | 32 | 8 | 10 |
| Push | 721 | ±8 | 34 | 31 | 27 | 13 | 10 |
| Push | 722 | ±2 | 48 | 50 | 48 | 0 | 2 |
| Push | 722 | ±8 | 47 | 49 | 48 | 0 | 2 |
| Push | 723 | ±2 | 32 | 37 | 33 | 6 | 11 |
| Push | 723 | ±8 | 35 | 38 | 31 | 8 | 11 |
| Kick | 721 | ±2 | 50 | 50 | 50 | 0 | 0 |
| Kick | 721 | ±8 | 49 | 47 | 46 | 3 | 1 |
| Kick | 722 | ±2 | 50 | 50 | 50 | 0 | 0 |
| Kick | 722 | ±8 | 50 | 49 | 49 | 1 | 0 |
| Kick | 723 | ±2 | 50 | 50 | 50 | 0 | 0 |
| Kick | 723 | ±8 | 48 | 46 | 48 | 2 | 0 |
| Chair | 721 | ±2 | 44 | 44 | 50 | 0 | 0 |
| Chair | 721 | ±8 | 32 | 33 | 31 | 9 | 10 |
| Chair | 722 | ±2 | 49 | 49 | 50 | 0 | 0 |
| Chair | 722 | ±8 | 41 | 46 | 41 | 2 | 7 |
| Chair | 723 | ±2 | 50 | 50 | 50 | 0 | 0 |
| Chair | 723 | ±8 | 50 | 50 | 50 | 0 | 0 |

- 按已有规定，表 1 只用本机 5070 的数，V3 的数不混进来。

## 文件

| 文件 | 内容 |
|---|---|
| `episodes.csv` | 918 行（18 个评测 × 51 回合），V3 收尾包 episodes.csv 的列，另加 scheme、seed、gpu_model、v3_run_id |
| `per_seed_summary.csv` | 每个任务、种子、档位一行 |
| `scheme_summary.csv` | 每个任务、档位，三个种子的均值和样本标准差 |
| `v3_vs_5070.csv` | 和 V3 收尾包的逐状态对照 |
| `configs/` | 每个评测的启动记录（含完整命令、git HEAD / status / diff、checkpoint / 状态 / Push 放置证明的 sha256）、退出记录、`run_manifest.json`，gzip 压缩 |
| `provenance.json` | 来源目录、参考的 sha256、手部 FK 自检、掌心后处理程序的 sha256、V3 episodes.csv 的 sha256、启动器日志的 sha256 |
| `SHA256SUMS` | |

原始评测输出（轨迹等）在本机 `logs_system1/post_train/d_final_5070/`，不进 git。

生成命令：

    hssim_s1/bin/python -B system1/t03/package_d_final_5070.py

## 和任务书的偏离

无。
