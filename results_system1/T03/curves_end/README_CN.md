# T03 云端曲线评测结束：没评的曲线点

依据：Kevin 2026-11-13 的云端第 1 条——取消截止后最多 4 小时的延长；到截止时间没评完的曲线点不评，写进 `curves_end.json`，曲线里留空，在 README 里列出。

## 做了什么

- 云端调度器（`system1/cloud/c2_scheduler.py`）2026-11-13 17:02:00 按 `--extend-hours 0` 重启，pid 373927。
- 17:15:06 它启动第一个评测 `c2_curve/cin_table_m5kg_seed722_fresh12000/it09000_pm2` 时出错退出。原因是重启所在的 shell 没有先 source `/root/autodl-tmp/system1/s1_env.sh`，环境变量 `S1_PY` 为空；`--python` 的默认值就是 `$S1_PY`，于是成了 None，`subprocess.Popen` 报 TypeError。证据：`c2_scheduler_log_tail.txt`、`c2_scheduler_pid_record.json`（argv 里没有 `--python`）。这是我重启时的失误。
- 17:15:06 以后没有再启动任何评测。当时已在运行的评测（Bin 721/722/723 和 Table 721 的第 9000 轮 ±2）都跑完了。
- 调度器退出后，`c2_decisions.json` 里没有记下"全部训练结束"、截止时间和曲线结束，`curves_end.json` 也没写出来。
- 最后一个训练（C-in Table 722）约 17:29 结束，按"训练结束 + 2 小时"的规则，截止时间约为 19:29。我 21:08 发现这件事时截止时间已过，所以按规则不补评。
- `curves_end.json` 由我在 21:08 用调度器自己的 `not_evaluated` / `est_minutes` 函数生成，文件里注明了不是调度器写的。21:3x 把其中的原因改成上面更准确的说法（`reason_corrected` 字段），其他内容没有改。

## 没评的曲线点（共 324 个曲线评测里有 94 个没评）

| 组 | run | 没评的轮次 |
|---|---|---|
| 主曲线 | C-in Bin 721、722、723 | 10000、11000、12000 |
| 主曲线 | C-in Table 721 | 10000、11000、12000 |
| 主曲线 | C-in Table 722、723 | 9000、10000、11000、12000 |
| 主曲线 | C-out Bin 721 | 9000、10000、11000、12000 |
| 主曲线 | C-out Table 721 | 9000、10000、11000、12000 |
| 质量扫描 | C-in Bin 2 kg、4 kg | 2000–12000（11 个） |
| 质量扫描 | C-in Bin 8 kg | 1000–12000（12 个） |
| 质量扫描 | C-in Table 10 kg、20 kg | 1000–12000（12 个） |
| 质量扫描 | C-in Pull 40 kg | 1050–8050（8 个） |

- 主曲线 28 个，质量扫描 66 个，D 曲线 0 个。
- 主曲线里各 run 第 12000 轮的 ±2，本机 5070 已经评过（最终对比表，见 `results_system1/T03/c2_finals/`）。按"曲线点全部来自云端 4090"的规则，本机的数不进曲线。
- Pull 的主曲线（C-in 721–723、C-out 721）全部评完。
- 每个点在 `curves_end.json` 里都有按调度器规则估计的云端用时（`est_minutes_cloud`）。在同时跑满评测的 4090 上，Bin 约 132 分钟，Table 约 184 分钟。

## 文件

- `curves_end.json`：没评的曲线点，以及原因、生成方式。
- `c2_scheduler_log_tail.txt`：调度器日志从 17:02 重启到出错退出的部分。
- `c2_scheduler_pid_record.json`：17:02 重启时的启动记录。
- `SHA256SUMS`。

云端完整日志已同步到本机 `exports_system1/cloud_sync/logs/`，不进 git。

## 和任务书的偏离

- 截止前约 2 小时 14 分钟（17:15 到 19:29）没有启动评测。按规则，这段时间本来会继续评上表里的点。这是调度器出错退出造成的，不是因为时间不够。
- 截止后我没有补评；要不要在租卡训练时顺便补上，由 Kevin 决定。

## 之后：Kevin 同意补评（2026-11-13 至 2026-11-14）

- **更正：** 上面写的"D 曲线 0 个"不对。D Bin 种子 721 第 0、2000、4000、6000、8000、10000 轮这 6 个点，从来没进过云端 D 曲线清单（`inputs/d_curve/catalog.json` 里这个 run 只有第 12000 轮），所以调度器的 `not_evaluated` 也没有列出它们。
- **Kevin 2026-11-13 22:1x：** 主曲线的 28 个点补评；质量扫描的 66 个点不评。
  - 用 `system1/cloud/run_curve_points.py` 在云端 5 张 4090 上评。它沿用调度器自己的评测命令、环境、记录和认领处理，见脚本说明。
  - 22:15:03 开始，2026-11-14 01:17:04 全部结束。28 个都是 rc 0、51 回合、complete。
  - 日志：`run_curve_points_20261113.log`。
- **Kevin 2026-11-13 23:0x：** D Bin 种子 721 的 6 个点也补评，和 28 个点并行。
  - 6 个 checkpoint 从本机只读目录上传到云端 `inputs/d_curve/D_bucket_paper_B-RZ0_seed721/`，sha256 核对一致。
  - 云端清单加了这 6 条：`cloud_d_curve_catalog.json`，改之前的备份是 `cloud_d_curve_catalog.before_dbin721_20261113.json`，差异见 `cloud_d_curve_catalog.diff`。
  - 用 `run_curve_points.py --names ... --rank-base 950` 评，启动时按清单核对 checkpoint 的 sha256。
  - 23:10:22 开始，2026-11-14 01:16:22 全部结束。6 个都是 rc 0、51 回合、complete。
  - 日志：`run_curve_points_dbin721_20261113.log`。
- 8 个第 12000 轮 ±2 的 C 点原先由本机认领。这次云端重评，认领仍归本机，另写了 `cloud_rerun.json`。曲线用云端的数，表 1 用本机的数。
- 补评后，主曲线和 D 曲线都没有缺点。曲线结果见 `results_system1/T03/train_efficiency/`。质量扫描的 66 个点仍然没评。
- **同步和关机：**
  - 2026-11-14 01:25 把云端 `logs/`、duet-hand 的 `logs_system1/`、`_to_delete/`、D 曲线清单和备份同步回本机 `exports_system1/cloud_sync/`。
  - 又补拉了两个云端仓库里的 `logs/`（各 2.3 MB），以及 duet-hand 仓库根目录下放错位置的 `.launch.json`（66 字节）。
  - 01:33:21 用 `rsync -anc --delete` 逐个文件核对，全部 0 处差异。
  - 云端两个仓库里改过、未提交的文件：92 个里 91 个和本机文件或本机提交一致，剩下 1 个就是上面那个 `.launch.json`。
  - 01:33:34 Kevin 指示后在云端执行 `shutdown`，只关机，没有释放实例。01:33:47 ssh 连接被拒绝。
  - 第一次核对时本机多出 1 个文件：21:31 中断的 rsync 留下的半截临时文件，1.5 MB，云端没有。已移到本机 `_to_delete/cloud_sync_partial_rsync_tmp_20261114/`，之后再核对为 0。
