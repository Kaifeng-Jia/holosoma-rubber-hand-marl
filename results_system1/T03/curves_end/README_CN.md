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
