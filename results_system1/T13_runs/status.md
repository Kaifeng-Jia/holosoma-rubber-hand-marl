# T13 训练状态（2026-11-25 09:03 纽约时间，每小时自动更新）

| run | 迭代（预热后计数）/ 上限 | 最近一次验证（轮: 完成数） | s/轮 | 预计结束 | 应接触时手力 > 1 N（r0 / r1，近 50 轮） | 异常 |
|---|---|---|---|---|---|---|
| d_table | 7719 / 10000 | 7000: 45 | 7.37 | — | 94% / 95% | 停止规则已触发；已结束：stopped by rule |
| d_push | 6350 / 10000 | 6000: 46 | 7.53 | — | 86% / 79% | 停止规则已触发；已结束：stopped by rule |
| d_table_pa | 7829 / 10000 | 7000: 49 | 6.81 | — | 96% / 97% | 停止规则已触发；已结束：stopped by rule |

GPU：d_table 0, 1 MiB, 24564 MiB, 0 %；d_push 0, 1 MiB, 24564 MiB, 0 %；d_table_pa 0, 1 MiB, 24564 MiB, 0 %

本机评测：d_table running；d_push 本机评测已完成（见 results_system1/T13_runs/d_push/）；d_table_pa running
