# T13 第一轮：云上测速、冒烟、开训记录（2026-11-24，纽约时间）

## 机器（Kevin 在 AutoDL 租，均从 596 克隆，内蒙 B 区）
| 机器 | SSH 端口 | 用途 | GPU | CPU | 配额 |
|---|---|---|---|---|---|
| 199 | 39500 | 测速（第 4 节 B），D-Table | RTX 4090 24 GB，驱动 595.71.05 | Xeon Gold 6430 | 16 核（cgroup cpu.max 1600000/100000；容器内 nproc 显示 128），120 GB |
| 196 | 47117 | 冒烟 D-Push、D-Table，D-Push | 同上 | 同上 | 同上 |
| 210 | 22255 | 冒烟 D-Table-PA，D-Table-PA | 同上 | 同上 | 同上 |

- 登录：本机公钥 `id_ed25519_autodl` 装进三台的 authorized_keys（密码只用一次，没有保存）。
- 代码：三台都 `git pull` 到 `bcff097`，无未提交改动。
- 环境核对：三台 hssim_s1 的 pip freeze 相同；和本机相比，共有的包版本全部一致（torch 2.7.0+cu128、triton 3.3.0、numpy 1.26.0、isaacsim 5.1.0.0），云上另有 SONIC 依赖带来的 11 个包。SONIC：commit 1e1ec4f5，代码包 sha256 cd5c3912…，`sonic_release/last.pt` e6bdab3f…，`config.yaml` f0818779…，与记录值一致。
- D-Push 的单机先验 `logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt` 从本机 scp 到 199、196，sha256 11ef1fe7…（与本机相同）。

## 偏离
- 任务书写"在第一张卡上测速"。210 起初密码登录被拒（Kevin 后来重发了密码），测速在 199 上做。
- 测速计划要求线程数 N 用 `nproc` 读；容器里 `nproc` 是宿主机的 128，N 取 cgroup 配额 16。

## 测速（`system1/t13v4/cloud_speed.py`，199，D-Table 正式命令行，各 100 轮，s/轮取第 3 轮以后的中位数）
| 设置 | s/轮 | GPU 利用率均值 | 进程树 CPU 均值 | 显存峰值 | 溢出警告行 | torch 线程（请求 / 实际） |
|---|---|---|---|---|---|---|
| 2048，默认线程 | 5.155 | 39.1 % | 182.5 % | 6413 MiB | 0 | — / 64 |
| 2048，OMP = MKL = 16，--torch-threads 16 | 5.182 | 38.6 % | 184.2 % | 6413 MiB | 0 | 16 / 16 |
| 4096，默认线程 | 6.166 | 43.7 % | 196.3 % | 9793 MiB | 0 | — / 64 |

- 线程：默认（5.155）比 16（5.182）快，用默认。三台环境里都有 `OPENBLAS_NUM_THREADS=32`，照实记录，没有改。
- 环境数：4096 / 2048 = 1.196 < 1.5；无溢出警告；显存 9793 / 24564 MiB → 三个条件都满足，用 4096。2048 时 5.155 s/轮，不慢于 6 s/轮。
- 原始记录：`speed4b/`（speed_summary.json、各设置的 json 和每 5 秒的采样）。

## 冒烟（`system1/t13v4/cloud_smoke.py`，2048 个环境，`cloud_runs.py --dry-run` 的命令行；预热照计划，主阶段 10 轮，最后一个 checkpoint 用 cloud_runs.py 的验证命令评一次）
| run | 机器 | 阶段（返回码，s/轮） | 溢出警告行 | 非有限值 | 验证 | 验证状态 sha256 |
|---|---|---|---|---|---|---|
| D-Push | 196 | 预热 50 轮（0，5.09），主阶段 10 轮（0，5.03） | 0 | 无 | 完整，51 回合，非 nominal 完成 0，用时 513 s | bb7cedf4… |
| D-Table | 196 | 主阶段 10 轮（0，5.20） | 0 | 无 | 完整，51 回合，非 nominal 完成 0，用时 237 s | 0caf56dd… |
| D-Table-PA | 210 | 主阶段 10 轮（0，5.02） | 0 | 无 | 完整，51 回合，非 nominal 完成 0，用时 246 s | 0caf56dd… |

## 开训（`system1/t13v4/cloud_launch.py`，本机 tmux `t13_launch`，测速和三个冒烟都通过后自动启动；Kevin 2026-11-24 "通过了自动开始"）
| run | 机器 | runner PID | 训练开始 | 环境数 | 轮数规则（4096，全部减半） | 预热 |
|---|---|---|---|---|---|---|
| D-Table | 199 | 7376 | 18:21:21 | 4096 | 存 500，验证 1000，4000 后看停止规则，窗口 2000，最早 6000，最多 10000 | 无 |
| D-Push | 196 | 7765 | 18:21（预热开始） | 4096 | 同上 | 25 轮 critic-only，从单机先验 |
| D-Table-PA | 210 | 8075 | 18:21:38 | 4096 | 同上 | 无 |

- 线程：默认（不设 OMP / MKL，不加 --torch-threads）。种子 721。轮数从 critic 预热结束后开始数（Kevin 2026-11-24），写在各 `launch.json` 的 `iteration_counting`。
- 各 run 的完整命令、git 状态在 `launch/*_launch_*.json`。
