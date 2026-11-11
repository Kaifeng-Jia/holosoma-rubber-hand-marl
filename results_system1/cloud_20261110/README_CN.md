# 云端返包（2026-11-10）：环境、Pull 复现验证、开训

本文件只写事实。时间为 America/New_York（云端系统时间为 CST，已换算）。

## 机器

- **AutoDL，5 × RTX 4090（24 GB）**，驱动 570.124.04，CUDA 12.8。
- **CPU：** Intel Xeon Platinum 8352V @ 2.10 GHz，可见 128 个逻辑核；容器 CPU 配额 `cpu.max = 8000000 100000`，相当于 80 核。
- **内存与磁盘：** 内存 1 TB；数据盘 `/root/autodl-tmp` 共 130 GB。
- **开机状态：** Kevin 开好时已经是有卡模式。无卡模式要在控制台切换，我在 ssh 里做不了，所以搭环境的约 15 分钟里 5 张卡空着计费。
- **工作目录：** `/root/autodl-tmp/system1/`。数据盘上之前项目留下的 `r2_*`、`v3_*` 目录只读，没有改动。

## 环境（`system1/cloud/setup_cloud_from_r2.sh`）

- **Isaac 环境：** 没有从头安装 Isaac Sim，而是复制数据盘上已有的同版本环境 `/root/autodl-tmp/r2_deps/hssim`，得到 `system1/deps/hssim_s1`。同时复制了 `r2_deps/IsaacLab`，并把 holosoma 和 isaaclab 两个 editable 安装改指到 `system1/` 下的副本。脚本里有断言检查，不允许再指向原目录。
  - 版本（`versions.json`）：Python 3.11.17，torch 2.7.0+cu128，isaacsim 5.1.0.0，isaaclab 0.47.2，numpy 1.26.0，scipy 1.15.3。与本机 `hssim_s1` 的这几项版本相同；其他周边小包的版本有差别。
- **代码：** 浅克隆 `system1-sonic`。云端访问 GitHub 不稳定：开学术加速时返回 503，直连时常超时，靠重试才拉下来。
- **SONIC 代码：** 用 `git archive` 打包固定 commit `1e1ec4f5`（LFS 指针不展开），从本机上传，sha256 为 `cd5c3912…`。
- **SONIC 权重：** `last.pt` 从 hf-mirror 下载，`config.yaml` 从 HF 下载。两者都是固定版本 `6733128a`，sha256 与固定值一致（`e6bdab3f…`、`f0818779…`）。
- **输入：** `cloud_inputs.tar`（sha256 `964f9a96…`），里面每个文件都按 SHA256SUMS 核对通过。
- **补装 `hydra-core==1.3.3`（`--no-deps`）：** 第一次开 C 的 run 时报 `ModuleNotFoundError: hydra`（gear_sonic 的 `torch_humanoid_batch.py` 要导入它）。本机环境本来就有这个包，R2 环境没有。补装前后的 pip freeze 见 `pip_freeze_{before,after}_hydra.txt`，唯一差别就是这一行。搭环境脚本已同步补上。

## Pull 复现验证（T01b 决定 1；开训前）

- **对象：** Table Pulling 论文模型 seed721，`model_08050.pt`（sha256 `727630e9…`），用冻结初始状态，51 个回合。脚本 `system1/cloud/repro_pull.sh`：±2 档在卡 0、±8 档在卡 1 并行。
- **运行时间：** 16:56–17:27。

| 档 | V3 对照值的来源 | 51 个回合逐项比较 | 结论 |
|---|---|---|---|
| ±8 | V3 云端 RTX 4090 | 所有字段一致；进度、物体 RMSE、物体 xy RMSE、高度 RMSE、奖励、手部误差的最大差全部为 0 | 4090 对 4090 逐位一致 |
| ±2 | 本机 RTX 5070 Laptop（T01 与 F2 都在本机复现过，逐项一致） | 完成与否 51/51 一致，执行步数 51/51 一致，终止原因 51/51 一致；连续量有差异 | 差异只在连续量的浮点位 |

±2 档连续量与 V3 的差：

| 指标 | 51 回合中位差 | 最大差 | 51 回合均值（云端 / V3） |
|---|---|---|---|
| 物体位置 RMSE（cm） | 0.053 | 4.845 | 1.943 / 2.032 |
| 手部位置误差（cm） | 0.046 | 0.945 | 3.776 / 3.780 |
| 奖励和 | 0.005 | 0.388 | 42.806 / 42.798 |

- **结论（Kevin 2026-11-10 确认）：** ±8 档 4090 对 4090 逐位一致；±2 档 51 个回合的完成情况、步数和终止原因全部一致，差异只在连续量的浮点位，来自 GPU 型号不同。据此开训。
- **文件：** `repro_pull/compare_pm{2,8}.{csv,json}`，以及 `checkpoint.sha256`、`gpu.txt`。

## 开训

- **启动：** 17:47:34 EST。`start_queues.py --assign`，每个 run 单独一个队列：

  | 卡 | run |
  |---|---|
  | 0 | D Pull 40 kg、C Pull 20 kg |
  | 1 | D Table 20 kg、C Table 5 kg |
  | 2 | D Bin 2 kg、D Bin 4 kg |
  | 3 | D Bin 8 kg、C Bin 1 kg |
  | 4 | D Table 10 kg（单独） |

  每个 run 绑定 14 个逻辑核（taskset），LOCAL_RANK 互不相同。详见 `assignment.json`。
- **C 的三个 run 第一次启动失败**（缺 hydra，见上）：都在第一步之前退出，没有开始训练。
  - 失败那次的目录和日志移到云端 `/root/autodl-tmp/system1/_to_delete/c_launch_attempt1_missing_hydra_20261110T1748/`，没有删除。
  - 补装后按原来的卡号、LOCAL_RANK 和 CPU 段，在 17:51:32 重新启动。
  - 按 T03 的剔除规则，这次不算无效 run：没有出现 NaN，也不是训练中断，而是环境缺包、根本没有开始。
- **启动时的 F6 对照：** 10 个已启动的段都是 0 个意外字段，其中包括本机没法预检的 D Pull joint8000 段（`ops/*.run_config_diff.json`）。
- **第二波调度：** `train_monitor.py c-mass` 已在云端后台运行。三个 C 721 run 都过 1000 轮且没有 NaN 后，依次开 T04-C 的 6 个 run，再开 C 补种子 722、723 的 6 个 run；每张卡最多 2 个。

## 训练速度（实测）

s/轮取各 run 日志里最近若干轮的中位数。原始记录：`speed_20261110T1759.txt`（开训约 10 分钟，最近 30 轮）、`speed_20261110T2000.json`（`train_monitor.py status`，最近 50 轮）。

| 卡 | run | 同卡另一个 run | 17:59 s/轮 | 20:01 轮数 | 20:01 s/轮 | 按 20:01 速度还需 |
|---|---|---|---|---|---|---|
| 4 | D Table 10 kg | 无 | 6.67 | 1296 / 12000 | 5.97 | 17.7 h |
| 1 | D Table 20 kg | C Table 5 kg | 7.95 | 1036 / 12000 | 7.64 | 23.3 h |
| 1 | C Table 5 kg | D Table 20 kg | 8.66 | 953 / 12000 | 8.07 | 24.8 h |
| 0 | D Pull 40 kg（joint8000 段） | C Pull 20 kg | 7.56 | 1052 / 8000 | 7.60 | 14.7 h |
| 0 | C Pull 20 kg（joint8000 段） | D Pull 40 kg | 7.87（critic50 段） | 975 / 8000 | 7.78 | 15.2 h |
| 2 | D Bin 2 kg | D Bin 4 kg | 8.76 | 967 / 12000 | 8.14 | 24.9 h |
| 2 | D Bin 4 kg | D Bin 2 kg | 8.46 | 981 / 12000 | 7.98 | 24.4 h |
| 3 | D Bin 8 kg | C Bin 1 kg | 8.48 | 962 / 12000 | 8.24 | 25.3 h |
| 3 | C Bin 1 kg | D Bin 8 kg | 9.43 | 922 / 12000 | 8.24 | 25.4 h |

- critic50 段：D Pull 6.05 s/轮、C Pull 7.87 s/轮，均已完成。
- **共用一张卡的影响：** 同配置只差质量的 D Table 20 kg（与 C 同卡）对 D Table 10 kg（单独）：17:59 为 7.95 / 6.67 = 1.19 倍，20:01 为 7.64 / 5.97 = 1.28 倍，都低于任务书的 2 倍，保持两个 run 同卡。
- **GPU 利用率：** 17:59 时卡 0–4 依次为 28 / 86 / 81 / 75 / 17 %；20:01 时为 95 / 87 / 91 / 81 / 38 %。显存每个 run 约 6.0–6.7 GB。
- **CPU：** 20:01 时负载均值 18.8（配额 80 核）。
- **和事前估算的差：** 事前按历史 4090 记录的 2.46 s/轮估算。单独一个 run 的实测是 5.97 s/轮，约为估算速度的 0.41 倍。
- **NaN：** 到 20:01，所有 run 的日志数值都是有限值（`n_nonfinite_values = 0`）。
- **调度上限：** 第二波仍按每卡最多 2 个 run，没有改动。
