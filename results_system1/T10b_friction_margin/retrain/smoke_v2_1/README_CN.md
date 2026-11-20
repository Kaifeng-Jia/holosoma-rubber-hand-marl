# T10b B：推扰改成逐分量后的 D 冒烟测试（第 2.1 版）

依据：`docs/system1/T10b_dr_ranges_zh.md` 2026-11-20 修订版（`f4ed7a6`），sha256 `1def9ac2266ca4379d81197ac5a6cb665e98ec178c186f7f0376a25e11a087ad`。本返包只写事实。

- **和第 2 版相比只改了推扰：**
  - 现在：线速度 x、y ±0.5 m/s，z ±0.2 m/s；角速度 roll、pitch ±0.52 rad/s，yaw ±0.78 rad/s。即 SONIC Table 4 的逐分量值，也是 holosoma WBT 预设的 `max_push_vel`。
  - 第 2 版：每个线速度分量 ±0.3 m/s，每个角速度分量 ±0.78 rad/s。
  - 不变的部分：推扰间隔 U[1, 3] s；速度在世界坐标系下叠加到根部速度上；两台机器人各自抽样。
  - 其他随机化项都没变，见上一级的 `../README_CN.md` 第 1 节。
- **范围：** 按 Kevin 的要求只重跑 D Table 种子 721。C-in Bin 没有重跑，它最近一次冒烟是第 2 版（推扰 ±0.3 / ±0.78），记录在 `../smoke/`。
- **时间：** 2026-11-20 15:41:09 – 15:46:18（America/New_York）。
- **机器：** 云端 RTX 4090 的 GPU 0（`autodl-container-cgna4vuzjk-b9c168f4`），环境见 `../cloud_env_20261120/README_CN.md`。第 2 版冒烟是在本机 5070 上跑的。
- **代码：**
  - HEAD `dc3ceae`，工作区干净，记录在 `smoke/d_table_seed721__launch.json`；
  - `randomization.py` 的 sha256 是 `250f587474087fc778e8558f98210a594c0bee515226990b9ac6816b08fc169a`。
- **命令：** `$S1_PY -B system1/t10b/run_smoke.py --out $S1_ROOT/logs/t10b_smoke_v2_1 --only d_table_seed721 --python $S1_PY --taskset ""`，`CUDA_VISIBLE_DEVICES=0`。
  - 50 次迭代，512 个 env，种子 721，其余和第 2 版冒烟相同。
  - 新加的 `--python`、`--taskset` 两个参数，默认值和原来一样。

## 结果

| 配置 | 退出码 | 迭代数 | 非有限值 | 启动时读回检查 | 结束时 PD / 推扰检查 | reward_mean 第 1 轮 → 第 50 轮 |
|---|---|---|---|---|---|---|
| D Table 721 | 0 | 50 | 无 | 全部通过 | 通过 / 通过 | −0.033 → 0.018 |

**推扰，结束时读回：**
- 触发 131 次，共推了 348 台次机器人；
- 各分量最大 |Δv|：x 0.499、y 0.498、z 0.199 m/s，roll 0.519、pitch 0.520、yaw 0.775 rad/s，都不超过新的上限；
- 前两次推扰的具体值在 `smoke/d_table_seed721__randomization_samples.json` 的 `final_readback.first_pushes` 里。

**其他读回，所有 env 的最小、最大值：**

| 项目 | 值 |
|---|---|
| 机器人 0 静 / 动摩擦 / 恢复系数 | 0.331–1.587 / 0.305–1.182 / 0.010–0.479 |
| 机器人 1 静 / 动摩擦 / 恢复系数 | 0.320–1.600 / 0.314–1.188 / 0.006–0.499 |
| 不同材质数（每台） | 64 |
| 躯干质心偏移 x / y / z（机器人 0） | −0.0248–0.0250 / −0.0500–0.0499 / −0.0750–0.0746 m |
| 躯干质心偏移 x / y / z（机器人 1） | −0.0250–0.0250 / −0.0500–0.0496 / −0.0740–0.0746 m |
| 其他刚体质心的最大变化 | 1.2e-7 m |
| 关节角偏移 | −0.0100–0.0100 rad（两台） |
| 物体质量 | 4.001–5.995 kg（名义 5.0） |
| 物体惯量系数 / 惯量总系数 | 0.501–1.499 / 0.435–1.782 |
| 物体静 / 动摩擦，恢复系数 | 0.109–0.595 / 0.105–0.586，0 |
| 物体质心偏移 x / y / z | −0.0798–0.0800 / −0.0791–0.0800 / −0.0796–0.0799 m |
| 结束时 kp / kd 系数 | 0.900–1.100 / 0.900–1.100，两台不同 |

- 种子和 env 数都和第 2 版相同。启动时读回检查 `startup_range_check` 和前几个 env 的抽样值（推扰间隔、PD 系数等），都和第 2 版逐项相等。
- 和第 2 版冒烟的 `run_config` 比，不同的键只有：
  - `push_max_vel`、`recipe`；
  - `randomization.py` 的 sha256；
  - `git_commit`、开始时间；
  - 参考文件的绝对路径（云端和本机不同）。
- **评测器兼容：** 冒烟 checkpoint `model_00050.pt`（sha256 `53d46960…9762`）的 `core4d_smalltable_mappo` 元数据和原 D Table checkpoint `f163d11a…1c50` 相等，actor、critic、优化器和归一化器的参数形状都相同。
- **和原训练配置的差别：** `configs/d_table_seed721_original_vs_smoke.diff.txt`。奖励、终止、观测、网络和 PPO 参数逐键相同。

## 文件

- `smoke_summary.json`：本次结果，冒烟和原 checkpoint 的 sha256，`randomization.py` 的 sha256，验证集的 sha256。
- `smoke/`：`launch.json`（命令、git HEAD、diff）、`exit.json`、`randomization_samples.json`、`metrics.jsonl.gz`。
- `configs/`：冒烟的 `run_config`、原训练的 `run_config`（gzip）、逐键 diff。
- 生成：`system1/t10b/package_retrain_smoke.py --smoke logs_system1/t10b/smoke_v2_1_cloud --only d_table_seed721`。
