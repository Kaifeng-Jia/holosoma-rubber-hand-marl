# T07 第 0 步：SONIC 带手检查（云端 4090）

## 做了什么

所有运行都在云端 RTX 4090 上做，SONIC 用官方 release checkpoint，PD 用 SONIC 的增益（和方案 A–C 相同）。每一项都同时跑了带 Sharpa 手的 G1 和原来的橡胶手 G1，作为对照。

1. **示例走路 10 s（任务书第 1 项）**
   - 参考：SONIC 自带的示例动作 A001（T01 用过的 `sonic_sample_A001`），取前 500 个策略步，即 10 s。
   - 脚本：`system1/sonic/validate_single.py --max-steps 500 --record-torques`；带手时加 `--sharpa`，手指锁在张开姿态，PD 目标为 0。
   - 带手用的 73 关节参考由 `system1/hand/sharpa_single_motion.py` 生成：
     - 手指关节恒为 0；
     - Sharpa 各连杆的位姿按 URDF 从 `<side>_wrist_yaw_link` 刚性推出；
     - SONIC 的参考文件不变。
   - 生成结果在本机和云端的 sha256 相同（06df8970…）。
2. **方案 A 的 nominal 回合（任务书第 2 项是 Pull）**
   - 实际做的是 Bin nominal，原因见"偏离"。
   - 冻结状态文件：`bucket_states.json`（sha256 631026d3…）的 nominal 状态。
   - 跑了 5 次：
     - 第一批 2 次：橡胶手；Sharpa 手指 track。
     - 第二批 3 次（加 `--record-torques`）：橡胶手；Sharpa 手指锁死；Sharpa 手指 track。
   - 手指 track 用的是 T06 第 2 步的测试手指参考（`test_A_track.npz`，sha256 aca0cfc5…），不是最终参考。
   - 两批里相同设置的结果完全一样：橡胶手都是 219 步，track 都是 221 步，各项误差也相同。
3. **力矩顶满**
   - 判定：|τ| ≥ 0.95 × 力矩上限，在每个物理子步上判断（和评测脚本的 `--record-torques` 同一规则）。
   - 身体 29 个关节和手指 44 个关节分开统计。

## 结果

**示例走路 10 s**（`walk.csv`）

| 机器人 | 执行步数 | 摔倒 | SONIC 关节跟踪误差（mm） | 骨盆最大高度误差（m） | 末帧骨盆水平误差（m） | 身体关节顶满比例 | 身体关节最大 |τ|/上限 |
|---|---|---|---|---|---|---|---|
| 橡胶手 | 500 / 500 | 否 | 20.47 | 0.0090 | 0.0478 | 0 | 0.534 |
| Sharpa，手指锁死 | 500 / 500 | 否 | 19.31 | 0.0091 | 0.0456 | 0 | 0.761 |

- Sharpa 的手指关节里，拇指 CMC_AA 两侧各有 0.05% 的子步顶满；其余手指关节都没有。

**方案 A，Bin nominal**（`step0.csv`；r0、r1 是两台机器人）

| 机器人 | 手指 | 步数 / 497 | 终止项 | SONIC 判摔倒 | SONIC 关节跟踪误差 r0 / r1（mm） | 末帧骨盆水平误差 r0 / r1（m） | 身体关节顶满比例最大值 r0 / r1 | 手指关节顶满比例最大值 r0 / r1 |
|---|---|---|---|---|---|---|---|---|
| 橡胶手 | — | 219 | joint_bad_tracking | 否 | 53.8 / 246.0 | 0.052 / 0.259 | 0.0046（左腕 yaw）/ 0 | — |
| Sharpa | 锁死 | 214 | joint_bad_tracking | 否 | 57.0 / 41.7 | 0.067 / 0.006 | 0.0035（左腕 pitch）/ 0 | 0.007 / 0.006 |
| Sharpa | track | 221 | joint_bad_tracking | 否 | 50.5 / 116.1 | 0.054 / 0.111 | 0.0045（左腕 pitch）/ 0 | 0.497 / 0.959 |

- 三次都在第 214–221 步因 joint_bad_tracking 终止，没有一次走完参考。SONIC 自己的摔倒判定一次都没有触发。
- 手指 track 时顶满的主要是拇指 CMC_AA（上限 3.3 Nm）。左手拇指在 r0 上顶满 49.7% 的子步，在 r1 上顶满 95.9%；右手拇指分别为 32.6% 和 17.0%。手指锁死时，所有手指关节顶满都不超过 0.7%。
- 每次运行的身体关节最大 |τ|/上限都到了 1.0（被截断），出现在左腕。

## 门槛（任务书：带手走路不摔倒，骨盆误差和摔倒判定不比橡胶手明显变差）

- 走路：两种手都没摔倒。Sharpa 的末帧骨盆水平误差 4.56 cm，橡胶手 4.78 cm；骨盆最大高度误差 0.91 cm 和 0.90 cm。
- Bin nominal：三次都没有触发 SONIC 摔倒判定。Sharpa 的末帧骨盆误差，r0 比橡胶手大 0.1–1.5 cm，r1 小 15–25 cm。

## 文件

- `walk.csv`、`step0.csv`：每行一次运行，含 summary 或 result、力矩文件的 sha256。
- `SHA256SUMS`。
- 原始输出（summary.json、轨迹、`*.torques.npz`、日志、代码 sha256 列表 `step0b_code_sha256.txt` / `walk_code_sha256.txt`）在本机 `exports_system1/cloud_sync/logs/t07_step0/`，不进 git。

生成命令：

    hssim_s1/bin/python -B system1/t07/step0_summary.py --step0 exports_system1/cloud_sync/logs/t07_step0 --out results_system1/T07/step0

## 和任务书的偏离

- **nominal 用的是 Bin，不是 Pull。** Pull、Push 走的是 plan5 训练路径，带手的接入还没写。Bin 走的是 CORE4D 双人路径，已经接好了。
- **手指 track 用的是测试参考。** 正式的手指参考还在做（T07 第 1 步）。
- **云端曾有 3 个误放的文件。** 第一批 Bin 运行时，我写的后台命令把 pid 和监控文件写到了 `duet-hand/` 和 `/`。已移到 `logs/t07_step0/misplaced_shell_files/`，没有删除，也已同步回本机。
