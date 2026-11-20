# T10b B：随机化重训，第 2 版随机化与冒烟测试返包

依据：
- `docs/system1/T10b_friction_margin_zh.md` B 部分；
- `docs/system1/T10b_dr_ranges_zh.md`（2026-11-20 第一版，Kevin 已确认），sha256 `4addabbb4444c1b3cb23461fca5f0390b07e235bdf5e03087ef8136be7a5a9a1`。

本返包只写事实。

> **2026-11-20 更新，第 2.1 版：**
> - **推扰：** 任务书 `f4ed7a6` 把推扰改成 SONIC Table 4 的逐分量值，即线速度 x、y ±0.5，z ±0.2 m/s；角速度 roll、pitch ±0.52，yaw ±0.78 rad/s。代码在 `fd96719`。下面第 1 节表格里的推扰一行，第 3 条"与同类工作的预设不同的地方"，以及第 2 节 D 的推扰读回，都是第 2 版的值，已被第 2.1 版取代。
> - **D 冒烟：** 在云端 GPU 0 上重跑并通过，见 `smoke_v2_1/README_CN.md`。C-in Bin 没有重跑，第 2 节里 C-in 的冒烟用的还是第 2 版推扰。
> - **云端环境：** 见 `cloud_env_20261120/README_CN.md`。这台实例的数据盘是空的，环境从头安装后，按本机 `hssim_s1` 逐包固定了版本。
> - **试点：** 15:48 在两张 4090 上启动，代码 `dc3ceae`。命令和本节第 4 条的 dry-run 相同：
>   - D Table 721 在 GPU 0，训练 PID 6812；
>   - C-in Bin 721 在 GPU 1，训练 PID 7452；
>   - 调度器 `pilot_cloud.py` 的 PID 是 6809。
>   - 两个 run 正式训练时，启动读回检查都通过了。

- 第 1 版的冒烟记录在 `v1_superseded/`，已作废，试点不用。
- **冒烟测试时间**（America/New_York，2026-11-20）：

| 配置 | 开始–结束 |
|---|---|
| D Table 721 | 12:43:02 – 12:48:58 |
| C-in Bin 721 | 12:48:58 – 12:55:24 |

- **机器：** 本机 RTX 5070 Laptop，同时有一路 T10b A 评测在跑。环境 `hssim_s1`。
- **代码：**
  - D 冒烟时 HEAD 是 `c744e69`，C-in 冒烟时 HEAD 是 `b9a4fe1`；两者之间只多了 T10s 的提交，没有动 B 的代码。
  - B 的改动在两次冒烟时都还没提交，随本返包一起提交。
  - 冒烟时各文件的 sha256：`randomization.py` `d92c1f03…`，`scheme_a.py` `25963b06…`，holosoma `observation/terms/marl.py` `f27661b8…`，`train_core4d_smalltable.py` `45583db8…`。
  - 完整 diff 在 `smoke/*__launch.json` 里。

## 1. 随机化项（D 和 C-in 用同一套；机器人的项两台各自抽样）

| 项目 | 范围 | 实现 | 何时抽 |
|---|---|---|---|
| 机器人所有碰撞形状的材质（含脚、橡胶手） | 静摩擦 U[0.3, 1.6]，动摩擦 U[0.3, 1.2]，恢复系数 U[0, 0.5] | holosoma `_isaacsim_randomize_rigid_body_material`，即 Isaac Lab `randomize_rigid_body_material`。64 个材质桶，每个形状随机取一个桶，和 holosoma WBT 预设相同。每台机器人调用一次 | 启动 |
| 躯干（`torso_link`）质心偏移 | x ±2.5 cm，y ±5 cm，z ±7.5 cm | holosoma `randomize_body_com`，WBT 预设在 Isaac Sim 上用的就是它。每台机器人调用一次；调用前核对第二台的初始质心和 holosoma 记录的默认值一致 | 启动 |
| 默认关节角偏移 | 每台、每个关节 U[−0.01, 0.01] rad | 按 holosoma `setup_dof_pos_bias` 的做法，改成每台机器人各自抽样，`env.default_dof_pos` 变成 [env, agent, dof]。效果相当于编码器零偏 b，见下面说明 | 启动 |
| 推扰 | 间隔 U[1, 3] s；根部线速度每个分量 ±0.3 m/s，角速度每个分量 ±0.78 rad/s，叠加到当前速度上 | 调度沿用 holosoma `PushRandomizerState` 和 `randomize_push_schedule`；施加部分用 `s1_apply_pushes`，两台机器人各自抽速度（holosoma 的 `apply_pushes` 只推第一台） | 每个 env 到点时 |
| PD 增益 | kp、kd 各 U[0.9, 1.1]，按每台、每个关节抽 | `s1_pd_gain_reset`，写入配对动作项自己的 [env, agent, dof] 缩放 | 每次重置 |
| 物体材质 | 静、动摩擦各 U[0.1, 0.6]，恢复系数 0 | 和 holosoma `randomize_object_rigid_body_material_startup` 相同的 helper 和 64 个桶，每个形状随机取一个桶 | 启动 |
| 物体质量 | × U[0.8, 1.2]，惯量按同一比例缩放 | holosoma 的质量 helper，即 Isaac Lab 的 `operation="scale"` 加 `recompute_inertia` | 启动 |
| 物体惯量 | 在质量缩放之上再 × U[0.5, 1.5]，每个 env 一个系数，作用于整个惯量张量 | `s1_object_inertia_startup` | 启动 |
| 物体质心偏移 | 每个轴 ±0.08 m | `s1_object_com_startup` | 启动 |
| 动作延迟 | 去掉 | — | — |
| 观测噪声 | 不变，沿用原训练配置 | 训练开、评测关 | — |

- **开关：** `scripts/train_core4d_smalltable.py --system1-randomization`。不加时 `CONFIG.randomization` 和训练行为都不变。
- **摩擦组合方式：** 保持 PhysX 和场景的现有设置。物体对地面的摩擦和物体对手的摩擦，都按各自材质组合出来。
- **关节角偏移的作用方式（编码器零偏 b，D 和 C-in 一致）：**
  - D：观测里的关节角是 q − default − b，PD 目标是 default + b + scale·a。这两处都由 holosoma 原有的观测项和动作项读取 `default_dof_pos` 实现。观测项 `paired_dof_pos` 改成在 default 为三维时直接使用；二维时和原来一样。
  - C-in：冻结的 SONIC 读到 q − b，它输出的目标 q* 加上 b 后再交给 PD（`system1/scheme_a.py`）。没有偏移时，这两处行为都不变。
- **自检：** setup 的最后一项 `s1_readback_check` 把以上每一项对所有 env 从 PhysX 或 env 读回来核对范围，超出就报错退出。训练结束时，`final_readback` 再核对 PD 系数和已施加的推扰。

**与同类工作的预设不同的地方：**
1. **物体惯量：** 用一个系数缩放整个张量。holosoma 的惯量函数对 Ixx、Iyy、Izz 和非对角元各自抽系数，可能违反三角不等式；以桌子为例，Ixx × 1.5 会大于 (Iyy + Izz) × 0.5。
2. **holosoma 的物体材质函数** 会检查模拟器类名必须是 `IsaacSim`，配对模拟器是 `DualRobotIsaacSim`，所以没有直接调用它，而是用了它内部的同一个 helper。
3. **推扰：** 任务书写"线速度 0.3 m/s，角速度 0.78 rad/s"，这里按六个分量各自 ±0.3 / ±0.78 实现。OmniRetarget 原文的逐分量取值没能核实。
4. **按形状随机：** 材质是每个碰撞形状随机取一个桶。同一台机器人的不同连杆、同一个物体的不同碰撞块（桌子 5 块，桶 16 块），摩擦各不相同。

## 2. 冒烟测试：5070，50 次迭代，512 个 env，种子 721

| 配置 | 退出码 | 迭代数 | 非有限值 | 读回检查 | 结束时 PD / 推扰检查 | reward_mean 第 1 轮 → 第 50 轮 |
|---|---|---|---|---|---|---|
| D Table 721 | 0 | 50 | 无 | 全部通过 | 通过 / 通过 | −0.033 → 0.015 |
| C-in Bin 721 | 0 | 50 | 无 | 全部通过 | 通过 / 通过 | 0.085 → 0.108 |

**读回结果（所有 env 的最小、最大值）：** 两份配置的种子相同，所以机器人的项抽到的值也相同。

| 项目 | D Table | C-in Bin |
|---|---|---|
| 机器人 0 静 / 动摩擦 / 恢复系数 | 0.331–1.587 / 0.305–1.182 / 0.010–0.479 | 同左 |
| 机器人 1 静 / 动摩擦 / 恢复系数 | 0.320–1.600 / 0.314–1.188 / 0.006–0.499 | 同左 |
| 不同材质数（每台） | 64 | 64 |
| 躯干质心偏移 x / y / z（机器人 0） | −0.0248–0.0250 / −0.0500–0.0499 / −0.0750–0.0746 m | 同左 |
| 其他刚体质心的最大变化 | 1.2e-7 m | 同左 |
| 关节角偏移 | −0.0100–0.0100 rad（两台） | 同左 |
| 物体名义质量 → 抽到的质量 | 5.0 → 4.001–5.995 kg | 1.0 → 0.800–1.199 kg |
| 物体惯量系数 / 惯量总系数（含质量） | 0.501–1.499 / 0.435–1.782 | 0.501–1.499 / 0.418–1.754 |
| 物体静 / 动摩擦 | 0.109–0.595 / 0.105–0.586 | 同左 |
| 物体质心偏移，各轴 | −0.0798–0.0800 m | −0.0800–0.0799 m |
| 结束时 kp / kd 系数 | 0.900–1.100 / 0.900–1.100，两台不同 | 同左 |
| 推扰：触发次数 / 机器人推扰数 / 各分量最大 \|Δv\| | 122 / 304 / 0.295, 0.300, 0.299 m/s, 0.780, 0.778, 0.775 rad/s | 999 / 4744 / 0.300, 0.300, 0.300, 0.780, 0.780, 0.780 |

- 前 5 个 env 的具体值见 `smoke/*__randomization_samples.json`，包括两台机器人的手和脚的材质、质心偏移、关节偏移、物体各项、推扰间隔和前几次推扰。
- **D 的推扰次数比 C-in 少：**
  - 推扰计数器在每次回合重置时清零，推扰间隔最短 1 s（50 步）。
  - 50 轮里，D 冒烟每次重置前的回合平均长度是 12.7–48.9 步，C-in 是 16.7–77.3 步，所以 D 的很多回合撑不到第一次推扰。
- **评测器能否接受 checkpoint：** 两份冒烟 checkpoint `model_00050.pt` 的 `core4d_smalltable_mappo` 元数据和对应原 checkpoint 逐项相等，actor、critic 的参数形状也相同。随机化只记在 `run_config` 里。

## 3. 两份配置和原配置的差别（`configs/`）

- 两份配置都只在原最终训练命令上加 `--system1-randomization`：
  - D Table 种子 721：原训练 `A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920`，checkpoint `f163d11a…1c50`；
  - C-in Bin 种子 721：原训练 `cin_bucket_m1kg_seed721_fresh12000`，checkpoint `c4c2fec5…3528`。
- `configs/*_original_vs_smoke.diff.txt` 逐键列出差别。除了随机化本身的项，只有这些不同：
  - 冒烟本身的设置：`iterations_requested` 50、`num_envs` 512、`save_interval` 50；
  - `git_commit`、开始时间；
  - 机器上的绝对路径：参考文件、SONIC 文件。内容相同，由 sha256 字段核对；
  - `object_physics`：原来记录每个 env 的固定值，现在记录范围和前几个 env 实际抽到的值。
- 奖励、终止、观测、网络和 PPO 参数逐键相同。
- 原训练的 `run_config` 以 gzip 存在 `configs/*_original.run_config.json.gz`。

## 4. 验证集和试点脚本（和上一版相同，没有改）

- **验证集：** 每个任务 50 个 ±2 状态，生成器种子 20261120，和冻结的 ±2、±8 测试文件逐状态比对，没有重复。
  - Table：`87a716562282bb838e253fa7578c28d9839fba6ac0a981f4f01368f06e50670d`
  - Bin：`a2222d1fb1c7ba325bfcc87f16fa4619fa10dc8cc212d0dbe1e9e0a5cc502d0d`
- **试点脚本：** `system1/t10b/pilot_cloud.py`，每张卡跑一个 run，命令见 `pilot_commands_dry_run.txt`（本机 dry-run；云端换成 `$S1_PY`）。
  - D Table：上限 20000 轮，每 1000 轮存 checkpoint，每 2000 轮在 Isaac 验证集上评一次，只看验证集。
  - C-in Bin：12000 轮。
- **停止规则（只用于 D Table）：** 4000 轮的观察窗口要整个落在 8000 轮之后，所以最早在 12000 轮那次验证之后停（T10b v1.4 已确认这个读法）。停下时用最后一个 checkpoint。

## 5. 和任务书的偏离与过程记录

1. 第 2 版第一次冒烟（12:35）在初始化时被读回检查拦下。原因是检查本身太严，随机化的值没有问题：
   - (default + b) − default 和 b 用 `torch.allclose` 默认容差比较，float32 舍入误差约 3e-8，超出了默认容差；
   - 其他刚体的质心要求和写入值完全相等，但写进 PhysX 再读出会有约 1e-7 的差别。
   - 两处都改成 1e-6 的容差，并打印全部读回值。失败那次的输出在 `_to_delete/t10b_smoke_v2_attempt1_check_tolerance/`。
2. 原 D Table 训练每 2000 轮存一次 checkpoint；试点按任务书每 1000 轮存一次。
3. 原 D Table 是在 `holosoma-core4d-base` 的 `a21858b` 上训练的；试点用本分支代码，配置逐键核对见第 3 节。
4. 第 1 节"与同类工作的预设不同的地方"的四条。

## 文件

- `smoke_summary.json`：两次冒烟的结果，冒烟和原 checkpoint 的 sha256，`randomization.py` 的 sha256，验证集的 sha256。
- `smoke/`：每次冒烟的 `launch.json`（命令、git HEAD、diff）、`exit.json`、`randomization_samples.json`（含 `startup_range_check` 和 `final_readback`）、`metrics.jsonl.gz`。
- `configs/`：冒烟的 `run_config`、原训练的 `run_config`（gzip）、逐键 diff。
- `pilot_commands_dry_run.txt`、`SHA256SUMS`。
- `v1_superseded/`：第 1 版，已作废。
- 生成脚本：`system1/t10b/package_retrain_smoke.py`。
- 离线测试：`system1/t10b/test_randomization_config.py`（范围和任务书一致、没有延迟项、函数路径都能解析）和 `system1/t10b/test_t10b_offline.py`。
