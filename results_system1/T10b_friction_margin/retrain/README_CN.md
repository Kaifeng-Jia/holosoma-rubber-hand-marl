# T10b B：随机化重训，准备阶段与冒烟测试返包

任务书：`docs/system1/T10b_friction_margin_zh.md`（v1.3）B 部分第 1–3 步，以及第 4 步的准备工作。试点训练**还没有开始**：按任务书，冒烟测试通过后先回报，云端 4090 由 Kevin 开。本返包只写事实。

- 冒烟测试时间（America/New_York，2026-11-20）：D Table 04:08:46–04:12:56；C-in Bin 04:12:56–04:18:47。
- 机器：本机 RTX 5070 Laptop。T10b A 的摩擦评测同时在跑，占一个评测位。
- 环境：`hssim_s1`。代码：`system1-sonic`，D 冒烟时 HEAD 是 `2bf2c9a`，C-in 冒烟时也是 `2bf2c9a`，外加一行日志改动（见“偏离”第 3 条）。

## 1. 随机化项清单

holosoma 已有的项在 `src/holosoma/holosoma/managers/randomization/terms/locomotion.py` 和 `marl.py`。

| 任务书要求 | holosoma 已有的项 | 是否复用 | 本轮用的实现 | 范围 |
|---|---|---|---|---|
| 物体和地面的摩擦 | `randomize_object_rigid_body_material_startup`（Isaac Lab 64 桶，静、动摩擦分别抽样）；配对训练现在用的是 `marl.set_object_rigid_body_material_startup`（固定 0.5/0.5） | 固定项保留，随后覆盖它的值 | `system1.t10b.randomization.s1_object_material_startup`：每个 env 一个系数，静摩擦 = 动摩擦 = 0.5 × U(0.5, 1.25) = [0.25, 0.625]，恢复系数 0。地面材质是 multiply × 1.0，所以物体–地面摩擦等于这个值 | 0.5–1.25 倍 |
| 手和物体的摩擦 | `randomize_robot_rigid_body_material_startup`：只作用于场景实体 `robot`（第一台），而且是所有刚体 | 否 | `s1_hand_material_startup`：只改两台机器人的橡胶手形状（找形状的方法同 `system1/t10b/hand_material.py`）。每个 env、每台机器人一个系数，两只手共用：1.0 × U(0.5, 1.25)。改完回读，并检查非手形状没有变。手–物体摩擦按 PhysX 的 average 规则 = (手 + 物体) / 2 | 0.5–1.25 倍 |
| 物体质量 | `randomize_object_rigid_body_mass_startup`：加法（`operation="add"`），惯量不跟着变 | 否 | `s1_object_mass_startup`：每个 env 质量 × U(0.8, 1.2)，惯量乘同一个系数，质心不变 | ±20% |
| PD 增益 | `randomize_pd_gains` / `ActuatorRandomizerState`：通过单机器人的 `JointPositionActionTerm` 找动作项，缩放张量是 [env, dof] | 否 | `s1_pd_gain_reset`：每次回合重置时，每个 env、每台机器人、每个关节抽一次 kp、kd 系数 U(0.9, 1.1)，写入配对动作项 `DualJointPositionActionTerm` 自带的 [env, agent, dof] 缩放（`update_pd_scales`） | ±10% |
| 观测噪声 | 训练配置 `plan5_actor_obs` 已经打开 `enable_noise=True`：motion_ref_ori_b 0.05，base_ang_vel 0.2，dof_pos 0.01，dof_vel 0.5，motion_command 和 actions 为 0；评测配置关掉 | **复用，不改** | 同原训练 | 同原训练 |
| 动作延迟 | `setup_action_delay_buffers` + `randomize_action_delay`：配对动作项也支持，但它延迟的是策略动作，C-in 的 PD 目标来自 SONIC 的 q*（`joint_target_override`），覆盖不到 | 否 | `s1_target_delay_setup` + `s1_target_delay_reset`：延迟实际的 PD 目标。D 是 default + scale × a，C-in 是 SONIC 的 q*。每个 env 每次重置时抽 {0, 1} 控制步；延迟为 1 时用上一步的目标；重置后的第一步不延迟 | 0–1 个控制步 |

- 开关：`scripts/train_core4d_smalltable.py --system1-randomization`。不加时，`CONFIG.randomization` 和训练行为都不变。
- 物理契约检查：只在开关打开时接受随机化后每个 env 的质量和摩擦范围；不加开关时仍做原来的精确检查。
- 范围记在 `run_config.json` 的 `system1_randomization` 里，抽到的值写在 `randomization_samples.json`。
- 物体–地面摩擦和手–物体摩擦来自两个材质（物体、手），不是两两独立抽样：物体系数同时影响两种接触。

## 2. 两份配置

两份配置都只在原最终训练命令上加 `--system1-randomization`：

- **D Table 种子 721**：
  - 原训练是 `/home/kevin/holosoma-core4d-base/logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/`，checkpoint `model_12000.pt` 的 sha256 是 `f163d11a…1c50`。
  - 原命令：`--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 2048 --steps-per-env 24 --seed 721`。
- **C-in Bin 种子 721**：
  - 原训练是 T03 的 `cin_bucket_m1kg_seed721_fresh12000`，checkpoint 的 sha256 是 `c4c2fec5…3528`。
  - 原命令：`--experiment bucket003 --sonic-residual --sonic-tf32 --sonic-residual-kind in --c-train-diag`，其余同 D。

`configs/*_original_vs_smoke.diff.txt` 逐键列出原 `run_config` 和冒烟 `run_config` 的所有差别。除随机化项外，只有这几项不同：

- 冒烟本身的设置：`iterations_requested` 50、`num_envs` 512、`save_interval` 50；
- 代码版本 `git_commit`、开始时间；
- 机器路径：参考文件、SONIC 文件的绝对路径不同，内容由 sha256 字段核对，相同；
- `object_physics`：原来记录的是每个 env 固定的质量、惯量、质心和材质；现在记录的是随机化范围和前几个 env 实际抽到的值。

奖励、终止、观测、网络、学习率、`gamma`/`lambda`、`steps_per_env`、种子都相同。

原训练的 `run_config` 以 gzip 存在 `configs/*_original.run_config.json.gz`。

## 3. 冒烟测试：5070，50 次迭代，512 个 env

| 配置 | 退出码 | 记录的迭代数 | 有没有非有限值 | 第 1 轮 reward_mean | 第 50 轮 reward_mean |
|---|---|---|---|---|---|
| D Table 721 | 0 | 50 | 无 | −0.0333 | 0.0133 |
| C-in Bin 721 | 0 | 50 | 无 | 0.0748 | 0.1128 |

实际抽到的参数，取前 5 个 env，见 `smoke/*__randomization_samples.json`。两份配置的种子都是 721，所以摩擦和 PD 抽到的值相同：

- 物体摩擦（同时也是物体–地面摩擦）：0.526、0.290、0.431、0.398、0.574。
- 手摩擦，每个 env 给出（机器人 0，机器人 1）：(1.040, 0.839)、(0.634, 0.522)、(0.603, 0.843)、(0.567, 0.642)、(0.872, 0.737)。
- 物体质量：
  - D Table：4.52、4.10、4.73、5.30、5.59 kg；
  - C-in Bin：0.905、0.820、0.947、1.060、1.118 kg。
- kp 系数（机器人 0，关节 0–3），第 1 个 env：1.042、1.029、0.941、0.956；其余 env 见文件。
- PD 重置：第一次重置 env 0–4 时记录一次，之后又记录了一次部分重置（D：env 17、34、89、115、136…；C-in：env 20）。
- PD 目标延迟：
  - 前 5 个 env 两次记录都是 1；
  - 全部 512 个 env 中延迟为 1 的比例是 0.523（只有 C-in 冒烟记录了这个比例，见“偏离”第 3 条）。

**评测器能否接受随机化训练出的 checkpoint：**

- 两份冒烟 checkpoint `model_00050.pt` 的 `core4d_smalltable_mappo` 元数据，和对应原 checkpoint 逐项相等；actor 和 critic 的参数形状也相同。所以评测器的 checkpoint 契约检查和原策略走同一条路径。
- 随机化只记在 `run_config` 里，不写进 checkpoint 元数据。
- 本机 GPU 当时被 A 的两路评测占着，没有在 Isaac 里实际跑一个验证回合。

## 4. 验证集（只用于挑 checkpoint 和停止判断）

- 生成器：`system1/t10b/make_validation_states.py`。和冻结测试文件用同一个生成器 `scripts/paper_evaluation_utils.make_states`，种子换成 20261120（测试文件是 241003）。
- 每个任务 1 个 nominal 加 50 个 ±2 档扰动状态。
- 和该任务的 ±2、±8 冻结测试文件逐状态比对，重复数为 0。

| 文件 | sha256 |
|---|---|
| `system1/t10b/validation_states/table_validation_states.json` | `87a716562282bb838e253fa7578c28d9839fba6ac0a981f4f01368f06e50670d` |
| `system1/t10b/validation_states/bucket_validation_states.json` | `a2222d1fb1c7ba325bfcc87f16fa4619fa10dc8cc212d0dbe1e9e0a5cc502d0d` |

## 5. 试点训练脚本（已准备，未运行）

`system1/t10b/pilot_cloud.py`：每个 run 占一张卡。命令见 `pilot_commands_dry_run.txt`；那是本机 dry-run 的输出，云端的 Python 路径换成 `$S1_PY`。

- **D Table 721：**
  - 上限 20000 轮，`--num-envs 2048`，每 1000 轮存一次 checkpoint；
  - 每个 2000 的整数倍 checkpoint 在同一张卡上用验证集评一次：Isaac，确定性动作，`scripts/evaluate_paper_supplement.py`，协议同 `d_final_5070`。
- **C-in Bin 721：** 12000 轮，其余同上；不用停止规则，用最后一个 checkpoint。
- **停止规则的执行方式（只用于 D Table）：**
  - 设最近一次验证的轮数为 `last`，要求 `last − 4000 ≥ 8000`，也就是 4000 轮的观察窗口整个落在 8000 轮之后。
  - 满足时，若窗口内（`last − 4000`, `last`] 的两个验证完成数都没有超过此前（≤ `last − 4000`）的最好值，就用启动时记下的进程号发 SIGINT 停止训练。
  - 按这个写法，最早在 12000 轮的验证之后停。
  - 最终用停下时最后保存的 checkpoint。
- 测试状态和 MuJoCo 结果都不在这个脚本里读取。
- 记录文件：
  - 每个 run 的 `launch.json`，含命令、PID、git HEAD、status、diff；
  - `validation/it*/`：评测器的完整输出；
  - `validation_curve.json`、`decision.json`。

## 6. 和任务书的偏离

1. **checkpoint 间隔：** 原 D Table 训练每 2000 轮存一次（`save_interval` 2000）；试点按任务书每 1000 轮存一次。原 C-in Bin 本来就是 1000。
2. **代码版本：** 原 D Table 是在 `holosoma-core4d-base` 的 `a21858b` 上训练的；试点用本分支代码。`run_config` 中奖励、终止、观测、网络和 PPO 参数逐键相同，见第 2 节。
3. **D 冒烟用的 `randomization.py` 少一行日志：** D 冒烟用的是 `2bf2c9a` 提交的版本（sha256 `ae53e03e…`）。C-in 冒烟前我加了一行日志，记录全部 env 中延迟为 1 的比例（sha256 `1726a8a1…`），没有改任何随机化逻辑。试点用的是带这行日志的版本，与本次提交一致。
4. **停止规则：** “8000 轮以后连续 4000 轮”按观察窗口整个落在 8000 轮之后来执行，即最早在 12000 轮停（第 5 节）。这是我对条文的读法，请 Kevin 确认。
5. **没有在 Isaac 里实际跑验证回合**测试随机化 checkpoint，原因见第 3 节；只做了元数据和参数形状的比对。

## 文件

- `smoke_summary.json`：两次冒烟的结果，冒烟和原 checkpoint 的 sha256，`randomization.py` 的 sha256，验证集 sha256。
- `smoke/`：每次冒烟的 `launch.json`（命令、git HEAD、diff）、`exit.json`、`randomization_samples.json`、`metrics.jsonl.gz`。
- `configs/`：冒烟 `run_config`、原训练 `run_config`（gzip）、逐键 diff。
- `pilot_commands_dry_run.txt`、`SHA256SUMS`。
- 生成脚本：`system1/t10b/package_retrain_smoke.py`。
