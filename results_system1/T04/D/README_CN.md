# T04 返包（第二部分）：D 原质量策略在其他质量档的直接评测

本文件只写事实。这部分是 T04"额外评测"一节：原质量的 D 策略直接放到其他档位上评测，不重训。它反映的是策略对质量变化的鲁棒性，与"重训后能搬多重"分开报告。重训的 6 个 D run 和 6 个 C run 还在云端训练。

## Receipt

- **任务书：** `docs/system1/T04_mass_sweep_zh.md`，sha256 `3d4f2787d6b82f34fae0ca1405bd4bd87bc489ccfccfcbe3a642c0419f9859f4`。
- **机器：** hostname `kevin-Legion-R9000P-ADR10`，GPU NVIDIA GeForce RTX 5070 Laptop。Python 环境 `hssim_s1`。
- **时间：** 2026-11-10 16:10:20 – 21:06:43（America/New_York）。
  - 主队列：`system1/t04/eval_d_direct.py`。
  - 从 17:39 起另开第二个队列，倒序跑 Bin 的 6 组，与主队列并行。主队列遇到已完成的组会跳过。
  - Pull 40 kg ±8 单独重跑，见下文"偏离"。
- **代码：** 18 组启动时的 git HEAD 不全相同，每组的 launch 记录里都存了 HEAD、`git status` 和 `git diff`。评测脚本的版本见下文"偏离"。

## 做了什么

- **策略：** 论文 D，种子 721 的最终 checkpoint，不重训。

  | 任务 | checkpoint | sha256 |
  |---|---|---|
  | Pull | `model_08050.pt` | `727630e9…` |
  | Table | `model_12000.pt` | `f163d11a…` |
  | Bin | `model_12000.pt` | `7ba039fb…` |

- **档位：** 只评测非原质量的档位，各 ±2 和 ±8 两档，每组 51 个回合。

  | 任务 | 评测档位 | 原质量 |
  |---|---|---|
  | Pull | 5、10、40 kg | 20 kg |
  | Table | 2.5、10、20 kg | 5 kg |
  | Bin | 2、4、8 kg | 1 kg |

  - 原质量档直接用 V3 收尾包里 Full 变体、种子 721 的行，数值照抄。
- **协议：** 与方案 A 和 T04-A 相同：GPU PhysX，TF32 关，初始状态用冻结文件。
  - 质量变体与 T04-A 是同一组 URDF：质量和惯量按同一比例缩放，其他内容不变。每组读回的质量都等于档位值。
- **checkpoint 约定检查：** `--checkpoint-mass original`，即按 checkpoint 训练时的原质量做检查。

## 结果

完成数 / 50 个扰动回合（nominal 不计）：

| 任务 | 质量 | ±2 | ±8 | 来源 |
|---|---|---|---|---|
| Pull | 5 kg | 49 | 37 | 本次 |
| Pull | 10 kg | 49 | 44 | 本次 |
| Pull | **20 kg（原）** | 50 | 43 | V3 |
| Pull | 40 kg | 2 | 1 | 本次 |
| Table | 2.5 kg | 0 | 2 | 本次 |
| Table | **5 kg（原）** | 35 | 46 | V3 |
| Table | 10 kg | 0 | 0 | 本次 |
| Table | 20 kg | 0 | 0 | 本次 |
| Bin | **1 kg（原）** | 50 | 38 | V3 |
| Bin | 2 kg | 50 | 40 | 本次 |
| Bin | 4 kg | 49 | 25 | 本次 |
| Bin | 8 kg | 0 | 0 | 本次 |

- **Table 各档的终止原因：** 非原质量档里没完成的回合，终止原因都是 `joint_bad_tracking`。Table 2.5 kg 在 ±2 档的 50 个回合全部如此。
- **GPU 型号：**
  - 本次的 18 组都在 RTX 5070 Laptop 上跑。
  - 原质量点：±2 档是 V3 在本机跑的，也是 5070 Laptop；±8 档来自 V3 云端，是 RTX 4090。
  - 汇总表 `mass_sweep_summary.csv` 有 `gpu_model` 列，逐行写明。
- **`delta_cm`（B1 的接触偏移 δ）：** 对 D 不适用，为空。

## 文件

| 文件 | 内容 |
|---|---|
| `D/mass_sweep_episodes_D.csv` | 本次 18 组和 V3 原质量点各 6 组，共 24 组 × 51 = 1224 行。列沿用 V3 `results/episodes.csv`，另有 `scheme`、`mass_kg`、`original_mass_kg`、`mass_ratio`、`retrained`、`seed`、`delta_cm`、`launch_git_head`、`gpu_model`、`gpu_model_source` |
| `D/mass_sweep_summary_D.csv` | 每个任务、质量和扰动档一行 |
| `D/provenance.json` | 手部 FK 自检、参考的 sha256、V3 表的 sha256、状态计数 |
| `D/evaluator_versions.json`、`D/evaluator_versions/*.diff` | 每组启动时的评测脚本版本，以及它与当前版本的 diff |
| `mass_sweep_episodes.csv`、`mass_sweep_summary.csv`（上一级目录） | A 和 D 合在一起 |

## 与任务书的偏离

- **Pull 40 kg ±8 跑了两次。**
  - 第一次在 18:09:10 启动，1 秒后退出（返回码 −6）：导入 torch 时报内部断言 `ApproximateClock ... getCount is non-monotonic`，没有跑任何回合。
  - 记录改名保留为 `ops/pull_m40kg_pm8.attempt1_torch_clock_abort.*`。
  - 20:08 用同样的参数重跑，完成。
- **18 组用了 4 个版本的评测脚本。** 评测期间，评测脚本加了方案 B 的分支和 GPU 型号字段；其中 2 组启动时，评测脚本还有未提交的改动。
  - 每组的脚本版本 = 启动时 HEAD 中的脚本 + launch 记录里的未提交 diff。按此重建每组的脚本，再与当前版本比较：差别只有 `sonic_planner`（方案 B）分支、`--b-variant` 参数、`gpu_name` 字段，以及把 `controller!='sonic'` 改写成 `controller not in ('sonic','sonic_planner')`。
  - 对 `--controller policy`，这些改动前后执行的代码相同。
- **其余：** 没有其他偏离。
