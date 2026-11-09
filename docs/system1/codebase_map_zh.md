# DuetMimic 代码地图（T00，只读核对）

核对时间：2026-11-09 15:32–15:52 EST。执行方：Ubuntu 本机 Claude Code。
依据：T00 任务书 sha256 `5f235f10183c02b3296c7174b0f450aaeeaf77397bd57252e67ca151f674b444`。

- 只写从代码、配置、日志里读到的内容；读不到或有矛盾的写"未确认"，集中列在第 12 节。
- 路径约定：
  - 仓库内路径相对于仓库根，例如 `src/holosoma/holosoma/...`。本分支 `system1-sonic` 的 HEAD 与 core4d-base 已提交代码在 `.py` 上一致（见第 2 节）。
  - 旧目录用绝对路径，`~` 指 `/home/kevin`。
  - `B=~/holosoma-core4d-base`，`M=~/holosoma-rubber-hand-marl`，`L=~/holosoma-rubber-hand-largetable`。
  - `V3=$B/paper_handoff_exports/V3_20261102`。
- 行号按本分支文件，核对于 HEAD `c7e763f`。

---

## 1. 机器与环境

### 1.1 本机

| 项 | 值 |
|---|---|
| hostname | `kevin-Legion-R9000P-ADR10` |
| OS | Ubuntu 22.04.5 LTS，内核 6.8.0-138-generic |
| CPU / 内存 | 32 逻辑核 / 31 GiB，无 swap |
| GPU | 1 × NVIDIA GeForce RTX 5070 Laptop GPU，8151 MiB |
| 驱动 / CUDA | 驱动 580.178.04；`nvidia-smi` 显示 CUDA 13.0 |
| 磁盘 | `/` 345 G，剩余 101 G |

### 1.2 conda 环境

conda 根目录是 `~/.holosoma_deps/miniconda3`，不在 PATH 上，系统里没有 `python` 命令。

| 环境 | Python | torch | 仿真相关 |
|---|---|---|---|
| `hssim` | 3.11.15 | 2.7.0+cu128（CUDA 12.8，可用） | isaacsim 5.1.0.0（全套子包），isaaclab 0.47.2，isaaclab_assets 0.2.3，isaaclab_rl 0.4.4，isaaclab_tasks 0.11.6，rsl-rl-lib 3.0.1，warp-lang 1.10.0，numpy 1.26.0。未装 IsaacGym。 |
| `hsretargeting` | 3.11.15 | 2.12.1+cu130 | mujoco 3.10.0，numpy 2.3.5；无 Isaac。 |

**注意 editable 安装：**
- `hssim` 里的 `holosoma` 以 editable 方式指向 `/home/kevin/holosoma/src/holosoma`，即 `push-bigbox-experiment` worktree，不是 core4d-base。
- `hsretargeting` 里的 `holosoma-retargeting` 同样指向 `/home/kevin/holosoma/src/holosoma_retargeting`。
- 训练和评测脚本自己会把 `<脚本所在根>/src/holosoma` 和 `src/holosoma_retargeting` 插到 `sys.path[0]`（例如 `scripts/train_plan5_push.py:17-19`、`scripts/train_core4d_smalltable.py:20-22`、`$B/scripts/evaluate_paper_supplement.py:12-14`），所以实际 import 的是脚本所在根的代码。
- 不经过这些脚本、直接 `import holosoma` 时，拿到的是 `/home/kevin/holosoma` 的代码。

### 1.3 云端（AutoDL）

- **V3（2026-11）：** `$V3/EXECUTION_REVIEW_20261102T215303Z/dispatch_receipt.json` 记录了云端环境：
  - 2 × RTX 4090；
  - Python 3.11.17，torch 2.7.0+cu128；
  - isaacsim 5.1.0.0，isaaclab 0.47.2；
  - 解释器 `/root/autodl-tmp/r2_deps/hssim/bin/python`。
- **R2（10 月）和 PaperSupplement（9/30–10/01）：** 云端解释器分别是 `/root/autodl-tmp/r2_deps/hssim` 和 `/root/autodl-tmp/core4d_deps/hssim`，来自 launch 记录。
  - 两者的 Isaac 版本和 GPU 型号：未确认。

### 1.4 论文主结果的仿真后端

- **结论：** Isaac Sim / Isaac Lab（PhysX）。
- **依据：**
  - run_config 没有显式的 `simulator` 字段。
  - 六个任务的配置都用 `simulator.isaacsim_dual_robot`，即 `_target_="holosoma.simulator.isaacsim.dual_robot_isaacsim.DualRobotIsaacSim"`：
    - Plan5 系列见 `src/holosoma/holosoma/config_values/marl/g1/experiment.py:41-51`、`kick_experiment.py:28-38`；
    - CORE4D 系列见 `core4d_smalltable_experiment.py:69-81`；
    - 定义在 `src/holosoma/holosoma/config_values/simulator.py:58-61`。
  - run_config 里的 `runtime_object_physics` / `object_physics` 是用 Isaac Lab 的 `root_physx_view.get_masses()/get_inertias()/get_coms()/get_material_properties()` 读回的（`scripts/train_plan5_push.py:261-300`、`scripts/train_core4d_smalltable.py:242-262`）。
  - 评测 manifest 的 `physics` 字段也来自同一接口（`$B/scripts/evaluate_paper_supplement.py:121-122`）。
- **本机训练：** 主结果的 721 种子是否在本机训练，未确认（见第 12 节）。

---

## 2. 三个旧目录的 git 状态

**结构：** 三个旧目录、`/home/kevin/holosoma` 和 `~/holosoma-push-largetable` 都是同一个仓库 `/home/kevin/holosoma/.git` 的 worktree。
- remote `origin`：`amazon-far/holosoma`；
- remote `private`：`Kaifeng-Jia/holosoma-rubber-hand-marl`，本目录 `duet-system1` 的 origin 也是它。

| 目录 | 分支 | HEAD | HEAD 是否在 GitHub | 修改的文件 | 未跟踪 |
|---|---|---|---|---|---|
| `$B` core4d-base | `core4d-base` | `7df11c92`（2026-11-07） | 是：`private/core4d-base` 的祖先。远端 core4d-base 是 `6acb096`，本地落后 1 个提交，该提交只改了 `docs/experiments/ubuntu_push_placement_more_seeds_20261107_zh.md` | 8 个，代码 6 个，见下 | 115 个顶层条目，代码文件见下 |
| `$L` rubber-hand-largetable | `rubber_hand_largetable_baseline` | `612713bc`（2026-08-12） | **否**：两个远端都没有。`git ls-remote` 两个 remote 都不含它，本目录 fetch 后也不存在该对象 | 6 个，全是代码，见下 | 0 |
| `$M` rubber-hand-marl | `rubber_hand_marl_baseline` | `c0102ace`（2026-09-04） | 是：`private/rubber_hand_marl_baseline` | 0 | 0 |

### 2.1 core4d-base 未提交的修改

共 8 个，其中代码 6 个；`git diff --stat` 为 +178/−10。

- `scripts/train_plan5_push.py`（+47）：新增开关：
  - `--push-placement-proof`；
  - `--from-scratch`、`--zero-teammate-obs`（V3 Pull 消融）；
  - `--episode-telemetry`；
  - 把 `Plan5PPO` 换成 `V3Plan5PPO`；
  - checkpoint 写入 `plan5_skill`、`placement_provenance`。
- `scripts/train_core4d_smalltable.py`（+19）：新增 `--reference-variant`（V3 Chair/Bucket 参考变体）。
- `scripts/smoke_core4d_smalltable.py`（+4）：同上。
- `scripts/export_recorded_demo_media.py`（+29）。
- `src/holosoma/holosoma/agents/mappo/core4d_smalltable_evaluation.py`（+8）：`deterministic_core4d_smalltable_actions` 增加 `zero_teammate_obs` 关键字参数。
  - **`evaluate_paper_supplement.py:146` 会用到这个参数；本分支的已提交版本没有它。**
- `src/holosoma/holosoma/agents/mappo/initialization.py`（+5）：`initialize_plan5_model_bundle` 增加 `from_scratch`。
- 另外 2 个是 md 文档。

### 2.2 core4d-base 未跟踪的代码文件

不含 `paper_handoff_exports/` 和 `CORE4D_UBUNTU_ENV_AUDIT*/` 里的副本。

- `scripts/`，59 个。与论文结果直接相关的有：
  - `evaluate_paper_supplement.py`：六任务统一评测；
  - `paper_evaluation_utils.py`；
  - `plan5_registered_placement.py`：Push 配准站位，评测和训练都要 import；
  - `prepare_experiments_r2_cpu.py`：手 FK，`BatchFK`；
  - `finalize_v3_results_20261104.py`、`export_v3_training_curves_20261104.py`；
  - `run_paper_evaluation_batch.py`。

  其余是 autodl/打包/渲染/V3 流程脚本：`audit_paper_supplement.py`、`autodl_*.py`、`backup_*.py`、`collect_*.py`、`finalize_*.py`、`finish_v3_local_20261104.py`、`measure_u2_shared_scene_handdist_cpu.py`、`organize_paper_evaluation.py`、`package_*.py`、`prepare_v3_*.py`、`push_2026110*_*.py`、`recompute_r2_wrist_cpu.py`、`reference_geometry_*.py`、`render_*.py`、`run_*.py`、`summarize_paper_evaluation.py`、`v314_*.py`、`verify_v3_pull_cpu.py`。
- `src/holosoma/holosoma/`：
  - `agents/mappo/v3_plan5_ablation.py`；
  - `config_values/marl/g1/v3_reference_variant.py`；
  - `utils/v3_episode_telemetry.py`；
  - 数据目录 `data/motions/g1_29dof/whole_body_tracking/v3_20261102/`。
- `tests/`，8 个：
  - `test_autodl_supplement_autorun.py`、`test_experiments_r2_cpu.py`、`test_paper_evaluation.py`、`test_plan5_registered_placement.py`；
  - `test_reference_geometry_pilot.py`、`test_reference_geometry_round2.py`；
  - `test_u2_shared_scene_handdist_cpu.py`、`test_wrist_reference_materials.py`。

### 2.3 rubber-hand-largetable 的修改

6 个代码文件，+63/−1，全部属于 object-goal 线：
- `agents/callbacks/recording.py`
- `config_types/experiment.py`
- `config_values/object_goal/g1/command.py`
- `eval_agent.py`
- `managers/command/terms/object_goal.py`
- `managers/object_goal/tests/test_object_goal_v1.py`

### 2.4 旧目录 HEAD 与 core4d-base 的代码差异

- **比较方式：**
  - 不算文档，只比 `*.py *.sh *.yaml *.toml`。
  - 用 `7df11c92` 代替 `6acb096` 比较，两者只差一个 md 文件。
- **`$M` c0102ace：**
  - 是 `7df11c92` 的祖先，中间 41 个提交，98 个代码文件不同。
  - 主要是新增文件，即 CORE4D 整条线：`scripts/*core4d*`、`src/holosoma/.../core4d_*`、`src/holosoma_retargeting/...core4d*`、对应 tests。
  - 修改的已有 `.py` 文件：
    - `config_types/robot.py`、`config_types/simulator.py`
    - `managers/observation/terms/marl.py`、`managers/reward/terms/marl.py`
    - `simulator/isaacsim/isaacsim.py`
    - retargeting 下的 `config_types/retargeter.py`、`examples/robot_retarget.py`、`src/interaction_mesh_retargeter.py`、`src/utils.py`
    - `tests/retargeting/test_pt_wrist_orientation.py`
- **`$L` 612713bc：**
  - 与 core4d-base 的分叉点是 `8038c092`（"milestone(wbt): freeze four rubber-hand action priors"）。
  - 只多 2 个提交：`1e8732b7` A1 goal-conditioned bridge、`612713bc` object-goal eval state，共 18 个文件 +1179，不在 core4d-base 里。
  - 到 core4d-base 有 274 个代码文件不同，以新增为主：Plan5/demo3/demo4/CORE4D 全部脚本和模块。
- **本分支与旧代码：**
  - `system1-sonic` HEAD 的 `.py` 与 `2b6668b1`（2026-09-25）完全一致：`git diff --stat 2b6668b1 HEAD -- '*.py'` 为空。
  - `2b6668b1` 是 PaperSupplement 和 V3 云端代码快照的 base（见第 4.3 节）。

---

## 3. 任务名对照

| 论文任务 | 代码/日志里的名字 | 示范来源 | 训练入口 | env 类 | 配置对象 |
|---|---|---|---|---|---|
| Table Pushing | Plan5 push，`--skill push`；模型 ID `Push-registered`；评测 case `push` | OMOMO 单人 `sub6_largetable_033`，角色等距补出同伴（配准间距 0.8780472 m） | `scripts/train_plan5_push.py --skill push` | `holosoma.envs.marl.plan5_push_manager.Plan5PushManager` | `g1_29dof_plan5_push_baseline`（`config_values/marl/g1/experiment.py:74`）+ `plan5_registered_placement.configure_push_placement` |
| Table Pulling | Plan5 pull，`--skill pull`；`Pull-baseline`；`pull` | OMOMO 单人 `sub3_largetable_010`，镜像角色 | `scripts/train_plan5_push.py --skill pull` | 同上 | `g1_29dof_plan5_pull_baseline`（`experiment.py:124`） |
| Table Kicking | Plan5 kick，`--skill kick`；`Kick-baseline`；`kick` | OMOMO 单人 `sub16_largetable_028`，镜像 | `scripts/train_plan5_push.py --skill kick` | 同上 | `g1_29dof_plan5_kick_baseline`（`kick_experiment.py:68`） |
| Chair Righting | CORE4D `chair021`；`C-01`；`chair` | CORE4D 双人 `20231020/074` | `scripts/train_core4d_smalltable.py --experiment chair021 --reward-variant baseline` | `holosoma.envs.marl.core4d_smalltable_manager.Core4DSmallTableManager` | `with_pair_experiment(g1_29dof_core4d_smalltable_smoke, get_core4d_pair_experiment('chair021'))` |
| Bin Transport | CORE4D `bucket003`；`B-RZ0`；`bucket` | CORE4D 双人 `20231020/071` | `train_core4d_smalltable.py --experiment bucket003 --reward-variant bucket_A_no_rel_no_height` | 训练：未确认是否用 `core4d_bucket_manager.Core4DBucketManager`。评测：用它，并设 `enable_object_hand_contact=True`（`evaluate_paper_supplement.py:44-46`） | 同上，`bucket003` |
| Table Transport | CORE4D `smalltable5kg_A`；`T-07`；`table` | CORE4D 双人 `20231030/001` | `train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height` | 评测：`Core4DBucketManager`；训练：未确认 | 同上，`smalltable5kg_A` |

- **runner / PPO：**
  - Plan5 训练用 `Plan5PPO`（`agents/mappo/ppo.py`），初始化走 `initialize_plan5_model_bundle`（`agents/mappo/initialization.py`），即从单人先验 warmstart actor。
  - CORE4D 训练用 `Core4DSmallTablePPO`（`agents/mappo/core4d_smalltable_ppo.py`，继承 `Plan5PPO`），初始化走 `initialize_core4d_smalltable_model_bundle`（`agents/mappo/core4d_smalltable_initialization.py`），是 fresh 初始化。
  - 评测六个任务都用 `initialize_core4d_smalltable_model_bundle` 加 `deterministic_core4d_smalltable_actions`。
- **不在论文六任务里：** `demo3`（tug，`train_demo3_tug.py`）、`demo4`（rotate，`train_demo4_rotate.py`）、`core4d_smalltable`（20 kg 原版小桌）、Bucket 的 `B-A` 等变体。本次没有找到它们和论文六任务的对应关系。
- **Fig.3 来源时刻：** 六任务的源序列和首帧见 `$V3/V3_CLOSEOUT_SUPPLEMENT_20261105/README_CN.md` 第 4 节。

---

## 4. 论文主结果的来源（Full 配置，种子 721/722/723）

**来源依据：**
- `$V3/V3_GPU_CLOSEOUT_20261104/results/source_index.json`（variant=Full）；
- 各评测目录的 `run_manifest.json`；
- `$V3/U5_F3_EXEC_20261104/jobs.json`。

18 个最终 checkpoint 本次全部在本地找到，逐个重算 sha256，都与评测 manifest 一致。

### 4.1 每任务一行

| 任务 | 训练命令（模板，取自 launch 记录） | 训练分段 | 运行时参考（相对 `src/holosoma/`）与 sha256 | 物体 URDF 与 sha256 | 冻结初始状态 ±2 / ±8 与 sha256 |
|---|---|---|---|---|---|
| Push | `train_plan5_push.py --skill push --seed S --num-envs 2048 --steps-per-env 24 --save-interval 1000 --actor-learning-rate 0.00001 --critic-learning-rate 0.001 --push-placement-proof <r2_evidence/preflight_push_registered.json>`；stage0 加 `--iterations 50 --critic-only`，stage1 `--iterations 8000 --resume <stage0/model_00050.pt>`，stage2 `--iterations 5000 --resume <stage1/model_08050.pt>`。来源：`$B/paper_handoff_exports/EXPERIMENTS_R2_20261026/sources/training_operations/push_seed72*_stage*.launch.json` | 50 critic-only + 8000 + 5000 → 13050 | 无 paired runtime 文件。用单人参考 `holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz`（`b669468a…ba2207`），由 `PairedA1Reference` 在线生成双人参考，间距 0.8780472 m | `.../objects_widetable_plan5_training.urdf`（`c6de78a4…b414c2a`） | ±2：`$B/logs/ExperimentsR2_evaluation_20261026_retry01/U5_push_registered/push_states.json`（`1799ea1e…ef895d`，与 `$B/logs/PaperSupplement20260929/unified_evaluation_20261003/push_states.json` 同 hash）；±8：`$B/logs/V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/states/push_pm8.json`（`a58b2fae…6c266`） |
| Pull | 同一脚本，`--skill pull`，`--actor-learning-rate 0.00001 --critic-learning-rate 0.001`；stage0 `--iterations 50 --critic-only`，stage1 `--iterations 8000 --resume <stage0/model_00050.pt>`。来源：`$B/logs/PaperSupplement20260929/autodl_received/operations/Pull_seed72[23]_stage*.launch.json`。721 的原始命令本次没有找到；run_config 记录 `start_iteration=50`、`optimizer_state_restored=True` | 50 critic-only + 8000 → 8050 | `.../rubber_hand_largetable_v1/pull/plan5_attempt08_mirrored_pair_runtime.npz`（`571f2305…8404d4`）；单人源 `.../pull/sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz`（`7c1c5548…d8cf7`） | `.../objects_widetable_plan5_pull_training.urdf`（`c260652d…a8c2`） | ±2：`$B/logs/PaperSupplement20260929/unified_evaluation_20261003/pull_states.json`（`473c1fc0…50c572`）；±8：`.../states/pull_pm8.json`（`3c2a60c7…c8d5b2`） |
| Kick | 同一脚本，`--skill kick --iterations 8000 --actor-learning-rate 0.001 --critic-learning-rate 0.001`，无 critic-only 段。来源：`.../EXPERIMENTS_R2_20261026/sources/training_operations/kick_seed72[23]_stage0_from0_plus8000.launch.json`。721 的原始命令未找到；原运行目录是 `$M/logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/`，旁边有 `.console.log` | 8000 | `.../rubber_hand_largetable_v1/kick/plan5_attempt09_dual_kick_mirrored_runtime.npz`（`8eba6106…ddd87d`） | 与 Pull 相同的 `objects_widetable_plan5_pull_training.urdf` | ±2：`$B/logs/ExperimentsR2_evaluation_20261026_retry01/U6_kick/kick_states.json`（`c1ac57c6…7527c4`）；±8：`.../states/kick_pm8.json`（`dba247ca…79319`） |
| Chair | `train_core4d_smalltable.py --experiment chair021 --reward-variant baseline --num-envs 2048 --steps-per-env 24 --seed S --iterations 12000 --save-interval 2000`。来源：`.../autodl_received/operations/C01_seed72[23]_*.launch.json` | fresh 12000 | `.../core4d_chair021_20231020_074_a1/core4d_pair_runtime_fps50.npz`（`9de64128…350ba5`） | `.../core4d_chair021_20231020_074_a1/chair021_training.urdf`（`78496902…69ee`） | ±2：`.../unified_evaluation_20261003/chair_states.json`（`5bb8ff9f…bc3`）；±8：`.../states/chair_pm8.json`（`f9a144ed…a5af7`） |
| Bucket | 同上，`--experiment bucket003 --reward-variant bucket_A_no_rel_no_height` | fresh 12000 | `.../core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz`（`071ac78b…b23a8`） | `.../core4d_bucket003_20231020_071_a1/bucket003_training.urdf`（`6c0ff9f6…cde`） | ±2：`.../bucket_states.json`（`631026d3…75d71f`）；±8：`.../states/bucket_pm8.json`（`3bae05a1…e6ff0`） |
| Table | 同上，`--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height`。722/723 分两段：`T07_seed72X_fresh12000_cache_isolated02` 跑到 10000 中断，再 `--iterations 2000 --resume .../model_10000.pt` | 721：fresh 12000；722/723：10000 + 2000 | `.../core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz`（`582e7669…ee153`） | `.../core4d_smalltable5kg_A/desk001_5kg_training.urdf`（`942cea0e…e9695d`） | ±2：`.../table_states.json`（`389e8da5…dac1`）；±8：`.../states/table_pm8.json`（`6a626f48…02fcad`） |

- **完整 sha256：** 上表为了排版截断了 sha256。完整值是：
  - 参考：Pull `571f23055f06648fb30b7abe5d90746a80c8555461ef91b6a23bccd9ad8404d4`；Push 单人 `b669468ade12f8c271119d6c8c120463223ab65489805c486e43b0c3a4ba2207`；Kick `8eba6106bf009b3561d61a4916d7cd4db7ed06d089cb5da14104ff9d15ddd87d`；Chair `9de64128209230229e40a675ed6b204e1c4b7655a81cbef773f76557a0350ba5`；Bucket `071ac78bac440fee72a2cf861e1c00ba3df9531b371292d930e04d61aa8b23a8`；Table `582e76693f877c61b0b09ab3b584922f330aeb85cae6035f6b7cd2ae729ee153`。
  - 物体 URDF：Push `c6de78a48eb7a35454a710ab77369e138112b2f62d9efa189f6ca7c05b414c2a`；Pull/Kick `c260652d23e31dc0978a167137c45727cddb156eebedab77c9a40aa30079a8c2`；Chair `78496902eb84489011d2a5241e32994cebb1ea3784b534bf8cd75151824469ee`；Bucket `6c0ff9f60d8ef7ef057a988f4e95a6b89cd7e0f5ae55ea9e6697596955f28cde`；Table `942cea0e35e45c67ba0377a42b382a28e2e52ba5b05238a3066697e9d1e9695d`。
  - 机器人 URDF：`src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf`，`7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25`，六任务相同。
  - 以上参考和 URDF 在本分支都已被 git 跟踪，与 `$B` 逐字节一致。
- **冻结状态 ±2：**
  - push `1799ea1eec1a3ff097d46e08e6492694d6455490ccf9f2de26586e079fef895d`
  - pull `473c1fc0563228a770393c2ccd78f3b051df258a408e6481d44c12fa2450c572`
  - kick `c1ac57c62345363c85f96f98363d778898a23e1f35a2596fbcd33a68367527c4`
  - chair `5bb8ff9f1cb7adb72b784c0d8475779cf68099aba551901114a2f78625960bc3`
  - bucket `631026d39f436ceca4008e2f5e684fff16102fba8895fafd23ae3639b575d71f`
  - table `389e8da5fb818572a48cbba049254585311079439265f5ab4cfc1523a185dac1`
- **冻结状态 ±8：**
  - push `a58b2fae20f488c082f1ac07c7d552cc85053e6d5bc8360a60df2a723d26c266`
  - pull `3c2a60c7982db92b59b1f3343103e711409a6c756faaf83b86016b8163c8d5b2`
  - kick `dba247caf71d5a80213f5eb96dd8c7a32ba0ff75d8fe2dd0f18411753c779319`
  - chair `f9a144ed0738975fef4453047c48382cee383b7daaa680e4cf43b468e8ea5af7`
  - bucket `3bae05a1b1d716d97c7025386dc3d7f832a7203a665d33f4f9979b3adb6e6ff0`
  - table `6a626f48c14598a646d40d35694e32b5be198ea09491b7f3557ab9c1fe02fcad`
  - ±4、±6 两档在同一 `states/` 目录。
- **状态文件格式：** `relative_initial_state_v1`。
  - 51 条：1 条 nominal + 50 条 perturbed。
  - 每条只有每台机器人的世界 XY 偏移 `robot_xy_m[2][2]` 和绕自身根的 yaw `robot_yaw_rad[2]`。
  - 物体不扰动，关节姿态和根高度不变。
  - ±8 档由 ±2 档同一组均匀抽样乘以 4 得到（`v3_provenance.scale_from_pm2=4.0`），`generation_seed=241003`。

### 4.2 18 个最终 checkpoint

| 任务 | seed | 最终 checkpoint（本地） | ckpt sha256 | run_config sha256 | 迭代 |
|---|---|---|---|---|---|
| Table Pushing | 721 | `~/holosoma-core4d-base/logs/ExperimentsR2_retry1/push_seed721_stage2_from8050_plus5000/model_13050.pt` | `f645adeadb60e047ec9c2b59ddd4d0ca311f11f01bbfb930dc26d4f1e1db62cb` | `2c7a4af032b1db2b0c9538ed4384bbbb11ab313797c5b85faf03ae7dbde7af4d` | 13050 |
| Table Pushing | 722 | `~/holosoma-core4d-base/logs/ExperimentsR2_retry1/push_seed722_stage2_from8050_plus5000/model_13050.pt` | `af0290071d24e7510b2c1beafada948bd2281d96c9b0df9bf77e49d6bcd1ac16` | `9681cd42ac1f12961386dcccaa3b39b77c73efe0620b79e32d5644894fee1fef` | 13050 |
| Table Pushing | 723 | `~/holosoma-core4d-base/logs/ExperimentsR2_retry1/push_seed723_stage2_from8050_plus5000/model_13050.pt` | `da4ec24cbe03a1d6416ab9579130e02ca2ad861c5ba10ad86582fb35c89ef2f8` | `3a5859f93ae2acfab4128974a64946e8ab0927e75c03e9908862701c8d9ca3ef` | 13050 |
| Table Pulling | 721 | `~/holosoma-rubber-hand-marl/logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt` | `727630e9cec654d88bfb454d5eaabd70d7db629478598a2039140653a57cbf43` | `bcaa2ed0d414e531655b7c7ce3df8980a339a793d3fc086a94a001d5882467bb` | 8050 |
| Table Pulling | 722 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/Pull_seed722_stage1_from50_plus8000_20261001/model_08050.pt` | `d6e545e5a1628aba4816a8cfb80c2fd1a962938e520186298b7b4157b16c85f7` | `1be1e9e721a32e13dad2f6e3988f347a11e3d2d00d7d36f66a44dd8a10f9b8b8` | 8050 |
| Table Pulling | 723 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/Pull_seed723_stage1_from50_plus8000_20261001/model_08050.pt` | `e4e5eaed6dba0a34e64305ec4c358759a44c80abdb2c0d840943fac53376c7ac` | `c837968ac29652956dd30ae83a82b63ca3511786868b76009f55b3ba316cf790` | 8050 |
| Table Kicking | 721 | `~/holosoma-core4d-base/logs/ExperimentsR2/deployment_20261024/evaluation_20261026/cloud_ops/historical_kick721/model_08000.pt`（与 `~/holosoma-rubber-hand-marl/logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/model_08000.pt` 同 hash） | `9c6a31809ba69eaf3fd510391ce5b5f56069d4378c514ce09c3b4f28e407b3a3` | `92a3412a5479ba3d67e0c12a6bc8d4e6ca60190930d4929b9c17ad41fc7f22d4` | 8000 |
| Table Kicking | 722 | `~/holosoma-core4d-base/logs/ExperimentsR2_retry1/kick_seed722_stage0_from0_plus8000/model_08000.pt` | `18ce1cb9e762d79133785cde7614fd0a2115ff3b224674aef62430878f16f412` | `540a4f3b2bf01999bb0ecb34e6d692f6cee5ceae6c901bc2216f05b13d194d2e` | 8000 |
| Table Kicking | 723 | `~/holosoma-core4d-base/logs/ExperimentsR2_retry1/kick_seed723_stage0_from0_plus8000/model_08000.pt` | `b521f7e45f25cf82cce610f8b8208ac4faf3b23a280d3dbaeafe65a945354719` | `5db3c372fcc73feca301a55f840b437e42c8233b4393d79df6779bbb25ff3576` | 8000 |
| Chair Righting | 721 | `~/holosoma-core4d-base/logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/model_12000.pt` | `03f6c8c0edbce762e2ef55bf8760414bbf54371e62dfb04071829a5ce3e457f7` | `244eb0b03d61b9ca095db6b4ae59faedf0b70ae6b548486f513c04d16b093a78` | 12000 |
| Chair Righting | 722 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/C01_seed722_fresh12000_concurrent_20260930/model_12000.pt` | `41c774d71cc5997e2c895f86ed59de452bb2f8fad576d18ca33305e6792c7d00` | `dc95a0156b9fc14ddf88c5b738ecb705bcae8e42c997d962e76734cdcb6cafc7` | 12000 |
| Chair Righting | 723 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/C01_seed723_fresh12000_concurrent_20260930/model_12000.pt` | `47395760932c87f05cb7a5072c30cf029a0428445b5fc2cba4c4bfcbc7e3d80c` | `de616296be928bc7a68e3c8dfc7062b09ae5ccb9ac10120690fb4de0851a4b65` | 12000 |
| Bin Transport | 721 | `~/holosoma-core4d-base/logs/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919/model_12000.pt` | `7ba039fbb608211f8b2866b4f80810c082313bca6232a028c07743391656a225` | `f07483a6dcd12696e0306c21cae1fe49939f7ddc7ce1dbaa4cb8f70e92bcd97c` | 12000 |
| Bin Transport | 722 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/BRZ0_seed722_fresh12000_concurrent_20260930/model_12000.pt` | `05850d5bcb26a2a01ef7939a6e7353484758ff949b8c861fb01c437a2b160892` | `89a28282450fc07bf049abf0f0aaa0a68beab08b6f9f05baaa1fbe7eae79003f` | 12000 |
| Bin Transport | 723 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/BRZ0_seed723_fresh12000_concurrent_20260930/model_12000.pt` | `f936aa17acb7ab790bbe8b19f4e14fbfcba7ddcd9f3f0e35fa96ef0ae2d08940` | `74b71ec66b878e211fa87951dadc9ff76c804e59d258f83f6043fa52301cb250` | 12000 |
| Table Transport | 721 | `~/holosoma-core4d-base/logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/model_12000.pt` | `f163d11a9629edae5dd72732c4ddec08a43024533a47b349a099a3275a8d1c50` | `1b0f4b953327dd1ffb23c3aaa36ea601229b30bae26658902db5d29b0fd6a330` | 12000 |
| Table Transport | 722 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/T07_seed722_resume10000_to12000_clone20260930/model_12000.pt` | `007161dc9ca5e431ac5367bc5655745ca2dcc367e2e9cdfad89eeb1c35c57867` | `5fbb8b8957a8d9869125c0f002fe49d7c2ea6bc719627100c190cb1c1ac31ef6` | 12000 |
| Table Transport | 723 | `~/holosoma-core4d-base/logs/PaperSupplement20260929/autodl_received/runs/T07_seed723_resume10000_to12000_clone20260930/model_12000.pt` | `df9bc014d1e4146d66df65c9f599f0cec99de9c09a20039026a6f4c164df7e26` | `32dca321e123bc81823ed9cd71b98f42218534bc32e72b8c8a5cd39c8d207bff` | 12000 |

- **run_config 位置：** 每个 checkpoint 同目录下的 `run_config.json`。
- **分段证据：** 各段的 run_config、metrics 和 hash 见 `$B/logs/PaperSupplement20260929/unified_evaluation_20261003/training_evidence/<模型ID>/stages.json`；Push/Kick 见 `$B/paper_handoff_exports/EXPERIMENTS_R2_20261026/sources/training_runs/`。
- **单人先验（Plan5 三任务的 warmstart actor）：**

| 任务 | 文件 | sha256 |
|---|---|---|
| Push | `$M/logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt` | `11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224` |
| Pull | `$M/logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt` | `f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364` |
| Kick | `$M/logs/WholeBodyTracking/marl_compat_kick_v1/model_07999_actor158.pt` | `1555968f678c2b69fcd6f09c64d0a6252683eab902edd773a84acc205d0f5491` |

  - 训练脚本里的路径写法是 `REPO_ROOT/logs/WholeBodyTracking/...`（`scripts/train_plan5_push.py:117/128/143`）。
  - CORE4D 三任务是 fresh 初始化，不用先验。

### 4.3 训练所用代码版本（run_config 的 `git_commit`）

| commit | 用到它的运行 | 本地是否有该 commit |
|---|---|---|
| `65f7ba5e` | Pull 721 | 有（`$M`） |
| `ba409225` | Kick 721 | 有 |
| `25c48aeb` | Chair 721 | 有 |
| `a21858b2` | Bucket 721、Table 721 | 有 |
| `b6944cff` | Push 721–723（stage2）、Kick 722/723 | **无**：R2 云端提交 |
| `7e28e9aa` | Pull 722/723、Chair 722/723、Bucket 722/723、Table 722/723 | **无**：PaperSupplement 云端提交 |

- **R2 云端代码：** 快照在 `$B/paper_handoff_exports/EXPERIMENTS_R2_20261026/sources/training_source_assets.tar.gz`，sha256 `208e60b8787f7fbcfda622b6c948001f6e352fbc20b29b9489d0d3f956046090`。补丁记录在 `sources/training_operations/source_patch_manifest.json`。
- **PaperSupplement 云端代码：** 快照在 `$B/logs/PaperSupplement20260929/deployment_v{1,2}/paper_supplement_source_assets.tar.gz`（v2 sha256 `243732c5…aac8e`）。
  - 打包时 base 是 `2b6668b1`，加当时工作区的未提交改动（`source_tracked_diff.patch`）。
  - 实际用的是 v1 还是 v2：未确认。

### 4.4 评测

- **脚本：** `$B/scripts/evaluate_paper_supplement.py`，未跟踪；sha256 `56c6450cb1d71d8fef3d7f114eb10255492581cba19a389d40eed73ac25f1f81`，与 V3 launch 记录里的 `evaluator_sha256` 一致。
- **依赖：** `$B/scripts/paper_evaluation_utils.py`（`051d626c…fa6`）和 `plan5_registered_placement.py`。
- **命令：** `python -B scripts/evaluate_paper_supplement.py --case <push|pull|kick|chair|bucket|table> --checkpoint <ckpt> --states <states.json> --model-id <id> --output-dir <dir>`。Push 另加 `--push-placement-proof`。
- **±2 档结果目录：**
  - Pull/Chair/Bucket/Table：`$B/logs/PaperSupplement20260929/unified_evaluation_20261003/results/<ID>_seed72X/`。Table 721 的目录名是 `T-07_seed721_retry_shutdown_20261004`。
  - Push：`$B/logs/ExperimentsR2_evaluation_20261026_retry01/U5_push_registered/results/`。
  - Kick：`.../U6_kick/results/`。
- **±4/6/8 档结果目录：** `$B/logs/V3_20261102/cloud_backup_20261104/raw/v3_extended_eval_20261104_ops/results/<ID>_seed72X_Full_pm{4,6,8}_attempt01/`。

---

## 5. 一次控制步的调用链（Table Pulling 写全，其他任务只写不同处）

### 5.0 主循环

一个控制步 = 0.02 s = 4 个物理子步。

1. `BaseTask.step(actor_state)`（`src/holosoma/holosoma/envs/base_task/base_task.py:405`）
2. `_pre_physics_step`：`Plan5PushManager._pre_physics_step`（`envs/marl/plan5_push_manager.py:15`）接受形状 `[E,2,29]` 或 `[E,58]`，压平成 `[E,58]` 后调用 `action_manager.process_actions`。
3. `_physics_step`（`base_task.py:417`）：循环 `control_decimation=4` 次：
   - `action_manager.apply_actions()`，即 `DualJointPositionActionTerm.apply_actions`，在 `managers/action/terms/marl.py:134`。每个子步重新读取实时关节状态并计算 PD。
   - `simulator.simulate_at_each_physics_step()`。
4. `_post_physics_step`（`base_task.py:427`）依次执行：
   - 刷新张量、`episode_length_buf += 1`、命令步进；
   - `_check_termination`、`_compute_reward`；
   - 重置结束的 env；
   - 计算观测。

### 5.1 actor 输入 158 维

- **组成：** `actor_obs`（154）在前，`teammate_obs`（4）在后，按 `[actor_obs, teammate_obs]` 拼接。见 `agents/mappo/runner.py:70` 和 `agents/mappo/core4d_smalltable_runner.py:20-21`。
- **组内顺序：** `ObservationManager.compute_group` 按 term 名的**字母序**拼接（`managers/observation/manager.py:139-142`），不是配置里的书写顺序。
- **历史帧：** 各组 `history_length=1`，不带历史。
- **term 实现：** `managers/observation/terms/marl.py`；组配置在 `config_values/marl/g1/observation.py:12-46`，继承 `config_values/wbt/g1/observation.py:5-41` 和 `141-159`。

| 下标 | term（字母序） | 维 | 内容 | 坐标系 | 训练噪声（均匀，±） |
|---|---|---|---|---|---|
| 0–28 | `actions` → `paired_actions`（`:75`） | 29 | 上一步原始动作。`action_manager.action` 未经裁剪，reshape 成 `[E,2,29]` | — | 0 |
| 29–31 | `base_ang_vel` → `paired_base_ang_vel`（`:54`） | 3 | 根（pelvis）角速度 | 本机器人根坐标系（完整姿态逆旋转） | 0.2 |
| 32–60 | `dof_pos` → `paired_dof_pos`（`:65`） | 29 | 关节角减默认姿态 | 关节空间 | 0.01 |
| 61–89 | `dof_vel` → `paired_dof_vel`（`:70`） | 29 | 关节角速度 | 关节空间 | 0.5 |
| 90–147 | `motion_command` → `paired_motion_command`（`:24`） | 58 | 当前参考帧关节位置 29（**绝对角，不减默认**）+ 参考关节速度 29 | 关节空间 | 0 |
| 148–153 | `motion_ref_ori_b` → `paired_motion_ref_ori_b`（`:29`） | 6 | 参考 `torso_link` 姿态相对实际 `torso_link` 姿态，取旋转矩阵前两列 | 实际 torso 坐标系 | 0.05 |
| 154–155 | `relative_position_b` → `real_teammate_relative_position_b`（`:213`） | 2 | 队友根相对本机根的平面位置 | 本机 heading（只取 yaw）坐标系，clip 到 [−1,1] | 无（组内 `enable_noise=False`） |
| 156–157 | `relative_velocity_b` → `real_teammate_relative_velocity_b`（`:218`） | 2 | 队友根相对本机根的平面线速度 | 同上，clip 到 [−1,1] | 无 |

- **噪声开关：**
  - 训练时 actor 组 `enable_noise=True`。
  - 评测时换成 `g1_29dof_core4d_smalltable_evaluation_observation`，其中 actor 组 `enable_noise=False`（`config_values/marl/g1/core4d_smalltable_observation.py`）。
- **归一化：** actor 输入经 `actor_obs_normalizer`（EmpiricalNormalization）。
  - Plan5 的 normalizer 来自单人先验，并且冻结：`FrozenEmpiricalNormalization`，在 `agents/mappo/initialization.py`。
  - CORE4D 的 normalizer 是新建的。
- **actor 不含物体状态：** 不含物体位姿，不含接触力。
- **critic 输入：** 527 维 = 2 × 228（每机器人）+ 共享 58 + 12 + 1。
  - 每机器人 228 = actions 29 + base_ang_vel 3 + base_lin_vel 3 + 14 个 body 的姿态 84 + 14 个 body 的位置 42 + dof_pos 29 + dof_vel 29 + ref_ori 6 + ref_pos 3。
  - 共享部分：`shared_motion_command` 取 `command[:,0]`（58）、`shared_object_tracking`（12）、`shared_phase`（1）。
  - 配置在 `config_values/marl/g1/observation.py:48-90`，实现在 `managers/observation/terms/marl.py:80-185`。
  - 每个 term 内两台机器人的排列顺序没有逐项核对。

### 5.2 动作：29 维 → 关节目标 → 力矩

实现：`DualJointPositionActionTerm`（`managers/action/terms/marl.py`），配置 `config_values/marl/g1/action.py`，其中 `scale=1.0`、`clip=None`。

1. **`process_actions`**（`:104`）：
   - `clip_actions=True`，`action_clip_value=100.0`：把动作裁剪到 ±100。
   - 动作延迟没有启用。
2. **`_compute_torques`**（`:148-192`），控制类型 `P`：
   - `q_target = a * action_scales + default_dof_pos`
   - `τ = Kp·(q_target − q) − Kd·q̇`
   - `clip_torques=True`：裁剪到 ±`torque_limits`。
   - `torque_limits` 取自 `dof_effort_limit_list`（`simulator/isaacsim/isaacsim.py:761-782`）。
3. **每个子步读实时状态：** `get_agent_dof_control_state()`（`simulator/isaacsim/dual_robot_isaacsim.py:128`）。
4. **施加力矩：** `apply_agent_torques()`（`:146`），对两台 articulation 调 `set_joint_effort_target`。
   - Isaac Lab 侧用 `IdealPDActuatorCfg(stiffness=0, damping=0)`（`isaacsim.py:303-315`），也就是说 PD 完全在 Python 里算，Isaac 只执行力矩。
   - `effort_limit` 和 `velocity_limit` 也在 actuator 中设置。
5. **动作缩放：** `action_scales_by_effort_limit_over_p_gain=True`，`action_scale=0.25`，所以 `scale_i = 0.25 × effort_i / Kp_i`。
   - 配置：`config_values/robot.py:1094-1105`（`g1_29dof_w_object`）。
   - 计算：`marl.py:278-284`。
6. **Kp/Kd 匹配规则：** `_configure_pd_gains`（`marl.py:263-276`）按关节名子串匹配 `control.stiffness/damping`（`config_values/robot.py:486-527`）。
7. **默认姿态：** `init_state.default_joint_angles`（`config_values/robot.py:346-376`）。
8. **其他物理参数：** armature 见 `dof_armature_list`；关节摩擦为 0。

### 5.3 关节顺序

- 动作向量顺序 = `robot_config.dof_names`（`config_values/robot.py:24-54`）。
- Isaac 侧用 `find_joints(..., preserve_order=True)`（`isaacsim.py:701`、`dual_robot_isaacsim.py:56-62`），所以顺序不变。
- 下表的增益和限位按代码配置计算。

| # | 关节 | Kp | Kd | 力矩限制 (N·m) | 动作缩放 (rad/单位动作) | 默认角 (rad) | 位置限位 (rad) |
|---|---|---|---|---|---|---|---|
| 0 | `left_hip_pitch_joint` | 40.179 | 2.558 | 88 | 0.5475 | -0.312 | -2.531 / +2.880 |
| 1 | `left_hip_roll_joint` | 99.098 | 6.309 | 139 | 0.3507 | +0.000 | -0.524 / +2.967 |
| 2 | `left_hip_yaw_joint` | 40.179 | 2.558 | 88 | 0.5475 | +0.000 | -2.758 / +2.758 |
| 3 | `left_knee_joint` | 99.098 | 6.309 | 139 | 0.3507 | +0.669 | -0.087 / +2.880 |
| 4 | `left_ankle_pitch_joint` | 28.501 | 1.814 | 50 | 0.4386 | -0.363 | -0.873 / +0.524 |
| 5 | `left_ankle_roll_joint` | 28.501 | 1.814 | 50 | 0.4386 | +0.000 | -0.262 / +0.262 |
| 6 | `right_hip_pitch_joint` | 40.179 | 2.558 | 88 | 0.5475 | -0.312 | -2.531 / +2.880 |
| 7 | `right_hip_roll_joint` | 99.098 | 6.309 | 139 | 0.3507 | +0.000 | -2.967 / +0.524 |
| 8 | `right_hip_yaw_joint` | 40.179 | 2.558 | 88 | 0.5475 | +0.000 | -2.758 / +2.758 |
| 9 | `right_knee_joint` | 99.098 | 6.309 | 139 | 0.3507 | +0.669 | -0.087 / +2.880 |
| 10 | `right_ankle_pitch_joint` | 28.501 | 1.814 | 50 | 0.4386 | -0.363 | -0.873 / +0.524 |
| 11 | `right_ankle_roll_joint` | 28.501 | 1.814 | 50 | 0.4386 | +0.000 | -0.262 / +0.262 |
| 12 | `waist_yaw_joint` | 40.179 | 2.558 | 88 | 0.5475 | +0.000 | -2.618 / +2.618 |
| 13 | `waist_roll_joint` | 28.501 | 1.814 | 50 | 0.4386 | +0.000 | -0.520 / +0.520 |
| 14 | `waist_pitch_joint` | 28.501 | 1.814 | 50 | 0.4386 | +0.000 | -0.520 / +0.520 |
| 15 | `left_shoulder_pitch_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.200 | -3.089 / +2.670 |
| 16 | `left_shoulder_roll_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.200 | -1.588 / +2.252 |
| 17 | `left_shoulder_yaw_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.000 | -2.618 / +2.618 |
| 18 | `left_elbow_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.600 | -1.047 / +2.094 |
| 19 | `left_wrist_roll_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.000 | -1.972 / +1.972 |
| 20 | `left_wrist_pitch_joint` | 16.778 | 1.068 | 5 | 0.0745 | +0.000 | -1.614 / +1.614 |
| 21 | `left_wrist_yaw_joint` | 16.778 | 1.068 | 5 | 0.0745 | +0.000 | -1.614 / +1.614 |
| 22 | `right_shoulder_pitch_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.200 | -3.089 / +2.670 |
| 23 | `right_shoulder_roll_joint` | 14.251 | 0.907 | 25 | 0.4386 | -0.200 | -2.252 / +1.588 |
| 24 | `right_shoulder_yaw_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.000 | -2.618 / +2.618 |
| 25 | `right_elbow_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.600 | -1.047 / +2.094 |
| 26 | `right_wrist_roll_joint` | 14.251 | 0.907 | 25 | 0.4386 | +0.000 | -1.972 / +1.972 |
| 27 | `right_wrist_pitch_joint` | 16.778 | 1.068 | 5 | 0.0745 | +0.000 | -1.614 / +1.614 |
| 28 | `right_wrist_yaw_joint` | 16.778 | 1.068 | 5 | 0.0745 | +0.000 | -1.614 / +1.614 |

- `soft_dof_pos_limit=0.95`，用于观测和限位计算。奖励项 `limits_dof_pos` 另用自己的参数 0.9。
- 本分支的六个任务都共用这一套动作设置，没有发现某个任务单独覆盖。

### 5.4 奖励、终止、重置、初始状态扰动

**奖励**
- 配置：`config_values/marl/g1/reward.py:8-73`（`g1_29dof_plan5_push_reward`，Push/Pull/Kick 共用）。
- 实现：`managers/reward/terms/marl.py`。
- `RewardManager.compute(dt)`（`managers/reward/manager.py:126`）对每项乘权重再乘 dt。
- 身体类各项先按每台机器人算，再在 agent 维取均值（`.mean(dim=1)`），得到每个 env 一个 team reward。

| 项 | 权重 | 参数 |
|---|---|---|
| `motion_global_ref_position_error_exp` | 0.5 | σ=0.3（torso_link 全局位置） |
| `motion_global_ref_orientation_error_exp` | 0.5 | σ=0.4 |
| `motion_relative_body_position_error_exp` | 1.0 | σ=0.3（14 个跟踪 body） |
| `motion_relative_body_orientation_error_exp` | 1.0 | σ=0.4 |
| `motion_global_body_lin_vel` | 1.0 | σ=1.0 |
| `motion_global_body_ang_vel` | 1.0 | σ=3.14 |
| `action_rate_l2`（函数 `penalty_action_rate`） | −0.1 | — |
| `limits_dof_pos` | −10.0 | soft_dof_pos_limit=0.9 |
| `undesired_contacts`（`UndesiredContacts`） | −0.1 | 阈值 1 N。脚底接触点、腕 yaw link、rubber hand link、踝 roll link 以外的 body 计入 |
| `object_global_ref_position_error_exp` | 1.0 | σ=0.3 |
| `object_global_ref_orientation_error_exp` | 1.0 | σ=0.4 |

- 14 个跟踪 body 是 `config_values/marl/g1/command.py:9-24` 里的 `body_names_to_track`。参考 body 是 `torso_link`。

**终止**（`config_values/marl/g1/termination.py`）
- `timeout`：`is_timeout=True`。
  - Push/Pull 的 `max_episode_length_s=10.0`，即 500 步（`experiment.py:48`）。
  - Kick 是 298/50 s（`kick_experiment.py:19-21`）。
- `joint_bad_tracking`：`JointBadTrackingZOnly`（`managers/termination/terms/marl.py:15-109`）。实际判据（`:109`）是以下三条取 OR：
  - 任一机器人 torso 投影重力 z 分量与参考相差 > 0.8；
  - 物体位置误差 > 0.25 m；
  - 物体姿态误差 > 0.8 rad。
- torso 高度误差、低高度、14 个 body 中 4 个的 z 误差只记诊断，不终止。
- 名字叫 `joint_...`，但判据和关节角误差无关。

**重置**（`PairedA1MotionCommand.reset`，`managers/command/terms/marl.py:80-99`）
- 训练时：从 `[0, N−1)` 均匀随机抽一帧作为起点（RSI）。评测时（`is_evaluating`）从第 0 帧开始。
- 抽到起点后，把参考的根状态、关节状态、物体状态直接写入仿真（`_write_reference_state`，`:107-129`）。
- 构造函数要求初始姿态噪声 `noise_to_initial_pose.overall_noise_scale == 0`（`:36-37`），所以训练时没有姿态扰动。
- 训练开始时 `episode_length_buf` 也随机化（`scripts/train_plan5_push.py:339-343`，`init_at_random_ep_len`）。
- **Plan5 的参考循环：** `step()` 里一旦 `time_steps >= N`，就对这些 env 调 `reset()`（`:101-105`）。
  - 训练时这会把机器人和物体传送到一个新的随机帧，而 env 的回合并不结束（timeout 是 500 步，参考只有 316 步）。
  - 评测在第 N 步停止，不会走到这一步。

**初始状态扰动**
- 只在评测时施加，见 `evaluate_paper_supplement.py:134-142`：
  1. `reset_envs_idx`，回到参考第 0 帧；
  2. 取 nominal 根状态，用 `yaw_transform` 对每台机器人加世界 XY 偏移和绕自身根的 yaw；
  3. `set_agent_root_states`；
  4. 校验实际状态与请求一致（atol 1e-5）。
- 关节、根高度、物体都不动。

### 5.5 MAPPO 超参数实际取值与论文 Table IV

- **本地没有论文源文件。** 只有 `docs/experiments/ubuntu_push_placement_more_seeds_20261107_zh.md:38` 转述了 Table IV 的部分取值：2048 env、24 步/迭代、actor 初始学习率 1e-5（adaptive）、critic 学习率 1e-3、clip 0.2、γ 0.99、λ 0.95。其余 Table IV 项：未确认。
- **基础配置：** `config_values/algo.py:12-56`（`ppo`）→ `config_values/wbt/g1/experiment.py:25-41`（`g1_29dof_wbt`）。Plan5 在 `scripts/train_plan5_push.py:345-359` 上覆盖；CORE4D 用 `_core4d_smalltable_algo`（`config_values/marl/g1/core4d_smalltable_experiment.py:36-55`）。

| 参数 | Push | Pull | Kick | Chair/Bucket/Table | Table IV（转述） |
|---|---|---|---|---|---|
| 并行 env | 2048 | 2048 | 2048 | 2048 | 2048 |
| 每迭代步数 | 24 | 24 | 24 | 24 | 24 |
| actor 学习率（run_config） | 1e-5 | 1e-5 | **1e-3** | **1e-3**。Table 722/723 记录为恢复时的当前值 1.7e-4 / 1.1e-4 | 1e-5 |
| 学习率调度 | adaptive，desired_kl=0.01 | 同 | 同 | 继承 `ppo`：adaptive，0.01。CORE4D run_config 没有记录该字段 | adaptive |
| critic 学习率 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| clip | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| γ / λ | 0.99 / 0.95 | 同 | 同 | 同 | 0.99 / 0.95 |
| 学习 epoch / minibatch | 5 / 4 | 5 / 4 | 5 / 4 | 继承 5 / 4，run_config 未记录 | 未确认 |
| entropy_coef | 继承 0.005，run_config 未记录 | 同 | 同 | 0.005，run_config 有记录 | 未确认 |
| init_noise_std | 继承 1.0。Plan5 的 actor 来自先验，噪声 std 取先验里的值，run_config 未记录 | 同 | 同 | 1.0 | 未确认 |
| max_grad_norm / value_loss_coef | 1.0 / 1.0 | 同 | 同 | 同 | 未确认 |
| 网络 | MLP [512,256,128] ELU，actor 和 critic 相同 | 同 | 同 | 同 | 未确认 |
| 迭代预算 | 50 critic-only + 13000 联合，共 13050 | 50 + 8000，共 8050 | 8000 | 12000 | 未确认 |
| actor 初始化 | 单人先验 | 单人先验 | 单人先验 | fresh | — |

- PPO 主干：`Plan5PPO.collect_rollout`（`agents/mappo/ppo.py:107`）→ `compute_team_returns_and_advantages`（`:180`）→ `update` / `_update_minibatch`（`:207/242`）。
- 每个 env 只有一条 team reward 和一条 GAE，算出的 team advantage 分给两台机器人各自的 actor 样本。

### 5.6 其他任务与 Pull 的不同

- **Push：**
  - 不用 paired 文件。由 `PairedA1Reference.from_motion_loader` 生成双人参考（`envs/marl/paired_a1_reference.py:49-79`）：
    - 两台机器人的关节轨迹和姿态都等于单人 A1 参考；
    - 根和 body 位置分别沿桌子局部 x 轴平移 ∓spacing/2，spacing=0.8780472 m，由 `plan5_registered_placement.configure_push_placement` 注入，默认值是 0.8。
  - 物体 URDF 是 `objects_widetable_plan5_training.urdf`：长轴沿局部 X，惯量的 Ixx 和 Izz 与 Pull 版对调。
- **Kick：**
  - 参考是 mirrored paired runtime。
  - 回合长度 298 步，与参考帧数相同。
  - actor 学习率 1e-3，没有 critic-only 段。
  - 物体 URDF 与 Pull 相同。
- **CORE4D（Chair/Bucket/Table）：**
  - 命令换成 `Core4DSmallTableMotionCommand`（`managers/command/terms/core4d_smalltable.py:47-140`）：
    - 参考**不循环**，到末帧后保持（`step()` 里 clamp）；
    - 重置时把物体原点速度转成 COM 速度后再写入。
  - 终止换成 `reference_horizon`（到末帧，`is_timeout=True`）+ 同一个 `joint_bad_tracking`（`config_values/marl/g1/core4d_smalltable_termination.py`）。
  - 回合长度 = 参考帧数 / 50 s。
  - 奖励：
    - Chair 用 baseline 11 项，同上表（`core4d_smalltable_reward.py`）。
    - Bucket 和 Table 用 `bucket_A_no_rel_no_height`：去掉两项物体奖励，换成一项 `bucket_interaction`，权重 1.0（`config_values/marl/g1/core4d_bucket_reward.py:38-40`）。
    - 这一项内部的块权重：位置/旋转/高度/关系 = 1/1/0/0；位置误差的 xyz 权重是 1/1/2；σ_pos=0.3，σ_rot=0.4；不启用接触门控（`core4d_bucket_contract.py:22-57`）。
  - 物理资产和契约由 `get_core4d_pair_experiment`（`config_values/marl/g1/core4d_pair_experiments.py`）按 sha256 锁定，包括参考、URDF、promotion manifest 和质量。
  - 评测时 Bucket/Table 打开 `enable_object_hand_contact`，记录手-物法向力 `hand_object_normal_force_w`。

---

## 6. 机器人模型

- **URDF：** `src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf`，sha256 `7ed217f2…a25`。
  - 配置：`config_values/robot.py:1094-1100`，`asset_root="@holosoma/data/robots"`。
- **USD：** 没有预制 USD（`usd_file=None`）。
  - Isaac Lab 运行时用 `UrdfFileCfg(force_usd_conversion=True, merge_fixed_joints=True)` 转换（`simulator/isaacsim/isaacsim.py:236-260`）。
  - 转换结果写在 `<asset_root>/converted_rank{LOCAL_RANK}`。
  - R2 记录过两个进程都用 `LOCAL_RANK=0` 导致转换缓存冲突的情况。
- **手 link：**
  - `left_rubber_hand_link` 和 `right_rubber_hand_link`，每只 0.170 kg。
  - 通过固定关节 `left_hand_palm_joint` / `right_hand_palm_joint` 挂在 `*_wrist_yaw_link` 上，偏移 xyz 分别是 (0.0415, ±0.003, 0)。
  - 因为 `collapse_fixed_joints=True`，仿真里 rubber hand 被并进 `*_wrist_yaw_link`，没有单独的刚体。R2 README 记录，实际选中的 32 个刚体里没有 rubber hand。
- **脚 link：**
  - `left_foot_contact_point` / `right_foot_contact_point`（0.001 kg），固定在 `*_ankle_roll_link` 下 z=−0.037 处。
  - 配置里 `foot_body_name="ankle_roll_link"`，`key_bodies` 是两个 foot_contact_point。
  - URDF 里另有 `LL_FOOT`、`LR_FOOT`，质量为 0。
- **论文手误差用的 link：** `left_rubber_hand_link` 和 `right_rubber_hand_link` 的 link origin。
  - 计算方法：对保存的根位姿和 29 个关节角做离线 URDF FK，`BatchFK` 基于 yourdfpy，见 `$B/scripts/prepare_experiments_r2_cpu.py:39-40,93`（`HANDS = ['left_rubber_hand_link','right_rubber_hand_link']`）。
  - 汇总：`$V3/V3_GPU_CLOSEOUT_20261104/sources/finalize_v3_results_20261104.py:84-120`。
  - 口径（`$V3/V3_GPU_CLOSEOUT_20261104/DATA_DICTIONARY.md`）：2 台机器人 × 2 只手 × 全部保存帧，世界坐标欧氏距离取均值，单位 cm。参考帧按 `reference_frame` 对齐，并加上物体参考原点的偏移；不做平面重对齐。只统计完成的扰动回合。
- **各 link 质量**（URDF `inertial/mass`，总和 33.343 kg）：
  - pelvis 3.813；torso_link 6.780；head_link 1.036
  - hip_pitch 1.350、hip_roll 1.520、hip_yaw 1.702、knee 1.932、ankle_pitch 0.074、ankle_roll 0.608（左右相同）
  - waist_yaw 0.214、waist_roll 0.086
  - shoulder_pitch 0.718、shoulder_roll 0.643、shoulder_yaw 0.734、elbow 0.600、wrist_roll 0.0854、wrist_pitch 0.484、wrist_yaw 0.0846（左右相同）
  - rubber_hand 0.170 × 2
  - 其余 contact/logo/imu/mid360 link 为 0 或 0.001。

---

## 7. 参考文件格式

### 7.1 paired runtime npz

用于 Pull、Kick、Chair、Bucket、Table。加载器是 `envs/marl/paired_motion_reference.py:15-140`（`PairedMotionReference`）；CORE4D 用子类 `Core4DSmallTableReference`。

| 键 | 形状（Pull） | 说明 |
|---|---|---|
| `fps` | () int64 = 50 | 帧率 |
| `joint_names` | (29,) | 与动作顺序相同；加载时按名字重排 |
| `body_names` | (51,) | 首项 `world`，然后 `pelvis`、`pelvis_contour_link`… 加载时按仿真 body 名重排 |
| `agent_joint_pos` / `agent_joint_vel` | (T,2,29) float64 | 两台机器人的关节角/速度，绝对角 |
| `agent_body_pos_w` | (T,2,51,3) | body 世界位置 |
| `agent_body_quat_w` | (T,2,51,4) | **文件里是 WXYZ**，加载时转成 XYZW（`_wxyz_to_xyzw`） |
| `agent_body_lin_vel_w` / `agent_body_ang_vel_w` | (T,2,51,3) | 世界系 |
| `object_pos_w` / `object_lin_vel_w` / `object_ang_vel_w` | (T,3) | **物体轨迹在这里**；位置是物体原点 |
| `object_quat_w` | (T,4) | WXYZ，加载时转成 XYZW |
| `provenance` | () str（JSON） | 生成器与参数，例如 `holosoma_retargeting.dual_pull_runtime_reference` |

- **根位姿约定：**
  - 根 = body 0（pelvis）：重置时用 `agent_body_pos_w[:, :, 0]` 和 `agent_body_quat_w[:, :, 0]`。
  - 仿真内部四元数是 XYZW。
  - 仿真的根状态张量是 `[pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]`，共 13 维。
- **各任务帧数 T（50 Hz）：** Pull 316，Kick 298，Chair 391，Bucket 497，Table 687。
- **没有接触标志：** 六个文件里都没有接触相关的键。

### 7.2 单人 A1 npz

用于 Push，由 `MotionLoader` 加载。键：`fps`(1,)、`joint_pos`(T,36)、`joint_vel`(T,35)、`body_pos_w`/`body_quat_w`/`body_lin_vel_w`/`body_ang_vel_w`(T,52,·)、`object_pos_w`/`object_quat_w`/`object_lin_vel_w`/`object_ang_vel_w`、`joint_names`(29)、`body_names`(52)。
- 36 = 浮动基座 7 + 关节 29；35 = 浮动基座 6 + 关节 29。
- 帧数：Push 309。Pull 的单人源文件是 317 帧，比 paired 文件多 1 帧。

### 7.3 采样

- `reference.sample(time_steps)` 返回当前帧的所有通道，形状 `[E,2,...]`。
- 命令对象每个控制步 `time_steps += 1`，也就是每 0.02 s 前进一帧参考，不插值。
- 评测开始时检查 `env.dt == 1/fps`（`evaluate_paper_supplement.py:97`）。

---

## 8. 物体

| 任务 | URDF（`src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/`） | 质量 (kg) | 惯量 | 碰撞体 |
|---|---|---|---|---|
| Push | `objects_widetable_plan5_training.urdf` | 20.0 | URDF 显式给出：Ixx 0.6277，Iyy 4.3604，Izz 3.9505；COM (0, 0.01511, 0) | box 组合（桌面 1.4×0.047×0.522 + 4 条腿） |
| Pull / Kick | `objects_widetable_plan5_pull_training.urdf` | 20.0 | Ixx 3.9505，Iyy 4.3604，Izz 0.6277（与 Push 对调 Ixx/Izz） | box 组合（长轴沿局部 Z） |
| Chair | `core4d_chair021_20231020_074_a1/chair021_training.urdf` | 5 | URDF 显式给出，含非对角项 | mesh `chair021_m.obj`，缩放 0.7442；契约 collider 为 convex_hull |
| Bucket | `core4d_bucket003_20231020_071_a1/bucket003_training.urdf` | 1 | URDF 显式给出 | mesh `bucket003_m.obj` |
| Table | `core4d_smalltable5kg_A/desk001_5kg_training.urdf` | 5.0 | URDF 显式给出：Ixx 0.3771，Iyy 0.2411，Izz 0.2676 | box 组合；collider convex_hull |

- **质量和惯量：** 都写在物体 URDF 的 `<inertial>` 里，没有找到代码里覆盖质量的地方。
- **摩擦和恢复系数：**
  - 由随机化项 `set_object_rigid_body_material_startup` 在启动时设置：静摩擦 0.5，动摩擦 0.5，restitution 0（`config_values/marl/g1/randomization.py`，六任务共用）。
  - CORE4D 的契约里也写了 `(0.5, 0.5, 0.0)`。
- **训练时的物理读回和校验：**
  - Plan5：`scripts/train_plan5_push.py:169-300`。
    - 硬编码 `EXPECTED_OBJECT_MASS_KG = 20.0`、`EXPECTED_OBJECT_COM_M`，以及按 skill 分的 `EXPECTED_OBJECT_INERTIA_KG_M2` 和材料；
    - 从 PhysX 读回 mass、inertia、com、material，与期望值不符就报错；
    - 读回的数组写进 run_config 的 `runtime_object_physics`（`:338,457`）。
  - CORE4D：`scripts/train_core4d_smalltable.py:183,242-262`。
    - 期望质量来自 `EXPERIMENT.object_mass_kg`；
    - 读回值写进 run_config 的 `object_physics`（`:472`）。
    - 契约（`core4d_pair_experiments.py`）同时锁定 URDF 的 sha256 和 promotion manifest。
- **评测：** manifest 的 `physics` 记录从 PhysX 读回的质量和材料（`evaluate_paper_supplement.py:121-122`）。

**T04 改质量应改哪里（按现有代码路径推断，供任务书参考）**

- 质量和惯量都只在 URDF 里，所以要让惯量和质量同步，需要按质量比例**新建 URDF**：
  - `<mass>` 和 `<inertia>` 的六个分量同乘 m/m0；
  - 几何、COM、碰撞体不变。
  - 已有先例：`core4d_smalltable5kg_A` 就是对 20 kg 小桌这样做的，惯量 ×0.25，promotion manifest 里写了 `"inertia_scale_from_20kg": .25`（`core4d_pair_experiments.py:89`）。
- **Table Pulling：**
  - 新建 `objects_widetable_plan5_pull_training_<m>kg.urdf`；
  - 配置里的 `object_urdf_path` 加开关切换；
  - `train_plan5_push.py` 的 `EXPECTED_OBJECT_MASS_KG` 和 `EXPECTED_OBJECT_INERTIA_KG_M2['pull']` 要跟着开关变，否则读回校验会报错；
  - `runtime_object_physics` 会自动把实际读回的质量和惯量记进 run_config。
- **Table Transport：**
  - 需要在 `get_core4d_pair_experiment` 里新增 experiment，例如 `smalltable{m}kg_A`；
  - 带新 URDF、它的 sha256、`object_mass_kg`，以及 promotion manifest；
  - `object_physics` 会记录读回值。
- **评测侧：** 评测脚本按 case 取配置，会校验 CORE4D 资产的 hash（`evaluate_paper_supplement.py:77-80`），所以也要能选到同一个新资产。

---

## 9. 仿真时序

| 项 | 值 | 位置 |
|---|---|---|
| 物理频率 | 200 Hz（`sim.fps=200`），dt = 0.005 s | `config_values/simulator.py:33-55`（`isaacsim`），`isaacsim_dual_robot` 继承 |
| decimation | 4 | 同上 |
| 策略频率 | 50 Hz，env.dt = 0.02 s。CORE4D run_config 记录 `control_hz=50.0`、`physics_hz=200.0`；评测 manifest 记录 `dt=0.02` | — |
| PhysX | solver_type 1，position iterations 8，velocity iterations 4，bounce 0.5，`contact_sensor_history_length=3` | 同上 |
| render | `render_interval=4`，训练和评测都 `headless=True` | — |
| 参考读取 | 50 fps，每个控制步前进 1 帧，不插值；参考文件本身已经是 50 Hz（30 Hz 源数据在生成 runtime 时已重采样） | `managers/command/terms/marl.py:101-102`；CORE4D `core4d_smalltable.py:129-132` |
| PD 更新 | 每个物理子步（200 Hz）用实时关节状态重新计算 | `managers/action/terms/marl.py:134-146` |

---

## 10. SONIC 接入点

### 10.1 动作 → 关节目标（A/B/C 要换成 SONIC 输出的地方）

- **唯一的转换点：** `DualJointPositionActionTerm._compute_torques`（`src/holosoma/holosoma/managers/action/terms/marl.py:148-192`）：
  - `q_target = a·action_scales + default_dof_pos`；
  - 之后在每个物理子步用 Kp/Kd 算力矩。
- **调用时序：**
  - `process_actions`（`:104`）每个控制步调用一次，接收 `[E,58]`；
  - `apply_actions`（`:134`）每个物理子步调用一次。
- **配置入口：** `config_values/marl/g1/action.py` 里的 `ActionTermCfg(func=...DualJointPositionActionTerm)`。
  - 可以新写一个 action term（放在 `system1/` 下），换掉 `func`，作为开关。
  - Kp/Kd、动作缩放、默认姿态目前取自 `robot_config.control` 和 `init_state`。改成 SONIC 自己的 PD 和缩放时，需要在新 term 里单独设置。
- **力矩输出：** `simulator.apply_agent_torques(torques[E,2,29])`（`simulator/isaacsim/dual_robot_isaacsim.py:146`）。Isaac 侧 actuator 的 stiffness 和 damping 都是 0。

### 10.2 参考帧被读取的位置（喂给 SONIC）

- **命令对象：** `env.command_manager.get_state("paired_motion_command")`，类型是 `PairedA1MotionCommand`，CORE4D 下是子类。
  - 当前帧：`command._sample()` → `self.reference.sample(self.time_steps)`（`managers/command/terms/marl.py:136-137`）。
  - 现成的属性（`:140-262`）：
    - `agent_joint_pos/vel [E,2,29]`；
    - `agent_body_pos_w/quat_w [E,2,14,·]`（跟踪 body）；
    - `agent_ref_pos_w/quat_w [E,2,·]`（torso）；
    - `agent_root_pos_w/quat_w`；
    - `object_pos_w/quat_w/lin_vel_w`。
- **完整轨迹：** `command.reference.agent_joint_pos` 等张量，形状 `[T,2,...]`，已转成 XYZW、按仿真顺序排好。需要未来帧窗口时，可以用 `time_steps + k` 直接索引。
- **当前时间步：** `command.time_steps [E]`（long）。
- **actor 读参考的位置：** `_paired_motion_command(env)`（`managers/observation/terms/marl.py:17`）。

### 10.3 两台机器人在 env 里的组织

- **仿真张量**（`dual_robot_isaacsim.py:69-126`，`refresh_sim_tensors`）：
  - `agent_root_states [E,2,13]`：世界系，`pos, quat_xyzw, lin_vel, ang_vel`；
  - `agent_dof_pos/vel [E,2,29]`；
  - `agent_rigid_body_pos/rot/vel/ang_vel [E,2,B,·]`；
  - `agent_contact_forces [E,2,B,3]`。
  - agent 0 是主 articulation（prim `/World/envs/env_.*/Robot`），agent 1 是 `_robot_1`（`/World/envs/env_.*/Robot_1`，`:35`）。
  - 物体 actor 名是 `"object"`。
- **写状态：** `set_agent_root_states(env_ids, [n,2,13])`、`set_agent_dof_states(env_ids, pos[n,2,29], vel[n,2,29])`（`:156-196`）。
- **动作：** 对外是 `[E,58]`，agent-major：前 29 维给 agent 0；内部用 `[E,2,29]`（`marl.py:15-21`）。
- **actor 批：** `HomogeneousAgentBatchLayout`（`agents/mappo/batch_layout.py:28-47`）把 `[E,2,D]` reshape 成 `[E·2, D]`，第 `e·2+a` 行对应 env e 的 agent a。actor 参数两台机器人共享。
- **评测：** `num_envs=1`，`seed=241003`（`evaluate_paper_supplement.py:47-48`）。

---

## 11. 日志与曲线

- **训练标量：** 每个运行目录下的 `metrics.jsonl`，每次迭代一行 JSON。由 `train_plan5_push.py` 和 `train_core4d_smalltable.py` 写入。字段包括：
  - `iteration`、`reward_mean`、`reward_min`、`reward_max`；
  - `reward_terms_raw_mean{各奖励项}`；
  - `value_loss`、`surrogate_loss`、`entropy`、`kl`；
  - `actor_learning_rate`、`critic_learning_rate`、`actor_grad_norm`、`critic_grad_norm`；
  - `action_noise_std_mean`、`reset_count`、`timeout_count`、`tracking_failure_count`；
  - `elapsed_seconds`、`policy_drift_*` 等。
  - 同目录还有 `run_config.json`、`status.json`、`model_XXXXX.pt`。
- **回合长度和回报：**
  - 只有加了 `--episode-telemetry` 的 V3 运行才有 `Episode/mean_length_last100_steps`、`Episode/mean_return_raw_last100`、`Episode/completed_window_count`。实现在 `$B/src/holosoma/holosoma/utils/v3_episode_telemetry.py`（未跟踪），另写 `episode_telemetry.jsonl.gz`。
  - 18 个历史主结果运行都没有这两个字段。
- **V3 `curves/*.csv` 的导出：** `$V3/V3_GPU_CLOSEOUT_20261104/sources/export_v3_training_curves_20261104.py`。
  - 输入：`cloud_backup_*/logs/**/metrics.jsonl`，加上每个历史 Full 模型最终 checkpoint 同目录下的 `metrics.jsonl`。
  - 每行映射：`mean_episode_length_steps←Episode/mean_length_last100_steps`，`mean_episode_return_raw←Episode/mean_return_raw_last100`，`mean_step_reward←reward_mean`。缺失就留空，标记 `metrics_status=missing_episode_length_not_substituted`。
  - 输出 `curves/training_scalars.csv`（372,456 行），清单在 `curves/training_curves_inventory.json`。
  - **历史主结果只导出了最终段：**
    - Push：8051–13050；
    - Pull：51–8050；
    - Table 722/723：10001–12000；
    - 其余任务从第 1 次迭代起完整。
    - 早期段的 metrics 在各自 stage 目录里，例如 `$B/logs/ExperimentsR2_retry1/push_seed72X_stage{0,1}_*`、`$B/logs/PaperSupplement20260929/autodl_received/runs/*_stage0_*` 和 `T07_*_fresh12000_cache_isolated02`。
  - Pull 三组 V3 曲线另见 `$V3/V3_CLOSEOUT_SUPPLEMENT_20261105/curves/pull_training_scalars.csv`。

---

## 12. 未确认事项与矛盾

1. **论文 Table IV：** 本地没有论文源文件，第 5.5 节的对照只依据 Push 任务书的转述。转述里 actor 学习率是 1e-5；Kick 和 CORE4D 三任务的 run_config 记录是 1e-3。Table IV 是否分任务列出，未确认。
2. **云端 commit 不在本地：** `b6944cff`（R2）和 `7e28e9aa`（PaperSupplement）在本地 git 里都找不到对象，代码只能从源码快照核对（见第 4.3 节）。PaperSupplement 用的是 deployment_v1 还是 v2 的快照，未确认。
3. **721 种子的训练机器：** Pull/Kick/Chair/Bucket/Table 721 的 run_config 里 `output_dir` 是本机路径，但实际在哪台机器上训练、用什么 GPU，未确认。Bucket/Table 721 的目录名带 `_cloud_2026091x`。
4. **721 的原始训练命令：** Pull、Kick、Chair、Bucket、Table 的 721 种子，本次没有找到完整命令行，只有 run_config。R2 README 写明 Kick 721 的历史训练退出码未知。
5. **Bucket/Table 训练时的 env 类：** 评测用 `Core4DBucketManager` 并打开 `enable_object_hand_contact`。训练时是否同样，本次没有核对 `train_core4d_smalltable.py` 的分支，未确认。
6. **同一任务的种子跨代码版本：** Pull 721 用的是 `$M` 的 `65f7ba5e`，722/723 用的是 PaperSupplement 快照；Chair、Bucket、Table 同理。各版本之间的差异没有逐一 diff。
7. **本分支缺少评测依赖：** 已提交的 `deterministic_core4d_smalltable_actions` 没有 `zero_teammate_obs` 参数，而 `evaluate_paper_supplement.py` 调用了它；`plan5_registered_placement.py` 和 `paper_evaluation_utils.py` 也未提交。T01 第 0 步的快照必须包含这些文件和 core4d-base 的未提交改动，否则在本分支上跑不起评测。
8. **旧文档与实际代码不一致：** `docs/writing_handoff/CODE_MAP.md` 第 6 节按"58 命令 + 6 + 3 + 29 + 29 + 29"的顺序描述 actor 的 154 维。实际拼接按字母序（第 5.1 节）。维数一致，只是顺序不同。
9. **editable 安装指向别处：** `hssim` 的 holosoma 指向 `/home/kevin/holosoma`（`push-bigbox-experiment`）。不经训练/评测脚本、直接 `import holosoma` 会拿到不同的代码。T01 克隆环境后需要注意这一点。
10. **训练中的参考循环：** Plan5 在训练时参考播完会把状态写回随机帧，回合不终止（第 5.4 节）；评测和 CORE4D 都不循环。这是代码行为，记在这里供方案 C 的训练设计参考。
11. **曲线缺段：** 历史主结果的曲线只导出了最终段（第 11 节），而且没有回合长度。T04/T03 若要比较"达到同一奖励所需的迭代数"，需要从各 stage 目录重新拼接。
12. **SONIC 与机器人模型：** SONIC 所用的 G1 模型和 rubber-hand URDF 是否一致，本次没有核对（属于 T01 范围）。rubber-hand 版每只手多 0.17 kg，而且仿真里手被并入 wrist_yaw_link。
13. **本次只读操作的一处副作用：** 第一次在三个旧目录运行 `git status --porcelain` 时没有设 `GIT_OPTIONAL_LOCKS=0`，可能刷新了 `/home/kevin/holosoma/.git/worktrees/*/index` 的 stat 缓存。git 对象和工作区文件内容没有变化。此后所有 git 命令都改用 `GIT_OPTIONAL_LOCKS=0`。
