# Ubuntu 实验补充交接说明（2026-10-24，自包含版）

给 Ubuntu 端执行者：可直接从 GitHub 仓库 `Kaifeng-Jia/holosoma-rubber-hand-marl` 的 `core4d-base` 分支读取 `docs/experiments/ubuntu_handoff_experiments_r2_20261024_zh.md`，或使用作者带来的本文件；仅获取说明即可，不需切换、重置或覆盖 Ubuntu 工作树。你不需要、也无法读取 Mac 上的论文、brief、图表或任何 Mac 路径。本文件给出任务、公式、Ubuntu 资产定位方式、执行顺序和交付契约。论文暂定名 DuetMimic，计划 11 月上旬投稿；这里只补 U1–U8，不扩展实验范围。

## 执行顺序与时间

1. **先做 U5 的开训核验，再启动 Push 配准摆位的三个种子 721、722、723。** 必须先核对下文 0.878 m 的计算来源、配准导出状态和实际训练引用的资产。作者本轮已明确要求重训；旧几何报告中的“不自动训练”是当时的范围，不是本轮需要再次征求许可的条件。
2. **U5 之后再跑 U6，补 Kick 的 722、723。** 721 用既有模型，三种子统一评测。不因旧任务单写了“显存够可同卡并行”而改变此处的优先顺序。
3. **U1–U4 和 U8 只用 CPU，可与 U5/U6 并行。** U3/U4 对新策略的部分在新轨迹交付后追加；对旧轨迹的部分现在就做。U4 可选，不阻塞 U5/U6。U7 在对应参考/轨迹齐备后渲染，避免挤占训练资源。
4. **U5 和 U6 均以 2026-10-31 前完成训练、统一评测和交付为目标。** 每个种子启动后记录预计结束时间；一旦判断任何一项赶不上，提前告诉作者，写明已完成种子、剩余工作、原因和最早可交付时间，不等到截止日才报告。最迟在 10/29 做一次截止风险检查。作者决定是否延后纳入结果，不由执行者静默放弃。
5. 分批交付：先交开训核验与 U8 事实；再交已完成 CPU 指标；最后交 U5/U6 完整结果与 U7 图像。未完成数值用 `--`，README 标明对应 U 编号、原因和预计时间；不估数、不把失败或缺失写成 0。

## Ubuntu 工作目录与已知入口

优先使用原 Ubuntu 工作副本：

- 当前研究副本：`/home/kevin/holosoma-core4d-base`，分支 `core4d-base`。
- 旧 Push/Pull/Kick 训练副本：`/home/kevin/holosoma-rubber-hand-marl`。
- 旧单机器人数据/模型副本：`/home/kevin/holosoma-rubber-hand-largetable`。
- 远程代码库身份：`Kaifeng-Jia/holosoma-rubber-hand-marl`。本说明不要求切换、重置或覆盖已有分支。
- 已知 CPU 环境：`/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python`；已知仿真环境：`/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python`。先确认本机实际可用环境。

以下以 `ROOT` 表示你核实后的研究副本根目录；未写绝对路径的代码路径均相对于 `ROOT`。若路径迁移，先在 Ubuntu 本地定位同名文件，核对文件 hash 和运行记录，再在交付 README 写出新的绝对路径。不要要求作者同步整份 Mac 工程。

已知 2026-10-17 几何执行时 `HEAD=2b6668b1205ee5c5b81ae9251744b16c799c7173`，指引提交为 `d4e934de3445e6cc17ad762c1c87cb3fcef16f1b`。当时部分脚本尚未跟踪，因此 **commit 不能单独证明实际执行代码**：每次交付同时保存脚本快照、配置及其 sha256、工作区差异。

| 内容 | Ubuntu 入口或资产 |
|---|---|
| 第一批源几何、接近窗口、原参考 | `logs/ReferenceGeometry/reference_geometry_pilot_20261017/`，每任务含 `input_manifest.json`、`preprocessed_source.npz`、`source_interactions.json`、`source_distances.npz`、`legacy_pair_reference.npz` |
| 第二轮配准 | `scripts/run_reference_geometry_round2.py`、`scripts/reference_geometry_round2_correspondence.py`、`scripts/reference_geometry_math.py`；第二轮输出在 `logs/ReferenceGeometry/` 下定位，按 `push/selection.json`、`push/new_runtime.npz` 和 `execution_identity.json` 校验，不只按文件夹名字猜 |
| 现有统一评测 | `logs/PaperSupplement20260929/unified_evaluation_20261003/`，含 `models.json`、`*_states.json`、`results/`、`training_evidence/` |
| 评测与汇总脚本 | `scripts/evaluate_paper_supplement.py`、`scripts/run_paper_evaluation_batch.py`、`scripts/paper_evaluation_utils.py`、`scripts/summarize_paper_evaluation.py`、`scripts/organize_paper_evaluation.py` |
| 旧 Push 构造和参考采样 | `src/holosoma/holosoma/config_values/marl/g1/command.py`、`src/holosoma/holosoma/managers/command/terms/marl.py`、`src/holosoma/holosoma/envs/marl/paired_motion_reference.py` |
| 几何变换及真实 FK | `scripts/reference_geometry_math.py` 的 `procrustes` / `map_robot`；`scripts/reference_geometry_pilot.py` 的 `RobotFK` |
| CORE4D 实际资产 | `src/holosoma/holosoma/config_values/marl/g1/core4d_pair_experiments.py`；对应 motion 目录下 `source_manifest.json`、`training_asset_manifest.json`、runtime NPZ 与 URDF |
| 回放渲染 | `scripts/export_recorded_demo_media.py`；已有 `paper_supplement_evaluation/media/<model_id>/manifest.json` 可在 `paper_handoff_exports/` 下定位，其 `rollout_source` 指向上述 Ubuntu 评测目录 |
| 腕部 | `scripts/render_wrist_reference_pairs.py`、`scripts/collect_wrist_bucket_materials.py`；`paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005/` 中的来源与腕部指标 |

`data/evaluation_20261005` 是 Mac 论文工程整理后的目录名，**不是要求 Ubuntu 新找的路径**。旧结果应追溯到上表统一评测目录，以及它的 `models.json`、模型运行清单和原始轨迹。旧批次包含 7 配置 × 3 种子 × 50 扰动回合 = 1050 个扰动回合；主表的五个配置是 `Push-livePD`、`Pull-baseline`、`C-01`、`B-RZ0`、`T-07`；`B-A` 与 `T-06` 是奖励附加对照，仍属于 U3 的全量范围，不能只算主表 750 回合。核查 `models.json` 与 CSV 后再确认数目。

## U5. Push 配准摆位重训（第一优先）

### 开训前核对 0.878 m，不把它硬编码为拟合输入

已核实的第二轮几何使用：源桌在分配轴上的宽度 `0.5219528 m`，目标桌外边界 `[-0.7, +0.7] m`。目标桌两端各取一段与源桌等宽的区域：

```text
role 0 interval = [-0.7, -0.1780472] m
role 1 interval = [ 0.1780472, 0.7] m
Q_0 ≈ I,  c_0 = (-0.4390236, 0, 0) m
Q_1 ≈ I,  c_1 = ( 0.4390236, 0, 0) m
η_0 = η_1 = +1
separation = ||c_1 − c_0|| = 0.8780472 m ≈ 0.878 m
           = 1.4 m − 0.5219528 m
```

这是 **Push 两个相同源机器人角色的局部平移差/间距**，不是目标桌长（目标轴向长度是 1.4 m），也不是每台机器人的单侧平移。正式数值从实际配准结果读入，保留全精度；不要给每台机器人都平移 0.878 m。

核验步骤：

1. 从第一批 `push/input_manifest.json` 读取实际源/目标 URDF 与源参考，验证 sha256；源参考已知为 `src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz`，历史 sha256 为 `b669468ade12f8c271119d6c8c120463223ab65489805c486e43b0c3a4ba2207`。以 manifest 指定的原资产为准。
2. 查第二轮 `push/correspondence_spec.json`、`push/selection.json` 的区域、`role_parameters`、`Q`、`c`、`parity`，重新计算间距；需要重算时运行上表第二轮脚本到**新输出目录**。`run_reference_geometry_round2.py` 读取第一批目录，不能在没有第一批资产时凭空重建输入。
3. **优先保留旧 `PairedA1Reference` 的原始参考通道和输入处理，只把配准求出的全精度 `lateral_spacing_m=0.8780472` 同时传给 setup/reset/step 的 `PairedA1MotionCommand`。** 先逐帧核对该路径生成的两角色 root、root orientation、joint、object 和相对摆位与 `push/new_pair_reference.npz` 一致；该数值来自已核验 Q/c，不是预设给拟合器的目标。两角色全段使用固定物体局部变换，物体轨迹不改，根部旋转/关节符合 `map_robot`。`push/new_runtime.npz` 用于离线交叉检查，**不能默认替换旧 reference loader**：新 runtime 的离线 FK/差分 body/velocity 通道与旧缓存通道未证明等价。若确实必须换 runtime，先逐项核验非摆位通道、采样/差分/缓存规则，保证两种摆位采用相同输入处理并记录数值差异；无法保持摆位单因素时提前报告，不静默替换，也不自行扩大为旧组重训。不要把旧 0.8 m 平移与新参考重复叠加。
4. 保存 `preflight_push_registered.json`（本轮新增核验记录）和核验日志：包括输入/输出 hash、两个 Q/c、全精度间距、物体质量、真实 training command、初始化 checkpoint hash、训练预算、诊断状态与处理方式。

**已有诊断不能抹掉：** 第二轮 `push/selection.json` 的状态是 `diagnostic_only`，因为冻结的源相对支撑漂移筛查未通过；报告中新参考最大新增支撑漂移 34.41 mm，旧构造 31.13 mm。配准残差接近浮点零，不等于动作已验证。作者本轮要求通过重训检验该参考，保留这条来源事实，不把旧状态改为 passed，不为了通过筛查加入逐帧 IK、修改阈值或改奖励。若重算结果与上述尺寸/变换不符，先定位资产、坐标系或脚本差异；未排除前不要启动错误摆位的长训练，并及时反馈作者，同时继续 CPU 工作。

### 训练对照

仅改变 Push 的参考摆位/参考资产，其余与旧 `Push-livePD` 链保持一致：20 kg 物体、奖励项及权重、PD、环境数量、初始化方式、训练预算。三个种子 721、722、723 从旧实验所用的**同一个单机器人跟踪 checkpoint** 初始化；不要把已有协作 Push 策略继续训练冒充重新训练。

原链定位：

```text
/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Push/
  a1_livepd_20kg_continue7000_from_08050_seed721_env2048/run_config.json
```

追该配置及其前序配置核实实际初始化 checkpoint。**本轮与正式主表对齐的固定预算为 13050：50 轮 critic warm-up + 8000 轮 + 5000 轮续训。** 依据是 Ubuntu 原评测材料 `evaluation_protocol.md` 中固定 Push13050 的规则，以及 `RESULTS_REVIEW_CN.md` 中“旧续训请求7000轮，但固定13050，只纳入8050之后前5000轮，与新增种子预算一致”的记录；这些文件可在原统一评测输出/`paper_handoff_exports/` 的 `paper_supplement_evaluation/` 中定位，并用 `training_evidence/Push-livePD_seed722/stage2_run_config.json` 的 `iterations_requested=5000`、各种子阶段配置与 checkpoint 迭代复核。目录名的 `continue7000` 是旧请求，不是三个正式种子的共同预算，不能据此扩为15050。保存每段起止迭代、critic-only 与 actor 更新数，固定在13050评测，不根据50个测试扰动挑模型；不重跑原 prescribed 组。每种子记录 checkpoint 路径与 sha256、训练/评测完整退出状态。

### 统一评测与交付

沿用 `evaluate_paper_supplement.py` 的 actor 均值动作、无 observation noise、每任务同一份 50 个扰动初始状态，加 1 个 nominal。Push 复用原 `push_states.json`，已知 sha256：`1799ea1eec1a3ff097d46e08e6492694d6455490ccf9f2de26586e079fef895d`。扰动为每台机器人初始平面位置 ±2 cm、yaw ±2°；物体及其它初始化规则不改。registered 与 prescribed 必须施加同一组扰动偏移，不把旧绝对根位置写回而覆盖新配准摆位。

旧评测脚本会直接取 `g1_29dof_plan5_push_baseline`，**不能只换 checkpoint 而仍加载旧 0.8 m command**。registered 评测采用与训练相同的原 `PairedA1Reference` 输入通道，setup/reset/step 均使用从 Q/c 核验得到的全精度配准间距；运行前导出实际采样结果并保存其 hash、参数及逐通道差异核查。不要仅为配准摆位切换成 `new_runtime.npz`，必要变更按上述非摆位通道等价性要求处理。原队列脚本目标数硬编码为 21，扩展本轮队列时按实际模型数调整验证，不能误把“未到 21”当成本轮失败或反之。新旧结果分目录保存，不覆盖旧 1050 回合。

每个种子交 50 个扰动 + 1 nominal 的轨迹、run manifest、初始状态与退出记录，另交一个 nominal 回放视频。保存并汇总为 `episodes.csv`、`per_seed_summary.csv`、`main_results.csv`；三个文件原字段见末尾“CSV 契约”。nominal 单列，不混入 50 回合统计。

## U6. Kick 补种子（U5 之后）

补种子 722、723，设置、单机器人初始化 checkpoint 和训练总预算与 seed 721 相同。现有入口：

```text
/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Kick/
  mirrored_kick_full8000_seed721_env2048/run_config.json
  mirrored_kick_full8000_seed721_env2048/model_08000.pt
src/holosoma/holosoma/config_values/marl/g1/kick_command.py
src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/
  rubber_hand_largetable_v1/kick/plan5_attempt09_dual_kick_mirrored_runtime.npz
```

**开训前先做 CPU 参考来源核验：** 将 seed721 的现有 mirrored runtime 与基于其实际源/目标区域、η=−1 分支得到的配准参考逐帧比较，至少包含两角色根部位置/旋转、关节、物体轨迹和物体局部相对摆位，保存最大值/RMS、坐标系、四元数符号无关的角距、输入 hash 与数值容差。若尚无该 Kick 配准输出，在 CPU 上按 U1 的对应与公式生成用于核查的参考，不把文件名中的 mirrored 当成已证明的 registration。若等价，可在 README 明确写出“旧参考与配准输出等价”的证据；若不等价或缺乏可检验的对应，三种子沿用真实 seed721 的 legacy 参考设置，交付中标为 legacy mirrored 并告知 Mac，不将 seed721 冒充新配准参考训练，也不自行追加第三个重训扩大范围。

历史记录为 8000 轮，仍需从运行配置核实真实预算和初始化来源。三种子全部按 U5 的统一协议评测。旧 `evaluate_paper_supplement.py --case` 只接受 push/pull/chair/bucket/table，没有 Kick；需要增加 Kick 配置分支并验证 action/observation、reference、物体和 checkpoint contract，不能把 Kick 伪装为 Pull。沿用 `paper_evaluation_utils.py` 中生成扰动列表的规则为 Kick 固定一次 50 个状态并加 nominal，三种子共用同一文件，保存生成参数及 sha256；不是复用另一任务的绝对初始状态。

交付格式同 U5，放 `U6_kick/`；包括 721 统一重评结果、722/723 新模型与各自 nominal 视频。U1/U3 追加 Kick 数据时沿用同一资产。目标同为 2026-10-31 前；预期无法完成必须提前告知作者。

## U1. 角色构造的参考质量（CPU，不训练）

任务：Push registered/prescribed、Pull registered；U6 完成则补 Kick。全部在**腕部优化之前的位置层面**测，使用重定向后的机器人手 link 原点，不用人手点、渲染网格像素或策略动作代替。

**接近帧定义：** 读取第一批 `source_interactions.json` 的每手持续接近窗口和 `source_thresholds.json`。已知检测来自 `reference_geometry_pilot.py`：手部表面样本到实际外露面的无符号距离 ≤0.03 m，至少 5% 样本满足，持续至少 0.1 s，填补 1 帧间断；保留竞争面。拟合 256 样本/手，独立几何检查 1024 样本/手。窗口是几何 proximity，不是真实接触力标签。U1 使用这些已冻结的源窗口；若本机实际版本不一致，记录差异而不调阈值凑结果。

设源物体局部手点为 `x_src(t)=R_o(t)^T[p_hand(t)−p_o(t)]`，对应的源有限表面集合为 `F_src`；配准角色 i 的点为 `x_i(t)=Q_i x_src(t)+c_i`，目标对应有限表面集合为 `F_i`。距离取点到**有限面片集合**的最小欧氏距离，不是到无限平面：

```text
d_src(t) = min_{y∈F_src} ||x_src(t) − y||₂
d_tgt(t) = min_{y∈F_i}   ||Q_i x_src(t) + c_i − y||₂
Δd(t) = |d_tgt(t) − d_src(t)|
```

按每个角色、每只手的源接近帧统计 `1000 × mean(Δd)` 和 `1000 × max(Δd)`，单位 mm。映射后的物理左右手遇到反射时按 `g1_sagittal_joint_map` 对应，不能把另一只手误配。

**配准及残差直接定义如下，不依赖论文公式编号：** 对候选对应锚点 `a_k→b_ik` 和归一化权重 `w_k`，

```text
(Q_i,c_i) = argmin Σ_k w_k ||Q_i a_k + c_i − b_ik||₂²
subject to Q_i^T Q_i = I, det(Q_i)=η_i, Σ_k w_k=1
E_i = Σ_k w_k ||Q_i a_k + c_i − b_ik||₂²
residual_rms_mm = 1000 sqrt(E_i)
```

实现为 `scripts/reference_geometry_math.py:procrustes`；窗口权重按 `reference_geometry_round2_correspondence.py:tie_safe_weights`，每只活跃手总权重 1/2，同一源支持样本的竞争面分摊一份证据。Push η 为 (+1,+1)，Pull 为 (+1,−1)。采用现有选定对应并保存锚点/权重/选择结果；不可为更低残差改删窗口。

Push prescribed 比较使用 `command.py` 原 `lateral_spacing_m=0.8`，即 Q=I、两角色 c=(±0.4,0,0) m；也在**同一套对应锚点和窗口**上重算残差，不能照抄 registered 最优残差。registered 用选出的实际 Q/c。

交 `ref_role_synthesis.csv`，原字段约定原样保留：

```csv
task,role,parity,placement(registered|prescribed),separation_m,residual_rms_mm,dd_mean_mm,dd_max_mm,n_contact_frames
```

这里 `placement(registered|prescribed)` 保留原表头字符串；单元格值只写 `registered` 或 `prescribed`。原表头没有 hand 字段；保留 `role` 为实际角色 `0`/`1`，主 CSV 每角色一行，`dd_mean_mm` 对该角色左右手各自的均值等权平均，`dd_max_mm` 取两手最大值的最大值，`n_contact_frames` 为两手有效 hand-frame 数之和。另交本轮新增的 `ref_role_synthesis_by_hand.csv` 明细（原字段加 `hand`，`hand` 为 `left`/`right`），保存每手均值、最大值和帧数；README 明确主表聚合规则及反射时物理角色/源手对应。`parity` 写 1 或 −1。任何一手无有效帧时单列缺失原因，不能默认为零。Kick 若没有可用的手部接近窗口，保留任务行并填 `--`/0 帧且说明原因，不把脚部指标改名填进手部字段。

## U2. 共享场景的参考质量（CPU，不训练）

任务：Chair、Small Table、Bucket。先用 `core4d_pair_experiments.py` 指向的 runtime、物体 URDF、`source_manifest.json`、`training_asset_manifest.json` 核实每任务实际采用共享缩放还是保留原尺寸后重新优化。Bucket 预期为后者，但必须以训练资产为准。README 按任务写出证据路径，不由图像外观猜测。

人身高 `h_i` 与机器人身高 `h_R` 从原 retargeting 输入/配置读取，`s_i=h_R/h_i`。不要猜身高。将地面归一化后源物体初始位置投影到地面得 c，源物体位置与朝向为 `p^H_o(t),R^H_o(t)`，物体局部表面为 `V_o`：

```text
S_i(x) = c + s_i (x − c)
p_nominal_i(t) = S_i(p^H_o(t))
mean scaling:
  s_obj = (s_1+s_2)/2
  V_shared = s_obj V_o
  p_shared(t) = [p_nominal_1(t)+p_nominal_2(t)]/2
  R_shared(t) = R^H_o(t)
one-sided scaling:
  s_obj = s_1
  V_shared = s_1 V_o
  p_shared(t) = p_nominal_1(t)
  R_shared(t) = R^H_o(t)
source-size option:
  s_obj=1, V_shared=V_o,
  p_shared(t)=p^H_o(t), R_shared(t)=R^H_o(t)
size_mismatch_pct(i) = |s_obj/s_i − 1| × 100
```

共享缩放任务比较 `one-sided` 与 `mean`；保留原尺寸任务比较 `nominal`（名义机器人运动直接配源尺寸物体）与 `re-optimized`（固定同一源尺寸物体和轨迹，重新优化机器人）。这四个字符串作为本轮 `variant` 的值，不改原字段名。均值缩放最小化两个 scale 的最大**绝对**偏差，不保证每个参与者的相对百分比都降低。

每参与者的 `d_src` 使用其示范经 `s_i` 缩放后的手–表面距离作为参照；`d_tgt` 用实际名义/重新优化机器人手 link 对变体共享物体的距离，Δd 同 U1，仅在源接近窗口、腕部优化前测。保存窗口、源/目标坐标系、人体到机器人手点对应规则；不得改用优化后的腕部姿态。每参与者先分别计算左右手均值，再等权平均为该参与者均值；最大值取两手最大值的最大值，`n_contact_frames` 在本轮明确表示该 participant 的有效 hand-frame 样本数，并在 README 写明两手计数，避免将不同窗口混为同帧。

交 `ref_shared_scene.csv`：

```csv
task,participant,height_m,s_i,variant,s_obj,size_mismatch_pct,dd_mean_mm,dd_max_mm,n_contact_frames
```

另附每任务两人的 `h_i`、采用的 `h_R`、名义/最终物体 scale 与轨迹来源。供 Mac 主表使用时，尺寸失配保留两人各自的百分比成对值；Δd 均值在角色/参与者的各手组间等权平均，最大值取所有组的最大值，不能按各组窗口长短不加说明地混池。若现有 hand-frame 定义或元数据不能支持计算，标 `--` 并说明缺少哪项，不拿零误差替代。

## U3. 身体与手部跟踪误差（CPU，从现有 NPZ 计算，不重跑）

处理旧评测全部 1050 个扰动回合、U5 新回合、U6 完成后的新回合。nominal 可另交诊断，但不得混入统计。

**已知存储事实：** 旧 `evaluate_paper_supplement.py` 的 trajectory NPZ 保存了 `root_pos`、`root_quat_xyzw`、`root_velocity`、`dof_pos`、`dof_vel`、`object_pos_w`、`object_quat_xyzw`、`reference_object_pos_w`、`reference_object_quat_xyzw`、`reference_frame`、`reward_reference_frame`、`episode_step`；未直接保存全身 link 世界位置。不要以“没有 body_pos 数组”就要求重跑仿真。可用原训练 URDF + 保存的根姿态和关节角做 CPU FK，再通过原 paired runtime/reference loader 得到参照。

先抽一条轨迹验证 joint_names 顺序、米/弧度、`xyzw` 与 FK 需要的四元数顺序、URDF 根 link、两机器人的轴序、环境原点及 reference 采样规则。可复用 `RobotFK` / `export_recorded_demo_media.py` 的 URDF 和 DOF 顺序校验；不要用渲染 mesh 质心代替 link 原点。`body_names_to_track` 从每个实际运行配置及 motion command 读取，旧 Push 的 14 links 是 pelvis、双侧 hip_roll/knee/ankle_roll、torso、双侧 shoulder_roll/elbow/wrist_yaw；U3 仍须按每配置实际列表执行。手部用 `left_rubber_hand_link`、`right_rubber_hand_link`，若实际资产命名不同，列出明确对应并核验，不能静默用 wrist_yaw 替代手。

以每回合 K 个有效记录状态、A=2 台机器人、身体 link 集合 B 定义（包含初始状态，见下文）：

```text
E_body_cm = 100/(K A |B|) × Σ_t Σ_a Σ_{l∈B} ||p_{a,l}(t) − p*_{a,l}(t)||₂
E_hand_cm = 100/(K A × 2) × Σ_t Σ_a Σ_{l∈{left_hand,right_hand}} ||p_{a,l}(t) − p*_{a,l}(t)||₂
```

全部为**世界坐标位置误差**，不使用身体跟踪奖励中的平面重对齐。按旧 `paper_evaluation_utils.py:metrics` 的帧口径：**包含初始化 `episode_step=0` 和终止前最后一帧**。旧评测代码用 `frames=[start]` 后追加各物理步，metrics 对全数组求均值，所以通常 `K=steps+1`。物理步 k 对齐记录的 `reference_frame=min(k,N−1)`，末帧在 autoreset 前已捕获。`reward_reference_frame` 仅作核对，不能不说明就混入另一套对齐。对象 RMSE和 U3 使用同一组记录状态；若另给排除初始状态的敏感性统计，必须单独标记，不能覆盖上述主口径。

两种统计口径都给：

- `completed`：只保留 `completed_reference=True` 的回合。
- `prefix`：所有回合截至终止时的有效前缀（含初始状态），包括已完成回合。

逐回合先求均值，然后每种子按所选回合集合平均，最后跨种子 mean 和 sample SD（ddof=1）。没有完成回合的种子，其 completed 指标是缺失，不填 0；说明有效种子数，不把其它种子均值复制过去。

交 `body_tracking_episodes.csv`，字段保持：

```csv
model_id,configuration,seed,state_id,completed,frames,E_body_cm,E_hand_cm
```

其中 `completed` 映射旧 `completed_reference`，`seed` 映射旧 `training_seed`，`frames=K`，一行一个扰动回合。**不要把旧评测 CSV 的字段也改名。**

交 `body_tracking_summary.csv`，每配置两种 scope 各一组。原任务单只指定文件名和统计要求，没有给 summary 字段；本轮新增明确约定如下，需在 README 标注“本轮汇总 schema”，不要声称这是旧表既有字段：

```csv
configuration,scope,training_seeds_evaluated,valid_seeds,episodes,E_body_cm_mean,E_body_cm_sample_sd,E_hand_cm_mean,E_hand_cm_sample_sd
```

`scope` 为 `completed` 或 `prefix`。另附 per-seed 中间结果和 FK/对齐核验记录。若原 reference、URDF 或 DOF 映射确实丢失，在 README 逐回合列出阻塞原因，保留 `--`；不自行启动 GPU 重新采集来补 U3。

## U4. 近距离代理（CPU，可选）

对象：Push prescribed/registered、Pull；有余力再做三个 CORE4D 任务。使用参考中已冻结的接近窗口，与实际回放时间对齐；提前终止回合只用其存在的窗口帧。使用机器人手、小腿、脚的真实碰撞几何，通过 U3 的 CPU FK 变到世界系；物体碰撞盒从本任务实际 URDF 读入，包含 collision 的局部 origin/rotation。

对每类部位，逐帧计算该类任一碰撞几何到物体碰撞盒并集的最小距离，统计距离 `<0.03 m` 的帧占比。交叠时距离记 0 即满足邻近，不扩展为穿透深度指标。固定并列出 hand/shin/foot 的左右 link 名单和几何类型；若采用包围盒代理而非真实几何，应另列说明，不把 SAT 分离下界冒充精确欧氏距离。每回合先算占比，再按配置/种子对有效回合等权平均；附分母与被排除回合清单。

交 `proximity.csv`：

```csv
configuration,seed,part(hand|shin|foot),proximity_pct
```

`part(hand|shin|foot)` 保留原表头字符串，单元格仅写 `hand`、`shin`、`foot`。这是 proximity 距离代理，不是接触力，也不能单独证明用哪一部位产生了推力。

## U7. 渲染与图像来源（不需要论文图文件）

所有交付静帧宽度 ≥1600 px，建议 1920×1080，直接按目标分辨率重渲染，不放大已有 1280×720 位图冒充新细节。使用真实 URDF、真实参考/执行状态；输出无字原图和标注建议，帧、时刻、阶段等写 manifest，便于 Mac 排版。

### 相机与灯光的 Ubuntu 来源

可直接核验 `scripts/export_recorded_demo_media.py` 及旧媒体 manifest；已知旧脚本 sha256 为 `d11abb4bc6e91c7dc21aee1a1a50543e1ac471d3409c132bda324cb2624cd8be`。这份脚本的可复现设置为：

- MuJoCo free camera，`--camera-mode follow-pair --azimuth 135 --elevation -22 --distance-scale 1.0`。
- lookat 的 XY 每帧取“两台机器人根位置 + 物体原点”三个点的均值，Z 固定 0.75 m。
- 每帧上述三点在 XY 的跨度向量为 span(t)，整段共用距离 `max(2.8, 0.65 × max_t ||span(t)||₂ + 1.3)` m。旧 Push/Pull/Chair/T-07 的 manifest 距离为 2.8 m；新轨迹仍按脚本核算，不盲套旧自动取景值导致裁切。
- headlight：diffuse=(0.5,0.5,0.5)、ambient=(0.3,0.3,0.3)、specular=(0.05,0.05,0.05)。顶光位置为全段包围范围中心（Z 改为0.75）+[0,0,5] m，方向(0,0,−1)，diffuse=(0.5,0.5,0.5)、specular=(0,0,0)，castshadow=true。shadowsize=2048、offsamples=4；其余材质、背景和地面保持该脚本。

先在 Ubuntu 定位旧 `media/Push-livePD_seed721/manifest.json`、`media/Pull-baseline_seed721/manifest.json`、`media/C-01_seed721/manifest.json` 和 `media/T-07_seed721/manifest.json`，验证脚本 hash、`rollout_source`、`outputs_sha256`。已核对现有拼图对应的原帧：Push-livePD_seed721 为 0000/0185/0309，Pull-baseline_seed721 为 0000/0190/0316，C-01_seed721 为 0000/0313/0391，T-07_seed721 为 0000/0412/0687；其中 T-07 的原回放目录是 `results/T-07_seed721_retry_shutdown_20261004/`。原拼图的小桌来自 T-07，与正式主表配置一致；T-06 是奖励附加对照，不能将二者互换。若另选配置取图，明确标成新的取图并给来源。若本机另一份生产渲染脚本才是原拼图来源，以该 manifest/渲染命令为准，保存完整配置及差异；**没有可核验 provenance 时，不声称“相机与原图完全相同”**，写出当前选用的已知设置和待核实项。无需作者带来 Mac 图文件即可完成这些来源核对。

每组图实际保存 `camera.json` 和 `render_manifest.json`：相机类型/投影、lookat（含逐帧 follow 值）、distance、azimuth、elevation、fovy/orthographic 状态、分辨率、灯光、脚本/资产 hash、输入 NPZ、帧号与秒数。不把以上取景规则当成未知 fovy 的替代值；fovy 从实际生成的 MuJoCo model/配置读出。

共享场景对比只需角色颜色一致：role 0 蓝、role 1 橙；无需查看论文中的颜色参考图。颜色是显示样式，RGBA 写入 manifest，不改变机器人/物体几何或运动。两对比面板固定完全相同的相机、光照、frame 和尺寸，不能每面板独立自动取景。此处图像要表达的内容已在下文写全，不需任何“参照图 2 / 图 4”的外部说明。

### 三组内容

1. **执行序列：** Push registered 取 3–4 个关键帧，Kick 完成则也取；每种子另交 U5/U6 的 nominal 视频。给 `model_id`、seed、checkpoint hash、rollout id、原 NPZ 帧号、`episode_step`、`reference_frame`、`time_s` 和阶段（接近、接触、搬运、放下；不存在的阶段写真实名称，不强行凑齐）。阶段标签是本轮排版所需，原图保持无标签。另对上文已列出的原 Push/Pull/Chair/Small Table 共12个帧补交真实阶段标签及依据；保留原取帧，不为凑四种阶段另选“更好看”的回合。若帧不处于上述阶段，写符合实际的简短阶段名。
2. **Chair 共享场景对比：** 同一帧两张图。(a) 两机器人分别配自己的 s_1/s_2 缩放物体，两个物体副本半透明并按角色着蓝/橙；使用各自名义物体位置。(b) 两机器人与一个共享物体，scale/pose 用 U2 核实的实际规则。附 h_1/h_2、s_1/s_2、s_obj、几何与帧来源，便于标明尺寸差；不凭图像测尺寸。
3. **Push 失败案例：** 从有记录的未完成回合选两帧，能清楚展示小腿/脚靠近或推动桌子的实际姿态；另交与其中一个时刻严格对齐的参考姿态。附 `configuration`、`model_id`、seed、`state_id`、失败原因、两时刻的 frame/秒数、参考对齐方式及选帧理由。原图 + 红色 callout 建议分别交。若实际回放没有展示该行为，应如实说明并交能支持的案例，不能为了符合预设解释编造动作或用参考帧冒充执行。

## U8. 事实确认（CPU，先交 README）

1. **提前终止准确含义。** 核对每配置实际保存的源代码和配置：`src/holosoma/holosoma/managers/termination/terms/marl.py:JointBadTrackingZOnly`、`.../terms/core4d_smalltable.py`，及 `config_values/marl/g1/termination.py`、`core4d_smalltable_termination.py`。已知版本的 `joint_bad_tracking` 不是关节角误差；它计算以下 OR：任一机器人参考/实际 torso 姿态下投影重力的 z 分量差绝对值 >0.8，或物体世界位置误差 >0.25 m，或物体姿态旋转误差 >0.8 rad。投影重力为 `R_torso^T (0,0,−1)`；姿态误差为归一化四元数相对旋转的最小夹角。躯干高度及特定身体点高度差只记录 diagnostics，在该版本不参与终止。确认 Kick 与新 Push 的实际配置是否相同；列出其它 active termination terms、阈值、函数、源码 hash，并从旧/新 episodes 的 `terminal_terms` 分别核实失败原因分布。“完成”是达到参考末端且未触发失败终止，不等价于证明接触方式正确。
2. **逐任务设置。** Push prescribed/registered、Pull、Kick、Chair、Bucket、Small Table 分别列出 partner reference 类型、初始化 checkpoint 或 scratch、所有训练阶段与累计更新数、采用 checkpoint 迭代、物体真实质量、参考帧数/秒数、reference FPS、控制频率、物理步长/decimation、`w_z`。CORE4D 再列 shared-scale/source-size 方案与资产 manifest；报告 h_R/h_i、s_i 的来源，支持 U2。参考秒数说明按 `N/fps` 还是首末采样时间 `(N−1)/fps`，避免混写。
3. **腕部。** 查 `scripts/render_wrist_reference_pairs.py`、原 wrist refinement/hand-keypoint 输入、`paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005/` 的源材料，以及 Ubuntu `docs/experiments/retarget_method_comparison_20260925.json` 的 Chair 报告，明确哪些任务有真实手关键点、哪些只有 fixed-palm 假设。Chair 既有指标是**完整掌框架的 SO(3) 角距**，不是仅掌面法向夹角。令 `R_target`、`R_actual` 为同一世界坐标系中的目标与实际掌框架旋转矩阵，计算 `θ=acos(clip((tr(R_target R_actual^T)−1)/2,−1,1))×180/π`；不得换成两个法向的点积，后者会遗漏绕掌面法向的旋转。先核对既有 Chair 掌框架轴、左右手约定、分母/帧窗口再推广。交 `wrist_orientation.csv`。原任务单没有给字段，故本轮明确新增约定为 `task,role,hand,target_source,n_frames,before_mean_deg,after_mean_deg,after_max_deg`，README 标注此为本轮 schema；`target_source` 区分真实关键点和假设目标框架。Chair 已有总体均值 62.72°→0.06°、**各手序列均值 <0.24°**，并非各手逐帧最大值都小于0.24°；最差帧为5.0077989301605434°（约5.008°）。这些数值用于同口径复算校验，不能复制成其它任务结果。缺完整目标掌框架的任务写明不可测原因；若只有法向，可以另给明确命名的法向指标，但不能填进上述 SO(3) 误差字段。
4. **摆位来源。** 交旧 `command.py` 中0.8 m的配置出处/运行时生成函数，以及 U5 中0.8780472 m的原 URDF、有限区域、Q/c、脚本、原 `PairedA1Reference` 采样导出 hash 和离线新 runtime hash；明确后者是角色间距、原配准导出为 `diagnostic_only`。交 U6 中 Kick 原 mirrored runtime 与 η=−1 配准参考的逐通道等价性核验及命名结论；镜像名称本身不能证明由配准得到。
5. **接触正则的实际范围。** 核对 `config_values/marl/g1/reward.py` / `core4d_smalltable_reward.py` 的 `undesired_contacts_body_names` 以及 `managers/reward/terms/marl.py:UndesiredContacts`。已知 regex 排除左右 foot_contact_point、ankle_roll、wrist_yaw、rubber_hand，而 knee 等链接匹配其中；不能写“所有腿/脚都不受罚”。在本机解析到实际 body_names，列出每个手/小腿/脚相关 link 是否受罚、threshold/weight、使用净接触力还是物体过滤接触传感器；区分软惩罚、硬接触限制与早停条件。不要据此额外设计消融或更改 U5/U6 奖励。

## CSV 契约与统计验收

所有原指定 CSV 文件名、原指定字段名保持。U5/U6 放在各自子目录，同名文件互不覆盖。旧统一评测表的列顺序和名称如下，单位也不变（对象误差原列仍为 m，不在原 CSV 中偷偷改为 cm）：

`episodes.csv`

```csv
model_id,case,state_id,kind,steps,horizon_steps,completed_reference,progress,terminal_terms,reward_sum_common,object_rmse_3d_m,object_rmse_xy_m,height_rmse_m,height_bias_m,trajectory,configuration,training_seed,training_reuse_status,checkpoint_sha256
```

`per_seed_summary.csv`

```csv
model_id,configuration,training_seed,reuse_status,episodes,completion_rate,progress_mean,completed_object_rmse_3d_m,completed_episodes,prefix_object_rmse_3d_m
```

`main_results.csv`

```csv
configuration,training_seeds_evaluated,target_training_seeds,complete_three_seed_result,completion_rate_valid_seeds,completion_rate_mean,completion_rate_sample_sd,progress_mean_valid_seeds,progress_mean_mean,progress_mean_sample_sd,completed_object_rmse_3d_m_valid_seeds,completed_object_rmse_3d_m_mean,completed_object_rmse_3d_m_sample_sd
```

新 Push configuration 可用 `Push-registered`，旧 `Push-livePD` 保留；Kick 的新增 configuration 名称在开训前冻结并记录。`training_reuse_status`/`reuse_status` 按实际来源写，不把新训练伪标历史复用。保留 `nominal_results.csv` 或等价独立 nominal 汇总以及原始 episodes JSON/NPZ。对无法导出字段写 `--` 与原因，不将新 summary schema 当成已有字段契约。

验收：每新配置三种子，每种子 50 个唯一 perturbed state_id + 1 nominal；三种子状态文件 hash 相同；同源任务比较扰动完全配对。Completion/progress/object RMSE 与 U3 的种子内/种子间统计分开复算，sample SD 使用 ddof=1；未完成回合不能进入 completed RMSE。失败回合保留，不按结果过滤；reference 质量只从参考算，执行质量只从实际轨迹算。

## 打包与交回 Mac

在 Ubuntu 本地创建全新的输出目录，例如 `ROOT/paper_handoff_exports/EXPERIMENTS_R2_20261030/`。下面是建议包结构；占位日期取实际交付日，不把例子当截止日变更：

```text
EXPERIMENTS_R2_<日期>/
  README.md
  provenance.json
  SHA256SUMS
  preflight_push_registered.json
  ref_role_synthesis.csv
  ref_role_synthesis_by_hand.csv    # 本轮明细补充，原主 CSV 字段不变
  ref_shared_scene.csv
  body_tracking_episodes.csv
  body_tracking_summary.csv
  proximity.csv                    # U4 可选；没做则 README 明确写
  wrist_orientation.csv
  U5_push_registered/
    episodes.csv
    per_seed_summary.csv
    main_results.csv
    nominal_results.csv
    manifests/  trajectories/  videos/  checkpoints/
  U6_kick/
    episodes.csv
    per_seed_summary.csv
    main_results.csv
    nominal_results.csv
    manifests/  trajectories/  videos/  checkpoints/
  U7_media/
    rollouts/  shared_scene/  failure_case/
    camera.json
    render_manifest.json
  sources/
    scripts/  configs/  code_diff.patch  reference_manifests/
  checks/
    state_inventory.json  statistics_check.json  fk_alignment_check.json
```

README 必须自包含，至少列出：

- U1–U8 各项完成/部分完成/未完成状态、剩余工作、是否影响 10/31，以及每项文件位置。
- **每份交付文件的来源：代码 commit、实际脚本路径及 sha256、完整命令、配置文件及 sha256、输入数据/参考/URDF 的原 Ubuntu 路径与 sha256、checkpoint 路径与 sha256。** 无 checkpoint 的 CPU 几何任务明确写“不适用”，不能留空冒充遗漏已补齐。
- 文件到配置/种子/任务/参考版本的对应；统计定义、单位、scope、有效回合/帧/种子数；U8 全部事实及源码证据。
- 新增和未跟踪代码的快照/差异，Python/关键依赖版本、运行时间、返回码、已知限制。commit 与实际执行文件存在差异时说明差异。
- 图像对应的真实 NPZ、物理帧、reference frame、时刻、相机/灯光、渲染命令及 sha256。不能只写“同论文设置”。

checkpoint 至少给原路径、sha256 和可追溯配置；新模型体积允许时放入包内 `checkpoints/`，太大则另打模型分卷并在 README 列出分卷与 hash，不能交只有 Ubuntu 绝对路径的不可移动清单。评测原轨迹、必要参考/manifest、CPU 计算脚本与配置随包带回，避免 Mac 只能看 CSV 而无法查来源。禁止用指向 Ubuntu 文件系统的符号链接代替文件；CSV 原 `trajectory` 路径字段不改，但额外清单给出它在包内的相对位置。旧 1050 轨迹若单独成数据包，在 README 写出归档名/hash及映射。

生成 `SHA256SUMS` 覆盖除校验清单本身外的所有交付文件，打包后另给归档的 sha256；在新的临时目录试解包并校验，确保没有绝对路径依赖、缺图、丢配置或仅本机可解析的链接。分阶段包各自标注覆盖范围，不把部分结果标记 complete。

**由作者把压缩包和校验和带回 Mac**，解包后的目标为：

```text
What my Ubuntu agent wants to tell you/EXPERIMENTS_R2_<日期>/
```

该路径相对于作者的 Paper 工作目录；不要求 Ubuntu 能访问它。执行者只负责在 Ubuntu 准备可搬运的自包含交付包，并把包名、大小、sha256、完成状态和截止风险报告给作者。
