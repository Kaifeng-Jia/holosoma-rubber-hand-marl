# DuetMimic v3：Ubuntu 两批任务执行令（自包含，2026-11-02）

**修订 1.4（2026-11-02 17:20 ET，Kevin 确认，Claude 下达）：见同目录 `ubuntu_v3_rev14_priority_gpu_20261102_zh.md`。内容：立即停训，从第 2000 轮续训并升级到 6 卡；P0/P1/P2 优先级；取消 V3-28–30；速度诊断；新截止。与本文件冲突时以修订 1.4 为准。**

**修订 1.3（2026-11-02，响应 Fig.2(c) 预检）：旧R6下图为优化前姿态、1920×1280，不能直接与新版C-01优化后上图配对。该项已按既定一致性要求处理：上下两图均用同一C-01优化后构型重新渲染，保留原视角、帧117对应源时刻和光照，统一1920×1080并记录新投影。继续渲染，不再因已报告的这一处旧图差异等待确认；训练照常。下方U5已同步。**

**修订 1.2（2026-11-02）：纠正 U3c 腕部数量——每只手3维、每台机器人双手6维、两台四手共12维。Chair/Bucket 的真实 before/after 参考与优化代码均已核实；原“共六”是交接文字错误。按12维腕部范围继续既定无腕部消融，测试通过后即可启动，无需为此计数错误等待再次确认。修订1.1的Fig.2(c)单侧尺度实体＋第二参与者nominal虚影保留；训练run数量、预算与截止不变。**

**授权：Kevin 已授权现在执行 U1–U5、27 个正式训练 run（条件满足时最多 30 个）及下列全部评测。** 不要再次等待“是否开训”确认。每个受影响项须先完成本文件的输入核验和 smoke test，再启动；独立项并行。旧交接中“未授权训练”“先看参考再确认”的限制被本轮明确授权覆盖，但不能擅改实验定义、缩短预算、增加 run 或改变资产来赶时间。

**时间统一为 America/New_York（本轮 UTC−05:00）。** 第一节点：2026-11-02 23:59 前回传 U1/U2、已完成 U5 和全部可独立完成的评测；最终节点：**2026-11-03 12:00−05:00 前回传训练与全部评测**。收到任务后立即登记接收时间和实际资源，并给各 run ETA。截止是目标，不能为了兑现日期伪报完成、降低训练预算或省种子。预计超期立即报告剩余 GPU 小时、已完成项、受阻项和最早可交付时间。

**状态规则：本文件是执行委托，不是启动回执。** 只有 Ubuntu 真实进程、PID/作业号、命令、run_config、日志首轮迭代齐备，状态才能写 `running`。仅上传 GitHub 写 `delivered_waiting_for_receiver`。当前 Mac 已检查历史材料，未发现可证明仍在运行的 Ubuntu GitHub 自动轮询/收任务机制；接收方需要实际读取本文件并执行。不得声称 GitHub 上传自动启动 GPU。

**截止风险与启动优先：** 历史Bucket单run12000轮纯迭代时间约9.065小时（RTX4090）/14.874小时（RTX5070 Laptop），见旧bucket_AB报告；这不是新run的速度保证，27run能否在11/3中午完成取决于实际并发资源与耗时。收到后立即以本机smoke/早期迭代测吞吐，给总GPU小时和排程。优先启动资产核验最简单且smoke通过的Chair/Bucket no-wrist与其他已就绪项；不要等全部U1–U5完成，也不因某学习开关仍开发而空置GPU。不得擅缩预算或增加付费资源来掩盖超期。

## 0. 不可改变的约束与 STOP 规则

1. 按实验整组报告，不能挑格、挑 seed、挑 checkpoint。四个参考模块整组保留；奇偶性唯一去留门槛为 U2 的残差是否 ≥10 mm。协作必要性、学习组件分别按整个实验判断；整体不支持则整组交补充材料，不写相应结论。扰动扫描整组保留。Ubuntu 交全部原始结果，不替作者删行。
2. 不制作失败案例图或逐种子散点；不把穿透、抖动、手–物表面距离（包括 `ref_shared_scene_handdist.csv`）纳入论文结果或图注。U2 明确要求的手–表面距离变化只交审计附件，标记 `audit_only_not_for_manuscript`。不能据此声称“共享场景改善接触”或“策略复现手部接触”。Table III 的 Hand 沿用原执行手位置跟踪误差，不能改成手–表面距离。
3. 只用真实人体/机器人渲染，禁止 AI 图片；全部原图无标签，1920×1080。Fig.1 作者暂缓，本轮不修改 Fig.1、不修改 manuscript。
4. **任务书与代码、数据、定义、hash 或实际运行配置不一致：停止受影响项并报告，不能自行变通。** 未知量填 null/缺失原因，不能估数、补零、用同名文件替换、按结果改变窗口或放宽通过阈值。仍继续无依赖的任务。
5. 每个 STOP 写 `stops/<item>.json`：`item,expected,observed,evidence_paths,evidence_sha256,affected_runs,unaffected_work_continued,question_to_author,created_at,eta_if_resolved`。记录文件与行号/数组字段。配置来源不明不启动长训练；不因一项 STOP 阻塞全部 27 run。
6. 保留旧资产、模型、结果、工作区改动。本轮写新目录。无需切换/reset/覆盖 Ubuntu 工作树；不安装或升级现有仿真依赖。已有获准计算资源可并行，新增付费资源不在本执行令中默许。

## 1. Ubuntu 根目录、来源与先行回执

仓库：`Kaifeng-Jia/holosoma-rubber-hand-marl`，分支 `core4d-base`。研究根 `ROOT=/home/kevin/holosoma-core4d-base`；旧 OMOMO 根 `/home/kevin/holosoma-rubber-hand-marl`；单机器人先验根 `/home/kevin/holosoma-rubber-hand-largetable`。CPU 解释器已知 `/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python`，仿真解释器已知 `/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python`；先核实本机存在且可用。

本轮建议输出 `ROOT/logs/V3_20261102/`，返包 `ROOT/paper_handoff_exports/V3_20261102/`。下列相对路径均相对于 ROOT；迁移时用原 manifest/hash 定位同一资产并登记新绝对路径。

| 用途 | 已存在入口/来源（本机仍须核验） |
|---|---|
| Plan5 Push/Pull/Kick 训练 | `scripts/train_plan5_push.py --skill {push,pull,kick}`；`scripts/train_plan5_kick.py` 是旧包装入口，不要臆造 `train_plan5_pull.py` |
| CORE4D 训练 | `scripts/train_core4d_smalltable.py --experiment {chair021,bucket003,smalltable5kg_A}` |
| 统一评测 | `scripts/evaluate_paper_supplement.py`、`scripts/paper_evaluation_utils.py`、`scripts/run_paper_evaluation_batch.py`；R2 的 `scripts/run_experiments_r2_evaluation.py` / retry01 和 collect 脚本有新 Push/Kick 接入证据，选实际最新且通过验证版本 |
| 六任务主模型来源 | 旧 `logs/PaperSupplement20260929/unified_evaluation_20261003/models.json` 与 `training_evidence/`；R2 的 `evaluation_models.json`、`evaluation_local_checkpoint_map.json`、`evaluation_provenance.json` 在 `paper_handoff_exports/EXPERIMENTS_R2_20261026/` 或其原输出中查找 |
| 主模型身份 | Push=`Push-registered`（不是旧 `Push-livePD` prescribed）；Pull=`Pull-baseline`；Kick=`Kick-baseline`；Chair=`C-01`；Bucket=`B-RZ0`；Small Table=`T-07` |
| R2 配准 | `scripts/run_reference_geometry_round2.py`、`scripts/reference_geometry_round2_correspondence.py`、`scripts/reference_geometry_math.py`；输入/输出在 `logs/ReferenceGeometry/`，按 `execution_identity.json` 和各任务 `selection.json` 定位 |
| CORE4D 共享物体导出 | **`scripts/export_core4d_small_table_pair.py`**（文件名有 `small_table`）；`export_small_table_pair` 函数。不能把它当作已支持本轮全部消融的 CLI |
| CORE4D 50Hz runtime 导出 | **`scripts/synthesize_core4d_smalltable_runtime_reference.py`**；底层 `src/holosoma_retargeting/holosoma_retargeting/core4d_pair_runtime_reference.py:build_core4d_pair_runtime_reference_file` |
| CORE4D 重定向/A1 | `scripts/retarget_core4d_pair.py`、`scripts/refine_core4d_pair_a1.py`；配置从真实 Bucket two_stage manifest 和源码读取 |
| Bucket 准备 | `scripts/prepare_core4d_bucket_training.py`（有已审定原资产 hash/帧数约束，不可直接拿新参考冒充原资产） |
| 训练资产索引 | `src/holosoma/holosoma/config_values/marl/g1/core4d_pair_experiments.py`，对应 motion 目录内 `source_manifest.json`、`training_asset_manifest.json` |
| 渲染 | `scripts/export_recorded_demo_media.py`；`scripts/render_wrist_reference_pairs.py`、`scripts/collect_wrist_bucket_materials.py`；批准图包 `paper_handoff_exports/FIGURES_R2_20261027_approved/sources/figround_*.py` |

收到文件后先交 `dispatch_receipt.json`：任务书 SHA256、接收时间、hostname、ROOT、git HEAD、dirty diff/hash、解释器、GPU 型号/数量/空闲显存、输出根、所有主模型的 checkpoint/config/reference/URDF/state 文件路径和 SHA256、27 条计划、各依赖状态、资源排程、预计完成时间。模型清单中云端 `/root/autodl-tmp/...` 只是原路径，必须用本地 checkpoint map 和 hash 找回真实本机文件。

主预算须逐个 seed 从主结果 run_config 验证：Chair/Bucket 12000；Pull **50 critic-only + 8000 actor+critic，总 8050**。环境数量、steps/env、奖励、PD、质量、噪声、初始化、学习率日程、终止、参考采样与主结果一致。历史协议可核对 Push13050、Pull8050、CORE4D12000；Kick R2 为8000。旧总协议“其余12000”不能覆盖后续 Kick 特例。

## 2. 第一批 U1–U3：事实与资产

### U1 Bucket 为何采用原尺寸

查 `CORE4D_BUCKET_AB_TRAINING_CN.md`、`docs/writing_handoff/evidence/reports/bucket_AB_20260919.md`（原始结果 `logs/Core4DBucket/comparison_20260919/RESULTS_CN.md`）、`MULTI_AGENT_EMERGENCE_ROADMAP.md`、原 retarget manifest/决定记录。交 `U1/bucket_size_decision.md` 与 `evidence.csv`，逐条列引用原句、文件、行号、SHA256、日期、是否能支持“当时为何这样选”。

已知历史材料证明 **采用了原尺寸、two-stage、A1、1kg**，不自动证明“缩小会破坏交接”“原尺寸必然更好”等原因。若未找到原始决策理由，明确 `reason_not_documented`，停在事实；不能把新推测写成历史决定。给方法选择准则时分“原记录支持的规则”和“待作者确认的推广”，不能用当前试验结果倒推当时原因。

### U2 Pull role 1 奇偶性

固定 R2 已选目标对应、源锚点、权重、窗口和坐标系。对 role 1 独立求 `det(Q)=+1` 与 `det(Q)=-1` 的最优刚性配准：

`min Σ_k w_k ||Q a_k + c − b_k||², QᵀQ=I, det(Q)=η, Σw=1`；`residual_rms_mm=1000 sqrt(Σw||...||²)`。

用 `reference_geometry_math.py:procrustes` 和 `reference_geometry_round2_correspondence.py:tie_safe_weights`。两只活跃手总权重各 1/2；竞争面分摊原证据。不能重新选一套更容易拟合的目标，也不能用已固定 reflection 的 Q/c 直接翻符号当 rotation 最优解。交 Q/c/det、锚点逐项残差、RMS、每只物理手有效帧数，反射后的左右手映射，以及手–有限表面距离变化审计附件。对应数据来自同一 R2 资产，单位 mm；不得替换为无限平面距离。

输出 `U2/pull_parity_summary.csv`、`pull_parity_by_hand.csv`、`fit_inputs.npz`、`fit_result.json`。summary 至少有 `role,parity,residual_rms_mm,n_hand_frames_left,n_hand_frames_right,source_hash,target_hash,weights_hash`。**η=+1 RMS ≥10.0 mm** 时导出合法 η=+1 pair/runtime/URDF（不得伪造有 reflection 的左右关节互换），完成 smoke 后启用 run 28–30，并做 Fig.2(b)。<10.0 mm 则留完整审计、明确 skip 条件，不做该列、不训练三 run。拟合/输入不一致则 STOP U2 及其条件分支，其他 27 run 不受影响。

### U3 消融参考：新目录输出，逐字段 diff

别名：

```
CHAIR=ROOT/logs/Core4DPreviews/chair021_20231020_074_baseline_20260908
BUCKET=ROOT/logs/Core4DPreviews/bucket003_20231020_071_20260918
```

| 资产 | 正确来源 |
|---|---|
| Chair 两人 nominal（30Hz） | `CHAIR/two_stage/person1/person1_nominal_scaled.npz`、`person2/person2_nominal_scaled.npz` |
| Chair pre-wrist shared mean | `CHAIR/shared_small_preview/core4d_pair_reference.npz`，SHA256 `d8e945b4ec08cc9d22d6493942cf05ed787cd42b87b5e18ac4e6d7d6423dd586` |
| Chair with wrist | `CHAIR/shared_small_a1_preview_tol6/` 的 pair，与 C-01 training_asset_manifest 逐级核对 |
| Bucket pre-wrist | **`BUCKET/two_stage/core4d_pair_reference.npz`**，SHA256 `089d3a36a39442e055f01a8111d6145475c062928c28727062f951887020ceec` |
| Bucket with wrist | `BUCKET/a1_wrist_only/core4d_pair_reference.npz`，历史 SHA256 `ea1e73c80d98ac15602871a543bf1aa6ca221a7039fc8ba98b25b3b09cc094af` |
| Bucket 主结果 runtime | `src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz`，历史 SHA256 `071ac78bac440fee72a2cf861e1c00ba3df9531b371292d930e04d61aa8b23a8` |

以上 hash 是来源核对值，当前有合法迁移须追来源，不能直接忽略。Bucket 无腕部必须用完成 Stage2 的 two_stage pair，**不能退回 Stage1 nominal**。Chair 无腕部必须用 shared mean pre-wrist，不能误取原尺寸 Stage2。单人 `qpos:T×43` 前36机器人、后7物体 `[xyz,quat_wxyz]`；pair `robot_qpos:T×2×36,object_qpos:T×7`；canonical `object_poses` 则是 `[quat_wxyz,xyz]`，须按 `[4,5,6,0,1,2,3]` 转换。

U3a Chair one-sided：物体 scale=`s1`，translation=`p_nominal_person1(t)`，rotation 与主结果相同；机器人两条动作（含腕部）**与 C-01 主参考逐值相同**。不重定向机器人，不重做 A1。历史 s1=`0.7616803111061523`，s2=`0.7267058142765197`，mean=`0.744193062691336`，必须从输入再次验证。质量、材料、PD 同主结果；导出 URDF 并报告几何/惯量处理，不顺带改变质量。

U3b Chair source-size+re-opt：物体源尺寸、源轨迹、源旋转，按 Bucket 的两阶段流程及**全构型 nominal prior**重新优化，后做 A1 wrist-only。不能以已有 Chair mean 动作直接配原尺寸物体当重优化；不能把仅关节 prior 称全构型 prior。保留原 canonical、Stage1、Stage2、A1 配置、命令、日志及 hash，明确各阶段哪些量允许改变；与主参考不同的机器人动作是该对照定义的一部分。若无法确定 Bucket 的原流程/全构型先验，STOP 该资产与三 run。

U3c Chair/Bucket no-wrist：从上表各自 pre-wrist 资产导出，与 with-wrist 比较时只允许两台机器人各自左右手腕的 roll、pitch、yaw 关节改变（每只手3维、每台双手6维、两台四手共12维），以及由此变化的 FK/速度等派生通道；root、非腕关节、物体、轨迹、时轴、采样、URDF/物理/任务参数逐值相同。不能把派生手 link 位置变化当作额外独立修改，也不能保留不匹配的新姿态旧 FK 缓存。

计数核验：Chair/Bucket 原始 pair 的 `robot_qpos[T,2,36]` 中，每台允许的零基列为 `[26,27,28,33,34,35]`，分别对应左/右 wrist roll、pitch、yaw（36维含7维浮动基）。对两个 robot index 都应用这个mask，合计12个 agent×joint 通道；不要只处理一只手或一台机器人。原G1 29关节顺序对应 `[19,20,21,26,27,28]`，runtime如重排须按 `joint_names` 映射，不能直接套索引。`scripts/refine_core4d_pair_a1.py` 的单机器人六索引与对两人的循环，以及两组实际before/after逐值差分均支持这一范围；Mac核查的非腕关节/root和物体保持一致。这是对既有腕部消融范围的准确说明，不新增优化自由度或训练run。

每个变体输出 `core4d_pair_compact_fps30.npz`（源原 fps 如不同则真实标注）、`core4d_pair_runtime_fps50.npz`、URDF/依赖mesh、`source_manifest.json`、`training_asset_manifest.json`、`diff_to_full.json`。所有反算 FK/速度/接触或关系派生缓存须与实际参考一致，并明确 B-RZ0 主奖励不含旧 A/B 的独立高度/关系项；不能因复用 Bucket 脚本就训练错奖励。生成器的 accepted hash/训练 promotion 契约必须显式拓展为新变体，不能把新数据伪装成已审定旧资产。

## 3. 第一批 U4：实现开关，先测试再开训

Mac 已检查09-25源码快照、R2执行快照，并于2026-11-02直接读取 GitHub `core4d-base` 当前训练/runner/初始化/观测源码。两个训练入口当前尚无本轮新增旗标；GitHub 的 `scripts/evaluate_paper_supplement.py` 返回404，历史Ubuntu导出快照中确有该文件，故必须定位Ubuntu本地实际版本而不能认为一次 git pull 即包含执行链。远程分支和旧导出仍不能代替Ubuntu未提交工作区的核验。以下功能是**本轮需实现并验证**，不能假定旧 CLI 已有。新选项必须进入 run_config、checkpoint contract、恢复校验和评测 manifest；关闭选项时旧行为保持。

### U4a `--zero-teammate-obs`

Plan5 和 CORE4D 均支持。actor 输入仍158维：自身154 + 队友平面相对位置2、相对速度2。真实定义在 `src/holosoma/holosoma/managers/observation/terms/marl.py` 的 `_real_teammate_planar_state_b`（按观察者 yaw frame）；Plan5 组在 `config_values/marl/g1/observation.py`，CORE4D 拼接在 `agents/mappo/core4d_smalltable_runner.py`。必须只遮去这4维，其他154维、团队奖励、central critic 不变。

测试输入不同队友状态时，actor 实际输入4维均为0且动作不受其影响；同时检验归一化**前后**，不能原始值0经旧均值归一化变为非零。保留其余154维归一化规则；单测覆盖 reset、step、batch、评测、保存/恢复。不要把 critic 对完整团队状态的读取也清零而混成另一消融。

### U4b `--local-critic`

IPPO 式 critic 跨机器人共享一套参数，逐机器人读**该机器人的158维 actor 观测**（包含本来可见的队友4维，不能顺带变成154维），输出 `V_i(o_i)`；不读其他机器人的 actor 观测或全局527维输入。共享 actor、团队奖励均不变。

旧实现 `agents/mappo/runner.py`、`core4d_smalltable_runner.py` 是每物理环境一个 team value；`storage.py` 区分 agent/team buffer。因此不能仅把网络首层从527改158：须同步 rollout、bootstrap、GAE、returns、advantages、value loss、mini-batch、normalizer 和 checkpoint 恢复。每台机器人用同一团队 reward 和团队 done/timeout，但其 value/advantage 按自身输入计算。测试：改变仅机器人1的局部输入不应改变机器人0的局部 value；两机器人交换输入后 value 交换；shape `[T,N,2,1]`；梯度来自两机器人且只一套 critic 参数；共享 reward 精确保持；timeout bootstrap 正确；保存/恢复等价。

### U4c Plan5 `--from-scratch`

不加载单机器人 actor 权重，随机初始化，Pull 仍50轮只训 critic +8000轮联合训练。旧 `agents/mappo/initialization.py:initialize_plan5_model_bundle` 强制验证/加载158维先验 actor 和 frozen actor normalizer，critic/optimizer 则新建；CORE4D 的 `core4d_smalltable_initialization.py` 是另一条 fresh 路径，不能整套换入而顺带改归一化/学习率等。

单独记录 actor 参数初始化、noise std、normalizer 来源与更新规则；按照“除 actor 初始化外其余与 Pull 主结果同配置”的单因素原则保持输入预处理，不能悄悄采用 CORE4D online normalizer。若当前主结果来源不足以落实此规则，STOP 从零组三 run 并给具体差异请求裁决。测试随机 actor 不等于先验；相同 seed 可复现；前50轮 actor 参数不变、critic 变化；第51轮 actor 允许变化；正式训练不复用 smoke 权重。

### U4d 评测 `--frozen-agent {0,1}`

被冻机器人每个控制步的 **PD joint target 固定为 t=0 参考关节构型**；不是 action=0、每步参考姿态、当前实际姿态或把整台机器人固定在世界。物理碰撞/质量/动力学保留。

提前终止不检查被冻结机器人躯干；只检查另一台机器人的躯干和物体误差，horizon/timeout保持。审计所有共享 termination 分支，避免被冻结机器人通过 team-any 的躯干检查间接终止。单测冻结 q_target 逐步不变；另一个 actor 正常动作；注入冻结躯干偏差不终止、非冻结躯干或物体超阈值仍终止。manifest 显式写此 termination 差异，不能称冻结与 Full 完全同终止规则。

### U4e 评测 `--copy-single-robot`

Push/Pull/Kick 两台均用对应单机器人 prior actor，actor mean action、相同 normalizer。优先50轮 critic-only checkpoint，但需证明 actor 与初始先验未变（含 normalizer）；否则从154→158无损构造：前154输入权重及后续层/bias原样复制，新增4列严格0，并正确扩展normalizer。不能加载已经联合训练的8000/8050/13050模型冒充先验。

测试随机与真实154维输入上原策略输出和转换后输出数值等价，改变队友4维输出不变，记录最大绝对误差及预设数值容差。如果三个 seed 指向同一个先验 actor，交代实际独立 actor 数量；不可把重复评测伪称三个独立训练先验。仍按已授权3个 seed 的评测键交齐，并标识重复来源。

### U4f 扰动状态文件

六任务各固定50个状态，±2 cm/±2°原文件原 hash 保留；新增±4/±4°、±6/±6°、±8 cm/±8°，均匀分布，沿原协议每台机器人初始平面 x/y 与 yaw 扰动，物体等其余初始化规则不变。使用固定、记录的生成 seed（可按已验证旧生成器的种子规则），**不是每训练 seed 重抽**。同任务同档全部变体/种子共用同一文件与 state_id；偏移应用在各变体自身 nominal 上，不能把绝对旧 root 写回覆盖新参考。档间如共用归一化随机样本必须记录，不强行修改旧±2文件。

### U4g Palm (°)

`R_robot_palm(t)=R_hand_link_FK(t) B_fixed_palm`；目标为**同一示范的掌面 `R_demo_palm*(t)`**，不能与各消融自己的参考掌面比。SO(3) 测地角为 `acos(clip((trace(R_demoᵀR_robot)-1)/2,-1,1))*180/pi`。记录手 link 名、固定基、左右手与角色映射、示范时轴/重采样、单位和hash；反射角色必须用已验证正确的掌面目标变换。

仅完成回合：两台×两手×所有有效帧平均，保留每手每帧明细与计数。Kick无目标记空值/`not_applicable`，展示“–”。其他任务缺目标为 `missing`，不得冒充不适用或零。单测同旋转0°、已知旋转角、四元数符号等价、剪裁数值稳定、目标固定、完成mask/帧数和左右手映射。

### Smoke 和正式启动门槛

每个开关交独立单元测试及 CPU 结果；然后每条受影响训练路径做小规模真实仿真2轮更新、保存、加载、再1轮恢复（独立新目录），并做 actor-mean 短评测。低环境数只用于 smoke，正式配置必须回到主结果；显存 smoke 再验证正式 env 数可用。核对有限 loss/梯度、158输入/29动作、正确 critic shape、reference/URDF/质量/奖励、初始与更新后参数 hash、恢复一致性。**进程退出0不充分：**还须 status passed、实际checkpoint可加载、预期日志和无Traceback。通过项立即开训；未通过只停相关 run，不绕过失败。

## 4. 第二批：27 个正式训练 run（最多30）

以下为独立 run，seed=721/722/723；每组都从对应主结果的初始化规则开始，不从 Full 协作模型续训。Pull 除 From-scratch 外加载同一个已验证单机器人先验；Chair/Bucket 沿主结果 fresh 初始化。每个 seed 的参考/参数除该变体外与 Full 一致。

| run_id | 任务/变体 | seed | 预算 | 依赖 |
|---|---|---:|---|---|
| V3-01 | Chair one-sided-scale | 721 | 12000 | U3a |
| V3-02 | Chair one-sided-scale | 722 | 12000 | U3a |
| V3-03 | Chair one-sided-scale | 723 | 12000 | U3a |
| V3-04 | Chair source-size-reopt | 721 | 12000 | U3b |
| V3-05 | Chair source-size-reopt | 722 | 12000 | U3b |
| V3-06 | Chair source-size-reopt | 723 | 12000 | U3b |
| V3-07 | Chair no-wrist | 721 | 12000 | U3c |
| V3-08 | Chair no-wrist | 722 | 12000 | U3c |
| V3-09 | Chair no-wrist | 723 | 12000 | U3c |
| V3-10 | Chair zero-teammate-obs | 721 | 12000 | U4a |
| V3-11 | Chair zero-teammate-obs | 722 | 12000 | U4a |
| V3-12 | Chair zero-teammate-obs | 723 | 12000 | U4a |
| V3-13 | Chair local-critic | 721 | 12000 | U4b |
| V3-14 | Chair local-critic | 722 | 12000 | U4b |
| V3-15 | Chair local-critic | 723 | 12000 | U4b |
| V3-16 | Bucket no-wrist | 721 | 12000 | U3c |
| V3-17 | Bucket no-wrist | 722 | 12000 | U3c |
| V3-18 | Bucket no-wrist | 723 | 12000 | U3c |
| V3-19 | Pull zero-teammate-obs | 721 | 50+8000 | U4a |
| V3-20 | Pull zero-teammate-obs | 722 | 50+8000 | U4a |
| V3-21 | Pull zero-teammate-obs | 723 | 50+8000 | U4a |
| V3-22 | Pull local-critic | 721 | 50+8000 | U4b |
| V3-23 | Pull local-critic | 722 | 50+8000 | U4b |
| V3-24 | Pull local-critic | 723 | 50+8000 | U4b |
| V3-25 | Pull from-scratch | 721 | 50+8000 | U4c |
| V3-26 | Pull from-scratch | 722 | 50+8000 | U4c |
| V3-27 | Pull from-scratch | 723 | 50+8000 | U4c |
| V3-28（条件） | Pull parity-plus | 721 | 50+8000 | U2 RMS≥10mm |
| V3-29（条件） | Pull parity-plus | 722 | 50+8000 | U2 RMS≥10mm |
| V3-30（条件） | Pull parity-plus | 723 | 50+8000 | U2 RMS≥10mm |

先核查真实 CLI 再生成 `launch_commands.sh`，不把本文件中的规格旗标直接传给尚未实现的旧程序。每次正式启动保存完整命令、环境、PID/job ID/GPU、起始时间、配置hash、预期总迭代、warm-up边界、checkpoint频率、日志路径与 ETA 到 `launch_receipts/<run_id>.json`。每个恢复段保留独立起止、父checkpoint hash；旧入口 `--iterations` 可能表示追加轮数，不能在恢复时再加完整12000。所有正式初始化排除 smoke权重。

## 5. 评测队列、较难档与配对统计

统一 final checkpoint、actor均值动作、无 observation noise、同任务同档50状态；不用最好 checkpoint，不把 nominal 混进50回合。失败保留完成标记、终止原因、有效步数和前缀轨迹，错误/缺文件不是失败回合0。

| 队列 | 数量（每单元50回合） | 档位/说明 |
|---|---:|---|
| Copied single robot | 3任务×3seed=9 | ±2；Push/Pull/Kick |
| Frozen one robot | 6任务×2被冻结角色×3seed=36 | ±2；冻结0和1分别交 |
| Full 扰动扫描 | 6任务×3seed×3新增档=54 | ±4/6/8；已有±2主结果一并验证 |
| 新模型 | 27×2=54；条件成立30×2=60 | 每模型±2与hard各一次；hard=±2时复用同次结果，不虚增独立样本 |

Full 总计18个主模型；六任务全部用最新正式配置（Push registered、Small Table T-07）。需要从原轨迹补 Palm 时可离线算；无法从完整状态/目标恢复则按相同协议补评并记录版本差异。较难档的 Full 对照、冻结/复制 Full 对照都必须具备同一个 state文件与全部150配对，不能拿只存 seed721 的结果替代。

**较难档冻结规则：** 对每档 d∈{2,4,6,8}，先算每任务三个seed完成率的均值，再六任务等权平均；选择该平均仍 ≥50% 的最大档。只读取 Full 数据，保存 `hard_level_selection.json`（全部输入CSV行/哈希、各任务/档均值、最终档、时间），之后才查看/汇总消融 hard结果。不得用消融数据选档，不因曲线不单调改变“最大满足档”的规则。若所有档包括±2都<50%，STOP hard列并报告，不能擅选±2或删hard列；其他评测继续。

汇总先每seed算，再三seed `mean ± sample std(ddof=1)`；完成率分母固定50，三seed共150。误差指标仅在完成回合聚合，保留完成数、帧数及加权方式，与现有表定义相同，不能混失败前缀使其显得更好。Frozen柱先在每seed内对冻结0/1两条件均值，再对三seed算均值和sample std；同时保留两角色原始分组，不能先在6个数上求std伪装seed差异。

配对键：`task,perturbation_level_cm,training_seed,state_id`，双方state文件hash必须一致且无重复/缺失。完成率 McNemar **精确双侧**，每比较150对，输出 `both_complete,only_full_complete,only_variant_complete,both_fail,n_pairs,p_exact`；discordant=0时p=1。†阈值p<0.01，不新增任意统计选择。误差只对双方均完成回合逐对比较，交 paired明细、每seed有效对数/差值与统计定义；无共同完成不能填0。旧 Push配对已知50/67/22/11，p≈1.9e−6仅供追溯校验，不能代替本轮重新匹配的来源。

## 6. U5 图用素材完整清单

全部PNG **1920×1080、无文字/标签/箭头/曲线烘焙**，每个before/after同相机/投影/裁切/光照/材质。二维叠加点、面片、轨迹另外JSON。可复用已批准原图，但必须核hash/尺寸/语义/选帧规则；旧 nominal 展示不能自动用于本轮“首个完成扰动回合”。

| asset_id | 内容与规则 | 交付 |
|---|---|---|
| F2a_push_prescribed / registered | Push prescribed0.80m与role isometry；同帧同相机 | 2图 + 与 `R4_push_0126_pair.json` 同格式的锚点面片、目标锚点、真实hand FK像素JSON |
| F2a_push_top_prescribed / registered | 同一比较俯视放大，能分辨39mm偏移 | 2图+独立camera/overlay JSON；数值从CSV追溯 |
| F2b_pull_plus / minus | 仅U2≥10mm时；η=+1/−1同帧相机，不能用旧“两种角色”画面冒充奇偶性消融 | 2图+同格式面片/锚点/FK JSON |
| F2c_chair_one_sided_ghost / shared | 上图U3a单侧尺度物体实体，叠加参与者2 nominal缩放物体的半透明虚影；下图一份mean共享物体 | 上下两图均用C-01优化后姿态重新渲染，1920×1080；保留旧R6视角、frame117对应源时刻（30Hz）与光照，记录新投影；旧图仅作排版占位 |
| F2d_chair_before / after | Chair wrist近景，同模型与物体 | 已有 `R5_chair_122_before/after`，核验复用；源before必须对应U3说明 |
| F3_human_{push,pull,kick,chair,bucket,table} | 六任务各一帧真实示范，风格同R1；Mac现有R1中Push/Pull/Chair/Bucket/Small Table均找到素材，Kick缺示范；逐图核验尺寸/来源后复用合格图，其余补渲 | 6图；人体模型/参数/faces/对象/时轴来源 |
| F3_exec_{task}_{000,025,050,075,100} | **seed721，按state_id顺序第一个完成的±2扰动回合**，取完整reference horizon的0/25/50/75/100% | 六任务各5图=30图；固定每任务相机，保存选择证明和帧号 |
| F3_traj_{task} | 同一回合reference与execution物体轨迹投影到末帧相机 | 六JSON，点序/时轴/坐标系/裁切变换/可见性；图内不画轨迹 |
| F4_training_curves | 所有本轮训练+主结果训练的TensorBoard scalars | CSV：iteration、mean episode length、mean reward、horizon、run/seed/阶段/来源tag；不抹掉warm-up或续训段 |

**Fig.2(c)作者已批准的画法（2026-11-02）：** 上图在 U3a 资产生成后渲染。参与者1的缩放物体（s₁、p_nominal_person1(t)、原旋转）为不透明实体，即单侧尺度训练/仿真实际采用的那个物体；参与者2的 nominal 缩放物体（s₂、p_nominal_person2(t)、原旋转）只画成半透明虚影，说明机器人2原先的重定向参照。虚影仅为渲染辅助，不加入训练URDF/仿真碰撞、质量或动力学，也不重定向机器人。机器人动作及腕角保持 U3a/C-01 规定。下图仍为一份平均尺度共享物体，不加第二份实体。

Ubuntu已报告旧 `shared_mean.png` 为腕部优化前姿态、1920×1280，与本轮C-01优化后构型、1920×1080规范不同。**上下两图都重新渲染，不再复用旧下图为正式素材。** 同一时刻两台机器人的完整构型（含左右腕角）在上下图逐值相同，均来自C-01正式优化后参考；只改变物体尺度/位置方案和已说明的nominal虚影。上图用U3a的s₁实体加s₂虚影；下图用平均尺度单一共享物体。不得为配合旧图将上图退回优化前姿态。

沿用R6原相机的观察位置/朝向、帧117对应的源时刻、光照与材料。两张新图统一1920×1080、相同投影及裁切；记录实际新相机内外参、画幅/FOV或正交尺度、裁切变换与源hash。画幅由1920×1280变为1920×1080时不能沿用旧像素投影JSON，也不能把旧图拉伸成新图。读取实际R6 manifest核对frame117/30Hz与runtime50Hz的时间映射，不能把117直接套到50Hz数据；选用该对应时刻的C-01构型，并记录源/目标帧映射。原 `individual_nominal.png` 和 `shared_mean.png` 都只作为Mac临时排版占位。若出现本次已说明差异之外的源资产/时间/构型冲突，再按通用STOP规则报告；本次两图重渲不影响任何训练run。

底部指标保持 `max size mismatch 4.81 → 2.41 %`，数值从对应 CSV 核准：one-sided 的 maxᵢ|s₁/sᵢ−1|×100%，mean 的 maxᵢ|s̄/sᵢ−1|×100%。前者为参与者1实体相对参与者2 nominal 尺度的差异；不是把虚影当作第二个实际物体。交 manifest 写明实体/虚影资产身份、各 scale/pose、透明度、物理开关、来源hash及CSV定位。若复算不符，只停该数字并报告，不估数。这个图修改不增加训练run、不阻塞U3a开训。

人体示范使用真实SMPL-X/原始mesh，不能拿机器人nominal或骨架替代。OMOMO缺失手姿态若采用官方default，manifest标记，不能声称真实抓握细节。六任务 execution 如seed721无完成扰动回合，STOP该行素材并报告，不能换seed、选nominal或按RMSE挑最好回合。统一选帧索引 `frame_index=round_half_up(frac*(horizon_frames-1))`，frac为0/0.25/0.5/0.75/1；保存horizon_frames定义与状态/phase映射。若实际采集从frame1开始或terminal状态另存而无法遵循该定义，先报告具体差异，不能近似偷换旧帧。视频25fps不能当状态50Hz；最后一帧取合法horizon终点，不越界，也不把提前终止末帧当100%。末帧轨迹投影JSON必须绑定 `image_sha256` 与该图同一个camera ID。旧素材若1920×1280仅可作Mac预览；正式U5仍需1920×1080并保持正确投影。Mac实查旧执行Push为perturbed_000但按八等分导出，Pull/Chair/Bucket为nominal，不能直接复用为本轮五帧规范。

每图 manifest至少：`asset_id,kind,task,configuration,training_seed,state_id,selection_rule,state_index,reference_frame,source_frame,fps,time_s,input_paths,input_sha256,checkpoint_sha256,robot_urdf_sha256,object_urdf_sha256,camera_intrinsics,camera_extrinsics,projection,image_size,crop_transform,light_material_spec,render_command,code_sha256,image_sha256`。相机复用给明确ID，不用“same camera”空文字。缺失TensorBoard时可交原metrics.jsonl与逐字段等价导出证明；缺真实mean episode length不以其他累计值替代。

**曲线日志预检须在新run开训前完成。** 当前GitHub Plan5训练入口写metrics.jsonl的reward_mean、reset_count、timeout_count等，未见mean episode length/TensorBoard写入；它们不能自动恢复真实mean episode length。先查Ubuntu实际版本及Full旧run是否另有真实episode-length记录。新run补无行为影响的真实episode length记录，统一单位、完成/失败回合纳入范围、滚动窗口、reset随机起点/参考后缀语义，并与Full日志定义一致；保留原始每回合记录以复算。若Full缺相应历史记录，STOP受影响Full曲线并报告，不能拿1/reset_count、存活env计数、mean reward或重跑的新Full训练替代旧曲线，不能为图自行扩张27/30run预算。

## 7. 交付目录与数据契约

```
V3_20261102/
  README_zh.md
  dispatch_receipt.json
  run_plan.csv
  launch_commands.sh
  launch_receipts/  exit_records/  stops/
  U1/ U2/ U3_assets/ U4_tests/ U5_media/
  states/                       # 六任务各档固定状态与生成器快照
  models/                       # 路径/hash清单；checkpoint可独立打包
  results/episodes.csv
  results/per_seed_summary.csv
  results/group_summary.csv
  results/paired_completion.csv
  results/paired_errors.csv
  results/hard_level_selection.json
  curves/training_scalars.csv
  sources/manifest.json  sources/code/  sources/configs/
  SHA256SUMS
```

`episodes.csv` 每回合一行，至少：

```
experiment_group,task,variant,run_id,training_seed,model_id,
checkpoint_iteration,checkpoint_sha256,perturbation_level_cm,
perturbation_level_deg,state_id,state_file_sha256,is_nominal,
completed_reference,termination_reason,executed_steps,reference_horizon_steps,
object_pos_rmse_cm,hand_pos_error_cm,palm_error_deg,
n_metric_frames,palm_valid_hand_frames,palm_status,
trajectory_path,trajectory_sha256,config_sha256,reference_sha256,
object_urdf_sha256,evaluation_code_sha256,exit_record_path
```

保留现有原字段/指标，不因采用上述规范丢原信息；将 Hand 原公式和聚合写进data_dictionary，若原指标不是mean error则沿用其正确名称/定义，不改数据语义。空值必须有 reason/status；不写NaN当实数，不适用、未完成、程序错误分开。原始轨迹至少能复算Object/Hand/Palm、完成与选帧。

`per_seed_summary.csv`：同组/任务/变体/档/seed，`n_episodes=50,n_completed,completion_pct`、误差均值及各有效样本数。`group_summary.csv`：三seed均值、sample std、seed数量、完成数、来源行键/CSVhash；不把150回合SE当3seed std。配对CSV另交双方episode行键与原始完成指示；所有论文数字能找到具体CSV行。

每正式run交真实启动/退出记录、完整run_config、训练日志/metrics/TensorBoard、最终checkpoint和hash、全部恢复段。退出码由真实监督进程记录；取不到写unknown，不能用已消失systemd服务默认值0。交付复核包括行数/唯一键/50状态×3seed、各组完整性、sha256、最终迭代、指标有限性、参数单因素diff、测试日志。

**分批回传：** 首先 dispatch/preflight/STOP/ETA；随后U1/U2、U5与评测；训练完成一个完整三seed组可先回传但必须标其余未完成，11/3中午交总包。任何异常即时回执，不等最终截止。Ubuntu不得直接改论文结论、图编号或主文/补充去留规则。
