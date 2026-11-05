# Demo、腕部对照与待办台账

核对更新：2026-09-25。本文是 `MULTI_AGENT_EMERGENCE_ROADMAP.md` 的证据台账，
不另立研究路线。训练评测结果依据截至09-20的归档；09-25完成身份统一并继续离线方法核查，
含NPZ逐值/哈希检查和已有viewer的validate-only，不重求解、不做动力学步进、不训练或加载策略模型，
资料先在本地完成，随后用户授权本轮同步到个人GitHub的 `core4d-base`；原始训练数据、模型和完整日志不随本轮提交。

**当前阶段：实验结果收束与论文材料整理。** 桶5组、小桌7组、椅子1组共13组正式训练均已完成12000轮，
各组最终模型、run_config、status和评测summary在本地；实验身份与入口统一见第9节。
小桌原20kg五组＋5kg两组，共84000轮；导入副本、smoke、同模型复测不重复计组。

| 系列 | 最终状态 | 当前用途 |
|---|---|---|
| 桶交接 | A/B/去关系/去独立高度/双删除：2/5、1/5、4/5、0/5、5/5完整 | 双删除为用户首选展示；其他组保留奖励对照与辅助接触素材 |
| 小桌5kg | 完整A 5/5、双删除3/5；完整回合位置RMSE 5.57/10.25cm，均未持续整桌离地>1cm | 已收件/校验/评测；回放待用户复核，不追加训练 |
| 椅子 | 基础11项、5kg，固定场景9/9完整；位置RMSE约3.42cm | 已认可的动作展示；参与不均作行为分析 |
| OMOMO及其他Demo | Push/Pull/Kick已有认可回放；Demo3/4保留诊断结果 | 完整分类见第2节，不混用奖励或模型 |

**奖励结论**：扶椅子是基础11项；桶A/B并非同一配置；桶双删除仍保留位置项wz2，
不是“不管高度”。桶结果不支持独立高度普遍必要或新增关系必然有效；小桌5kg完整A的跟踪反而更好，
应按任务和组合讨论。公式见第3节，桶五组完整对照见第6.2节。

身份统一及本轮方法对照核查已完成，见第4.1节；椅子误差图已生成。
8组训练回放素材与Mac交接已整理，见[交接入口](docs/writing_handoff/README.md)及[素材清单](docs/writing_handoff/media/README.md)。
腕部前后同相机图、共同几何质量指标和定量行为时序仍可后续补充。
OMOMO单人奖励验证尚未配置/训练，需另行确认。两次小桌有限复试额度已用完。
09-22写作是历史目标，不当作未来排期；本工作区提供实验材料，用户在别处撰写正文。

“用户认可”“固定场景到达参考末尾”“跨条件稳健”分别记录；评测回合不是独立训练种子。
后续带日期的启动/收件段落是历史过程，不表示当前任务仍运行。旧Viser端口是回放索引，
不保证09-25仍在线；本次未连接或停止云端实例。

结果入口：[桶最终对照](logs/Core4DBucket/ablation_comparison_20260920/RESULTS_CN.md)、
[5kg小桌结果](logs/Core4DSmallTableA/comparison_20260920/RESULTS_CN.md)、
[GitHub可读小桌快照](docs/experiments/core4d_smalltable_5kg_20260920.md)。
`logs/`为本地证据，模型/完整日志/回放不随文档提交上传。

## 1. 已对齐的定位

- 论文主线仍是老师确认的人类交互示范迁移、重定向与参考引导的多机器人协作学习。
  固定橡胶手不是替代该主线的新主题。
- 固定手型理由1：部分以推压、支撑为主的 loco-manipulation 不必依赖主动手指；
  选配灵巧手会增加硬件与控制复杂度。未测量的成本、可靠性与训练效率不写成实验证实的优势。
- 理由2：不同来源的手部信息质量不一，尽量减少对高质量逐指示范的依赖。
  CORE4D 实际有动态手指参数，不能声称所有数据都没有手指；当前掌面方法仍需腕姿态及标定信息。
- 技术缺口：稀疏关键点位置和交互关系不一定充分约束末端接触面朝向。
  官方 OmniRetarget G1 基线优化腕关节并考虑手部碰撞，但未显式匹配解剖掌面方向。
  不泛化成“所有主流重定向都不支持朝向”，也不将所有质量问题归因于手部表示。
- 掌面映射与腕部优化属于方法改进/候选贡献；是否改善物理执行和下游训练仍需对照。
  人体逐指抓握到固定手型的可实现性限制单独讨论，不能仅由训练失败断言物理不可能。

## 2. 已有多机器人 Demo：包括未成功的结果

| Demo | 来源与状态 | 奖励分类/额外设计 |
|---|---|---|
| Push | OMOMO 单人 A1 构造双机器人参考、对应 WBT 初始化；已有用户认可回放。当前 live-PD 链训练到15050，13050复测1/9完成、14050与15050为0/9；较好展示不等于稳健成功。用户决定未来可续训，但暂不启动。 | 基础11项；旧平滑实验另列，实时PD刷新修复不是奖励项。 |
| Pull | 单人拉动参考镜像组成双机器人参考、Pull WBT初始化；08050有完整回放，用户认可。尚无跨随机场景稳健性结论。 | 基础11项；镜像是参考构造，不是奖励修改。 |
| Kick | 单人踢动参考镜像组成双机器人参考、Kick WBT初始化；8000轮及完整回放，用户认可，未要求严格复现脚尖发力。 | 基础11项。 |
| CORE4D 小桌 | 原始双人 `20231030/001`、desk001，fresh12000；能够移动物体，但未复现双手抬起搬运，参与不均衡。各奖励变体见下表。 | 基础11项，以及独立的交互图/高度消融。 |
| CORE4D 椅子 | 原始双人 `20231020/074`、chair021，A1腕优化参考，5kg，fresh12000；用户认可语义并保留，但一人主要操作、另一人参与不足。 | 基础11项，没有额外交互图。固定配置9/9完成、位置RMSE约3.42cm；不同seed的确定性回放相同，不当作泛化验证。 |
| Demo3 对抗拉动 | Pull派生独立参考；实际训练停止于10100。最终5次评估均约6步跌倒，胜负标记不能当作有效拔河。保留collapse诊断。 | 竞争奖励，不是共同物体轨迹模仿；见第3节。 |
| Demo4 协作旋转 | Pull派生机器人先验；10000轮后快速达到+90°，但猛烈甩转、偏离先验。保留奖励设计失败案例，不列为动作质量合格Demo。 | yaw进度与首次达标奖励；未来先验约束/平滑目标重设计尚未实施。 |
| CORE4D 桶交接 A/B | `20231020/071`、bucket003，A1 wrist-only参考、原尺寸1kg，两组fresh12000；用户已认可两组为Demo并观察到额外支撑。共同本地评测A为2/5、B为1/5到达末尾；不能据选优视频宣称稳定交接或B整体更优。 | 原身体跟踪/正则 + 竖直误差加权的物体位置/朝向、独立正高度和距离加权相对向量；B另乘软接触因子。不同于小桌旧Laplacian；非attention/信用分配。 |

### 2.1 小桌历史五组20kg实验（另两组5kg已完成，见6.5/6.6）

不计smoke、重复评测、导入副本，实际为**5组×12000轮，共60000 iterations**。
加上09-20完成的两组5kg，目前小桌合计7组正式训练、84000 iterations；新增两组已完成本地评测。
共同参考为`20231030/001 / desk001`、共享缩小物体的Stage1 nominal；687状态、50Hz。
五组均使用**20kg桌子**（仿真读回已核对）、静/动摩擦0.5/0.5、恢复系数0；几何缩小没有
同时降低质量。均fresh seed721、2048env×24步、每2000保存、Actor158/Critic527、物理200Hz。
四组本地、一组云端。完整参考执行不等于持续双手搬运。

| 配置 | 状态 | 当前证据 |
|---|---|---|
| 无图、wz=1 | 本地12000完成 | 固定场景5/5完成，物体3D RMSE均值9.100cm；搬运语义未解决。 |
| 交互图、wz=1 | 本地12000完成 | 2/5完成；用户认为交互更明显，仍主要用腿。代表成功回放关系RMS约5.481→4.021cm，不代表全回合/跨seed稳定改善。 |
| 无图、wz=2 | 本地12000及评估完成，用户已看并认为变化不明显 | 5/5完成，3D RMSE均值7.576cm；代表回放z RMSE3.532cm，旧3.552cm；整桌最低点高于地面1cm比例均0%。 |
| 交互图、wz=2 | 已接收最终12000模型，实际平台为RunPod RTX4090；本地评测退出码0 | 固定场景5/5完成，3D RMSE均值9.686cm；选优完整回放z RMSE3.239cm，整桌离地>1cm比例1.89%（13/687，三个短段），仍未复现持续搬运。 |
| 交互图、wz=2、独立高度误差惩罚 | 本地12000轮及新旧各5回合评测完成，三个真实进程退出码均0 | 新4/5完成、旧5/5；完整回合原点高度RMSE均值1.512cm（旧3.259cm），但整桌最低点>1cm仅0.109%（旧1.921%），最长单段0.04s（旧0.10s；参考5.32s）。原点升高可由倾斜产生，未复现持续搬运，不证明手部承重或分工改善。 |

前四组及新增高度组是探索性对照：一个训练seed，云端与本地训练硬件不同，历史评估与源码版本需保留。
不能直接比较不同定义的总reward来证明提升。

**历史五组未覆盖什么**：截至这五组完成，小桌未训练桶式正高度`+exp(-ez²/0.10²)`或桶式新相对向量；
09-19启动的第六组5kg组合同时采用二者，第七组在相同5kg条件下删除二者，均已完成并收件，见6.5/6.6。
旧独立高度为`-(ez/0.05)²`，旧图为Laplacian（权重1、尺度6cm）。桶是1kg、新向量权重2/
尺度4cm，并有独立正高度。不能将两种物体的表现差别归给某个奖励因素。

**对下一步最有用的机制**：小桌参考整桌最高离地约6.42cm，旧高度惩罚确实改善了物体原点z，
但倾斜也能让原点升高。报告中的episode4/frame283原点上升5.547cm、桌子偏离参考姿态
11.499°，最低碰撞点仍接近地面。改成正指数奖励仍追踪原点，并不自动消除这个替代方式。
09-19用户曾选择先做5kg完整桶A配方的组合试验，再比较两项均删除；这两次有限复试现已完成，
结果见6.5/6.6，不再追加训练，也不以修好这条数据为论文前提。组合试验不是历史单因素消融的延续。

高度新旧对照报告与所有回合：
`logs/Core4DSmallTable/interaction_mesh_z2_height1_scale005_fresh12000_save2000_actor158_seed721_env2048_20260916/evaluation_20260917/RESULTS_CN.md`。
本次新4个完整回合、旧5个完整回合分别求均值，未把失败前缀混作全片。新失败为桌子位置误差
越过原25cm阈值，主要是水平偏离；训练95.243%是随机参考后缀完成率，不是从头抬升成功率。

## 3. 哪些奖励相同，哪些有额外设计

### 3.1 共同底座：基础11项

Push最终live-PD链、Pull、Kick、小桌baseline、椅子baseline使用同11项及同权重/sigma：

| 奖励项 | 权重 | sigma/说明 |
|---|---:|---|
| 参考锚点位置/朝向 | 0.5 / 0.5 | 0.3 / 0.4 |
| 身体相对位置/朝向 | 1 / 1 | 0.3 / 0.4 |
| 身体线速度/角速度 | 1 / 1 | 1 / 3.14 |
| action rate | -0.1 | 相邻动作变化 |
| joint limit | -10 | soft limit比例0.9 |
| undesired contact | -0.1 | 按原配置指定部位，不是新增hand-only限制 |
| 物体位置/朝向 | 1 / 1 | 0.3 / 0.4 |

机器人奖励按两人取均值，物体项仅计算一次，形成共享team reward。上述为乘控制dt前的权重；
这些MARL配置的RewardManager统一乘dt=0.02。基础奖励已包含身体动作跟踪和动作变化正则，
“没有新增交互项”不等于“只奖励物体移动”。原不期望接触项也不是新增的手专用接触规则。
同奖励不等于同协议：前三个Demo采用WBT Actor初始化，CORE4D从零开始；物体、参考、
质量、训练预算和学习率等另有差异，不能把跨Demo差异直接归因于奖励。

### 3.2 已训练的交互/高度变体矩阵

以下各行都保留上面的六项身体跟踪及三项正则，物体朝向权重均为1、sigma均为0.4。
表中的“无”是没有**独立新增项**，不表示完全不约束高度/相对身体运动。

| 实际训练版本 | 物体位置中wz | 旧Laplacian图 | 新相对向量 | 独立高度项 | 软接触调制 |
|---|---:|---|---|---|---|
| Push live-PD / Pull / Kick / 扶椅子 / 小桌baseline | 1 | 无 | 无 | 无 | 无 |
| 小桌interaction、wz1 | 1 | 权重1，sigma=0.06m | 无 | 无 | 无 |
| 小桌baseline、wz2 | 2 | 无 | 无 | 无 | 无 |
| 小桌interaction、wz2 | 2 | 权重1，sigma=0.06m | 无 | 无 | 无 |
| 小桌interaction、wz2、高度惩罚 | 2 | 权重1，sigma=0.06m | 无 | −(ez/0.05m)² | 无 |
| 桶A | 2 | 无 | 权重2，sigma=0.04m | +exp(−ez²/0.10²)，权重1 | 无 |
| 桶B | 2 | 无 | 与A相同 | 与A相同 | 对物体位置、朝向、高度、关系整个正奖励块乘g |
| 桶去关系 | 2 | 无 | 无 | 与A相同 | 无 |
| 桶去独立高度 | 2 | 无 | 与A相同 | 无 | 无 |
| 桶双删除 | 2 | 无 | 无 | 无 | 无 |
| 小桌5kg完整A | 2 | 无 | 权重2，sigma=0.04m；小桌自身点集 | 与A相同 | 无 |
| 小桌5kg双删除 | 2 | 无 | 无 | 无 | 无 |

物体位置统一写作 `rp(wz)=exp(-(ex²+ey²+wz*ez²)/0.3²)`。
因此扶椅子与桶A至少有三处奖励差别：wz从1变2、增加独立高度、增加相对向量；
还同时换了参考、物体、质量（椅子5kg/桶1kg）等。不能用两段视频证明某一项造成了改善。

桶A/B展开为：

```text
R0(wz) = Rbody + Preg + rp(wz) + rrot
rz     = exp(-ez²/0.10²)
rrel   = exp(-mean_agent(Erel)/0.04²)
RA     = R0(2) + rz + 2*rrel
RB     = Rbody + Preg + g*(rp(2) + rrot + rz + 2*rrel)
```

`Erel`比较对应身体—物体点的world相对向量；参考/实际距离权重分别归一化再平均，
按人独立计算。每人19点、桶89个固定表面点（预算100），每手三点的基础权重各1/3。
这与小桌在物体坐标系下、参考冻结的Laplacian二次型不同，不能统称为“同一个交互图”。
`g=0.5+0.5*exp(-2*Ec)`；Ec只度量示范要求时段内缺失的手—桶接触，不惩罚额外接触。
完整接触公式及几何软标签局限见[桶A/B说明](CORE4D_BUCKET_AB_TRAINING_CN.md)。
所有身体/关系项都是奖励约束，不是把参考姿态强制写入仿真，也不是Actor增加物体输入。

### 3.3 其他独立设计：不要混入共同底座

- 小桌interaction：在11项上加 `interaction_mesh`，权重1、sigma=0.06m、不退火。
  它是几何关系奖励，不是attention、信用分配网络、触觉标签或承重测量。
- 小桌wz=2：物体位置项变为 `exp(-(dx²+dy²+2dz²)/0.3²)`；仍是原来的一个奖励项。
- Demo3：保留6项机器人跟踪、action-rate及joint-limit；去掉undesired-contact和两项物体
  位姿跟踪，增加双方相反拉轴方向速度奖励（权重10），使用各自reward/value/GAE。
- Demo4：同样保留上述8项，增加团队yaw进度（10）与首次达标奖励（5），没有物体轨迹跟踪。
- Push历史smooth实验：确曾加入joint acceleration代价，目标系数-5e-9、500轮ramp；
  当前live-PD续训run的 `joint_acceleration_schedule=null`，不能把旧项算入当前模型。
- 双机器人每个物理子步读取实际关节状态的PD刷新修复属于控制实现，不是reward trick。
  不把旧single-agent路线的退火讨论写成这些有效MARL基线已经启用的机制。

配置依据：
[共同奖励](src/holosoma/holosoma/config_values/marl/g1/reward.py)、
[小桌奖励](src/holosoma/holosoma/config_values/marl/g1/core4d_smalltable_reward.py)、
[Demo3](src/holosoma/holosoma/config_values/marl/g1/demo3_reward.py)、
[Demo4](src/holosoma/holosoma/config_values/marl/g1/demo4_reward.py)。

### 3.4 查证入口：配置、算法和实际训练身份

- 扶椅子复用[小桌基础奖励配置](src/holosoma/holosoma/config_values/marl/g1/core4d_smalltable_reward.py)，
  不是另造一套椅子奖励；[实际run_config](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/run_config.json)
  记录`reward_variant=baseline`、`interaction_reference=null`及11项名称。
- 小桌五组实际配置依次是：[baseline/wz1](logs/Core4DSmallTable/paired_reference_fresh12000_save2000_actor158_seed721_env2048/run_config.json)、
  [旧图/wz1](logs/Core4DSmallTable/interaction_mesh_fresh12000_save2000_actor158_seed721_env2048/run_config.json)、
  [baseline/wz2](logs/Core4DSmallTable/baseline_z2_fresh12000_save2000_actor158_seed721_env2048_20260915/run_config.json)、
  [旧图/wz2](logs/Core4DSmallTable/interaction_mesh_z2_fresh12000_save2000_actor158_seed721_env2048_cloud_20260915/run_config.json)、
  [旧图/wz2/高度惩罚](logs/Core4DSmallTable/interaction_mesh_z2_height1_scale005_fresh12000_save2000_actor158_seed721_env2048_20260916/run_config.json)。
- 桶[实际A配置](logs/Core4DBucket/A_fresh12000_env2048_seed721/run_config.json)、
  [实际B配置](logs/Core4DBucket/B_fresh12000_env2048_seed721_cloud_20260918/run_config.json)与
  [冻结奖励契约](src/holosoma/holosoma/config_values/marl/g1/core4d_bucket_contract.py)共同确定身份。
  A/B的新增块在`reward_terms`中只有一个`bucket_interaction`，不能仅数配置项判断奖励内容。
  特别注意：桶run_config顶层旧通用字段`object_z_error_weight=1`不控制桶专用块；
  实际`bucket_reward_contract`记录`[1,1,2]`，计算实现也固定wz2。本台账记录实际生效值，
  未为统一展示而修改原日志或当前代码。
- 算法位置：[旧Laplacian](src/holosoma/holosoma/managers/reward/terms/interaction_mesh.py)、
  [通用相对向量计算](src/holosoma/holosoma/managers/reward/terms/interaction_vectors.py)、
  [桶组合/接触项](src/holosoma/holosoma/managers/reward/terms/core4d_bucket.py)、
  [桶配置接线](src/holosoma/holosoma/config_values/marl/g1/core4d_bucket_reward.py)。
- 单人OMOMO使用[WBT带物体奖励](src/holosoma/holosoma/config_values/wbt/g1/reward.py)的11项，
  不是不含物体的9项preset。单人项没有双人均值，不能直接套桶的双人环境包装器。

OMOMO双机器人历史正式产物仍在`/home/kevin/holosoma-rubber-hand-marl/logs/`，本次只读，
不搬入当前工作树、不混用checkpoint。可核对以下实际配置：

| 运行链 | 历史run_config位置（以上述logs为根） |
|---|---|
| Push原11项、至8050 | `Plan5Push/a1_officialequiv20kg_fullactor8000_from_critic50_seed721_env2048/run_config.json` |
| Push旧smooth、8050→10050 | `Plan5Push/a1_8050_jointacc_smooth2000_seed721_env2048/run_config.json` |
| Push旧smooth、10050→15000 | `Plan5Push/a1_10050_jointacc_continue_to15000_seed721_env2048/run_config.json` |
| Push live-PD、8050→15050 | `Plan5Push/a1_livepd_20kg_continue7000_from_08050_seed721_env2048/run_config.json` |
| Pull、至8050 | `Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/run_config.json` |
| Kick、至8000 | `Plan5Kick/mirrored_kick_full8000_seed721_env2048/run_config.json` |
| Demo3、6800恢复至10100 | `Demo3Tug/square_table_diagonal_tug_full15000_seed721_env2048/run_config_resume_from_06800.json` |
| Demo4、至10000 | `Demo4Rotate/rectangular_pull_pull_rotate90_full10000_save1000_seed721_env2048/run_config.json` |

旧smooth的run_config记录提交`398dc04e`，但该提交本身不含smooth新增源码，说明当时有未提交
改动；复现还须核对实跑terms/schedule和保存实现，不能只靠HEAD。smooth15000与live-PD15050
是两条不同链。Demo3目录含`full15000`不代表完成15000，实际恢复终点为10100。

每个新实验保留自己的run_config、参考/资产哈希、奖励版本、预算、checkpoint及全部评测索引；
后来的默认代码不追溯改变旧模型的训练身份。比较效果看共同物理指标，不按不同定义的总reward排名。

## 4. 同一数据的腕部前后对照

| 数据/对照 | 可复核结论 | 不能混淆的限制 |
|---|---|---|
| CORE4D chair `20231020/074`，相同小椅子与身体baseline→A1 | 最清楚的用户认可样例。仅六腕关节变化，物体轨迹及FPS相同；四手平均朝向误差约42.42/86.34/53.17/68.97°降至约0/0/0/0.236°。 | 有一只手最大残差约5.008°；用户批准6°导出阈值。未重解碰撞；朝向改善不等于接触点不动或承重已实现。 |
| OMOMO `sub6_largetable_033`，相同Stage2 baseline→旧A1 | 有严格配对文件，只有六腕关节变化，掌面朝向显著改善。用户历史讨论认为动作更像示范，允许作为训练参考。 | 手—桌穿透明显；不是物理质量全方面更好。对照视频应保留这个缺点。 |
| 同OMOMO片段，matched A1→新掌面避碰候选 | 采样帧最深穿透85.606→0.100mm。 | 新候选改14个双臂关节，朝向和连续性有代价，帧间仍可穿透；未获训练批准，不记为整体优选。 |
| CORE4D小桌 `20231030/001` | 当前训练用的认可参考是共享缩小物体的Stage1 nominal。 | 未找到当前认可版本对应的严格A1前后配对。不能将缩小物体、身体解和求解阶段改变的收益算作腕部收益。 |
| CORE4D桶 `20231020/071`，相同两阶段baseline→wrist-only A1 | 已核对严格配对，仅六腕关节变化；四手平均方向误差约45.03/62.32/44.80/49.87°降至约0。 | 几何代理穿透和部分帧间角变化增加；作为跨数据适用性材料，不将A/B训练比较说成腕部消融。 |

可复现入口：

- Chair baseline：`logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/shared_small_preview/core4d_pair_reference.npz`。
- Chair A1：同目录 `shared_small_a1_preview_tol6/core4d_pair_reference.npz`；
  [指标与认可记录](logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/shared_small_a1_preview_tol6/README.md)。
- OMOMO历史pair：`/home/kevin/holosoma-rubber-hand-largetable/data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/a1/`，
  `sub6_largetable_033_fixed_object_base.npz` 与 `sub6_largetable_033_fixed_object.npz`。
- OMOMO新三方pair：[来源与指标](logs/OMOMOPush/retarget_pt_collision_20260911/README.md)，
  使用其 `pt_palm_collision/` 内同次求解的base、matched_a1和collision文件，不能跨次baseline归因。
- 小桌当前来源：[source_manifest](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/source_manifest.json)。
- 桶pair：`logs/Core4DPreviews/bucket003_20231020_071_20260918/`下
  `two_stage/core4d_pair_reference.npz`与`a1_wrist_only/core4d_pair_reference.npz`；
  指标见`a1_wrist_only/comparison/README_CN.md`。

09-19本轮已重新CPU核对：chair235帧、桶299帧、OMOMO186帧均30Hz，配对中仅qpos列
`[26,27,28,33,34,35]`变化，物体轨迹不变。椅子是优先制作的方向改善正例；OMOMO可用
09-11同次`fixed_object_base`与`matched_a1`作配对，区别于未获认可的14关节避碰版本。

### 4.1 09-25方法对照整理结果

[专项报告](docs/experiments/retarget_method_comparison_20260925.md)、
[指标及配对哈希](docs/experiments/retarget_method_comparison_20260925.json)、
[椅子方向误差图](docs/experiments/assets/chair_wrist_orientation_20260925.svg)已经生成。
本轮重新检查三类腕部配对；椅子非腕/物体/尺度逐值相同、URDF及报告来源哈希匹配，
最大关节步长复算一致，两个viewer validate-only均退出0。椅子四手全帧平均方向误差
62.725°→0.059°；方向数值引用哈希匹配的已有FK报告，不冒称新跑动力学或训练消融。

原尺寸小桌现有单阶段140帧、两阶段413帧，共同前140帧的人体/物体目标及资产匹配；
但脚高、限位覆盖、弹性约束和nominal先验也改变，可作pipeline整体比较，不能单归因于阶段数。
140帧是命令主动截断，不是已知失败；尚缺同定义的共同前缀质量指标。
训练用小桌来自Stage1共享缩小预览，与原尺寸Stage2不同；椅子为Stage1共享小物体+A1，
桶为原尺寸两阶段+A1。没有找到相同共享小桌目标下另行求解的Omni结果，不能用原尺寸组替代。

待办：报告第6节已列同帧/同相机素材清单；实际机器人截图和视频还未制作。
桶和OMOMO的方向/几何代价分别呈现，不串用两次求解的baseline；不将现有训练成功率归因为腕部收益。

## 5. 下一批数据：现有候选与新来源分开

用户选过15条CORE4D，其中chair `20231020/074`、bin `20231020/071`已训练；
其余13条未找到正式训练记录（小桌`20231030/001`不在这15条中）：

| 浏览分类（不是动作标签） | 剩余候选 |
|---|---|
| table | 20231108/060、20231108/070、20231023/040、20231020/120、20231020/130、20231020/132 |
| long table | 20231023/050 |
| chair | 20231023/100、20231020/090、20231018/080 |
| bin | 20231020/065、20231020/064、20231018/001 |

它们都在本地，具体动作要回看原始数据，不能凭“桌子/桶”认定是推而不是抓握搬运。
2026-09-18选中的桶交接已完成参考及A/B训练：共同采用原尺寸两阶段后的wrist-only A1，
不是Stage2-only；1kg为实验质量，不是数据集实测。源预览/腕部对照及离线诊断保留在
`logs/Core4DPreviews/bucket003_20231020_071_20260918/`，正式结果见第2/3节。
不会因为新奖励效果看起来好，就把尚未训练的候选或旧Demo改称采用了桶奖励。
060及chair100已有物体矩阵异常记录，保留候选，选中后单独处理，不默默放宽检查。
新动作不自动要求先做单智能体WBT。建议优先看无需主动抓握、物体留地面的双人操作片段；
这只是选片方向，不是已经证明哪条可行。

新数据集目前仅完成调研，未下载接入或训练：

- 首选候选 [FORCE](https://github.com/xz6014/FORCE_dataset)：真实推/拉箱子、椅子，
  有人体、物体轨迹及网格，大箱标注含0/15/30kg添加配重（非总质量）。
  是单人示范，若构造双人必须标明合成来源并检查配对，不称真实双人协作；许可仍需核实。
- 备选 [OmniContact](https://huggingface.co/datasets/lightcone02/OmniContact-Dataset)：有推箱子子集，
  原始动捕/物体轨迹/mesh及G1预览；重量与抓握依赖尚未逐片核验，访问需同意数据条款。
- 当前没有选定新来源，不同时启动多个adapter。先看少量原始片段，再决定是否沿用Omni入口接入。

## 6. 接下来要解决的事情与真实进度

### 6.1 当前执行次序：统一进度 → 方法对照 → 展示与分析（09-25）

| 类别 | 工作 | 当前状态/下一步 |
|---|---|---|
| 已完成训练 | 桶A/B、去关系、去独立高度、双删除 | 五组12000及共同本地评测已完成；完整对照见6.2 |
| 已完成训练 | 小桌20kg五组、5kg完整A与双删除 | 七组均已完成；新两组收件/校验/评测完成，见6.5/6.6 |
| 已完成训练 | 椅子baseline | 12000及固定场景重复评测完成，用户认可 |
| 阶段1 已完成 | 统一路线图/story/台账；核对身份和文件入口 | 09-25本地同步；第9节列模型、配置、参考和评测 |
| 阶段2 本轮核查完成 | 腕部同数据对照；单阶段/两阶段条件审计 | 报告、JSON和椅子数值图完成，见4.1；共同几何指标/实景图待制作 |
| 阶段3 待整理 | 主展示、参考/实际交接时序、统一结果表 | 保留全部回合，代表视频单列；现有时序索引见第7节 |
| 另行确认 | OMOMO单人历史A1奖励比较 | 原WBT与加关系项只是候选；配方/物理/预算未定，不恢复被暂停的新避碰参考 |

桶两项单删除、双删除和两组5kg对照都是从零训练，不续接A或短检查模型。
原启动、安装及收件记录不重写：
[桶消融启动](logs/Core4DBucket/ablations_20260919_launch/LAUNCH_CN.md)、
[去高度收件](logs/Core4DBucket/A_no_height_fresh12000_env2048_seed721_cloud_20260919/IMPORT_CN.md)、
[桶双删除启动](logs/Core4DBucket/both_removed_20260919_launch/LAUNCH_CN.md)、
[小桌收件](logs/Core4DSmallTableA/collection_20260920/RECEIPT_CN.md)。

这一轮不追加训练/评测，不更改任何旧Demo奖励或checkpoint；如材料整理发现必须新增实验，
先给出问题、比较条件和预算，再与用户确认。不以短检查代替完整学习评价，也不凑实验次数。

### 6.2 桶的最小消融矩阵

| 版本 | R0中的wz | 独立高度权重 | 相对向量权重 | 软接触g | 与现有A相比 |
|---|---:|---:|---:|---|---|
| A（已有） | 2 | 1 | 2 | 无 | 原对照 |
| A-no-rel（评测4/5完整） | 2 | 1 | 0 | 无 | 只删关系项 |
| A-no-height（评测0/5完整） | 2 | 0 | 2 | 无 | 只删独立高度项 |
| A-no-rel-no-height（评测5/5完整） | 2 | 0 | 0 | 无 | 同时删两项，已完成并收件 |
| B（已有） | 2 | 1 | 2 | 有 | 只加软接触调制 |

两个“no”组都**不是扶椅子的R0(1)**。A-no-height也仍在物体3D位置奖励中约束z，
不是“不管物体高度”。不改成小桌的负平方高度惩罚；随后经用户批准新增双删除组，
位置wz2仍保留，不是桶纯R0(1)组。若检验整个配方相对wz1基础奖励，仍需另作对照。
两项单删除只说明各项在A配方中的条件作用；09-20双删除组完成评测，补齐两因素比较。
去独立高度但保留关系0/5，两项均去掉5/5；因此独立高度不是完成任务的普遍必要条件。
仅比较完整回合，原A/仅去关系/双删除的高度RMSE均值为2.02/1.56/3.36cm，交出者示范加权
手接触覆盖71.7%/67.7%/24.2%。任务完成、轨迹精度和交互方式需分别评判。
最新全量结果及回放：[09-20报告](logs/Core4DBucket/ablation_comparison_20260920/RESULTS_CN.md)。

两项单删除的逐回合结果与回放见
[`ablation_comparison_20260919/RESULTS_CN.md`](logs/Core4DBucket/ablation_comparison_20260919/RESULTS_CN.md)。
去关系版完整回合更多，不支持当前关系项已证明改善完成率；去高度版仍抬到接近参考峰值，
但在290–309步因跟踪偏差终止，不能表述为“抬不起来”。5个重复评测并非5个训练种子，
完整执行与语义反馈分别记录。旧单删除回放用户已复核；新双删除episode000亦获用户认可：
“这个看下来最好了，和原视频一样”。当前优先展示最终12000双删除模型，旧A/B和消融仍保留。
较低接触覆盖不等同语义变差；本数据无需新增两项也能产生认可动作，OMOMO配方仍待共同选择。

### 6.3 OMOMO单人Push的复用边界

2026-09-25状态：桶五组结果已齐，但尚未共同选择OMOMO配方。下述“只加关系”保留为候选，
不是已确定的配方或当前执行任务；结合最终对照重新说明目的与配置，再确认是否实施。

- 用户指的是历史认可的单人`sub6_largetable_033` A1，不是双机器人Push，也不是09-11
  被暂停的新14关节避碰候选；两组使用同一参考，不把换参考的效果算作奖励收益。
- 复用桶A的**相对向量计算与采样组权重原则**，首选外部权重2、sigma=0.04m作为待验证配置；
  桌子重新固定采样并保存对应关系，不能套桶89点的坐标；单人无需对两人取均值。
- 此前提出的最小候选对照是原单人WBT带物体的11项 vs 这11项+2*rrel，尚未确定。
  不自动连带复制桶的wz2、独立高度、B接触调制或1kg资产。整套A配方不是“同款关系项”。
- Actor保留原154维；不添加虚拟队友、物体观测或新的腿接触禁令。原WBT身体跟踪/正则保留，
  首轮不再叠加额外强姿态正则。单人通过是迁移证据，不保证双人的力矩、接触时序也自动成功。
- 历史A1存在明显参考手—桌穿透，关系项可能强化不可实现几何；如实记录，不因此暗换参考，
  也不预先保证加奖励一定能学会手推。若接入中发现必须改变参考/契约，先和用户讨论。
- 原单人运行曾有物体质量/摩擦随机化，不能借用双人宽桌20kg固定配置冒充原版；
  质量、摩擦、随机化、批量、预算等在训练前单独冻结。旧checkpoint只作历史参照，
  不冒充同协议新训练的对照。当前未改奖励代码、未生成这两组正式配置。

### 6.4 评测、论文材料和暂不扩展的范围

- 对桶用同协议全部回合比较：执行长度、物体轨迹误差、持续离地、交接时序、双方实际接触、
  身体动作和额外支撑；对Push记录手/腿交互与物体移动，不只看总reward或最佳视频。
  失败前缀与完整回合分开。单训练seed和跨GPU的局限保留；必要重复种子后续确认，不自动开跑。
- 额外支撑不默认惩罚：先看是否维持物体并帮助交接。已有法向力不能单独量化承重/因果贡献；
  不强制50:50、不把接触覆盖提升等同信用分配已解决。椅子缺少接触力记录，暂作现象分析。
- 小桌七组全部保留，两次5kg有限复试已完成，不再自动追加；椅子/Push/Pull/Kick不换奖励、不覆写模型；Demo3/4诊断记录保留。
  FORCE、更多CORE4D片段、人—人图、attention、信用分配新网络暂不同时展开。
- 论文仍按老师主线；动作语义优先，分工能解释则分析。两阶段/掌面模块作为支撑贡献；
  整理同一数据腕部前后对照与统一视频，不把缩物体收益算成腕部收益。
  09-25先统一进度，再整理方法对照、展示与行为分析；旧09-22日期仅保留为历史目标。
  不把小桌成功或彻底消除参与不均当作交付前提，不再按旧5–6次容量自动安排新实验。
  当前顺序与授权范围以路线图第1.1节为准；本次只同步本地文档。

结果入口：
[椅子复核](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/evaluation/TRAINING_REVIEW.md)、
[小桌交互版](logs/Core4DSmallTable/interaction_mesh_fresh12000_save2000_actor158_seed721_env2048/evaluation/RESULTS_20260908.md)、
[本地高度消融](logs/Core4DSmallTable/baseline_z2_fresh12000_save2000_actor158_seed721_env2048_20260915/evaluation_20260916/RESULTS_CN.md)、
[云端组合实验与四组对照](logs/Core4DSmallTable/cloud_interaction_z2_received_20260916/evaluation_20260916/RESULTS_CN.md)。

云端交付已按本地实验布局整理，后续优先使用
[标准实验目录说明](logs/Core4DSmallTable/interaction_mesh_z2_fresh12000_save2000_actor158_seed721_env2048_cloud_20260915/README_CN.md)。
原收件夹和ZIP保留未删；历史JSON/NPZ中的原路径保持溯源，不能把云端记录改称本地训练。

### 6.5 小桌5kg完整A配方：云端完成并本地归档（2026-09-20）

用户已选择质量5kg、优先一次完整配方，而非先做无图/高度两组对照，并要求检查云端部署。
旧5组均是已完成的20kg实验；本组单列为第6个配置，已完成12000轮及5回合本地评测。

| 项目 | 本轮约定/状态 |
|---|---|
| 数据 | 沿用`20231030/001 / desk001`的现有687状态、50Hz参考，不重新retarget |
| 物理 | 5kg、原几何/碰撞/质心，惯量按旧20kg乘0.25；摩擦0.5/0.5、恢复0 |
| 奖励 | 原身体跟踪及正则；物体位置/朝向/独立正高度/相对关系权重1/1/1/2 |
| 奖励尺度 | 位置0.3m且xyz权重1/1/2，朝向0.4rad，高度0.10m，关系0.04m |
| 不启用 | B接触调制、旧Laplacian、旧负平方高度惩罚 |
| 关系数据 | 已按小桌生成每人19点、物体85点（预算100）；不复制桶点或源接触标签 |
| 策略 | 共享Actor158、全局Critic527，实际从零训练；未加载桶checkpoint |
| 已确认预算 | 从零12000轮、2048env×24步、seed721、每2000保存；不加载短检查模型 |
| 执行位置 | 云端独立目录；09-20 13:25:54 UTC正常结束，真实退出0；已按同名本地目录归档 |

以下为09-19准备/启动时的历史记录；当前已完成训练及评测，结果见6.6节，不表示云端仍运行。

最初只读核对：远端RTX4090共24564MiB，已用6396MiB、利用率71%，一个训练进程214432；
桶双删除组读取时为3780轮，最近100轮平均2.649秒/轮。它仍在运行，未启动/停止其他进程。
环境metadata：torch2.7.0+cu128、IsaacSim5.1.0.0、IsaacLab0.47.2、numpy1.26.0。
旧小桌runtime/20kg URDF、训练入口、相对向量计算模块均存在；存在性检查不等于新实验已部署。
同卡再开一组有显存余量，但双任务初始化峰值/吞吐未实测；当前只建议最多额外一组，
不据显存空闲承诺加速。共享存储df值不是用户独享配额，没有据此宣称可用配额。

随后用户批准接入与云端准备，已完成：

- 新增`smalltable5kg_A`身份及资产；原687状态参考逐字节不变，惯量乘0.25，五碰撞盒/COM不变。
- A公式及权重复用，checkpoint绑定新实验身份；桶/旧20kg默认不变，禁止误用B和桶artifact。
- 小桌未生成源接触标签；格式中的全零掩码明确not_used_for_smalltable_A，实际四手接触仍记录。
- 267项CPU回归通过；旧20kg、桶A、桶B最终模型均通过身份及Actor/Critic/normalizer严格加载。
- 本地1环境8步物理检查真实退出0，5kg、0.5/0.5/0、200/50Hz和158/527维核对通过。
- 453文件冻结上传并校验，约24MiB；独立云端目录`/workspace/core4d_smalltable5kg_A_20260919`。
- 云端2048环境×24步×2轮检查于20:47:53–20:50:44 UTC运行，PID249291，真实退出0。
  初始与第2轮模型保存，指标有限、gate=1；第2轮模型CPU严格加载通过。
- 采样到合计显存约12.1GiB；两次更新5.746/5.450s。旧桶4290→4351轮，未停止或改代码。
  两轮短检查不是学习评价，也不能证明长时间并行有吞吐收益。
- 上述准备结束时未启动正式训练；随后用户明确confirm，于21:01:13 UTC启动正式12000轮，
  PID249989，tmux=`smalltable5kg-A-train12000`。从零初始化，不续接短检查模型。
  输出目录=`logs/Core4DSmallTableA/A_5kg_fresh12000_env2048_seed721_cloud_20260919`（云端）。
  启动前再次校验453文件；旧桶任务此时4589轮，未停止。无自动重启或自动续训。
  独立授权及真实启动记录为云端operations目录`formal.launch.json`/`formal.started.json`；
  原冻结说明与资产manifest保留准备时状态，不改写历史快照。没有新增/升级依赖、GitHub推送或覆盖旧实验。

配置说明：[SMALLTABLE5KG_A_TRAINING_CN.md](SMALLTABLE5KG_A_TRAINING_CN.md)；
完整证据：[准备记录](logs/Core4DSmallTableA/preparation_20260919/PREPARATION_CN.md)，
原始退出/配置/指标在同目录`cloud_records/`。

### 6.6 小桌5kg双删除对照（2026-09-20云端完成并本地归档）

这是小桌第7个正式配置、有限复试的第2个名额，已完成12000轮，计入已完成轮数。
用户不做新增桶腕部消融；本组对照第6组完整A，不更换参考或重做资产。

| 配置 | 完整A（6.5节） | 本次双删除 |
|---|---|---|
| 质量/几何/惯量/参考/材质 | 5kg固定资产 | 完全相同，资产SHA逐一一致 |
| 物体位置/朝向/独立高度/相对关系 | 1/1/1/2 | 1/1/0/0 |
| 物体位置xyz误差权重 | 1/1/2 | 1/1/2，不是取消所有高度信息 |
| 身体跟踪/正则/终止/网络 | 原配置 | 不变，Actor158、Critic527 |
| 训练 | fresh12000，2048×24，seed721，每2000保存 | 相同，不加载短检查模型 |

以下准备与启动信息保留09-20时序；当前完成状态见本节末尾。

实验入口 `--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height`：
前者是共用场景身份，后者才是本组奖励身份，明确写入run_config和checkpoint。
没有旧Laplacian或B接触调制，不强制手接触；高度及关系诊断仍记录，只有奖励贡献归零。
268项CPU回归通过；本次包453文件、25014148字节，仅4个实现文件与技术说明不同于完整A包。
包SHA256：`e9120f233ae451f5a9b5b2496324a21cf7b6a9ac581621ebc45713061e1259f7`。
云端独立目录 `/workspace/core4d_smalltable5kg_no_rel_no_height_20260920`；完整A进程不停止。
两轮技术检查真实退出0、模型CPU严格加载通过；随后09:17:54 UTC正式从零启动。
PID288351，tmux=`smalltable5kg-bothremoved-train12000`；旧A PID249989不停止。
首次运行核对：新任务已到8轮、旧A为9216轮，指标数值有限，合计显存约12GB；
两份正式配置除奖励身份/两项权重、独立路径、启动时间外一致，实际从零初始化。
输出 `logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920`。
只表示已启动，不表示完成或学习效果通过；不将早期学习效果当成启动条件。
操作证据：`logs/Core4DSmallTableA/both_removed_20260920_launch/`。

09-20收件更新（覆盖上述启动时状态）：双删除于19:48:01 UTC正常结束，真实退出0。
两组共37原始文件393865900字节已归档；14个模型（含2个最终12000）全部SHA256/CPU严格加载通过，
每组12000条连续指标均有限。在这个收件时点，云端原件未删、实例未停，尚未运行本地评测；
当日晚间评测已完成（见下段），随后09-20文档/源码已按用户授权推送。09-25未核查远端实例状态。
完整配置仍只差奖励两项贡献/身份、独立路径和开始时间。
目录入口与全部校验见[收件报告](logs/Core4DSmallTableA/collection_20260920/RECEIPT_CN.md)。

随后同协议评测完成：两组真实退出0，全部10个回放保留。完整A 5/5、双删除3/5完整；
完整回合位置RMSE均值5.57/10.25cm，高度RMSE 2.22/3.44cm。完整A的agent0手接触更常见，
但两组均未持续整桌离地>1cm，不能当作已成功复现抬桌语义，也不把法向接触比例当承重贡献。
8080为完整A episode001，8081为双删除episode003，8082为双删除失败episode000。
只对本次单训练seed固定场景比较作结论；不能分别归因两项奖励，不新增小桌训练。
完整口径和失败前缀见[评测报告](logs/Core4DSmallTableA/comparison_20260920/RESULTS_CN.md)。

## 7. 论文材料的CPU事件索引（2026-09-19）

论文主张/英文介绍在[story.md](story.md)，方法技术细节在[retarget-rubberhand-A1.md](retarget-rubberhand-A1.md)。
本节是素材索引，不另立论文路线；视频尚未新录制。主图优先组织：多源流程、多动作结果、
椅子同数据腕部前后、桶参考时序与物理接触、小桌奖励因素。

### 7.1 桶交接与延长参与

已读A episode004、B episode002、去关系 episode001，均为已展示的完整物理回放，497记录、50Hz。
文件在各运行目录`evaluation_20260919_seed721/episode_XXX.npz`。使用`episode_step/50`定位播放时间，
不是直接`reference_frame/50`：本次phase比step大1，最后夹到496。初始无物理采样记录排除。
按既有手—桶法向力模长>1N判定；手索引0/1为左/右，agent0为源参考先交出者、agent1为接收者。

源人体几何软标签：agent0正置信度phase165–278，agent1为222–466；对应本次回放
3.28–5.54s和4.42–9.30s。共同标签区间4.42–5.54s，从5.56s起agent0标签结束。

| 回放 | 两人同时手接触的主要段（秒） | agent0标签结束后右手继续接触的主要段（秒） |
|---|---|---|
| A / 004 | 4.04–4.60、4.82–6.10、6.20–7.24、7.28–7.38 | 5.56–6.10、6.20–7.24 |
| B / 002 | 3.94–4.28、4.32–5.08、5.12–5.76、5.96–7.88 | 5.56–5.76、5.96–7.88 |
| 去关系 / 001 | 4.04–4.46、4.50–5.84、5.90–7.68、7.78–7.92 | 5.56–5.84、5.90–7.94 |

仅列连续跨度至少0.10s的主要片段，未平滑或补空隙。共同截图候选时间为4.44、5.56、6.40、
7.00、7.80、9.00s。可用表述：物理执行相对源标签形成了交出者延长参与的阶段。
三组都有这个现象，所以它不是关系奖励独有的效果。结合用户观察可讨论补偿接触，法向力
时间索引本身不等于承重或因果互助证明；标签是几何接近而非动捕实测接触力。

复算口径：从NPZ读取`reference_frame`、`episode_step`、`contact_valid_after_physics`；
对`hand_object_normal_force_w`最后一维取范数并阈值1N，按手取any得到各agent接触；
两agent逻辑与为共同接触，`agent0_contact & (reference_frame>=279)`为标签后接触，
取连续True段的首尾step并除以50。没有访问checkpoint、启动Isaac或修改训练。

## 8. 可复用处理与自动检查：论文依据（2026-09-19只读源码核对）

对应老师16:41–17:17提出的可重复数据处理要求。这是现有代码清单，不是本次新建算法，
也不代表已经重新跑完所有样例。可按“规范化→重定向→可选腕部优化→参考导出→物理检查”
复用各入口；动作选择、目标尺寸取舍和语义验收仍由人工确认。

| 环节与实际入口 | 已实现的可重复检查或记录 | 判断范围 |
|---|---|---|
| [源数据规范化](scripts/prepare_core4d_sequence.py) / [adapter](src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_adapter.py) | 帧数/形状/有限值、旋转矩阵与单位四元数、关节顺序/FPS；配对时序重采样，保存后重读验证 | 校验格式与数值；接触方式是否适合固定手型仍需看片 |
| [双人重定向](scripts/retarget_core4d_pair.py) | 记录物体采样数量、seed和点集hash，以及关节限位、脚约束、非穿透等实际配置 | 求解器按配置实施约束；不能把配置记录等同于物理可执行保证 |
| [腕部优化](scripts/refine_core4d_pair_a1.py) | 物体轨迹冻结、允许修改的关节集合检查；方向残差与关节变化/速度摘要 | wrist-only模式支持严格前后配对；速度摘要不等于已实现突变自动筛除 |
| [运行参考导出](scripts/synthesize_core4d_smalltable_runtime_reference.py) | 自定义参考需来源manifest；来源hash、独立输出、训练资产声明验证；复用FK/差分生成运行通道 | 保持来源可追溯；training_ready不是自动语义判定 |
| [物理与接口检查](scripts/smoke_core4d_smalltable.py) | 资产hash、观测/动作维度、参考FPS、物理/控制频率、实际质量/材质；有限步执行及可选PPO更新 | 检查接线与运行有效性，不将短检查表现当作完整训练效果 |

当前可支持的表述：已有覆盖数据接入、参考加工到物理执行的可复用工具及分层检查，
能以同一套入口处理已接入来源的新序列，并保存来源与运行配置。
未做的全数据集自动选优/自动修复、统一语义评分和批量通过率，不作为现有贡献。
论文先把已实现步骤、已有案例及人工确认位置讲清楚，无需为补故事新造复杂筛选系统。

## 9. 实验身份与文件入口（2026-09-25统一核查）

这是第2/3节的可执行文件索引，不是新路线或重新运行结果。只读核对run_config、status、
逐回合summary、报告及文件存在性；没有加载模型、重新计算所有资产hash或启动仿真。
表中短ID仅方便引用，不重命名目录、不搬运文件、不修改历史JSON。

### 9.1 CORE4D十三组正式实验

全部从零12000轮；2048环境×24步、训练seed721、每2000保存、共享Actor158/全局Critic527。
13个最终模型、配置、status和对应summary均存在，status均记录final_iteration=12000、passed=true。
这不是从status推断正常退出；真实进程退出码须看各组原始exit/收件报告。完成数由summary逐回合
completed_reference统计，指到参考末尾，不等同于操作语义或泛化成功。

| ID | 配置/奖励身份 | 质量/参考 | 最终模型与实际配置 | 最终模型评测 | 完整回合 |
|---|---|---|---|---|---|
| B-A | 桶A：R0(2)+rz+2rrel | 1kg / BKT | [12000](logs/Core4DBucket/A_fresh12000_env2048_seed721/model_12000.pt) · [配置](logs/Core4DBucket/A_fresh12000_env2048_seed721/run_config.json) | [summary](logs/Core4DBucket/A_fresh12000_env2048_seed721/evaluation_20260919_seed721/summary.json) | 2/5 |
| B-B | 桶B：A物体/关系块乘g | 1kg / BKT | [12000](logs/Core4DBucket/B_fresh12000_env2048_seed721_cloud_20260918/model_12000.pt) · [配置](logs/Core4DBucket/B_fresh12000_env2048_seed721_cloud_20260918/run_config.json) | [summary](logs/Core4DBucket/B_fresh12000_env2048_seed721_cloud_20260918/evaluation_20260919_seed721/summary.json) | 1/5 |
| B-R0 | 桶去关系：R0(2)+rz | 1kg / BKT | [12000](logs/Core4DBucket/A_no_rel_fresh12000_env2048_seed721_20260919/model_12000.pt) · [配置](logs/Core4DBucket/A_no_rel_fresh12000_env2048_seed721_20260919/run_config.json) | [summary](logs/Core4DBucket/A_no_rel_fresh12000_env2048_seed721_20260919/evaluation_20260919_seed721/summary.json) | 4/5 |
| B-Z0 | 桶去独立高度：R0(2)+2rrel | 1kg / BKT | [12000](logs/Core4DBucket/A_no_height_fresh12000_env2048_seed721_cloud_20260919/model_12000.pt) · [配置](logs/Core4DBucket/A_no_height_fresh12000_env2048_seed721_cloud_20260919/run_config.json) | [summary](logs/Core4DBucket/A_no_height_fresh12000_env2048_seed721_cloud_20260919/evaluation_20260919_seed721/summary.json) | 0/5 |
| B-RZ0 | 桶双删除：R0(2)；当前首选展示 | 1kg / BKT | [12000](logs/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919/model_12000.pt) · [配置](logs/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919/run_config.json) | [summary](logs/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919/evaluation_20260920_seed721/summary.json) | 5/5 |
| T-01 | 小桌基础11项，wz1 | 20kg / ST20 | [12000](logs/Core4DSmallTable/paired_reference_fresh12000_save2000_actor158_seed721_env2048/model_12000.pt) · [配置](logs/Core4DSmallTable/paired_reference_fresh12000_save2000_actor158_seed721_env2048/run_config.json) | [summary](logs/Core4DSmallTable/paired_reference_fresh12000_save2000_actor158_seed721_env2048/evaluation/model12000_seed721/summary.json) | 5/5 |
| T-02 | 小桌wz1+旧Laplacian | 20kg / ST20+LAP | [12000](logs/Core4DSmallTable/interaction_mesh_fresh12000_save2000_actor158_seed721_env2048/model_12000.pt) · [配置](logs/Core4DSmallTable/interaction_mesh_fresh12000_save2000_actor158_seed721_env2048/run_config.json) | [summary](logs/Core4DSmallTable/interaction_mesh_fresh12000_save2000_actor158_seed721_env2048/evaluation/model12000_seed721/summary.json) | 2/5 |
| T-03 | 小桌基础11项，wz2 | 20kg / ST20 | [12000](logs/Core4DSmallTable/baseline_z2_fresh12000_save2000_actor158_seed721_env2048_20260915/model_12000.pt) · [配置](logs/Core4DSmallTable/baseline_z2_fresh12000_save2000_actor158_seed721_env2048_20260915/run_config.json) | [summary](logs/Core4DSmallTable/baseline_z2_fresh12000_save2000_actor158_seed721_env2048_20260915/evaluation_20260916/model12000_seed721/summary.json) | 5/5 |
| T-04 | 小桌wz2+旧Laplacian | 20kg / ST20+LAP | [12000](logs/Core4DSmallTable/interaction_mesh_z2_fresh12000_save2000_actor158_seed721_env2048_cloud_20260915/model_12000.pt) · [配置](logs/Core4DSmallTable/interaction_mesh_z2_fresh12000_save2000_actor158_seed721_env2048_cloud_20260915/run_config.json) | [summary](logs/Core4DSmallTable/interaction_mesh_z2_fresh12000_save2000_actor158_seed721_env2048_cloud_20260915/evaluation/model12000_seed721/summary.json) | 5/5 |
| T-05 | 小桌wz2+旧图+独立高度负惩罚 | 20kg / ST20+LAP | [12000](logs/Core4DSmallTable/interaction_mesh_z2_height1_scale005_fresh12000_save2000_actor158_seed721_env2048_20260916/model_12000.pt) · [配置](logs/Core4DSmallTable/interaction_mesh_z2_height1_scale005_fresh12000_save2000_actor158_seed721_env2048_20260916/run_config.json) | [summary](logs/Core4DSmallTable/interaction_mesh_z2_height1_scale005_fresh12000_save2000_actor158_seed721_env2048_20260916/evaluation_20260917/model12000_seed721/summary.json) | 4/5 |
| T-06 | 5kg小桌完整A：R0(2)+rz+2rrel | 5kg / ST5 | [12000](logs/Core4DSmallTableA/A_5kg_fresh12000_env2048_seed721_cloud_20260919/model_12000.pt) · [配置](logs/Core4DSmallTableA/A_5kg_fresh12000_env2048_seed721_cloud_20260919/run_config.json) | [summary](logs/Core4DSmallTableA/A_5kg_fresh12000_env2048_seed721_cloud_20260919/evaluation_20260920_seed721/summary.json) | 5/5 |
| T-07 | 5kg小桌双删除：R0(2) | 5kg / ST5 | [12000](logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/model_12000.pt) · [配置](logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/run_config.json) | [summary](logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/evaluation_20260920_seed721/summary.json) | 3/5 |
| C-01 | 椅子基础11项，wz1 | 5kg / CHR | [12000](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/model_12000.pt) · [配置](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/run_config.json) | [summary](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/evaluation/model12000_seed/summary.json) | 9/9（3×3） |

椅子的另外两份summary：[seed722](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/evaluation/model12000_seed722/summary.json)、
[seed723](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/evaluation/model12000_seed723/summary.json)；第一份目录确实叫
`model12000_seed`，不要补造seed721路径。三份回放数值相同，仅作重复性记录。
T-04另有[09-17同模型复测](logs/Core4DSmallTable/interaction_mesh_z2_fresh12000_save2000_actor158_seed721_env2048_cloud_20260915/evaluation_20260917/model12000_seed721/summary.json)，
与T-05进行高度惩罚比较；不另算一次训练。

**共同物理与配置追溯**：上述CORE4D资产采用静/动摩擦0.5/0.5、恢复0，物理200Hz/控制50Hz。
具体reference和资产清单见下表；精确训练参数、运行身份和hash以每行实际配置及云端冻结清单为准。
云端旧绝对路径保留原样，本表提供本地对应文件，不重写历史来源。

| 参考代号 | 数据与处理路径 | 本地运行参考 | 资产清单/关系artifact |
|---|---|---|---|
| BKT | 20231020/071 / bucket003；原尺寸、两阶段后wrist-only A1 | [runtime](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz) | [manifest](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/training_asset_manifest.json) · [关系](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/interaction_vectors_v1.npz) |
| ST20 | 20231030/001 / desk001；共享缩小物体Stage1 nominal | [runtime](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/core4d_pair_runtime_fps50.npz) | [manifest](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/training_asset_manifest.json) |
| ST5 | 与ST20同一运动参考；5kg资产独立身份 | [runtime](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz) | [manifest](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/training_asset_manifest.json) · [关系](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/interaction_vectors_v1.npz) |
| CHR | 20231020/074 / chair021；共享小椅子A1 | [runtime](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_chair021_20231020_074_a1/core4d_pair_runtime_fps50.npz) | [manifest](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_chair021_20231020_074_a1/training_asset_manifest.json) |

LAP使用[小桌旧Laplacian artifact](src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/interaction_mesh_100_v1.npz)；
BKT与ST5各用自己的采样点及相对向量，不能互换。ST20/ST5在run_config记录相同runtime SHA，
与“不重新retarget、只换质量资产”的设计一致；本次未重新计算NPZ hash。

身份解释与保留范围：

- T-01旧配置没有显式reward_variant/wz字段，T-02也无显式wz；基础项和wz1由实际reward_terms、
  旧报告与配置说明联合确定，不回填原始JSON。
- 桶与ST5顶层旧通用object_z_error_weight字段不控制专用块；实际contract为xyz=[1,1,2]。
  第3节矩阵记录生效奖励，不据顶层旧字段误写wz1。
- B-B和T-04标准目录本地只保留最终model_12000；其余11组各有0、2000至12000七个模型。
  最终模型均在；不声称未收取的中间模型在其他机器上仍可用。
- 旧收件目录与标准归档目录不是两个实验，smoke及失败预评测也不追加计入正式比较。
- 代表回放：桶B-RZ0为evaluation_20260920_seed721/episode_000.npz；
  T-06为同名评测目录episode_001.npz；T-07完整代表为episode_003.npz，
  失败例为episode_000.npz。全部回合保留，旧端口不作为唯一定位方式。

报告汇总：
[桶A/B](logs/Core4DBucket/comparison_20260919/RESULTS_CN.md)、
[桶最终消融](logs/Core4DBucket/ablation_comparison_20260920/RESULTS_CN.md)、
[旧小桌高度对照](logs/Core4DSmallTable/interaction_mesh_z2_height1_scale005_fresh12000_save2000_actor158_seed721_env2048_20260916/evaluation_20260917/RESULTS_CN.md)、
[5kg两组](logs/Core4DSmallTableA/comparison_20260920/RESULTS_CN.md)、
[椅子](logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/evaluation/TRAINING_REVIEW.md)。

### 9.2 历史OMOMO五类Demo（只索引，不搬入当前工作树）

根目录为 `/home/kevin/holosoma-rubber-hand-marl`。下表均有实际文件；质量从各run_config物理读回为20kg，
机器人为rubber-hand G1。精确摩擦、参考构造、预算、保存间隔及控制实现以各自配置为准，
不套用CORE4D的fresh12000。前三类WBT初始化，Demo3/4奖励及桌轨迹用途不同。

| Demo | 当前保留模型/配置 | 评测入口 | 奖励身份与结果 |
|---|---|---|---|
| Push live-PD | [model_13050.pt](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Push/a1_livepd_20kg_continue7000_from_08050_seed721_env2048/model_13050.pt) · [配置](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Push/a1_livepd_20kg_continue7000_from_08050_seed721_env2048/run_config.json) | [日志/回放](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Push/eval_livepd_13050_multiseed_20260827) | 基础11项；无joint-acceleration schedule；13050为1/9；14050、15050均0/9 |
| Pull | [model_08050.pt](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt) · [配置](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/run_config.json) | [日志/回放](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Pull/eval_full8000_seed721/model_08050_object_centric.npz) | 基础11项；已有单条完整回放316/316帧 |
| Kick | [model_08000.pt](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/model_08000.pt) · [配置](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/run_config.json) | [日志/回放](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Kick/eval_model08000_seed721/model_08000_actor_only_rollout.npz) | 基础11项；已有单条完整回放step0–297 |
| Demo3对抗 | [model_10100.pt](/home/kevin/holosoma-rubber-hand-marl/logs/Demo3Tug/square_table_diagonal_tug_full15000_seed721_env2048/model_10100.pt) · [配置](/home/kevin/holosoma-rubber-hand-marl/logs/Demo3Tug/square_table_diagonal_tug_full15000_seed721_env2048/run_config_resume_from_06800.json) | [日志/回放](/home/kevin/holosoma-rubber-hand-marl/logs/Demo3Tug/eval_model10100_seed721/evaluation.json) | 8项身体/正则+相反拉轴速度；5回合均6步跌倒；诊断用途 |
| Demo4旋转 | [model_10000.pt](/home/kevin/holosoma-rubber-hand-marl/logs/Demo4Rotate/rectangular_pull_pull_rotate90_full10000_save1000_seed721_env2048/model_10000.pt) · [配置](/home/kevin/holosoma-rubber-hand-marl/logs/Demo4Rotate/rectangular_pull_pull_rotate90_full10000_save1000_seed721_env2048/run_config.json) | [日志/回放](/home/kevin/holosoma-rubber-hand-marl/logs/Demo4Rotate/eval_model10000_seed721/evaluation.json) | 8项身体/正则+yaw进度/首次达标；5/5触发角度目标，但猛烈甩转；非质量合格 |

| Demo | 实际参考入口/构造 | 配置中的物体资产名 |
|---|---|---|
| Push live-PD | [单人A1](/home/kevin/holosoma-rubber-hand-marl/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz)；运行时横向配对，间距0.8m，paired_reference_file=null | objects_widetable_plan5_training.urdf |
| Pull | [镜像双人参考](/home/kevin/holosoma-rubber-hand-marl/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/pull/plan5_attempt08_mirrored_pair_runtime.npz) | objects_widetable_plan5_pull_training.urdf |
| Kick | [镜像双人参考](/home/kevin/holosoma-rubber-hand-marl/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/kick/plan5_attempt09_dual_kick_mirrored_runtime.npz) | objects_widetable_plan5_pull_training.urdf |
| Demo3 | [对角对抗参考](/home/kevin/holosoma-rubber-hand-marl/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/demo3_tug/sub3_010_diagonal_tug_runtime.npz)；桌轨迹用于reset/schema，不作共同轨迹奖励 | objects_squaretable_demo3_training.urdf |
| Demo4 | [机器人拉动/静态桌reset参考](/home/kevin/holosoma-rubber-hand-marl/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/demo4_rotate/demo4_pull_pull_static_table_runtime.npz)；不跟踪桌轨迹 | objects_widetable_plan5_pull_training.urdf |

**Push两条链必须区分**：表中为live-PD 13050，不是
[旧smooth 13050](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Push/a1_10050_jointacc_continue_to15000_seed721_env2048/model_13050.pt)。
旧smooth带joint-acceleration schedule（目标−5e−9、ramp500）；其
[旧评测目录](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Push/a1_10050_jointacc_continue_eval/)不能支持live-PD的成功率。
live-PD续训目录另保留14050/15050，同链前驱为
[live-PD 08050](/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Push/a1_livepd_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt)。
后续统一视频必须标出模型完整路径，不能只写“13050”。

Demo3目录名包含15000，但恢复运行实际按要求止于10100，不把原计划终点当完成量。
Demo4当前只保留最终10000，不能依据save1000目录名推断中间模型还在。
Push/Pull/Kick对应评测没有独立summary.json，已有日志/NPZ及
[历史路线图报告](/home/kevin/holosoma-rubber-hand-marl/MULTI_AGENT_EMERGENCE_ROADMAP.md)；不编造summary路径。
这些结果与CORE4D评测口径分别标注，不把单条完整回放和多回合比例当作同一统计。

### 9.3 09-25本次统一完成项与剩余材料

- 已完成：三份现有文档同步；CORE4D十三组和OMOMO五类的模型/配置/参考/奖励/评测入口核对。
  已存在模型采用索引；原日志、配置和资产未改。可读配置/评测副本另存交接证据目录，按本轮授权同步GitHub。
- 已完成方法核查：三类腕部配对、椅子数值图、同目标单/两阶段条件表和参考来源追溯，见4.1。
- 已整理Mac交接：五类主展示＋桶A/B＋5kg小桌分析，共8段MP4、48张PNG、8张六宫格；
  从已有物理回放离线重新渲染，不重新求解、运行策略或动力学步进。
  同时提供源码导航、主张索引和13组便携配置/评测（原日志不动）。
- 可选补充：腕部前后同相机图、方法共同几何指标、桶交接定量时序；不混称已完成。
- 待用户复核：最新5kg小桌两组回放；不把这项反馈设为继续写作的障碍。
- 需另行确认：OMOMO单人奖励实验、任何新训练/新数据/改奖励/重求解。
- 用户已授权本轮文档、便携证据、展示素材及独立导出工具提交/推送；未删除备份或checkpoint。
