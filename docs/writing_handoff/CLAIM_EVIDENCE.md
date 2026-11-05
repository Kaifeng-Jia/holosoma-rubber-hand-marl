# 主张—证据索引：Mac 写作端

更新：2026-09-25。本文是现有结果的写作查证表，不是论文正文，也不提出新研究或训练计划。叙事定位见 [story.md](../../story.md)，实验身份见 [台账](../../DEMO_AND_EXPERIMENT_INVENTORY.md)，实现导航见 [CODE_MAP.md](CODE_MAP.md)。

主线：**多源人类示范 → 交互感知重定向与形态适配 → 双机器人/共同物体参考 → shared actor、global critic 下的物理学习与执行。** 主体是已经贯通且可复用的处理与学习流程；腕部对照、奖励对照和行为分析分别解释其中的具体环节。

## 统一口径

- 当前实证范围：OMOMO 单人来源的派生配对、CORE4D 真实双人来源，双 G1、固定橡胶手、物理仿真、按动作分别训练。可复用的是来源接口和处理/学习模块，不是一个覆盖所有动作的统一策略。
- CORE4D 十三组正式实验均为训练 seed721、fresh12000、2048环境×24步；5回合或椅子3×3回合是固定场景重复执行，不是多个独立训练seed。“完整”指到达参考末尾；语义认可、轨迹误差、离地和接触分别报告。
- 证据优先看本地可携带 [evidence/README.md](evidence/README.md) 与 [index.json](evidence/index.json)：`config_extract.json` 是可读派生配置，旁边的 `run_config.json.gz` 保存原始字节，`summary.json` 保存既有评测。历史报告中的绝对路径是来源记录；下表链接均可在 clone 中打开。

## 1. 主线与已实现能力

| 现有结果 / 事实 | 适用论点 | 可携带证据文件 | 准确表述要点 |
| --- | --- | --- | --- |
| OMOMO 的单人动作通过平移/镜像配对形成双机器人参考；CORE4D 保留原两人时序和共享物体，两条路径进入 paired/runtime 与物理训练接口 | 不同示范资源可进入共同协作学习流程；研究不局限于一个手工双人 demo | [台账§2、§8、§9](../../DEMO_AND_EXPERIMENT_INVENTORY.md)、[Pull 配置](evidence/historical/Pull/config_extract.json)、[Kick 配置](evidence/historical/Kick/config_extract.json)、[椅子配置](evidence/C-01/config_extract.json)、[代码入口](CODE_MAP.md) | “流程贯通了单人示范派生配对与真实双人示范两种来源。”来源身份分别保留：OMOMO 配对是构造参考，CORE4D 是原始双人动捕。LAFAN/AMASS 源码入口可作为接口能力介绍，不计入已有物理训练的实证来源数。 |
| 复用 OmniRetarget 交互关系求解，加入显式尺度路径、共同物体配对与 wrist-only 掌框架目标；runtime 补齐 FK、速度及统一时钟 | 交互信息与形态适配能够被组织为可重复的处理模块 | [重定向核查报告](../experiments/retarget_method_comparison_20260925.md)、[指标与来源哈希](../experiments/retarget_method_comparison_20260925.json)、[CODE_MAP§2–4](CODE_MAP.md) | “在交互感知重定向基础上，形成面向双机器人参考的尺度、配对和掌面方向适配接口。”Omni 基础与本项目新增接口分别归属；具体数值收益用第2节对照支撑。 |
| OMOMO 路线使用 WBT 运动先验初始化；CORE4D 路线从零联合学习；均在物理仿真中以参考引导的 PPO/MAPPO 训练 | 同一框架可利用已有单人动作先验，也可直接利用真实双人参考 | [历史配置索引](evidence/README.md)、[CORE4D 初始化配置](evidence/C-01/config_extract.json)、[初始化与 PPO 导航](CODE_MAP.md) | “参考提供跟踪奖励与动作引导，策略通过物理交互学习执行；单人 WBT 可提供初始化先验。”WBT 是物理跟踪强化学习，不写成监督行为克隆；两来源路线不是 warmstart 优于 fresh 的受控比较。 |
| 共享 actor 每机器人输入158维、输出29维；集中式 critic 每物理环境527维；评测只调用 actor | 集中式训练、分散式执行的双机器人实现已经落地 | [椅子配置](evidence/C-01/config_extract.json)、[actor-only 评测标记](evidence/C-01/summary.json)、[Pull 配置](evidence/historical/Pull/config_extract.json)、[维度与控制导航](CODE_MAP.md) | “同一动作内两个机器人共享 actor，各自使用本体/参考及队友相对信息；global critic 仅参与训练。”actor-only 仍使用参考输入；新意定位在完整流程与适配接口，不把 PPO/MAPPO 本身写成新算法。 |
| 接入、重定向、腕部冻结、runtime 导出、物理接线均有格式/数值/哈希/配置检查 | 可复用性有具体工程机制支撑，而非只展示几段动画 | [台账§8](../../DEMO_AND_EXPERIMENT_INVENTORY.md)、[源码与回归测试导航](CODE_MAP.md)、[证据索引与校验方式](evidence/README.md) | “处理链保存来源与运行契约，并提供分层自动检查；动作选择、物体尺度取舍和语义验收由人工确认。”以实际已实现检查支撑可重复处理能力。 |

## 2. 重定向对照：支持哪一层结论？

| 现有结果 | 适用论点 | 证据文件 | 准确表述要点 |
| --- | --- | --- | --- |
| 椅子 W-CHAIR：全235帧、30Hz，四手等权平均完整掌框架 SO(3) 误差 **62.725°→0.059°**；逐手均值42.421/86.345/53.167/68.967°→近0/近0/近0/0.236° | 显式示范掌面目标补足位置关键点不能唯一确定的末端方向 | [核查报告§2–3](../experiments/retarget_method_comparison_20260925.md)、[JSON：chair](../experiments/retarget_method_comparison_20260925.json)、[方向误差图](../experiments/assets/chair_wrist_orientation_20260925.svg) | “在保持同一输入、采样率、物体及非腕 qpos 不变的对照中，wrist-only A1 显著降低参考掌框架方向误差。”每机器人仅左右腕共6关节、两机器人共12变量；指标是参考方向匹配，不是逐指/接触误差或下游策略成功率的因果提升。 |
| 椅子人2右腕存在16帧残差，最大5.008°；腕角帧间变化增大。桶和 OMOMO matched A1 也已核对严格六腕配对并改善方向，但碰撞代理几何可变差 | 同一腕部目标接口可跨来源复用，同时方向、连续性与接触几何是不同评价量 | [核查报告§3–4](../experiments/retarget_method_comparison_20260925.md)、[JSON：chair/bucket/OMOMO 对应记录](../experiments/retarget_method_comparison_20260925.json) | “跨数据对照进一步验证方向接口的可复用性；几何和时间连续性单独报告。”桶凸代理穿透不等于真实空心桶壁侵入；历史 OMOMO 与09-11复跑分别匹配各自 baseline。 |
| 原尺寸 desk001：单阶段140帧，两阶段413帧；共同前140帧输入、最终物体轨迹、资产与采样点匹配，双方无额外A1 | 已具备同输入/目标下的 pipeline 整体比较材料，显式展示 nominal→physical 适配路径 | [核查报告§5.1](../experiments/retarget_method_comparison_20260925.md)、[条件与哈希 JSON](../experiments/retarget_method_comparison_20260925.json) | “共同前140帧可比较已记录的单阶段 pipeline 与两阶段适配 pipeline。”140帧由 `--max-frames` 主动截断，不是失败点；脚高锚定、限位、弹性约束及 nominal prior 同时变化，且共同质量指标尚缺，故这里报告条件审计而非阶段数单变量优势。 |
| 训练小桌来自 Stage1 共享缩小物体；椅子为 Stage1 共享缩小+A1；桶为原尺寸 two-stage+A1 | 目标尺度选择与最终阶段适配是可独立说明的流程决策 | [核查报告§5.2](../experiments/retarget_method_comparison_20260925.md)、[小桌来源 manifest](../../src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/source_manifest.json)、[实验配置索引](evidence/index.json) | “流程容纳共享缩小目标与原尺寸适配两种参考选择，并记录各训练资产的来源。”小桌和椅子的训练表现用于对应参考路径，不能归为 Stage2 恢复原尺寸的验证。 |

## 3. 多动作物理执行：展示与完整记录并列

| 现有结果 | 适用论点 | 证据文件 | 准确表述要点 |
| --- | --- | --- | --- |
| Pull `08050`：固定单回放316/316帧，实际/参考净位移0.58082/0.58428m；Kick `08000`：固定单回放 step0..297 完整，均有用户认可记录 | OMOMO 单人动作经配对与物理学习可用于双机器人拉动、踢动执行 | [来源可核对的历史摘录](evidence/historical/rollout_history_excerpts.md)、[Pull 配置](evidence/historical/Pull/config_extract.json)、[Kick 配置](evidence/historical/Kick/config_extract.json) | “在记录的固定初始化 actor-only 回放中，派生配对参考产生了完整的双机器人 Pull/Kick 执行。”这是各自单条物理记录；没有独立 summary 的历史结果用原摘录呈现。 |
| Push live-PD：已有认可回放；`13050`固定重复评测1/9完整，`14050`、`15050`均0/9 | Push 是已实现的动作实例，也是训练回报、展示质量与从头执行完整性需要分别检查的案例 | [27次结果字段及来源哈希](evidence/historical/Push-livePD/evaluation_outcomes_27runs.json)、[live-PD 配置](evidence/historical/Push-livePD/config_extract.json)、[历史记录](evidence/historical/rollout_history_excerpts.md) | “Push 展示配对动作的物理执行，同时保留该链完整重复评测结果。”模型身份写明 live-PD，不混入旧 joint-acceleration 平滑链；不按最好回放宣称稳定完成。 |
| CORE4D 椅子5kg：fresh12000，三轮固定条件评测合计9/9完整，物体位置 RMSE 约3.42cm；动作获认可，参与不均有记录 | 真实双人示范能够直接支撑参考引导的双机器人物理执行 | [配置](evidence/C-01/config_extract.json)、[721](evidence/C-01/summary.json)、[722](evidence/C-01/summary_seed722.json)、[723](evidence/C-01/summary_seed723.json)、[复核报告](evidence/reports/chair_training_review.md) | “椅子参考经腕部适配后，从零联合训练得到固定场景完整执行。”三轮轨迹相同，9/9是执行记录；腕部方向对照与此策略结果分别列出。 |
| CORE4D 桶 B-RZ0：fresh12000，5/5完整、每回合496步；桶源网格最低点>1cm的最长连续段均3.88s；episode000获用户最优语义反馈 | 真实双人交接时序可转为具有持续物体抬升的物理执行 | [配置](evidence/B-RZ0/config_extract.json)、[全5回合 summary](evidence/B-RZ0/summary.json)、[最终报告](evidence/reports/bucket_ablations_final_20260920.md) | “桶交接双删除组在固定场景完成5/5回合，并产生持续物体抬升；展示回合经用户认可。”展示取完整回合中3D误差最小的episode000，全体结果另列。离地为源网格几何口径，不等于手部承重比例。 |

## 4. 奖励与行为分析：保留完整反差

桶五组和小桌5kg两组均使用最终12000模型；先报告全部完成数，再对明确标注的完整回合集合比较误差。`wz2` 指物体三维位置误差中的 z² 系数为2；删除独立高度项仍保留该竖直跟踪。

| 现有结果 | 适用论点 | 证据文件 | 准确表述要点 |
| --- | --- | --- | --- |
| 桶两因素矩阵：A **2/5**、去关系 **4/5**、去独立高度 **0/5**、双删除 **5/5**；另加软接触调制 B **1/5** | 奖励项的作用取决于组合；完成度与配方复杂度不是单调关系 | [五组最终报告](evidence/reports/bucket_ablations_final_20260920.md)、[A](evidence/B-A/summary.json)、[R0](evidence/B-R0/summary.json)、[Z0](evidence/B-Z0/summary.json)、[RZ0](evidence/B-RZ0/summary.json)、[B](evidence/B-B/summary.json) | “桶的完整五组比较显示，双删除仍可产生认可的交接动作；新增关系与独立高度项的收益依赖组合。”四个无接触调制组构成两因素矩阵，B另列；双删除保留身体/物体/朝向及正则，并非不跟踪高度或姿态。 |
| 桶仅统计完整回合：A / 去关系 / 双删除分别 n=2/4/5，z RMSE **2.02/1.56/3.36cm**，交出者示范加权手接触覆盖 **71.7/67.7/24.2%** | 完成、跟踪精度和接触方式提供互补评价维度 | [最终报告中的完整样本表](evidence/reports/bucket_ablations_final_20260920.md)、[A/B 补充报告](evidence/reports/bucket_AB_20260919.md) | “双删除完成更多回合且展示语义获认可；保留高度但去关系的完整回合高度误差更小。不同指标反映不同执行属性。”接触覆盖由几何软标签和手部法向力阈值定义，不是负载分配或整体语义得分。 |
| 小桌5kg同条件：完整A **5/5**、双删除 **3/5**；完整回合3D RMSE **5.57/10.25cm**，z RMSE **2.22/3.44cm**；两组均无持续整桌最低点>1cm | 桶上较好的展示配方不直接对应另一物体/动作上的最好跟踪；奖励效果具有场景依赖 | [同协议报告](evidence/reports/smalltable_5kg_20260920.md)、[分析 JSON](evidence/reports/smalltable_5kg_analysis_20260920.json)、[完整A](evidence/T-06/summary.json)、[双删除](evidence/T-07/summary.json) | “与桶形成反差，完整A在这对小桌实验中完成更多回合且跟踪误差更小；持续整桌抬升仍未实现。”这是配方整体对比，不分别归因于同时删除的两项。双删除失败前缀另列，不与完整轨迹混算排名。 |
| 小桌20kg：旧图+wz2→追加负高度惩罚，完整回合原点z RMSE **3.2588→1.5116cm**；完成 **5/5→4/5**，整桌持续离地未建立 | 物体原点跟踪改善与整体几何动作实现可分离，指标表征影响执行方式 | [同协议高度报告](evidence/reports/smalltable_height_penalty_20260917.md)、[旧模型复测](evidence/T-04/summary_recheck_20260917.json)、[新模型](evidence/T-05/summary.json) | “独立高度惩罚改善了原点高度精度，但该指标改善没有转化为持续整桌离地。”此负平方惩罚与桶的正指数高度奖励、旧Laplacian图与桶相对向量均分开命名；完整小桌七组总表保留在[证据索引](evidence/README.md)。 |
| 桶A/B/去关系的既有完整回放中，交出者在源几何接触标签结束后仍有手部接触；共同事件时段已索引 | 参考引导执行可出现参考标签以外的延长参与，适合作行为分析 | [台账§7.1事件索引](../../DEMO_AND_EXPERIMENT_INVENTORY.md)、[A/B 接触分析](evidence/reports/bucket_AB_20260919.md)、[最终五组解释](evidence/reports/bucket_ablations_final_20260920.md) | “物理执行出现交出者延长参与阶段，可结合同步画面讨论补偿接触。”三组都有这一现象；时序观察不等于通过干预证明必要互助，参与不均本身也不定义协作失败。 |

## 5. 补充材料与写作取证

Demo3 / Demo4 可支持“目标指标与动作质量分开验收”的补充分析：Demo3 最终模型五次均约6步跌倒；Demo4 五次达到 yaw 阈值，但台账记录猛烈甩转、偏离先验。分别使用 [Demo3 原始评测](evidence/historical/Demo3/evaluation.json)、[Demo4 原始评测](evidence/historical/Demo4/evaluation.json) 和 [台账§2](../../DEMO_AND_EXPERIMENT_INVENTORY.md)，不计入合格协作展示的数量。

写作组织顺序可沿上表：先展示两种来源和多动作物理执行，再用椅子严格腕部对照说明方向适配，用桶全部五组与小桌反差说明奖励/行为分析。范围集中放在实验设置中：固定场景、单训练seed、指定参考、双机器人、按动作训练；现有证据不承担跨场景泛化、多训练seed统计或信用分配机制已解决的结论。

取证时以历史运行契约认定实际启用项，而不是当前源码默认值；保留完整回合数、失败前缀、展示选例规则和奖励版本。表内措辞是可选准确表述片段，数值与图表应继续绑定对应文件及模型身份，不作为 abstract、introduction 或 method 正文直接代写。
