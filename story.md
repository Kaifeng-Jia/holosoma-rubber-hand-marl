# 研究故事：从人类交互示范到多机器人协作

> 套磁信与研究介绍用简版。依据老师会议及当前路线整理；不是新的执行路线图。

## 1. 一句话主线

研究如何将不同来源的单人、双人人类交互示范，转化为人形机器人可在物理仿真中执行的协作行为，建立可复用、面向更多动作与场景扩展的数据处理、重定向和多智能体学习流程。

核心不只是让两台机器人“同时动起来”，而是让它们围绕共享物体形成有效交互，并尽可能保留示范的操作方式。

## 2. 方法框架

1. **多来源示范接入。** 已接入 OMOMO 单人动捕与 CORE4D 原始双人动捕。前者通过配对、平移或镜像构造双机器人参考，后者保留原始双人交互时序；两种来源明确区分，不把合成参考称为真实双人示范。
2. **交互关系驱动的重定向。** 基于 OmniRetarget，针对人体与机器人尺寸、可达范围和末端形态差异，实现两阶段处理及掌面朝向映射与腕部优化。目标是获得更合理的机器人参考；运动学参考改善不等于已经证明物理可执行。
3. **参考引导的多智能体强化学习。** 在 Isaac 仿真中，通过 PPO/MAPPO 学习真实碰撞动力学下的执行策略。采用集中式训练、分散式执行：同一动作内共享 Actor，训练时使用全局 Critic，各机器人执行时使用自身/参考及队友相对信息。不同动作分别训练，不混成一个动作策略；可用单人 WBT 先验初始化，也支持直接从零联合训练。

这不是对参考关节角做简单行为克隆：参考提供奖励与动作引导，策略仍需通过物理交互学习如何完成动作。

## 3. 已有工作与当前研究问题

- **已搭建完整仿真流程：** 从示范接入、重定向、训练到物理评测和回放；已有推动、拉动、踢动、椅子操作等双机器人演示，并完成小桌操作实验。展示案例不等于跨场景稳健性已验证。
- **重定向侧的候选贡献：** 两阶段处理与掌面腕部优化已实现，部分同数据对照的姿态更合理；普适收益及对下游训练的独立贡献仍需系统消融。
- **学习侧发现的关键问题：** 物体轨迹接近目标，不一定意味着交互语义正确，例如用腿顶动代替用手搬运；同时可能出现一人主要操作、另一人只跟随参考的参与不均衡现象。
- **正在验证的改进：** 用参考中的身体—物体相对关系和接触时序引导学习，探索能否减少操作方式偏离。桶交接正在开展对应实验；参与不均衡目前主要作为机制分析问题，尚未证明解决，也不强制两人均分出力。

论文的整体价值是将多来源示范与机器人协作学习连接起来，并研究“动作跟踪、物体运动与有效协作并不等价”这一实际问题。两阶段处理、腕部优化及交互关系奖励是支撑这条主线的方法模块，而不是相互独立的故事。

## 4. 对外介绍的边界

当前平台为固定橡胶手的双 G1，主要结果来自仿真。固定手型是硬件设定及接触几何适配的动机，不是文章主线，也不意味着能够替代依赖主动抓握的灵巧操作。暂不声称已有实机验证、视频重建、大规模自动数据筛选、任意机器人数量泛化或稳定的自主角色涌现。论文仍处于实验与整理阶段，投稿、接收状态应按实际填写。

## 5. 可用于英文套磁信的简短介绍

I am working on a simulation-based framework for learning cooperative humanoid loco-manipulation from human–object interaction demonstrations. My work connects single-person and two-person motion-capture data with interaction-aware retargeting and reference-guided multi-agent reinforcement learning. I have implemented a two-stage retargeting pipeline and palm-orientation-based wrist refinement, and developed dual-G1 simulation demonstrations involving pushing, pulling, kicking, and chair manipulation. I am currently investigating a key challenge: achieving the desired object motion does not necessarily preserve the demonstrated interaction pattern or ensure meaningful participation from both robots. My ongoing experiments explore geometry- and contact-informed rewards to address this gap.

---

内部依据：`meeting_93.docx` 的研究框架讨论、[当前主路线图](MULTI_AGENT_EMERGENCE_ROADMAP.md)、[实验台账](DEMO_AND_EXPERIMENT_INVENTORY.md)。以后续已确认的研究范围为准；文中的改进目标不等同于已经验证的结论。
