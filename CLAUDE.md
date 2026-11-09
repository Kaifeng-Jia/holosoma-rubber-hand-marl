# DuetMimic System 1 对比实验：Ubuntu 执行方常驻规则

本文件是常驻规则。具体任务写在 `docs/system1/T*.md`，按 Kevin 转交的顺序执行。任务书和本文件冲突时以任务书为准；任务书没写到的按本文件。

## 分工

- 你在 Kevin 的 Ubuntu 机器上，负责写代码、跑训练和评测、交结果。
- 任务书由云端的 Claude 起草，Kevin 确认后放进 `docs/system1/`。Kevin 是决策人。
- 用中文汇报。时间一律写 America/New_York。

## 项目背景

- **论文：** DuetMimic，两台 Unitree G1（29 自由度）从人类示范学习合作移动操作。
  - 参考来自两类示范：单人示范用角色等距补出同伴；双人示范用共享场景把两人各自缩放的物体合并成一个。
  - 执行用 MAPPO：共享 actor 加中心化 critic。actor 输入 158 维（不含物体状态），输出 29 维关节位置目标给 PD。策略 50 Hz，仿真 200 Hz。
- **六个任务：** Table Pushing、Table Pulling、Table Kicking（物体 20 kg）；Chair Righting（5 kg）；Bin Transport（1 kg，一台举起交给另一台）；Table Transport（5 kg）。
- **新方向：** 把论文放进"快慢脑"框架，回答"面向交互的多机器人快系统（System 1）该怎么搭"。用同一份参考比较四种执行方式：
  - A 静态目标 + SONIC：参考直接喂 SONIC，不训练，开环。
  - B 动态目标 + SONIC：按物体实际状态在线修改目标，不训练。
  - C MARL + SONIC：分层，高层学习，SONIC 冻结执行。
  - D 端到端 MARL：现有 DuetMimic。
  - 总体设计见 `docs/system1/README_zh.md`。
- **SONIC：** NVIDIA GEAR-SONIC 全身运动跟踪模型。
  - 仓库 `NVlabs/GR00T-WholeBodyControl`，文档 https://nvlabs.github.io/GR00T-WholeBodyControl/，模型 https://huggingface.co/nvidia/GEAR-SONIC。
  - 代码是 Apache 2.0；权重是 NVIDIA Open Model License，不提交、不转发。

## 隔离（最重要）

- **只在本目录写代码和输出。** 本目录是新克隆的 `system1-sonic` 分支。
  - 运行输出放 `logs_system1/`，不进 git。
  - 返包里的小文件放 `results_system1/<任务号>/`，提交并 push，云端 Claude 直接从 GitHub 读。小文件包括 README_CN.md、CSV、配置、diff、SHA256SUMS，单个文件不超过 5 MB。
  - 视频和 checkpoint 只留本地 `exports_system1/`，不进 git。
- **以下目录只读：**
  - `/home/kevin/holosoma-core4d-base`
  - `/home/kevin/holosoma-rubber-hand-marl`
  - `/home/kevin/holosoma-rubber-hand-largetable`
  - 以及它们的 `logs/`、`paper_handoff_exports/`。
  - 可以读其中的参考、checkpoint、评测状态和脚本。不得写入、改文件，也不得在里面切分支、git pull、commit 或 stash。
- **旧目录里未提交的代码改动**（例如 `scripts/evaluate_paper_supplement.py`），按 T01 第 0 步做成快照，提交到本分支，然后只改本分支的副本。
- **旧的 conda 环境（`hssim`、`hsretargeting`）不改、不升级。**
  - SONIC 自己的评测用新环境。
  - 需要在 holosoma 仿真里加载 SONIC 时，用 `conda create --clone hssim -n hssim_s1` 复制一份，只在副本里装依赖。
- **不碰不是你启动的进程：** 不 kill、不 renice。开训前先看 `nvidia-smi`；需要和正在跑的训练共用 GPU 时，先回报。
- **不删除任何文件。** 确实需要清理的，移到本目录 `_to_delete/`，并在汇报里列出。
- **GPU：** 本轮所需的 GPU 由 Kevin 提供，用 Kevin 指定的机器。不自己开新的付费云资源，也不自己续租。

## Git

- 只提交和 push 到 `system1-sonic` 分支。
  - 不 push 到 `core4d-base`、`main` 或其他分支。
  - 开工前先 `git pull`，拿到云端新发的任务书。
  - 不 force-push，不改写历史。
- 不提交大文件：checkpoint、视频、SONIC 权重、仿真缓存。
- 每完成一个能用的阶段提交一次，提交信息写清楚做了什么。

## 数据诚信

- **只报真实运行得到的数字。** 没跑出来就写"未完成"和原因，不用占位数、估计值或别的运行结果代替。
- **不挑种子、不挑 checkpoint、不挑回合。** 剔除只按任务书事先写明的规则。所有种子、所有回合都进交付表。
- **评测初始状态用冻结文件，** 记录 sha256。
- **每个训练或评测都要保存：**
  - 完整命令和 run_config，以及它与对照配置的 diff；
  - 代码 git hash，以及未提交改动的 diff；
  - 所用 checkpoint、参考、URDF 的 sha256。

## 对比的公平性

- **四种方案共用同一套设置：** 机器人模型、物体（质量、尺寸、摩擦）、参考、初始状态集、终止条件和评测指标。
- **低层控制按各方案原样：**
  - A–C 用 SONIC 自己的 PD 增益、动作缩放和控制频率；
  - D 用现有 DuetMimic 的设置；
  - 两套设置都要记录。
- **任何一项不得不不同时**（例如 SONIC 只能用它自己的 G1 模型），先停下回报，不自行决定。

## 节奏

- Kevin 要求越快越好。任务书里写了"通过后直接继续"的地方，不等回复，直接往下做。
- GPU 空着就是浪费：不依赖 SONIC 的训练（例如 T04 的 D 部分）可以先开。
- 只有在任务书写明的停止条件，或下一节列出的情况下，才停下来等回复。

## 什么时候停下来问

- 需要同意许可证或条款，例如 HuggingFace 上的模型许可：请 Kevin 本人操作。
- 需要付费资源、删除文件，或动别人的进程。
- 要改评测协议、机器人或物体模型、奖励。
- 结果和预期差很多：不要改设置去"修"结果，先如实回报。

## 汇报格式

- **每个任务开始时交 receipt：**
  - 任务书文件名和 sha256、开始时间；
  - hostname、GPU 型号和空闲显存；
  - 本目录 git HEAD、所用 Python 环境。
- **每个阶段结束交一个返包：**
  - `README_CN.md`：只写事实，包括做了什么、用时、和任务书有没有偏离。不解释结论。
  - 结果 CSV、配置与 diff、`SHA256SUMS`。
- **结果表字段：** 沿用 V3 收尾包 `results/episodes.csv` 的口径；新增字段另外写说明。

## 代码习惯

- 环境、奖励、终止、评测尽量复用现有 holosoma 代码路径，不重写。新代码放在 `system1/` 下。
- 新功能加开关；开关关闭时，旧行为不变。
- 每个新组件都写 smoke test。
