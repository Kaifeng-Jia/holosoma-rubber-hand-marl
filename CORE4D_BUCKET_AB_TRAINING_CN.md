# CORE4D 桶交接 A/B：本地与云端并行实验

日期：2026-09-18。工作树 `/home/kevin/holosoma-core4d-base`，分支 `core4d-base`。
这是主路线图的执行说明，不另立研究路线。**截至2026-09-19 UTC，本地A、云端B均已完成
12000轮，云端最终模型已接收并在本地统一评测。** 首轮各5回合到达参考末尾A为2/5、B为1/5；
B接触时段覆盖提高，不等于已解决交接或整体优于A。结果及回放见
`logs/Core4DBucket/comparison_20260919/RESULTS_CN.md`。原启动记录仍保留。

后续用户已观看并认可A/B作为Demo，观察到交接时额外的手部支撑；这是定性认可与待分析
现象，不改变上述完整评测，也不自动授权续训。2026-09-19用户授权准备并启动两组新消融：
本地`A-no-rel`、云端`A-no-height`，具体命令与进度见第9节。OMOMO**单人**先等待两组结果，
再选择最终要迁移的奖励组合，不预定只能加关系项。总体待办统一见
[`DEMO_AND_EXPERIMENT_INVENTORY.md`第6节](DEMO_AND_EXPERIMENT_INVENTORY.md#6-接下来要解决的事情与真实进度)。
本文继续描述已完成的A/B，不把新消融混入既有冻结配置。

第1–8节为已完成A/B的冻结记录；新消融仅按第9节启动，勿重复执行旧A/B命令。

## 1. 目标与唯一对照差异

使用 CORE4D `20231020/071`，学习一人拿起桶并交给另一人。复用两阶段及 A1 wrist-only
参考；不新增单人 WBT，不改变 Actor 观测，不要求两人始终同时接触或各承担一半负载。
参考腕姿态可以包含几何穿透；实际训练中的橡胶手、机器人与桶均进行物理碰撞。

- A 在本地：原身体跟踪与正则 + 物体位置/朝向 + 独立高度 + 新相对向量关系。
- B 在云端：与 A 完全相同，仅对上述物体/关系正奖励乘软接触因子。
- A/B 均启用同一个轻量手—桶传感器；A 的新增传感器只用于诊断，不调制奖励块。
  原有不期望身体接触正则仍保留，不能将A理解为完全没有接触相关奖励。
- 相同代码、参考、资产、采样批量、种子及预算。不同 GPU 不保证逐位确定性；一个训练
  seed 的对照是探索性证据，不冒充多 seed 稳健性结论。

## 2. 冻结配置

| 项目 | 两组共同设置 |
| --- | --- |
| 机器人 | 两台 G1，固定橡胶手，每台 29 DoF |
| Actor | 共享参数；158 = 自身/参考154 + 队友平面相对位置速度4；输出29 |
| Critic | 全局527维；评测仅使用 Actor 和其归一化器 |
| 网络 | 隐层512/256/128，ELU；不额外增加接触/物体 Actor 输入 |
| 初始化 | 两组从零，seed721；不加载 smoke 或旧 WBT/MARL 权重 |
| 桶 | 原尺寸、总质量1 kg；这是实验设定，非数据集实测质量 |
| 碰撞 | 源mesh + Isaac convex decomposition；不是无条件保证保留所有空心开口 |
| 材料 | 静/动摩擦0.5/0.5，恢复系数0 |
| 物理/控制 | 200/50 Hz，PD每5 ms读取实时关节状态 |
| 参考 | 497帧/50 Hz，最后采样9.92秒；原299帧/30 Hz，不循环 |
| PPO | actor/critic初始LR均0.001；actor按KL自适应，critic独立；clip0.2，gamma0.99，lambda0.95，entropy0.005 |
| 正式预算 | 2048环境 × 每轮24步 × 12000 iterations |
| 保存 | 每2000 iterations，另保存初始和最终完整checkpoint |

不要因云端显存更大就改成4096环境；不要自行改摩擦/质量/termination、加入隐藏辅助力。
若云端无法容纳相同采样配置，报告后讨论，不把不同批量结果称作严格A/B。

## 3. 奖励公式及尺度

记 `Rbody` 为原六项身体跟踪，`Preg` 为原动作变化、限位、不期望身体接触正则：

```text
rp   = exp(-(ex² + ey² + 2ez²) / 0.3²)
rrot = exp(-angle_error² / 0.4²)
rz   = exp(-ez² / 0.10²)
rrel = exp(-mean_agent(Erel) / 0.04²)
I    = rp + rrot + rz + 2*rrel
RA   = Rbody + I + Preg
RB   = Rbody + g*I + Preg
```

距离为米、角度为弧度。RewardManager统一乘dt=0.02一次。
新的向量关系项替换旧Laplacian项，不同时叠加；独立高度使用上述有界正奖励，
不是小桌的 `-(ez/0.05)²`。现有termination不因接触缺失或短期学习表现而收紧。

每人19点、物体请求100点实际89点，seed42，固定物体局部对应。实际/参考身体到物体点
的world向量作差；两侧各按 `prior/max(distance²,0.1²)` 归一化后平均。
每个agent独立归一化；每手3点各分配1/3基础权重，避免仅因多采点而三倍加权。
0.04 m是聚合关系误差尺度，不是允许的手—桶距离。0.10 m是距离权重下限，不是接触阈值。

接触部分：

```text
alpha_a = clip((0.04 - min(source_left_palm_distance, source_right_palm_distance))/0.02, 0, 1)
c_a     = 任一橡胶手与桶的当前法向接触力模长 > 1 N
Ec      = sum_a(alpha_a * (1-c_a)) / max(1, sum_a(alpha_a))
g       = 0.5 + 0.5*exp(-2*Ec)
```

分母至少1是本轮明确的数值细化：避免仅有0.01置信度也被归一化成完整接触要求。
两人满置信度时仍取缺失比例；无参考接触时g=1；额外接触不被这一项惩罚。
不奖励接触力越大越好，不把“碰到”解释成“承担重量”。传感器是法向力，**不包含切向摩擦力**。

### 参考接触不是力真值

原数据没有提供接触力标签。掌部点由源SMPL-X wrist/index1/middle1/pinky1的中心估计，
到源桶三角面计算无符号最近距离，再将15 Hz距离插值到50 Hz。不是从A1穿模反推接触。
2–4 cm由源距离与交接时序选择为首轮技术参数；用户表示不熟悉该参数，因此不能写成
用户确认了最优阈值。它是几何代理，正式开跑前展示时间轴并说明局限。

| 人物 | 接触置信度非零 | 满置信度 |
| --- | --- | --- |
| 人1 | 3.30–5.56秒 | 3.62–5.50秒 |
| 人2 | 4.44–9.32秒 | 4.50–9.28秒 |

重叠期符合交接概念，但不证明真实承重或夹持一定可行。

## 4. 代码与资产位置

- `scripts/prepare_core4d_bucket_training.py`：离线导出、源接触代理和资产哈希；不运行物理。
- `config_values/marl/g1/core4d_pair_experiments.py`：新增bucket003，不替换小桌/椅子默认设置。
- `config_values/marl/g1/core4d_bucket_reward.py`：A/B配置；只替换原两个物体奖励为共同交互块。
- `managers/reward/terms/core4d_bucket.py`：真实状态向量评分、软接触因子、逐轮诊断。
- `config_values/marl/g1/core4d_bucket_contract.py`：冻结公式/尺度/参考/机器人哈希，恢复不能混用A/B。
- `envs/marl/core4d_bucket_manager.py`：桶专用传感器reset，不改变Actor观察。
- `config_types/simulator.py`、`simulator/isaacsim/isaacsim.py`：新增默认关闭的轻量手接触开关；
  保留旧完整诊断模式，桶不收集每个接触点的位置。
- 现有train/smoke/evaluate入口：显式bucket选项；记录独立奖励版本与共同评测口径。

以上包路径相对于 `src/holosoma/holosoma/`。桶数据目录：
`src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/`。
不会把旧Stage2机器人点位缓存当成A1参考；两者只共用已冻结的物体表面采样。

关键身份：

```text
A1 pair: ea1e73c80d98ac15602871a543bf1aa6ca221a7039fc8ba98b25b3b09cc094af
runtime: 071ac78bac440fee72a2cf861e1c00ba3df9531b371292d930e04d61aa8b23a8
vectors: 3de13d54c9aca213e3a41aac67e4d4d01d2f53b3d77e458faceb1fc461fb358f
URDF:    6c0ff9f60d8ef7ef057a988f4e95a6b89cd7e0f5ae55ea9e6697596955f28cde
```

## 5. 验证记录

- 离线：497@50Hz、89固定样本；固定链映射误差5.09e-16m，源文件未覆盖。
- CPU：最终桶/旧图奖励、checkpoint、评测及入口回归182项全部通过（3.31秒）。
- 真实训练入口：A/B分别完成2048环境×24步×2轮，产生完整checkpoint与passed状态。
  核对两组初始Actor/Critic参数逐值相同，更新后均发生变化；优化器与归一化器已保存。
  这只是更新/保存与显存适配检查，不是完整交接成功证据。
- B完整恢复：第2轮checkpoint恢复后的模型、优化器、归一化器及元数据逐字段一致，
  再运行1轮成功保存第3轮，模型参数继续更新且全部有限。正式训练不使用这些权重。
- Actor-only评测：短检查模型的1回合录制完成，保存实际状态、手—桶法向力、
  接触有效标志、参考帧和episode步数；回放数组行号不能直接代替奖励计算的参考phase。
- 物理：A起始帧、B第230帧，各2环境24步通过；1kg与全部材料0.5/0.5/0已核对；
  cooked USD为convexDecomposition。B接触因子均值约0.868，非恒为1。
- 初次重型接触诊断曾出现接触点buffer不足/CUDA索引越界，改用所需的四路法向力读取后
  重测通过。不以此认为动作不可学，也没有改变手的碰撞或训练目标。
- 早期B诊断指定帧被默认reset覆盖的脚本问题已修复，旧`smoke_B_frame230.json`不能用于
  交接阶段结论；应使用`smoke_B_frame230_v2.json`。
- 所有日志与短检查输出：`logs/Core4DBucket/preparation_20260918/`。这些不是正式训练结果。
- 判定运行通过需同时看报告、实际产物和错误日志。Isaac快速关闭可能掩盖Python的退出码，
  **不能只凭shell退出0判断成功**。没有重新安装或修复环境依赖。

## 6. 已完成训练的命令记录——勿重复运行

在工作树根目录，使用已经成功运行Isaac的Python；下面的`python`应替换为该解释器。
本地解释器是 `/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python`。

本地A：

```bash
python -B scripts/train_core4d_smalltable.py \
  --experiment bucket003 --reward-variant bucket_A \
  --num-envs 2048 --steps-per-env 24 --iterations 12000 \
  --save-interval 2000 --seed 721 \
  --output-dir logs/Core4DBucket/A_fresh12000_env2048_seed721
```

云端B：

```bash
python -B scripts/train_core4d_smalltable.py \
  --experiment bucket003 --reward-variant bucket_B \
  --num-envs 2048 --steps-per-env 24 --iterations 12000 \
  --save-interval 2000 --seed 721 \
  --output-dir logs/Core4DBucket/B_fresh12000_env2048_seed721
```

建议将控制台输出保存到对应run外的独立日志。使用持久化存储及时备份每个完整checkpoint、
`run_config.json`、`metrics.jsonl`和`status.json`；云端runtime本地磁盘不是可靠的长期备份。
保存为临时文件后原子更名的逻辑已存在；同步后核验SHA256。

中断恢复：`--resume`指定同组checkpoint，`--iterations`是**追加轮数**，不是总目标。
例如从model_08000.pt恢复到12000，应传4000，并选择新的输出目录。不得拿A checkpoint续B，
不得用这次烟测checkpoint作为正式初始化。云端结束后带回完整checkpoint，不只导出Actor。

## 7. 发给云端/Mac执行端的说明

交接包输出位置：
`/home/kevin/CORE4D_BUCKET_AB_TRANSFER_20260918/core4d_bucket_transfer.tar.gz`；
同目录`archive.sha256`为压缩包校验，包内`SHA256SUMS`为逐文件校验。
通过`scripts/package_core4d_bucket_transfer.py --include-reference-robot`冻结本地当前文件，
同时包含训练/参考机器人mesh，不含转换缓存。打包不等于已传给云端或已push GitHub。
包内手册和资产清单保留打包时“未授权/未启动”状态；本地后续授权与启动单独记在上述
启动记录，不修改冻结资产或代码，因此不影响A/B一致性及checkpoint恢复校验。

优先复用上次成功部署的仿真环境，不重新进行依赖升级或改物理后端。
本轮源码尚有工作区改动，**只git pull旧HEAD不够**：使用附带SHA清单的冻结源码/资产包，
覆盖到同base HEAD的干净checkout，先核验清单。不要覆盖云端尚未保存的用户修改。
交接包不含环境、数据集全量、日志或checkpoint，不上传token或其他凭据。

```text
本轮只运行CORE4D桶交接B，A在Ubuntu本地。阅读CORE4D_BUCKET_AB_TRAINING_CN.md。
使用交接包所记录的base HEAD与payload文件哈希，保持同一源码和桶资产。
复用已验证的Isaac环境；先做bucket_B少量环境smoke，确认JSON passed和日志、实际输出。
正式配置固定：1kg，摩擦0.5/0.5，Actor158共享，Critic527，2048env×24步×12000轮，
seed721，每2000保存。从零开始，不导入smoke权重。只有用户授权开始正式训练后才执行。
不要加环境数、修改接触标签/奖励或自行选其他参考。有问题先说明具体证据。
运行结束交付完整checkpoint、run_config、metrics、status、控制台日志、SHA256及中断记录。
```

## 8. 本地统一评测

两组都用同一版本评测脚本、相同评测种子及每种子相同次数：

```bash
python -B scripts/evaluate_core4d_smalltable.py \
  --experiment bucket003 --checkpoint /实际路径/model_12000.pt \
  --episodes 5 --seed 721 --output-dir /新的评测目录
```

评测`reward_sum`为共同旧tracking口径，排除新增高度/向量/软接触；不是两组各自训练reward。
同时看完整执行率、物体3D/高度误差、持续离地与交接、身体稳定性及手部接触。
额外保存手—桶法向力、参考帧和episode步数。法向力不是总承重；不能用它单独证明摩擦支撑，
也不能把物体原点高度当作整个桶离地高度。短失败前缀与完整轨迹要分别统计。

## 9. 新消融：本地去关系、云端去独立高度（2026-09-19授权）

用户确认先完成这两组，再根据结果决定OMOMO单人A1究竟加什么。
OMOMO现在不训练、不固定为“只加关系项”，也不改参考。

| 本轮运行 | CLI选项 | 位置/朝向/独立高度/关系权重 | 软接触调制 |
|---|---|---|---|
| 本地 A-no-rel | `bucket_A_no_rel` | 1 / 1 / 1 / 0 | 无 |
| 云端 A-no-height | `bucket_A_no_height` | 1 / 1 / 0 / 2 | 无 |

两组都以原A为比较基准，不是在B上删项。共同保留物体位置中的wz=2，独立高度的
sigma=0.10m、关系sigma=0.04m定义不变；被去掉的项仍计算诊断，但对总奖励贡献为零。
其他9项、参考、固定点位、轻量接触传感器、网络、重置、终止、优化器设置及物理完全沿用A。
因此原A对两组分别回答“关系项的条件作用”和“独立高度项的条件作用”，不是整个配方
相对于纯基础奖励的比较。原A/B契约和已有checkpoint不改、不覆盖。

共同正式配置：fresh seed721、2048env×24步×12000轮、每2000保存；共享Actor158→29、
全局Critic527→1、512/256/128 ELU；桶1kg、静/动摩擦0.5/0.5、恢复系数0；physics200Hz、
control50Hz、实时PD不变。训练从零，不从12000的A/B或本轮短检查权重开始。

### 本地命令

在`/home/kevin/holosoma-core4d-base`、原hssim解释器运行：

```bash
/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python -B -u scripts/train_core4d_smalltable.py \
  --experiment bucket003 --reward-variant bucket_A_no_rel \
  --num-envs 2048 --steps-per-env 24 --iterations 12000 --save-interval 2000 --seed 721 \
  --output-dir logs/Core4DBucket/A_no_rel_fresh12000_env2048_seed721_20260919
```

### 云端同步与命令（本次由Ubuntu直接连接执行）

本轮用户已授权云端训练A-no-height，并选择由Ubuntu直接连接启动，连接信息待用户提供。
下述冻结包仍用于两端一致性；不需要Mac中转。打包不等于云端已启动，远端沿用之前成功的
仿真环境，不升级依赖、不重配物理、不启动旧B；缺失环境或需要安装时先报告再讨论。

本次冻结包输出目录：`/home/kevin/CORE4D_BUCKET_ABLATIONS_TRANSFER_20260919/`，
包含`core4d_bucket_transfer.tar.gz`、`archive.sha256`、`package_manifest.json`和`SHA256SUMS`。
不修改09-18旧包，不上传checkpoint/日志/完整数据集或环境。

1. 收到本次新`core4d_bucket_transfer.tar.gz`及`archive.sha256`，先校验压缩包。
   旧09-18包没有新消融代码，不能用旧包或只git pull来代替。
2. 解压到新目录，执行`sha256sum -c SHA256SUMS`，核对`package_manifest.json`的
   `base_git_head`、required_base_files和required_absent_paths。使用匹配HEAD的干净研究
   源码副本再叠加`payload/`；不要覆盖旧运行目录或未保存的云端改动。无需复制Python环境。
   使用该新副本的scripts入口，其显式sys.path会选中同副本源码；不要从旧研究目录启动。
3. 复用实际能运行Isaac的解释器，设置`PYTHONDONTWRITEBYTECODE=1`、`PYTHONUNBUFFERED=1`、
   `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`CUDA_VISIBLE_DEVICES=0`。
   确认单GPU当前空闲、持久化盘足够。云端若需新开付费实例或资源规格变化，先询问用户。
4. 可用新空目录做`bucket_A_no_height`两轮2048环境短检查，核对配置/保存与有限指标；
   不用短检查评判动作质量。随后从零启动下方完整命令，输出到新的持久化目录。
5. 不自动续训或改系数。断点恢复仅限同一个A_no_height，`--iterations`仍表示追加轮数，
   每个恢复段使用新目录；不得拿A/B或A_no_rel权重恢复。

```bash
python -B -u scripts/train_core4d_smalltable.py \
  --experiment bucket003 --reward-variant bucket_A_no_height \
  --num-envs 2048 --steps-per-env 24 --iterations 12000 --save-interval 2000 --seed 721 \
  --output-dir logs/Core4DBucket/A_no_height_fresh12000_env2048_seed721_cloud_20260919
```

`python`必须替换为云端已验证环境的解释器。启动后核对run_config的variant、
`bucket_reward_contract.block_weights_position_rotation_height_relation=[1,1,0,2]`、
fresh/resume=None、参考/资产哈希、158/527维、2048×24、12000轮及2000保存间隔。
桶专用契约的`position_error_weights_xyz=[1,1,2]`才是实际位置权重；顶层旧通用字段
`object_z_error_weight=1`不控制桶奖励，不要为了它而修改实验。

记录真实子进程退出码及完整启动命令，并同时检查最终模型、连续12000条指标与status；
不要用不存在systemd服务的默认值补退出码。返还最终完整checkpoint、run_config、metrics、
status、stdout、进程记录、源码manifest/SHA及任何恢复段说明。中间模型暂留云端持久盘，
不要自行删除。评测沿第8节在Ubuntu进行，使用共同口径而不是各组训练总reward排名。

### 验证与启动状态

两组代码/入口及旧实验回归284项CPU测试通过。两组各自2048环境×24步×2轮真实更新完成，
真实退出码均0，model_00002落盘/CPU加载验证通过，指标有限、gate=1。两组初始Actor/Critic、
优化器及归一化器逐值相同，正式运行不使用这批短检查权重。原A/B的12000模型实际读取验证
仍通过，四种variant之间恢复会被拒绝。此处验证程序正确性，不评价早期学习质量。
正式启动证据另记于`logs/Core4DBucket/ablations_20260919_launch/`，原A/B身份和资产不变。
未收到云端实际启动命令、PID或run_config前，状态只能写“交接准备/待接收方启动”。

## 10. 用户批准的双删除对照（2026-09-19）

本节为新增实验；前述A/B和单项消融的历史契约不修改。用户已明确批准云端继续训练
`A_no_rel_no_height`，不是继续训练已完成的去高度checkpoint。

- 从零12000轮、2048环境×24步、seed721、每2000轮保存。
- 共享Actor158→29、全局Critic527→1；512/256/128 ELU，PPO配置沿用A。
- 相同参考/机器人/桶资产；桶1kg、静/动摩擦0.5/0.5、恢复系数0、200/50Hz和实时PD。
- 仅删除独立高度和相对向量奖励：块权重`[1,1,0,0]`，位置误差xyz权重仍`[1,1,2]`。
  不启用B接触调制，gate始终1；原身体跟踪与正则不变。保留两项诊断和接触传感器。
- 它不是位置wz1的椅子baseline，而是固定wz2下的两因素空白对照。
- 新variant恢复身份独立，不加载旧A/B、单删除模型或短检查模型；不更改现有训练。
- 新云端研究快照`/workspace/core4d_bucket_no_rel_no_height_20260919`；旧云端源码和结果不覆盖。
- 冻结包`/home/kevin/CORE4D_BUCKET_BOTH_REMOVED_TRANSFER_20260919/`。
- 云端沿用`/workspace/core4d_deps/hssim/bin/python`及刚完成12000轮的环境，不安装/升级依赖。
  先做2048环境×2轮更新/保存检查，通过后正式从零训练；短检查不评判动作质量。
- 操作记录存入`logs/Core4DBucket/both_removed_20260919_launch/`；评测仍在本地进行。

```bash
/workspace/core4d_deps/hssim/bin/python -B -u scripts/train_core4d_smalltable.py \
  --experiment bucket003 --reward-variant bucket_A_no_rel_no_height \
  --num-envs 2048 --steps-per-env 24 --iterations 12000 --save-interval 2000 --seed 721 \
  --output-dir logs/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919
```

CPU回归258项已通过，覆盖五种variant契约/交叉恢复、奖励算术、dt、入口和其他场景兼容。
实际云端启动与结果以后续独立进程记录为准。该补充对照不设为OMOMO准备的前置门槛；
OMOMO仍待此前两项单删除的共同评测后选择配方，不进行大规模多种子训练。
