# CORE4D 小桌高度跟踪：Colab 训练 / Ubuntu 评测交接

更新时间：2026-09-14。本文是部署交接，不是第二份路线图；研究方向以
`MULTI_AGENT_EMERGENCE_ROADMAP.md` 最新条目为准。

## 1. 本轮要做什么

- 用户希望释放 Ubuntu 笔记本供日常使用：**Colab 只训练，Ubuntu 做物理评测与 ViSER 回放**。
- Mac 上 Codex 负责阅读代码、准备 Colab notebook、协助用户连接并操作真实 Colab runtime。
  Mac 不是训练机器；不能在 Mac 上安装并运行 Isaac Sim 来冒充云训练。
- 仓库为 `https://github.com/Kaifeng-Jia/holosoma-rubber-hand-marl`，分支 `core4d-base`，私有。
  不推 upstream `amazon-far/holosoma`，不切换其他工作树，不更改仓库可见性。
- 当前实验：CORE4D 小桌搬运。保留当前交互图，只提高竖直误差权重。
  椅子、OMOMO 新腕部预览、Demo 3/4 均不在本次执行范围。

## 2. 已确定的实验，不要重新设计

物体位置奖励为

```text
r_position = exp(-(dx**2 + dy**2 + wz * dz**2) / 0.3**2)
```

`wz=1` 是旧对照，`wz=2` 是新候选。这里是误差平方权重，不是拆成
`r_xy + 2*r_z`，也不是物体位置奖励总权重乘二。
目标是逐帧跟踪参考高度，不是越高越好。身体参考、交互图、姿态奖励、终止条件不变。
不增加接触奖励、attention、信用分配网络，不提高质量，不用灵巧手或半球手。

| 项目 | 配置 |
| --- | --- |
| 机器人 | 两台 rubber-hand G1，每台 29 DoF |
| Actor | 两台共享参数；每台输入 158 = 自身/参考 154 + 队友平面位置速度 4；输出 29 |
| Critic | centralized team Critic，输入 527；不用于本地 actor-only 推理 |
| 网络 | Actor / Critic 隐层 512、256、128，ELU；沿用当前代码 |
| 算法 | reference-guided MAPPO/PPO，不是行为克隆 |
| 参考 | `core4d_pair_runtime_fps50.npz`，687 帧，50 Hz |
| 物体 | 小桌，20 kg，5 个 box 碰撞体 |
| 材料 | 静/动摩擦 0.5/0.5，恢复系数 0 |
| 频率 | 物理 200 Hz，控制 50 Hz；保持双机器人每个物理子步刷新 PD 状态 |
| 交互图 | `interaction_mesh_100_v1.npz`，权重 1，尺度 0.06 m，不重新采样 |
| 正式预算 | 2048 environments × 每轮采样 24 步 × 12000 iterations |
| 保存 | 每 2000 iterations，另保存初始和最终 checkpoint |
| 初始化 | 第一轮从零开始，seed 721；不加载历史 WBT / MARL 权重 |

更换 GPU 不自动授权改变 env 数、采样长度、奖励、控制频率或训练预算。
不要根据前几百轮成绩判断方法失败；烟测只检查安装、资产、维度、数值与保存/恢复是否工作。
本次先启动一个 `wz=2, seed=721` 正式 run，后续多种子对照按用户确认推进，不自动耗尽额度。

## 3. 先检查 Colab，不要先安装一大堆东西

用户是学生 Colab **Pro**，不是已确认的 Pro+。不能假设可以连续后台跑 24 小时。

先在真实 Colab GPU runtime 执行：

```python
!nvidia-smi
!cat /etc/os-release
!ldd --version
!df -h /content
!free -h
```

检查 GPU 型号、驱动、系统、内存和可用磁盘。Isaac Sim 5.1 官方不支持没有 RT Core 的
A100 / H100。不能因为 PyTorch 能用 CUDA 就说本项目可运行，也不能把 `headless=True`
当作已验证的兼容性豁免。其他 GPU 仍需要真实的 Isaac Sim 启动检查。

如果资源不适配，说明具体问题并停在这里与用户讨论；不要切模拟器、修改物理方法、升级
订阅、购买云服务器、反复重连刷卡或设置绕过 Colab 回收限制的保活脚本。

官方依据（实际执行时可复查）：

- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>
- <https://research.google.com/colaboratory/faq.html>
- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>

## 4. 安装与数据准备

先记录实际下载的 git commit。使用 `--single-branch --branch core4d-base --depth 1` 获取源码，
不拉所有分支历史。GitHub 登录、Colab 登录和 Google Drive 授权由用户完成，或使用当前客户端
明确提供且已经授权的能力；不得假定 Codex 自动拥有这些账号会话。

私有仓库认证：使用 GitHub 已授权连接、credential helper 或 Colab Secrets；不把 token 写入
仓库、notebook 明文、clone URL、日志或最终回复。若浏览器自动化不可用，请生成 notebook，
请用户执行必要单元，再根据真实输出继续；不要声称尚未执行的单元已经成功。

本地可工作的版本：

```text
Python 3.11.15
Isaac Sim 5.1.0.0
IsaacLab checkout v2.3.0 (Python package metadata isaaclab=0.47.2)
torch 2.7.0+cu128
torchvision 0.22.0+cu128
numpy 1.26.0
warp-lang 1.10.0
```

`scripts/setup_isaacsim.sh` 是现有 Linux 安装来源。读取后再按 Colab 路径适配；不要修改宿主
驱动，不盲目使用最新 Isaac Sim/IsaacLab。Colab 内核的默认 Python 不一定是 3.11，可另建
Python 3.11 环境并通过该环境的 python 启动脚本。

注意训练与重定向是两个环境：不要在训练环境完整安装 `src/holosoma_retargeting` 的依赖，
其 `numpy==2.3.5` 与训练的 NumPy 1.x 要求冲突。入口已把其源码路径加入 `sys.path`；
本次直接使用预生成数组和图缓存，不运行 retarget。训练依赖以 `src/holosoma` 和上述版本为准。

必要资产已在此分支中，无需下载整个 CORE4D 或 Ubuntu 上的 logs：

```text
src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/
src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_core4d_desk001_small_training.urdf
```

训练 robot 配置指向的 rubber-hand G1 URDF 及其 mesh 也已 tracked。小桌目录约 6.9 MiB，
该机器人 URDF 及 35 个 mesh 约 18.8 MiB。不要更改 manifest、参考或 mesh 来绕过哈希校验。
manifest 内的 `/home/kevin/...` 是来源记录，不是要求云端重建相同 home 路径。
椅子资产和原始数据本轮不上传；共享 descriptor 内存在 chair 选项不代表本次可直接训练它。

## 5. 正式训练命令

先完成少量环境的安装/加载检查，并用独立输出目录检查一次 checkpoint 保存和恢复。
烟测结束后正式 run 必须重新从零开始，不复用烟测权重。
在源码根目录，以下 `python` 必须是已安装上述依赖的 Python 3.11：

```bash
python -B scripts/train_core4d_smalltable.py \
  --experiment smalltable \
  --reward-variant interaction_mesh \
  --interaction-reference src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/interaction_mesh_100_v1.npz \
  --object-z-error-weight 2 \
  --num-envs 2048 \
  --steps-per-env 24 \
  --iterations 12000 \
  --save-interval 2000 \
  --seed 721 \
  --output-dir /content/core4d_runs/z2_seed721_segment01
```

GPU 显存不足时先报告，不静默减少 env 数。不要用 nohup 的存在证明 Colab 已能无人值守运行；
它不能延长 Colab runtime 的寿命。正式启动前简要报告 GPU、实测速度、持久化位置与配置。
`--output-dir` 必须为空或不存在：notebook不要提前在里面创建stdout、环境清单或说明文件，
否则训练入口会拒绝启动。训练stdout可放在旁边，例如
`/content/core4d_runs/z2_seed721_segment01.stdout.log`，再与该segment一起备份。

## 6. 持久化与断线续训：本轮必须做好

Colab 临时磁盘会随 runtime 回收而丢失。训练数据在 `/content` 读取；在开始正式训练前挂载
用户授权的 Google Drive 并确认可以写入指定的新实验目录。不要把模型上传 GitHub。

推荐 notebook 启动训练子进程，同时运行一个简短同步循环：

1. 训练代码先原子地生成完成的 `model_XXXXX.pt`；不复制写到一半的 `.tmp` 文件。
2. 每遇到一个新完成的 checkpoint，复制到 Drive 临时文件，校验 SHA256 后更名；保留旧文件。
3. 定期同步 `run_config.json`、`metrics.jsonl`、训练 stdout，完成后再同步 `status.json`。
4. runtime 回收后，以 Drive 中通过完整性检查的最近 checkpoint 续训；只能保住最近一次
   **完整同步到Drive**的进度，本地刚保存但尚未同步的文件也可能丢失。不要声称能逐 bit
   恢复物理世界和随机数状态。

若无法可靠部署同步循环，可以把 `--output-dir` 直接设为已挂载的 Drive 新目录，但要检查
远程 I/O 和中断写入；保存前后的校验仍有必要。不要将同步误称为延长 Colab 会话。

**`--iterations` 在恢复时表示“追加多少轮”，不是总目标轮数。**
例如已保存 `model_08000.pt`，目标仍是总计 12000，则用：

```bash
python -B scripts/train_core4d_smalltable.py \
  --experiment smalltable \
  --reward-variant interaction_mesh \
  --interaction-reference src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/interaction_mesh_100_v1.npz \
  --object-z-error-weight 2 \
  --num-envs 2048 --steps-per-env 24 --seed 721 \
  --iterations 4000 --save-interval 2000 \
  --resume /content/drive/MyDrive/core4d_height/z2_seed721_segment01/model_08000.pt \
  --output-dir /content/core4d_runs/z2_seed721_segment02
```

这里 Drive 路径只是示例，以用户实际选择的位置为准。每次恢复必须用新的空输出目录，
不得覆盖前一段；保持相同 git commit、参考哈希、reward 变体、`wz=2` 和采样配置。
不能把旧 `wz=1` checkpoint 当成新实验的中断点。

## 7. 交回 Ubuntu 的内容

- 完整最终 `model_12000.pt`，以及按约定保存的 2000 间隔 checkpoint。
- 所有 segment 的 `run_config.json`、`metrics.jsonl`、stdout、`status.json`。
- git commit、Python/包版本、GPU/驱动、完整启动/续训命令、文件 SHA256。
- 说明是否有 runtime 回收、续训次数、异常；区分“训练结束”和“评测通过”。

不要只导出 Actor 权重：完整 checkpoint 中的 Actor normalizer 是推理所需的，optimizer / Critic
等还是续训所需的。评测时仅使用 Actor 和相应 normalizer，Critic 不参与动作生成。

Ubuntu 本地评测沿用：

```bash
/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python -B \
  scripts/evaluate_core4d_smalltable.py \
  --experiment smalltable \
  --checkpoint /实际下载位置/model_12000.pt \
  --episodes 5 --seed 721 \
  --output-dir /新的本地评测目录
```

现有评测用于检查/预览，默认仍按旧基础 reward 打分，不等于新版训练总 reward。
现有固定初态多次回放不能当作独立随机初态评测；本周后续在 Ubuntu 另做高度/真实离地、
操作部位和分工分析。本次云端部署不增加接触传感器或改变 Actor 信息权限。

## 8. 给 Mac 上 Codex 的提示词

```text
请用中文协助我执行一次 CORE4D 小桌搬运训练。我是 Colab Pro 学生用户，希望训练只在
Colab NVIDIA GPU runtime 上运行，Mac 只用于操作，物理评测留给 Ubuntu。

请获取我的私有仓库 https://github.com/Kaifeng-Jia/holosoma-rubber-hand-marl 的
core4d-base 分支，首先完整阅读 COLAB_CORE4D_HEIGHT_HANDOFF_CN.md 和总路线图中
2026-09-14 的当前执行条目，记录实际 commit，检查是否存在后续修改。

先确认你能否实际操作我的 Colab。登录和授权请让我本人完成。如果你没有浏览器/会话
操作能力，生成可直接打开的 notebook，并只让我执行不可代办的单元；根据真实输出推进，
不要说尚未执行的步骤已经完成。

先检查 GPU/驱动/系统是否满足 Isaac Sim 5.1，尤其不要在 A100/H100 上盲目安装。
兼容才安装固定版本环境。若不兼容，请说明证据并和我商量，不要更换模拟器、改实验设计、
升级订阅、租用其他 GPU 或绕过 Colab 限制。

本次只做一个新实验：smalltable + 原 interaction_mesh + object-z-error-weight=2，
其公式是 exp(-(dx²+dy²+2dz²)/0.3²)，总位置奖励权重仍为1。两机器人共享158维Actor，
527维全局Critic，每人29维动作；20kg桌子，静/动摩擦0.5，物理200Hz/控制50Hz。
保留已有参考与所有其他奖励，不加接触、attention、信用分配或新观测。

从零训练，seed721，2048env，每轮24步，总计12000iterations，每2000保存。
先用独立目录做安装和保存/恢复烟测；随后给我一段简短启动配置说明，确认资源与备份就绪后
执行这一个正式run。不根据早期成绩自行停训或改设计；遇到技术错误或费用/资源选择再问我。

先配置Drive持久化，确保checkpoint完整同步。断线后从最新完整checkpoint恢复，追加轮数
为12000减去已有iter，使用新segment目录，wz与其他配置不变。每段日志和源码版本都要保留。
不要把token、原始完整CORE4D、训练checkpoint或日志push进GitHub。

结束后给我可下载的完整checkpoint与日志、SHA256、实际轮数、运行时中断/恢复记录。
不要在云端安排正式评测，不自动启动额外种子，不把训练完成说成语义问题已解决。
如需要改代码，先解释修改哪些文件、用途和原因；保护我其他分支、实验和既有改动。
```

本交接已按官方 OpenAI 文档对浏览器/账号授权能力作条件说明，不保证另一台设备上的
Codex 拥有本线程的工具或会话：<https://developers.openai.com/codex/app/browser>。

## 9. 本次交接的验证边界

Ubuntu 本轮只检查代码、数值公式、checkpoint兼容与保存/恢复，以及必要资产的存在和哈希；
没有启动本轮正式训练，也没有连接用户的Colab runtime。GitHub准备完成不代表Colab兼容性
已通过。云端GPU/驱动、完整依赖安装、真实仿真加载和Drive同步仍由接手端按本文验证。

干净源码中的重点CPU检查可运行（不启动Isaac Sim）：

```bash
PYTHONPATH=src/holosoma:src/holosoma_retargeting python -B -m pytest -q -p no:cacheprovider \
  src/holosoma/tests/agents/mappo/test_core4d_vertical_tracking.py \
  src/holosoma/tests/agents/mappo/test_core4d_smalltable_interaction_checkpoint.py \
  src/holosoma/tests/agents/mappo/test_core4d_pair_experiment_checkpoint.py \
  src/holosoma/tests/envs/marl/test_core4d_smalltable_config.py \
  src/holosoma/tests/envs/marl/test_core4d_smalltable_interaction_config.py \
  src/holosoma/tests/envs/marl/test_core4d_smalltable_asset.py
```
