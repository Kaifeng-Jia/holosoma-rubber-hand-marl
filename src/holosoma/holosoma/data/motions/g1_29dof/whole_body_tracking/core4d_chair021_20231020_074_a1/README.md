# CORE4D chair021 基础训练准备

源序列 `20231020/074`。用户认可8091所示共享缩小椅子A1腕部版本作为后续训练准备的参考。
2026-09-09已按用户指定5 kg生成独立物理资产并完成短程物理加载检查；2026-09-10用户确认
后已启动独立chair正式训练，配置与运行路径见末节。

## 已完成

- 源：`logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/shared_small_a1_preview_tol6/core4d_pair_reference.npz`。
- 源SHA256：`1e1999a179655abaf577f78fa6eeea5a7ecd4b1e8ae86a1144118b4dca54679b`。
- 源235帧30 Hz → 运行参考391帧50 Hz，首末采样时间差均7.8秒，不循环、不加速、无末尾裁切。
- 机器人位置和关节角线性插值，四元数SLERP；随后复用既有MuJoCo FK与速度计算。
  包含双机器人29关节的位置/速度、51个body的位姿/线角速度，及共享物体位姿/线角速度。
- 共同采样时刻与原参考一致；数值均有限；9项相关测试通过。原腕部参考未被修改。
- 本目录runtime SHA256：`9de64128209230229e40a675ed6b204e1c4b7655a81cbef773f76557a0350ba5`。
- 选定的源参考及原manifest另存为本目录 `core4d_pair_compact_fps30.npz`、`source_manifest.json`，
  字节不变；不改写原预览的 `training_ready=false`。
- 独立 `training_asset_manifest.json` 记录5 kg物理准备及实际加载证据，技术状态
  `training_ready=true`；2026-09-10用户确认开跑后另记录 `formal_training_authorized=true`。

复用现有脚本，通过显式参考哈希选择新数据；没有新增椅子专用转换算法。
脚本/底层部分命名保留smalltable历史名称，默认行为保持旧实验兼容，不代表复用桌子几何。

```bash
cd /home/kevin/holosoma-core4d-base
chair_py=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python
chair_source=logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/shared_small_a1_preview_tol6
"$chair_py" -B scripts/synthesize_core4d_smalltable_runtime_reference.py \
  --source "$chair_source/core4d_pair_reference.npz" \
  --source-manifest "$chair_source/manifest.json" \
  --expected-source-sha256 1e1999a179655abaf577f78fa6eeea5a7ecd4b1e8ae86a1144118b4dca54679b \
  --output src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_chair021_20231020_074_a1/core4d_pair_runtime_fps50.npz \
  --target-fps 50
```

以上命令已执行，重跑请改用新的输出目录；入口拒绝覆盖已有文件。

## 已准备的物理与训练配置

- 几何：共享缩放0.744193062691336；独立 `chair021_training.urdf`，相对路径mesh
  `chair021_m.obj`，保持预览的尺寸、原点和质心。质量由用户指定5 kg；同几何惯量按原1 kg
  占位值乘5。这是选定的实验负载，不是数据集实测质量。
- 静/动摩擦0.5/0.5、恢复系数0；椅子使用IsaacLab已有 `convex_decomposition`，
  旧实验仍默认 `convex_hull`。没有复制小桌几何或新建碰撞算法。
- `--experiment chair021` 复用原train/eval/smoke与双机器人环境；默认smalltable保持兼容。
  新实验参考、资产、日志及checkpoint来源校验独立，不需要chair专用环境类。
- 两台rubber-hand G1共享Actor：158 → 512 → 256 → 128 → 29，隐藏层ELU；
  每台输入为原基础观测154加队友4。全局Critic：527 → 512 → 256 → 128 → 1。
  椅子状态不新增到Actor中；MAPPO、基础reward/termination和控制逻辑沿用原baseline，
  不加人—物或人—人交互图奖励。
- 物理200 Hz、控制50 Hz，每5 ms刷新两台机器人的PD状态；参考391帧、采样跨度7.8秒，
  非循环，环境时间上限按391/50=7.82秒配置。

## 实际检查结果

- 104项相关CPU测试通过，包括旧小桌默认配置、checkpoint兼容与新入口选择。
- 1环境、8控制步、不更新PPO的Isaac加载检查通过；实际质量5 kg，16个物理shape的
  静摩擦/动摩擦/恢复系数均为0.5/0.5/0，USD碰撞近似为 `convexDecomposition`。
- Actor输入[1,2,158]、Critic输入[1,527]、动作[1,2,29]，参考帧数/频率及物体
  原点—质心速度转换和reset写入检查通过。报告：
  `logs/Core4DChair/preparation_20260909/physics_smoke.json`。
- 该报告保存检查时的 `training_ready=false` 和当时manifest哈希；通过后才记录技术准备完成。
  参考、URDF和mesh没有在检查后更换。不改写历史报告以假装检查前已准备完毕。
- 本检查不证明已学会动作，也不证明凸分解精确保留全部开口；不以8步未训练动作质量作为
  否定训练方案的门槛。该检查本身没有更新PPO。

## 正式训练（2026-09-10已确认并启动）

沿用此前CORE4D基础版预算：从零初始化共享Actor与全局Critic，2048环境、每环境每次
采样24步、12000 iterations、每2000保存checkpoint、seed721。用户已确认；
不加载旧小桌或旧Push/Pull/Kick的checkpoint。

```bash
cd /home/kevin/holosoma-core4d-base
/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python -B -u \
  scripts/train_core4d_smalltable.py --experiment chair021 \
  --num-envs 2048 --steps-per-env 24 --iterations 12000 --save-interval 2000 --seed 721 \
  --output-dir logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910
```

服务为 `train-core4d-chair021-20260910.service`；不重复运行上面的命令。
启动记录、完整stdout及启动时源码/资产快照：`logs/Core4DChair/launch_20260910/`。
初始checkpoint单独保存，之后保存2000/4000/6000/8000/10000/12000。完成情况以实际
`status.json`和指标为准；本说明不提前宣称训练完成或动作合格。

唯一执行路线仍是仓库根目录 `MULTI_AGENT_EMERGENCE_ROADMAP.md`。
