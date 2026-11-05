# 小桌5kg完整桶A配方：独立部署说明

本说明保留原 A 配方技术快照，不替代总路线。A 正式训练已启动；2026-09-20 用户另批准下述双删除对照。

- experiment=`smalltable5kg_A`，reward-variant=`bucket_A`。CLI沿用奖励实现名字，不代表桶数据。
- 原小桌687状态/50Hz参考逐字节不变，物体5kg、惯量为20kg旧版的0.25倍，COM/几何/五碰撞盒不变。
- 静/动摩擦0.5/0.5，恢复0，物理200Hz/控制50Hz、每物理子步刷新实际关节状态。
- 共享Actor158→29，集中Critic527→1；MLP512/256/128 ELU，原PPO/跟踪/正则/终止不变。
- 位置/朝向/独立正高度/相对关系权重1/1/1/2；位置xyz误差权重1/1/2；尺度0.3m/0.4rad/0.10m/0.04m。
- 不加旧Laplacian、负平方高度或B接触调制。每人19点、物体85点（预算100），手组三点各1/3权重。
- 小桌专用关系数据保留物体link原点定义、wxyz参考转xyzw的约定，不复制桶点或源接触标签。
- 全零reference_contact_weights仅兼容现有数据格式，明确not_used_for_smalltable_A；不表示示范不接触。
  B及未批准的单项消融对该实验禁止；2026-09-20 开放双删除对照。原始四手法向接触仍记录，Bucket/contact_error不作小桌语义指标。
- 正式候选：fresh seed721，2048env×24步，12000轮，每2000保存；不加载桶或短检查checkpoint。
- 短检查：2轮、2048环境，独立输出。它只验证程序、更新、模型保存和资源，不评判学习效果。
- 旧20kg小桌及桶实验源码快照、模型、日志均保留，不覆盖云端正在运行目录。

源资产目录：`src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/`。
本地原始验证记录：`logs/Core4DSmallTableA/preparation_20260919/`。
267项CPU回归通过；本地1环境8物理控制步真实退出0，加载质量/材质/维度核对通过。
training_ready表示资产已通过物理检查，不代表已授权正式长训练；formal_training_authorized仍false。

正式候选命令（此处仅说明，不自动执行）：

```bash
python -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A \
  --iterations 12000 --num-envs 2048 --steps-per-env 24 --seed 721 --save-interval 2000 \
  --output-dir logs/Core4DSmallTableA/A_5kg_fresh12000_env2048_seed721
```

`run_config`保留旧通用参数object_z_error_weight=1；实际A组合位置权重由bucket_reward_contract的
position_error_weights_xyz=[1,1,2]决定。不要把CLI默认字段误当实际奖励，也不额外再加一次z项。

## 2026-09-20 已批准：5 kg 小桌双删除对照

- 使用同一 `smalltable5kg_A` 场景身份，显式指定 `--reward-variant bucket_A_no_rel_no_height`。
- 仅将独立高度、相对向量关系贡献置零：位置/朝向/高度/关系为 `[1,1,0,0]`；完整 A 为 `[1,1,1,2]`。
- 保留原位置奖励中的 xyz 权重 `[1,1,2]`，因此不是删除一切高度信息；诊断仍记录高度和关系误差。
- 5 kg 质量、惯量、参考、几何、材质、网络、身体跟踪、正则、重置和终止均与完整 A 相同。
- 从零 12000 轮，2048 环境，每轮 24 步，seed721，每 2000 轮保存；不从两轮检查模型续训。
- 资产清单中 `allowed_reward_variant: A` 是原始资产晋级记录，不修改其字节或资产哈希；本次配方由入口允许列表及运行/模型的 `bucket_reward_contract` 单独识别。
- 打包器的 `formal_training_authorized=false` 仅表示打包不自动授权启动；本次实际授权及命令由独立启动记录保存。
- 云端新目录 `core4d_smalltable5kg_no_rel_no_height_20260920`，不修改运行中的完整 A 源码快照。
- 本地验证：268 项 CPU 测试通过；只变更配方身份及两项权重的回归断言通过。

本次操作证据目录：`logs/Core4DSmallTableA/both_removed_20260920_launch/`。
