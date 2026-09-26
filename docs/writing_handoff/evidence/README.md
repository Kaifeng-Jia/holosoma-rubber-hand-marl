# 可携带实验依据（2026-09-25）

这里把原来仅存在 Ubuntu `logs/` 中的关键文本依据整理成小包。Mac 无需安装 Isaac、
Conda 或恢复 checkpoint，即可检查实验配置、奖励身份和已有评测结果。
原训练目录、模型、数据和日志未改动；本次没有重新训练或运行物理评测。

## 怎么看

1. 总体实验分组和公式先看[实验台账](../../../DEMO_AND_EXPERIMENT_INVENTORY.md)第3、9节。
2. 本目录的 [index.json](index.json)将每个实验连接到可读配置、原始配置和评测摘要。
3. 每组 `config_extract.json` 可直接阅读；`summary.json` 是既有评测原件。
4. 下方比较报告提供行为分析和指标解释；展示视频与切片由上级交接目录另行索引。

## 已收齐的 CORE4D 正式实验

下表统计的是固定参考场景中**到达参考末尾的回合数**，不是“用手承重”、双人贡献均衡或
跨物体泛化的自动成功标签。13组均为一次训练seed721、fresh12000轮，2048环境，每轮24步。
椅子另有固定条件复测；这些评测seed不是多次独立训练。

| ID | 物体/奖励身份 | 完整回合 | 可读配置 | 原始评测 |
|---|---|---:|---|---|
| B-A | 桶1kg；完整A：wz2 + 独立正高度 + 相对向量 | 2/5 | [配置](B-A/config_extract.json) | [summary](B-A/summary.json) |
| B-B | 桶1kg；A的物体/关系块加软接触调制 | 1/5 | [配置](B-B/config_extract.json) | [summary](B-B/summary.json) |
| B-R0 | 桶1kg；A去相对向量项 | 4/5 | [配置](B-R0/config_extract.json) | [summary](B-R0/summary.json) |
| B-Z0 | 桶1kg；A去独立高度项 | 0/5 | [配置](B-Z0/config_extract.json) | [summary](B-Z0/summary.json) |
| B-RZ0 | 桶1kg；两项均删除，保留wz2；用户首选展示 | 5/5 | [配置](B-RZ0/config_extract.json) | [summary](B-RZ0/summary.json) |
| T-01 | 小桌20kg；基础11项、wz1 | 5/5 | [配置](T-01/config_extract.json) | [summary](T-01/summary.json) |
| T-02 | 小桌20kg；wz1 + 旧Laplacian图 | 2/5 | [配置](T-02/config_extract.json) | [summary](T-02/summary.json) |
| T-03 | 小桌20kg；基础11项、wz2 | 5/5 | [配置](T-03/config_extract.json) | [summary](T-03/summary.json) |
| T-04 | 小桌20kg；wz2 + 旧Laplacian图 | 5/5 | [配置](T-04/config_extract.json) | [summary](T-04/summary.json) |
| T-05 | 小桌20kg；旧图 + wz2 + 独立高度负惩罚 | 4/5 | [配置](T-05/config_extract.json) | [summary](T-05/summary.json) |
| T-06 | 小桌5kg；完整桶A配方、使用小桌点集 | 5/5 | [配置](T-06/config_extract.json) | [summary](T-06/summary.json) |
| T-07 | 小桌5kg；独立高度与关系双删除、仍有wz2 | 3/5 | [配置](T-07/config_extract.json) | [summary](T-07/summary.json) |
| C-01 | 椅子5kg；基础11项、wz1 | 3/3；三次复测合计9/9 | [配置](C-01/config_extract.json) | [首轮](C-01/summary.json) / [722](C-01/summary_seed722.json) / [723](C-01/summary_seed723.json) |

椅子三组的确定性轨迹相同，所以9/9表示固定条件执行记录，不作为随机场景泛化结论。
T-04另收录[09-17同模型复测](T-04/summary_recheck_20260917.json)，不重复计训练组。
每组同时保留`status.json`；其中`passed=true`不是操作系统进程退出码的替代证据。

## 奖励身份的关键读法

- 桶式配置的`bucket_reward_contract`记录实际专用奖励块：位置xyz权重为`[1,1,2]`。
  若顶层旧通用字段`object_z_error_weight`为1，不代表桶式块实际用了wz1。
  T-06/T-07也是这一专用块；查嵌套契约及对应代码，不根据通用残留字段重新命名实验。
- B-RZ0/T-07虽无两个新增项，仍保留身体跟踪、动作/限位/接触正则、物体位置及朝向。
  因而不能称为“只管任务、不管姿态”或“不管高度”。
- 旧Laplacian图和桶式相对向量是不同项；独立负高度惩罚与正指数高度奖励也不同。
- 历史配置中缺失的字段原样缺失，索引用`null`表示未记录；不以当前代码默认值补写旧事实。
- 历史`git_commit`是记录中的提交号，不保证当时工作区无未提交代码。
  精确源码溯源仍需对应冻结源码清单，不能把全部模型都标成当前HEAD训练。

## 已有比较报告

- [桶A/B](reports/bucket_AB_20260919.md)
- [桶先完成的两组消融](reports/bucket_ablations_20260919.md)
- [桶五组最终对照](reports/bucket_ablations_final_20260920.md)
- [5kg小桌两组对照](reports/smalltable_5kg_20260920.md)及[分析JSON](reports/smalltable_5kg_analysis_20260920.json)
- [20kg小桌独立高度惩罚对照](reports/smalltable_height_penalty_20260917.md)
- [扶椅子训练复核](reports/chair_training_review.md)

这些报告是**原文副本**，没有为了Mac移动路径而修改。报告中的Ubuntu绝对路径、旧端口、
`logs/`链接是历史来源标识，不代表Mac可点击文件或当前在线服务；可移植入口以本README和
index中的相对路径为准。报告中早于后续消融的判断，也应结合五组最终结果解读。

## 历史 OMOMO 路线与诊断 Demo

| 实验 | 便携配置 | 本次收录证据 |
|---|---|---|
| Push live-PD | [配置](historical/Push-livePD/config_extract.json) | [27次既有日志的结果字段](historical/Push-livePD/evaluation_outcomes_27runs.json)：13050为1/9，14050和15050为0/9完成 |
| Pull | [配置](historical/Pull/config_extract.json) | [历史记录摘录](historical/rollout_history_excerpts.md)：8050在固定单回放完成316/316帧 |
| Kick | [配置](historical/Kick/config_extract.json) | [历史记录摘录](historical/rollout_history_excerpts.md)：8000固定单回放完整执行step0..297 |
| Demo3 对抗 | [恢复配置](historical/Demo3/config_extract.json) | [原始评测](historical/Demo3/evaluation.json)：10100，5次约6步跌倒；胜负标记不是有效拔河 |
| Demo4 旋转 | [配置](historical/Demo4/config_extract.json) | [原始评测](historical/Demo4/evaluation.json)：10000，5/5达到yaw阈值；动作质量另按猛烈甩转诊断记录 |

Pull/Kick所选回放目录未找到独立summary JSON；保留来源可核对的历史摘录，不伪造新的统计摘要。
Push的27条是原日志结果块的选定字段，日志源SHA逐条记录；没有复制大日志、重跑评测或把它
误标为旧joint-acceleration平滑链。该多seed协议没有随机化初态、质量或摩擦。

## 原件、压缩与校验

原run_config中有2048环境的重复物理读回数组，合计数十MB。因此每组提供：

- `run_config.json.gz`：原始字节无损gzip，展开后就是完整原件；没有删字段或改路径。
- `config_extract.json`：可读派生摘要；保留所有字典字段，只把超过32项的数组替换为
  长度、首末值、是否全相同的摘要。文件头明确标识不是原配置。
- `index.json`：原件路径、原件SHA256、复制文件路径/SHA256、转换方式和评测身份。
- `SHA256SUMS`：本证据目录文件的校验清单（不包含自身）。

Mac终端在本目录下可以使用系统工具，不需要安装项目依赖：

```bash
shasum -a 256 -c SHA256SUMS
gzip -dc B-A/run_config.json.gz > /tmp/core4d_bucket_A_run_config.json
```

`collect_evidence.py`是本次收集脚本，仅供Ubuntu原工作区复查/重建；Mac不需运行。
本包不含模型、原始数据、NPZ轨迹、环境、token或训练目录备份。这里提供的是**写作可核查证据**，
而不是能够脱离外部模型与数据直接复现实验的完整运行包。
