# T09：第四稿的消融和奖励变体在本机 5070 上重评（2026-11-17 第一版）

## 目的

论文的所有最终评测都在同一张卡上做。

- 已经在本机 5070 上的：主结果（表 1）和 SONIC 对比。
- 还是 V3 时在云端 4090 上的：第四稿的消融和奖励变体。

两张卡混用的后果有两个：
- 同一个策略在两张表里的数字不一样，例如 Push 在表 1 是 84%，在消融表里是 78%；
- 配对检验里混进了换卡带来的差别。

## 范围

- 只评测已有的最终 checkpoint，不训练，不改任何设置。
- 只用本机 5070，最多同时跑 2 个评测，不开云端。
- **通过后直接继续，做完交付。**

## 评测清单

1. **V3 收尾包里的全部变体。** 即 `results/episodes.csv` 里除 `Full` 和 `Full-rerun` 以外的所有变体：
   - `pull` 的 `from-scratch`、`zero-teammate-obs`；
   - `bucket` 和 `chair` 的 `no-wrist`；
   - `chair` 的 `source-size-reopt`、`one-sided-scale`；
   - `push`、`pull`、`kick` 的 `copied-single-prior`。

   种子、档位和 V3 相同：V3 评了哪几档就评哪几档，至少包括 ±2 和 ±8。
2. **第四稿里另外两组，不在收尾包里：**
   - 消融表的 Push "prescribed placement"；
   - 奖励表里 Bin 和 Table 的 "+ Height + Relation"。

   找到当时用的最终 checkpoint，按 sha256 和当时的记录核对。某一组找不到或无法确认，就停下回报这一组，其余照做，不拿相近的 checkpoint 代替。

**顺序：**
1. Pull 的两组（from-scratch、zero-teammate-obs），正文引用最多；
2. 两组 no-wrist；
3. Push prescribed placement；
4. copied-single-prior；
5. Chair 的两组；
6. 奖励变体。

**这次不做：** ±4、±6 档。等论文的图定下来再说。

## 协议

- 和 `results_system1/T03/d_final_5070/` 完全相同：
  - 冻结的初始状态；
  - 确定性动作，TF32 关闭；
  - 用最终 checkpoint；
  - 每档 50 个回合，nominal 不计。
- checkpoint 和状态文件按 sha256 与 V3 的记录逐一核对。

## 指标

- V3 `results/episodes.csv` 的全部字段，口径不变。
- **补算身体误差：** 每台机器人 14 个跟踪 link 的平均位置误差，口径和第四稿表 I 相同，从保存的轨迹算。
  - 同时给 D 的六个任务补算（d_final_5070 和 c2_main 里的 D），这样表 1 的误差列全部来自 5070。
  - 先找第四稿表 I 当时算身体误差的代码，复用它，不重写。

## 配对检验

每个变体与同任务的 D（5070 上的结果）按初始状态配对，口径和第四稿相同：
- 三个种子合并做精确 McNemar；
- 每个种子也单独报。

## 交付

`results_system1/T09_ablation_5070/` 下：
- `episodes.csv`、`per_seed_summary.csv`、`group_summary.csv`；
- `paired_completion.csv`；
- `table1_errors.csv`：D 六个任务的物体、身体、手、掌面误差；
- `v3_vs_5070.csv`：每组和 V3 的数并排，写差几个回合，不解释；
- `README_CN.md`、配置与 diff、`SHA256SUMS`。

做完 push，按常规格式回报。
