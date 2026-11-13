# T04 承重边界：每档重训的 D 和 C-in（2026-11-13，本机 5070）

依据：`docs/system1/T04_mass_sweep_zh.md` 第四版补充（sha256 `2e911a1c…a4cf`）、`docs/system1/paper_results_plan_zh.md` 第二节第 3 条。本文件只写事实。

## 做了什么

- **重训 run**：D 和 C-in 各 6 个，种子 721，在云端 4090 上训练。
  - Pull 40 kg；
  - Table 10、20 kg；
  - Bin 2、4、8 kg。
- **评测**：最终 checkpoint，在本机 RTX 5070 上评 ±2、±8。
  - 冻结初始状态，sha256 见 episodes 表的 `state_file_sha256` 列；
  - TF32 关闭；
  - 本机流水线 `system1/t03/post_train.py` 跑的。
- **原质量档**：也是本机 5070 的结果。
  - D 是论文 seed721 主结果的最终 checkpoint（与 `results_system1/T03/c2_finals/finals.csv` 同一批文件）；
  - C-in 是 T03 的 721 主 run。
- **逐回合指标**：沿用 `system1/t04/package_d_direct.py` 的口径。手误差用 URDF 正运动学算，Table 和 Bin 的掌面误差用 V3 后处理，物体误差取自评测 summary。
- **读回质量**：每个评测 summary 里 PhysX 读回的物体质量，都等于该档位的质量（脚本里有断言）。

## 结果

| 任务 | 质量（kg） | 倍数 | D ±2 | D ±8 | C-in ±2 | C-in ±8 |
|---|---|---|---|---|---|---|
| Pull | 20（原） | 1 | 50/50 | 43/50 | 50/50 | 22/50 |
| Pull | 40 | 2 | 12/50 | 21/50 | 36/50 | 26/50 |
| Table | 5（原） | 1 | 35/50 | 45/50 | 31/50 | 24/50 |
| Table | 10 | 2 | 22/50 | 25/50 | 50/50 | 47/50 |
| Table | 20 | 4 | 49/50 | 46/50 | 44/50 | 42/50 |
| Bin | 1（原） | 1 | 50/50 | 39/50 | 49/50 | 30/50 |
| Bin | 2 | 2 | 13/50 | 11/50 | 23/50 | 26/50 |
| Bin | 4 | 4 | 45/50 | 32/50 | 49/50 | 35/50 |
| Bin | 8 | 8 | 16/50 | 18/50 | 0/50 | 0/50 |

- **单种子**：每个格子都只有 1 个种子（721）。
- **完成率随质量不单调的格子**：
  - D：Table 10 kg 低于 5 kg 和 20 kg；Bin 2 kg 低于 1 kg 和 4 kg。
  - C-in：Bin 2 kg 低于 1 kg 和 4 kg。
- **其他列**：平均进度、物体误差、手误差、掌面误差见 `mass_sweep_summary_retrained.csv`。

## 加档、插档规则检查（规则在看到结果之前写在任务书里）

**边界**：±2 完成率不低于 50% 的最大质量。

| 任务 | D 边界 | C-in 边界 | 加档规则（最高档上 D 或 C-in ±2 ≥ 50%） | 插档规则（两者边界在同一对相邻档之间） |
|---|---|---|---|---|
| Pull | 20 kg（40 kg 为 24%） | 40 kg（72%） | **触发**：C-in 在 40 kg 达 72%，按规则加 80 kg | 不触发：C-in 的上界还没夹住 |
| Table | 20 kg（98%） | 20 kg（88%） | **触发**：两者在 20 kg 都 ≥ 50%，按规则加 40 kg | 不触发：上界都没夹住 |
| Bin | 4 kg（90%；8 kg 为 32%） | 4 kg（98%；8 kg 为 0%） | 不触发 | **触发**：两者都在 4 kg 过、8 kg 不过，按规则在几何中点 4√2 ≈ 5.66 kg 加一档 |

- 按 Kevin 的指示，只检查、回报，没有开训。
- 物理上限核算见 `../physics_check/`。若某档物理上做不到，按规则标"超出物理上限"，不用于比较架构。

## 文件

| 文件 | 内容 |
|---|---|
| `mass_sweep_episodes_retrained.csv` | 36 组 × 51 回合，含 nominal，列同 V3 收尾包 |
| `mass_sweep_summary_retrained.csv` | 每组的完成数、平均进度、误差、checkpoint、初始状态和 summary 的 sha256、评测目录 |
| `provenance.json` | 手部正运动学自检、参考和掌面后处理的 sha256、每组的来源目录 |
| `SHA256SUMS` | 本目录文件的 sha256 |

`../mass_sweep_summary.csv` 已重建，合并了三部分：
- 方案 A 的全部档位；
- 原质量 D 直接评其他档位（`../D/mass_sweep_summary_D.csv`，未改）；
- 本表的重训档和原质量点。原质量点改用 5070 的重评，原来那一行来自 V3 收尾包的 D，已不再放进合并表。

命令：

    hssim_s1/bin/python -B system1/t04/package_retrained.py
