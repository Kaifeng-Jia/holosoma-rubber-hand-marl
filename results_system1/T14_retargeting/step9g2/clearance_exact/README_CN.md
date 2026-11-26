# T14 第 9g 步再修，第 1 条：两机间隙改按实际碰撞形状重判

- **依据：**Kevin 2026-11-26（10:29 收到）"第 9g 步再修"第 1 条："两机间隙改用实际碰撞形状来量：和'两机最小距离'一样，用 PhysX 连杆形状，算 mj_geomDistance，不再用包围盒代理。门槛不变（≥ −1 cm）。第 8 步（step8b）、9f、9g 的已有结果全部按新量法重判（只重算判定），新旧并排列出。"
- **时间：**2026-11-26 10:35（纽约），只读 JSON，几秒钟。
- **代码：**`system1/t14/objects/rejudge_clearance_exact.py`，提交 `b170599`；smoke test `system1/t14/objects/test_step9g2.py`（通过）。

## 两种量法

| | 旧（包围盒代理） | 新（实际碰撞形状） |
|---|---|---|
| 几何 | `reference_geometry_round2_checks.CollisionProxy`，两台机器人各连杆的 OBB 代理 | `dual_stretch.PairCollision(depth="exact")`：两台机器人各 23 个连杆的 PhysX 碰撞形状（求解器、训练仿真同一批），跨机器人所有连杆对，`mj_geomDistance` |
| 量 | `robot_robot_clearance_m` 全片段最小值 | 评测里已报的"两机最小距离" `robot_robot_min_distance_cm` 全片段最小值 |
| 门槛 | ≥ −0.01 m | ≥ −0.01 m（不变） |

- 两种量都是评测时已经算好、存在每个 `eval.json` 里的，帧也相同。这次没有重新求解，也没有重算几何。
- 档位协议里"1.5 以上第一个不相交档"一直用的是新量法（最小距离 < 0 算相交），这次没有改。

## 三种判定并排

- **A：**非手穿透和两机间隙都用包围盒。这是各轮自己的判定；step8b 是它报告时用的第 6 轮重判。
- **B：**非手穿透用碰撞形状，两机间隙用包围盒，也就是 9g 续第 1 条的结果。
- **C：**两项都用碰撞形状，即本次结果。其余判定项照旧。

## 结果

| 集合 | 评测档 | 两机间隙改判 | 改判方向 | 旧量法不过 / 新量法不过 | 适用档 A / B / C | 适用片段 A / B / C |
|---|---|---|---|---|---|---|
| step8b（57 个候选，1.0 倍） | 57 | 0 | — | 36 / 36 | 7 / 15 / 15 | — |
| 9f（22 条，全部评测档） | 138 | 33 | 全部从不过变成过 | 70 / 37 | 19 / 50 / 70 | — |
| 9f 报告档 | 70 | 16 | 同上 | 32 / 16 | 8 / 24 / 34 | 3 / 10 / 12 |
| 9g（22 条，全部评测档） | 118 | 27 | 同上 | 72 / 45 | 22 / 40 / 61 | — |
| 9g 报告档 | 73 | 19 | 同上 | 39 / 20 | 12 / 28 / 41 | 5 / 13 / 14 |

- **没有一个档从过变成不过。**改判的档里：
  - 旧量法的值在 −7.2 到 −1.0 cm 之间；
  - 新量法的值在 −0.99 到 +6.1 cm 之间，多数是正值，即碰撞形状之间没有接触。
- **新量法下仍然不过的报告档，**两台机器人的碰撞形状都实际穿进 2 cm 以上：
  - 9f 16 个，−16.1 到 −2.0 cm；
  - 9g 20 个，−13.8 到 −2.5 cm。
- **step8b：**36 个不过的候选，新量法在 −20.1 到 −3.1 cm 之间，两种量法完全一致。
  - Push / Pull 按规则选出的摆法，B 和 C 相同：Push c00，Pull c05。
  - A 里 Push 是 c09，9g 续第 1 条已经报过。
- **9f，从 B 到 C：**
  - 新增适用片段：smallboxa（1.5、1.55）、sub4_suitcase_042（0.85）；
  - 新增适用的报告档共 10 个。
- **9g，从 B 到 C：**
  - 新增适用片段：sub8_smallbox_045（2.25）；
  - 新增适用的报告档共 13 个，明细见 `step9g_all22_clearance_exact.csv`。
- **9g 在 C 下的未通过项**（73 个报告档）：两机间隙 20，保持率 14，主运动 9，支撑脚漂移 r1 4。
- **9g 在 C 下适用的 14 条片段及档位：**
  - largeboxa 1.25、1.5；
  - suitcasea 0.95、1.0、1.25、1.5；
  - sub3_largebox_003 1.5；
  - sub3_largebox_029 0.95、1.0、1.25、1.5；
  - sub8_smallbox_045 2.25；
  - sub1_suitcase_031 0.7、1.0、1.25、1.5；
  - sub1_suitcase_034 1.25、1.5；
  - sub4_suitcase_006 1.0、1.25、1.5；
  - sub4_suitcase_042 0.9、1.0、1.25、1.5；
  - sub1_suitcase_036 1.0、1.25、1.5；
  - sub4_suitcase_043 0.8、1.0、1.25、1.5；
  - sub8_suitcase_020 0.65、1.0、1.25、1.5；
  - sub5_suitcase_046 1.5；
  - sub1_suitcase_039 0.95、1.0、1.25、1.5。

## 文件

- `step8b_clearance_exact.csv / .json`、`step9f_clearance_exact.csv / .json`、`step9g_all22_clearance_exact.csv / .json`：
  - 每档一行，列出两种量法的值和判定；
  - A / B / C 三种判定及各自的未通过项；
  - 对应的 `eval.json` 路径。
  - json 是按片段（step8b 按序列）的汇总，step8b 含 A / B / C 各自选出的摆法。
- 输入：
  - 9g 续第 1 条的非手重判表（`../../step9g/nonhand_exact/*_nonhand_exact.csv`），提供 A、B 的判定；
  - `logs_system1/t14/{step8b,step9f,step9g}/` 下的 `eval.json`。
- 命令：

  ```
  N=results_system1/T14_retargeting/step9g/nonhand_exact; O=results_system1/T14_retargeting/step9g2/clearance_exact
  python3 -B system1/t14/objects/rejudge_clearance_exact.py --set step8b --nonhand $N/step8b_nonhand_exact.csv --out $O
  python3 -B system1/t14/objects/rejudge_clearance_exact.py --set step9f --nonhand $N/step9f_nonhand_exact.csv --out $O
  python3 -B system1/t14/objects/rejudge_clearance_exact.py --set step9g --nonhand $N/step9g_all22_nonhand_exact.csv --out $O --name step9g_all22_clearance_exact
  ```

## 与任务书的偏离

- 无。
- 9g 的 3 条调试片段没有单独列表，它们的档都包含在 22 条的表里。
- 9f 有一个档（sub15_suitcase_051 1.65），旧量法的值记为 −0.01000 m，但判定是不过：存的值四舍五入到 0.01 mm，实际值略小于 −1 cm。表里旧判定照 `eval.json` 原样列出。
