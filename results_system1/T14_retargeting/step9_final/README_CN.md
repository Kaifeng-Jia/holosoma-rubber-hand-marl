# T14 最终测试：冻结的 9g2 构造，plasticbox、smalltable 直走片段

- **依据：**Kevin 2026-11-26 决定（"T14 第 9g2 步之后"）第一项：
  - "9g2 的构造从现在起冻结，之后不再改"；
  - "在 plasticbox、smalltable 的直走片段上一次跑完……只跑一次，结果不管好坏都如实报，不再改构造。trashcan 不进"；
  - 交 README、每条片段的结果表、适用片段的并排图和各档位图，视频传 Drive，只用 CPU。
- **只跑了一次。**构造、求解、判定、门槛都没有因为这次的结果改动。

## 1. 冻结的构造

- **构造：**9g2 的三条规则，加上"走过的距离 = 骨盆和两踝沿走路方向位移的平均"（`--fb-walk-distance pelvis_feet`）。
- **冻结的代码：**`b170599`，即 9g2 运行时的 commit。以下五个文件此后没有改动，sha256 前 16 位：

  | 文件 | sha256 |
  |---|---|
  | `frontback.py` | 5cdd5479652e6b6a |
  | `role_solve.py` | 5335add623b22f22 |
  | `partner.py` | f33aafafdd4c2874 |
  | `step9_eval.py` | 2915b244a2afbf67 |
  | `step9_gates.py` | 2740d616237cd450 |

- **运行时的代码：**`42669f0`。它和 `b170599` 之间，`system1/t14/objects/` 只加了开关，默认行为不变：
  - `step9_turns.py --objects`：挑片段的对象，默认输出逐字节相同；
  - `run_step9.py --turns`：片段表；
  - `rejudge_*_exact.py --set step9_final`：重判；
  - `step9e_package.py --turns`：打包；
  - 出图标签：`render_box.py`、`step9g_render_both.py`。
- **打包后另加的：**
  - `step9e_package.py` 修了一行，见第 5 节；
  - 新增 `step9_final_table.py`，用来出本返包的两张结果表。

## 2. 和 22 条调试集相同的部分

- **挑片段：**`step9_turns.py`，每帧朝向和走路方向差 45° 以内算直走，规则同 9g2。
  - plasticbox：a 类 37 条，直走 3 条；
  - smalltable：a 类 73 条，直走 4 条；
  - 一共 7 条全跑。trashcan 没有进。
  - 片段表：`class_a_turns.csv / .json`。
- **档位：**
  - 先跑 1.0、1.25、1.5；
  - 1.0 档两机不相交时往下找 a 档；1.5 档两机相交时往上找第一个不相交档。两个判断用的都是求解驱动自己的量法，22 条调试集也一样。
  - 7 条里只有 sub15_plasticbox_008 往下找了（0.90、0.95，a 档 = 0.95）；没有一条往上找。
- **判定 C：**非手穿透和两机间隙都用实际碰撞形状量，两机间隙门槛 ≥ −1 cm。其余判定项和门槛照旧：
  - 重力 5°；
  - 支撑脚漂移 1 cm；
  - 非手穿透 1 cm；
  - 接触保持率 ≥ 0.99；
  - 主运动：两个人、两台机器人的骨盆水平净位移都 > 1 m（求解尺度下按 SMPL 缩放）。
- **命令：**和 9g2 的 `run_all22.sh` 只差三处：
  - `--step step9_final`；
  - `--turns` 指向本次的片段表；
  - 不用 `--source-root`：源解全部重算，因为这 7 条在调试集里没有源解。

  ```
  CUDA_VISIBLE_DEVICES="" taskset -c 16-31 sonic_s1/bin/python -B system1/t14/objects/run_step9.py --step step9_final \
    --turns results_system1/T14_retargeting/step9_final/class_a_turns.csv --fb-turn world --fb-legs mirror \
    --fb-walk-distance pelvis_feet --fb-human-frame target --skip-failed-up --all-main --r1-source construction \
    --parallel 6 --deadline 23:59 --no-render
  # 之后：records/post.sh（重判、出表、出图、打包）
  ```

## 3. 结果（只列事实）

- **适用（判定 C）：**7 条里 3 条，22 个报告档里 6 档：
  - sub12_plasticbox_053：1.00、1.25、1.50；
  - sub12_smalltable_016：1.25、1.50；
  - sub9_smalltable_020：1.50。
- **不适用的 4 条：**
  - sub15_plasticbox_008：四个报告档都只差接触保持率（0.966 / 0.969）；
  - sub12_plasticbox_057：三档都有 r1 支撑脚漂移（1.42 cm）和保持率（0.976 / 0.980），1.00 和 1.25 还有两机间隙；
  - sub3_smalltable_008：三档都有 r1 支撑脚漂移（2.81 cm）和主运动，1.00 和 1.25 还有两机间隙；
    - 主运动：示范者骨盆 1.30 m，副本人 0.94 m，机器人 r0 0.90 m，r1 1.28 m，门槛 1 m。
  - sub3_smalltable_007：三档都有保持率（0.942），1.00 和 1.25 还有两机间隙。
- **22 个报告档的未通过项：**
  - 保持率 10 档，两机间隙 9 档，r1 支撑脚漂移 6 档，主运动 3 档；
  - 非手穿透 0 档：报告档里碰撞形状量的最大值 0.22 cm，在 sub15_plasticbox_008 r1 右腕 pitch 连杆（没报告的 0.90 档是 0.23 cm）。
- **两机间隙（1.00 档）：**
  - 7 条里 5 条相交超过 1 cm，最深 −12.8 cm（sub12_plasticbox_057）；
  - sub12_plasticbox_053 是 −0.52 cm，只有 1 帧，在门槛以内；
  - sub15_plasticbox_008 是 +5.3 cm，不相交。
- **判定 A、B 对照：**和 9g2 的 `clearance_exact` 同表。
  - A：两项都用包围盒，0 档适用；
  - B：只有两机间隙用碰撞形状，4 档适用；
  - C：6 档适用。

### 逐条结果（`clips.csv`；每档明细在 `levels.csv`）

| 片段 | 物体 | 跑过的档 | 适用档（C） | 各报告档的结论 | 保持率最低 | 支撑脚漂移最大 | 两机最小距离（碰撞形状） | 倒走开头人脚 / 机器人踝 / 原示范走路段（cm） | 腿最大拉长 |
|---|---|---|---|---|---|---|---|---|---|
| sub12_plasticbox_053 | plasticbox | 1.00 1.25 1.50 | **1.00 1.25 1.50** | 三档都适用 | 1.0 | 0.43 cm | −0.52 cm（1.00 档，1 帧） | 4.1 / 5.6 / 6.3 | −0.5 cm |
| sub15_plasticbox_008 | plasticbox | 0.90 0.95 1.00 1.25 1.50 | 无 | 0.95–1.50：保持率 | 0.966 | 0.44 cm | 1.0 cm | 4.3 / 4.3 / 7.0 | 3.3 cm |
| sub12_plasticbox_057 | plasticbox | 1.00 1.25 1.50 | 无 | 1.00、1.25：漂移 r1、两机间隙、保持率；1.50：漂移 r1、保持率 | 0.976 | 1.42 cm | −12.8 cm | 5.9 / 7.1 / 7.6 | 3.2 cm |
| sub12_smalltable_016 | smalltable | 1.00 1.25 1.50 | **1.25 1.50** | 1.00：两机间隙 | 1.0 | 0.62 cm | −10.7 cm（1.00 档） | 6.6 / 7.0 / 7.3 | 2.9 cm |
| sub9_smalltable_020 | smalltable | 1.00 1.25 1.50 | **1.50** | 1.00、1.25：两机间隙 | 1.0 | 0.47 cm | −12.3 cm（1.00 档） | 5.1 / 7.8 / 7.6 | 3.0 cm |
| sub3_smalltable_008 | smalltable | 1.00 1.25 1.50 | 无 | 1.00、1.25：漂移 r1、两机间隙、主运动；1.50：漂移 r1、主运动 | 1.0 | 2.81 cm | −12.2 cm | 3.9 / 4.3 / 6.8 | 2.5 cm |
| sub3_smalltable_007 | smalltable | 1.00 1.25 1.50 | 无 | 1.00、1.25：两机间隙、保持率；1.50：保持率 | 0.942 | 0.22 cm | −10.9 cm | 3.3 / 3.6 / 6.5 | 2.5 cm |

- **"保持率最低""支撑脚漂移最大""两机最小距离"：**在该片段所有报告档上取最值。支撑脚漂移用判定时的值（第 6 轮的量法：源解同一帧）。
- **"倒走开头"：**副本走路段前 15 帧里单帧最大位移，定义和 9g2 的 `compare.csv` 相同：
  - 人脚 = 副本人的两踝和两脚尖；
  - 机器人踝 = r1（1.0 倍）的两个踝 roll 连杆；
  - "原示范走路段" = 示范者自己走路段的最大单帧位移。
- **"腿最大拉长"：**走路段镜像髋到倒放踝的距离减大腿加小腿长，两条腿取大。完整数据在 `feet_ik.csv`。

## 4. 图和视频

- **出了哪些片段：**适用的 3 条，图在 `figures/`。每条有：
  - `<片段>_sidebyside.jpg`：人、机器人 0 特写、两台全景，各取 3 帧，用 1.0 倍；
  - `<片段>_levels_grid.jpg`：各档位图，每个报告档一行。左侧写"新量法"（判定 C）和"旧量法"（评测自己的判定，两项都用包围盒）的结论和未通过项；
  - `<片段>_render.json`。
- **出图的输入：**`levels/<片段>/levels_both.json`。
- **视频：**3 个 `<片段>_levels.mp4`，在 Drive `DuetMimic_transfer/T14_retargeting/step9_final/`。sha256 和大小都核对一致，见 `drive_manifest.csv`。

## 5. 时间、资源和偏离

- **时间（纽约，2026-11-27）：**

  | 时间 | 事项 |
  |---|---|
  | 00:13:25–00:42:25 | 7 条片段的源解、求解和评测，最多 6 个进程，绑在 16–31 号核 |
  | 00:42:25–00:42:31 | 非手穿透按碰撞形状重判，6 个进程；之后两机间隙重判、`feet_ik.csv` |
  | 00:42–00:54:33 | 出图等可用内存到 14 GB；当时一前一后训练任务的本机冒烟在跑 |
  | 00:54:33–00:59:50 | 出图和视频，3 条，一次一条 |
  | 00:59:50 | 打包脚本报错，见下 |
  | 01:5x | 修好重跑，产出 `inputs_outputs_sha256.csv`，共 67 行；01:56:50 视频上传 Drive |

- **资源：**只用 CPU（`CUDA_VISIBLE_DEVICES=""`），没有占 5070。
- **打包脚本报错：**
  - 原因：`step9e_package.py` 默认读 driver 记录里的 `source_root` 列表；这次没有用 `--source-root`，值是 None。
  - 修改：None 时当作空列表，源解从本次的 `src/` 读。原有目录的行为不变。
- **一个空目录：**`diag9_final/` 是打包脚本按步骤名建的。本次没有分段诊断，所以是空的；git 不收空目录。
- **.log 文件：**`.gitignore` 里有 `*.log`，拷进返包的日志一律改名为 `*_log.txt`。

## 6. 文件

- **结果表：**
  - `clips.csv`：每条片段一行；
  - `levels.csv`：每个评测档一行；
  - `totals.json`：合计。
  - 脚本 `system1/t14/objects/step9_final_table.py`。
- **重判：**
  - `clearance_exact/step9_final_clearance_exact.csv / .json`：每档两种两机间隙量法，以及 A / B / C 三种判定；
  - `nonhand_exact/step9_final_nonhand_exact.csv / .json`：每档新旧非手穿透。
- **其他表：**`feet_ik.csv`；`class_a_turns.csv / .json`（片段表）。
- **明细：**
  - `evals/`：23 个评测档的 eval.json，是评测自己的判定（两项都用包围盒）；
  - `diag/`：7 条片段的求解前诊断。
- **图：**`figures/`、`levels/`。
- **记录：**`records/`
  - 命令：`run.sh`（求解）、`post.sh`（重判、出表、出图、打包）；
  - 日志：`*_log.txt`；
  - `driver.json`、`run_records/`、`gates_logs/`、`failed_logs/`。
- **其他：**
  - `inputs_outputs_sha256.csv`：序列、源解、目标 URDF、配对、视频的 sha256；
  - `drive_manifest.csv`；
  - `SHA256SUMS`：本目录所有文件。
