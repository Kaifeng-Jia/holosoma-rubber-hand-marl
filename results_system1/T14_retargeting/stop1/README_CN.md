# T14 停点 1：适用范围清单和 1.0 倍物体尺寸

只读数据，只用 CPU。没有生成参考，没有训练。

## 0. Receipt

- 任务书：`docs/system1/T14_retargeting_zh.md`，sha256 `a1ed6e1570bc276a006e5db5ec02fe4e5b8f57cdbc91aa943c5287c46d69b5cf`。文件开头写"草稿"；按 Kevin 2026-11-23 的消息执行。
- 开始：2026-11-23 09:40 EST。
- 机器：kevin-Legion-R9000P-ADR10，RTX 5070 Laptop，开始时空闲显存 7341 MiB。本停点没有用 GPU。
- 本目录 git HEAD（开始时）：`9822434`。
- Python 环境：
  - `hssim_s1`：算物体尺寸；
  - `sonic_s1`：跑统计和截图，用到 smplx、trimesh、MuJoCo EGL CPU 渲染；
  - `tools_s1`：新建的环境，只装了 conda-forge 的 tmux 3.8。

## 1. 每个任务 1.0 倍物体的实际尺寸

- **1.0 倍的定义：** 现在 D 训练用的物体 URDF，也就是最终评测 `summary.json` 里的 `object_urdf`。
- **尺寸怎么量：** 取碰撞几何，按物体自身坐标轴量，单位米。
- **"两机器人连线方向"：** 参考第 0 帧两台机器人骨盆连线的水平方向。

| 任务 | 数据 | 物体 | 沿两机器人连线 | 横向 | 高 | 来源和缩放 |
|---|---|---|---|---|---|---|
| Push | OMOMO 单人 `sub6_largetable_033` | 5 块盒子拼成的桌子 | **1.400** | 0.522 | 0.428 | 源桌宽 0.522；1.0 倍把桌面沿局部 X 加长到 1.4 m。两机器人间距 0.878 m（角色等距配准值）。 |
| Pull | OMOMO 单人 `sub3_largetable_010` | 同一张桌子，加长方向在局部 Z | **1.400** | 0.522 | 0.428 | 同上，镜像角色。第 0 帧骨盆间距 1.052 m。 |
| Table | CORE4D 双人 20231030/001 desk001 | 5 块盒子近似的 desk001 | 0.592 | 0.444 | 0.541 | 原 desk001 网格 0.799 × 0.599 × 0.731，乘共享缩放 0.7412（两人 0.7477、0.7347 的平均）。第 0 帧骨盆间距 1.129 m。 |
| Bin | CORE4D 双人 20231020/071 bucket003 | bucket003 网格 | 0.541 | 0.466 | **0.763** | **没有缩放**：原网格原尺寸，manifest 写 `baseline_kind = fixed-object`。两人缩放 0.7615、0.7223。 |
| Chair | CORE4D 双人 20231020/074 chair021 | chair021 网格 × 0.7442 | 0.400 | 0.381 | 0.573 | 原网格 0.538 × 0.512 × 0.769，乘共享缩放 0.7442（两人 0.7617、0.7267 的平均）。第 0 帧椅子侧躺。 |

- **Bin 和任务书的定义对不上。** 任务书第 3 步写："双人的共享物体 = 原物体按两人缩放的平均值缩放，再乘 k。"
  - 照这个定义，Bin 在 k = 1 时是原桶 × 0.7419，即 0.401 × 0.346 × 0.566 m。
  - 现在训练用的是原尺寸桶。
  - Table 和 Chair 两种定义一致。
- 明细：`object_sizes_1p0.csv` / `.json`，含 URDF、原网格和参考的 sha256，以及第 0 帧的世界坐标尺寸。脚本：`system1/t14/object_sizes.py`。

## 2. 规则 1：能不能变成双人参考

- **CORE4D：** 本来就是双人，全部适用。
  - 本机共 920 条序列，6 类：box、board、bucket、stick、chair、desk。任务书里的 barrel 在数据里叫 bucket。
  - 有 8 条（20231030/065–072）用的 board021，本机两份 CORE4D 里都没有它的网格，没有统计。
- **OMOMO：现有的角色等距检查不能自动判定。**
  - 现有检查是 `scripts/run_reference_geometry_round2.py`，以及同目录的 `reference_geometry_round2_checks.py`、`_correspondence.py`、`reference_geometry_math.py`。本分支的文件和旧目录逐字节相同。
  - **它要求的输入：**
    - 源物体和加长后的目标物体都是"盒子并集"（`BoxUnion`）；
    - 已有的单台机器人参考；
    - 第一轮 pilot 标好的手–物体接触窗口。
  - **只跑过 Push 和 Pull 两例：**
    - **Pull：** 1/4 对角色通过；
    - **Push：** 0/36 对通过。冻结的源支撑脚漂移筛查没过，状态 `diagnostic_only`。论文用的 0.878 m 是"失败组合里最好的一对"。
  - **对其余 13 个物体：** 没有盒子近似，没有加长后的 1.0 倍物体，也没有单人重定向结果，所以这个检查现在跑不了。
  - **字面执行规则 1 时：** Push 本身也会被判"不适用"。
- **数据版本：** OMOMO 用的是原始发布数据：15 个物体，5882 条序列（train 5280、test 602），SMPL-X 参数，物体网格在 `captured_objects/`。
  - holosoma 重定向流水线用的是 InterMimic 处理过的 `OMOMO_new`，只有 13 个物体，没有 mop 和 vacuum；它的 `models/` 下也只有 largebox 和 largetable 的网格。

## 3. 规则 2：手掌贴面，还是手指包住细的部位

规则写在 `system1/t14/contact_parts.py` 开头，在算任何数之前定下。要点：

1. **手掌宽 W = 10.44 cm：** 机器人橡胶手碰撞网格有向包围盒的中间边长（4.4 × 10.4 × 14.2 cm）。作参照，SMPL-X 人手在指根处宽 8.5–9.6 cm（女 / 男模板）。
2. **人手：** SMPL-X 手部顶点（腕关节和 15 个手指关节的蒙皮顶点，每只手约 780 个）。
   - **CORE4D：** 用数据里存的逐帧顶点，15 Hz，每帧都算。
   - **OMOMO：** 用 SMPL-X 参数算顶点，按序列的性别模型和 16 个 beta，取每 3 帧一帧（10 Hz）。OMOMO 没有手指动作，手是平的。
3. **接触：** 手部顶点的最大有符号距离 ≥ −1.5 cm，同 T12。有符号距离用物体表面的密集采样点算：约每 (3 mm)² 一个点，符号按面法向定。
4. **细杆点（rod）：** 在表面一点周围半径 W 内，看表面法向：
   - 法向绕某一轴转满一圈（最大角缺口 < 105°），并且这些点在轴的横截面上的宽度 < W，就算细杆点；
   - 含义是这一段的截面比手掌还小，手只能包住它：棍、杆、腿、把手、窄横档。
   - 桌面或板的边缘只转半圈，不算细杆点；盒子的角也不算。
5. **一帧手是"细杆接触"：** 这只手的接触顶点里，超过一半的最近表面点是细杆点。
6. **辅助指标"对面接触"：** 同一只手同时贴着两个相对的面（法向点积 < −0.5），而且两面相距 < W，即手夹着或包着比手掌薄的部分。
   - 这个指标只对 CORE4D 有行为含义。OMOMO 的平手会直接穿过薄板，所以它的这一项不说明手在做什么。
7. **判定提议（Claude 提的阈值，没有外部出处，请 Kevin 确认或修改）：** 按细杆接触在所有接触手帧里的占比判：
   - < 20%：适用；
   - > 50%：不适用；
   - 其余：待定。

### 结果

"接触手帧"是一个人的一只手在一帧里接触物体。明细在 `rule2_summary.csv`（按类别和按物体）和 `rule2_per_sequence.csv`。

| 数据 | 类别 / 物体 | 有接触的序列 | 接触手帧 | 细杆接触 | 对面接触 | 规则 7 判定 | 备注（看截图和分物体统计） |
|---|---|---|---|---|---|---|---|
| CORE4D | box | 187/192 | 44 623 | 0.0% | 26% | 适用 | box022 的 4 条序列没有任何接触帧，原因未查 |
| CORE4D | board | 116/116 | 31 024 | 2.9% | **77%** | 适用 | 截图里手指托在板下沿：是捏板边，不是贴面。board005：细杆 47% |
| CORE4D | bucket（barrel） | 196/197 | 43 463 | 0.0% | 57% | 适用 | 手在桶沿或侧面凹槽处 |
| CORE4D | chair | 145/146 | 34 211 | 24.7% | 65% | **待定** | 分物体：chair022 57%，chair006 29%，chair021（我们 Chair 任务用的）12%，chair005 9%，chair020 4% |
| CORE4D | desk | 201/204 | 54 122 | 2.8% | **81%** | 适用 | 截图里手指勾在桌面下沿。desk001（我们 Table 任务用的）对面接触 90% |
| CORE4D | stick | 57/57 | 10 618 | 100% | 78% | 不适用 | |
| OMOMO | largebox | 336/379 | 28 249 | 0.0% | — | 适用 | |
| OMOMO | smallbox | 290/329 | 19 816 | 0.0% | — | 适用 | |
| OMOMO | plasticbox | 319/355 | 28 533 | 2.8% | — | 适用 | 截图里手在薄箱沿 |
| OMOMO | suitcase | 313/351 | 26 920 | 0.0% | — | 适用 | |
| OMOMO | trashcan | 288/332 | 19 105 | 0.0% | — | 适用 | |
| OMOMO | monitor | 332/332 | 27 382 | 1.9% | — | 适用 | 手在屏幕板面上 |
| OMOMO | largetable | 334/343 | 30 682 | 11.6% | — | 适用 | 细杆接触来自抓桌腿 |
| OMOMO | smalltable | 321/359 | 24 989 | 6.1% | — | 适用 | |
| OMOMO | clothesstand | 341/346 | 23 990 | 27.7% | — | **待定** | 截图里抓的是细立杆。立杆上的挂钩让局部截面超过 W，规则 4 漏判了这一段，所以 27.7% 偏低 |
| OMOMO | whitechair | 498/502 | 33 612 | 26.5% | — | **待定** | |
| OMOMO | woodchair | 490/490 | 46 062 | 34.2% | — | **待定** | |
| OMOMO | vacuum | 486/486 | 30 835 | 68.3% | — | 不适用 | |
| OMOMO | floorlamp | 325/338 | 23 868 | 99.9% | — | 不适用 | |
| OMOMO | mop | 475/477 | 27 676 | 100% | — | 不适用 | |
| OMOMO | tripod | 461/463 | 25 734 | 99.7% | — | 不适用 | |

- **截图：** `examples/core4d_examples.jpg`、`examples/omomo_examples_1.jpg`、`_2.jpg`。
  - **选帧规则**（渲染前定）：每类取按序列顺序第一条有接触的序列，再取它所有接触手帧按时间排序后的中间一帧。
  - **画面：** 左边是近景，右边是全景。物体浅蓝，规则 4 判的细杆部分为橙红。
  - 选中的帧记录在 `examples/examples.json`。
- **数据问题（如实记录，没有修）：**
  - OMOMO 有 39 条序列的部分帧物体位姿无效（scale ≤ 0 或非有限值），多数是 vacuum 下半部分。这些帧里无效的部件不参与；没有有效部件的帧不用。共 64 帧。
  - CORE4D 20231002/015（stick）的两个人帧数不同（物体 283 帧，其中一人 100 帧），按较短的算。
  - OMOMO 网格大多不封闭；desk020 和 monitor 的面朝向不一致，在内存里用 `fix_normals` 修了，文件没动。清单见 `rule2_provenance.json`。

## 4. 需要 Kevin 定的

1. **规则 2 的清单：**
   - 判定阈值（第 3 节第 7 条）要不要改；
   - "待定"的 chair、clothesstand、whitechair、woodchair 怎么判；
   - 捏板边、勾桌沿（board、desk 的对面接触 77–81%）算不算"手掌贴面"。我们的 Table 任务（desk001）就属于这一种。
2. **规则 1 对 OMOMO 无法自动判定。** 可选：
   - (a) OMOMO 只保留能做成盒子近似的桌子类（largetable、smalltable），用现有检查判；
   - (b) 把检查推广到网格物体，这是新代码；
   - (c) 规则 1 改成别的判据。

   另外，Push 本身没有通过现有检查（0/36）。
3. **其余 OMOMO 物体的 1.0 倍是多少，没有定义。** 现在只有 Push、Pull 两张桌子有加长后的 1.0 倍物体。
4. **Bin 的 1.0 倍：** 用现在的原尺寸桶，还是按任务书第 3 步缩到 0.742 倍？
5. **第 2 节以后要用的"论文设置表里的两机器人间距阈值"：** 本机没有论文源文件，没找到。到第 2 节时需要 Kevin 给值。现有检查里只有"两机器人包围盒重叠 ≤ 1 cm"（`pair_proxy_penetration_m`），而它的出处写的是"v0 工程筛查，非论文参数"。

## 5. 过程记录和偏离

- **规则 4 跑数后改了两处，均在第一次 30 序列 smoke 之后、全量之前：**
  - **(1) 邻域半径由 W/2 改为 W。** stick006（截面 3.7 × 7.2 cm）在偏离中心的点上，W/2 的球够不到对侧。
  - **(2) 加了"横截面宽度 < W"的条件。** 原来只看法向转一圈；桌面边缘下面有裙板时，法向也会转满一圈，但截面有 11–17 cm。
  - 两处都写进了脚本开头的规则说明。改完后用合成几何的 smoke test 核对过（`system1/t14/test_contact_parts.py`，2 项通过）。
- **另外加了两条数据处理规则，都是全量跑到一半出错后加的，不涉及判定阈值：**
  - **3b：** OMOMO 无效位姿帧跳过；
  - 缺网格的序列跳过并列出。
- **运行事故：** `contact_parts.py` 用 24 个进程跑全量，每个 1.3–1.6 GB，两次超过内存。时间和内核日志对得上：
  - **10:04 EST 那次：** 10:06 OOM 杀掉了 VS Code 和 Firefox 的页面进程。这次运行本身随后又因无效位姿报错退出。
  - **15:15 EST 那次：** 15:18–15:19 OOM 杀掉了 VS Code 和 Firefox，Kevin 手动停掉了脚本。
  - **15:28 EST 那次：** 我在看到 Kevin 的新规矩之前又用 24 个进程开了一次，14 s 后按记录的进程组自行停掉，当时还在第 0 条序列。
  - 改正：
    - 每个进程最多缓存 3 个物体（原来缓存全部 53 个）；
    - SMPL-X 每个性别只建一个固定批大小的模型（原来每个序列长度建一个）；
    - 任务按物体排序；
    - 在 tmux 里先用 4 个进程实测：每个进程峰值 0.97 GB；
    - 全量用 8 个进程，在 tmux 里跑。实测整组峰值 8.3 GB（主进程 0.83 GB，每个 worker 不超过 1.02 GB），用时 903 s，15:31–15:46 EST。
- **tmux：** 系统没有装 tmux，也没有 screen，apt 要 sudo，所以新建了 conda 环境 `tools_s1`，只装了 conda-forge 的 tmux 3.8（32 MB）。旧环境没动。
- **移到 `_to_delete/` 的**（都是本任务自己产生的中间输出，没有删除）：
  - `t14_obj_cache_radius_w2`、`t14_obj_cache_no_size_rule`：旧规则下的物体缓存；
  - `t14_smoke_*`：3 份 smoke 输出；
  - `t14_full_failed_board021`、`t14_full_failed_invalid_pose`、`t14_full_24workers_oom`、`t14_full_stopped_24workers`：4 次中断的全量日志；
  - `t14_memtest_4workers`：内存实测；
  - `t14_render_failed_len`：截图第一次失败的日志。
- **磁盘：** 剩 19 GB（95%）。本停点新增约 160 MB，在 `logs_system1/t14`，不进 git。

## 6. 复现

```bash
hssim_s1/bin/python -B system1/t14/object_sizes.py --out results_system1/T14_retargeting/stop1
# 在 tmux 里，8 个进程：
OMP_NUM_THREADS=1 sonic_s1/bin/python -B system1/t14/contact_parts.py --out results_system1/T14_retargeting/stop1 --workers 8
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  sonic_s1/bin/python -B system1/t14/render_contact_examples.py --out results_system1/T14_retargeting/stop1/examples
```

逐帧结果（`per_frame_flags.npy`）和物体缓存在 `logs_system1/t14/`，不进 git。
