# T06 第 2 步，方案 A：按修订门槛核查（2026-11-13，CPU）

依据：`docs/system1/T06_dexterous_hand_zh.md` 第 2 步，2026-11-13 修订版（2e62b67，sha256 `98c55f9c…ab7bf89`）。第一版（v1，`../step2_gate/`）门槛没过以后，Kevin 选了方案 A，任务书随之修订了门槛。

全程只用 CPU（0–15 号核）。拼图也是 CPU 软件渲染，没有用 GPU。

**本目录取代 2bd7878 的版本。** 那一版是在任务书修订之前按草稿定义算的：
- 拇指掌骨（thumb_MC）算在手指里；
- 手掌组是手掌加腕座；
- 门槛不和橡胶手比较。

那一版的结论作废，日志改名留在本地 `logs_system1/t06/finger_ref/A_*_draftgroups_2bd7878/`。

## 修订后的门槛（任务书原文的意思）

1. 手掌组（手掌 + 拇指掌骨 thumb_MC）每帧的穿插，不超过同一帧原 D 橡胶手参考的穿插；
2. 其余手指的穿插不超过 1 cm；
3. 按机器人手判定的接触帧里，至少 80% 满足"手掌 + 至少 2 根手指"接触；
4. 提交关键帧拼图后停下，Kevin 看过点头。

## 做了什么

代码开关：`system1/hand/finger_ref.py --variant A`。默认是 `v1`，v1 的行为不变。

**接触帧**
- 用机器人手掌点判定：Sharpa 的腕点加食指、中指、小指根关节点，四点取平均，和人手掌点的构造相同。
- 该点到桶表面的有符号距离 < 4 cm 的帧算接触帧，4 cm 和现有流程的外沿阈值相同。

**优化**
- 每只手的所有帧一起优化。
- 贴合项每帧乘以 0.2 s smoothstep 过渡权重：接触帧为 1，远离接触为 0。
- 穿插、平滑、张开正则作用在每一帧。
- 这些量沿用 v1 不变：
  - 权重：贴合 1e4、穿插 1e6、平滑 10、张开 0.01、人手 0；
  - Adam 迭代 400 步；
  - 关节限位作为硬约束。

**分组**
- 手掌组 = `hand_C_MC` + `thumb_MC` + `hand_wrist`。
- 任务书写的是手掌 + thumb_MC。腕座 `hand_wrist` 和 flange 固连，位置由身体参考决定，任务书没提。我把它算进手掌组，不算进手指。
- 不含腕座的手掌组（palm_core）也单独算了，门槛 1、3 的结果和含腕座完全相同（见 summary.json）。
- 拇指组不含 thumb_MC。

**门槛评估**
- Sharpa 手和原橡胶手都按约 3 mm 间距密集采样：所有网格顶点，加上面积 / (3 mm)² 个表面均匀采样点。每只 Sharpa 手 39552 个点。
- 有符号距离场仍是 3 mm 网格。
- 橡胶手的穿插：`main_mesh_collision_rubberhand.urdf` 里 `<side>_rubber_hand_link` 的碰撞网格，按原 D 参考里该连杆的逐帧位姿计算。
- "接触"：该组最近点离表面不超过 5 mm。

**用时**

| 项 | 用时 |
|---|---|
| track | 1023 s |
| locked | 约 10 s |
| 拼图渲染 | 约 2 分钟 |

命令（CPU 0–15 号核，8 线程）：

    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/finger_ref.py --mode track  --variant A --out logs_system1/t06/finger_ref/A_track
    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/finger_ref.py --mode locked --variant A --out logs_system1/t06/finger_ref/A_locked

## 结果：门槛 1、2 没过，门槛 3 过了

### track 模式

| 手 | 接触帧 | 门槛 1：手掌组比橡胶手深的帧（其中深 > 1 mm） | 最多深出 | 手掌组最深 / 橡胶手最深 | 门槛 2：其余手指最深 | 手指 > 1 cm 的帧（其中接触帧） | 最深的手指 | 门槛 3：掌 + 2 指 |
|---|---|---|---|---|---|---|---|---|
| 第 1 人左 | 27 | 44（30） | 0.4 cm | 3.0 / 2.8 cm | 4.3 cm | 95（11） | 拇指 94 帧 | 27/27 = 100% |
| 第 1 人右 | 0 | 0 | — | 0 / 0.4 cm | 0 | 0 | — | 无接触帧 |
| 第 2 人左 | 107 | 28（8） | 0.2 cm | 3.5 / 3.7 cm | 2.8 cm | 90（89） | 拇指 90 帧 | 107/107 = 100% |
| 第 2 人右 | 106 | 7（5） | 0.3 cm | 8.5 / 9.9 cm | 14.2 cm | 87（87） | 食指 87 帧 | 106/106 = 100% |

- **门槛 1 不过**：
  - 三只手都有手掌组比橡胶手深的帧，共 79 帧，其中 43 帧深出超过 1 mm。
  - 最多深出 0.4 cm。
- **门槛 2 不过**：
  - 其余手指最深 14.2 cm（第 2 人右手的食指）。
  - 这一只手有 40 帧，机器人手掌点本身在桶网格里面，最深 5.9 cm。
- **门槛 3 通过**：有接触帧的 3 只手都是 100%。
- **关节限位**：全部在内。

手指超过 1 cm 的那些帧，同一帧的手掌组和原橡胶手是什么情况：

| 手 | 手指 > 1 cm 的帧 | 其中手掌组也 > 1 cm | 其中原橡胶手也 > 1 cm |
|---|---|---|---|
| 第 1 人左 | 95 | 94 | 95 |
| 第 2 人左 | 90 | 90 | 90 |
| 第 2 人右 | 87 | 87 | 87 |

也就是说，手指穿插超过 1 cm 的帧，全部是原 D 橡胶手在同一帧也插进桶网格超过 1 cm 的帧。

**手指相邻帧跳变**（没有调，不在门槛里）：

| 手 | 一帧变化 > 0.1 rad 的帧 | 其中 > 0.5 rad |
|---|---|---|
| 第 1 人左 | 17 | 4 |
| 第 1 人右 | 2 | 2 |
| 第 2 人左 | 27 | 4 |
| 第 2 人右 | 33 | 5 |

最大一帧 1.92 rad，相当于 96 rad/s，出现在拇指 MCP_FE。平滑权重 10 相对穿插权重 1e6 很小。

### locked 模式（手指全程张开，接触帧按 A 判定）

| 手 | 门槛 1：手掌组比橡胶手深的帧 | 门槛 2：其余手指最深 / > 1 cm 的帧 | 门槛 3：掌 + 2 指 |
|---|---|---|---|
| 第 1 人左 | 72 | 8.0 cm / 132 | 100% |
| 第 1 人右 | 0 | 3.1 cm / 50 | 无接触帧 |
| 第 2 人左 | 156 | 7.4 cm / 343 | 65.4% |
| 第 2 人右 | 16 | 16.2 cm / 120 | 92.5% |

## 关键帧拼图（`montage_track/`，用 track 参考）

**渲染方式**
- 只做运动学回放，不步进物理。
- 场景、材质、地面、光照沿用 U5 / Fig. 4 的流水线（`system1/render/u5` 的 NativeScene），只改了两处：
  - 去掉橡胶手和 29 个关节的断言；
  - 每台机器人设置 73 个关节。
- 渲染器：Mesa llvmpipe（EGL，纯 CPU）。

**颜色**
- G1 用 Unitree 原色。
- Sharpa 原装配色是浅蓝灰，指尖弹性体是亮绿。这里改成 G1 的灰色（0.7），指尖改成 G1 的深灰（0.2）。
- 桶的颜色沿用 U5 F3 Bin 的记录（灰色 0.4），不是浅蓝。

**图**
- `overview.png`：全段等间隔取 6 帧。
  - 相机的距离、方位角、俯角用 F3 Bin 的值：2.98 m、135°、−22°。
  - 注视点改成每帧两个骨盆和桶的中心。F3 原来的注视点在评测的世界坐标里，参考在另一个位置。
- `closeups.png`：每只手取它接触区间 25 %、50 %、75 % 处的 3 帧；第 1 人右手没有接触帧，取离桶最近的一帧。
  - 相机离机器人手掌点 0.55 m，方向自动选：
    - 相机不在桶内；
    - 视线不穿过桶、不贴近机器人身体；
    - 尽量沿桶表面的法向看过去。
  - 左上角写了：帧号、其余手指穿插、手掌组穿插、同一帧橡胶手穿插、接触手指数。
- `closeups_xray_diagnostic.png`：同样的视角，桶改为 30 % 透明度。只用来诊断，不是论文风格。
- `render.json`：输入文件和 sha256、每张图的相机、数值。

**图里能看到的事实**
- 4 只手都平贴在方形垃圾桶的侧壁上，手指基本张开，没有搭在桶沿上的动作。手的位置来自 D 的身体参考。
- 第 2 人右手第 237、263 帧，这只手在自己的躯干和桶之间，所有候选视角都被躯干挡住，图里看不到。

## 和任务书的偏离

- 门槛 1、2 没过。按任务书停下回报，没有自行改规则或调权重。
- 手掌组多算了腕座 hand_wrist（理由见上）。不算腕座时，门槛 1、3 的结果相同。

## 文件

| 文件 | 内容 |
|---|---|
| `track_summary.json`、`locked_summary.json` | 每只手的诊断；`gate_revised` 是修订门槛的结果；`hands.*.gate_revised` 是逐手数字 |
| `track_per_frame.csv`、`locked_per_frame.csv` | 逐帧数据。`rev_*` 是修订门槛的量，`rubber_hand_penetration_m` 是同一帧橡胶手的穿插 |
| `montage_track/` | 拼图（见上） |
| `SHA256SUMS` | 本目录文件的 sha256 |

参考本身只留在本地：

| 参考 | sha256 |
|---|---|
| `logs_system1/t06/finger_ref/A_track/finger_reference.npz` | `69e404bd6c1c8ce1c2ac3f1efa1e7830270687844e521ac1bf01e0a6ed01010c` |
| `logs_system1/t06/finger_ref/A_locked/finger_reference.npz` | `4fc9e3a9a05aa9ee01b37f801d940a962b9459d27f5c54ca4feb360aec45092e` |

渲染命令：

    CUDA_VISIBLE_DEVICES= __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json EGL_PLATFORM=surfaceless \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl taskset -c 0-15 sonic_s1/bin/python -B system1/hand/render_keyframes_sharpa.py \
        --reference logs_system1/t06/finger_ref/A_track --out-dir results_system1/T06/step2_A/montage_track
