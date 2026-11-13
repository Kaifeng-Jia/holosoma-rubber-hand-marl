# T06 第 2 步，方案 A：手指参考门槛核查（2026-11-13，CPU）

依据：`docs/system1/T06_dexterous_hand_zh.md` 第 2 步。11/13 第一版（v1，`../step2_gate/`）门槛没过，Kevin 选了方案 A："改手指规则，不动身体参考"。本目录是 A 的结果。全程只用 CPU（0–15 号核），没有用 GPU，拼图也是 CPU 软件渲染。

## A 和 v1 的区别

代码开关是 `system1/hand/finger_ref.py --variant A`，默认 `v1`，v1 的行为不变。

| | v1 | A |
|---|---|---|
| 接触帧 | 人手掌点到桶表面 < 4 cm（`source_contact_diagnostics.npz`） | 机器人手掌点到桶表面的有符号距离 < 4 cm |
| 优化哪些帧 | 只优化接触帧；非接触帧张开；过渡段事后按 smoothstep 混合 | 每只手所有帧一起优化 |
| 贴合项 | 接触帧权重 1 | 每帧乘以 0.2 s smoothstep 过渡权重（接触帧为 1，远离接触为 0） |
| 穿插、平滑、张开正则 | 只在接触帧 | 每一帧 |
| 1 cm 门槛 | 手上所有碰撞点 | 只算五根手指的连杆（含拇指掌骨、小指掌骨） |
| 手掌和腕座 | 算进门槛 | `hand_wrist`、`hand_C_MC` 和 flange 固连，由身体参考决定位置，单独报告 |

- **机器人手掌点**：Sharpa 的腕点加食指、中指、小指根关节点，四点取平均。和人手掌点的构造相同（腕、食指根、中指根、小指根取平均）。
- **没变的部分**：腕部重解、权重（贴合 1e4、穿插 1e6、平滑 10、张开 0.01、人手 0）、Adam 迭代 400 步、关节限位硬约束、接触容差 5 mm、桶的有符号距离场（3 mm 网格）。

## 用时

| 项 | 用时 |
|---|---|
| track | 973 s |
| locked | 2 s |
| 拼图渲染 | 约 2 分钟 |

命令（CPU 0–15 号核，8 线程）：

    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/finger_ref.py --mode track  --variant A --out logs_system1/t06/finger_ref/A_track
    taskset -c 0-15 hssim_s1/bin/python -B system1/hand/finger_ref.py --mode locked --variant A --out logs_system1/t06/finger_ref/A_locked

## 结果：门槛没过

### track 模式

| 手 | 接触帧 | 手指穿插 > 1 cm 的帧（全部帧中 / 其中接触帧） | 手指最深 | 最深的手指 | 手掌和腕座最深 | 手掌和腕座 > 1 cm 的帧 | 接触帧中掌 + 2 指 | 接触帧中 ≥ 2 指 | 机器人手掌点到桶最近 |
|---|---|---|---|---|---|---|---|---|---|
| 第 1 人左 | 27 | 97 / 11 | 4.3 cm | 拇指（96 帧） | 0.5 cm | 0 | 27/27 | 27/27 | 3.5 cm |
| 第 1 人右 | 0 | 0 / 0 | 0 | — | 0 | 0 | — | — | 4.8 cm |
| 第 2 人左 | 107 | 113 / 105 | 3.5 cm | 拇指（113 帧） | 1.8 cm | 20 | 102/107 | 107/107 | 1.8 cm |
| 第 2 人右 | 106 | 90 / 90 | 14.2 cm | 食指（85 帧）、拇指（5 帧） | 8.4 cm | 86 | 96/106 | 106/106 | −5.9 cm（在桶内） |

**门槛结果**：

| 项 | 结果 | 数值 |
|---|---|---|
| 手指穿插 ≤ 1 cm | 不过 | 最深 14.2 cm |
| 接触帧中掌 + 2 指 | 不过 | 两只手不满足：第 2 人左 102/107，第 2 人右 96/106 |
| 接触帧中 ≥ 2 指 | 全部满足 | — |
| 关节限位 | 全部在内 | — |

- **第 2 人右手**：有 40 帧机器人手掌点本身在桶网格里面（最深 5.9 cm），手指接在手掌上，出不去。
- **两只左手**：超过 1 cm 的穿插几乎都是拇指。
- **第 1 人右手**：机器人手掌点始终离桶 4.8 cm 以上，没有接触帧。

**手指相邻帧跳变**：

| 手 | 一帧变化 > 0.1 rad 的帧 | 其中 > 0.5 rad |
|---|---|---|
| 第 1 人左 | 17 | 4 |
| 第 1 人右 | 2 | 2 |
| 第 2 人左 | 27 | 4 |
| 第 2 人右 | 31 | 5 |

- 最大一帧 1.92 rad（96 rad/s），出现在拇指 MCP_FE 关节。
- 平滑权重 10 相对穿插权重 1e6 太小。这一项没有调。

### locked 模式（手指全程张开，接触帧按 A 判定）

| 手 | 接触帧 | 手指最深 | 手指 > 1 cm 的帧 | 接触帧中掌 + 2 指 |
|---|---|---|---|---|
| 第 1 人左 | 27 | 8.0 cm | 132 | 27/27 |
| 第 1 人右 | 0 | 3.1 cm | 50 | — |
| 第 2 人左 | 107 | 7.4 cm | 342 | 69/107 |
| 第 2 人右 | 106 | 16.2 cm | 120 | 96/106 |

### 和 v1 的对比（track）

| 手 | v1 接触帧 | v1 接触帧中穿插 > 1 cm | A 接触帧 | A 接触帧中手指 > 1 cm |
|---|---|---|---|---|
| 第 1 人左 | 64 | 64 | 27 | 11 |
| 第 1 人右 | 114 | 0 | 0 | 0 |
| 第 2 人左 | 237 | 113 | 107 | 105 |
| 第 2 人右 | 228 | 62 | 106 | 90 |

全局最深：v1 是 16.2 cm（所有碰撞点），A 是手指 14.2 cm。

## 关键帧拼图（`montage_track/`）

**渲染方式**
- 只做运动学回放，不步进物理。
- 场景、材质、地面、光照沿用 U5 / Fig. 4 的流水线（`system1/render/u5` 的 NativeScene），改动只有两处：
  - 去掉橡胶手和 29 关节的断言；
  - 每台机器人设置 73 个关节。
- 渲染器：Mesa llvmpipe（EGL，纯 CPU），没有用 GPU。

**颜色**
- G1 保持 Unitree 原色。
- Sharpa 原装配色是浅蓝灰，指尖弹性体是亮绿。为了和机器人统一，Sharpa 改成 G1 的灰色（0.7），指尖改成 G1 的深灰（0.2）。
- 桶的颜色沿用 U5 F3 Bin 的记录（灰色 0.4），不是浅蓝。

**图**
- `overview.png`：全段等间隔取 6 帧。
  - 相机的距离、方位角、俯角用 F3 Bin 的值（2.98 m、135°、−22°）。
  - 注视点改成每帧两个骨盆和桶的中心。F3 原来的注视点在评测的世界坐标里，参考在另一个位置，原样用的话机器人会偏到画面一角。
- `closeups.png`：每只手取它接触区间 25 %、50 %、75 % 处的 3 帧；第 1 人右手没有接触帧，取离桶最近的一帧。
  - 相机放在离机器人手掌点 0.55 m 处，方向自动选：
    - 相机不在桶内；
    - 视线不穿过桶，也不贴近机器人身体；
    - 尽量沿桶表面的法向看过去。
  - 每张图左上角写了帧号、手指穿插、手掌和腕座穿插、手掌点到桶的距离、接触手指数。
- `closeups_xray_diagnostic.png`：同样的视角，桶改成 30 % 透明度。只用来诊断，不是论文风格。
- `render.json`：输入文件和 sha256、每张图的相机、穿插数值。

**图里能看到的事实**
- 4 只手都是平贴在方形垃圾桶的侧壁上，手指基本张开，没有搭在桶沿上的动作。手的位置来自 D 的身体参考。
- 第 2 人右手的第 237、263 帧：这只手在自己的躯干和桶之间，从所有候选视角看都被躯干挡住，图里看不到这只手；透明桶的版本也看不到，因为挡住它的是机器人。

## 和任务书的偏离

- 门槛没过，按任务书停下回报，没有自行改规则或调权重。
- 拼图用 CPU 软件渲染，没用 GPU，没有和最终评测抢 GPU。

## 文件

| 文件 | 内容 |
|---|---|
| `track_summary.json`、`locked_summary.json` | 每只手的诊断、门槛结果、输入文件 sha256 |
| `track_per_frame.csv`、`locked_per_frame.csv` | 逐帧数据，比 v1 多出 `finger_penetration_m`、`rigid_penetration_m`、`robot_palm_point_sd_m`、`human_palm_distance_m` |
| `montage_track/` | 拼图（见上） |
| `SHA256SUMS` | 本目录文件的 sha256 |

参考本身只留在本地：

| 参考 | sha256 |
|---|---|
| `logs_system1/t06/finger_ref/A_track/finger_reference.npz` | `f520f80a3ebd1574c5bb24b6c44ba953956e9382c93afc057f24766ce8f5bfb8` |
| `logs_system1/t06/finger_ref/A_locked/finger_reference.npz` | `4fc9e3a9a05aa9ee01b37f801d940a962b9459d27f5c54ca4feb360aec45092e` |

渲染命令：

    CUDA_VISIBLE_DEVICES= __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json EGL_PLATFORM=surfaceless \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl taskset -c 0-15 sonic_s1/bin/python -B system1/hand/render_keyframes_sharpa.py \
        --reference logs_system1/t06/finger_ref/A_track --out-dir results_system1/T06/step2_A/montage_track
