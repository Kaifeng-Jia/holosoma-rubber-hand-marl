# T06 第 0 步：调研（2026-11-13，不用 GPU）

依据：`docs/system1/T06_dexterous_hand_zh.md`（sha256 `23478d6c…eabc8a`，提交 5ef497e）第 0 步。本步只读数据和代码，没有用 GPU。

## 手的模型

- **仓库**：`sharpa-robotics/sharpa-urdf-usd-xml`，克隆到 `assets/sharpa/`，未进 git。
  - 提交号 `d323b371d02f4fc04a0873c19c5234cae6d803dd`，2026-11-05。
  - 含 Git LFS 共 718 MB。
  - 许可证 Apache 2.0（`LICENSE.txt`、`NOTICE.txt`）。
- **所用文件**：`wave_01/{left,right}_sharpa_wave/*_with_flange.{urdf,usda,xml}`。

  | 文件 | sha256 |
  |---|---|
  | left `…with_flange.urdf` | `6ea8d2c563c6d4ba56f5f4b04aa7d339f5b703ac8e5ec02404ca5f632d7c28a2` |
  | left `…with_flange.usda` | `db13e532452033b754da1c0783e6bb871d9244551e25203668eb5e556bf0e5f7` |
  | left `…with_flange.xml` | `8041d8ec725ad0aad0c14fc92b34641d4c0168e5a32833d53dc6dd34570fc62e` |
  | right `…with_flange.urdf` | `261c1a5284e04bc9690e0f5ac39d7a7cace0a16d335d663a1d3c7fcf4e4daf31` |
  | right `…with_flange.usda` | `93b536684aa4e140718e52363c4f5fada518211cd053fbf42b499aee57ad4ef7` |
  | right `…with_flange.xml` | `8b1dd21a7ab1217b7fd78789b455f3f03029b21bfdee25b5fa9381c8e5b5e643` |

- **单手结构**（右手 URDF）：
  - 35 个 link、34 个 joint，其中 22 个可动转动关节，没有 mimic 耦合；
  - 总质量 1.310 kg，和任务书一致；
  - 27 个碰撞体，25 个 STL 网格；
  - 根是 `right_hand_flange`，各指从 `right_hand_C_MC` 分出；
  - 每个指尖有三组固定坐标系：`DP`、`elastomer`、`fingertip`。
- **手指 PD**：取自官方 USD（为 Isaac Lab 配的 position mode 参数），22 个关节每只手都齐全，逐关节数值见 `sharpa_finger_pd.csv`。
  - **单位**：USD 转动驱动的刚度和阻尼以"每度"为单位存放，表里已乘 180/π 换算成每弧度。这个换算在第 1 步加载后从 PhysX 读回来核对。
  - **主要数值**：

    | 关节 | kp（N·m/rad） | kd（N·m·s/rad） | 力矩上限（N·m） |
    |---|---|---|---|
    | MCP 屈伸 | 4.760 | 0.183 | 1.864 |
    | MCP 侧摆 | 6.622 | 0.208 | 1.864 |
    | PIP | 0.908 | 0.040 | 0.638 |
    | DIP | 0.904 | 0.0315 | 0.189 |
    | 拇指 CMC 屈伸 | 6.955 | — | 3.3 |
    | 拇指 CMC 侧摆 | 13.20 | — | 3.3 |
    | 小指 CMC | 1.380 | — | 0.5285 |

  - **其他物理参数**：各关节 armature 为 0.00012–0.0032，关节摩擦 0.0004–0.132。
  - **learn 模式的动作缩放**（0.25 × 力矩上限 / kp）：MCP 屈伸 0.098 rad，侧摆 0.070，PIP 0.176，DIP 0.052，拇指 CMC 0.119 和 0.063。

## 1. 数据

**OMOMO**（本机 `/home/kevin/holosoma/OMOMO_new`，InterMimic 格式，4421 条，每帧 591 维）
- 第 162–318 列是 52 个 SMPL-H 关节的位置，其中含两手各 15 个手指关节。
- 随机抽 40 条检查：手指"腕–指尖距离 / 腕–掌指关节距离"在整段内的标准差，中位数 1.4e-6，最大 2.4e-6。也就是说手指是固定姿态，**OMOMO 没有手指运动数据**。

**CORE4D**（本机 `/home/kevin/datasets/CORE4D-V1-full`）
- **格式**：每人每帧是 SMPL-X 参数。
  - `left_hand_pose`、`right_hand_pose` 各 12 维 MANO PCA 系数；
  - 另附 10475 个网格顶点和 127 个关节；
  - 关节里含两手各 15 个手指关节（索引 25–54）和 10 个指尖点（索引 66–75）。
- **Bin 那一段**（20231020/071，物体 bucket003）：两个人都有手指数据，150 帧，15 Hz，约 10 s。
- **数据质量**：

  | 人/手 | 指尖到桶表面最小距离 | ≥2 根指尖在 3 cm 内的帧 | ≥2 根指尖在 1 cm 内的帧 |
  |---|---|---|---|
  | person1 左 | 0.000 m | 37/150 | 14/150 |
  | person1 右 | 0.000 m | 40/150 | 25/150 |
  | person2 左 | 0.000 m | 84/150 | 6/150 |
  | person2 右 | 0.000 m | 73/150 | 56/150 |

  - 平滑性：手指 PCA 系数的帧间变化平均约 0.01；同一系数在整段内的标准差约 0.16–0.24，是帧间变化的十几到二十几倍。
  - 穿插测不准：CORE4D 和仿真共用同一个桶网格 `bucket003_m.obj`（sha256 `2a95d824…`）。它是封闭实体，体积 0.0998 m³，为凸包的 77%，竖直射线穿过中心只交上下两面，说明桶内腔被填实了。所以手伸进桶口也算作"在物体内部"。按这个网格算，手部关节最大"穿插"为 1.2–7.2 cm，person2 两手各约 70 帧超过 1 cm。这个数反映的是封闭网格，不是真实的手–桶穿插。
  - 现有流程里 Bin 接触帧的定义（`source_contact_diagnostics.json`）：
    - 掌点：取 SMPL-X 的腕、食指根、中指根、小指根四个关节的平均；
    - 距离：掌点到桶网格三角面的精确最近距离；
    - 软带：2 cm 内权重为 1，4 cm 外为 0，中间线性；
    - 15 Hz 线性插到 50 Hz。

    第 2 步沿用这个定义。
  - **可用性**：可以给 w > 0 的演示参考提供 q_f^人手。办法是用 CORE4D 的手指关节点和指尖点，做 Sharpa 手指的位置 IK。不需要 MANO 模型文件，因为已经有逐帧关节位置。

## 2. 代码：橡胶手怎么装、腕部优化在哪、换手要改什么

**橡胶手的安装**
- **仿真模型**：`src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf`，由 `config_values/robot.py` 的 `g1_29dof_w_object` 引用。
  - 安装方式：固定关节 `{left,right}_hand_palm_joint`，父连杆是 `{side}_wrist_yaw_link`，原点 xyz=(0.0415, ±0.003, 0)，rpy=0。
  - 每只手 0.170 kg，碰撞体是 STL 网格。
  - 动作缩放 0.25 × 力矩上限 / kp（`action_scales_by_effort_limit_over_p_gain=True`）。
- **重定向模型**：`src/holosoma_retargeting/.../models/g1/g1_29dof.xml` 和 `.urdf`，橡胶手是挂在 wrist_yaw 上的 geom。

**腕部优化**
- **脚本**：`scripts/refine_core4d_pair_a1.py`，即 A1 阶段的 `wrist-only` 模式。
  - 只解两臂的腕部 roll/pitch/yaw，qpos 索引 26–28 和 33–35；其余关节和物体位姿逐帧保持不变。
  - 求解器误差上限 0.01°。
  - Bin 参考 `core4d_bucket003_20231020_071_a1` 的 `source_manifest.json` 里有记录：`solver_mode: wrist-only`，四个人手的标定误差都 ≤ 0.09°。
- **人手掌面系**：`holosoma_retargeting/src/utils.py: build_pt_wrist_palm_orientation_targets`。
  - 前向：腕 → 中指根；横向：食指根 − 小指根；法向由前两者叉乘得到。
  - 用整段估一个常值的腕到掌旋转。
- **机器人掌面系**：`src/interaction_mesh_retargeter.py: _init_hand_orientation`。
  - 在 `{side}_wrist_yaw_link` 坐标系里，写死了橡胶手的掌法向：左 (−0.0751, −0.9954, −0.0594)，右 (−0.0751, 0.9954, −0.0593)。
  - 手指方向取 +x 在掌面上的投影，横向由叉乘得到。
  - 换 Sharpa 时只换这组掌面基，求解器不变。

**换手要改的地方**（工作量是我估的人日）

| 项 | 要改的内容 | 估计 |
|---|---|---|
| 模型 | 新 URDF：G1 去掉两只橡胶手，在同一 palm 关节位置接两只 Sharpa flange 版（新文件，旧文件不动）。仿真用 URDF 导入；重定向、可视化另用一份 MJCF 或 URDF | 1 |
| 碰撞 | 每只手 27 个网格碰撞体。保留 USD 里的手内过滤对，确认和腕、前臂之间没有初始穿插。G1 现在不开自碰撞（`enable_self_collisions=False`） | 0.5 |
| 参考生成 | 腕部：换 Sharpa 掌面基后重跑 wrist-only。手指：新写逐帧优化（第 2 步）。参考 npz 增加两只手各 22 个手指关节 | 2 |
| 观测 | D 的 158 维之外，加本机器人两只手 44 个手指关节位置和 44 个手指参考，actor 和 critic 都加 | 0.5 |
| 动作 | 现在 `DualJointPositionActionTerm` 每个 agent 的动作维数等于 `env.num_dof`。要新写带 `finger_mode` 的动作项：locked、track 下策略仍输出 29 维，手指目标取张开姿态或参考；learn 输出 29+44 维，手指部分做有界修正，输出层初始化为 0 | 1 |
| PD | 22×2 个手指关节的 kp/kd/力矩上限写进机器人配置。armature 和关节摩擦是逐关节的，现有配置只有全局 armature，需要确认能不能按关节设 | 0.5 |
| 评测 | 评测器支持 `--finger-mode`，冻结状态要补手指初值，新增接触指标 | 1 |
| 渲染 | U5 渲染要加 Sharpa 的视觉网格和手指关节轨迹 | 0.5–1 |
| 合计 | | 约 7–7.5 |

**待定**：±2、±8 冻结初始状态只有 29 个关节。手指初值用张开姿态还是参考的第 0 帧，需要定一个规则，我先按参考第 0 帧设计。

## 3. 训练开销（估计，第 1 步加载后实测）

- **基线**：D Bin 论文主结果，种子 721。单独一张 RTX 4090，2048 个环境，每轮中位数 5.29 s，显存 6040 MiB，GPU 利用率 66%。
- **规模变化**：每个环境 2 台机器人。

  | 项 | 橡胶手 | Sharpa（4 只手 88 个手指关节） | 倍数 |
  |---|---|---|---|
  | 自由度 | 58 | 146 | 2.5 |
  | 刚体 | 约 61 | 约 153 | 2.5 |
  | 碰撞体 | 约 82 | 约 186 | 2.3 |

- **显存估计**：12–16 GB。2048 个环境能放进一张 24 GB 的 4090，但不能和别的训练共卡。
- **每轮耗时估计**：11–16 s，12000 轮约 37–53 小时，所以每个 run 要独占一张 4090。
- **风险**：PhysX 的 `gpu_max_rigid_patch_count` 在 `simulator/isaacsim/isaacsim.py` 写死为 10·2^15。手指接触变多后可能不够；如果出现接触丢失，要先回报再改。
- 本机 5070（8 GB）放不下 2048 个环境。第 1 步只用少量环境做加载测试。

## 和任务书的偏离

没有。

## 文件

- `sharpa_finger_pd.csv`：44 个手指关节的 USD 原值、换算后的 kp/kd、armature、摩擦、URDF 限位、力矩、速度，以及 learn 模式的动作缩放。
- `SHA256SUMS`。
