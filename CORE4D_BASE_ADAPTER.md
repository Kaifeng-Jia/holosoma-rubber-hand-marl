# CORE4D → OmniRetarget Base Adapter

## 1. 文档目的

本文只定义 CORE4D 数据进入现有 OmniRetarget 基线的接口契约。它不是新的总路线图，也不改变 OmniRetarget 的优化器、机器人模型、reward、训练环境或多智能体网络。

本阶段固定链路如下：

```text
CORE4D-Real V1（官方完整 scene bundle）
  person1_poses.npz
  person2_poses.npz
  smooth_objposes.npy
  object_metadata.json + canonical object mesh
                    │
                    ▼
        CORE4D adapter（一次性可信输入边界）
                    │
                    ▼
       规范、纯数值、可校验的 NPZ + provenance
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  OmniRetarget(person 1)  OmniRetarget(person 2)
          │                   │
          └─────────┬─────────┘
                    ▼
       paired robot-object reference
                    ▼
                 ViSER 检查
                    ▼
          Isaac 单环境物理可行性检查
```

“分别运行两次”指同一段双人数据中的两个人分别进入现有单机器人 OmniRetarget 求解器；两次求解共享同一条物体轨迹、同一物体网格和同一时间轴。它不表示复制数据，也不表示当前就开始 MARL。

## 2. 官方数据事实

本节仅记录官方论文、官方数据定义和官方代码能够确认的事实。

### 2.1 一段 CORE4D-Real V1 序列

每段交互包含两个人和一个刚体物体。与本项目有关的文件是：

```text
<date>/<sequence>/
  person1_poses.npz
  person2_poses.npz
  smooth_objposes.npy
  object_metadata.json
  aligned_frame_ids.txt
```

物体网格位于 `object_models/<category>/<object_name>_m.obj`。官方说明网格位于物体类别的 canonical space，单位为米。

### 2.2 V1 中每个人的 SMPL-X 字段

官方加载方式为：

```python
human_motion = np.load(path, allow_pickle=True)["arr_0"].item()
```

字典字段如下：

| 字段 | 官方形状与类型 | 含义 |
|---|---:|---|
| `betas` | `(T, 10)`, `float32` | SMPL-X 体型；各帧实际相同 |
| `global_orient` | `(T, 3)`, `float32` | 根节点世界朝向，axis-angle |
| `transl` | `(T, 3)`, `float32` | 根节点世界位置 |
| `body_pose` | `(T, 21, 3)`, `float32` | 21 个身体关节的局部 axis-angle |
| `left_hand_pose` | `(T, 12)`, `float32` | 左手 12 维 PCA 参数 |
| `right_hand_pose` | `(T, 12)`, `float32` | 右手 12 维 PCA 参数 |
| `joints` | `(T, 127, 3)`, `float32` | SMPL-X 世界坐标关节位置 |
| `vertices` | `(T, 10475, 3)`, `float32` | SMPL-X 世界坐标顶点 |

官方可视化代码使用 neutral SMPL-X、10 个 beta、每只手 12 个 PCA 分量。原始 NPZ **不包含** wrist quaternion、joint name、FPS、速度或接触标签。

### 2.3 物体字段

- `smooth_objposes.npy` 是 `(T, 4, 4)` 的 `float64` 数组。
- 每帧矩阵表示 canonical object model 到 world coordinate system 的刚体变换。
- `object_metadata.json` 至少包含官方可视化程序使用的 `obj_name`，由此解析类别和 OBJ 文件。
- 数据没有提供仿真需要的质量、惯量、摩擦系数或简化 collision mesh。这些不是 adapter 可以从运动轨迹中可靠猜出的字段。

### 2.4 时间与同步

官方论文明确说明整个采集系统工作在 **15 FPS**；官方可视化程序也用 `1000 / 15` 毫秒生成每帧动画。因此 raw motion 的基准采样率设为 15 FPS，而不是沿用 OmniRetarget 配置中的默认 30 FPS。

Motion NPZ 本身没有逐帧时间戳。视觉数据目录另有 `timestamp.txt`，序列目录另有 `aligned_frame_ids.txt`，但官方文件定义没有说明 `aligned_frame_ids.txt` 的内部语义。本阶段只处理已对齐的人体—物体运动数组，不使用 RGB/RGB-D，因此不猜测该文件的含义。

### 2.5 坐标与标定

官方说明人体与物体轨迹均位于同一个 world coordinate system；物体网格单位为米。世界原点由三枚固定标记定义，外部相机通过世界坐标下的 3D 标记和图像 2D 点执行 PnP 标定。

官方文档没有明确写出 handedness、forward axis 和每根轴的正方向。官方 benchmark 代码把坐标第 1 维用于脚部高度，并在 X–Z 平面做朝向 canonicalization，因此 **Y-up 是有代码依据的判断**；它仍须在一个真实样本上用脚底、重力方向和物体落地姿态进行验证。不能只根据数组形状硬编码轴交换。

### 2.6 V2 与完整 V1 scene 的关系

远端真实包核查表明，`CORE4D_Real_human_object_motions_v2/batch*.zip` 不是上述完整五文件布局。V2 每段只包含更新后的人体 `person_1/result.npz`、`person_2/result.npz` 和少量抽帧 mesh/joints；物体轨迹、metadata 和对齐表仍须来自 V1。

V2 的 `result.npz` 也不是 V1 的 `arr_0` NumPy 字典：顶层键为 `results`，内部包含 Torch Tensor/Parameter。当前 adapter 因此先支持官方文档和可视化程序直接使用的 V1 完整序列。若以后采用 V2，应新增单独的可信 loader，并用同一序列逐项验证帧数、person identity、世界坐标、人体—物体相对位置和接触时序；provenance 必须明确记录 `human_version=V2`、`scene_version=V1`，不能将 hybrid 产物标成纯 V2。

## 3. Adapter 必须锁定的四项规范

### 3.1 人体关节规范

- 每段固定两个人，身份顺序保持 `person1`、`person2`，不按左右位置重新编号。
- OmniRetarget 使用的身体关键点固定为 22 个 SMPL-X body joints，顺序必须与本仓库的 `SMPLX_DEMO_JOINTS` 一致。
- 前 22 个关节按官方 SMPL-X body 顺序映射到本仓库的 `SMPLX_DEMO_JOINTS`；首次真实样本仍须通过可视化核验左右侧和腕部位置。
- 腕部世界 quaternion 不是 CORE4D 的原始字段，应由 `global_orient + body_pose` 沿 SMPL-X 运动链计算。为了直接复用现有 A1 校准接口，腕部输出使用 `xyzw`，归一化并执行时间符号连续化。
- Adapter 保存两个人各自的 10 维 beta；进入 Omni 前再用 neutral/rest SMPL-X 计算身高并确定**共享场景尺度**，不能让两次独立求解各自缩放同一条物体轨迹。

### 3.2 坐标与单位规范

- adapter 输出统一为 OmniRetarget 使用的 Z-up 世界坐标，长度单位为米。
- 同一个经过样本验证的刚体坐标变换必须同时作用于两个人、腕部朝向和物体世界轨迹。物体 canonical mesh 的局部坐标保持不变，world pose 左乘坐标变换；这与把 mesh 实例变换到新的世界系等价。
- adapter 阶段不偷偷缩放数据。人体到 G1 的比例缩放继续由 OmniRetarget 基线负责，并且人体、物体和接触关系必须采用一致的缩放策略。
- 输出前检查物体旋转正交性、`det(R)≈1`、齐次矩阵最后一行、四元数范数和所有数值的有限性。

### 3.3 时间规范

- raw FPS 固定记录为 15，不从文件扩展名或默认配置推测。
- `person1`、`person2` 和 object 必须具有相同 `T`；不允许静默截断到最短长度。
- 若 OmniRetarget 或后续跟踪需要 30/50 Hz，平移和关节位置做带明确时间戳的插值，旋转使用 SLERP；速度由重采样后的统一时间轴计算。
- 保存 source frame index/time mapping，使结果能够回溯到原始 15 FPS 帧。

### 3.4 物体规范

- 保留 `object pose = canonical object model → world` 的变换方向。
- 标准输出四元数顺序固定为 `[qw, qx, qy, qz]`，物体 pose 固定为 `[qw, qx, qy, qz, x, y, z]`。
- mesh、pose、人体必须共享同一单位、坐标变换和尺度策略。
- `obj_name`、源 mesh 路径、文件哈希和 sequence ID 放入 provenance；训练资产的质量、摩擦、惯量和 collision approximation 在进入 Isaac 前单独定义并记录，不能冒充 CORE4D 原始标注。

## 4. 规范纯数值 NPZ

Adapter 的下游产物不得包含 Python dict、object dtype 或 pickle。当前实现一个序列只输出一个 paired NPZ：

| 键 | 形状 | 约定 |
|---|---:|---|
| `fps` | scalar | 初始为 15 |
| `human_joints` | `(T, 2, 22, 3)` | Z-up、米、Omni body joint 顺序 |
| `human_joints_full` | `(T, 2, 127, 3)` | 保留手部/接触几何用途 |
| `betas` | `(2, 10)` | 两个人的固定 SMPL-X shape 参数 |
| `wrist_quat_xyzw` | `(T, 2, 2, 4)` | person × left/right × xyzw |
| `object_poses` | `(T, 7)` | wxyz + xyz |
| `joint_names` | `(22,)` | `SMPLX_DEMO_JOINTS` |
| `aligned_frame_ids` | `(T, K)` | 若官方文件存在则原样保留；否则为顺序帧号，不解释列语义 |

Sequence ID、mesh 路径、转换约定等 provenance 以普通 Unicode/JSON 字符串数组嵌入同一个 NPZ；它们不是 object dtype，因此整个文件仍可用 `allow_pickle=False` 安全读取。真实样本审计阶段再记录源文件 SHA-256。

现有 OmniRetarget 的 `smplx` 路径已经能读取单人的 `global_joint_positions + height`，但当前 `object_interaction` 路径仍硬编码读取 InterMimic `.pt`。因此 adapter 接入点应只新增一个 CORE4D loader，将 paired NPZ 的某个 person view 转成优化器已经使用的内存接口：

```text
human_joints: (T, 22, 3)
object_poses: (T, 7)
human_height: scalar
optional wrist quaternions: (T, 2, 4)
```

不要把 CORE4D 重新伪装成 InterMimic `.pt`，也不要修改优化器来理解 CORE4D 的原始 pickle 结构。

## 5. Pickle 信任边界

CORE4D 官方 person NPZ 把字典放在 `arr_0` object array 中，读取它必须启用 `allow_pickle=True`。Pickle 加载可能执行对象构造，因此边界固定如下：

1. 只接收官方仓库链接下载、固定 revision 并记录 SHA-256 的 raw 文件。
2. 只在 adapter 原始导入步骤启用 `allow_pickle=True`；这一步视为可信输入边界。
3. 立即验证顶层类型、允许的键集合、shape、dtype、frame count 和有限性；未知字段只记录，不传入下游。
4. 立即转存为纯数值 NPZ；此后 ViSER、OmniRetarget 和 Isaac 均使用默认 `allow_pickle=False`。
5. 第三方重新打包的 CORE4D NPZ 不进入该边界。

## 6. 阶段边界与验收顺序

### Stage A：最小官方样本与 raw audit

- 使用 CORE4D-Real V1 的完整五文件序列，而不是先下载全部 255 GB 数据、拼接尚未校验的 V1/V2 hybrid，或直接接入未充分文档化的 Synthetic 分支。
- 首个样本固定为 `20231003_2/000`：官方标签 `move2_obs0`、物体 `desk001`、197 帧。通过 HTTP Range 只提取该序列及其 object model，不下载约 34 GB 的完整 motion ZIP。
- 打印并保存字段、shape、dtype、帧数、数值范围和 SHA-256；验证 15 FPS、Y-up、两人—物体长度一致和 mesh/metadata 解析。

输出：raw audit 报告。此阶段不运行 OmniRetarget。

### Stage B：adapter 与 source visualization

- 生成规范纯数值 NPZ 和 provenance。
- 同时显示两个人、物体 mesh、世界坐标轴、腕部朝向和地面，确认左右身份、接触、尺度、上轴和物体变换方向。

输出：通过校验的 paired source NPZ。此阶段不生成机器人动作。

### Stage C：分别 OmniRetarget 两次

- Person 1 和 Person 2 使用同一个基线配置分别求解。
- 共享物体轨迹和时间轴；腕部处理沿用已经验证过的 rubber-hand 思路。
- 每次求解单独保存结果和 provenance，不让一个人的优化状态影响另一个人。

输出：两个 robot-object kinematic references。此阶段不训练。

### Stage D：paired reference 与 ViSER

- 将两条机器人 reference 按原时间轴合并，只保留一条共享 object reference。
- 检查机器人互穿、机器人—物体接触、脚底高度、双方相位和物体运动是否与 CORE4D 源动作一致。

输出：人工确认的 paired reference。

### Stage E：Isaac 单环境物理可行性

- 只建立一个双机器人、单物体环境。
- 明确写出质量、惯量、摩擦、碰撞几何、PD 参数和控制频率。
- 先做 reset/FK/contact/短 rollout 检查，不直接启动大规模 PPO 或 MARL。

输出：物理可行性报告。是否进入 WBT/MARL 是之后的独立决策。

## 7. 当前明确不做的事项

- 不修改现有 Push、Pull、Kick、Demo 4 的配置或 checkpoint。
- 不把两个机器人合并成一次 58 维动作输出。
- 不引入新的 reward、task-oriented 目标或混合动作训练。
- 不先接 CORE4D-Synthetic；官方目录只列出 `human_poses.npy / object_mesh.obj / object_poses.npy`，但没有在公开 file definition 中给出 `human_poses.npy` 的完整数组契约。
- 不依赖 RGB/RGB-D、camera calibration 或 `aligned_frame_ids.txt` 完成本阶段的 motion retarget。
- 不在数据 adapter 中猜测仿真质量、摩擦和惯量。

## 8. 下载与许可注意事项

- 官方 Hugging Face 数据仓库总计约 255 GB；CORE4D-Real human-object motion V2 单独约 39.6 GB，分为四个压缩包。Base adapter 首轮不需要 RGB/RGB-D 和 segmentation。
- V1 完整 motion ZIP 约 34.16 GB，但官方 Hugging Face endpoint 支持 HTTP Range；首个样本实际只需提取约 44.4 MB 的 motion/metadata 和一个约 0.17 MB 的 desk mesh。
- 官方 GitHub README 将该 work 标为 CC BY 4.0；Hugging Face dataset metadata 当前显示 MIT，部分 benchmark code 也单独使用 MIT。三处标记并不完全一致。内部研究阶段保留来源和引用即可；若要重新分发数据或转换产物，应先向作者确认 dataset 本体适用的许可。
- SMPL-X 模型需要从 SMPL-X 官方渠道单独取得并遵守其许可；它不是 CORE4D 文件许可的自动组成部分。

## 9. 官方依据

- [CORE4D 官方主页](https://core4d.github.io/)
- [CORE4D 官方代码与下载入口](https://github.com/leolyliu/CORE4D-Instructions)
- [官方文件定义](https://github.com/leolyliu/CORE4D-Instructions/blob/main/docs/file_definitions.md)
- [官方论文及补充材料（arXiv HTML）](https://arxiv.org/html/2406.19353)
- [官方数据仓库（Hugging Face）](https://huggingface.co/datasets/leolyliu/CORE4D/tree/main)
- [官方人体—物体可视化代码](https://github.com/leolyliu/CORE4D-Instructions/blob/main/dataset_utils/visualize_human_object_motion.py)
- [官方 benchmark 预处理代码](https://github.com/leolyliu/CORE4D-Instructions/blob/main/benchmarks/motion_forecasting/MDM_InterDiff/interdiff/data/prepare_hho.py)
