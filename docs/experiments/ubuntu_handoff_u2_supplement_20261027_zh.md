# U2 补测交接：共享场景中的机器人手–物表面距离

2026-10-27。Ubuntu 端从本文件即可执行，不需要访问 Mac、论文或评审文件。**只用 CPU，截止 2026-10-29。** 如果缺输入、定义核查失败或预计赶不上，请提前向作者反馈具体缺项及预计交付时间；能独立完成的任务继续做。不要估数。

本轮测量 Chair、Small Table、Bucket 的既有腕部优化前参考。只新增测量脚本和独立输出目录；不训练、不运行策略、不重新优化机器人、不运行腕部优化，也不改动既有 R2 交付物或训练资产。

## 1. 固定的比较与输出

| `task` | 变体 A | 变体 B | 机器人参考 |
|---|---|---|---|
| `Chair` | `one-sided` | `mean` | 两变体均为同一套 Stage 1 nominal_scaled 机器人姿态 |
| `Small Table` | `one-sided` | `mean` | 两变体均为同一套 Stage 1 nominal_scaled 机器人姿态 |
| `Bucket` | `nominal` | `re-optimized` | Stage 1 nominal_scaled 与已完成的 Stage 2 fixed-object 姿态 |

主文件名和字段原样保留：

```csv
task,variant,participant,hand,n_window_frames,dist_mean_mm,dist_median_mm,dist_p90_mm,frac_inside
```

文件名为 `ref_shared_scene_handdist.csv`，**恰好 24 行数据**：每任务两个变体 × `participant` 为 `0,1` × `hand` 为 `left,right`。`participant=0` 对应原数据 `person1`，`participant=1` 对应 `person2`，左右手不互换。每个变体各占一行，通过 `(task,participant,hand)` 配对比较；两个变体使用完全相同的窗口帧。

这是窗口内机器人手点到表面的**绝对距离**，不是人手距离减机器人手距离，也不是原 U2 的 `dd_mean_mm`。源人手仅负责确定窗口，不要求它与机器人手点一一重合。

## 2. Ubuntu 已有输入及查找入口

仓库为 `Kaifeng-Jia/holosoma-rubber-hand-marl`，已有工作目录 `/home/kevin/holosoma-core4d-base`，以下简称 `ROOT`。保持现有工作树和资产，不执行 reset。已知 CPU Python：`/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python`。

沿用 R2 的来源记录：`checks/u2_definition_gap.json`、`CPU_first_batch/checks/shared_scene_sources.json`、`CPU_first_batch/checks/wrist_recomputed.json`、`sources/file_mapping.json`。它们只作为核验入口，本文件已给出执行定义。R2 缺的是距离定义和冻结窗口，不是要求重做 retargeting。

### 2.1 源人和源物体

| 任务 | canonical NPZ（Ubuntu 绝对路径） | 原帧数 / FPS | 本轮参考帧数 / FPS |
|---|---|---|---|
| Chair | `/home/kevin/holosoma-core4d-base/logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/source_canonical.npz` | 118 / 15 | 235 / 30 |
| Small Table | `/home/kevin/datasets/CORE4D-V1-full/derived/20231030_001_desk001_v1_canonical.npz` | 207 / 15 | 413 / 30 |
| Bucket | `/home/kevin/holosoma-core4d-base/logs/Core4DPreviews/bucket003_20231020_071_20260918/source_canonical.npz` | 150 / 15 | 299 / 30 |

对应已记录 SHA256：

```text
Chair canonical:       e496e71901d50808b1d29d533b358117c64a1301b5244dfa134850fb8d729acc
Small Table canonical: 5e27c929999bde64187ec5d71d4da02ea93bb12f78fedfcb36f7200e7990ed90
Bucket canonical:      1208deb13bf70daee2d42343c6728656cca8586ae46efc7a2bb718d514079083
```

使用 canonical 的 `human_joints_full`（形状 `T×2×127×3`）、`object_poses`、`fps` 和 `object_mesh_path`。这些人点已经转到米制、Z-up 世界系；mesh 保留 CORE4D 原物体局部坐标。**不要再旋转 mesh 或再做人点坐标变换，也不要对窗口的人点做身高缩放或减去每人的 ground offset。**

源表面是以下原始三角网格，尺寸系数为 1：

| 任务 | 源 mesh 路径 | SHA256 |
|---|---|---|
| Chair | `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models/chair/chair021_m.obj` | `7fb3283f1b67c8d08bb1f68c51d15aab5e6c940b8571e704bca265701b776161` |
| Small Table | `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models/desk/desk001_m.obj` | `f8f9dffa9f1ccdbd2ff90125cd39c96e79bf84ccebd8a81fcef656c4fbc7c00b` |
| Bucket | `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models/bucket/bucket003_m.obj` | `2a95d824af8facbd363ce00eba0534cd7c341b51f9997e3b137990f5fb503bdf` |

### 2.2 既有机器人参考

定义三个目录别名，仅用于减少下面的路径重复：

```text
CHAIR = /home/kevin/holosoma-core4d-base/logs/Core4DPreviews/chair021_20231020_074_baseline_20260908
TABLE = /home/kevin/datasets/CORE4D-V1-full/retarget_results/method_comparison_20260904/20231030_001_desk001_move
BUCKET = /home/kevin/holosoma-core4d-base/logs/Core4DPreviews/bucket003_20231020_071_20260918
```

- Chair 两人的 nominal：`CHAIR/two_stage/person1/person1_nominal_scaled.npz` 和 `CHAIR/two_stage/person2/person2_nominal_scaled.npz`。已有 mean 配对参考：`CHAIR/shared_small_preview/core4d_pair_reference.npz`，SHA256 `d8e945b4ec08cc9d22d6493942cf05ed787cd42b87b5e18ac4e6d7d6423dd586`。
- Small Table 两人的 nominal：`TABLE/two_stage_fullnominal/person1/person1_nominal_scaled.npz` 和 `TABLE/two_stage_fullnominal/person2/person2_nominal_scaled.npz`。已有 mean 配对参考：`TABLE/small_table_nominal_preview/core4d_pair_reference.npz`，SHA256 `d9d17e96f5f0b73aa76a1bf32f1ec50cfb7fa7fe78a847dd890882add4bbfd05`。不要把 `two_stage_fullnominal/core4d_pair_reference.npz` 的 Stage 2 机器人误当作该任务的 mean 机器人。
- Bucket nominal 的入口：`BUCKET/two_stage/person1/person1_nominal_scaled.npz` 和 `BUCKET/two_stage/person2/person2_nominal_scaled.npz`；开算前核对这两个 Stage 1 中间产物实际存在，并用原 two_stage manifest / 生成脚本确认来源。Bucket re-optimized 使用 **`BUCKET/two_stage/core4d_pair_reference.npz`**，SHA256 `089d3a36a39442e055f01a8111d6145475c062928c28727062f951887020ceec`。这里的 re-optimized 已经完成，不是要求现在再次优化。

不得使用 Chair `shared_small_a1_preview_tol6`、Bucket `a1_wrist_only`、训练用 `core4d_pair_runtime_fps50.npz` 或任何 rollout 状态代替这些腕部优化前参考。若文件迁移，按 manifest 和 SHA256 查找同一资产；若原 nominal 中间产物缺失，明确报告并保留对应缺失行，不重新求解。

数组布局：单人文件的 `qpos` 应为 `T×43`；前 36 列是机器人 `[root_xyz,root_quat_wxyz,29 joints]`，末 7 列是该人的 nominal 物体 `[xyz,quat_wxyz]`。配对文件为 `robot_qpos: T×2×36`、`object_qpos: T×7`。**canonical `object_poses` 的顺序不同，是 `[quat_wxyz,xyz]`**；用 `core4d_retarget._physical_object_qpos` 的 `[4,5,6,0,1,2,3]` 重排规则转为 qpos。

## 3. 先固定时间轴和源窗口

1. 读取原 canonical，在 `src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_adapter.py` 中复用 `load_canonical_core4d_sequence` 和 **`resample_core4d_pair_sequence(sequence, 30)`**。它对人点/物体平移做线性插值，对物体旋转做 SLERP，并保留首尾时刻。窗口由这份源数据在 **30 Hz 参考时间轴**上的取值计算，不先在 15 Hz 二值化再复制 mask。
2. 验证两个变体及两个人都是表中预期的 30 Hz 帧数，时间 `t=k/30`，从源 canonical 第 0 行开始且没有裁剪、移相或截断。235、413、299 是本轮几何参考帧数；391、687、497 是训练重采样后的帧数，不在本轮使用。若实际元数据不符，报告来源差异，不以归一化相位或重复末帧补齐。
3. 原 `aligned_frame_ids` 是保留的多源对应元数据，不是一个可直接替代 `k/fps` 的时间列。`windows.json` 同时保存参考帧号、秒数、canonical 插值左右行号与权重，并保存相应原 `aligned_frame_ids` 行，便于回查。
4. 每只手使用 A1 已有的四个关键点的**等权质心**作为本次窗口代理点：

   | 手 | canonical `human_joints_full` 中的名字与零起始索引 |
   |---|---|
   | left | `L_Wrist:20`, `L_Index1:25`, `L_Middle1:28`, `L_Pinky1:31` |
   | right | `R_Wrist:21`, `R_Index1:40`, `R_Middle1:43`, `R_Pinky1:46` |

   依据是 `src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_retarget.py` 的 `CORE4D_A1_PALM_LANDMARKS`。这里是腕点加 index/middle/pinky 三个指根，共四点；不另加入 thumb/ring，也不只用腕点。A1 原函数构造的是掌面方向，**四点质心是本次明确选定的窗口点，不声称是原函数已经输出的掌心原点。**
5. 设该源代理点为 `c[k,i,h]`，源 mesh 的局部三角表面为 `M`，原尺寸源物体轨迹为 `(Rsrc[k],psrc[k])`：

   ```text
   source_distance[k,i,h] = min_{x in M} ||c[k,i,h] - (Rsrc[k] x + psrc[k])||
   W[i,h] = { k : source_distance[k,i,h] < 0.050 m }
   ```

   严格小于 5 cm；不平滑、不扩张、不剔除短段、不按机器人距离或任务结果筛帧。每任务冻结四个窗口，供两个变体共用。即使该源点位于物体内，也按无符号表面距离取窗。

## 4. 每个变体的唯一共享物体

R2 中的缩放值如下，用完整精度，不用表格显示时的舍入值：

| 任务 | `s_0` | `s_1` | `mean` / 目标缩放 |
|---|---:|---:|---:|
| Chair | 0.7616803111061523 | 0.7267058142765197 | 0.744193062691336 |
| Small Table | 0.7477371848535067 | 0.7347195863845842 | 0.7412283856190455 |
| Bucket | 0.7614736204510514 | 0.7222941753069272 | 1.0（两个变体共同目标） |

**Chair、Small Table：**

- 两变体的两台机器人都使用第 2.2 节的 nominal `qpos[:,:36]`，必须逐元素相同。
- `one-sided`：唯一共享物体取 `participant=0/person1` 的 nominal 物体轨迹 `qpos[:,36:43]` 和缩放 `s_0`。两台机器人的距离都对这一物体计算，不各算各的物体。
- `mean`：使用已有 pre-wrist mean 配对文件的 `object_qpos` 和 `shared_object_scale`。验证其平移等于两个 nominal 物体平移的算术均值，旋转与两者相同（四元数正负等价），缩放为 `(s_0+s_1)/2`；同时验证配对文件的 `robot_qpos` 等于两份 nominal 机器人姿态。
- 这些规则见既有 `scripts/export_core4d_small_table_pair.py` 的 `export_small_table_pair`；Chair 的 `scripts/render_r2_shared_chair.py` 也验证了 nominal 与 mean 的机器人姿态相同。只读取/复现其组合规则，不运行导出器覆盖资产。

**Bucket：**

- `nominal` 和 `re-optimized` **都对同一个原尺寸（scale=1）源物体、同一条源轨迹评估**。使用 `BUCKET/two_stage/core4d_pair_reference.npz` 的 `object_qpos`，并验证它等于源 canonical 在 30 Hz 上的物理轨迹。
- `nominal` 的两台机器人来自 Stage 1，虽然它们当时分别围绕缩放 `s_0`、`s_1` 的临时场景求解，本次将其原姿态放在同一世界系中对源尺寸目标测距。不得按各自缩小的临时物体测距，也不得再平移/缩放机器人来消除差异。
- `re-optimized` 使用已有 Stage 2 `robot_qpos`；不用腕部优化后的姿态。此对照只比较既有两阶段结果，不重跑求解。

对所有任务，使用原始 OBJ 的三角表面乘上述缩放。不要用稀疏 object-sampling 点、mesh 顶点最近邻、碰撞 convex hull/decomposition 或 Small Table 的五盒碰撞代理替代表面。若 URDF/MJCF 中还含局部 origin/rotation/mesh scale，先核对它与上述 canonical mesh 定义的关系，并把完整变换写进来源记录，防止重复缩放。

## 5. 机器人手点、距离和 inside

机器人手点为 `left_rubber_hand_link`、`right_rubber_hand_link` 的 **FK 连杆原点**。这是 U1 实际代码 `scripts/prepare_experiments_r2_cpu.py:HANDS`、`scripts/collect_experiments_r2_geometry.py:u1` 使用的点。不要换成 `wrist_yaw_link`、视觉 mesh 中心、掌面质心或额外平移的接触点；本轮与 U1 只共享手点定义，指标和窗口各自按定义执行。

使用机器人 URDF：`ROOT/src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf`，R2 已记录 SHA256 为 `7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25`。复用 `prepare_experiments_r2_cpu.py:BatchFK` 的 FK 逻辑及关节顺序核对，或原 retarget MJCF 的 `mj_kinematics`；不调用 `mj_step`。运行已有 FK 独立核查，并在少量实际参考帧上核对两个实现/原场景的手点差异，记录最大值。

设机器人手点为 `h`，变体的共享物体为 `(R,p,s)`：

```text
d = min_{x in M} ||h - (R (s x) + p)||
  = s * distance_to_triangle_surface(R^T (h-p) / s, M)   # s>0，均匀缩放
dist_mm = 1000 * d
```

用精确点–三角面最近距离，允许 CPU 分块/BVH 加速；不以采样点近似。保留穿透帧：内部点的无符号距离通常大于 0，**不能将 inside 点的距离截为 0**。

`inside` 是额外的几何诊断，不能反过来选择窗口或替换距离：

- 对能定义封闭实体内部的 mesh，记录 `is_watertight`、`is_winding_consistent`、体积和判定方法/软件版本；只在拓扑与判定可靠时计算严格内部标志。距离小于等于 `1e-8 m` 的边界点按 `inside=false`，单独记边界计数。
- 对非封闭、方向不一致或内部判定不可靠的 mesh，逐帧 `inside=--`，对应汇总 `frac_inside=--`，并在 `checks.json` 写明原因。**无符号距离仍照常计算。** 不补洞、不换 convex hull、不把包围盒内部当作物体内部，不将未知 inside 写成 false/0。
- 若使用 mesh 预处理，仅允许不改变三角表面的索引整理；记录操作。不得通过修复或替换几何来获得更方便的 inside 结果。

## 6. 统计口径与缺失值

每个 `(task,variant,participant,hand)` 的四项统计只使用其冻结窗口 `W`：

- `n_window_frames = len(W)`；同一任务/人/手的两个变体必须相等。
- `dist_mean_mm` 是所有窗口帧距离的算术均值；`dist_median_mm` 是中位数；`dist_p90_mm` 使用 NumPy `percentile(...,90,method="linear")`。先用完整精度计算，再写 CSV。
- `frac_inside = 窗口内严格内部帧数 / len(W)`，范围 `[0,1]`；仅在窗口非空且所有内部标志有效时填写。若任一标志未知，不通过丢弃该帧改变分母，填 `--`。
- 真正空窗口填写 `n_window_frames=0`，三个距离统计及 `frac_inside` 都填 `--`。输入/时间映射缺失与空窗口不同：窗口尚未确定时 `n_window_frames=--`，在状态记录中说明。
- 若一个窗口内任一距离计算失败，该组距离统计填 `--` 并报告错误，不静默删帧。保留其他已完成组及全部 24 行。

**本轮任务级 mean 固定为四组等权平均：**

```text
task_variant_mean_mm = (1/4) * sum_{participant in {0,1}, hand in {left,right}} dist_mean_mm
```

把六个任务–变体均值写入 `aggregation.json`，标明 `equal_weight_participant_hand_means`、四个组成值及各自帧数；不混入主 CSV，不把各组帧拼池后得到的均值混称为此 mean。四组中任何一组空窗或未测，该任务–变体均值填 `--`；不要改成对剩余组取平均。不得在得到结果后变更聚合方式、阈值或点位。

## 7. 执行顺序与验收

1. **输入预检。** 保存当前代码 commit 和 dirty diff，核对文件 SHA、人体点索引、mesh 单位/坐标、Stage 1/Stage 2/wrist 阶段、FPS/帧数和机器人关节顺序。缺少 Bucket nominal 时先报告，其他任务继续。不要为消除 hash 差异覆盖或重新生成资产。
2. **只从源数据生成并保存窗口。** 冻结四点定义、5 cm 阈值、30 Hz 时间网格、每人每手 mask 和源距离。两个变体从同一份窗口文件读取。
3. **组装只读变体并跑 CPU FK/表面距离。** 逐项断言第 4 节的同姿态/同目标关系。先用少量帧核对世界系和局部系两种距离计算相等，再计算全序列。禁止重新求解或策略仿真。
4. **汇总和自检。** 检查唯一键正好 24 个、两个变体窗口一致、距离非负、`n_window_frames` 与原始记录相符，并从逐帧结果独立重算主 CSV 和六个等权均值。保存完整命令、退出码和检查结果。
5. **打包交付。** 原始数值完整交付，不筛选任务、人、手或帧，不写“改善/变差”等结论。

Ubuntu 需要新增一个只做上述计算的脚本，例如 `scripts/measure_u2_shared_scene_handdist_cpu.py`；该名称是本次拟新增接口，不是假定已有程序。实现后可用以下命令运行，输出目录必须新建：

```bash
cd /home/kevin/holosoma-core4d-base
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 /home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python scripts/measure_u2_shared_scene_handdist_cpu.py --output paper_handoff_exports/EXPERIMENTS_R2_U2_SUPPLEMENT_20261029
```

日期改为实际交付日期；不要复用已存在的输出目录。可复用 NumPy、SciPy、yourdfpy、trimesh 等已有 CPU 依赖。缺少加速依赖时可使用精确的 CPU 分块点–三角面实现；不能换成近似顶点距离。

## 8. 交付包

```text
EXPERIMENTS_R2_U2_SUPPLEMENT_<YYYYMMDD>/
  README.md
  ref_shared_scene_handdist.csv
  windows.json
  aggregation.json
  checks.json
  per_frame_handdist.csv
  sources/manifest.json
  sources/code/                    # 测量脚本和实际使用的辅助函数快照
  sources/inputs/                  # 本次测量所用 canonical、pre-wrist qpos、mesh、机器人URDF及必要配置
  SHA256SUMS
```

`per_frame_handdist.csv` 保存所有参考帧，至少含 `task,variant,participant,hand,frame,time_s,in_source_window,source_dist_mm,dist_mm,inside`；这里允许额外诊断字段，主 CSV 字段不变。它用于复算，不要求额外实验。`windows.json` 保存第 3 节的点名/索引、插值映射、窗口帧号、时间、mask 和输入 hash；也可将大数组放入随包 NPZ 并在 JSON 指明路径/字段/hash。

README 和 `sources/manifest.json` 对每个输入及输出写明原始 Ubuntu 路径、打包相对路径、SHA256、代码 commit、脚本、完整命令、配置、环境版本和父来源；dirty 工作树需附本次实际执行代码的快照/diff，不能只给 commit。本任务未加载 checkpoint，明确记 `checkpoint: not_used`。保留 canonical 的来源帧映射和 Stage 1/Stage 2 manifest；不要只交无法追溯来源的匿名数组。

为包内所有文件生成 `SHA256SUMS`（清单本身除外），压缩后另附压缩包 SHA256。由作者将交付包带回 Mac，放入 `What my Ubuntu agent wants to tell you/EXPERIMENTS_R2_U2_SUPPLEMENT_<YYYYMMDD>/`。不覆盖已交付的 `EXPERIMENTS_R2_20261026`。

本说明明确化了原请求中未固定的四点掌心代理、15→30 Hz 时间匹配、Bucket 唯一共享目标、非封闭网格的 inside 缺失值和任务级 mean；保留原六个变体、主 CSV 名称/字段和 24 行要求。
