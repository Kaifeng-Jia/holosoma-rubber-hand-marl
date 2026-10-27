# 图 1–3 合并渲染交接（F1 / F2 / F3）

2026-10-27。Ubuntu 从本文件即可执行，不需要 Mac 文件、论文或另一本 brief。只处理已有数据的显示和几何导出；不训练、不重新评测、不重新 retarget/配准/腕部优化、不修改参考、模型或训练资产。图的文字、公式、箭头和最终排版由 Mac 完成。

## 0. 工作目录、CPU 预检和执行顺序

- 仓库：`Kaifeng-Jia/holosoma-rubber-hand-marl`，分支 `core4d-base`；已知工作目录 `ROOT=/home/kevin/holosoma-core4d-base`。不要 reset/clean 工作树；先记录 commit 和 dirty diff。
- 已有 Python：`/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python`。本任务与 U2 补测独立，不改它的脚本、定义或交付物。
- **只用 CPU 软件渲染，先 smoke test。** 已有 MuJoCo 渲染器默认 `MUJOCO_GL=egl`，不能直接沿用。启动进程前显式选软件后端，例如已有 OSMesa，设置 `MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`LIBGL_ALWAYS_SOFTWARE=1`、`CUDA_VISIBLE_DEVICES=""`；所有人体模型张量也放 CPU。
- 上述环境变量本身不是证明：预检记录实际 GL vendor/renderer/version、后端、Python/依赖版本，确认 renderer 为 CPU 软件实现；用一个已有姿态做低分辨率试渲并检查输出。后端缺失或实际使用 GPU 时停止渲染并反馈，不静默改用 EGL/GPU。可继续输入核验和 CPU 几何导出。
- 顺序：①输入与软件后端核验；②建立共用素材/相机清单；③F1 人体与执行、F2 参考、F3 几何；④自检、打包。相同输入/帧/相机/材质只渲一次，通过 manifest 复用，不复制为另一个时间点。
- 可新增 `scripts/export_figures_r2_cpu.py` 等只读导出脚本；这是待实现接口，不是假定现有命令。沿用现有 FK/loader/渲染逻辑，保存实际执行代码。只向新输出目录写文件。

## 1. 共用显示规格

- 机器人必须为真实 G1 视觉网格：`ROOT/src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf`；已记录 SHA256 `7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25`。关节顺序与参考的 `joint_names` 核对。
- 角色0/源角色：蓝 `#266BD1`；角色1/伙伴：橙 `#F27A1F`；人体中性浅灰，物体中性灰，背景 `#FAFAFA`。颜色跟随角色 ID，不随屏幕左右位置交换。材质仅改变显示，不修改 URDF 或物理属性。
- 所有正式原图为 **1920×1440、4:3 PNG**，无文字、时间、坐标轴、UI、公式、箭头或标记。无损保存，不拉伸原16:9图片；主体四周留约5–8%安全边距，头、脚、手和物体完整。需要局部图时另交原分辨率局部视角并记录相机。
- 同一任务的时间序列固定相机和 framing，不逐帧自动居中/缩放。先用候选帧整体范围确定视角与取景，再一次锁定；F3左右面板观察方向一致，允许各自固定距离/中心。F1人体与对应执行行用一致观察方向，不要求不同世界原点直接共享未经变换的外参。
- 旧 U7 的 `follow-pair` 会逐帧移动 lookat，不能直接称为固定相机。可参考旧 azimuth=135°、elevation=−22°、fovy=45°，但须为本轮重新冻结实际相机；视角以看清两角色和操作区域为准，不改变姿态。
- 若为显示统一坐标，整段“人/机器人＋物体”一起施加同一刚体变换，保存原始→显示的4×4矩阵；不得分别移动人和物体、逐帧对齐或为构图缩放人体。源物体与目标共享物体保留各自真实尺寸。
- 同组固定背景、地面、光源、阴影、曝光和抗锯齿；记录设置及软件版本。复用 `VisualScene.mapping` 按 agent/body/geom 修改角色颜色，避免共享 mesh/material 误改两角色；保存角色→body/link/geom 的映射，不依赖跨场景不稳定的数字 ID。
- 每张图保存相机模式、图像大小、投影类型、fovy/内参、world-to-camera外参、裁切/显示变换；给出明确的坐标与像素约定。不要仅交方位角和一个“默认相机”名称。

## 2. Ubuntu 输入与可复用脚本

下表路径以 `ROOT` 为根；若文件迁移，用旧 manifest/hash 找同一资产，不重新生成或替换。

| 用途 | Ubuntu 路径/入口 |
|---|---|
| 第一批源姿态与接近窗口，记 `PILOT` | `logs/ReferenceGeometry/reference_geometry_pilot_20261017/` |
| 第二轮配准，记 `GEOM` | `logs/ReferenceGeometry/reference_geometry_round2_20261017/` |
| R2已交付素材，记 `R2` | `paper_handoff_exports/EXPERIMENTS_R2_20261026/` |
| 旧统一评测，记 `OLD_EVAL` | `logs/PaperSupplement20260929/unified_evaluation_20261003/` |
| 新Push评测，记 `NEW_EVAL` | `logs/ExperimentsR2_evaluation_20261026_retry01/` |
| 参考渲染 | `scripts/render_reference_geometry_round2.py`、`scripts/render_reference_geometry_pilot.py`、`scripts/render_wrist_reference_pairs.py` |
| 执行回放渲染 | `scripts/export_recorded_demo_media.py`、`scripts/render_r2_handoff.py` |
| Chair共享场景 | `scripts/render_r2_shared_chair.py`；已有输出 `R2/U7_media/shared_scene/` |
| 几何/FK/映射入口 | `scripts/reference_geometry_pilot.py`、`scripts/reference_geometry_round2_correspondence.py`、`scripts/reference_geometry_math.py`；`src/holosoma_retargeting/holosoma_retargeting/dual_a1_layout.py`、`dual_pull_reference.py` |

保留上述脚本现有资产校验，显示时只做 FK 和场景更新，不调用 `mj_step` 或策略。部分脚本曾为 Ubuntu 未提交文件，不能只凭 Git commit 追溯；交付实际代码快照及 hash。

## F1. Teaser：Push、Pull、Chair 三行

Mac 将排为每行1帧真实人体示范＋4帧机器人执行。Ubuntu 每任务交6张执行候选、最多3张人体候选；Mac再选择。候选帧只表示时间位置，未经核查不标“接触/受力/开始移动”。

### F1-A 机器人执行（保存的真实回合）

| 任务 | 保存轨迹 | 建议候选源状态帧（零起始） | 回合身份 |
|---|---|---|---|
| Push | `NEW_EVAL/U5_push_registered/results/Push-registered_seed721/perturbed_000.trajectory.npz` | 0,62,103,206,247,309 | 配准策略，seed721，扰动成功回合 |
| Pull | `OLD_EVAL/results/Pull-baseline_seed721/nominal_viser.npz` | 0,63,126,190,253,316 | seed721，nominal成功回合 |
| Chair | `OLD_EVAL/results/C-01_seed721/nominal_viser.npz` | 0,78,156,235,313,391 | C-01，seed721，nominal成功回合 |

三者状态帧率均为50 Hz；帧数分别310/317/392。核对同目录 `episodes.json` 的 `completed_reference` 和 R2 `U7_media/render_manifest.json`，记录 checkpoint SHA256与轨迹SHA256。Push同seed的nominal失败回合不属于本组，不能换用 `U5_push_registered/videos/` 的 nominal 视频。

已知轨迹 SHA256：Push `4bcb6a3eaa3f66fb3b67e8b308e6ae145bd8bc8887d37adb70cfc91d9f479443`；Pull `fa69703a6cc71c73df7281be33e63ad023ca7d19d5eee7e78932fb0a932a24aa`；Chair `b895b60421b039d75476ce2634f6f471b3992eab875423433b4e5abe7c2ff74a`。

物体使用各回合 manifest 的 `object_urdf_source` 和训练资产 hash。旧视频是25 fps，不能将视频帧号当50 Hz状态帧；直接回放保存状态。图中不添加物理效果、估算力或改变物体轨迹。

### F1-B 真实人体示范（与F2共用）

- Push：OMOMO `sub6_largetable_033`；Pull：OMOMO `sub3_largetable_010`。已知数据根 `/home/kevin/holosoma/OMOMO_new`，现有 loader 按任务名读取 `.pt`，格式 `smplh`；入口 `ROOT/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py:load_motion_data` 及其 `load_intermimic_data`。这里只复用读取/坐标转换，不运行重定向求解。
- Chair：原始目录 `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions/20231020/074/`，读取 `person1_poses.npz`、`person2_poses.npz`、物体轨迹及 metadata；两人必须同帧。原源物体 `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models/chair/chair021_m.obj`。
- Chair canonical：`ROOT/logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/source_canonical.npz`。用 `core4d_adapter.py` 核对时序与 `Rx(+90°)` 的Y-up→Z-up转换。原人体已有10475个surface vertices；须配正确SMPL-X faces，保留原人物形状。canonical的关节点不能代替surface mesh；`visualize_core4d_source.py`现有骨架预览不是人体网格图。
- 先确认真实 mesh/原始pose/相应人体模型和faces存在；可用现有模型在CPU恢复原始mesh，不从稀疏关节点拟造人体。若输入缺失，报告准确缺项，交付其他素材，不用AI图片、机器人或通用人体替代。
- 每任务选最多3个原示范中手部靠近物体、姿态可辨的候选帧，交原帧号、FPS、秒数、读取配置和坐标转换。若与机器人参考有重采样/裁剪，记录真实时间映射；没有明确映射时只标“示范候选”，不声称与执行帧同步。人体保持源尺寸、中性浅灰；Pull/Chair人体图供F2复用。

## F2. 方法总览需要的真实小图

1. **人体输入：** 复用F1的Pull一人和Chair两人素材，不再渲同一图。
2. **重定向输出：** Pull原始单机器人retarget参考1–3帧；Chair两人的nominal参考1帧。必须是保存的retarget结果，不以策略rollout冒充。
   - Pull定位 basename `sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz`（既有 `src/holosoma/tests/envs/marl/test_plan5_push_manager.py` 有引用），追溯其原始 retarget/source manifest及生成配置后使用；只按文件名不能证明来源。找不到可验证的retarget产物时记缺失，不重新求解。
   - `GEOM/pull/preprocessed_source.npz` 来自成功单机器人rollout的预处理，**可供F3显示配准实际输入，但不可用来填F2的“重定向输出”节点**。如仅交此资产，manifest类型必须标 `source_skill_rollout`，该节点保留缺失。
   - Chair读取 `ROOT/logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/two_stage/person1/person1_nominal_scaled.npz` 及person2同名文件；机器人取 `qpos[:,:36]`。不混入后续腕部优化结果。
3. **共享场景：** 复用 `R2/U7_media/shared_scene/` 的 `individual_nominal` / `shared_mean` 来源；原文件为pre-wrist第117帧、30 Hz（3.9 s），两面板机器人姿态完全相同。若为统一4:3/材质需重渲，只渲这一对，保持相同相机和真实物体缩放；来源规则见 `render_r2_shared_chair.py`。
4. **协作参考：** 复用F3的Pull右侧候选，或另取同一 `GEOM/pull/new_pair_reference.npz` 的2–3帧；优先共用F3，不用执行图替代参考。无需做残影合成，Mac可从原图选择。
5. **执行：** 复用F1的一张执行候选。角色等距变换俯视图和网络结构由Mac画矢量；只交F3已有几何数据，不另造网络/机器人图片。
6. 腕部小图若Mac需要，可复用 `ROOT/paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005/figure3/chair/frame_110_before_hand.png` 与 `frame_110_after_hand.png` 及该包render manifest；只建立引用，本轮不补腕部实验。若包内部层级不同按其manifest定位。

## F3. 单栏两行的角色等距变换机制图

最终由Mac按论文实际 `\columnwidth` 排版（当前模板为3.4 in宽），高约2.4 in，上Push、下Pull；每行左窄右宽，左为单机器人与源物体，右为同帧协作参考与共享物体。Ubuntu只交原图和独立几何，版式/字号不在Ubuntu固定。

### F3-A 输入、帧与手点

- 每任务读 `GEOM/{push,pull}/preprocessed_source.npz`、`new_pair_reference.npz`、`input_manifest.json`、`correspondence_spec.json`、`fitted_candidates.json`、`selection.json`、`hand_samples.npz`。源 `robot_qpos:T×36`；配对 `T×2×36`；物体 `T×7=[xyz,wxyz]`；机器人 `[root_xyz,root_quat_wxyz,29 joints]`。Push为309帧、Pull为316帧，均50 Hz；同帧源与配对物体时间轴必须一致。
- 使用 `selection.json` 指定的role候选及实际 `Q,c`，不重拟合。读取源/目标物体URDF和hash自 `input_manifest.json`。本轮Push已存状态为 `diagnostic_only`、Pull为 `passes_source_relative_geometry`，原样保留在内部manifest；这不妨碍画映射机制，但不得把配准图说成新的可执行性证明。
- 接近窗口读 `correspondence_spec.json:weighted_source.windows`，结合 `PILOT/{case}/source_interactions.json`、`source_distances.npz` 的真实支持帧；保留原阈值和窗口定义，不把填补gap本身视为支持。每任务选3个双手均有几何接近证据、手部不被躯干/物体遮住且可辨认的候选帧；不要求link原点穿出自身mesh。隐藏anchor如实标可见性，不移动点或几何；不足3帧时交实际数量和原因，不伪称动力学接触。
- `hand_samples.npz` 的 `*_fit` 为256×3、`*_validation`为1024×3，都是**手link局部网格采样**，不是轨迹。通过 `reference_geometry_pilot.py:RobotFK` / `render_wrist_reference_pairs.py:VisualScene` 对当前qpos做FK后才能投到世界系；不把第k个采样点当第k帧。
- 主手点固定为 `left_rubber_hand_link`、`right_rubber_hand_link` 的FK原点，明确 `local_point=[0,0,0]`。另保留整组validation表面采样的FK结果用于接近/遮挡诊断；若原点位于自身网格内，诚实标注自遮挡，不把它移动到表面来伪造可见性。
- 对每个候选帧/源手，只在该手当帧有支持的face角点中选一次最近源anchor，等距时按稳定face/角点索引排序。之后所有目标角色用对应的anchor索引，不在目标侧重新找最近点。保留 `hand,window_id,anchor_slice,anchor_index,source_face_id,target_face_id,corner_permutation`；同时导出该面全部角点供Mac选择稀疏标记。
- Pull的手对应由现有构型映射确认：源L→伙伴R、源R→伙伴L；记录role的parity、关节index/sign映射及hand link对应。角色0仍蓝、角色1仍橙，手的左右与角色颜色是两个概念。

### F3-B 几何公式与导出

下面 `a_m,a'_im,Q_i,c_i` 都在物体局部坐标，`R_o,p_o` 为该帧物体到原世界的刚体变换。FK返回原世界点；显示用刚体变换只在投影前施加。

```text
h_s^local = R_o^T (h_s^FK_world - p_o)
h_i^ideal_local = Q_i h_s^local + c_i
h_i^ideal_world = R_o h_i^ideal_local + p_o
a_m^world = R_o a_m + p_o;   a'_im^world = R_o a'_im + p_o
r_im = Q_i a_m + c_i - a'_im
E_i = sum_m w_m ||r_im||^2,   sum_m w_m = 1
```

- `w_m` 原样用 `correspondence_spec.json:weighted_source.weights`，不按显示点子集重新归一化；源anchors与该selected候选的 `target_anchors` 按原顺序一一对应，复用face/permutation检查。
- **分别导出**源FK手点、配对参考中实际FK手点 `h_i^actual_world`、理想映射点 `h_i^ideal_world`，以及actual−ideal差值。箭头的端点类型在JSON明确；不能用ideal点覆盖actual点，或宣称实际G1网格满足严格镜像等式。
- 右侧虚线表示已放置role0→role1的相对反射：`Q10=Q1 Q0^T`，`c10=c1-Q10 c0`，固定平面满足 `(I-Q10)x=c10`。本Pull两平移约为local z方向±0.4390236 m，故平面是共享物体local `z=0`；**不是world z=0**。
- 将该平面与真实tabletop平面求交、按实际有限桌面范围截取线段，再变换到世界/显示系并投影；记录两平面方程、坐标系、3D端点和2D端点。无交线/投影退化时报告，不手画一条看似合理的虚线。
- 每图一个JSON：任务/asset ID、帧/秒数/FPS、输入hash、源/目标URDF、角色/关节/手映射；相机K、world-to-camera矩阵、投影类型、near/far、图像尺寸；display变换；全部标记的local/world/display/camera/2D坐标、点类型、对应ID和权重。
- 像素约定固定为原图左上角原点、u向右/v向下；注明像素中心、相机轴方向、深度定义及矩阵乘法约定。投影和深度图必须使用渲染该PNG的同一相机/场景，而不是另写近似投影。
- 导出线性深度（NPZ即可）及基于同相机深度缓冲的可见性：`in_front,in_image,projected_depth,buffer_depth,depth_tolerance_m,visibility,occluder_geom_id`。区分 `visible/occluded_self/occluded_other/out_of_frame/behind_camera/unknown`；容差和取样邻域写清。可借助geom/segmentation ID区分自遮挡；无法区分则记unknown，不猜测。手原点在网格内时同时报告对应手表面是否可见，供Mac决定标记显示。
- 内部 `checks.json` 重算全anchor的 `E_i`、`max_m||r_im||`、det Q、正交性、权重和、FK/ideal差值、像素投影一致性，与已存fit核对。`E_i`单位为m²，与存档 `weighted_rmse_m` 比较时用 `sqrt(E_i)`；最大残差单位为m。保留实际误差，不改数据以追求“零”；这些数值/微米误差/调试说明不放进论文原图。

### F3-C 渲染和自检预览

- 每任务3候选×左右2面板，最多12张原PNG，统一1920×1440；左右同帧、相同方位角/俯仰角，分别锁定framing。源左面板用源物体，协作右面板用真实共享物体；左面板机器人蓝色，右侧角色蓝/橙。
- 原PNG完全无标记。另交一张独立 `diagnostics/geometry_overlay_preview.png`，在原图上按JSON投影叠加稀疏anchor/手点/偏移及Pull反射线，供检查投影；明确标diagnostic，Mac只用raw PNG＋JSON重新做矢量叠加。
- 遮挡多时先从现有窗口换候选或为整行统一调整视角，不改手、物体或参考。允许少于3候选并说明；不以新仿真“证明接触”，不导出假想作用力。

## 3. 交付与复用清单

输出目录必须新建：`ROOT/paper_handoff_exports/FIGURES_R2_<YYYYMMDD>/`。`manifest.json` 用独立asset ID列出文件及 `used_by:[F1,F2,F3]`，避免重复渲染或混淆reference/execution/source_skill/retarget/human类型。

```text
FIGURES_R2_<YYYYMMDD>/
  README.md
  manifest.json
  cameras.json
  raw/                          # 无标记PNG；按asset ID组织，共用素材只存一份
  geometry/                     # F3逐图JSON、深度/必要表面采样NPZ
  diagnostics/                  # CPU smoke test、checks.json、overlay preview
  sources/                      # 执行脚本/配置快照、输入manifest、必要小型NPZ
  SHA256SUMS
```

- README逐项写F1/F2/F3完成状态、缺失项、候选选择依据和共享资产ID；记录Ubuntu源路径、代码commit、dirty diff、执行脚本/配置/命令、环境/软件后端、每个输入与输出SHA256。模型/mesh/人体faces也要追溯；大原始数据无需全复制，保留路径和hash及复现步骤。
- 保存所有参考和执行的真实来源、时间映射、模型checkpoint hash（只有执行图关联checkpoint；未加载者记 `not_loaded`）。不要只给commit或匿名图片；源姿态与训练资产本轮不得改变。
- 核查相机固定、角色颜色、时间帧、文件数、图像尺寸、2D投影/深度、输入输出hash；为包内文件生成 `SHA256SUMS`（自身除外），压缩后另附压缩包SHA256。
- 由作者把包带回Mac，放入 `What my Ubuntu agent wants to tell you/FIGURES_R2_<YYYYMMDD>/`。只交素材，不改论文正文或最终图编号；遇到无法核验的来源/CPU后端/人体输入，提前反馈具体问题并继续可独立完成的部分。
