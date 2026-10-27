# 图表轮 Ubuntu 交接：本版替代 commit 4ce7fc0 的旧 F1–F3；停止旧蓝橙、三行方案的批量渲染

2026-10-27。仓库 `Kaifeng-Jia/holosoma-rubber-hand-marl`，分支 `core4d-base`；发布时沿用 `docs/experiments/ubuntu_handoff_figures_r2_20261027_zh.md`。本文件自包含，无需 Mac 文件、论文或旧 brief。旧图表渲染若正在运行，在当前输出文件写完后停止，不再启动其后续批量步骤；保留已完成素材、脚本和记录，按本版重新清点，不删除或覆盖旧结果。

**当前只授权 R0、输入清点和 CPU metadata/几何准备。R1–R6 的批量渲染必须等作者明确批准本次 R0 样张后才运行。** 本文描述完整后续规格不等于已批准后续执行；之前的 C+ 改色截图、旧任务批准、GitHub 文档更新、超时或无人回复都不是本次样张批准。

## 执行边界、路径与 CPU 预检

- 只读已有参考和保存轨迹；不训练、不重新评测、不重跑 retarget/配准/腕部优化，不改模型、参考或训练资产。不调用策略或 `mj_step`。本任务不改 U2 补测及其交付。
- `ROOT=/home/kevin/holosoma-core4d-base`；已有 Python 为 `/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python`。先记录 commit、dirty diff；不得 reset/clean 工作树。
- **只用 CPU 软件渲染。** 现有脚本默认 `MUJOCO_GL=egl`，不能直接视为 CPU。启动前选已有软件后端，例如 OSMesa：`MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`LIBGL_ALWAYS_SOFTWARE=1`、`CUDA_VISIBLE_DEVICES=""`；人体模型张量也放 CPU。
- 环境变量不是验收证明：先用一个已有姿态低分辨率 smoke test，记录实际 GL vendor/renderer/version、后端及依赖版本，确认 CPU 软件实现。若后端缺失或实际使用 GPU，停止渲染并反馈，可继续只读清点和 CPU 几何导出；不静默退回 EGL/GPU。
- 可新增只读导出脚本（例如 `scripts/export_figures_round_cpu.py`，是待实现入口），复用现有 loader/FK/render 逻辑，保存实际代码和 hash；只写新输出目录。不要把下表中的旧命令直接当作已满足新材质/CPU/固定相机的命令。

| 简称/用途 | Ubuntu 路径（相对 ROOT，除非写绝对路径） |
|---|---|
| `PILOT`、`GEOM` | `logs/ReferenceGeometry/reference_geometry_pilot_20261017/`、`logs/ReferenceGeometry/reference_geometry_round2_20261017/` |
| `R2` | `paper_handoff_exports/EXPERIMENTS_R2_20261026/` |
| `OLD_EVAL` | `logs/PaperSupplement20260929/unified_evaluation_20261003/` |
| `NEW_EVAL` | `logs/ExperimentsR2_evaluation_20261026_retry01/` |
| `CHAIR` | `logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/` |
| `WRIST` | `paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005/` |
| 参考与执行渲染 | `scripts/render_reference_geometry_round2.py`、`render_reference_geometry_pilot.py`、`render_wrist_reference_pairs.py`；`scripts/export_recorded_demo_media.py`、`render_r2_handoff.py` |
| 共享场景渲染 | `scripts/render_r2_shared_chair.py`；原输出 `R2/U7_media/shared_scene/` |
| 几何/FK/映射 | `scripts/reference_geometry_pilot.py`、`reference_geometry_round2_correspondence.py`、`reference_geometry_math.py`；`src/holosoma_retargeting/holosoma_retargeting/dual_a1_layout.py`、`dual_pull_reference.py` |

路径若迁移，按既有 manifest/hash 定位同一资产，不能挑名称相近的替代。部分脚本可能未提交，交付 commit 之外还须有实际脚本快照与 SHA256。

## 共用材质、相机和原图规格

- 真实 G1 视觉网格；执行及 R6 已知 URDF 为 `ROOT/src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf`，SHA256 `7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25`。其他参考特别是 R5 按其原 manifest 的 URDF/关节顺序读取，不能静默换机器人模型。
- C+ 基础材质：角色0/源角色 `#AAC0DE`，角色1/伙伴 `#E8C8A0`；正常明暗、粗糙、弱高光，用现有渲染器支持的低specular/shininess等设置实现，不假定MuJoCo有roughness字段。物体中性灰 `#B5B5B5`、基本不透明；人体中性浅灰且比物体略浅。固定角色 ID 与颜色，不随屏幕左右交换。
- 深色标记/副本色为 `#4A6FA5`、`#B07D3E`；角色等距变换强调色为低饱和紫 `#6B5B95`（Mac 矢量绘制）。反射线灰色虚线。R6(a) 是半透明物体的指定例外，其余不要浅蓝半透明桌或蓝灰椅。
- 纯色浅背景、无棋盘地面，有接触阴影。光源、曝光、抗锯齿同组固定；只在渲染场景中覆盖显示材质，不修改资产物理属性。用 agent/body/link/geom 映射着色，保存映射，勿因共享 material/mesh 把两角色一起改色。
- 原 PNG **不烧入文字、时间、UI、坐标轴、框、箭头、公式或标记**。字体、标签、矢量叠加、论文版心由 Mac 做。R1/R2 为 1920×1280（3:2）；其余原图宽统一 1920 px，高度按下述取景，保留少量安全边距。不得将旧低分辨率图拉大当作重渲。
- 同一任务执行序列锁定相机和 framing，以所有候选姿态的范围一次确定；禁止逐帧跟踪中心/自动缩放。旧 U7 `follow-pair` 会移动 lookat，不能称固定相机。R4 左右保持观察方向一致，各自 framing 固定；R5 同一行 Before/After 相机完全相同；R6 两版相机完全相同。
- 人体与对应执行行采用同一观察方向与投影规格；如世界原点/轴不同，先用已知坐标转换建立共同显示系，再选锁定相机。整段“人/机器人＋物体”一起施加同一刚体变换，记录 4×4 矩阵；不独立挪人物/物体、不逐帧对齐、不为构图缩放人体，不抹掉真实物体尺寸差异。
- 保存内参/投影类型、world-to-camera 外参、near/far、图像尺寸、显示变换、裁切规则、光照/材质与软件版本；仅方位角和“默认相机”不足以复现。原图应支持最终放置尺寸 ≥300 dpi，实际排版与 dpi 由 Mac 验收。

## R0：先交材质样张，等待作者确认

1. 完成 CPU smoke test 与输入清点。可并行准备下面各项的路径/hash、候选帧 metadata、FK/几何 JSON；**暂不生成 R1–R6 批量图片、contact sheet 或批量深度渲染**。
2. 只渲两张正式材质样张：Chair 双机器人＋灰色椅子（用 R6 共享版 frame117），Push 机器人＋灰色不透明桌（用 R2 的配准策略 `perturbed_000` 既有中段帧，记录所选帧）。使用上述 C+、新背景和阴影；这两张不是新的成功评测。
3. 输出 `R0/chair_material.png`、`R0/push_material.png`、`R0/manifest.json`、CPU 预检和 `input_inventory.json`；manifest 记录样张及材质/灯光配置 SHA256。另交一张R0 contact sheet，同时含两张全图和缩到图1单格大小的预览，检查浅色材质是否仍能辨认。交作者查看后，记录 `approval_status=pending` 并停止渲染。
4. **必须等作者明确确认这次样张，才可启动R1–R6。** 收到后在README记录确认原文/时间/来源及对应两图、manifest SHA256；没有确认不执行。不得将以前C+改色截图视为通过，也不因为本文已发布而自动通过。
5. 作者要求改色时只重做两张样张再等确认；若已确认样张或材质/灯光配置变化，重新确认。保持已批准材质不变的相机取景调整按以下规格做并记录。无需另建审批系统。

## R1：图1/图2的人类示范（批准后渲染）

四任务各交接触阶段1帧＋2备选，真实人体网格；图1最终每行1帧人体＋5帧执行。图2复用 Pull/Chair 人体，不重复渲染。

| 任务 | 原始输入与已有读取入口 |
|---|---|
| Push | OMOMO `sub6_largetable_033`，根 `/home/kevin/holosoma/OMOMO_new`，按任务名定位 `.pt`、格式 `smplh` |
| Pull | 同根 `sub3_largetable_010`；loader 为 `ROOT/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py:load_motion_data` 及 `load_intermimic_data` |
| Chair（两人） | `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions/20231020/074/{person1_poses.npz,person2_poses.npz}`；已有 `CHAIR/source_canonical.npz` 及 provenance |
| Bucket（两人） | 同数据根 `human_object_motions/20231020/071/{person1_poses.npz,person2_poses.npz}`；已有 `ROOT/logs/Core4DPreviews/bucket003_20231020_071_20260918/source_canonical.npz`（150帧、15Hz，SHA256 `1208deb13bf70daee2d42343c6728656cca8586ae46efc7a2bb718d514079083`） |

- 只复用 loader、已有 pose/vertices 与正确人体模型/face topology。CORE4D 原始人体含 `vertices[T,10475,3]`，canonical 文件不能代替缺失网格；`visualize_core4d_source.py` 的骨架/点云也不是人体网格。若原始 pose/模型/faces 不全，报告缺项，不能从 joints 假造 mesh 或用机器人代替人体。
- Chair 对象为 `object_models/chair/chair021_m.obj`，Bucket 为 `object_models/bucket/bucket003_m.obj`，都在 `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/` 下。复用 canonical provenance 中坐标/物体转换，避免重复旋转或把仿射物体变换误当纯旋转。
- 源与机器人 FPS/裁剪范围可能不同，保存原始帧、原始时间、裁剪起点、重采样/索引对应；不能以相同帧号冒充同时刻。依据已有接近窗口/真实几何选帧，仅表达几何接近，不新增物理接触判定。

## R2：图1的四任务执行（批准后渲染）

表I Bucket配置为 **B-RZ0**，使用该配置 seed721，不能改用 B-A 或旧 `bucket_best`。四回合均已有成功记录；本任务只回放保存状态。

| 行 | Ubuntu 保存状态 | 回合与状态帧数 |
|---|---|---|
| Push | `NEW_EVAL/U5_push_registered/results/Push-registered_seed721/perturbed_000.trajectory.npz` | seed721，扰动成功；310帧，50Hz |
| Pull | `OLD_EVAL/results/Pull-baseline_seed721/nominal_viser.npz` | seed721，nominal成功；317帧，50Hz |
| Chair | `OLD_EVAL/results/C-01_seed721/nominal_viser.npz` | C-01 seed721，nominal成功；392帧，50Hz |
| Bucket | `OLD_EVAL/results/B-RZ0_seed721/nominal_viser.npz`（同目录有 `nominal.trajectory.npz`） | B-RZ0 seed721，nominal成功；498帧，50Hz |

- 以上四份显示状态 SHA256 依次为 `4bcb6a3eaa3f66fb3b67e8b308e6ae145bd8bc8887d37adb70cfc91d9f479443`、`fa69703a6cc71c73df7281be33e63ad023ca7d19d5eee7e78932fb0a932a24aa`、`b895b60421b039d75476ce2634f6f471b3992eab875423433b4e5abe7c2ff74a`、`c16e43a05b33583d6c9342faaf88223a0980a47e9b7fa3775e3f9cc09db8c96f`。
- 核对同目录 `episodes.json`/旧 media manifest 的回合身份、`completed_reference`、时间轴及 checkpoint；Push同seed的nominal失败回合不能替代本行。旧视频25fps不能当50Hz状态帧索引。
- Bucket nominal trajectory SHA256 `459ac575428da98758c1c688ead61e3f80315faeda2aa4d2662fbc91ed3640c3`；checkpoint 为 `ROOT/logs/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919/model_12000.pt`，SHA256 `7ba039fbb608211f8b2866b4f80810c082313bca6232a028c07743391656a225`。目录名含 A 不能据此改标 B-A，当前来源审计状态为 `historical_effective_code_verified`。
- Bucket 物体 `ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/bucket003_training.urdf`，SHA256 `6c0ff9f60d8ef7ef057a988f4e95a6b89cd7e0f5ae55ea9e6697596955f28cde`。其他物体同样使用各回合 manifest 的训练资产及 hash。
- 每任务用保存数据确定交互/几何接近阶段起点 `start` 与成功回合末帧 `end`，再以 `linspace(start,end,8)` 取整得到8个不同状态候选，记录起止依据和精确索引，不沿用旧手选6帧；没有可核验起点则先交完整时域8候选并明确“接触起点未确定”，不猜阶段、不重仿真证明contact。分别保存 `state_frame` 与已存 `reference_frame`（缺失记null，不假定相等）及时间对应。Mac最终选5帧，不重复图片充当新时间点。
- 四行最终顺序 Push、Pull、Chair、Bucket，3:2取景紧贴主体，整行固定相机。无人类输入与策略执行共享时间轴的证据时只说明是同任务示范，不声称逐帧复现。图注条件需保留 Push perturbed、其余 nominal。

## R3：图2真实缩略图（批准后渲染）

图2上道为单人示范→重定向→角色等距变换，下道为双人示范→重定向→共享场景；合并协作参考后进入多智能体RL。Ubuntu只提供真实缩略图/几何，箭头、公式和网络框由Mac画。

- Pull 单机器人 **原始 retarget** 1帧：定位 `sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz`（`src/holosoma/tests/envs/marl/test_plan5_push_manager.py` 有引用），核验原始 retarget/source manifest、配置、模型及 hash。找不到可验证产物则报缺，不重新求解。
- `GEOM/pull/preprocessed_source.npz` 是成功单机器人 rollout 的预处理，适合 R4 配准实际输入，**不可填入 R3“重定向输出”节点**；不得以策略 rollout 冒充 retarget。
- Chair 重定向双机器人1帧：`CHAIR/two_stage/person1/person1_nominal_scaled.npz` 与 `CHAIR/two_stage/person2/person2_nominal_scaled.npz`；机器人取 `qpos[:,:36]`，不混入后续腕部优化。
- Pull 协作参考复用 R4 右图的2–3帧，不必额外合成残影。共享场景缩略图直接复用 R6 两版；人体复用 R1；执行复用 R2。所有复用在 manifest 用 `used_by`/asset ID表达。
- 可选交互网格仅在原重定向器已保存真实边/点时导出；缺失时跳过，不为图重优化或凭空连线。角色等距变换俯视小图使用 R4 的真实几何JSON，强调色与公式不烧入PNG。

## R4：图3单栏两行，Push / Pull 各3个候选

每候选帧左为单机器人＋源物体，右为同帧协作参考＋共享物体；上Push（η=+1）、下Pull（η=−1）。每任务最多3帧×左右2张，批准后渲染；不足3个可见几何接近帧就报实际数量。左右观察方向一致，同序列framing固定，不通过新物理仿真证明接触。

- 每任务读取 `GEOM/{push,pull}/{preprocessed_source.npz,new_pair_reference.npz,input_manifest.json,correspondence_spec.json,fitted_candidates.json,selection.json,hand_samples.npz}`。源 `robot_qpos:T×36`、配对 `T×2×36`；机器人 `[root_xyz,root_quat_wxyz,29 joints]`，物体 `[xyz,wxyz]`；Push309帧、Pull316帧、均50Hz。核验源/配对同帧时间映射与物体姿态。
- 复用 `selection.json` 选定的 `Q_i,c_i,parity,part_map`，不重拟合；源/目标物体URDF、hash按 `input_manifest.json`。内部保留既有状态 Push `diagnostic_only`、Pull `passes_source_relative_geometry`，不把机制图标成新可执行性证明。
- 源接近窗口读 `correspondence_spec.json:weighted_source.windows`，结合 `PILOT/{case}/source_interactions.json`、`source_distances.npz` 的支持帧；沿用阈值与窗口定义，gap填补本身不视为支持。选双手几何接近且可辨认、未被躯干/物体严重遮住的帧。
- `hand_samples.npz` 中 fit256×3/validation1024×3 是 **手link局部表面采样，不是时间轨迹**。用 `reference_geometry_pilot.py:RobotFK` / `render_wrist_reference_pairs.py:VisualScene` 对每帧做FK；主手点固定为左右 `rubber_hand_link` 的 `local_point=[0,0,0]`，另存表面样本的FK结果用于接近/遮挡诊断。
- 每候选帧/源手只在当帧有支持的face角点中选一次最近源anchor；等距时按稳定face/角点索引。所有目标角色复用其对应anchor索引，不在目标侧重找最近点。保留 `hand,window_id,anchor_slice,anchor_index,source_face_id,target_face_id,corner_permutation`，并导出该face全部角点。
- Pull保留源L→伙伴R、源R→伙伴L及真实关节index/sign映射。手左右与角色色是两件事。源/目标anchors与selected候选 `target_anchors` 按存档顺序对应，保留face/permutation。

令源/目标物体姿态分别为 `(R_s,p_s)`、`(R_t,p_t)`，anchors及 `Q_i,c_i` 在物体局部系；两物体姿态只有经核验相同才可共用一个变换。导出定义：

```text
h_s_local = R_s^T (h_s_FK_world - p_s)
h_i_ideal_local = Q_i h_s_local + c_i
h_i_ideal_world = R_t h_i_ideal_local + p_t
a_m_world = R_s a_m + p_s;  a'_im_world = R_t a'_im + p_t
r_im = Q_i a_m + c_i - a'_im
E_i = sum_m w_m ||r_im||^2;  sum_m w_m = 1
```

- 权重原样取 `weighted_source.weights`（两手总权重相同），不用显示子集重归一化。**分开保存**源FK点、配对参考实际FK点 `h_i_actual_world`、理想映射点及 actual−ideal，不能用ideal替换真实G1手点。
- Pull右图反射线表示已放置role0→role1相对映射：`Q10=Q1 Q0^T`、`c10=c1-Q10 c0`；固定平面 `(I-Q10)x=c10`。本例平移约local z方向±0.4390236m，固定平面为共享物体local `z=0`，不是world z=0。与真实tabletop平面求交，按有限桌面截取，再变换/投影；导出平面坐标系和3D/2D端点，退化则报告。
- 每图JSON含输入hash、帧/秒/FPS、坐标约定、相机K/外参/near/far/投影、显示矩阵；每点含local/world/display/camera/2D、类型/对应ID/权重。像素左上为原点、u右v下，注明像素中心及深度定义；使用该PNG的真实相机投影。
- 批准后导出线性深度NPZ及 `in_front,in_image,projected_depth,buffer_depth,depth_tolerance_m,visibility,occluder_geom_id`；区分visible/self-occluded/other-occluded/out-of-frame/behind/unknown。手link原点常在自身mesh内，保留自遮挡结果，并以对应手部可辨认为选帧标准，不能移动原点/anchor求好看。
- 内部 `checks.json` 核验全anchor的 `E_i`、`max_m||r_im||`、det/正交性、权重和、FK/ideal差及投影；E单位m²，与存档RMSE比较用sqrt(E)。真实残差不调成零，不把微米误差/调试数字放论文图。另交独立diagnostic overlay preview供检查；论文只用raw PNG+JSON矢量重画。

## R5：图4腕部2×2，保持当前帧/时间

原渲染脚本 `ROOT/scripts/render_wrist_reference_pairs.py`，规格 `WRIST/render_spec.json`，来源manifest `WRIST/figure3/render_manifest.json`；按包内manifest实际层级核验。如下before/after顺序固定，不使用旧建议的Chair110帧。

| 行与手部 | 帧/时间 | Before参考 → After参考 | 原图（相对 WRIST/figure3） |
|---|---|---|---|
| Push，agent0右手 | 原始30Hz frame117，3.9s | `/home/kevin/holosoma-rubber-hand-largetable/data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/a1/sub6_largetable_033_fixed_object_base.npz` → 同目录 `sub6_largetable_033_fixed_object.npz` | `push/frame_117_before_hand.png`、`push/frame_117_after_hand.png` |
| Chair，person1/agent0左手 | 原始30Hz frame122，4.0666666667s | `CHAIR/shared_small_preview/core4d_pair_reference.npz` → `CHAIR/shared_small_a1_preview_tol6/core4d_pair_reference.npz` | `chair/frame_122_before_hand.png`、`chair/frame_122_after_hand.png` |

- 四参考SHA256依序为 `27f9f6c73579aab479aadd4991b08231e4459431c50b625976f8fcc4775cbc29`、`a902e79abb69974c4b4af41ee73efa3101b150e4f9a41d06c7c48d0ebfceb511`、`d8e945b4ec08cc9d22d6493942cf05ed787cd42b87b5e18ac4e6d7d6423dd586`、`1e1999a179655abaf577f78fa6eeea5a7ecd4b1e8ae86a1144118b4dca54679b`。
- 拉近到手和接触面占画面一半以上；**无箭头、无高亮框**。Push相机从桌面上方向下看手，桌子中性灰且基本不透明，不隔半透明桌面看手。Chair同帧近景、两机器人保持角色颜色。每张宽1920px。
- 每行before/after使用完全相同相机/光照/裁切，非腕部参考字段按原manifest核验不变；不得重做腕优化。Push保留 `historical_wrist_only_reference`，Chair保留 `diagnostic_preview_not_training_asset` 的内部来源状态，不能把这两对替换成正式策略rollout。

## R6：图5共享场景两版，保持当前Chair117帧

- 使用 `R2/U7_media/shared_scene/render_manifest.json` 的原始30Hz **frame117=3.9s**，比较在腕优化前。输入为 `CHAIR/two_stage/person1/person1_nominal_scaled.npz`、`CHAIR/two_stage/person2/person2_nominal_scaled.npz`、`CHAIR/shared_small_preview/core4d_pair_reference.npz`；R2中对应 `source_0.npz/source_1.npz/source_2.npz`。
- 三输入SHA256依次 `6c36ccd2f826b4ad9fbe2d3133dc04540543566fba20f3f12dfd09d3786f0d78`、`45bbeeba07af1bee87402386b0c3224c7b64312089b1b31167f50a225235d2c9`、`d8e945b4ec08cc9d22d6493942cf05ed787cd42b87b5e18ac4e6d7d6423dd586`。源椅mesh `.../CORE4D_Real/object_models/chair/chair021_m.obj`，SHA256 `7fb3283f1b67c8d08bb1f68c51d15aab5e6c940b8571e704bca265701b776161`。
- (a)沿用原 `individual_nominal`：两份椅子使用各自真实缩放/位姿，颜色深蓝 `#4A6FA5`、深沙 `#B07D3E`，alpha0.35–0.45（建议0.42）；(b)沿用 `shared_mean`：一个中性灰实体椅子，平均缩放。两panel机器人qpos完全相同；不移动两份椅子扩大差别。
- 原person尺度约0.7616803111/0.7267058143，共享0.7441930627；以NPZ/manifest完整精度为准。参考 `render_r2_shared_chair.py` 和旧 `individual_nominal_visual_only.urdf` / `shared_mean_visual_only.urdf`，不改真实缩放规则。
- 新机位稍高，从机器人不挡椅子的一侧观察；人物保留头到膝，椅子居中约占每格1/3，减少空白。两panel同相机/光照，独立raw PNG，无(a)/(b)或子标题。R3共享场景缩略图直接复用此对。

## 分阶段交付与验收

新建 `ROOT/paper_handoff_exports/FIGURES_R2_<YYYYMMDD>/`，若同日已有输出则加版本后缀避免覆盖。R0阶段先交小包；批准后补齐完整包，README明确哪阶段完成、哪些因审批/缺输入待执行。

```text
FIGURES_R2_<YYYYMMDD>/
  README.md, manifest.json, input_inventory.json, cameras.json
  R0/                         # 两张样张、manifest、全图与缩略contact sheet
  raw/                        # 批准后的无标记PNG；共用素材只存一份
  geometry/                   # 独立JSON、批准后的深度NPZ
  diagnostics/                # CPU预检、checks、批准后的各R1–R6 contact sheet/overlay
  sources/                    # 实际代码/配置快照、必要输入manifest、dirty diff
  SHA256SUMS
```

- 每资产manifest记录 `used_by:[R1,...,R6]`、类型human/retarget/source_skill/reference/execution、源路径及hash、模型/任务/seed/回合、帧/时间/FPS/时间映射、相机、材质、文件名、选择依据。参考不冒充执行，回合条件不混用；执行须追溯checkpoint SHA256，未加载checkpoint明确记 `not_loaded`。
- README记录代码commit、实际脚本/配置/命令、输入与输出SHA256、人体模型/faces与对象mesh来源、CPU后端版本、真实审批证据、缺失/退化项及复现方式。大数据无需全复制，但路径和hash必须充分；不用Mac路径作为输入。
- 批准后每R1–R6交一张诊断contact sheet（允许诊断标签，与raw分开），R4另交投影overlay；共用素材用引用避免重渲。核验固定相机、角色色、帧身份、PNG尺寸、投影/深度和完整hash。为包内文件生成SHA256SUMS（自身除外），压缩包另附SHA256。
- 由作者带回Mac，放入 `What my Ubuntu agent wants to tell you/FIGURES_R2_<YYYYMMDD>/`。Ubuntu只交素材与来源，不修改论文、字号、最终图布局或图注；任何无法核验的输入/软件后端及时报告，不虚构替代，不自行扩展为实验。
