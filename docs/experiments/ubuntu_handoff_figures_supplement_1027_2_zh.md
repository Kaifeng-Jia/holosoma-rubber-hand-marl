# 1027_2 图表素材补交：Small Table 替换 teaser 的 Bucket，并补 R1/R3、修 R4 Pull / R5 Push

2026-10-27。作者已确认图1第4行改为 **Small Table（T-07 seed721）**；本条决定优先于旧复评中的“待作者决定”。图1顺序为 Push、Pull、Chair、Small Table，每行1帧人体示范＋5帧执行。Bucket已有素材保留在原包，表I和补充视频不因本轮改动。Ubuntu依据本文件执行，无需访问Mac文件或论文。

只做已有数据核验、人体表面重建和CPU渲染，不训练、不重新评测、不改参考/轨迹、不重新retarget、配准或腕部优化，不运行策略或物理步进。以下沿用R1–R6编号；已通过素材只引用复用，不整包重做。

## 路径、环境与本次输出

- `ROOT=/home/kevin/holosoma-core4d-base`，仓库 `Kaifeng-Jia/holosoma-rubber-hand-marl`、分支 `core4d-base`。
- `APPROVED=ROOT/paper_handoff_exports/FIGURES_R2_20261027_approved`；其 `sources/figround_*.py`、manifest和原图是上一包实物。**不要直接执行其中写死OUT的脚本**；复制必要导出逻辑并将OUT改到本次新目录，不覆盖已批准包。
- `OLD_EVAL=ROOT/logs/PaperSupplement20260929/unified_evaluation_20261003`；`GEOM=ROOT/logs/ReferenceGeometry/reference_geometry_round2_20261017`；`PILOT=ROOT/logs/ReferenceGeometry/reference_geometry_pilot_20261017`；`WRIST=ROOT/paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005`。
- 本次输出 `ROOT/paper_handoff_exports/FIGURES_R2_SUPPLEMENT_1027_2_<YYYYMMDD>/`，重试加版本后缀。原包及训练资产保持只读；新增脚本、配置、诊断和图片只写新目录。
- 已有Python：`/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python`。沿用上一包已确认的OSMesa/CPU软件后端，重新记录实际GL vendor/renderer/version；设置 `MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`LIBGL_ALWAYS_SOFTWARE=1`、`CUDA_VISIBLE_DEVICES=""`，所有人体模型参数/张量也在CPU。缺CPU后端就报告，不退回GPU。
- 记录当前commit与dirty diff，不reset/clean。复用上一包配置及其hash，不再次索要已批准C+配色；本轮只需R5 Push先交一对改善后的近景供视觉确认。

## 显示规格与复用范围

- R1人体：中性灰真实网格；R2执行、R3重定向：G1原灰白外壳/黑色部件，物体、地面、光照沿用已批准的本色执行风格；桌子保留其原浅蓝材质。不要把R2整组改成C+或白地。
- R4 Pull继续C+：源/role0 `#AAC0DE`、伙伴/role1 `#E8C8A0`；物体中性灰，纯浅背景。深色标记 `#4A6FA5` / `#B07D3E`、灰色反射线由Mac做矢量；PNG无标记。
- R5 Push保持本色灰，桌子浅蓝、降低透明度以便看清掌面；透明度、灯光、相机在before/after之间严格相同。R5 Chair、R6和R4 Push本次不重渲。
- R1/R2/R3/R4原图1920×1280、R5原图1920×1440，无文字/时间/UI/框/箭头/公式。接近主体但保留完整相关手部、物体和身体；同任务执行8帧固定相机与framing，不逐帧跟随/居中。
- 每张图保存真实相机内外参、near/far、图像尺寸、显示变换、材质和光照；整段“人/机器人＋物体”一同转换坐标，不独立挪对象、不缩放人体凑构图。
- R2 Push/Pull/Chair、R1 Chair、R4 Push、R5 Chair、R6引用APPROVED中的asset ID/文件hash即可。R3人体、协作参考及共享场景缩略图也按对应素材复用；缺项不得用机器人、骨架或AI人体填格。

## R1-A：补 Small Table 双人示范3帧

1. 正确原始任务是 **CORE4D `20231030/001`、对象 `desk001`**，不是OMOMO的同名smalltable。原始目录 `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions/20231030/001/`，定位 `person1_poses.npz`、`person2_poses.npz`、`smooth_objposes.npy`、`object_metadata.json`；保存实际hash、字段和FPS。已有 `/home/kevin/datasets/CORE4D-V1-full/derived/20231030_001_desk001_v1_canonical.npz`（SHA256 `5e27c929999bde64187ec5d71d4da02ea93bb12f78fedfcb36f7200e7990ed90`）的provenance记录207帧、15Hz、person1/person2、Y-up经Rx(+90°)转Z-up；用它核对原始时间/坐标，不能把其joints/betas直接当已有表面网格。
2. 对象网格 `/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models/desk/desk001_m.obj`，已知SHA256 `f8f9dffa9f1ccdbd2ff90125cd39c96e79bf84ccebd8a81fcef656c4fbc7c00b`，开工核对。人体使用原始逐帧vertices及正确faces；可复用上一包 `sources/figround_r1.py` 的CORE4D loader/坐标转换，不能把机器人nominal参考当人体。
3. 已有参考来源入口：`ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/source_manifest.json`；其上游目录为 `/home/kevin/datasets/CORE4D-V1-full/retarget_results/method_comparison_20260904/20231030_001_desk001_move/`，内有 `small_table_nominal_preview/` 和 `two_stage_fullnominal/`。用这些provenance核对序列、裁剪/重采样、坐标轴；不要把413帧30Hz机器人源参考自动当成原始人体帧率。
4. 参考已验证的SMPL-X拓扑位于 `/home/kevin/CS6758Project/data/smplx_models/smplx/SMPLX_NEUTRAL.npz`，SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`。只有当原始vertices确实匹配该拓扑时才复用faces；这里不替换原始人体形状/性别。
5. 从已有双人手部几何接近阶段选1主选＋2备选，优先清楚呈现两人共同扶桌。可读取同序列已有窗口/provenance或对原始几何做CPU距离检查来选帧，不新增动力学接触判定。保存原始帧/秒、窗口和对应索引依据，不能用相同数字帧号代替时轴核对。
6. 与R2 Small Table用相同观察方向、投影规格和场景风格；保留人/物真实尺度和相对位置，不宣称与策略执行逐帧同步。

## R1-B：补 Push / Pull 的原始 OMOMO 人体，各3帧

目标序列是 `sub6_largetable_033`（Push）和 `sub3_largetable_010`（Pull）。现有 `/home/kevin/holosoma/OMOMO_new/<seq>.pt` 为 `[T,591]` 状态/关节张量，不能单独证明人体表面、betas或gender。本轮允许从可核实原始参数重建人体表面，不是重新生成动作。

1. 先在Ubuntu查当初生成OMOMO_new的转换脚本、配置与上游文件：从 `/home/kevin/holosoma/`、`ROOT` 及本机已保存的OMOMO原始数据目录搜索序列名，以及 `train_diffusion_manip_seq_joints24.p` / `test_diffusion_manip_seq_joints24.p`。按记录中的 `seq_name` 精确选序列，不能仅凭文件名近似匹配。
2. 找不到本地原始数据，再由[OMOMO官方仓库](https://github.com/lijiaman/omomo_release) README 的 dataset 链接取得对应官方数据，保存下载URL、日期、文件hash与实际所用记录；不运行其训练、测试或diffusion生成脚本。官方入口：[项目页](https://lijiaman.github.io/projects/omomo/)、[README的数据链接](https://github.com/lijiaman/omomo_release#testing)。受登录/模型许可阻塞时报告具体缺项，不虚构下载成功。
3. **先核模型类型再重建。** [官方dataset loader](https://github.com/lijiaman/omomo_release/blob/main/manip/data/hand_foot_dataset.py) 的表面模型为SMPL-X，按原始gender选male/female、使用16维betas；同文件的SMPL-H路径用于骨架父表，不能仅凭本地格式标签`smplh`就换表面模型。检查原记录的 `betas,gender,trans,root_orient,pose_body`、手姿态是否提供，以及 `obj_trans,obj_rot,obj_scale`、FPS/裁剪信息；实际格式不同则据元数据和官方读取代码处理并报告。
4. 只取真实记录参数做CPU body-model forward。对应模型通常为原OMOMO数据根 `smpl_all_models/smplx/SMPLX_MALE.npz` / `SMPLX_FEMALE.npz`；先找已有模型。已有NEUTRAL文件不能代替记录所需的male/female或猜betas。保留原shape、gender、root/body姿态和物体变换，不重新拟合动作。
5. [官方 `run_smplx_model`](https://github.com/lijiaman/omomo_release/blob/main/trainer_full_body_manip_diffusion.py) 给出参数到mesh的forward逻辑；只抽取必要逻辑，不导入训练入口。其缺失手部姿态时使用零手姿态的约定如被采用，须在manifest写明 `hand_pose_source=official_default`，不把默认手指当采集的抓握细节；若有原始手姿态优先保留。核对实际输出顶点数与faces，不依赖代码注释中的旧顶点数。
6. 对象使用官方数据 `captured_objects/largetable_cleaned_simplified.obj` 或原转换链明确使用的同一mesh；按原 `obj_scale,obj_rot,obj_trans` 恢复，保留hash。人/物采用同一坐标转换；检查root平移是否需与 `trans2joint` 或canonical转换配合，避免重复偏移，核对转换后人体关节点与原始/已存张量的帧对应及数值一致性。
7. 每任务1主选＋2备选，选原动作中可辨认的几何接近阶段；与已批准R2对应行观察方向一致。保存原始帧/FPS/秒、裁剪起点、重采样映射、model/参数/mesh/faces hash。只缺一项时明确列出缺失参数/模型文件，继续其他独立任务；不要退回用joints假造表面。

## R2：新增 Small Table / T-07 seed721执行8候选

正式模型是 **T-07，smalltable5kg_A，bucket_A_no_rel_no_height**；reward名字带bucket不代表桶，不能换成T-06。已保存nominal回合成功完成687控制步，共688状态、50Hz。

| 输入 | Ubuntu路径与指纹 |
|---|---|
| 回放状态 | `OLD_EVAL/results/T-07_seed721_retry_shutdown_20261004/nominal_viser.npz`；SHA256 `5c6a399a0f25ef1c853a87a00ed392aebafb2f97948ab273748d882bc9d0e5d6` |
| 原trajectory | 同目录 `nominal.trajectory.npz`；文件SHA256 `652ccc77cc3e68b369cd9f38e6c550ac1b519a2af84307de89c60cb33e5d06cc`；旧manifest中另有 `recorded_metadata.states_sha256=389e8da5fb818572a48cbba049254585311079439265f5ab4cfc1523a185dac1`，不能把后者当NPZ文件hash |
| 原媒体manifest | `OLD_EVAL/media/T-07_seed721/manifest.json`；实际trajectory目录带`retry_shutdown_20261004`，不要擅自省略 |
| checkpoint身份 | `ROOT/logs/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920/model_12000.pt`；SHA256 `f163d11a9629edae5dd72732c4ddec08a43024533a47b349a099a3275a8d1c50`，仅溯源不加载策略 |
| G1视觉模型 | `ROOT/src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf`；SHA256 `7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25` |
| 训练桌子资产 | `ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf`；SHA256 `942cea0e35e45c67ba0377a42b382a28e2e52ba5b05238a3066697e9d1e9695d` |
| 参考身份 | 同资产目录 `core4d_pair_runtime_fps50.npz` SHA256 `582e76693f877c61b0b09ab3b584922f330aeb85cae6035f6b7cd2ae729ee153`；源pair SHA256 `d9d17e96f5f0b73aa76a1bf32f1ec50cfb7fa7fe78a847dd890882add4bbfd05` |

- 读取该回合 `episodes.json` 和原媒体manifest核对 `completed_reference=true`、seed721、nominal及上述hash。旧视频25fps不是状态50Hz；直接使用保存state做FK回放。
- 用已有交互/接近阶段证据确定start，end=687，再 `round(linspace(start,end,8))` 取8个不同状态帧；若接近起点无法核实，允许完整保存时域候选 `0,98,196,294,393,491,589,687`，明确是full-domain fallback，不宣称8帧均为接触。Mac最终选5帧。
- `state_frame` 与 `reference_frame` 分开记录；末状态687可能对应reference686，按保存字段读取，不自动设相等。写明time=state_frame/50、terminal是在reset前抓取；不得补循环首帧作为末帧。
- 同序列固定相机，尽量让机器人和桌子占足够画幅，保留整段移动范围。沿用APPROVED R2本色和原物体材质/灯光；只修改新导出脚本，不改T-07资产或状态。

## R3：补 Pull A1运行参考缩略图

图2“重定向”节点包含腕部优化，**A1后参考允许直接使用**，不再要求寻找优化前baseline。本次只核模型/关节顺序并渲一张单机器人＋源桌子；最多补两张候选，不用单机器人rollout代替。

- 使用 `ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/pull/sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz`，SHA256 `7c1c554851b1ac1a0ab3998755dda2eddf76800e2460fa29da5e4d3093bd8cf7`。
- 已知来源：`/home/kevin/holosoma/OMOMO_new/sub3_largetable_010.pt` SHA256 `43773e87ff49e7dc1c65be424cc876e859029acb4cbb7c263835e521c9c509bb` → `ROOT/data/retargeted/rubber_hand_largetable_v1/sub3_largetable_010/pull_a1_tol1deg/sub3_largetable_010_fixed_object.npz` SHA256 `6ac57802dbd6c202e80ae97b31678d850751c37ba133cb49f968351ba9178c05` → 上述fps50参考。
- 读取该runtime字段与 `src/holosoma/tests/envs/marl/test_plan5_push_manager.py` 的载入配置，确认实际机器人模型、29关节顺序、四元数约定和物体URDF；已有生成manifest可辅助核对，但不以找齐旧生成配置作为补图前置条件；不要因名称相同就拿训练模型覆盖原模型。用原retarget模型/对象做FK，保留A1状态。
- 本色G1，不上C+；manifest类型记 `retarget_reference_after_A1`，不是未经优化的输出，不是策略rollout。渲染原始单机器人和原源物体，无伙伴。图2其他缩略图按其现有节点复用。

## R4：仅补 Pull侧视，简化为一组可见镜像手对应

保留当前数值通过的数据，优先frame118；需要备选时用当前153/188，同为50Hz。输入 `GEOM/pull/preprocessed_source.npz`、`new_pair_reference.npz`、`correspondence_spec.json`、`fitted_candidates.json`、`selection.json`、`hand_samples.npz`，也可用APPROVED已封存同hash副本；不重拟合/重选目标对应。

- 侧视方向由真实展开轴/物体坐标确定，使视线近似垂直于两角色展开方向，俯视约20–30°（MuJoCo elevation约−20至−30°）；不能把object-local轴硬当world轴。源单机器人图与pair图同观察方向，各自固定framing。希望两角色左右展开、反射线居中，但必须服从实际投影，不后处理拉歪、平移几何或强制竖直线。
- **只显示近相机的一只源手及其固定对应**：若选源L，则role0 L、伙伴role1 R；若选源R，则role0 R、伙伴role1 L。保留源手→伙伴手L↔R映射，不独立挑一个更好看的伙伴手。以手部实际可辨认和对应anchor可见为依据，逐图记录 `paper_visible_subset`；被遮挡的点不画，图注由Mac注明“visible hand shown”。
- 复用当前候选的源anchor索引、window/face/corner permutation及目标对应；只改相机和显示子集。若该固定对应无法同时可见，先调侧视角或用既有备选帧；仍不足则报告，不挪点、不另选目标最近点。自遮挡link原点不能冒充可见表面点；实际FK和ideal mapped points继续分开保存。
- 每个保留role只画一条从匹配anchor指向手点的偏移箭头（anchor → hand，对应 h−a），端点类型在JSON写清。`hand_samples`是link局部网格采样，必须FK到当前帧；不把采样下标当时间。
- 更新全部点的3D/2D及depth visibility，保留旧的完整数据与实际误差，新增可见子集。原图无标记；另交只绘可见子集的diagnostic overlay，不把被遮挡的线画穿物体。无关的被遮挡点仍在JSON中，不删证据。
- 反射线继续使用摆位后相对映射 `Q10=Q1 Q0^T`、`c10=c1−Q10 c0`，固定平面 `(I−Q10)x=c10`；本例共享物体local z=0，非world z=0。与真实tabletop平面求交并裁到桌面有限边界，再用新相机投影；保存平面坐标系及3D/2D端点。
- 内部复核 `r_im=Q_i a_m+c_i−a'_im`、`E_i=Σ_m w_m||r_im||²`（原全量权重和为1），以及FK、实际/ideal差、重投影和深度。只更换显示子集不改变这些全量核验；残差数值不进入论文图片。
- 交frame118源/pair的一组侧视小样和更新JSON；仅当该组仍不可读时追加现有153/188候选，不再广泛扫相机或重渲Push。

## R5：先核同模型，再补 Push近景一对

现有证据支持before/after为同一模型：两图robot URDF、37条mesh来源和53条geom映射一致，上一包脚本在循环外只建一个scene，再改变FK姿态。因此“after换成rubber hand”目前不是事实。开工仍须用实际输入断言一致，并用诊断图解释外形差别，不虚写两种模型图注。

- 保留原始30Hz **frame117=3.9s**。Before：`/home/kevin/holosoma-rubber-hand-largetable/data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/a1/sub6_largetable_033_fixed_object_base.npz`，SHA256 `27f9f6c73579aab479aadd4991b08231e4459431c50b625976f8fcc4775cbc29`；After同目录 `sub6_largetable_033_fixed_object.npz`，SHA256 `a902e79abb69974c4b4af41ee73efa3101b150e4f9a41d06c7c48d0ebfceb511`。
- 原机器人URDF：`/home/kevin/holosoma-rubber-hand-largetable/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf`，SHA256 `1c5a19d8440d246fd6a21645de0b9bc8b92c99ee5cd895607a402db9eb02a32b`；原桌URDF同retargeting目录 `models/largetable/largetable.urdf`，SHA256 `7e69319366888310890e1accac8fd137bb9cbd85ad04a922e20e97cade3109e2`。使用 `WRIST/render_spec.json`、`APPROVED/sources/figround_r5r6.py` 与 `render_wrist_reference_pairs.py` 的已核验来源。
- 同一scene、模型/mesh/faces/primitive尺寸/局部origin/材质/关节顺序断言一致。沿用 `validate_pair`：非腕部qpos列、object与human_joints逐值不变；允许改变的qpos列为26/27/28/33/34/35。保存断言结果和两次渲染geom指纹，不改qpos。
- 额外交 **两张仅诊断用隐藏桌面的before/after图**，同相机同模型，显示实际手掌/手指是否因转腕露出不同表面；诊断图不是论文成图，也不能拿移开的桌子或补画的手当最终图。若核验发现实际不同模型，停止这一对并报告具体差异，不静默替换。
- 正式一对从斜上方看到掌面/腕部方向与桌沿，手和相关接触面占画面一半以上；保留真实姿态，不抬手、不降桌、不删遮挡网格。桌子浅蓝，相对旧版半透明材质向更不透明方向调整，但不机械锁死alpha=1；两图alpha必须相同，记录实际值与原值。若手主要在桌面下方而无法兼顾，交最清楚的一对及可见性限制，不假造无遮挡。
- 弱化抢眼的硬阴影、提高抗锯齿；无箭头、无框。保持before/after相机/投影/裁切/光照完全一致。先交这一对近景＋隐藏桌面的诊断对供作者确认，本轮不重做Chair或整组腕图。

## R6：复用，无新增渲染

继续用 `APPROVED/raw/R6_chair_individual_nominal.png` 与 `R6_chair_shared_mean.png`，原frame117、30Hz、3.9s，C+与两个副本alpha0.42保持。Mac统一裁切并检查最终单栏轮廓；本轮只在复用清单引用两图及其JSON/hash，不重渲、不改变副本位置/比例。

## 交付与验收

- 优先顺序：①输入清点及R5模型断言；②R5近景小样与R4 Pull侧视小样；③可独立并行的R1原始数据定位/人体、R2 Small Table、R3 Pull；④完整性核验。某项缺模型/数据时写明缺项并继续其余工作，不把整包状态写complete。
- 新包包含 `README.md`、`manifest.json`、`reuse_manifest.json`、各R1/R2/R3/R4/R5子项manifest、`raw/`、`geometry/`、`diagnostics/`、实际`sources/`快照和`SHA256SUMS`。每个新增子项一张contact sheet；R4另有可见子集overlay，R5隐藏桌面诊断独立命名。无需把已批准原图复制一遍。
- 逐图记录类型human/reference/execution、任务/序列/模型ID/seed/回合、state/reference/source帧与FPS/秒、输入路径+SHA256、相机/显示变换、材质/光照、文件hash；执行图附checkpoint指纹但不加载策略。人体另记录model_type/gender/betas来源、模型/faces hash、重建参数与手姿态默认约定。
- README给出commit、dirty diff、实际脚本/配置/命令及SHA256、CPU后端、各项完成/缺失状态和复用资产ID。对新增输入和原图生成SHA256SUMS；压缩包另附SHA256。来源或hash不符时停止该项并列明差异，不换相似资产。
- 交作者带回Mac的新目录 `What my Ubuntu agent wants to tell you/FIGURES_R2_SUPPLEMENT_1027_2_<YYYYMMDD>/`。本文件不授权上传GitHub、不直接修改论文，也不要求重新运行训练或评测。
