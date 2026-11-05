# 图2交互网格：Pull A1 第118帧的精确来源核验与坐标导出

2026-10-28。本任务仅补图2的交互网格数据，**CPU、无训练、无评测、不重新优化、不改变已保存动作或已批准图片**。优先恢复原重定向网格；只有能从原配置和原时间映射核实复现时才交可绘制数据。不能精确复现就交原因报告，Mac将保留原图、不画网格。不得以近似网格、最近帧、另做机器人Delaunay或人工挑边替代。本文件不包含Small Table重渲任务。

## 1. 固定对象与可见路径

- `ROOT=/home/kevin/holosoma-core4d-base`；`PKG=ROOT/paper_handoff_exports/FIGURES_R2_SUPPLEMENT_1027_2_20261028`；`RT=ROOT/src/holosoma_retargeting/holosoma_retargeting`。仓库为 `Kaifeng-Jia/holosoma-rubber-hand-marl`，分支 `core4d-base`，不reset或覆盖现有改动。
- 底图 `PKG/raw/R3_pull_a1_0118.png`，1920×1280，SHA256 `3fa13e4f4884f15d7cb0804825adf721255acb3078ad8a5357b7ae736727bc8f`；相机和来源取同名JSON，JSON SHA256 `c11480fb2d2a220f1cb22d20ed1e1641ebe4b6724909e4ea025a2c260cb0075a`。不要改相机、摆位或底图。
- 50Hz输入：`ROOT/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/pull/sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz`，SHA256 `7c1c554851b1ac1a0ab3998755dda2eddf76800e2460fa29da5e4d3093bd8cf7`；取0-based frame118，t=2.36 s。
- 30Hz A1输入：`ROOT/data/retargeted/rubber_hand_largetable_v1/sub3_largetable_010/pull_a1_tol1deg/sub3_largetable_010_fixed_object.npz`，SHA256 `6ac57802dbd6c202e80ae97b31678d850751c37ba133cb49f968351ba9178c05`。原序列191帧，runtime317帧。同目录定位 `_fixed_object_base.npz`、`_nominal_scaled.npz`、原命令/配置/日志；找不到不能假称已查到。
- 人体原输入 `/home/kevin/holosoma/OMOMO_new/sub3_largetable_010.pt`，SHA256 `43773e87ff49e7dc1c65be424cc876e859029acb4cbb7c263835e521c9c509bb`。这是网格源关键点链，不用本轮另建的人体表面或R4的`preprocessed_source.npz`替代。
- 原视觉模型 `RT/models/g1/g1_29dof.urdf` SHA256 `1c5a19d8440d246fd6a21645de0b9bc8b92c99ee5cd895607a402db9eb02a32b`；桌 `RT/models/largetable/largetable.urdf` SHA256 `7e69319366888310890e1accac8fd137bb9cbd85ad04a922e20e97cade3109e2`，网格 `RT/models/largetable/largetable.obj` SHA256 `3bf0d31812f2fa05f3bb5502d3014bb7c6bdce858bcb4ec496bf91008b86f701`。原FK模型还需核对实际使用的 `RT/models/g1/g1_29dof_w_largetable.xml` 及include；不能拿训练机器人覆盖。

## 2. 先核可复现性，再导出；不启动求解器

1. 核对以上输入hash，定位生成这一条Pull的原有效配置/代码commit。入口为 `ROOT/STEP1_MOTION_MANIFEST.yaml` 的 `sub3_largetable_010`（variant为`a1_pt_wrist_orientation_tol1deg_bounded_pull`）、`ROOT/retarget-rubberhand-A1.md` 和上述retarget目录；后者通用说明不能代替该Pull实际配置。历史工作树 `/home/kevin/holosoma-rubber-hand-largetable` 也可只读查找来源，不切换/更新其资产。保存实际命令、配置、dirty diff、代码hash和依赖版本。
2. 核查 `RT/data_conversion/convert_data_format_mj.py:MotionLoader` 与实际转换参数，特别是 `line_range`、`use_omniretarget_data`、fps和索引起点。已收包有同脚本快照 `PKG/sources/convert_data_format_mj.py`。按原`_compute_frame_blend`、线性插值和四元数SLERP在CPU检查A1输入能重现runtime118的root/joints/object，并记录实际误差；不运行转换脚本的主入口。
3. **时间歧义须显式解决。** 在无裁剪且30→50Hz的通常路径中，118/50×30=70.8，对应原索引70/71及约0.8混合，而不是原始118或最近71。原代码在30Hz逐帧建离散拓扑；不能对四面体索引插值、取并集或擅自选择一个邻帧。优先取原保存的该时刻网格或原链明确规定的网格/源关键点重采样规则。若原流程没有定义这个插值时刻的网格，就记录`status=not_reproducible`，停止可绘制网格导出；不要新增人体插值规则来制造一个“原网格”。可以检查70/71拓扑用于解释原因，但两者相同本身不证明新插值规则属于原流程。
4. 核查A1与前两阶段的关联：`RT/examples/robot_retarget.py:run_fixed_object_size_adaptation`中Stage 2采用源nominal场景拓扑去适应physical object，Stage 3仅腕部优化、继承`human_joints`。以实际base/A1文件验证root、非腕部、物体及human_joints不变；允许腕部qpos列26/27/28/33/34/35变化。图上点位置必须来自底图的**A1后runtime118 FK**，不能用preopt姿态点覆盖，也不能因此给A1后点重新Delaunay。

## 3. 可精确复现时采用的原定义

以下是已查到的代码定义，仍须与该Pull运行的原版本/覆盖配置核对；无需读论文。

- `RT/config_types/data_type.py:JOINTS_MAPPINGS[("smplh","g1")]`规定有序映射：Pelvis、L/R_Hip、L/R_Knee、L/R_Shoulder、L/R_Elbow、L/R_Ankle、L/R_Toe、L/R_Wrist，共15点；源索引取同文件`SMPLH_DEMO_JOINTS`，目标link取映射value，保留顺序和名称。不要换成人体表面采样或任意机器人关节点。
- `RT/examples/robot_retarget.py:setup_object_data`的object_interaction分支调用 `RT/src/utils.py:load_object_data(..., sample_count=100)`；已知实现使用`trimesh.sample.sample_surface_even`、默认seed42。使用原mesh、scale、采样实现与原依赖版本，优先恢复保存的实际样本；实际点数可能少于100，不补点凑数。导出原始object-local样本及`smpl_scale`后的demo样本、顺序和hash，查原height_dict/配置，不能猜scale。
- 源人体须沿原`load_motion_data`、`preprocess_motion_data`、`create_grounded_nominal_object_poses`处理，或使用经一致性核验的已保存`human_joints`和nominal物体轨迹。不能重复缩放/落地。Stage 2的Delaunay输入为：
  `V_source = vstack([R_demo.T @ (human_mapped_world − t_demo), object_points_local_demo])`。
  `RT/src/interaction_mesh_retargeter.py:retarget_motion`调用 `RT/src/utils.py:create_interaction_mesh`，即`scipy.spatial.Delaunay(V_source).simplices`。保留原SciPy/Qhull选项与点序，不额外joggle/随机扰动“修好”退化输入。
- 显示在机器人图上的点为 `V_robot = vstack([FK(q_A1_118, ordered_robot_links), R_physical_118 @ object_points_local + t_physical_118])`；**继承上述源拓扑**。可参照原`visualize_motion`/`_get_robot_link_positions`，只调用FK和几何函数；不能运行`retarget_motion`/`iterate`或腕部优化。每个四面体的6条无向边去重得到edge list，保留全部原四面体和点ID，不为美观重新连边。
- 验证A1 runtime、原模型的关节顺序/四元数约定与 `PKG/sources/supp1027_r3.py`一致；至少核对pelvis、两手、两踝FK与runtime body_pos_w。人/物整体坐标不移动，不使用R4配准后的双机器人数据。

## 4. 同相机投影、可见性与交付

- 直接读取 `PKG/raw/R3_pull_a1_0118.json` 的完整`camera`，包括K、world_to_camera、resolution、near/far、display_transform；不要仅用azimuth/elevation重新估相机。投影：`Xc=Rcw Xw+tcw`，`u=fx Xc.x/Xc.z+cx`、`v=fy Xc.y/Xc.z+cy`，坐标是原1920×1280图左上角像素中心，u右、v下。不要提前应用Mac裁切；Mac的裁切框为(500,330,1260,1030)。
- 若需depth/segmentation，只允许CPU OSMesa/llvmpipe、同scene/相机/姿态/材质做离屏几何检查；使用 `PKG/sources/figround_common.py:render(depth=True)` 的实际实现作参考，另写新目录输出，不能直接执行写死OUT的旧脚本。`MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`、`LIBGL_ALWAYS_SOFTWARE=1`、`CUDA_VISIBLE_DEVICES=""`，记录GL renderer。
- 各点导出3D、2D、camera-depth、in-frustum、depth visibility及occluder geom ID；link原点可能被自己的mesh挡住，单列self-occlusion，不移动点。桌为半透明，记录depth如何处理透明物体。边导出端点ID及遮挡/可见分段或采样mask；不能只用两端可见就断言整条边可见。Mac自行画深色细线/黄色圆点，不烧字进底图。
- 新输出目录 `ROOT/paper_handoff_exports/FIGURES_FINAL_MESH_<YYYYMMDD>/`，不覆盖已批准包。成功时仅需 `interaction_mesh.json`（点/边/相机/可见性）、`interaction_mesh_arrays.npz`（源/目标3D、样本、tetrahedra、edges）、一张同底图诊断overlay、`README.md`、实际导出脚本/原配置快照和`SHA256SUMS`。README写明原retarget与当前导出commit、脚本/配置/input/model hash、依赖版本、帧映射、A1关联、核验结果和`status=exact_reconstruction`的依据。
- 不能精确复现时只交`README.md`＋`reproducibility_report.json`＋相关证据/脚本与hash，列明卡在原配置、样本、模型、时间规则或拓扑哪一步，写`status=not_reproducible`；无需重训、重retarget或生成替代图片。交作者带回Mac放入 `What my Ubuntu agent wants to tell you/FIGURES_FINAL_MESH_<YYYYMMDD>/`。
