#!/usr/bin/env python3
"""T14 step 7, layer 1 (Kevin 2026-11-24 part 3 item 2): single-robot retarget of one OMOMO sequence of a box-approximated
object, with the round-6 C settings (system1/t14/single_shared.py round 6 C, method unchanged):

  * Stage 1 (nominal scaled; the existing pipeline's first stage, examples/robot_retarget.run_fixed_object_size_adaptation
    lines "Stage 1"): default RetargeterConfig with the refinements off, the scene's object uniformly scaled by smpl_scale
    (create_uniformly_scaled_object_scene_xml), human-side object points for both sides, original=True. The existing Push /
    Pull Stage-1 files were made by that code; new sequences have none, so it is run here with the same code.
  * Stage 2 = single_shared.py round 6 C: interaction mesh with palm points (L_Palm / R_Palm, omomo_palms), simulator hand
    hulls and simulator kinematics / shapes (sim_scene), nominal prior on all 36 qpos, lambda 5, tau 10, anchor term weight 10
    (contact windows = SMPL-X hand vertex within 1.5 cm of the captured mesh; anchor = median palm point in the object frame
    projected onto the scene's collision box), palm tracking weight 5 (palm normal and finger direction), support foot
    locked at its segment start (+-1 mm; support = person toe speed < 0.01 m/s after Gaussian sigma 8), elastic object 1e5 /
    foot 1e4, foot height +-5 mm.
Object (robot side) = box_assets.py box at the captured size; human side = the captured mesh: the interaction-mesh object
points are sampled on the box (load_object_data, 100 points, seed 42) and their human-side counterparts are the nearest
points of the captured mesh (box frame, metres) x smpl_scale (as round 6 uses the capture table the hands touch).
Object pose = the captured pose in OMOMO_new (box_assets.poses_omomo_new), not InterMimic's object pose (InterMimic's meshes
for these objects are not in this repository).

    hsretargeting_s1/bin/python -B system1/t14/objects/single_object.py --object smallbox --sequence sub10_smallbox_007 --tag smallboxb \
        --out logs_system1/t14/step7/solves/single_smallboxb_C
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RT = ROOT / "src/holosoma_retargeting/holosoma_retargeting"
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

FOOT_XY_TOL, ELASTIC_OBJ, ELASTIC_FOOT, FOOT_H_TOL = 1e-3, 1.0e5, 1.0e4, 5e-3  # single_shared.py (round 6)
PALM_KEYS = {"L_Palm": "left_rubber_hand_link", "R_Palm": "right_rubber_hand_link"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def human_hand_sd(v, name):
    """[T, 2] largest signed distance of the person's SMPL-X hand vertices to the captured mesh (contact_parts rule 4 with its
    distance guard, every frame, 30 Hz; -inf = far); and the object speed (m/s) of the stored translation (as
    dual_stretch.human_hand_sd)."""
    import smplx
    import torch
    import contact_parts as cp
    T = len(v["trans"])
    mdl = smplx.create(str(cp.SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        V = mdl(betas=torch.tensor(np.repeat(v["betas"], T, 0), dtype=torch.float32), global_orient=torch.tensor(v["root_orient"], dtype=torch.float32),
                body_pose=torch.tensor(v["pose_body"], dtype=torch.float32), transl=torch.tensor(v["trans"], dtype=torch.float32)).vertices.numpy().astype(np.float64)
    hv = cp.hand_vertex_ids()
    o = cp.obj(f"omomo_{name}")
    out = np.full((T, 2), -np.inf)
    s, R, tr = np.asarray(v["obj_scale"], float), np.asarray(v["obj_rot"], float), np.asarray(v["obj_trans"], float)[:, :, 0]
    for hi, side in enumerate(("left", "right")):
        for t in range(T):
            if not (np.isfinite(s[t]) and s[t] > 0):
                continue
            pl = ((V[t, hv[side]] - tr[t]) @ R[t]) / s[t]
            if np.linalg.norm(pl.mean(0) - o["c"]) * s[t] > o["r"] * s[t] + 0.25:
                continue  # contact_parts.hand_frame's guard: a hand this far is not in contact (open meshes give wrong signs far away)
            dd, j = o["tree"].query(pl)
            sgn = -np.sign(np.einsum("ij,ij->i", pl - o["p"][j], o["n"][j]))
            out[t, hi] = (sgn * dd * s[t]).max()
    speed = np.linalg.norm(np.gradient(tr, axis=0), axis=1) * 30.0
    return out, speed


def anchor_term(h_sd, palms_raw, object_poses_raw, scene_xml, body, weight):
    """single_shared.step2_anchor_term_omomo with the object's own captured contact (h_sd) and box frame."""
    from scipy.spatial.transform import Rotation
    from object_surface import UnionSurface
    from single_shared import scene_object_boxes
    from table_shared import contact_windows, smoothstep_window
    T = len(palms_raw)
    R = Rotation.from_quat(object_poses_raw[:, [1, 2, 3, 0]]).as_matrix()
    local = np.einsum("tji,thj->thi", R, palms_raw - object_poses_raw[:, None, 4:7])
    surf = UnionSurface(scene_object_boxes(scene_xml, body))
    anchors, act, normals = np.zeros((T, 2, 3)), np.zeros((T, 2)), np.zeros((T, 2, 3))
    rec = dict(weight=weight, ramp_frames=5, surface="scene collision box (box_assets.py)", windows=[])
    for hi, side in enumerate(("left", "right")):
        for s0, s1 in contact_windows(h_sd[:, hi] >= -0.015):
            med = np.median(local[s0:s1, hi], axis=0)
            anc, dist, nrm = surf.nearest(med[None], normals=True)
            anchors[s0:s1, hi] = anc[0]
            normals[s0:s1, hi] = nrm[0]
            act[s0:s1, hi] = smoothstep_window(s1 - s0)
            rec["windows"].append(dict(hand=side, start=int(s0), stop=int(s1), median_local_m=med.round(5).tolist(), anchor_local_m=anc[0].round(5).tolist(),
                                       projection_m=round(float(dist[0]), 5), outward_normal=nrm[0].round(4).tolist()))
    return dict(keys=("L_Palm", "R_Palm"), weight=weight, anchors=anchors, activation=act, normals=normals), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--tag", required=True, help="file prefix (no underscore): <tag>_new_qpos.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    assert "_" not in a.tag
    t0 = time.time()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import box_assets as ba
    import omomo_motion as om
    from single_shared import OMOMO_NEW, omomo_palms
    seq, name = a.sequence, a.object
    v = om.load_sequences({name})[seq]
    palms, fit = omomo_palms(seq)
    bf = ba.box_frame(name, v)
    assets = ba.write_assets(name, bf, out / "assets")
    box_poses, invalid = ba.poses_omomo_new(v, bf, fit)
    h_sd, obj_speed = human_hand_sd(v, name)
    np.savez(out / "human_contact.npz", h_sd=h_sd, object_speed=obj_speed)
    os.chdir(RT)
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import (_compute_q_init_base, build_retargeter_kwargs_from_config, convert_object_poses_to_mujoco_order,
                                                              create_grounded_nominal_object_poses, create_task_constants, load_motion_data, setup_object_data)
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src.utils import (create_uniformly_scaled_object_scene_xml, extract_foot_sticking_sequence_velocity, preprocess_motion_data,
                                                build_pt_wrist_palm_orientation_targets, load_intermimic_wrist_quaternions)
    from palm_retargeter import PalmInteractionMeshRetargeter
    from table_shared import palm_point, patch_hand_meshes, write_sim_hand_meshes
    from sim_scene import simify_scene
    import hand_landmarks as hl
    import support
    import trimesh
    import holosoma_retargeting

    robot_config = RobotConfig(robot_type="g1")
    robot_config = replace(robot_config, nominal_tracking_indices=np.arange(7 + robot_config.ROBOT_DOF, dtype=np.int64))
    motion_config = MotionDataConfig(data_format="smplh", robot_type="g1")
    task_config = TaskConfig(object_name=name)
    base = RetargeterConfig(visualize=False, debug=False)
    constants = create_task_constants(robot_config, motion_config, task_config, "object_interaction")
    constants.OBJECT_URDF_FILE, constants.OBJECT_MESH_FILE = assets["urdf"], assets["mesh"]
    human, _intermimic_obj, smpl_scale = load_motion_data("object_interaction", "smplh", OMOMO_NEW, seq, constants, motion_config)
    assert len(human) == len(box_poses), (len(human), len(box_poses))
    object_poses = box_poses.copy()
    human_raw52, names52 = human.copy(), list(constants.DEMO_JOINTS)
    # ---------------------------------------------------------------- Stage 1 (existing first stage, nominal scaled scene)
    constants.SCENE_XML_FILE = assets["scene"]
    obj_pts, obj_pts_demo_box, urdf_path = setup_object_data("object_interaction", constants, None, smpl_scale, task_config, False)
    mesh_box = ba.mesh_in_box_frame(name, bf)
    cap_pts = trimesh.proximity.closest_point(mesh_box, obj_pts)[0]
    obj_pts_demo = cap_pts * smpl_scale  # human side: the captured mesh the hands touch, scaled with the person
    k1 = build_retargeter_kwargs_from_config(base, constants, urdf_path, "object_interaction")
    k1["nominal_tracking_tau"] = base.nominal_tracking_tau
    tmp1 = out / "stage1_scene"
    tmp1.mkdir()
    k1["scene_xml_path"] = create_uniformly_scaled_object_scene_xml(assets["scene"], name, smpl_scale, tmp1 / Path(assets["scene"]).name)
    ret1 = InteractionMeshRetargeter(**k1)
    human1, object_poses1, _ = preprocess_motion_data(human, ret1, motion_config.toe_names, scale=smpl_scale, object_poses=object_poses)
    nominal = create_grounded_nominal_object_poses(object_poses1, smpl_scale)
    nominal_mj = convert_object_poses_to_mujoco_order(nominal)
    phys_mj = convert_object_poses_to_mujoco_order(object_poses1)
    n = a.max_frames or len(human1)
    foot1 = extract_foot_sticking_sequence_velocity(human1[:n], ret1.demo_joints, motion_config.toe_names)
    foot1[0][motion_config.toe_names[0]] = False
    foot1[0][motion_config.toe_names[1]] = False
    q_init1 = _compute_q_init_base("object_interaction", "smplh", human1[:n], nominal[:n], constants)
    t1 = time.time()
    q1, *_ = ret1.retarget_motion(human_joint_motions=human1[:n], object_poses=nominal_mj[:n], object_poses_augmented=nominal_mj[:n],
                                  object_points_local_demo=obj_pts_demo, object_points_local=obj_pts_demo, foot_sticking_sequences=foot1,
                                  q_a_init=q_init1, q_nominal_list=None, original=True, dest_res_path=str(out / f"{a.tag}_nominal_scaled.npz"))
    q1 = np.asarray(q1, dtype=np.float64)
    stage1_seconds = round(time.time() - t1, 1)
    del ret1
    # ---------------------------------------------------------------- Stage 2 (round 6 C)
    human = np.concatenate([human, palms], 1)
    constants.DEMO_JOINTS = list(constants.DEMO_JOINTS) + list(PALM_KEYS)
    constants.JOINTS_MAPPING = dict(constants.JOINTS_MAPPING) | PALM_KEYS
    sim_hands, dump_sha = write_sim_hand_meshes(out)
    scene = patch_hand_meshes(Path(assets["scene"]), out / "scene_simhands.xml", sim_hands)
    scene, sim_rec = simify_scene(scene, out / "scene_sim.xml", out / "sim_meshes", f"{name}_link")
    constants.SCENE_XML_FILE = str(scene)
    offsets = {k: palm_point(trimesh.load(sim_hands[s], force="mesh").vertices, s) for k, s in (("L_Palm", "left"), ("R_Palm", "right"))}
    cfg = replace(base, apply_manual_joint_limit_overrides=False, foot_sticking_tolerance=FOOT_XY_TOL,
                  elastic_constraints=replace(base.elastic_constraints, enable=True, object_collision_weight=ELASTIC_OBJ, foot_kinematics_weight=ELASTIC_FOOT))
    kw = build_retargeter_kwargs_from_config(cfg, constants, urdf_path, "object_interaction")
    kw["nominal_tracking_tau"] = 10.0
    kw["anchor_nominal_foot_height"] = True
    kw["nominal_foot_height_tolerance"] = FOOT_H_TOL
    kw["scene_xml_path"] = str(scene)
    at, at_rec = anchor_term(h_sd, palms, object_poses, scene, f"{name}_link", 10.0)
    wq = load_intermimic_wrist_quaternions(OMOMO_NEW / f"{seq}.pt")
    palm_rot, cal_err = build_pt_wrist_palm_orientation_targets(human_raw52, wq, names52)
    orient = dict(weight=5.0, activation=np.ones((len(palm_rot), 2)), palm_world=palm_rot[..., :, 2], finger_world=palm_rot[..., :, 0],
                  palm_normal_local=hl.PALM_NORMAL_LOCAL, finger_weight=None)
    ret = PalmInteractionMeshRetargeter(**kw, palm_offsets=offsets, anchor_term=at, orient_term=orient)
    assert list(ret.track_nominal_indices) == list(range(36))
    human2, object_poses2, _ = preprocess_motion_data(human, ret, motion_config.toe_names, scale=smpl_scale, object_poses=object_poses)
    prep_err = float(np.abs(human2[:, :52] - human1).max())
    assert prep_err < 1e-6, prep_err  # float32 input (52 joints) vs float64 after the palm points are appended
    assert np.abs(object_poses2 - object_poses1).max() < 1e-12
    toes = human2[:n][:, [ret.demo_joints.index(t) for t in motion_config.toe_names]]
    sup = support.support_flags(toes)
    foot = support.as_solver_sequence(sup)
    ret.enable_stance_lock(sup)
    q_nom = q1[:n].copy()
    q_nom[:, -7:] = phys_mj[:n]
    t2 = time.time()
    q, *_ = ret.retarget_motion(human_joint_motions=human2[:n], object_poses=nominal_mj[:n], object_poses_augmented=phys_mj[:n],
                                object_points_local_demo=obj_pts_demo, object_points_local=obj_pts, foot_sticking_sequences=foot,
                                q_a_init=q_nom[0], q_nominal_list=q_nom, original=False, dest_res_path=str(out / f"{a.tag}_new.npz"))
    q = np.asarray(q)
    np.savez(out / f"{a.tag}_new_qpos.npz", qpos=q, object_qpos_mj=phys_mj[:n], human_joints=human2[:n], fps=30, smpl_scale=smpl_scale)
    import resource
    rec = dict(object=name, sequence=seq, tag=a.tag, variant="round6C", python=sys.executable, holosoma_retargeting_module=str(Path(holosoma_retargeting.__file__).resolve()),
               box=assets, palm_fit=fit, object_pose="captured pose (original release) composed with the box frame, mapped by the palm fit",
               invalid_object_pose_frames=invalid, smpl_scale=smpl_scale, r_box_over_scaled_person_object=round(1.0 / float(smpl_scale), 4),
               stage1=dict(file=str(out / f"{a.tag}_nominal_scaled.npz"), sha256=sha(out / f"{a.tag}_nominal_scaled.npz"), seconds=stage1_seconds,
                           config="RetargeterConfig() defaults, refinements off, scene object x smpl_scale, original=True", nominal_tracking_tau=base.nominal_tracking_tau),
               foot=dict(rule="person toe speed < 0.01 m/s after Gaussian smoothing sigma 8 frames; support foot locked at segment start +-1 mm",
                         support_share=np.round(sup.mean(0), 4).tolist(), segments=[int(np.sum(np.diff(np.concatenate([[0], sup[:, i].astype(int), [0]])) == 1)) for i in (0, 1)]),
               anchor_term=at_rec, palm_track=dict(weight=5.0, finger_weight=5.0, calibration_p90_deg=np.round(cal_err, 6).tolist(), prior_tau=10.0),
               sim_scene=sim_rec, prior=dict(scope="all", qpos_indices="0..35", tau=10.0, w_nominal=ret.w_nominal_tracking_init), physx_dump=dump_sha, scene=str(scene),
               robot_palm_offsets_m={k: np.round(x, 6).tolist() for k, x in offsets.items()}, interaction_mesh_keys=list(ret.laplacian_match_links),
               object_points=int(len(obj_pts)), human_side_points="nearest captured-mesh points of the box samples x smpl_scale",
               human_side_points_gap_p95_m=round(float(np.percentile(np.linalg.norm(cap_pts - obj_pts, axis=1), 95)), 4), frames=n,
               profile=dict(foot_xy_tolerance=FOOT_XY_TOL, elastic_object=ELASTIC_OBJ, elastic_foot=ELASTIC_FOOT, foot_height_tolerance=FOOT_H_TOL, manual_joint_limits=False),
               preprocess_stage1_vs_stage2_max_abs=prep_err, seconds=round(time.time() - t0, 1), stage2_seconds=round(time.time() - t2, 1),
               finite=bool(np.isfinite(q).all()), qpos_shape=list(q.shape), object_locked_max_err=float(np.abs(q[:, -7:] - phys_mj[:n]).max()),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3), command=" ".join(sys.argv))
    (out / "run_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("seconds", "finite", "qpos_shape", "object_locked_max_err", "peak_rss_gb")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
