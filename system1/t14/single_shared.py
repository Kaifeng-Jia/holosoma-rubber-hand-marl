#!/usr/bin/env python3
"""T14 item 1 (single person -> two robots), part A: new single-robot retarget for Push (OMOMO sub6_largetable_033) and
Pull (OMOMO sub3_largetable_010), T14 section 1 step 1 (same settings as the Table step-1 reference):

* interaction mesh = the existing 15 SMPL-H body joints + 2 palm points (L_Palm / R_Palm) + object points; human palm point =
  mean of the 67 palm-side SMPL-X vertices per hand (table_shared.palm_vertex_ids), computed from the original OMOMO SMPL-X
  parameters (flat hands) and mapped into the OMOMO_new (InterMimic) world by the rigid transform fitted on the 22 body
  joints of the same sequence (the fit residual is recorded; it is 0.0 for both sequences); robot palm point =
  _derive_palm_contact_point's construction on the simulator's hand hull;
* hand non-penetration with the simulator's rubber-hand hull (PhysX-cooked, T10 dump) instead of the retargeting OBJ;
* object = the existing physical large table of the retargeting scene (five boxes; the training table is its stretch),
  source mesh points as before;
* initial pose and nominal prior = the existing Stage-1 solve of the sequence (unchanged files, sha256 recorded); no hand
  correction mode (A1 / Plan B off);
* solver profile = the Table step-1 profile (existing CORE4D two-stage Stage 2: elastic object 1e5 / foot 1e4, foot xy
  tolerance 1 mm, URDF joint limits, nominal prior lambda 5 on all 36 robot qpos, nominal foot height +-5 mm), so that one
  parameter set is used for all tasks. (The existing Push / Pull Stage 2 used the default profile with foot tolerance 0.05 m
  and the A1 wrist step.)

    hsretargeting/bin/python -B system1/t14/single_shared.py --task push --out logs_system1/t14/single/push
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

ROOT = Path(__file__).resolve().parents[2]
RT = ROOT / "src/holosoma_retargeting/holosoma_retargeting"
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OMOMO_NEW = Path("/home/kevin/holosoma/OMOMO_new")
OMOMO_P = Path("/home/kevin/holosoma/data")
SMPLX_DIR = Path("/home/kevin/CS6758Project/data/smplx_models")
STAGE1 = Path("/home/kevin/holosoma-rubber-hand-largetable/data/retargeted/rubber_hand_largetable_v1")
TASKS = dict(push=("sub6_largetable_033", STAGE1 / "sub6_largetable_033/a1/sub6_largetable_033_nominal_scaled.npz"),
             pull=("sub3_largetable_010", STAGE1 / "sub3_largetable_010/pull_a1/sub3_largetable_010_nominal_scaled.npz"))
SMPLH_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 33, 34, 35, 36]  # OMOMO_new SMPL-H names ...
SMPLX_BODY = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]  # ... and the same SMPL-X joints
PALM_KEYS = {"L_Palm": "left_rubber_hand_link", "R_Palm": "right_rubber_hand_link"}
# existing CORE4D two-stage Stage-2 profile (scripts/retarget_core4d_pair.py constants)
FOOT_XY_TOL, ELASTIC_OBJ, ELASTIC_FOOT, FOOT_H_TOL = 1e-3, 1.0e5, 1.0e4, 5e-3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def omomo_palms(seq):
    """[T, 2 (L, R), 3] palm points in the OMOMO_new world, and the fit record."""
    import joblib
    import smplx
    import torch
    from table_shared import palm_vertex_ids
    v = None
    for split in ("train", "test"):
        d = joblib.load(OMOMO_P / f"{split}_diffusion_manip_seq_joints24.p")
        c = [x for x in d.values() if x["seq_name"] == seq]
        del d
        if c:
            v, src = c[0], split
            break
    T = len(v["trans"])
    m = smplx.create(str(SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        o = m(betas=torch.tensor(np.repeat(v["betas"], T, 0), dtype=torch.float32), global_orient=torch.tensor(v["root_orient"], dtype=torch.float32),
              body_pose=torch.tensor(v["pose_body"], dtype=torch.float32), transl=torch.tensor(v["trans"], dtype=torch.float32))
    V, Jx = o.vertices.numpy().astype(np.float64), o.joints.numpy().astype(np.float64)
    J = torch.load(OMOMO_NEW / f"{seq}.pt", map_location="cpu").detach().numpy()[:, 162:162 + 52 * 3].reshape(-1, 52, 3).astype(np.float64)
    assert len(J) == T, (len(J), T)
    A, B = Jx[:, SMPLX_BODY].reshape(-1, 3), J[:, SMPLH_BODY].reshape(-1, 3)
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    res = np.linalg.norm(A @ R.T + t - B, axis=1)
    pv = palm_vertex_ids()
    palms = np.stack([V[:, pv[s]].mean(1) for s in ("left", "right")], 1) @ R.T + t
    return palms, dict(source_split=src, frames=T, rotation=R.round(8).tolist(), translation=t.round(8).tolist(), fit_residual_max_m=float(res.max()),
                       fit_residual_mean_m=float(res.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    t0 = time.time()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seq, st1 = TASKS[a.task]
    os.chdir(RT)  # the retargeting constants use paths relative to the package (models/...)
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import (build_retargeter_kwargs_from_config, convert_object_poses_to_mujoco_order,
                                                              create_grounded_nominal_object_poses, create_task_constants, load_motion_data,
                                                              setup_object_data)
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity, preprocess_motion_data
    from palm_retargeter import PalmInteractionMeshRetargeter
    from table_shared import palm_point, patch_hand_meshes, write_sim_hand_meshes
    import holosoma_retargeting
    import trimesh

    robot_config = RobotConfig(robot_type="g1")
    robot_config = replace(robot_config, nominal_tracking_indices=np.arange(7 + robot_config.ROBOT_DOF, dtype=np.int64))
    motion_config = MotionDataConfig(data_format="smplh", robot_type="g1")
    task_config = TaskConfig(object_name="largetable")
    base = RetargeterConfig(visualize=False, debug=False)
    cfg = replace(base, apply_manual_joint_limit_overrides=False, foot_sticking_tolerance=FOOT_XY_TOL,
                  elastic_constraints=replace(base.elastic_constraints, enable=True, object_collision_weight=ELASTIC_OBJ, foot_kinematics_weight=ELASTIC_FOOT))
    constants = create_task_constants(robot_config, motion_config, task_config, "object_interaction")
    human, object_poses, smpl_scale = load_motion_data("object_interaction", "smplh", OMOMO_NEW, seq, constants, motion_config)
    assert human.shape[1] == 52 and len(constants.DEMO_JOINTS) == 52
    palms, fit = omomo_palms(seq)
    human = np.concatenate([human, palms], 1)  # [T, 54, 3]: ..., L_Palm, R_Palm (raw OMOMO_new world, before preprocessing)
    constants.DEMO_JOINTS = list(constants.DEMO_JOINTS) + list(PALM_KEYS)
    constants.JOINTS_MAPPING = dict(constants.JOINTS_MAPPING) | PALM_KEYS
    sim_hands, dump_sha = write_sim_hand_meshes(out)
    # scene as the existing fixed-object path derives it (robot_retarget.py run_fixed_object_size_adaptation)
    src_scene = Path(getattr(constants, "SCENE_XML_FILE", None) or constants.ROBOT_URDF_FILE.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml"))
    src_scene = src_scene if src_scene.is_absolute() else RT / src_scene
    scene = patch_hand_meshes(src_scene, out / "scene_simhands.xml", sim_hands)
    # the patched scene lives in out/: make its mesh paths absolute
    import xml.etree.ElementTree as ET
    tree = ET.parse(scene)
    root = tree.getroot()
    comp = root.find("compiler")
    if comp is not None and comp.get("meshdir") and not Path(comp.get("meshdir")).is_absolute():
        comp.set("meshdir", str((src_scene.parent / comp.get("meshdir")).resolve()))
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f and not Path(f).is_absolute() and comp is None:
            mesh.set("file", str((src_scene.parent / f).resolve()))
    tree.write(scene)
    constants.SCENE_XML_FILE = str(scene)
    obj_pts, obj_pts_demo, urdf_path = setup_object_data("object_interaction", constants, None, smpl_scale, task_config, False)
    offsets = {k: palm_point(trimesh.load(sim_hands[s], force="mesh").vertices, s) for k, s in (("L_Palm", "left"), ("R_Palm", "right"))}
    kw = build_retargeter_kwargs_from_config(cfg, constants, urdf_path, "object_interaction")
    kw["nominal_tracking_tau"] = cfg.nominal_tracking_tau
    kw["anchor_nominal_foot_height"] = True
    kw["nominal_foot_height_tolerance"] = FOOT_H_TOL
    kw["scene_xml_path"] = str(scene)
    retargeter = PalmInteractionMeshRetargeter(**kw, palm_offsets=offsets)
    human, object_poses, _ = preprocess_motion_data(human, retargeter, motion_config.toe_names, scale=smpl_scale, object_poses=object_poses)
    z1 = np.load(st1)
    q1 = z1["qpos"].astype(np.float64)
    prep_err = float(np.abs(human[:, :52] - z1["human_joints"]).max())  # our preprocessing vs the Stage-1 input
    nominal = create_grounded_nominal_object_poses(object_poses, smpl_scale)
    nominal_mj = convert_object_poses_to_mujoco_order(nominal)
    phys_mj = convert_object_poses_to_mujoco_order(object_poses)
    n = a.max_frames or len(human)
    foot = extract_foot_sticking_sequence_velocity(human[:n], retargeter.demo_joints, motion_config.toe_names)
    foot[0][motion_config.toe_names[0]] = False
    foot[0][motion_config.toe_names[1]] = False
    q_nom = q1[:n].copy()
    q_nom[:, -7:] = phys_mj[:n]
    import mujoco
    gid = mujoco.mj_name2id(retargeter.robot_model, mujoco.mjtObj.mjOBJ_GEOM, "left_rubber_hand_link")
    rec = dict(task=a.task, sequence=seq, python=sys.executable, holosoma_retargeting_module=str(Path(holosoma_retargeting.__file__).resolve()),
               smpl_scale=smpl_scale, stage1=str(st1), stage1_sha256=sha(st1), preprocess_vs_stage1_human_joints_max_abs=prep_err,
               palm_fit=fit, robot_palm_offsets_m={k: v.round(6).tolist() for k, v in offsets.items()}, physx_dump=dump_sha, scene=str(scene),
               left_hand_mesh_vertices=int(retargeter.robot_model.mesh_vertnum[retargeter.robot_model.geom_dataid[gid]]),
               interaction_mesh_keys=list(retargeter.laplacian_match_links), object_points=int(len(obj_pts)), frames=n,
               profile=dict(foot_xy_tolerance=FOOT_XY_TOL, elastic_object=ELASTIC_OBJ, elastic_foot=ELASTIC_FOOT, foot_height_tolerance=FOOT_H_TOL,
                            nominal_indices="0..35", w_nominal=retargeter.w_nominal_tracking_init, manual_joint_limits=False))
    print(json.dumps({k: rec[k] for k in ("preprocess_vs_stage1_human_joints_max_abs", "palm_fit", "left_hand_mesh_vertices", "interaction_mesh_keys")}), flush=True)
    assert prep_err < 1e-6, prep_err  # float32 rounding of the stored Stage-1 input
    q, *_ = retargeter.retarget_motion(human_joint_motions=human[:n], object_poses=nominal_mj[:n], object_poses_augmented=phys_mj[:n],
                                       object_points_local_demo=obj_pts_demo, object_points_local=obj_pts, foot_sticking_sequences=foot,
                                       q_a_init=q_nom[0], q_nominal_list=q_nom, original=False, dest_res_path=str(out / f"{a.task}_new.npz"))
    q = np.asarray(q)
    np.savez(out / f"{a.task}_new_qpos.npz", qpos=q, object_qpos_mj=phys_mj[:n], human_joints=human[:n], fps=30, smpl_scale=smpl_scale)
    import resource
    rec.update(seconds=round(time.time() - t0, 1), finite=bool(np.isfinite(q).all()), qpos_shape=list(q.shape), object_locked_max_err=float(np.abs(q[:, -7:] - phys_mj[:n]).max()),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3))
    (out / "run_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("seconds", "finite", "qpos_shape", "object_locked_max_err", "peak_rss_gb")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
