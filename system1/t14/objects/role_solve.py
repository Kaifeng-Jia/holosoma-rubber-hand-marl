#!/usr/bin/env python3
"""T14 step 8: solve one robot of the new partner construction (partner.py) around the shared object trajectory, round 6 C
settings (single_shared.py round 6 C / single_object.py Stage 2: prior on all 36 qpos lambda 5 tau 10, palm / finger tracking
5 / 5, anchor weight 10, simulator kinematics and shapes, support foot locked at its segment start +-1 mm with the person's
toe-speed support rule, elastic object 1e5 / foot 1e4, foot height +-5 mm).

Modes
  source  (box family) role 0 alone with its own object = the oriented box x s (r = 1, nominal poses): Stage 1 (existing
          first stage, default RetargeterConfig, original=True) + Stage 2. Its robot is the source of the role check and the
          V0 comparisons; its Stage 1 is the prior of the role solves.
  role    one role r of one candidate pair at level m: the CORE4D shared-scene steps 3-4 (table_shared.py): the role's Stage-1
          solve (role 0: the source's; role 1: the source's mapped by S) shifted horizontally by the mean displacement of the
          role's hand anchors from its own object to the shared object = prior and initial pose; the robot is solved around the
          shared target (anchors in the shared object frame = the role's region map applied to the source anchors, projected
          onto the target's collision boxes); the person = the role's demonstration (role 1 = S applied in the world; an
          improper S also swaps left / right joints, hands, contact flags and palm targets); human-side object = the role's own
          object (role 1: S applied); robot-side object points = the source object points moved by the role's region map.
Families: box (step-7 objects; capture mesh nearest points as the human side) and table (Push / Pull; round 6: the training
table at its real size with phys poses, human side on the OMOMO capture table moved by omomo_table's similarity).

Step 9 switches (off = the step-8 behaviour above, unchanged):
  --frames A:B           solve only frames A..B-1 of the sequence (the screen's carry episode); everything (person, palms, object
                         poses, contact flags, palm targets) is cut before preprocessing; the record holds clip_start = A.
  --construction frontback   (role mode, box family; no --cand) the front-back partner of frontback.py at level m: shared
                         target P, role 0 = the demonstration, role 1 = the constructed partner person (upper body and palms
                         carried by G(t) = P T180 A^-1, legs reversed); role 1's own object = G O, its palm targets rotated by G,
                         its contact flags = role 0's; role 1's prior = the existing first stage (as in source mode) solved on the
                         partner person with its own object (the partner is not an isometric copy, so the source's first stage
                         cannot be mapped); role 0's prior = the source's first stage (as step 8). Region maps = frontback's.

    hsretargeting_s1/bin/python -B system1/t14/objects/role_solve.py --family box --object smallbox --sequence sub10_smallbox_007 \
        --mode source --tag smallboxb --out logs_system1/t14/step8/<...>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RT = ROOT / "src/holosoma_retargeting/holosoma_retargeting"
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

FOOT_XY_TOL, ELASTIC_OBJ, ELASTIC_FOOT, FOOT_H_TOL = 1e-3, 1.0e5, 1.0e4, 5e-3
PALM_KEYS = {"L_Palm": "left_rubber_hand_link", "R_Palm": "right_rubber_hand_link"}
STAGE1_TABLE = Path("/home/kevin/holosoma-rubber-hand-largetable/data/retargeted/rubber_hand_largetable_v1")  # read only
STAGE1_FILES = dict(push=STAGE1_TABLE / "sub6_largetable_033/a1/sub6_largetable_033_nominal_scaled.npz",
                    pull=STAGE1_TABLE / "sub3_largetable_010/pull_a1/sub3_largetable_010_nominal_scaled.npz")
LARGETABLE_SCENE = RT / "models/g1/g1_29dof_w_largetable.xml"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def swap_names(names):
    out = []
    for n in names:
        if n.startswith("L_"):
            out.append("R_" + n[2:])
        elif n.startswith("R_"):
            out.append("L_" + n[2:])
        else:
            out.append(n)
    return out


def write_box_scene(name, size, out):
    """box_assets.write_assets without the capture-mesh quality record (the target box is not a capture box)."""
    import trimesh
    import box_assets as ba
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bf = dict(extents_m=np.asarray(size, float))
    s = " ".join(f"{x:.9g}" for x in size)
    urdf = out / f"{name}.urdf"
    urdf.write_text(f'<?xml version="1.0" ?>\n<robot name="{name}"><link name="{name}_link"><inertial><mass value="0.1"/><origin xyz="0 0 0"/>'
                    f'<inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial><visual name="{name}_box"><origin rpy="0 0 0" xyz="0 0 0"/>'
                    f'<geometry><box size="{s}"/></geometry></visual><collision name="{name}_box"><origin rpy="0 0 0" xyz="0 0 0"/><geometry><box size="{s}"/>'
                    f'</geometry></collision></link></robot>\n')
    mesh = out / f"{name}.obj"
    trimesh.creation.box(extents=bf["extents_m"]).export(mesh)
    tree = ET.parse(ba.TEMPLATE)
    root = tree.getroot()
    comp = root.find("compiler")
    comp.set("meshdir", str((ba.TEMPLATE.parent / comp.get("meshdir")).resolve()))
    asset = root.find("asset")
    for m in list(asset.findall("mesh")):
        if m.get("name") == "largebox_mesh":
            asset.remove(m)
    body = next(b for b in root.find("worldbody").findall("body") if b.get("name") == "largebox_link")
    body.set("name", f"{name}_link")
    g = body.find("geom")
    g.set("name", name)
    g.set("type", "box")
    del g.attrib["mesh"]
    g.set("size", " ".join(f"{x / 2:.9g}" for x in size))
    scene = out / f"g1_29dof_w_{name}.xml"
    tree.write(scene)
    return dict(urdf=str(urdf), mesh=str(mesh), scene=str(scene), size=list(map(float, size)))


def write_table_scene(urdf, out):
    """the largetable retargeting scene with its object boxes replaced by the given box-union URDF's boxes (training table
    of the level); the visual mesh geom of the 0.52 m table is removed."""
    from reference_geometry_math import BoxUnion
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = BoxUnion(urdf)
    tree = ET.parse(LARGETABLE_SCENE)
    root = tree.getroot()
    comp = root.find("compiler")
    comp.set("meshdir", str((LARGETABLE_SCENE.parent / comp.get("meshdir")).resolve()))
    for m in root.iter("mesh"):
        f = m.get("file")
        if f and f.startswith("../"):
            m.set("file", str((LARGETABLE_SCENE.parent / comp.get("meshdir") / f).resolve()))
    body = next(x for x in root.find("worldbody").findall("body") if x.get("name") == "largetable_link")
    proto = None
    for g in list(body.findall("geom")):
        if g.get("type") == "box" and proto is None:
            proto = dict(g.attrib)
        body.remove(g)
    for p in b.parts:
        att = dict(proto, name=f"largetable_{p['id'].split('_', 1)[1]}", size=" ".join(f"{x / 2:.9g}" for x in p["size"]),
                   pos=" ".join(f"{x:.9g}" for x in p["center"]))
        ET.SubElement(body, "geom", att)
    scene = out / "g1_29dof_w_largetable.xml"
    tree.write(scene)
    return dict(urdf=str(urdf), scene=str(scene))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("box", "table"), required=True)
    ap.add_argument("--object", default=None, help="box family: OMOMO object name")
    ap.add_argument("--task", default=None, choices=("push", "pull"), help="table family")
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--mode", choices=("source", "role"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source", type=Path, default=None, help="role mode, box family: the source solve dir")
    ap.add_argument("--cand", type=Path, default=None, help="role mode: json with one candidate pair (partner.candidate_pairs row)")
    ap.add_argument("--role", type=int, choices=(0, 1), default=0)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--target-urdf", type=Path, default=None, help="table family: the level's training-table URDF")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frames", default=None, help="step 9: A:B, solve frames A..B-1 only")
    ap.add_argument("--construction", choices=("partner", "frontback"), default="partner", help="step 9: frontback (role mode, box family)")
    ap.add_argument("--fb-turn", choices=("box", "world"), default="box", help="step 9c: frontback partner turn about the world vertical")
    ap.add_argument("--fb-legs", choices=("all", "walk"), default="all", help="step 9c: reverse only the walking segment")
    ap.add_argument("--fb-human-frame", choices=("own", "target"), default="own",
                    help="frontback, Stage 2: 'target' = the person-side object frame of the interaction mesh is the role's own object "
                         "frame carried by its region map (own_r M_r^-1, points M_r demo), so that it differs from the robot-side "
                         "(shared target) frame by a translation and the small one-sided rotation only; 'own' = as steps 8 / 9b / 9c run 1")
    ap.add_argument("--human-frame", choices=("own", "target"), default="own",
                    help="step 8 rerun: the same fix for every role-mode construction (partner and frontback): person-side frame = "
                         "the person-side object pose x (M_r F)^-1, points (M_r F) applied to the person-side points (F = the table mirror "
                         "of an improper role 1, else identity); world points unchanged. 'own' = the earlier behaviour")
    a = ap.parse_args()
    if a.construction == "frontback" and not (a.mode == "role" and a.family == "box"):
        raise SystemExit("--construction frontback is for role mode, box family")
    t0 = time.time()
    for k in ("source", "cand", "target_urdf"):  # the solver changes directory below
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import partner as pt
    import box_assets as ba
    import omomo_motion as om
    from single_shared import OMOMO_NEW, omomo_palms, scene_object_boxes
    from single_object import human_hand_sd
    if a.family == "box":
        name, seq = a.object, a.sequence
        v = om.load_sequences({name})[seq]
    else:
        name, seq = "largetable", pt.TABLES[a.task]["seq"]
    palms, fit = omomo_palms(seq)
    os.chdir(RT)
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import (_compute_q_init_base, build_retargeter_kwargs_from_config, convert_object_poses_to_mujoco_order,
                                                              create_grounded_nominal_object_poses, create_task_constants, load_motion_data, setup_object_data)
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity, preprocess_motion_data, build_pt_wrist_palm_orientation_targets, load_intermimic_wrist_quaternions
    from palm_retargeter import PalmInteractionMeshRetargeter
    from table_shared import palm_point, patch_hand_meshes, write_sim_hand_meshes, contact_windows, smoothstep_window
    from sim_scene import simify_scene
    from object_surface import UnionSurface
    from reference_geometry_math import map_robot
    from reference_geometry_pilot import G1_29DOF_JOINT_NAMES, g1_sagittal_joint_map
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
    human, intermimic_obj, smpl_scale = load_motion_data("object_interaction", "smplh", OMOMO_NEW, seq, constants, motion_config)
    s = float(smpl_scale)
    # ---------------------------------------------------------------- source object (raw OMOMO_new world) and its geometry
    if a.family == "box":
        bf = ba.box_frame(name, v)
        raw_poses, invalid = ba.poses_omomo_new(v, bf, fit)
        h_sd, _ = human_hand_sd(v, name)
        src_size = bf["extents_m"] * s  # r = 1: the person's object x s
        src_assets = write_box_scene(name, src_size, out / "assets_source")
        mesh_box = ba.mesh_in_box_frame(name, bf)
        mesh_box.vertices = np.asarray(mesh_box.vertices) * s
        constants.OBJECT_URDF_FILE, constants.OBJECT_MESH_FILE = src_assets["urdf"], src_assets["mesh"]
    else:
        import dual_stretch as ds
        import omomo_table as ot
        raw_poses, invalid = intermimic_obj.copy(), 0
        h_sd, _ = ds.human_hand_sd(seq)
        constants.OBJECT_URDF_FILE = str(RT / "models/largetable/largetable.urdf")
        constants.OBJECT_MESH_FILE = str(RT / "models/largetable/largetable.obj")
    assert len(human) == len(raw_poses)
    human_raw52, names52 = human.copy(), list(constants.DEMO_JOINTS)
    wq = load_intermimic_wrist_quaternions(OMOMO_NEW / f"{seq}.pt")
    palm_rot, cal_err = build_pt_wrist_palm_orientation_targets(human_raw52, wq, names52)
    clip_start = 0
    if a.frames is not None:  # step 9: the clip only (cut before preprocessing, as step9_diag.load_role0)
        fa, fb_ = (int(x) for x in a.frames.split(":"))
        clip_start = fa
        human, raw_poses, palms, h_sd, palm_rot = human[fa:fb_], raw_poses[fa:fb_], palms[fa:fb_], h_sd[fa:fb_], palm_rot[fa:fb_]
        human_raw52 = human_raw52[fa:fb_]
        assert len(human) == fb_ - fa
    T = len(human)
    n = a.max_frames or T
    # ---------------------------------------------------------------- level geometry (target) and scene
    cand = None
    FB = None
    if a.mode == "role" and a.construction == "frontback":
        import frontback as fbm
        from types import SimpleNamespace as _NS
        names54 = list(constants.DEMO_JOINTS) + list(PALM_KEYS)
        h0_, p0_, _ = preprocess_motion_data(np.concatenate([human, palms], 1), _NS(demo_joints=names54), motion_config.toe_names, scale=s, object_poses=raw_poses)
        O_ = pt.mj_to_mat(convert_object_poses_to_mujoco_order(create_grounded_nominal_object_poses(p0_, s)))
        fa_, fb_e = (int(x) for x in a.frames.split(":")) if a.frames else (0, len(human))
        walk = fbm.walking_segment(v, fa_, fb_e) if a.fb_legs == "walk" else None
        if a.fb_legs == "walk" and walk is None:
            raise SystemExit("no swing segment in the clip")
        FB = fbm.build(h0_, names54, O_, src_size, a.m, turn=a.fb_turn, legs=a.fb_legs, walk=walk)
        cand = FB["cand"]
        axis, L1 = int(cand["axis"]), float(cand["L1"])
        Lm = a.m * L1
        assert abs(Lm - FB["Lm"]) < 1e-9
        tsize = np.array(src_size, float)
        tsize[axis] = Lm
        tgt = write_box_scene(name, tsize, out / "assets_target")
    elif a.mode == "role":
        cand = json.loads(a.cand.read_text())
        axis, L1 = int(cand["axis"]), float(cand["L1"])
        Lm = a.m * L1
        if a.family == "box":
            tsize = np.array(src_size, float)
            tsize[axis] = Lm
            tgt = write_box_scene(name, tsize, out / "assets_target")
        else:
            import dual_stretch as ds
            tgt = write_table_scene(a.target_urdf, out / "assets_target")
    else:
        if a.family != "box":
            raise SystemExit("source mode is for the box family (tables use the round-6 single solve)")
        axis, L1, Lm = 0, float(src_size[0]), float(src_size[0])
        cand = dict(Q0=np.eye(3).tolist(), c0=[0, 0, 0], Q1=np.eye(3).tolist(), c1=[0, 0, 0], axis=0, L1=L1, parity0=1, parity1=1, id="identity (source)")
        tgt = src_assets
    # ---------------------------------------------------------------- Stage-2 retargeter (scene of the target)
    human_p = np.concatenate([human, palms], 1)
    constants.DEMO_JOINTS = list(constants.DEMO_JOINTS) + list(PALM_KEYS)
    constants.JOINTS_MAPPING = dict(constants.JOINTS_MAPPING) | PALM_KEYS
    sim_hands, dump_sha = write_sim_hand_meshes(out)
    scene = patch_hand_meshes(Path(tgt["scene"]), out / "scene_simhands.xml", sim_hands)
    scene, sim_rec = simify_scene(scene, out / "scene_sim.xml", out / "sim_meshes", f"{name}_link")
    constants.SCENE_XML_FILE = str(scene)
    offsets = {k: palm_point(trimesh.load(sim_hands[sd], force="mesh").vertices, sd) for k, sd in (("L_Palm", "left"), ("R_Palm", "right"))}
    cfg = replace(base, apply_manual_joint_limit_overrides=False, foot_sticking_tolerance=FOOT_XY_TOL,
                  elastic_constraints=replace(base.elastic_constraints, enable=True, object_collision_weight=ELASTIC_OBJ, foot_kinematics_weight=ELASTIC_FOOT))
    obj_urdf_for_kw = tgt["urdf"]
    kw = build_retargeter_kwargs_from_config(cfg, constants, obj_urdf_for_kw, "object_interaction")
    kw["nominal_tracking_tau"] = 10.0
    kw["anchor_nominal_foot_height"] = True
    kw["nominal_foot_height_tolerance"] = FOOT_H_TOL
    kw["scene_xml_path"] = str(scene)
    # placeholder retargeter for preprocessing (demo joints / toe names); the solving one is built after the anchors
    from types import SimpleNamespace
    human0, poses0, _ = preprocess_motion_data(human_p, SimpleNamespace(demo_joints=constants.DEMO_JOINTS), motion_config.toe_names, scale=s, object_poses=raw_poses)
    nominal0 = create_grounded_nominal_object_poses(poses0, s)
    own_nom_mj = convert_object_poses_to_mujoco_order(nominal0)
    own_phys_mj = convert_object_poses_to_mujoco_order(poses0)
    own_robot_mj = own_nom_mj if a.family == "box" else own_phys_mj  # r = 1 boxes: nominal; tables: the real-size training table (round 6)
    O = pt.mj_to_mat(own_robot_mj)
    if FB is None:
        C = pt.construction(O, cand, axis, L1, Lm)
    else:  # front-back: the construction was built above on the same preprocessed data
        assert np.allclose(pt.mj_to_mat(own_nom_mj), O_) and np.allclose(human0, h0_)
        Q0_, c0_, Q1_, c1_ = pt.region_maps(cand, axis, L1, Lm)
        assert np.allclose(pt.affine(Q0_, c0_), FB["M0"]) and np.allclose(pt.affine(Q1_, c1_), FB["M1"])
        C = dict(S=FB["S"], P=FB["P"], M0=FB["M0"], M1=FB["M1"])
    S, P = C["S"], C["P"]
    improper = np.linalg.det(S[:3, :3]) < 0
    shared_mj = pt.mat_to_mj(P)
    # ---------------------------------------------------------------- the role's person, own object, flags, palm targets
    hidx = [0, 1] if (a.role == 0 or not improper) else [1, 0]
    F = np.diag([1.0, 1.0, -1.0]) if (improper and a.role == 1) else np.eye(3)  # role-1 own-object frame S O F is proper (table local XY mirror)
    names_role = list(constants.DEMO_JOINTS)
    if a.role == 0:
        human_r, nom_r = human0, own_nom_mj
        pw, fw = palm_rot[..., :, 2], palm_rot[..., :, 0]
        own_r = O
    elif FB is not None:  # front-back partner: constructed person, own object G O, palm targets rotated by G
        G = FB["G"]
        human_r = FB["human1"]
        own_r = G @ O
        nom_r = pt.mat_to_mj(G @ pt.mj_to_mat(own_nom_mj))
        pw = np.einsum("tij,thj->thi", G[:, :3, :3], palm_rot[..., :, 2])
        fw = np.einsum("tij,thj->thi", G[:, :3, :3], palm_rot[..., :, 0])
    else:
        Rs, ts = S[:3, :3], S[:3, 3]
        human_r = human0 @ Rs.T + ts
        if improper:
            perm = [names_role.index(x) for x in swap_names(names_role)]
            human_r = human_r[:, perm]
        N = pt.mj_to_mat(own_nom_mj)
        Fm = pt.affine(F, np.zeros(3))
        nom_r = pt.mat_to_mj(S @ N @ Fm)
        own_r = S @ O @ Fm
        pw = (palm_rot[..., :, 2] @ Rs.T)[:, hidx]
        fw = (palm_rot[..., :, 0] @ Rs.T)[:, hidx]
    h_sd_r = h_sd[:, hidx]
    # ---------------------------------------------------------------- interaction-mesh object points
    obj_pts, _demo_unused, _ = setup_object_data("object_interaction", constants, None, s, task_config, False)
    Mr = C["M0"] if a.role == 0 else C["M1"]
    robot_pts = obj_pts @ Mr[:3, :3].T + Mr[:3, 3]  # source object points moved into the target frame by the role's region map
    if a.family == "box":
        demo = trimesh.proximity.closest_point(mesh_box, obj_pts)[0]  # already x s (r = 1)
    else:
        import omomo_table as ot
        demo = ot.from_training(seq, obj_pts) * s
    demo_r = demo @ F.T
    # ---------------------------------------------------------------- anchors (source frame medians -> region map -> target boxes)
    from scipy.spatial.transform import Rotation
    Rraw = Rotation.from_quat(raw_poses[:, [1, 2, 3, 0]]).as_matrix()
    local_raw = np.einsum("tji,thj->thi", Rraw, palms - raw_poses[:, None, 4:7])
    unit = s if a.family == "box" else 1.0
    surf = UnionSurface(scene_object_boxes(scene, f"{name}_link"))
    anchors, act, normals = np.zeros((T, 2, 3)), np.zeros((T, 2)), np.zeros((T, 2, 3))
    disp = []
    arec = dict(weight=10.0, ramp_frames=5, surface="scene collision boxes of the shared target", windows=[])
    for h in (0, 1):  # robot / role hand
        hs = hidx[h]  # the source person's hand it corresponds to
        for s0, s1 in contact_windows(h_sd_r[:, h] >= -0.015):
            med = np.median(local_raw[s0:s1, hs], axis=0)
            if a.family == "table":
                med = ot.to_training(seq, med)
            a_src = med * unit
            a_tgt = Mr[:3, :3] @ a_src + Mr[:3, 3]
            anc, dist, nrm = surf.nearest(a_tgt[None], normals=True)
            anchors[s0:s1, h], normals[s0:s1, h] = anc[0], nrm[0]
            act[s0:s1, h] = smoothstep_window(s1 - s0)
            w_own = np.einsum("tij,j->ti", own_r[s0:s1, :3, :3], F @ a_src) + own_r[s0:s1, :3, 3]
            w_sh = np.einsum("tij,j->ti", P[s0:s1, :3, :3], a_tgt) + P[s0:s1, :3, 3]  # the same anchor point carried by the shared object (table_shared rule)
            disp.append(w_sh - w_own)
            arec["windows"].append(dict(hand=["left", "right"][h], source_hand=["left", "right"][hs], start=int(s0), stop=int(s1), median_source_local=np.round(a_src, 5).tolist(),
                                        target_local=np.round(a_tgt, 5).tolist(), anchor_local=np.round(anc[0], 5).tolist(), projection_m=round(float(dist[0]), 5)))
    shift = np.zeros(3)
    if disp:
        shift[:2] = np.concatenate(disp)[:, :2].mean(0)
    # ---------------------------------------------------------------- prior: the role's Stage-1 solve, shifted
    if a.mode == "source" or (FB is not None and a.role == 1):  # step 9: the partner's own first stage on the partner person
        constants52 = SimpleNamespace(**vars(constants))
        constants52.DEMO_JOINTS = names52
        constants52.JOINTS_MAPPING = {k_: v_ for k_, v_ in constants.JOINTS_MAPPING.items() if k_ not in PALM_KEYS}
        constants52.SCENE_XML_FILE = src_assets["scene"]
        k1 = build_retargeter_kwargs_from_config(base, constants52, src_assets["urdf"], "object_interaction")
        k1["nominal_tracking_tau"] = base.nominal_tracking_tau
        k1["scene_xml_path"] = src_assets["scene"]  # r = 1: the scene's box is already x s (Stage 1 = the nominal-scaled scene)
        ret1 = InteractionMeshRetargeter(**k1)
        h52 = (human0 if a.mode == "source" else human_r)[:, :52]
        s1_nom_mj = own_nom_mj if a.mode == "source" else nom_r  # MuJoCo order; nominal0 order = (qw qx qy qz x y z)
        foot1 = extract_foot_sticking_sequence_velocity(h52[:n], ret1.demo_joints, motion_config.toe_names)
        foot1[0][motion_config.toe_names[0]] = False
        foot1[0][motion_config.toe_names[1]] = False
        q_init1 = _compute_q_init_base("object_interaction", "smplh", h52[:n], s1_nom_mj[:n][:, [3, 4, 5, 6, 0, 1, 2]], constants52)
        t1 = time.time()
        q1, *_ = ret1.retarget_motion(human_joint_motions=h52[:n], object_poses=s1_nom_mj[:n], object_poses_augmented=s1_nom_mj[:n],
                                      object_points_local_demo=demo, object_points_local=demo, foot_sticking_sequences=foot1,
                                      q_a_init=q_init1, q_nominal_list=None, original=True, dest_res_path=str(out / f"{a.tag}_nominal_scaled.npz"))
        stage1 = np.asarray(q1, dtype=np.float64)
        stage1_rec = dict(file=str(out / f"{a.tag}_nominal_scaled.npz"), seconds=round(time.time() - t1, 1), config="RetargeterConfig() defaults, original=True, box x s scene")
        del ret1
    else:
        sp = a.source / f"{json.loads((a.source / 'run_record.json').read_text())['tag']}_nominal_scaled.npz" if a.family == "box" else STAGE1_FILES[a.task]
        stage1 = np.load(sp)["qpos"].astype(np.float64)
        stage1_rec = dict(file=str(sp), sha256=sha(sp))
    prior = stage1[:n, :36].copy()
    if a.role == 1 and FB is None:
        index, sign = g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES)
        ident = np.tile(np.array([0, 0, 0, 1.0, 0, 0, 0]), (n, 1))
        prior = map_robot(prior, ident, S[:3, :3], S[:3, 3], -1 if improper else 1, index, sign)
    prior[:, :2] += shift[:2]
    q_nom = np.concatenate([prior, shared_mj[:n]], 1)
    # ---------------------------------------------------------------- solve
    at = dict(keys=("L_Palm", "R_Palm"), weight=10.0, anchors=anchors, activation=act, normals=normals)
    orient = dict(weight=5.0, activation=np.ones((T, 2)), palm_world=pw, finger_world=fw, palm_normal_local=hl.PALM_NORMAL_LOCAL, finger_weight=None)
    ret = PalmInteractionMeshRetargeter(**kw, palm_offsets=offsets, anchor_term=at, orient_term=orient)
    assert list(ret.track_nominal_indices) == list(range(36))
    toes = human_r[:n][:, [names_role.index(x) for x in motion_config.toe_names]]
    sup = support.support_flags(toes)
    ret.enable_stance_lock(sup)
    t2 = time.time()
    nom2, demo2 = nom_r, demo_r
    human_frame = "target" if (a.human_frame == "target" or (FB is not None and a.fb_human_frame == "target")) else "own"
    if a.mode == "role" and human_frame == "target":  # the same world points, local coordinates in the role's copy of the target
        Meff = Mr @ pt.affine(F, np.zeros(3))
        assert np.linalg.det(Meff[:3, :3]) > 0
        Nr = pt.mj_to_mat(nom_r)
        nom2 = pt.mat_to_mj(Nr @ np.linalg.inv(Meff))
        demo2 = demo_r @ Meff[:3, :3].T + Meff[:3, 3]
        assert np.allclose(np.einsum("tij,pj->tpi", pt.mj_to_mat(nom2)[:, :3, :3], demo2) + pt.mj_to_mat(nom2)[:, None, :3, 3],
                           np.einsum("tij,pj->tpi", Nr[:, :3, :3], demo_r) + Nr[:, None, :3, 3], atol=1e-7)
    q, *_ = ret.retarget_motion(human_joint_motions=human_r[:n], object_poses=nom2[:n], object_poses_augmented=shared_mj[:n], object_points_local_demo=demo2,
                                object_points_local=robot_pts, foot_sticking_sequences=support.as_solver_sequence(sup), q_a_init=q_nom[0], q_nominal_list=q_nom,
                                original=False, dest_res_path=str(out / f"{a.tag}_new.npz"))
    q = np.asarray(q)
    # object speed of the shared trajectory on the person's real scale (CoCarry moving condition)
    spd = np.zeros(n)
    spd[1:] = np.linalg.norm(np.diff(P[:n, :3, 3], axis=0), axis=1) * 30.0 / s
    np.savez(out / f"{a.tag}_new_qpos.npz", qpos=q, object_qpos_mj=shared_mj[:n], human_joints=human_r[:n], fps=30, smpl_scale=s, own_object_mj=nom_r[:n],
             S=S, P=P[:n])
    np.savez(out / "human_contact.npz", h_sd=h_sd_r[:n], object_speed=spd)
    import resource
    rec = dict(family=a.family, object=name, task=a.task, sequence=seq, tag=a.tag, mode=a.mode, role=a.role, m=a.m, candidate=cand, axis=axis, L1=L1, Lm=Lm,
               improper=bool(improper), S=np.round(S, 9).tolist(), shift_xy_m=np.round(shift[:2], 5).tolist(), smpl_scale=s, r="1 (box x s)" if a.family == "box" else "training table (round 6)",
               target=tgt, stage1=stage1_rec, anchor_term=arec, palm_fit=fit, palm_track=dict(weight=5.0, finger_weight=5.0, calibration_p90_deg=np.round(cal_err, 6).tolist()),
               foot=dict(rule="person toe speed < 0.01 m/s after Gaussian sigma 8; support foot locked at segment start +-1 mm", support_share=np.round(sup.mean(0), 4).tolist()),
               sim_scene=sim_rec, physx_dump=dump_sha, scene=str(scene), robot_palm_offsets_m={k: np.round(x, 6).tolist() for k, x in offsets.items()},
               invalid_object_pose_frames=invalid, frames=n, clip_start=clip_start, clip_frames=a.frames, construction=a.construction,
               human_frame=human_frame if a.mode == "role" else None,
               frontback=None if FB is None else dict(human_frame=a.fb_human_frame, turn=FB["turn"], legs=FB["legs"], walk=FB["walk"], walk_axis_angle_deg=FB["walk_axis_angle_deg"], sigma=FB["sigma"],
                                                     leg_offset_m=FB["leg_offset"].tolist(), gravity_deg_S=pt.gravity_deg(FB["S"]),
                                                     M0=FB["M0"].tolist(), M1=FB["M1"].tolist(), pelvis_mismatch=fbm.diag(FB["D"], s)),
               python=sys.executable, holosoma_retargeting_module=str(Path(holosoma_retargeting.__file__).resolve()),
               seconds=round(time.time() - t0, 1), stage2_seconds=round(time.time() - t2, 1), finite=bool(np.isfinite(q).all()), qpos_shape=list(q.shape),
               object_locked_max_err=float(np.abs(q[:, -7:] - shared_mj[:n]).max()), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3),
               command=" ".join(sys.argv))
    (out / "run_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("seconds", "finite", "qpos_shape", "object_locked_max_err", "peak_rss_gb", "shift_xy_m")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
