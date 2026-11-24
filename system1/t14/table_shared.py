#!/usr/bin/env python3
"""T14 step 1 (+ step 3 at k = 1) for Table (CORE4D 20231030/001 desk001): re-solve one robot around the shared object.

Variants (--variant):
  new        T14 step 1 + step 3 (k = 1):
             * interaction mesh = the existing 15 SMPL-X body joints + 2 palm points (L_Palm / R_Palm) + object points;
               human palm point = mean of 67 palm-side SMPL-X vertices per hand (template rule in palm_vertex_ids());
               robot palm point = _derive_palm_contact_point's construction on the simulator's hand hull;
             * hand non-penetration with the simulator's rubber-hand collision shape (the PhysX-cooked convex hull of
               the T10 dump, also what the T10 MuJoCo scene uses) instead of the retargeting OBJ;
             * nominal prior and initial pose = the person's existing Stage-1 solve shifted horizontally by the mean
               displacement of this person's hand anchors from the person-scaled object to the shared object;
             * no new cost term.
  omni_asis  comparison "OmniRetarget as is": the same solve around the shared object without the palm points, with the
             retargeting hand mesh, and with the unshifted Stage-1 solve as prior and initial pose.
Both: shared object = desk001 five-box proxy x s_shared, s_shared = s_bar * k (s_bar = mean of the two person scales,
0.7412283856; --k = T14 step 3 uniform scale, 1.0 = the current training table), about the existing ground anchor, pose
translation anchor + s_shared (x - anchor); the person's own scaled
human joints and person-scaled object points form the source mesh (Stage 2's nominal / physical split); all solver
parameters are the existing CORE4D two-stage Stage-2 profile (scripts/retarget_core4d_pair.py: elastic object 1e5 /
foot 1e4, foot xy tolerance 1 mm, URDF joint limits, nominal prior lambda 5 on all 36 robot qpos, nominal foot height
+-5 mm, 100 requested object points seed 42, scene mass 1 kg); Stage 1 is the existing person{N}_nominal_scaled.npz.

Hand anchors (step 2 definition, used here only for the step-3 shift): per hand, the frames in which the person's hand
touches the raw desk001 mesh in the source (any hand vertex signed distance >= -1.5 cm, the T12 / stop-1 rule); anchor =
per-coordinate median over those frames of the palm point in the object frame, projected to the nearest raw-mesh surface
point. Displacement of an anchor a at a frame = (s_shared - s_i) * ((x(t) - anchor_ground) + R(t) a); the shift is the mean
over both hands' anchors and their contact frames (30 Hz), horizontal part only.

    hsretargeting/bin/python -B system1/t14/table_shared.py --variant new --person 1 --out logs_system1/t14/table/new
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CORE4D = Path("/home/kevin/datasets/CORE4D-V1-full")
PREVIEWS = Path("/home/kevin/holosoma-core4d-base/logs/Core4DPreviews")  # read only
WBT = Path(__file__).resolve().parents[2] / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
# --task: the CORE4D two-person sequences of the paper tasks. base = the 1.0x object scale: "mean" = mean of the two person
# scales (Table, Chair: the current training objects), "one" = the original size (Bin, T14 section 7 item 2).
TASKS = dict(
    table=dict(raw="20231030/001", canon=CORE4D / "derived/20231030_001_desk001_v1_canonical.npz",
               stage1=CORE4D / "retarget_results/method_comparison_20260904/20231030_001_desk001_move/two_stage_fullnominal",
               obj_key="core4d_desk001", current=WBT / "core4d_smalltable/20231030_001_desk001_move_compact_fps30.npz", base="mean"),
    bin=dict(raw="20231020/071", canon=PREVIEWS / "bucket003_20231020_071_20260918/source_canonical.npz",
             stage1=PREVIEWS / "bucket003_20231020_071_20260918/two_stage", obj_key="core4d_bucket003",
             current=WBT / "core4d_bucket003_20231020_071_a1/core4d_pair_compact_fps30.npz", base="one"),
    chair=dict(raw="20231020/074", canon=PREVIEWS / "chair021_20231020_074_baseline_20260908/source_canonical.npz",
               stage1=PREVIEWS / "chair021_20231020_074_baseline_20260908/two_stage", obj_key="core4d_chair021",
               current=WBT / "core4d_chair021_20231020_074_a1/core4d_pair_compact_fps30.npz", base="mean"))
SEQ_RAW = CORE4D / "CORE4D_Real/human_object_motions" / TASKS["table"]["raw"]
CANON = TASKS["table"]["canon"]
STAGE1 = TASKS["table"]["stage1"]
OBJ_KEY = TASKS["table"]["obj_key"]
SMPLX_NEUTRAL = Path("/home/kevin/CS6758Project/data/smplx_models/smplx/SMPLX_NEUTRAL.npz")
OUTPUT_FPS = 30
CONTACT = -0.015
PALM_KEYS = {"L_Palm": "left_rubber_hand_link", "R_Palm": "right_rubber_hand_link"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_pair_script():
    spec = importlib.util.spec_from_file_location("retarget_core4d_pair", ROOT / "scripts/retarget_core4d_pair.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------- human palm points
def palm_vertex_ids():
    """Per hand: SMPL-X template vertices whose largest skinning weight is the wrist, whose template normal points to the
    palm side (dot > 0.5 with the palm-plane normal, palm side = down in the T-pose template), and that lie between 25 %
    and 85 % of the wrist -> middle-knuckle distance along that axis. 67 vertices per hand."""
    z = np.load(SMPLX_NEUTRAL)
    V, F = z["v_template"], z["f"]
    J = z["J_regressor"] @ V
    lab = z["weights"].argmax(1)
    N = trimesh.Trimesh(V, F, process=False).vertex_normals
    out = {}
    for side, w, i1, m1, p1 in (("left", 20, 25, 28, 31), ("right", 21, 40, 43, 46)):
        ax = J[m1] - J[w]
        L = np.linalg.norm(ax)
        ax /= L
        pn = np.cross(J[i1] - J[w], J[p1] - J[w])
        pn /= np.linalg.norm(pn)
        if pn[1] > 0:
            pn = -pn
        hv = np.flatnonzero(lab == w)
        s = (V[hv] - J[w]) @ ax
        out[side] = hv[(N[hv] @ pn > 0.5) & (s > 0.25 * L) & (s < 0.85 * L)]
    return out


def source_palms_and_contacts(core4d_to_omni):
    """Raw CORE4D (15 Hz, Y-up): palm points of both persons in the canonical Z-up world [T, 2, 2(L,R), 3], and per
    person / hand the contact flags and the palm point in the raw object frame (for the anchors)."""
    import contact_parts as cp  # stop-1 module: raw-mesh signed distance from dense surface samples
    pv = palm_vertex_ids()
    hv = cp.hand_vertex_ids()
    To = np.load(SEQ_RAW / "smooth_objposes.npy")
    palms = []
    contact, palm_local = {}, {}
    for pi in (1, 2):
        P = np.load(SEQ_RAW / f"person{pi}_poses.npz", allow_pickle=True)["arr_0"].item()
        V, Jraw = P["vertices"].astype(np.float64), P["joints"].astype(np.float64)
        pp = []
        for side in ("left", "right"):
            c = V[:, pv[side]].mean(1)
            pp.append(c)
            flags, loc = [], []
            for t in range(len(V)):
                R, x = To[t, :3, :3], To[t, :3, 3]
                r = cp.hand_frame([((V[t, hv[side]] - x) @ R, OBJ_KEY)], None, None)
                flags.append(r is not None)
                loc.append((c[t] - x) @ R)
            contact[(pi, side)] = np.array(flags)
            palm_local[(pi, side)] = np.array(loc)
        palms.append(np.stack(pp, 1))
        canon_check = np.einsum("ij,tkj->tki", core4d_to_omni, Jraw)
        palms[-1] = (palms[-1], canon_check)
    return palms, contact, palm_local


def resample_linear(x, fps_in, fps_out):
    """As resample_core4d_pair_sequence for joint positions: linear in time, first and last samples kept."""
    n = len(x)
    duration = (n - 1) / fps_in
    m = int(round(duration * fps_out)) + 1
    t_in = np.arange(n) / fps_in
    t_out = np.linspace(0.0, duration, m)
    flat = x.reshape(n, -1)
    return np.stack([np.interp(t_out, t_in, flat[:, k]) for k in range(flat.shape[1])], 1).reshape((m,) + x.shape[1:])


# --------------------------------------------------------------------------------------------- simulator hand hulls
def write_sim_hand_meshes(out_dir):
    sys.path.insert(0, str(ROOT / "system1/t12v2"))
    import contact_stats as cs
    shapes = dict(cs.dump_shapes("table", "/World/envs/env_0/Robot/")[0])
    paths = {}
    for side in ("left", "right"):
        m = shapes[f"{side}_rubber_hand_link"]
        p = out_dir / f"{side}_rubber_hand_physx_hull.obj"
        m.export(p)
        paths[side] = p
    return paths, dict(scene_sha256=cs.sha(cs.DUMP / "d_table/dump/scene.json"), convexes_sha256=cs.sha(cs.DUMP / "d_table/dump/collision_convexes.npz"))


def patch_hand_meshes(scene_xml, out_xml, mesh_paths):
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    done = 0
    for mesh in root.iter("mesh"):
        for side, p in mesh_paths.items():
            if mesh.get("name") == f"{side}_rubber_hand_link":
                mesh.set("file", str(Path(p).resolve()))
                mesh.attrib.pop("content_type", None)
                done += 1
    if done != 2:
        raise RuntimeError(f"expected 2 rubber-hand mesh assets, patched {done}")
    tree.write(out_xml)
    return out_xml


def palm_point(vertices, side):
    n = dict(left=np.array([-0.0751, -0.9954, -0.0594]), right=np.array([-0.0751, 0.9954, -0.0593]))[side]
    n = n / np.linalg.norm(n)
    v = np.asarray(vertices, float)
    c = 0.5 * (v.min(0) + v.max(0))
    return c + (float(np.max(v @ n)) - c @ n) * n


# --------------------------------------------------------------------------------------------- step 2
def smoothstep_window(L, ramp=5):
    """activation inside one contact window of L frames: smoothstep ramps over the first and last `ramp` frames, 1 between."""
    j = np.arange(L)
    x = np.minimum(1.0, np.minimum((j + 1) / ramp, (L - j) / ramp))
    return x * x * (3.0 - 2.0 * x)


def contact_windows(flags):
    """[(start, stop)] of the runs of True (stop exclusive)."""
    f = np.concatenate([[False], np.asarray(flags, bool), [False]])
    e = np.flatnonzero(np.diff(f.astype(int)))
    return list(zip(e[0::2], e[1::2]))


def step2_anchor_term(a, contact, palm_local, fps_src, T, s_shared):
    """T14 section 1 step 2 (CoCarry form), per hand of this person: contact = the source rule (any hand vertex within 1.5 cm of
    the raw mesh), 15 Hz flags resampled to 30 Hz (> 0.5); per contact window, anchor = per-coordinate median of the palm point
    in the object frame over the window, scaled with the object (x s_shared), projected to the nearest point of the
    simulator's collision surface of this level (sim_scene.object_pieces x k); activation = 1 inside the window with 5-frame
    smoothstep ramps at both ends, 0 outside; weight = the interaction-mesh weight (10)."""
    from object_surface import UnionSurface
    from sim_scene import object_pieces
    surf = UnionSurface(object_pieces(a.task, a.k))
    anchors, act, normals = np.zeros((T, 2, 3)), np.zeros((T, 2)), np.zeros((T, 2, 3))
    rec = dict(weight=a.anchor_weight, ramp_frames=5, surface="simulator collision shape x k", windows=[])
    for hi, side in enumerate(("left", "right")):
        c30 = resample_linear(contact[(a.person, side)].astype(float)[:, None], fps_src, OUTPUT_FPS)[:, 0] > 0.5
        loc30 = resample_linear(palm_local[(a.person, side)], fps_src, OUTPUT_FPS)
        assert len(c30) == T, (len(c30), T)
        for s0, s1 in contact_windows(c30):
            med = np.median(loc30[s0:s1], axis=0) * s_shared
            anc, dist, nrm = surf.nearest(med[None], normals=True)
            anchors[s0:s1, hi] = anc[0]
            normals[s0:s1, hi] = nrm[0]
            act[s0:s1, hi] = smoothstep_window(s1 - s0)
            rec["windows"].append(dict(hand=side, start=int(s0), stop=int(s1), median_local_m=med.round(5).tolist(), anchor_local_m=anc[0].round(5).tolist(),
                                       projection_m=round(float(dist[0]), 5), outward_normal=nrm[0].round(4).tolist()))
    return dict(keys=("L_Palm", "R_Palm"), weight=a.anchor_weight, anchors=anchors, activation=act, normals=normals), rec


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("new", "omni_asis"), required=True)
    ap.add_argument("--task", choices=tuple(TASKS), default="table")
    ap.add_argument("--person", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke only")
    ap.add_argument("--k", type=float, default=1.0, help="T14 step 3 uniform object scale relative to the 1.0x shared table (s_bar)")
    # T14 round 3 (section 8) switches; defaults = rounds 1-2 behaviour
    ap.add_argument("--sim-scene", action="store_true", help="simulator kinematics and robot / object collision shapes (sim_scene.py); "
                    "Bin / Chair object = the 16 PhysX pieces x k")
    ap.add_argument("--prior", choices=("all", "body"), default="all", help="nominal prior lambda 5 on all 36 qpos, or on root 7 + legs 12 + waist 3")
    ap.add_argument("--anchor-weight", type=float, default=0.0, help="T14 step 2 CoCarry anchor term weight (0 = off)")
    # T14 round 4 (section 9) switches; defaults = round-3 behaviour
    ap.add_argument("--landmarks", action="store_true", help="A: Index1 / Pinky1 / middle fingertip per hand in the interaction mesh (hand_landmarks.py)")
    ap.add_argument("--palm-orient-weight", type=float, default=0.0, help="B: palm normal toward the inward surface normal at the anchor, in human-contact "
                    "frames (anchor activation); 0 = off; round 4 B uses HandOrientationConfig.palm_direction_weight = 5.0")
    a = ap.parse_args()
    if a.palm_orient_weight > 0:
        assert a.anchor_weight > 0 and a.landmarks, "B = A + palm orientation; needs the anchor term"
    global SEQ_RAW, CANON, STAGE1, OBJ_KEY
    cfg_t = TASKS[a.task]
    SEQ_RAW, CANON, STAGE1, OBJ_KEY = CORE4D / "CORE4D_Real/human_object_motions" / cfg_t["raw"], cfg_t["canon"], cfg_t["stage1"], cfg_t["obj_key"]
    t0 = time.time()
    out = a.out / f"person{a.person}"
    out.mkdir(parents=True, exist_ok=False)
    pair = load_pair_script()
    from holosoma_retargeting.data_utils.core4d_adapter import CORE4D_TO_OMNI_ROTATION, load_canonical_core4d_sequence, resample_core4d_pair_sequence
    from holosoma_retargeting.data_utils.core4d_object_assets import create_core4d_retarget_assets
    from holosoma_retargeting.data_utils.core4d_retarget import _scale_points_about_ground_anchor, prepare_core4d_person_retarget_input
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import (build_retargeter_kwargs_from_config, convert_object_poses_to_mujoco_order,
                                                              create_task_constants, setup_object_data)
    from holosoma_retargeting.src.utils import create_uniformly_scaled_object_scene_xml, extract_foot_sticking_sequence_velocity
    from palm_retargeter import PalmInteractionMeshRetargeter

    pidx = a.person - 1
    src = load_canonical_core4d_sequence(CANON)
    seq = resample_core4d_pair_sequence(src, OUTPUT_FPS)
    persons = [prepare_core4d_person_retarget_input(seq, i) for i in range(2)]
    pin = persons[pidx]
    s_i = pin.human_to_robot_scale
    s_bar = float(np.mean([p.human_to_robot_scale for p in persons])) if cfg_t["base"] == "mean" else 1.0
    s_shared = s_bar * a.k  # k = 1: the current training object
    anchor = pin.scale_anchor
    assert np.allclose(persons[0].scale_anchor, persons[1].scale_anchor)
    phys = pin.physical_object_poses  # [T, 7] wxyz + xyz, Z-up
    shared = phys.copy()
    shared[:, 4:] = anchor + s_shared * (phys[:, 4:] - anchor)
    import holosoma_retargeting
    rec = dict(task=a.task, base=cfg_t["base"], holosoma_retargeting_module=str(Path(holosoma_retargeting.__file__).resolve()), python=sys.executable, variant=a.variant, person=a.person, s_i=s_i, s_bar=s_bar, k=a.k, s_shared=s_shared, scale_anchor=anchor.tolist(), canonical=str(CANON), canonical_sha256=sha(CANON),
               frames=len(phys), fps=OUTPUT_FPS)

    # Stage-1 solve of this person (existing, unchanged), checked against the current reference
    st1_path = STAGE1 / f"person{a.person}/person{a.person}_nominal_scaled.npz"
    q1 = np.load(st1_path)["qpos"].astype(np.float64)
    rec.update(stage1=str(st1_path), stage1_sha256=sha(st1_path))
    cur = np.load(cfg_t["current"])
    rec["stage1_equals_current_reference_robot"] = (float(np.abs(q1[:, :36] - cur["robot_qpos"][:, pidx]).max())
                                                    if cur["robot_qpos"].shape[0] == len(q1) else f"frames differ {cur['robot_qpos'].shape[0]} vs {len(q1)}")
    assert len(q1) == len(phys)

    human = pin.human_joints
    palm_info = None
    if a.variant == "new":
        palms, contact, palm_local = source_palms_and_contacts(CORE4D_TO_OMNI_ROTATION)
        pz, jz = palms[pidx]
        canon_err = float(np.abs(jz[:, :22] - src.human_joints[:, pidx]).max())  # axis conversion check vs canonical
        pz = np.einsum("ij,tkj->tki", CORE4D_TO_OMNI_ROTATION, pz)  # [T15, 2, 3]
        p30 = resample_linear(pz, src.fps, OUTPUT_FPS)
        j30 = resample_linear(src.human_joints[:, pidx], src.fps, OUTPUT_FPS)
        resample_err = float(np.abs(j30 - seq.human_joints[:, pidx]).max())
        p30 = _scale_points_about_ground_anchor(p30, anchor=anchor, scale=s_i, ground_offset_m=pin.ground_offset_m)
        human = np.concatenate([human, p30], axis=1)  # [T, 24, 3]: ..., L_Palm, R_Palm
        if a.landmarks:
            import hand_landmarks as hl
            P = np.load(SEQ_RAW / f"person{a.person}_poses.npz", allow_pickle=True)["arr_0"].item()
            Jr, Vr = P["joints"].astype(np.float64), P["vertices"].astype(np.float64)
            lm = np.concatenate([hl.human_points(Jr, Vr, side) for side in ("left", "right")], 1)  # [T15, 6, 3] raw Y-up
            lm = np.einsum("ij,tkj->tki", CORE4D_TO_OMNI_ROTATION, lm)
            lm30 = _scale_points_about_ground_anchor(resample_linear(lm, src.fps, OUTPUT_FPS), anchor=anchor, scale=s_i, ground_offset_m=pin.ground_offset_m)
            human = np.concatenate([human, lm30], axis=1)  # [T, 30, 3]: ..., L_Index1, L_Pinky1, L_MiddleTip, R_...
        # anchors (raw object frame, metres) and the step-3 shift
        raw = trimesh.load(seq.object_mesh_path, force="mesh")
        disp = []
        anchors = {}
        for side in ("left", "right"):
            c15 = contact[(a.person, side)]
            loc = palm_local[(a.person, side)][c15]
            if len(loc) == 0:
                continue
            med = np.median(loc, axis=0)
            # raw CORE4D object frame is the mesh frame; the canonical pose maps the same mesh frame to the Z-up world
            surf = trimesh.proximity.closest_point(raw, med[None])[0][0]
            anchors[side] = dict(contact_frames_15hz=int(c15.sum()), median_local_m=med.round(5).tolist(), anchor_local_m=surf.round(5).tolist())
            c30 = resample_linear(c15.astype(float)[:, None], src.fps, OUTPUT_FPS)[:, 0] > 0.5
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat(phys[:, [1, 2, 3, 0]]).as_matrix()
            d = (s_shared - s_i) * ((phys[:, 4:] - anchor) + R @ surf)
            disp.append(d[c30])
        shift = np.concatenate(disp).mean(0)
        shift_xy = np.array([shift[0], shift[1], 0.0])
        q_prior = q1.copy()
        q_prior[:, :2] += shift_xy[:2]
        sim_hands, dump_sha = write_sim_hand_meshes(out)
        offsets = {k: palm_point(trimesh.load(sim_hands[s], force="mesh").vertices, s) for k, s in (("L_Palm", "left"), ("R_Palm", "right"))}
        if a.landmarks:
            import hand_landmarks as hl
            lm_rec = {}
            for side in ("left", "right"):
                off, lm_rec[side] = hl.robot_offsets(trimesh.load(sim_hands[side], force="mesh").vertices, side)
                for k, key in zip(hl.LANDMARKS, hl.keys(side)):
                    offsets[key] = off[k]
                lm_rec[side]["offsets_m"] = {k: np.round(v, 6).tolist() for k, v in off.items()}
            rec["landmarks"] = dict(human=dict(joints=hl.JOINTS["left"] | {"fingers": "25-39"}, tip_vertex=hl.TIP_VERTEX), robot=lm_rec)
        palm_info = dict(palm_vertices_per_hand={k: int(len(v)) for k, v in palm_vertex_ids().items()}, axis_conversion_max_err_m=canon_err,
                         resample_max_err_m=resample_err, anchors=anchors, shift_mean_m=shift.round(6).tolist(), shift_xy_m=shift_xy[:2].round(6).tolist(),
                         robot_palm_offsets_m={k: v.round(6).tolist() for k, v in offsets.items()}, sim_hand_meshes={k: str(v) for k, v in sim_hands.items()},
                         physx_dump=dump_sha)
        rec["palm"] = palm_info
        print(json.dumps(palm_info), flush=True)
    else:
        q_prior = q1.copy()
        offsets = None

    # constants, scene and retargeter exactly as the two-stage CORE4D run's Stage 2, except the listed changes
    robot_config = RobotConfig(robot_type="g1", robot_urdf_file=str(pair.BASE_G1_URDF.resolve()))
    n_prior = 7 + robot_config.ROBOT_DOF if a.prior == "all" else 7 + 12 + 3  # qpos order: root 7, legs 12, waist 3, arms 14
    robot_config = replace(robot_config, nominal_tracking_indices=np.arange(n_prior, dtype=np.int64))
    motion_config = MotionDataConfig(data_format="smplx", robot_type="g1")
    task_config = TaskConfig(object_name=seq.object_name)
    base_cfg = RetargeterConfig(visualize=False, debug=False)
    cfg = replace(base_cfg, apply_manual_joint_limit_overrides=False, foot_sticking_tolerance=pair.CORE4D_STAGE2_FOOT_XY_TOLERANCE_M,
                  elastic_constraints=replace(base_cfg.elastic_constraints, enable=True, object_collision_weight=pair.CORE4D_STAGE2_ELASTIC_OBJECT_WEIGHT,
                                              foot_kinematics_weight=pair.CORE4D_STAGE2_ELASTIC_FOOT_WEIGHT))
    constants = create_task_constants(robot_config, motion_config, task_config, "object_interaction")
    assets = create_core4d_retarget_assets(seq.object_mesh_path, pair.BASE_G1_XML, seq.object_name, out / "assets", mass_kg=1.0)
    constants.OBJECT_MESH_FILE = seq.object_mesh_path
    constants.OBJECT_URDF_FILE = str(assets.object_urdf_path)
    shared_scene = create_uniformly_scaled_object_scene_xml(assets.scene_xml_path, seq.object_name, s_shared, out / "assets" / "shared_scene.xml")
    if a.variant == "new":
        shared_scene = patch_hand_meshes(shared_scene, out / "assets" / "shared_scene_simhands.xml", sim_hands)
        extra = dict(PALM_KEYS)
        if a.landmarks:
            import hand_landmarks as hl
            extra |= {k: "left_rubber_hand_link" for k in hl.keys("left")} | {k: "right_rubber_hand_link" for k in hl.keys("right")}
        assert len(constants.DEMO_JOINTS) == 22 and human.shape[1] == 22 + len(extra), (len(constants.DEMO_JOINTS), human.shape)
        constants.DEMO_JOINTS = list(constants.DEMO_JOINTS) + list(extra)
        constants.JOINTS_MAPPING = dict(constants.JOINTS_MAPPING) | extra
    if a.sim_scene:
        from sim_scene import object_pieces, simify_scene
        pcs = object_pieces(a.task, a.k) if a.task in ("bin", "chair") else None
        shared_scene, rec["sim_scene"] = simify_scene(shared_scene, out / "assets" / "shared_scene_sim.xml", out / "assets" / "sim_meshes", seq.object_name, pcs)
    constants.SCENE_XML_FILE = str(shared_scene)
    obj_pts, obj_pts_demo, urdf_path = setup_object_data("object_interaction", constants, None, s_i, task_config, False)
    obj_pts_shared = obj_pts * s_shared
    rec["object_points"] = dict(count=int(len(obj_pts)), demo_over_raw=float(np.median(np.linalg.norm(obj_pts_demo, axis=1) / np.maximum(np.linalg.norm(obj_pts, axis=1), 1e-9))))
    kw = build_retargeter_kwargs_from_config(cfg, constants, urdf_path, "object_interaction")
    kw["nominal_tracking_tau"] = cfg.nominal_tracking_tau
    kw["anchor_nominal_foot_height"] = True
    kw["nominal_foot_height_tolerance"] = pair.CORE4D_NOMINAL_FOOT_HEIGHT_TOLERANCE_M
    kw["scene_xml_path"] = str(shared_scene)
    anchor_term, orient_term = None, None
    if a.anchor_weight > 0:
        anchor_term, rec["anchor_term"] = step2_anchor_term(a, contact, palm_local, src.fps, len(phys), s_shared)
    if a.palm_orient_weight > 0:
        import hand_landmarks as hl
        orient_term = dict(weight=a.palm_orient_weight, activation=anchor_term["activation"], targets_obj=-anchor_term["normals"],
                           palm_normal_local=hl.PALM_NORMAL_LOCAL)
        rec["orient_term"] = dict(weight=a.palm_orient_weight, form="w * act * ||n_palm(q) - n*||^2 (HandOrientationConfig form, 7 arm joints)",
                                  target="inward object-surface normal at the anchor", activation="anchor activation (5-frame smoothstep in contact windows)")
    retargeter = PalmInteractionMeshRetargeter(**kw, palm_offsets=offsets, anchor_term=anchor_term, orient_term=orient_term)
    import mujoco
    jadr = lambda n: int(retargeter.robot_model.jnt_qposadr[mujoco.mj_name2id(retargeter.robot_model, mujoco.mjtObj.mjOBJ_JOINT, n)])
    assert jadr("left_hip_pitch_joint") == 7 and jadr("waist_pitch_joint") == 21 and jadr("left_shoulder_pitch_joint") == 22
    assert list(retargeter.track_nominal_indices) == list(range(n_prior)), retargeter.track_nominal_indices
    rec["prior"] = dict(scope=a.prior, qpos_indices=f"0..{n_prior - 1}", weight=retargeter.w_nominal_tracking_init)
    gid = mujoco.mj_name2id(retargeter.robot_model, mujoco.mjtObj.mjOBJ_GEOM, "left_rubber_hand_link")
    rec["scene"] = dict(path=str(shared_scene), left_hand_mesh_vertices=int(retargeter.robot_model.mesh_vertnum[retargeter.robot_model.geom_dataid[gid]]),
                        interaction_mesh_keys=list(retargeter.laplacian_match_links), w_nominal_tracking_init=retargeter.w_nominal_tracking_init)

    n = a.max_frames or len(phys)
    nominal_mj = convert_object_poses_to_mujoco_order(pin.nominal_object_poses)[:n]
    shared_mj = convert_object_poses_to_mujoco_order(shared)[:n]
    foot = extract_foot_sticking_sequence_velocity(human[:n], retargeter.demo_joints, motion_config.toe_names)
    foot[0][motion_config.toe_names[0]] = False
    foot[0][motion_config.toe_names[1]] = False
    q_nom = q_prior[:n].copy()
    q_nom[:, -7:] = shared_mj  # object part of the prior = the locked shared object (not used by the prior indices 0..35)
    print("solving", a.variant, "person", a.person, n, "frames", flush=True)
    q, *_ = retargeter.retarget_motion(human_joint_motions=human[:n], object_poses=nominal_mj, object_poses_augmented=shared_mj,
                                       object_points_local_demo=obj_pts_demo, object_points_local=obj_pts_shared, foot_sticking_sequences=foot,
                                       q_a_init=q_nom[0], q_nominal_list=q_nom, original=False, dest_res_path=str(out / f"person{a.person}_{a.variant}.npz"))
    q = np.asarray(q)
    np.savez(out / f"person{a.person}_{a.variant}_qpos.npz", qpos=q, shared_object_qpos_mj=shared_mj, human_joints=human[:n], fps=OUTPUT_FPS)
    rec.update(seconds=round(time.time() - t0, 1), finite=bool(np.isfinite(q).all()), qpos_shape=list(q.shape),
               object_locked_max_err=float(np.abs(q[:, -7:] - shared_mj).max()))
    import resource
    rec["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3)
    (out / "run_record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("seconds", "finite", "qpos_shape", "object_locked_max_err", "peak_rss_gb", "stage1_equals_current_reference_robot")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
