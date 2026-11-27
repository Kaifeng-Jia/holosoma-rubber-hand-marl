#!/usr/bin/env python3
"""T14 front-back training reference (sub8_suitcase_020, level 1.0, frozen step-9g2 construction): palm-to-box gap of the
HUMAN palms vs the ROBOT palms, per frame of the 30 fps pair (Kevin 2026-11-27, report only). Read only, CPU, nothing of the
reference / construction / training is changed; outputs only under logs_system1/t14/frontback_train/palm_gap and a copy
under results_system1/T14_frontback/palm_gap.

Question: in the 50 Hz runtime reference three of the four robot palms sit 4-6 cm from the box while should_contact is true
all the time (contact_labels.py --task frontback). (a) How far are the human palms from the box in the same clip? (b) Is robot
0's left-hand gap already in the demonstration or introduced by the retargeting solve?

Frames / scale: the solver world (role_solve.py / step9_diag.load_role0: OMOMO sequence frames 103:163 at 30 Hz, person
preprocessed with smpl_scale s, box = the captured suitcase's oriented bounding box x s, nominal grounded poses). Pair frame t
= clip frame t; robot 0 = the demonstrator, robot 1 = the 9g2 mirror partner (frontback.build human1 = the demonstrator mirrored
at the same frame, left / right exchanged). Phases: lift [0, w0), walk [w0, w1), put_down [w1, T) of frontback.walking_segment
(w0 = 13, w1 = 56, as eval.json). All signed distances in metres, POSITIVE = OUTSIDE, negative = inside (frontback.box_sd).

Palm points
  human  = mean of the 67 palm-side SMPL-X wrist vertices (table_shared.palm_vertex_ids via single_shared.omomo_palms), carried
           with the person through preprocess_motion_data (x s): the solver person's "L_Palm" / "R_Palm" rows (the same rows the
           interaction mesh and step9_gates.prediag_9g use). Robot 1's L_Palm = the mirror of the demonstrator's R_Palm.
  robot  = the rubber-hand palm point (table_shared.palm_point on the simulator hand hull: hull bbox centre moved along the
           palm normal PALM_NORMAL_LOCAL to the hull's support plane), offset in the <side>_rubber_hand_link frame =
           run_record robot_palm_offsets_m (contact_labels.py palm_offset agrees to < 1e-5 m). FK: MuJoCo on the solve's own
           scene_sim.xml (the solver's kinematics); check: the PhysX-URDF FK of contact_labels.py (contact_stats.Robot("physx")).
Boxes
  shared    = the shared target P(t), size L x w x h = the solve's target URDF (= the training URDF): the robots' box (pair
              object_qpos); the step9_gates.prediag_9g "sd_to_shared_box" definition, here for every frame;
  level_own = the person's levelled own box (robot 0: Ob, robot 1: Sm Ob), size of the own box: the Stage-2 person-side
              object frame of the interaction mesh (--fb-human-frame target);
  real_own  = the person's real own box (robot 0: O, the captured box at its nominal poses; robot 1: Sm O), size of the own box;
  mesh      = the captured suitcase mesh x s in the real own box frame (box_assets.mesh_in_box_frame): palm -> nearest surface
              point, sign from trimesh contains (watertightness recorded); also the hand-vertex rule of should_contact
              (human_contact.npz h_sd, reported as -h_sd = the smallest signed distance of any hand vertex to the mesh).
Solver hand targets (role_solve.py, construction frontback)
  anchor    = the anchor-term point of that hand (weight 10 x smoothstep activation, one window = the whole clip), constant in
              the shared-target frame (run_record anchor_term.windows.anchor_local, projected onto the target box);
  im        = the person's palm in its Stage-2 person-side frame carried into the shared target, Meff Nr^-1 x (robot 0:
              M0 Ob^-1; robot 1: M1 F (Sm Ob F)^-1): where the interaction mesh would put the robot palm if its deformation were
              the identity (approximation: its person-side object points are capture-mesh points, its robot-side points box
              points);
  hull      = the hand's simulator collision hull (the physx_<side>_rubber_hand_link geom of scene_sim.xml, the geom of the
              solver's object non-penetration constraint >= -1 mm): min signed distance to the shared box, exact
              (mujoco.mj_geomDistance, the solver's function) and over the hull vertices; the exact witness point on the box
              (its face) and its distance from the palm point (where the hand touches the box);
  normals   = robot palm normal (link R x PALM_NORMAL_LOCAL) and the person's palm-normal target of the orient term (weight 5,
              build_pt_wrist_palm_orientation_targets column 2; robot 1 mirrored, hands exchanged, as role_solve): angle to the
              inward normal of the shared-box face the hand's anchor lies on, and to each other; the box face nearest each
              point (robot palm, person palm in the shared box, im target, hull contact) is recorded.
50 Hz runtime check: the robot palms of the runtime npz (contact_labels.robot_palms_obj path) vs the training URDF box,
signed, next to the labels' palm_to_surface_m (unsigned, sampled surface).

Outputs: palm_gap.csv (per pair frame x robot x hand), palm_gap_runtime50.csv, palm_gap_summary.json (per hand and phase:
n, median, p95, min, max, mean in cm; input sha256s; consistency checks), copied to results_system1/T14_frontback/palm_gap/.

    CUDA_VISIBLE_DEVICES="" /home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python -B \
        system1/t14/frontback_train/palm_gap_check.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OBJ = ROOT / "system1/t14/objects"
for p in (ROOT / "system1/t13v4", ROOT / "system1/t12v2", ROOT / "scripts", ROOT / "src/holosoma_retargeting", ROOT / "system1/t14", OBJ):
    sys.path.insert(0, str(p))
RT = ROOT / "src/holosoma_retargeting/holosoma_retargeting"

OBJECT, SEQ, FRAMES, LEVEL = "suitcase", "sub8_suitcase_020", (103, 163), 1.0
SOLVE = ROOT / "logs_system1/t14/step9g2/sub8_suitcase_020/m1p00"
REF = ROOT / "exports_system1/T14_references/frontback_train/sub8_suitcase_020_m1p00"
PAIR = REF / "sub8_suitcase_020_m1p00_pair_fps30.npz"
MANIFEST = REF / "source_manifest.json"
RTDIR = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14fb/table"
RUNTIME = RTDIR / "core4d_pair_runtime_fps50.npz"
TRAIN_URDF = RTDIR / "frontback_box5kg_training.urdf"
LABELS = ROOT / "logs_system1/t14/frontback_train/labels/frontback_m1p00_labels.npz"
OUT = ROOT / "logs_system1/t14/frontback_train/palm_gap"
RES = ROOT / "results_system1/T14_frontback/palm_gap"
TAG = "sub8_suitcase_020"
SIDES = ("left", "right")
HN = ("L", "R")
PHASES = ("lift", "walk", "put_down")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats_cm(x, unit=100.0):
    """n, median, p95, min, max, mean of the finite values, x unit (metres -> cm by default; unit=1 for degrees)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    r = lambda v: round(unit * float(v), 2)  # noqa: E731
    return dict(n=int(len(x)), median=r(np.median(x)), p95=r(np.percentile(x, 95)), min=r(x.min()), max=r(x.max()), mean=r(x.mean()))


def angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.einsum("...i,...i->...", a, b), -1, 1)))


def nearest_face(q, size):
    """local point(s) q [N, 3] in a centred box of full size `size`: (axis, sign) of the face nearest (largest |q| - h)."""
    d = np.abs(q) - np.asarray(size) / 2
    ax = d.argmax(-1)
    sg = np.sign(np.take_along_axis(q, ax[..., None], -1)[..., 0])
    return ax, sg


def memoize(mod, name):
    """cache a loader by its arguments (the original keeps writing its own .last attributes)."""
    f = getattr(mod, name)
    cache = {}

    def g(*args):
        key = repr(args)
        if key not in cache:
            cache[key] = f(*args)
        return cache[key]
    setattr(mod, name, g)


def main():
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation
    import single_shared as ss
    import omomo_motion as om
    memoize(ss, "omomo_palms")  # load_role0 / this script share one SMPL-X pass and one joblib load
    memoize(om, "load_sequences")
    import box_assets as ba
    import frontback as fb
    import partner as pt
    import step9_gates as sg
    from step9_diag import load_role0
    import hand_landmarks as hl
    from table_shared import smoothstep_window

    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    # ------------------------------------------------------------------ construction rebuilt with the prediag's own code
    h0, names, nom, size, s = load_role0(OBJECT, SEQ, FRAMES)
    O = pt.mj_to_mat(nom)
    v = om.load_sequences({OBJECT})[SEQ]
    walk = fb.walking_segment(v, *FRAMES)
    C = fb.build(h0, names, O, size, LEVEL, turn="world", legs="mirror", walk=walk, walk_distance="pelvis_feet")
    T = len(O)
    w0, w1 = int(C["walk"]["w0"]), int(C["walk"]["w1"])
    phase = np.array(["lift" if t < w0 else ("walk" if t < w1 else "put_down") for t in range(T)])
    tsize = np.array(size, float)
    tsize[C["k"]] = C["Lm"]
    ev = json.loads((SOLVE / "eval.json").read_text())
    pd = sg.prediag_9g(C, h0, names, O, size, s)["palms_vs_contact_points"]
    checks["prediag_9g_palms_reproduced"] = pd == ev["pelvis_mismatch"]["prediag_9g"]["palms_vs_contact_points"]
    recs = [json.loads((SOLVE / f"r{r}" / "run_record.json").read_text()) for r in (0, 1)]
    zq = [np.load(SOLVE / f"r{r}" / f"{TAG}_new_qpos.npz") for r in (0, 1)]
    pair = np.load(PAIR, allow_pickle=True)
    rq, oq = pair["robot_qpos"].astype(np.float64), pair["object_qpos"].astype(np.float64)
    checks["solve_person_r0_equals_rebuilt_h0_max_m"] = float(np.abs(zq[0]["human_joints"] - h0).max())
    checks["solve_person_r1_equals_rebuilt_human1_max_m"] = float(np.abs(zq[1]["human_joints"] - C["human1"]).max())
    checks["solve_P_equals_rebuilt_P_max"] = float(max(np.abs(zq[r]["P"] - C["P"]).max() for r in (0, 1)))
    checks["pair_robot_equals_solve_qpos_max"] = float(max(np.abs(rq[:, r] - zq[r]["qpos"][:, :36]).max() for r in (0, 1)))
    Pp = pt.mj_to_mat(oq)
    checks["pair_object_equals_P_max"] = float(np.abs(Pp - C["P"]).max())
    tgt_size = np.array(recs[0]["target"]["size"], float)
    import yourdfpy
    tr_size = np.array(yourdfpy.URDF.load(str(TRAIN_URDF)).robot.links[0].collisions[0].geometry.box.size, float)
    checks["shared_box_size_m"] = np.round(tsize, 6).tolist()
    checks["target_urdf_size_m"] = np.round(tgt_size, 6).tolist()
    checks["training_urdf_size_m"] = np.round(tr_size, 6).tolist()
    checks["own_box_size_m"] = np.round(size, 6).tolist()
    checks["smpl_scale"] = s
    checks["phases"] = dict(w0=w0, w1=w1, frames=T)

    # ------------------------------------------------------------------ human palms (solver scale) vs the boxes
    F = pt.affine(np.diag([1.0, 1.0, -1.0]), np.zeros(3))
    persons = {0: h0, 1: C["human1"]}
    level_own = {0: C["Ob"], 1: C["own1"]}
    real_own = {0: O, 1: C["own1_real"]}
    Meff = {0: C["M0"], 1: C["M1"] @ F}
    Nr = {0: C["Ob"], 1: C["own1"] @ F}
    jp = {"L": names.index("L_Palm"), "R": names.index("R_Palm")}
    bf = ba.box_frame(OBJECT, v)
    mesh = ba.mesh_in_box_frame(OBJECT, bf)
    mesh.vertices = np.asarray(mesh.vertices) * s
    checks["capture_mesh"] = dict(watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                                  sha256=bf["mesh_sha256"], bounds_in_box_frame_m=np.round(mesh.bounds, 5).tolist())
    # raw box frame (role_solve anchors): palm in the raw captured-box frame x s, vs the nominal box O used here
    palms_raw, fit = ss.omomo_palms(SEQ)
    raw_poses, _ = ba.poses_omomo_new(v, bf, fit)
    a_, b_ = FRAMES
    Rraw = Rotation.from_quat(raw_poses[a_:b_, [1, 2, 3, 0]]).as_matrix()
    local_raw = np.einsum("tji,thj->thi", Rraw, palms_raw[a_:b_] - raw_poses[a_:b_, None, 4:7]) * s  # [T, 2 (L, R), 3]
    loc_O = np.stack([np.einsum("tji,tj->ti", O[:, :3, :3], h0[:, jp[h]] - O[:, :3, 3]) for h in HN], 1)
    checks["palm_local_nominal_box_vs_raw_box_x_s_max_m"] = float(np.abs(loc_O - local_raw).max())
    dw = np.einsum("tij,thj->thi", O[:, :3, :3], loc_O - local_raw)  # the same difference in world axes
    checks["palm_nominal_vs_raw_world_diff_m"] = dict(mean=np.round(dw.reshape(-1, 3).mean(0), 5).tolist(), std=np.round(dw.reshape(-1, 3).std(0), 5).tolist(),
                                                     note="person grounded (toe z_min of the clip) then x s; object nominal = raw x s (preprocess_motion_data + "
                                                          "create_grounded_nominal_object_poses): expected constant world-z offset -s z_min")
    hc = [np.load(SOLVE / f"r{r}" / "human_contact.npz") for r in (0, 1)]

    H = {}
    for r in (0, 1):
        for hi, h in enumerate(HN):
            x = persons[r][:, jp[h]]
            sd_sh = np.array([fb.box_sd(x[t][None], C["P"][t], tsize)[0] for t in range(T)])
            sd_lo = np.array([fb.box_sd(x[t][None], level_own[r][t], size)[0] for t in range(T)])
            sd_ro = np.array([fb.box_sd(x[t][None], real_own[r][t], size)[0] for t in range(T)])
            q = np.einsum("tji,tj->ti", real_own[r][:, :3, :3], x - real_own[r][:, :3, 3])  # local in the real own box (mirror: = source hand's)
            cpt, dist, tri = trimesh.proximity.closest_point(mesh, q)
            in_n = np.einsum("ij,ij->i", q - cpt, mesh.face_normals[tri]) < 0  # sign from the nearest face's normal
            try:
                inside = mesh.contains(q)
                checks.setdefault("mesh_sign_contains_vs_face_normal_disagree_frames", {})[f"r{r}_{h}"] = int((inside != in_n).sum())
            except Exception as ex:  # noqa: BLE001
                inside = in_n
                checks["mesh_sign_method"] = f"face normal (contains failed: {ex})"
            sd_mesh = np.where(inside, -dist, dist)
            im_loc = np.stack([(Meff[r] @ np.linalg.inv(Nr[r][t]) @ np.r_[x[t], 1.0])[:3] for t in range(T)])
            sd_im = fb.box_sd(im_loc, np.eye(4), tsize)
            H[(r, h)] = dict(x=x, sd_shared=sd_sh, sd_level_own=sd_lo, sd_real_own=sd_ro, sd_mesh=sd_mesh, local_real_own=q, im_loc=im_loc, sd_im=sd_im,
                             hand_vertex_min_sd_mesh=-hc[r]["h_sd"][:T, hi], source_hand=h if r == 0 else ("R" if h == "L" else "L"))
    # mirror consistency: robot 1's person palm = robot 0's other palm, mirrored (distances equal)
    checks["mirror_r1L_vs_r0R_real_own_max_m"] = float(np.abs(H[(1, "L")]["sd_real_own"] - H[(0, "R")]["sd_real_own"]).max())
    checks["mirror_r1R_vs_r0L_real_own_max_m"] = float(np.abs(H[(1, "R")]["sd_real_own"] - H[(0, "L")]["sd_real_own"]).max())

    # ------------------------------------------------------------------ person palm-normal targets (role_solve orient term)
    cwd = os.getcwd()
    os.chdir(RT)
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import create_task_constants, load_motion_data
    from holosoma_retargeting.src.utils import build_pt_wrist_palm_orientation_targets, load_intermimic_wrist_quaternions
    mc = MotionDataConfig(data_format="smplh", robot_type="g1")
    constants = create_task_constants(RobotConfig(robot_type="g1"), mc, TaskConfig(object_name=OBJECT), "object_interaction")
    human_raw, _, _ = load_motion_data("object_interaction", "smplh", ss.OMOMO_NEW, SEQ, constants, mc)
    wq = load_intermimic_wrist_quaternions(ss.OMOMO_NEW / f"{SEQ}.pt")
    palm_rot, cal = build_pt_wrist_palm_orientation_targets(human_raw.copy(), wq, list(constants.DEMO_JOINTS))
    os.chdir(cwd)
    checks["orient_calibration_p90_deg"] = np.round(cal, 6).tolist()
    checks["orient_calibration_p90_deg_run_record"] = recs[0]["palm_track"]["calibration_p90_deg"]
    palm_rot = palm_rot[a_:b_]
    pw = {0: palm_rot[..., :, 2], 1: (palm_rot[..., :, 2] @ C["Ref"].T)[:, [1, 0]]}

    # ------------------------------------------------------------------ robot palms (pair, 30 fps): MuJoCo solver FK + PhysX-URDF FK
    import contact_stats as cs
    robot = cs.Robot("physx")
    Rb = {}
    for r in (0, 1):
        m = mujoco.MjModel.from_xml_path(str(SOLVE / f"r{r}" / "scene_sim.xml"))
        d = mujoco.MjData(m)
        assert m.nq == 43, m.nq
        offs = {s_: np.asarray(recs[r]["robot_palm_offsets_m"][f"{h}_Palm"], float) for s_, h in zip(SIDES, HN)}
        bid = {s_: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s_}_rubber_hand_link") for s_ in SIDES}
        gbox = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, OBJECT)
        gnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        ghull = {}
        for s_ in SIDES:  # the hand geom that collides with the box (box contype 4 & conaffinity 4): physx_<side>_rubber_hand_link_*
            cand_ = [g for g, n_ in enumerate(gnames) if n_.startswith(f"physx_{s_}_rubber_hand_link")]
            assert len(cand_) == 1 and (m.geom_contype[gbox] & m.geom_conaffinity[cand_[0]]), cand_
            ghull[s_] = cand_[0]
        checks.setdefault("hull_geoms", {})[f"r{r}"] = {s_: gnames[ghull[s_]] for s_ in SIDES}
        assert np.allclose(m.geom_size[gbox] * 2, tsize, atol=1e-7), (m.geom_size[gbox], tsize)
        hverts = {s_: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[ghull[s_]]]:m.mesh_vertadr[m.geom_dataid[ghull[s_]]] + m.mesh_vertnum[m.geom_dataid[ghull[s_]]]].astype(np.float64)
                  for s_ in SIDES}
        jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        dof_names = [jn[j] for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        assert [int(m.jnt_qposadr[jn.index(n_)]) for n_ in dof_names] == list(range(7, 36)), "robot dofs at qpos 7..35"
        out = {s_: dict(palm=np.zeros((T, 3)), R=np.zeros((T, 3, 3)), hull_exact=np.zeros(T), hull_vert=np.zeros(T), wit_box=np.zeros((T, 3)),
                        wit_hand=np.zeros((T, 3))) for s_ in SIDES}
        ft = np.zeros(6)
        for t in range(T):
            d.qpos[:36] = rq[t, r]
            d.qpos[36:43] = oq[t]
            mujoco.mj_kinematics(m, d)
            for s_ in SIDES:
                Rl = d.xmat[bid[s_]].reshape(3, 3).copy()
                out[s_]["R"][t] = Rl
                out[s_]["palm"][t] = d.xpos[bid[s_]] + Rl @ offs[s_]
                out[s_]["hull_exact"][t] = mujoco.mj_geomDistance(m, d, ghull[s_], gbox, 0.5, ft)
                out[s_]["wit_box"][t] = (ft[3:6] - Pp[t, :3, 3]) @ Pp[t, :3, :3]  # witness point on the box, shared-target frame
                out[s_]["wit_hand"][t] = Rl.T @ (ft[0:3] - d.xpos[bid[s_]]) - offs[s_]  # witness point on the hull, link frame, minus the palm point
                gR = d.geom_xmat[ghull[s_]].reshape(3, 3)
                vw = hverts[s_] @ gR.T + d.geom_xpos[ghull[s_]]
                out[s_]["hull_vert"][t] = fb.box_sd(vw, Pp[t], tsize).min()
        # PhysX-URDF FK of contact_labels.py
        dof = rq[:, r, 7:][:, [dof_names.index(n_) for n_ in robot.dofs]]
        tl = robot.fk(rq[:, r, :3], np.concatenate([rq[:, r, 4:7], rq[:, r, 3:4]], -1), dof)
        for s_ in SIDES:
            Mx = tl[f"{s_}_rubber_hand_link"]
            out[s_]["palm_physx"] = np.einsum("tij,j->ti", Mx[:, :3, :3], offs[s_]) + Mx[:, :3, 3]
        Rb[r] = out
    checks["robot_palm_mujoco_vs_physx_urdf_fk_max_m"] = float(max(np.abs(Rb[r][s_]["palm"] - Rb[r][s_]["palm_physx"]).max() for r in (0, 1) for s_ in SIDES))
    lab = np.load(LABELS)
    checks["robot_palm_offsets_run_record_vs_labels_max_m"] = float(max(np.abs(np.asarray(recs[r]["robot_palm_offsets_m"][f"{h}_Palm"]) - lab["palm_offset"][i]).max()
                                                                       for r in (0, 1) for i, h in enumerate(HN)))

    # ------------------------------------------------------------------ per-frame table
    from collections import Counter
    rows, per, per_faces = [], {}, {}
    face = lambda ax, sg_: f"{'+' if sg_ > 0 else '-'}{'xyz'[ax]}"  # noqa: E731
    for r in (0, 1):
        anc = {w["hand"]: w for w in recs[r]["anchor_term"]["windows"]}
        for hi, (s_, h) in enumerate(zip(SIDES, HN)):
            hh, rb = H[(r, h)], Rb[r][s_]
            pl = np.einsum("tji,tj->ti", Pp[:, :3, :3], rb["palm"] - Pp[:, :3, 3])  # robot palm in the shared-target frame
            sd_r = fb.box_sd(pl, np.eye(4), tsize)
            sd_rx = np.array([fb.box_sd(rb["palm_physx"][t][None], Pp[t], tsize)[0] for t in range(T)])
            A = np.full((T, 3), np.nan)
            act = np.zeros(T)
            w = anc.get(s_)
            if w is not None:
                A[w["start"]:w["stop"]] = np.asarray(w["anchor_local"], float)
                act[w["start"]:w["stop"]] = smoothstep_window(w["stop"] - w["start"])
            sd_anchor = np.where(np.isfinite(A[:, 0]), fb.box_sd(np.nan_to_num(A), np.eye(4), tsize), np.nan)
            d_anchor = np.linalg.norm(pl - A, axis=1)
            d_im = np.linalg.norm(pl - hh["im_loc"], axis=1)
            # normals: robot palm normal and the person's orient-term target vs the inward normal of the box face the hand's
            # anchor lies on (the face the solver targets; shared target frame); nearest faces recorded separately
            n_r = rb["R"] @ hl.PALM_NORMAL_LOCAL[s_] / np.linalg.norm(hl.PALM_NORMAL_LOCAL[s_])
            axr, sgr = nearest_face(pl, tsize)
            hl_loc = np.einsum("tji,tj->ti", C["P"][:, :3, :3], hh["x"] - C["P"][:, :3, 3])
            axh, sgh = nearest_face(hl_loc, tsize)
            axi, sgi = nearest_face(hh["im_loc"], tsize)
            axa, sga = nearest_face(np.asarray(w["anchor_local"], float)[None], tsize)
            inward_a = -np.einsum("tij,j->ti", Pp[:, :3, :3], np.eye(3)[axa[0]] * sga[0])
            ang_r = angle_deg(n_r, inward_a)
            ang_h = angle_deg(pw[r][:, hi], inward_a)
            ang_rh = angle_deg(n_r, pw[r][:, hi])
            axw, sgw = nearest_face(rb["wit_box"], tsize)
            d_wit = np.linalg.norm(rb["wit_hand"], axis=1)
            per[(r, h)] = dict(human_sd_shared=hh["sd_shared"], human_sd_level_own=hh["sd_level_own"], human_sd_real_own=hh["sd_real_own"],
                               human_sd_mesh=hh["sd_mesh"], human_hand_vertex_min_sd_mesh=hh["hand_vertex_min_sd_mesh"],
                               robot_sd_shared=sd_r, robot_sd_shared_physx_fk=sd_rx, robot_minus_human_shared=sd_r - hh["sd_shared"],
                               robot_minus_human_real_own=sd_r - hh["sd_real_own"], im_target_sd=hh["sd_im"], robot_to_im_target=d_im,
                               anchor_sd=sd_anchor, robot_to_anchor=d_anchor, hull_exact_sd=rb["hull_exact"], hull_vertex_min_sd=rb["hull_vert"],
                               robot_minus_human_level_own=sd_r - hh["sd_level_own"], robot_minus_human_mesh=sd_r - hh["sd_mesh"],
                               hull_contact_to_palm_point=d_wit,
                               robot_palm_normal_vs_anchor_face_deg=ang_r, human_palm_normal_target_vs_anchor_face_deg=ang_h, robot_vs_human_palm_normal_deg=ang_rh)
            per_faces[(r, h)] = dict(anchor_face=face(axa[0], sga[0]), robot_palm_nearest_face=dict(Counter(face(a1, b1) for a1, b1 in zip(axr, sgr))),
                                     human_palm_nearest_face_shared=dict(Counter(face(a1, b1) for a1, b1 in zip(axh, sgh))),
                                     im_target_nearest_face=dict(Counter(face(a1, b1) for a1, b1 in zip(axi, sgi))),
                                     hull_contact_face=dict(Counter(face(a1, b1) for a1, b1 in zip(axw, sgw))))
            for t in range(T):
                rows.append(dict(frame=t, omomo_frame=FRAMES[0] + t, phase=phase[t], robot=r, hand=h, person_palm=("demonstrator " if r == 0 else "mirror partner (demonstrator ") + f"{hh['source_hand']}_Palm" + ("" if r == 0 else ")"),
                                 human_palm_sd_shared_m=hh["sd_shared"][t], human_palm_sd_level_own_m=hh["sd_level_own"][t], human_palm_sd_real_own_m=hh["sd_real_own"][t],
                                 human_palm_sd_mesh_m=hh["sd_mesh"][t], human_hand_vertex_min_sd_mesh_m=hh["hand_vertex_min_sd_mesh"][t],
                                 robot_palm_sd_shared_m=sd_r[t], robot_palm_sd_shared_physx_fk_m=sd_rx[t],
                                 robot_minus_human_sd_shared_m=sd_r[t] - hh["sd_shared"][t],
                                 anchor_activation=act[t], anchor_sd_m=sd_anchor[t], robot_palm_to_anchor_m=d_anchor[t],
                                 im_target_sd_m=hh["sd_im"][t], robot_palm_to_im_target_m=d_im[t],
                                 hull_exact_sd_m=rb["hull_exact"][t], hull_vertex_min_sd_m=rb["hull_vert"][t],
                                 anchor_face=face(axa[0], sga[0]), robot_palm_nearest_face=face(axr[t], sgr[t]), human_palm_nearest_face_shared=face(axh[t], sgh[t]),
                                 im_target_nearest_face=face(axi[t], sgi[t]), hull_contact_face=face(axw[t], sgw[t]), hull_contact_to_palm_point_m=d_wit[t],
                                 hull_contact_local_x=rb["wit_box"][t, 0], hull_contact_local_y=rb["wit_box"][t, 1], hull_contact_local_z=rb["wit_box"][t, 2],
                                 hull_contact_on_hand_minus_palm_link_x=rb["wit_hand"][t, 0], hull_contact_on_hand_minus_palm_link_y=rb["wit_hand"][t, 1],
                                 hull_contact_on_hand_minus_palm_link_z=rb["wit_hand"][t, 2],
                                 robot_palm_normal_vs_anchor_face_deg=ang_r[t], human_palm_normal_target_vs_anchor_face_deg=ang_h[t], robot_vs_human_palm_normal_deg=ang_rh[t],
                                 robot_palm_local_x=pl[t, 0], robot_palm_local_y=pl[t, 1], robot_palm_local_z=pl[t, 2],
                                 human_palm_local_x=hl_loc[t, 0], human_palm_local_y=hl_loc[t, 1], human_palm_local_z=hl_loc[t, 2],
                                 anchor_local_x=A[t, 0], anchor_local_y=A[t, 1], anchor_local_z=A[t, 2],
                                 im_target_local_x=hh["im_loc"][t, 0], im_target_local_y=hh["im_loc"][t, 1], im_target_local_z=hh["im_loc"][t, 2]))
    csv_p = OUT / "palm_gap.csv"
    with open(csv_p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for row in rows:
            wr.writerow({k: (f"{x:.6f}" if isinstance(x, (float, np.floating)) else x) for k, x in row.items()})

    # ------------------------------------------------------------------ 50 Hz runtime (contact_labels path), signed
    import contact_labels as cl
    z = np.load(RUNTIME, allow_pickle=True)
    jn50 = [str(n_) for n_ in z["joint_names"]]
    dof50 = z["agent_joint_pos"][:, :, [jn50.index(n_) for n_ in robot.dofs]].astype(np.float64)
    bi = [str(n_) for n_ in z["body_names"]].index("pelvis")
    offs_l = {s_: lab["palm_offset"][i].astype(np.float64) for i, s_ in enumerate(SIDES)}
    palms50 = cl.robot_palms_obj(robot, offs_l, z["agent_body_pos_w"][:, :, bi].astype(np.float64), cl.quat_wxyz_to_xyzw(z["agent_body_quat_w"][:, :, bi].astype(np.float64)),
                                 dof50, z["object_pos_w"].astype(np.float64), cl.quat_wxyz_to_xyzw(z["object_quat_w"].astype(np.float64)))  # [F, 2, 2, 3]
    F50 = len(palms50)
    sd50 = fb.box_sd(palms50.reshape(-1, 3), np.eye(4), tr_size).reshape(F50, 2, 2)
    src50 = np.clip(np.round(np.arange(F50) / 50.0 * 30).astype(int), 0, T - 1)
    ph50 = phase[src50]
    checks["labels_palm_to_surface_vs_abs_signed_runtime_max_m"] = float(np.abs(lab["palm_to_surface_m"] - np.abs(sd50)).max())
    rows50 = []
    for f_ in range(F50):
        for r in (0, 1):
            for hi, h in enumerate(HN):
                rows50.append(dict(frame50=f_, source_frame30=int(src50[f_]), phase=ph50[f_], robot=r, hand=h, robot_palm_sd_training_box_m=f"{sd50[f_, r, hi]:.6f}",
                                   labels_palm_to_surface_m=f"{float(lab['palm_to_surface_m'][f_, r, hi]):.6f}", should_contact=int(lab["should_contact"][f_, r, hi])))
    csv50 = OUT / "palm_gap_runtime50.csv"
    with open(csv50, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows50[0]))
        wr.writeheader()
        wr.writerows(rows50)

    # ------------------------------------------------------------------ summary
    def by_phase(x, ph, unit=100.0):
        return {"all": stats_cm(x, unit), **{p_: stats_cm(x[ph == p_], unit) for p_ in PHASES}}
    summary = {}
    for (r, h), dct in per.items():
        key = f"r{r}_{h}"
        summary[key] = dict(person_palm=("demonstrator " if r == 0 else "mirror partner = mirrored demonstrator ") + f"{H[(r, h)]['source_hand']}_Palm",
                            faces=per_faces[(r, h)],
                            **{k: by_phase(x, phase, 1.0 if k.endswith("_deg") else 100.0) for k, x in dct.items()},
                            runtime50_robot_sd_training_box=by_phase(sd50[:, r, HN.index(h)], ph50),
                            runtime50_labels_palm_to_surface=by_phase(lab["palm_to_surface_m"][:, r, HN.index(h)].astype(float), ph50),
                            anchor_local_m=next((w["anchor_local"] for w in recs[r]["anchor_term"]["windows"] if w["hand"] == SIDES[HN.index(h)]), None),
                            anchor_median_source_local_m=next((w["median_source_local"] for w in recs[r]["anchor_term"]["windows"] if w["hand"] == SIDES[HN.index(h)]), None),
                            anchor_projection_m=next((w["projection_m"] for w in recs[r]["anchor_term"]["windows"] if w["hand"] == SIDES[HN.index(h)]), None))
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--short", "--", "system1/t14", "system1/t13v4"], capture_output=True, text=True).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        head, dirty = None, None
    inputs = [PAIR, MANIFEST, SOLVE / "eval.json", RUNTIME, TRAIN_URDF, LABELS, Path(recs[0]["target"]["urdf"]), ss.OMOMO_NEW / f"{SEQ}.pt"]
    for r in (0, 1):
        inputs += [SOLVE / f"r{r}" / n_ for n_ in ("run_record.json", f"{TAG}_new_qpos.npz", "scene_sim.xml", "left_rubber_hand_physx_hull.obj",
                                                    "right_rubber_hand_physx_hull.obj", "human_contact.npz")]
    code = [Path(__file__), OBJ / "frontback.py", OBJ / "role_solve.py", OBJ / "step9_gates.py", OBJ / "step9_diag.py", OBJ / "box_assets.py",
            ROOT / "system1/t14/single_shared.py", ROOT / "system1/t14/table_shared.py", ROOT / "system1/t14/palm_retargeter.py", ROOT / "system1/t13v4/contact_labels.py"]
    res = dict(question="Kevin 2026-11-27: human vs robot palm-to-box distance, front-back training reference sub8_suitcase_020 level 1.0 (frozen 9g2)",
               units="cm (angles: degrees); signed distances: positive = outside the box, negative = inside", phases=dict(lift=[0, w0], walk=[w0, w1], put_down=[w1, T]),
               definitions=__doc__, checks=checks, per_hand=summary,
               inputs_sha256={str(p_.relative_to(ROOT)) if str(p_).startswith(str(ROOT)) else str(p_): sha(p_) for p_ in inputs},
               code_sha256={str(p_.relative_to(ROOT)): sha(p_) for p_ in code}, git_head=head, git_status_system1_t14_t13v4=dirty,
               capture_mesh_sha256=bf["mesh_sha256"], python=sys.executable,
               command="CUDA_VISIBLE_DEVICES=\"\" " + sys.executable + " -B " + " ".join(sys.argv),
               outputs={"palm_gap.csv": sha(csv_p), "palm_gap_runtime50.csv": sha(csv50)})
    js = OUT / "palm_gap_summary.json"
    js.write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (bool(x) if isinstance(x, np.bool_) else float(x))) + "\n")
    RES.mkdir(parents=True, exist_ok=True)
    for p_ in (csv_p, csv50, js):
        shutil.copy2(p_, RES / p_.name)
    print(json.dumps(checks, default=str))
    for key in summary:
        print(key, summary[key]["faces"], {k: summary[key][k]["all"]["median"] for k in ("human_sd_shared", "human_sd_level_own", "human_sd_real_own", "human_sd_mesh",
                                                                   "robot_sd_shared", "runtime50_robot_sd_training_box", "anchor_sd", "robot_to_anchor", "im_target_sd",
                                                                   "hull_exact_sd", "hull_contact_to_palm_point", "robot_palm_normal_vs_anchor_face_deg",
                                                                   "human_palm_normal_target_vs_anchor_face_deg")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
