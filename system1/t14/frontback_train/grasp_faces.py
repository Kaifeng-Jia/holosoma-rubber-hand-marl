#!/usr/bin/env python3
"""Grasp faces of the frozen 9g2 construction (Kevin 2026-11-27, report only; CPU; nothing of any reference is changed).

Question: on which box face does each of the four robot hands touch the box, is it the face the corresponding person hand is
on, and how far is each palm from the box (median)? Clips: the step-9g2 debug clips with at least one applicable level (verdict
C, results_system1/T14_retargeting/step9g2/compare.csv applicable_levels_after) and the 3 applicable clips of the final test
(results_system1/T14_retargeting/step9_final/levels.csv applicable_C); level 1.00 if applicable, else the smallest applicable
level. Not checked (Kevin): sub1_suitcase_036 (shoulder carry) and sub5_suitcase_046 (squat walk).

Measures: the same as palm_gap_check.py (the front-back training reference report), per pair frame of the 30 Hz solve:
  construction  = frontback.build(..., turn="world", legs="mirror", walk_distance="pelvis_feet") rebuilt with the prediag's
                  own loaders (step9_diag.load_role0); checked against the solve's saved person joints and shared target P;
  robot palm    = rubber-hand palm point (run_record robot_palm_offsets_m) by MuJoCo FK on the solve's own scene_sim.xml
                  (robot qpos and object qpos of <tag>_new_qpos.npz); signed distance to the shared target box (frontback.box_sd,
                  POSITIVE = OUTSIDE);
  touched face  = the box face of the exact closest point between the hand's simulator collision hull
                  (physx_<side>_rubber_hand_link geom, the solver's non-penetration geom) and the box (mujoco.mj_geomDistance
                  witness on the box, nearest face in the shared-target frame) = palm_gap_check "hull_contact_face";
  person face   = the face of the person palm carried into the shared target by the interaction-mesh map (Meff Nr^-1, the
                  robot palm target if the deformation were the identity) = palm_gap_check "im_target_nearest_face"; also the
                  nearest face of the person palm in the person's real own box (local axes = the shared target's) and the
                  anchor-term face (run_record anchor_term.windows anchor_local);
  person palm   = mean of the palm-side SMPL-X wrist vertices (solver person "L_Palm" / "R_Palm"); robot 1's person hand =
                  the 9g2 mirror partner's hand (the demonstrator's other hand, mirrored); signed distance to the person's
                  real own box.
Face names: +/-x, +/-y, +/-z in the box frame; role per clip: "end" = the carry axis k (the axis stretched between the two
robots; the end face towards robot 0 / robot 1 by the side of each robot's pelvis), "top" / "bottom" = the local axis most
aligned with world z, "side" = the remaining axis (left / right as seen by robot 0 facing the box); per hand also as seen by
its own robot facing the box ("own left side" / "own right side" / "near end" = the end face towards itself / "far end"). Per hand and clip: frame
counts of each face, the mode; "match" = the touched-face mode equals the person-face mode. Summary class_ per hand: "match" /
"mismatch" when the three person faces (by label, geometric, interaction-mesh target) agree and the touched face equals / differs
from them; "person faces disagree" otherwise. Distances to a face = unsigned
distance to that face's rectangle (to_face; the person palm = its interaction-mesh target point in the shared target), medians
over all frames: robot palm to the touched face and to the person face, person palm to the person face and to the touched face
(a small value = the person palm also lies near the touched face, e.g. at an edge).

Main comparison (match_label): the robot's touched face on the shared box vs the person palm's nearest face on the person's
REAL OWN box (the box the person holds; robot 1's person = the mirror partner and its mirrored box), carried to the shared box
BY LABEL, the construction's correspondence: real own face (a, s) = levelled own face (a, s) (frontback rule 3 keeps the box
axes: long axis k turned to the walking direction, the frame-0 vertical axis vertical), then the rotation part of M0 (role 0)
/ M1 (role 1) (the own-box labels +/-x ... are not the shared target's labels: candidate Q0 = diag(-1, -1, 1) for the "sigma-1"
clips). Second, geometric (match_mapped): the real face's direction carried by Meff Nr^-1 x real own box frame (includes the
real box's yaw / tilt against the levelled box, so near 45 degrees it can switch faces). Also reported: the interaction-mesh target face
(person_face: the person palm POINT carried into the shared box; near an edge it can differ from the mapped face because the
own box is one end of the stretched shared box), and viewer roles (view_roles: the robot's touched face as seen from the
robot's pelvis, the person's own-box face as seen from the person's pelvis; role_match) - the person may face the box
differently from the robot, so the viewer roles are a second view, not the face correspondence.

Outputs: logs_system1/t14/frontback_train/grasp_faces/<tag>.json + frames_<tag>.csv; summary grasp_faces.csv / .json; copies
under results_system1/T14_frontback/grasp_faces/. Check: sub8_suitcase_020 1.00 per-frame values equal palm_gap.csv.

    CUDA_VISIBLE_DEVICES="" .../hsretargeting_s1/bin/python -B system1/t14/frontback_train/grasp_faces.py --clip TAG   # one clip
    CUDA_VISIBLE_DEVICES="" .../hsretargeting_s1/bin/python -B system1/t14/frontback_train/grasp_faces.py --summary    # tables
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OBJ = ROOT / "system1/t14/objects"
for p in (ROOT / "system1/t13v4", ROOT / "system1/t12v2", ROOT / "scripts", ROOT / "src/holosoma_retargeting", ROOT / "system1/t14", OBJ,
          Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))
OUT = ROOT / "logs_system1/t14/frontback_train/grasp_faces"
RES = ROOT / "results_system1/T14_frontback/grasp_faces"
SKIP = ("sub1_suitcase_036", "sub5_suitcase_046")  # Kevin 2026-11-27
SIDES, HN = ("left", "right"), ("L", "R")
FACE = lambda ax, sg_: f"{'+' if sg_ > 0 else '-'}{'xyz'[ax]}"  # noqa: E731


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lv_dir(level):
    return "m" + f"{level:.2f}".replace(".", "p")


def clips():
    """tag -> dict(step, level, object, sequence, frames, applicable) from the committed tables and the drivers."""
    out = {}
    for r in csv.DictReader(open(ROOT / "results_system1/T14_retargeting/step9g2/compare.csv")):
        app = [float(x) for x in r["applicable_levels_after"].split() if x != "无"]
        if app:
            out[r["clip"]] = dict(step="step9g2", applicable=app)
    fin = {}
    for r in csv.DictReader(open(ROOT / "results_system1/T14_retargeting/step9_final/levels.csv")):
        if r["applicable_C"] == "True":
            fin.setdefault(r["clip"], []).append(float(r["level"]))
    for k, v in fin.items():
        out[k] = dict(step="step9_final", applicable=sorted(v))
    for st in ("step9g2", "step9_final"):
        for s in json.loads((ROOT / f"logs_system1/t14/{st}/driver.json").read_text())["steps"]:
            tag = s["name"][6:]
            if s["name"].startswith("gates_") and tag in out and out[tag]["step"] == st:
                c = s["cmd"].split()
                g = lambda f: c[c.index(f) + 1]  # noqa: E731
                assert (g("--fb-turn"), g("--fb-legs"), g("--fb-walk-distance")) == ("world", "mirror", "pelvis_feet"), s["cmd"]
                a, b = map(int, g("--frames").split(":"))
                out[tag].update(object=g("--object"), sequence=g("--sequence"), frames=(a, b))
    for k in SKIP:
        out.pop(k, None)
    for v in out.values():
        v["level"] = 1.0 if 1.0 in v["applicable"] else min(v["applicable"])
    return out


def to_face(q, f, size):
    """unsigned distance of local point(s) q [N, 3] to the face rectangle f (e.g. "+x") of a centred box of full size `size`."""
    ax, sg_ = "xyz".index(f[1]), (1.0 if f[0] == "+" else -1.0)
    h = np.asarray(size, float) / 2
    p = np.clip(q, -h, h)
    p[:, ax] = sg_ * h[ax]
    return np.linalg.norm(q - p, axis=1)


def view_roles(Rw, q_faces, centre, pelvis):
    """per frame: role of the box face q_faces[t] = (axis, sign) as seen from `pelvis` facing the box (world): outward normal
    n = sign x Rw[t][:, axis]; |n_z| > 0.7 -> top / bottom; else with d = horizontal unit vector pelvis -> box centre:
    n.d < -0.7 -> near (the face towards the viewer), n.d > 0.7 -> far, otherwise own left / own right side by n.(z x d).
    Returns the roles and the per-frame max(|n_z|, |n.d|, |n.left|) (1 = face square to the viewer)."""
    out, sq = [], []
    for t, (ax, sg_) in enumerate(q_faces):
        n = sg_ * Rw[t][:, ax]
        n = n / np.linalg.norm(n)
        d = centre[t] - pelvis[t]
        d[2] = 0
        d = d / np.linalg.norm(d)
        left = np.cross([0, 0, 1.0], d)
        cz, cd, cl = n[2], n @ d, n @ left
        sq.append(max(abs(cz), abs(cd), abs(cl)))
        if abs(cz) > 0.7:
            out.append("top" if cz > 0 else "bottom")
        elif cd < -0.7:
            out.append("near end")
        elif cd > 0.7:
            out.append("far end")
        else:
            out.append("own left side" if cl > 0 else "own right side")
    return out, np.array(sq)


def measure(tag, spec):
    import mujoco
    import omomo_motion as om
    import frontback as fb
    import partner as pt
    from step9_diag import load_role0
    from palm_gap_check import nearest_face

    OBJECT, SEQ, FRAMES, LEVEL = spec["object"], spec["sequence"], tuple(spec["frames"]), spec["level"]
    SOLVE = ROOT / "logs_system1/t14" / spec["step"] / tag / lv_dir(LEVEL)
    checks = {}
    h0, names, nom, size, s = load_role0(OBJECT, SEQ, FRAMES)
    O = pt.mj_to_mat(nom)
    v = om.load_sequences({OBJECT})[SEQ]
    walk = fb.walking_segment(v, *FRAMES)
    C = fb.build(h0, names, O, size, LEVEL, turn="world", legs="mirror", walk=walk, walk_distance="pelvis_feet")
    T = len(O)
    w0, w1 = int(C["walk"]["w0"]), int(C["walk"]["w1"])
    tsize = np.array(size, float)
    tsize[C["k"]] = C["Lm"]
    recs = [json.loads((SOLVE / f"r{r}" / "run_record.json").read_text()) for r in (0, 1)]
    zq = [np.load(SOLVE / f"r{r}" / f"{tag}_new_qpos.npz") for r in (0, 1)]  # named by the clip tag
    checks["frames"] = T
    checks["solve_frames"] = [int(z["qpos"].shape[0]) for z in zq]
    assert all(n == T for n in checks["solve_frames"]), (T, checks["solve_frames"])
    checks["solve_person_r0_equals_rebuilt_h0_max_m"] = float(np.abs(zq[0]["human_joints"] - h0).max())
    checks["solve_person_r1_equals_rebuilt_human1_max_m"] = float(np.abs(zq[1]["human_joints"] - C["human1"]).max())
    checks["solve_P_equals_rebuilt_P_max"] = float(max(np.abs(zq[r]["P"] - C["P"]).max() for r in (0, 1)))
    checks["solve_object_qpos_equals_qpos_36_43_max"] = float(max(np.abs(zq[r]["object_qpos_mj"] - zq[r]["qpos"][:, 36:43]).max() for r in (0, 1)))
    checks["object_qpos_r0_vs_r1_max"] = float(np.abs(zq[0]["object_qpos_mj"] - zq[1]["object_qpos_mj"]).max())
    checks["target_urdf_size_m"] = recs[0]["target"]["size"]
    checks["shared_box_size_m"] = np.round(tsize, 6).tolist()
    checks["shared_box_size_equals_target_urdf"] = bool(np.allclose(tsize, recs[0]["target"]["size"], atol=1e-6))
    checks["own_box_size_m"] = np.round(size, 6).tolist()
    checks["carry_axis_k"] = "xyz"[int(C["k"])]
    checks["phases"] = dict(w0=w0, w1=w1, frames=T)

    # person palms (as palm_gap_check)
    F = pt.affine(np.diag([1.0, 1.0, -1.0]), np.zeros(3))
    persons = {0: h0, 1: C["human1"]}
    real_own = {0: O, 1: C["own1_real"]}
    Meff = {0: C["M0"], 1: C["M1"] @ F}
    Nr = {0: C["Ob"], 1: C["own1"] @ F}
    jp = {"L": names.index("L_Palm"), "R": names.index("R_Palm")}

    # face roles: carry axis k (end faces, the robot on each side), vertical axis, side axis
    Pp = pt.mj_to_mat(zq[0]["object_qpos_mj"].astype(np.float64))
    k = int(C["k"])
    zl = np.einsum("tji,j->ti", Pp[:, :3, :3], np.array([0, 0, 1.0]))  # world z in the box frame
    zmed = np.median(zl, 0)
    vert = int(np.argmax(np.abs(zmed)))
    checks["vertical_axis"] = dict(axis="xyz"[vert], median_world_z_in_box_frame=np.round(zmed, 4).tolist())
    assert vert != k, (vert, k)
    side = 3 - vert - k
    pel = {r: np.einsum("tji,tj->ti", Pp[:, :3, :3], zq[r]["qpos"][:, :3] - Pp[:, :3, 3]) for r in (0, 1)}
    end_r0 = float(np.sign(np.median(pel[0][:, k])))
    checks["robot_pelvis_box_local_median_m"] = {f"r{r}": np.round(np.median(pel[r], 0), 4).tolist() for r in (0, 1)}
    assert np.sign(np.median(pel[1][:, k])) == -end_r0, "robots on opposite ends of the carry axis"
    # left of robot 0 facing the box (world): z x (box centre - pelvis); side faces named by it
    c0 = Pp[:, :3, 3] - zq[0]["qpos"][:, :3]
    c0[:, 2] = 0
    left0 = np.cross(np.array([0, 0, 1.0]), c0)
    side_w = np.einsum("tij,j->ti", Pp[:, :3, :3], np.eye(3)[side])
    left_sign = float(np.sign(np.median(np.einsum("ti,ti->t", side_w, left0))))
    zsign = float(np.sign(zmed[vert]))

    def role(f):
        ax, sg_ = "xyz".index(f[1]), (1.0 if f[0] == "+" else -1.0)
        if ax == k:
            return "end towards r0" if sg_ == end_r0 else "end towards r1"
        if ax == vert:
            return "top" if sg_ == zsign else "bottom"
        return "side, left of r0" if sg_ == left_sign else "side, right of r0"
    roles = {FACE(a, b): role(FACE(a, b)) for a in range(3) for b in (1, -1)}

    def own_role(r, f):  # the same face as seen by robot r facing the box
        g = roles[f]
        if g.startswith("end"):
            return "near end" if g.endswith(f"r{r}") else "far end"
        if g.startswith("side"):
            left = g.endswith("left of r0") == (r == 0)
            return "own left side" if left else "own right side"
        return g

    rows, hands = [], {}
    for r in (0, 1):
        m = mujoco.MjModel.from_xml_path(str(SOLVE / f"r{r}" / "scene_sim.xml"))
        d = mujoco.MjData(m)
        assert m.nq == 43, m.nq
        rq, oq = zq[r]["qpos"][:, :36].astype(np.float64), zq[r]["object_qpos_mj"].astype(np.float64)
        Pr = pt.mj_to_mat(oq)
        offs = {s_: np.asarray(recs[r]["robot_palm_offsets_m"][f"{h}_Palm"], float) for s_, h in zip(SIDES, HN)}
        bid = {s_: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s_}_rubber_hand_link") for s_ in SIDES}
        gbox = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, OBJECT)
        gnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        ghull = {}
        for s_ in SIDES:
            cand = [g for g, n_ in enumerate(gnames) if n_.startswith(f"physx_{s_}_rubber_hand_link")]
            assert len(cand) == 1 and (m.geom_contype[gbox] & m.geom_conaffinity[cand[0]]), cand
            ghull[s_] = cand[0]
        assert np.allclose(m.geom_size[gbox] * 2, tsize, atol=1e-6), (m.geom_size[gbox] * 2, tsize)
        out = {s_: dict(palm=np.zeros((T, 3)), hull=np.zeros(T), wit=np.zeros((T, 3)), wit_hand=np.zeros(T)) for s_ in SIDES}
        ft = np.zeros(6)
        for t in range(T):
            d.qpos[:36] = rq[t]
            d.qpos[36:43] = oq[t]
            mujoco.mj_kinematics(m, d)
            for s_ in SIDES:
                Rl = d.xmat[bid[s_]].reshape(3, 3).copy()
                out[s_]["palm"][t] = d.xpos[bid[s_]] + Rl @ offs[s_]
                out[s_]["hull"][t] = mujoco.mj_geomDistance(m, d, ghull[s_], gbox, 0.5, ft)
                out[s_]["wit"][t] = (ft[3:6] - Pr[t, :3, 3]) @ Pr[t, :3, :3]
                out[s_]["wit_hand"][t] = np.linalg.norm(Rl.T @ (ft[0:3] - d.xpos[bid[s_]]) - offs[s_])
        anc = {w["hand"]: w for w in recs[r]["anchor_term"]["windows"]}
        for s_, h in zip(SIDES, HN):
            x = persons[r][:, jp[h]]
            pl = np.einsum("tji,tj->ti", Pr[:, :3, :3], out[s_]["palm"] - Pr[:, :3, 3])
            sd_r = fb.box_sd(pl, np.eye(4), tsize)
            sd_ro = np.array([fb.box_sd(x[t][None], real_own[r][t], size)[0] for t in range(T)])
            q_own = np.einsum("tji,tj->ti", real_own[r][:, :3, :3], x - real_own[r][:, :3, 3])
            im_loc = np.stack([(Meff[r] @ np.linalg.inv(Nr[r][t]) @ np.r_[x[t], 1.0])[:3] for t in range(T)])
            sd_im = fb.box_sd(im_loc, np.eye(4), tsize)
            fw = [FACE(a, b) for a, b in zip(*nearest_face(out[s_]["wit"], tsize))]
            fr = [FACE(a, b) for a, b in zip(*nearest_face(pl, tsize))]
            fi = [FACE(a, b) for a, b in zip(*nearest_face(im_loc, tsize))]
            fo = [FACE(a, b) for a, b in zip(*nearest_face(q_own, size))]
            nf_w = list(zip(*nearest_face(out[s_]["wit"], tsize)))
            rv, rsq = view_roles(Pr[:, :3, :3], nf_w, Pr[:, :3, 3], zq[r]["qpos"][:, :3])
            nf_o = list(zip(*nearest_face(q_own, size)))
            pv, psq = view_roles(real_own[r][:, :3, :3], nf_o, real_own[r][:, :3, 3], persons[r][:, names.index("Pelvis")])
            # person face on the real own box carried to the shared box by the construction's own map (the interaction-mesh map
            # Meff Nr^-1 applied to the real own box frame: own-box local direction -> shared-target local direction)
            fm, fm_al = [], []
            for t, (ax, sg_) in enumerate(nf_o):
                Rm = (Meff[r] @ np.linalg.inv(Nr[r][t]) @ real_own[r][t])[:3, :3]
                vv = Rm @ (sg_ * np.eye(3)[ax])
                j_ = int(np.argmax(np.abs(vv)))
                fm.append(FACE(j_, np.sign(vv[j_])))
                fm_al.append(abs(vv[j_]) / np.linalg.norm(vv))
            checks.setdefault("person_face_map_alignment_min", {})[f"r{r}_{h}"] = round(float(min(fm_al)), 4)
            # by label (the construction's correspondence): real own box face (a, s) = levelled own box face (a, s) (Ob keeps the
            # box axes: rule 3), carried to the shared target by the rotation part of M0 (role 0) / M1 (role 1, own1 frame)
            Lm_ = (C["M0"] if r == 0 else C["M1"])[:3, :3]
            fl = []
            for ax, sg_ in nf_o:
                vv = Lm_ @ (sg_ * np.eye(3)[ax])
                j_ = int(np.argmax(np.abs(vv)))
                fl.append(FACE(j_, np.sign(vv[j_])))
            checks.setdefault("label_map_rotation", {})[f"r{r}"] = np.round(Lm_, 4).tolist()
            checks.setdefault("robot_view_roles_equal_k_axis_roles_frames", {})[f"r{r}_{h}"] = int(sum(a1 == own_role(r, f1) for a1, f1 in zip(rv, fw)))
            w = anc.get(s_)
            A = np.asarray(w["anchor_local"], float) if w else None
            fa = FACE(*[z[0] for z in nearest_face(A[None], tsize)]) if w else None
            d_anchor = np.linalg.norm(pl - A, axis=1) if w else np.full(T, np.nan)
            src = h if r == 0 else ("R" if h == "L" else "L")
            mode = lambda L_: Counter(L_).most_common(1)[0][0]  # noqa: E731
            med = lambda a_: round(100 * float(np.median(a_)), 2)  # noqa: E731
            hands[f"r{r}_{h}"] = dict(
                person_hand=("demonstrator " if r == 0 else "mirror partner (demonstrator ") + f"{src}_Palm" + ("" if r == 0 else ", mirrored)"),
                touched_face_counts=dict(Counter(fw)), touched_face=mode(fw), touched_face_role=roles[mode(fw)], touched_face_own=own_role(r, mode(fw)),
                touched_face_share=round(Counter(fw)[mode(fw)] / T, 3),
                person_face_counts=dict(Counter(fi)), person_face=mode(fi), person_face_role=roles[mode(fi)], person_face_own=own_role(r, mode(fi)),
                person_face_share=round(Counter(fi)[mode(fi)] / T, 3),
                person_own_box_face_counts=dict(Counter(fo)), anchor_face=fa, anchor_face_role=roles.get(fa),
                robot_palm_nearest_face_counts=dict(Counter(fr)),
                person_face_label_counts=dict(Counter(fl)), person_face_label=mode(fl), person_face_label_own=own_role(r, mode(fl)),
                person_face_label_share=round(Counter(fl)[mode(fl)] / T, 3), match_label=mode(fw) == mode(fl),
                match_label_frames=int(sum(a1 == b1 for a1, b1 in zip(fw, fl))),
                robot_palm_to_person_face_label_median_cm=med(to_face(pl, mode(fl), tsize)),
                person_face_mapped_counts=dict(Counter(fm)), person_face_mapped=mode(fm), person_face_mapped_own=own_role(r, mode(fm)),
                person_face_mapped_share=round(Counter(fm)[mode(fm)] / T, 3), match_mapped=mode(fw) == mode(fm),
                match_mapped_frames=int(sum(a1 == b1 for a1, b1 in zip(fw, fm))),
                robot_palm_to_person_face_mapped_median_cm=med(to_face(pl, mode(fm), tsize)),
                robot_touched_role_counts=dict(Counter(rv)), robot_touched_role=mode(rv), robot_touched_role_share=round(Counter(rv)[mode(rv)] / T, 3),
                person_role_counts=dict(Counter(pv)), person_role=mode(pv), person_role_share=round(Counter(pv)[mode(pv)] / T, 3),
                role_match=mode(rv) == mode(pv), role_match_frames=int(sum(a1 == b1 for a1, b1 in zip(rv, pv))),
                view_squareness_min=dict(robot=round(float(rsq.min()), 3), person=round(float(psq.min()), 3)),
                match=mode(fw) == mode(fi), touched_in_person_faces=mode(fw) in Counter(fi),
                robot_palm_to_touched_face_median_cm=med(to_face(pl, mode(fw), tsize)), robot_palm_to_person_face_median_cm=med(to_face(pl, mode(fi), tsize)),
                person_palm_to_person_face_median_cm=med(to_face(im_loc, mode(fi), tsize)), person_palm_to_touched_face_median_cm=med(to_face(im_loc, mode(fw), tsize)),
                robot_palm_sd_median_cm=med(sd_r), robot_palm_sd_walk_median_cm=med(sd_r[w0:w1]),
                person_palm_sd_own_box_median_cm=med(sd_ro), person_palm_im_target_sd_median_cm=med(sd_im),
                hull_exact_sd_median_cm=med(out[s_]["hull"]), hull_contact_to_palm_point_median_cm=med(out[s_]["wit_hand"]),
                robot_palm_to_anchor_median_cm=med(d_anchor) if w else None)
            for t in range(T):
                rows.append(dict(frame=t, robot=r, hand=h, robot_touched_role=rv[t], person_role=pv[t], person_face_label=fl[t], person_face_mapped=fm[t],
                                 robot_palm_sd_shared_m=f"{sd_r[t]:.6f}", hull_exact_sd_m=f"{out[s_]['hull'][t]:.6f}",
                                 touched_face=fw[t], person_face_im=fi[t], person_own_box_face=fo[t], robot_palm_nearest_face=fr[t], anchor_face=fa,
                                 person_palm_sd_real_own_m=f"{sd_ro[t]:.6f}", im_target_sd_m=f"{sd_im[t]:.6f}",
                                 robot_palm_to_anchor_m=f"{d_anchor[t]:.6f}", hull_contact_to_palm_point_m=f"{out[s_]['wit_hand'][t]:.6f}"))
    return dict(tag=tag, **{k_: v_ for k_, v_ in spec.items()}, solve_dir=str(SOLVE.relative_to(ROOT)), face_roles=roles, checks=checks, hands=hands), rows


def check_vs_palm_gap(rows):
    """sub8_suitcase_020 1.00: the per-frame values equal palm_gap_check.py's palm_gap.csv."""
    ref = {(int(r_["frame"]), int(r_["robot"]), r_["hand"]): r_ for r_ in csv.DictReader(open(ROOT / "logs_system1/t14/frontback_train/palm_gap/palm_gap.csv"))}
    res = dict(rows=len(rows), ref_rows=len(ref), max_abs_m={}, face_mismatch={})
    for a, b in (("robot_palm_sd_shared_m", "robot_palm_sd_shared_m"), ("hull_exact_sd_m", "hull_exact_sd_m"),
                 ("person_palm_sd_real_own_m", "human_palm_sd_real_own_m"), ("im_target_sd_m", "im_target_sd_m"),
                 ("robot_palm_to_anchor_m", "robot_palm_to_anchor_m"), ("hull_contact_to_palm_point_m", "hull_contact_to_palm_point_m")):
        res["max_abs_m"][a] = max(abs(float(r_[a]) - float(ref[(r_["frame"], r_["robot"], r_["hand"])][b])) for r_ in rows)
    for a, b in (("touched_face", "hull_contact_face"), ("person_face_im", "im_target_nearest_face"), ("anchor_face", "anchor_face"),
                 ("robot_palm_nearest_face", "robot_palm_nearest_face")):
        res["face_mismatch"][a] = sum(r_[a] != ref[(r_["frame"], r_["robot"], r_["hand"])][b] for r_ in rows)
    res["passed"] = len(rows) == len(ref) and max(res["max_abs_m"].values()) < 2e-6 and not any(res["face_mismatch"].values())
    return res


def summary(spec_all):
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    tab, allj = [], {}
    for tag in spec_all:
        j = json.loads((OUT / f"{tag}.json").read_text())
        allj[tag] = j
        for key, hd in j["hands"].items():
            same = hd["person_face_label"] == hd["person_face_mapped"] == hd["person_face"]
            klass = ("match" if hd["touched_face"] == hd["person_face_label"] else "mismatch") if same else "person faces disagree"
            tab.append(dict(clip=tag, class_=klass, set="final test" if j["step"] == "step9_final" else "9g2 debug", object=j["object"], level=f"{j['level']:.2f}",
                            applicable_levels=" ".join(f"{x:.2f}" for x in j["applicable"]), robot=key[:2], hand=key[3], person_hand=hd["person_hand"],
                            touched_face_main=hd["touched_face"], touched_face_main_own=hd["touched_face_own"], touched_face_main_share=hd["touched_face_share"],
                            person_face_label=hd["person_face_label"], person_face_label_own=hd["person_face_label_own"],
                            person_face_label_share=hd["person_face_label_share"],
                            person_face_label_counts=" ".join(f"{k}:{v}" for k, v in sorted(hd["person_face_label_counts"].items())),
                            match_label=hd["match_label"], match_label_frames=hd["match_label_frames"],
                            robot_palm_to_person_face_label_median_cm=hd["robot_palm_to_person_face_label_median_cm"],
                            person_face_mapped=hd["person_face_mapped"], person_face_mapped_own=hd["person_face_mapped_own"],
                            person_face_mapped_share=hd["person_face_mapped_share"],
                            person_face_mapped_counts=" ".join(f"{k}:{v}" for k, v in sorted(hd["person_face_mapped_counts"].items())),
                            match_mapped=hd["match_mapped"], match_mapped_frames=hd["match_mapped_frames"],
                            robot_palm_to_person_face_mapped_median_cm=hd["robot_palm_to_person_face_mapped_median_cm"],
                            robot_touched_role=hd["robot_touched_role"], robot_touched_role_share=hd["robot_touched_role_share"],
                            robot_touched_role_counts=" ".join(f"{k}:{v}" for k, v in sorted(hd["robot_touched_role_counts"].items())),
                            person_role=hd["person_role"], person_role_share=hd["person_role_share"],
                            person_role_counts=" ".join(f"{k}:{v}" for k, v in sorted(hd["person_role_counts"].items())),
                            role_match=hd["role_match"], role_match_frames=hd["role_match_frames"],
                            touched_face=hd["touched_face"], touched_face_role=hd["touched_face_role"], touched_face_own=hd["touched_face_own"],
                            touched_face_share=hd["touched_face_share"],
                            touched_face_counts=" ".join(f"{k}:{v}" for k, v in sorted(hd["touched_face_counts"].items())),
                            person_face=hd["person_face"], person_face_role=hd["person_face_role"], person_face_own=hd["person_face_own"],
                            person_face_share=hd["person_face_share"],
                            person_face_counts=" ".join(f"{k}:{v}" for k, v in sorted(hd["person_face_counts"].items())),
                            person_own_box_face_counts=" ".join(f"{k}:{v}" for k, v in sorted(hd["person_own_box_face_counts"].items())),
                            anchor_face=hd["anchor_face"], match=hd["match"], touched_in_person_faces=hd["touched_in_person_faces"],
                            robot_palm_to_touched_face_median_cm=hd["robot_palm_to_touched_face_median_cm"], robot_palm_to_person_face_median_cm=hd["robot_palm_to_person_face_median_cm"],
                            person_palm_to_person_face_median_cm=hd["person_palm_to_person_face_median_cm"], person_palm_to_touched_face_median_cm=hd["person_palm_to_touched_face_median_cm"],
                            robot_palm_sd_median_cm=hd["robot_palm_sd_median_cm"], robot_palm_sd_walk_median_cm=hd["robot_palm_sd_walk_median_cm"],
                            person_palm_sd_own_box_median_cm=hd["person_palm_sd_own_box_median_cm"],
                            person_palm_im_target_sd_median_cm=hd["person_palm_im_target_sd_median_cm"],
                            hull_exact_sd_median_cm=hd["hull_exact_sd_median_cm"], hull_contact_to_palm_point_median_cm=hd["hull_contact_to_palm_point_median_cm"],
                            robot_palm_to_anchor_median_cm=hd["robot_palm_to_anchor_median_cm"], frames=j["checks"]["frames"],
                            carry_axis=j["checks"]["carry_axis_k"], vertical_axis=j["checks"]["vertical_axis"]["axis"]))
    p = OUT / "grasp_faces.csv"
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(tab[0]))
        wr.writeheader()
        wr.writerows(tab)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    js = dict(question="Kevin 2026-11-27: grasp faces of the frozen 9g2 construction (report only)", definitions=__doc__, skipped=list(SKIP),
              clips=allj, git_head=head, code_sha256={str(x.relative_to(ROOT)): sha(x) for x in (Path(__file__), ROOT / "system1/t14/frontback_train/palm_gap_check.py",
                                                                                                     OBJ / "frontback.py", OBJ / "step9_diag.py")},
              csv_sha256=sha(p))
    (OUT / "grasp_faces.json").write_text(json.dumps(js, indent=1, ensure_ascii=False) + "\n")
    import shutil
    for x in [p, OUT / "grasp_faces.json"] + [OUT / f"frames_{t}.csv" for t in spec_all]:
        shutil.copy2(x, RES / x.name)
    print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    spec_all = clips()
    if a.list:
        for k, v in spec_all.items():
            print(k, json.dumps(v))
        return 0
    if a.summary:
        summary(spec_all)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    res, rows = measure(a.clip, spec_all[a.clip])
    if a.clip == "sub8_suitcase_020" and res["level"] == 1.0:
        res["checks"]["equals_palm_gap_csv"] = check_vs_palm_gap(rows)
    res["command"] = "CUDA_VISIBLE_DEVICES=\"\" " + sys.executable + " -B " + " ".join(sys.argv)
    with open(OUT / f"frames_{a.clip}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (OUT / f"{a.clip}.json").write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (bool(x) if isinstance(x, np.bool_) else float(x))) + "\n")
    print(a.clip, json.dumps(res["checks"], default=str)[:600])
    for k_, hd in res["hands"].items():
        print(" ", k_, "robot", hd["robot_touched_role_counts"], "| person (own box)", hd["person_role_counts"], "| role match", hd["role_match"], "|",
              hd["touched_face"], hd["touched_face_own"], "| im", hd["person_face_own"], hd["person_face"], hd["person_face_counts"], "| match", hd["match"],
              "| robot sd", hd["robot_palm_sd_median_cm"], "person sd", hd["person_palm_sd_own_box_median_cm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
