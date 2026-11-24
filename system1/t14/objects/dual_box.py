#!/usr/bin/env python3
"""T14 step 7, layer 1: single robot -> two robots on the lengthened box (role isometry of role_check_box.py), levels m,
metrics as round 6. Read only, CPU.

Levels: 1.0 x = the role check's target (the box twice as long along the selected axis k: two copies end to end); m x = the box
m x (2 w) long along k, everything else unchanged; each role keeps its region at its end: role r's map (Q_r, c_r) of the role
check, with c_r moved by -/+ (L(m) - L(1)) / 2 along k (role 0 / role 1). Exact for every m (rigid maps of the solve).
Lower bound a (T14 section 7 item 5, Kevin 2026-11-24 upper limit 1.5): from m = 1.0 down in 0.05 steps (not below 0.55: at
0.5 the two regions coincide), the last level before the first one at which the two robots' shapes intersect in any frame;
reported levels = a, 1.0, 1.25, 1.5. When 1.0 already intersects, a is undefined; then, as extra information only, the
first level above 1.0 (0.05 steps, up to 3.0) without intersection is searched and reported.
Per level (geometry = the simulator's: robot = PhysX link shapes of the T10 dump, exact convex distances; object = the box):
  * contact retention (CoCarry): source frames where the person's hand is within 0.10 m of the captured mesh and the object
    moves faster than 0.03 m/s; retained when the robot hand's signed distance >= -0.02 m; robot 1 inherits the source labels
    (proper maps: left stays left); pooled = mean of the two robots;
  * max penetration (any link) per robot, share of frames penetrating; hand penetration;
  * robot-robot: smallest exact distance between the two robots' shapes, frames intersecting;
  * 3-D contact-position error (contact_error3d.py's definition): frames where the person's hand touches the object (hand
    vertex within 1.5 cm); target = the person's palm point -> nearest point of the captured mesh in the box frame (real
    size; the robot's box is the captured object's oriented box), moved by the role's map; n = that face normal toward the
    palm; e = robot palm point - target: |e|, signed normal component (+ = outside / hovering), tangential component;
  * hand metrics (hand_metrics.py): palm normal / finger direction vs the person's (object frame, person's directions moved
    by Q_r), palm vs inward box-surface normal, wrist near-limit share and peak speed;
  * foot slide per robot (posture_metrics.py's rule, T14 section 4): support frames = the solve person's toe speed
    < 0.01 m/s after Gaussian sigma 8; robot toe = front-most ankle_roll PhysX vertex; largest horizontal drift from the
    segment start; share of support frames with toe speed > 0.01 m/s (robot 1 differs from robot 0 when the object turns:
    the role map is fixed in the object frame).

    sonic_s1/bin/python -B system1/t14/objects/dual_box.py --solve <solve dir> --role <role_check.json> --out <dir> --refs-out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(HERE))
import contact_gap as cg  # noqa: E402
import dual_stretch as ds  # noqa: E402
import eval_core4d_hulls as eh  # noqa: E402
import hand_metrics as hm  # noqa: E402
from reference_geometry_math import map_robot  # noqa: E402
from reference_geometry_pilot import G1_29DOF_JOINT_NAMES, g1_sagittal_joint_map  # noqa: E402
from role_check_box import box_urdf  # noqa: E402

LEVELS_REPORT = (1.0, 1.25, 1.5)
FFK = None  # posture_metrics.FullFK (robot toe = front-most ankle_roll PhysX vertex), set in main


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x, float)
    return dict(frames=int(len(x)), mean_cm=round(100 * float(x.mean()), 2), p95_cm=round(100 * float(np.percentile(x, 95)), 2)) if len(x) else None


class Human:
    """person's palm targets / directions in the box frame (real size) and contact flags, from the solve's inputs."""

    def __init__(self, solve, rec):
        import box_assets as ba
        import omomo_motion as om
        import single_shared as ss
        import trimesh
        from scipy.spatial.transform import Rotation
        seq, name = rec["sequence"], rec["object"]
        v = om.load_sequences({name})[seq]
        palms, fit = ss.omomo_palms(seq)
        J, tips = ss.omomo_palms.last["joints_omomo_new"], ss.omomo_palms.last["tips_omomo_new"]
        bf = ba.box_frame(name, v)
        poses, _ = ba.poses_omomo_new(v, bf, fit)
        R = Rotation.from_quat(poses[:, [1, 2, 3, 0]]).as_matrix()
        local = np.einsum("tji,thj->thi", R, palms - poses[:, None, 4:7])
        mesh = ba.mesh_in_box_frame(name, bf)
        T = len(local)
        self.target, self.normal = np.zeros((T, 2, 3)), np.zeros((T, 2, 3))
        for h in (0, 1):
            cp_, _, tri = trimesh.proximity.closest_point(mesh, local[:, h])
            n = mesh.face_normals[tri]
            n = n * np.where(np.einsum("ij,ij->i", local[:, h] - cp_, n) < 0, -1.0, 1.0)[:, None]
            self.target[:, h], self.normal[:, h] = cp_, n
        z = np.load(solve / "human_contact.npz")
        self.h_sd, self.speed = z["h_sd"], z["object_speed"]
        self.flags = self.h_sd >= -0.015
        tf = hm.hl.template_frames()
        self.dirs = {k: np.zeros((T, 2, 3)) for k in ("palm_normal", "finger", "long_axis")}
        for hi, side in enumerate(("left", "right")):
            V = np.zeros((T, max(hm.hl.TIP_VERTEX.values()) + 1, 3))
            V[:, hm.hl.TIP_VERTEX[side]] = tips[side]
            d = hm.hl.human_directions(J, V, side, tf)
            for k in self.dirs:
                self.dirs[k][:, hi] = np.einsum("tji,tj->ti", R, d[k])
        self.palm_fit_residual_max_m = fit["fit_residual_max_m"]
        import support
        from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
        hj = np.load(solve / f"{rec['tag']}_new_qpos.npz")["human_joints"]
        names = list(SMPLH_DEMO_JOINTS)
        self.support = support.support_flags(hj[:, [names.index("L_Toe"), names.index("R_Toe")]])  # the solve's person (posture_metrics rule)


def level(ctx, m, out_dir):
    robot, pc, rh, H, q, obj, roles, axis, size, name, index, sign = ctx
    L1 = 2 * size[axis]
    Lm = m * L1
    tag = f"m{m:.2f}".replace(".", "p")
    tsize = np.array(size, float)
    tsize[axis] = Lm
    urdf = box_urdf(out_dir / tag / f"{name}_{tag}.urdf", name, tsize)
    e = np.eye(3)[axis]
    cs_ = [np.asarray(r["c"], float) + sgn * (Lm - L1) / 2 * e for r, sgn in zip(roles, (-1, 1))]
    Qs = [np.asarray(r["Q"], float) for r in roles]
    pair = np.stack([map_robot(q, obj, Q, c, 1, index, sign) for Q, c in zip(Qs, cs_)], 1)
    ev = eh.ObjectRobotDistance(eh.urdf_box_meshes(urdf), robot, depth="exact")
    Hd, P = ev.frames(pair, obj)
    n = min(len(pair), len(H.h_sd))
    cond = (H.h_sd[:n] >= -0.10) & (H.speed[:n, None] > 0.03)
    ret = {}
    for r in (0, 1):
        ok = Hd[:n, r] >= -0.02
        ret[f"r{r}"] = round(float(ok[cond].mean()), 4) if cond.any() else None
    ret["pooled"] = round(float(np.mean([ret["r0"], ret["r1"]])), 4) if cond.any() else None
    rr = pc.min_distance(pair)
    # 3-D contact-position error
    Pl = cg.robot_palms_local(robot, pair[:n], obj[:n], cg.palm_offsets())  # [n, 2 robots, 2 hands, 3]
    ce = {}
    for r in (0, 1):
        tgt = H.target[:n] @ Qs[r].T + cs_[r]
        nrm = H.normal[:n] @ Qs[r].T
        err = Pl[:, r] - tgt
        nc = np.einsum("thi,thi->th", err, nrm)
        tang = np.linalg.norm(err - nc[..., None] * nrm, axis=-1)
        dist = np.linalg.norm(err, axis=-1)
        for hi, side in enumerate(("left", "right")):
            f = H.flags[:n, hi]
            ce[f"r{r}_{side}"] = dict(distance=stats(dist[f, hi]), tangential=stats(tang[f, hi]), normal_abs=stats(np.abs(nc[f, hi])),
                                      normal_signed_mean_cm=round(100 * float(nc[f, hi].mean()), 2) if f.any() else None)
        f = H.flags[:n]
        ce[f"r{r}"] = dict(distance=stats(dist[f]), normal_signed_mean_cm=round(100 * float(nc[f].mean()), 2) if f.any() else None)
    # hand metrics
    from object_surface import UnionSurface
    rd = rh.directions_local(pair[:n], obj[:n])
    _, _, nr = UnionSurface(eh.urdf_box_meshes(urdf)).nearest(Pl, normals=True)
    psa = hm._ang(rd["palm_normal"], -nr)
    hd = {k: np.stack([H.dirs[k][:n] @ Qs[r].T for r in (0, 1)], 1) for k in H.dirs}
    fl = np.stack([H.flags[:n], H.flags[:n]], 1)
    hand = hm.summarize(rd, hd, fl, psa, rh.wrist_stats(pair))
    foot = {}
    for r in (0, 1):
        L_ = FFK.fk(pair[:, r])
        for si, side in enumerate(("left", "right")):
            toe = np.einsum("tij,j->ti", L_[f"{side}_ankle_roll_link"][:, :3, :3], FFK.toe[side]) + L_[f"{side}_ankle_roll_link"][:, :3, 3]
            v = np.zeros(len(toe))
            v[1:] = np.linalg.norm(np.diff(toe[:, :2], axis=0), axis=1) * 30.0
            sp = H.support[:len(toe), si]
            f_ = np.concatenate([[False], sp, [False]])
            e_ = np.flatnonzero(np.diff(f_.astype(int)))
            drift = [float(np.linalg.norm(toe[a0:b0, :2] - toe[a0, :2], axis=1).max()) for a0, b0 in zip(e_[0::2], e_[1::2])]
            foot[f"r{r}_{side}"] = dict(segments=len(drift), drift_max_cm=round(100 * max(drift), 3) if drift else None,
                                        share_above_0p01mps=round(float((v[sp] > 0.01).mean()), 4) if sp.any() else None,
                                        toe_speed_max_mps=round(float(v[sp].max()), 4) if sp.any() else None)
    p = out_dir / tag / f"{name}_{tag}_pair_fps30.npz"
    np.savez(p, robot_qpos=pair, object_qpos=obj, fps=np.int64(30), m=m, length_m=Lm, axis=axis, Q=np.stack(Qs), c=np.stack(cs_), box_size_m=tsize)
    return dict(m=m, box_size_m=np.round(tsize, 5).tolist(), length_along_axis_m=round(float(Lm), 5), frames=len(pair), condition_frames=int(cond.sum()),
                contact_retention=ret, max_penetration_cm={f"r{r}": round(100 * float(max(P[:, r].max(), 0.0)), 3) for r in (0, 1)},
                frames_penetrating={f"r{r}": round(float((P[:, r] > 0).mean()), 4) for r in (0, 1)},
                hand_max_penetration_cm={f"r{r}": round(100 * float(max(Hd[:, r].max(), 0.0)), 3) for r in (0, 1)},
                robot_robot_min_distance_cm=round(100 * float(rr.min()), 3), robot_robot_intersect_frames=int((rr < 0).sum()),
                first_intersect_frame=int(np.argmax(rr < 0)) if (rr < 0).any() else None, contact_position_error_3d=ce, hand=hand, foot_slide=foot,
                finite=bool(np.isfinite(pair).all()), urdf=str(urdf), urdf_sha256=sha(urdf), pair=str(p), pair_sha256=sha(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", type=Path, required=True)
    ap.add_argument("--role", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs-out", type=Path, default=None, help="copy the reported levels' pair npz and urdf here")
    a = ap.parse_args()
    rec = json.loads((a.solve / "run_record.json").read_text())
    role = json.loads(a.role.read_text())
    name, tag = rec["object"], rec["tag"]
    if role["selected_roles"] is None:
        raise SystemExit("no role pair")
    axis = int(role["selected"]["roles"][0].split("/")[0][4:])
    z = np.load(a.solve / f"{tag}_new_qpos.npz")
    qq = z["qpos"].astype(np.float64)
    q, obj = qq[:, :36], qq[:, 36:43]
    size = np.array(role["box_size_m"], float)
    global FFK
    import posture_metrics as pm
    FFK = pm.FullFK()
    robot = eh.cs.Robot("physx")
    pc = ds.PairCollision(robot, depth="exact")
    rh = hm.RobotHands()
    H = Human(a.solve, rec)
    index, sign = g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES)
    ctx = (robot, pc, rh, H, q, obj, role["selected_roles"], axis, size, name, index, sign)
    a.out.mkdir(parents=True, exist_ok=True)
    levels, search = {}, []
    m = 1.0
    while m > 0.549:
        mm = round(m, 2)
        levels[mm] = level(ctx, mm, a.out)
        search.append(dict(m=mm, robot_robot_min_distance_cm=levels[mm]["robot_robot_min_distance_cm"], intersect_frames=levels[mm]["robot_robot_intersect_frames"]))
        print("search", json.dumps(search[-1]), flush=True)
        if levels[mm]["robot_robot_intersect_frames"]:
            break
        m -= 0.05
    up = []
    if search[0]["intersect_frames"]:
        a_level, a_note = None, "1.0 x already intersects"
        m = 1.05
        while m < 3.001:  # extra information only: the first level above 1.0 without intersection (0.05 steps, up to 3.0)
            mm = round(m, 2)
            r_ = levels[mm] = level(ctx, mm, a.out)
            up.append(dict(m=mm, robot_robot_min_distance_cm=r_["robot_robot_min_distance_cm"], intersect_frames=r_["robot_robot_intersect_frames"]))
            print("search up", json.dumps(up[-1]), flush=True)
            if not r_["robot_robot_intersect_frames"]:
                break
            m += 0.05
    elif search[-1]["intersect_frames"]:
        a_level, a_note = search[-2]["m"], "last level before the first intersecting one"
    else:
        a_level, a_note = search[-1]["m"], "no intersection down to 0.55 (search floor); a = 0.55"
    for mm in LEVELS_REPORT:
        if mm not in levels:
            levels[mm] = level(ctx, mm, a.out)
    report = ([a_level] if a_level is not None else []) + list(LEVELS_REPORT)
    if a.refs_out is not None:
        import shutil
        for mm in report:
            d = a.refs_out / f"{tag}_m{mm:.2f}".replace(".", "p")
            d.mkdir(parents=True, exist_ok=True)
            for k in ("pair", "urdf"):
                shutil.copy(levels[mm][k], d / Path(levels[mm][k]).name)
    out = dict(object=name, tag=tag, sequence=rec["sequence"], solve=str(a.solve), role_check=str(a.role), role_status=role["status"], axis=axis,
               source_box_m=size.tolist(), a=a_level, a_note=a_note, search=search, search_up_when_1_intersects=up,
               first_free_level_above_1=next((x["m"] for x in up if not x["intersect_frames"]), None), report_levels=report,
               levels={f"{k:.2f}": v for k, v in sorted(levels.items())}, person_palm_fit_residual_max_m=H.palm_fit_residual_max_m,
               physx_dump_scene_sha256=robot.dump_scene_sha256, rules=__doc__, command=" ".join(sys.argv))
    (a.out / "levels.json").write_text(json.dumps(out, indent=1) + "\n")
    print(name, tag, "a", a_level, json.dumps({k: (v["contact_retention"]["pooled"], v["robot_robot_min_distance_cm"]) for k, v in out["levels"].items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
