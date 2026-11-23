#!/usr/bin/env python3
"""T14 section 4 metrics for Table references (30 Hz compact pairs). Read only, CPU.

Geometry = the simulator's: robot links = the PhysX-cooked shapes of the T10 Isaac dump (T12-v2 Robot('physx')); object =
the training Table (desk001_5kg_training.urdf, five boxes = desk001 proxy x 0.7412, exact union-of-boxes signed distance,
positive inside). Robot r <-> CORE4D person r+1 (robot 0 = person1). Source 15 Hz frame t <-> reference 30 Hz frame 2t.

* palm signed distance: per robot and hand, the hand link's largest signed distance (exact, no culling).
* contact retention (CoCarry definition as written in T14 section 4): condition frames = source frames where that person's
  hand is within 0.10 m of the source object's surface (any SMPL-X hand vertex, raw desk001 mesh, stop-1 surface-sample
  signed distance) and the source object moves faster than 0.03 m/s (object origin, central difference, 15 Hz); retained
  when the robot hand's signed distance >= -0.02 m. Per robot / hand and pooled.
* penetration: a frame penetrates when any link of the robot has signed distance > 0 (also reported with OmniRetarget's
  1 cm tolerance, > 0.01 m); share of frames and the largest depth; hands and other links separately.
* foot slip: robot toe = left/right_ankle_roll_sphere_5_link of the retargeting MJCF (OmniRetarget's evaluator), human toe
  = SMPL-X L_Foot / R_Foot of the canonical sequence resampled to 30 Hz (the solve input); horizontal displacement per
  frame. Two definitions: (T14 text) support = human toe speed < 0.01 m/s, slide = robot toe speed > 0.01 m/s;
  (OmniRetarget code, eval_retargeting.py) support = human toe moves < 0.01 m per frame, slide = robot toe moves > 0.01 m
  per frame. Share of support frames with a slide, and the largest robot toe speed in a sliding frame (m/s).
* robot-robot clearance: smallest distance between the two robots' link sample points (every 3rd frame).
* T12-diag phases (reference_hand_table_distance_per_frame.csv, 50 Hz labels mapped by time): per robot, the closer hand's
  signed distance in lift / push / carry / lower: median, share within 1.5 cm, share penetrating.

    sonic_s1/bin/python -B system1/t14/eval_table_refs.py --ref current=... --ref new=... --out results_system1/T14_retargeting/step1_table
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import contact_stats as cs  # noqa: E402
import contact_parts as cp  # noqa: E402

OBJ_URDF = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf"
MJCF = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
SEQ = Path("/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions/20231030/001")
CANON = Path("/home/kevin/datasets/CORE4D-V1-full/derived/20231030_001_desk001_v1_canonical.npz")
PHASES = ROOT / "results_system1/T12_diag/reference_hand_table_distance_per_frame.csv"
ACTIVE = ("lift", "push", "carry", "lower")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:], q[..., :1]], -1)


def hand_sd(robot, obj, Tl, To_inv, link):
    M = To_inv @ Tl[link]
    p = robot.points[link]
    q = np.einsum("...ij,nj->...ni", M[..., :3, :3], p) + M[..., None, :3, 3]
    s = obj.sdf(q.reshape(-1, 3)).reshape(q.shape[:-1])
    return s.max(-1)


def human_hand_sd():
    """[15 Hz frames, person, hand] largest signed distance of the SMPL-X hand vertices to the raw desk001 mesh (m)."""
    hv = cp.hand_vertex_ids()
    To = np.load(SEQ / "smooth_objposes.npy")
    o = cp.obj("core4d_desk001")
    out = np.full((len(To), 2, 2), -np.inf)
    for pi in (1, 2):
        V = np.load(SEQ / f"person{pi}_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"].astype(np.float64)
        for hi, side in enumerate(("left", "right")):
            for t in range(min(len(V), len(To))):
                R, x = To[t, :3, :3], To[t, :3, 3]
                pl = (V[t, hv[side]] - x) @ R
                d, j = o["tree"].query(pl)
                sgn = -np.sign(np.einsum("ij,ij->i", pl - o["p"][j], o["n"][j]))
                out[t, pi - 1, hi] = (sgn * d).max()
    speed = np.linalg.norm(np.gradient(To[:, :3, 3], axis=0), axis=1) * 15.0
    return out, speed


def toe_positions(qpos):
    """[T, 2 robots, 2 toes, 3] from the retargeting MJCF (robot only: free joint + 29 joints = 36 qpos)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    assert m.nq == 36, m.nq
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_sphere_5_link") for s in ("left", "right")]
    out = np.zeros(qpos.shape[:2] + (2, 3))
    for t in range(qpos.shape[0]):
        for r in range(2):
            d.qpos[:] = qpos[t, r]
            mujoco.mj_kinematics(m, d)
            out[t, r] = d.xpos[ids]
    return out


def human_toes_30hz():
    from holosoma_retargeting.data_utils.core4d_adapter import load_canonical_core4d_sequence, resample_core4d_pair_sequence
    seq = resample_core4d_pair_sequence(load_canonical_core4d_sequence(CANON), 30)
    return seq.human_joints[:, :, [10, 11]]  # L_Foot, R_Foot (SMPLX_DEMO_JOINTS)


def slip(robot_toe, human_toe, fps=30):
    """robot_toe / human_toe [T, 2 feet, 3]; returns both definitions."""
    rv = np.concatenate([[np.zeros(2)], np.linalg.norm(np.diff(robot_toe[..., :2], axis=0), axis=-1)])  # m per frame
    hv = np.concatenate([[np.full(2, np.inf)], np.linalg.norm(np.diff(human_toe[..., :2], axis=0), axis=-1)])
    out = {}
    for name, h_thr, r_thr in (("t14_text_mps", 0.01 / fps, 0.01 / fps), ("omniretarget_m_per_frame", 0.01, 0.01)):
        sup = hv < h_thr
        sl = sup & (rv > r_thr)
        n = int(sup.any(axis=1).sum())
        out[name] = dict(support_frames=n, sliding_share=round(float(sl.any(axis=1).sum() / n), 4) if n else None,
                         max_sliding_toe_speed_mps=round(float((rv * sl).max() * fps), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", required=True, help="name=path to a compact 30 Hz pair npz")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    robot = cs.Robot("physx")
    obj = cs.Object(OBJ_URDF, workers=1, geometry="physx", task="table")
    h_sd, obj_speed = human_hand_sd()
    htoe = human_toes_30hz()
    phase50 = np.array([r["phase"] for r in csv.DictReader(PHASES.open())])
    summary, per_frame = {}, {}
    for spec in a.ref:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        q = z["robot_qpos"].astype(np.float64)
        o = z["object_qpos"].astype(np.float64)
        T = len(q)
        root_pos, root_quat = q[:, :, :3], wxyz_to_xyzw(q[:, :, 3:7])
        dof = q[:, :, 7:]
        obj_pos, obj_quat = o[:, :3], wxyz_to_xyzw(o[:, 3:7])
        Tl = robot.fk(root_pos, root_quat, dof)
        To_inv = np.linalg.inv(cs.pose_mat(obj_pos, obj_quat))[:, None]
        H = np.stack([hand_sd(robot, obj, Tl, To_inv, l) for l in cs.HANDS], -1)  # [T, 2, 2]
        D = cs.link_distances(robot, obj, root_pos, root_quat, dof, obj_pos, obj_quat)  # [T, 2, links], culled
        nonhand = D[:, :, 2:].max(-1)
        allmax = np.maximum(H.max(-1), nonhand)
        rec = dict(path=str(path), sha256=sha(path), frames=T)
        # contact retention
        t15 = np.arange(min(len(h_sd), (T + 1) // 2))
        cond = (h_sd[t15] >= -0.10) & (obj_speed[t15, None, None] > 0.03)
        robo = H[2 * t15] >= -0.02
        ret = {}
        for r in range(2):
            for hi, side in enumerate(("left", "right")):
                c = cond[:, r, hi]
                ret[f"r{r}_{side}"] = dict(condition_frames=int(c.sum()), retention=round(float(robo[c, r, hi].mean()), 4) if c.any() else None)
            c = cond[:, r]
            ret[f"r{r}_both_hands_pooled"] = round(float(robo[:, r][c].mean()), 4) if c.any() else None
        ret["pooled"] = round(float(robo[cond].mean()), 4)
        rec["contact_retention"] = ret
        # penetration
        pen = {}
        for r in range(2):
            pen[f"r{r}"] = dict(frames_any_link_gt0=round(float((allmax[:, r] > 0).mean()), 4), frames_any_link_gt1cm=round(float((allmax[:, r] > 0.01).mean()), 4),
                                max_depth_cm=round(100 * float(max(allmax[:, r].max(), 0)), 2), hand_max_depth_cm=round(100 * float(max(H[:, r].max(), 0)), 2),
                                nonhand_max_depth_cm=round(100 * float(max(nonhand[:, r].max(), 0)), 2), frames_hand_gt0=round(float((H[:, r].max(-1) > 0).mean()), 4))
        rec["penetration"] = pen
        # foot slip
        toes = toe_positions(q)
        rec["foot_slip"] = {f"r{r}": slip(toes[:, r], htoe[:T, r]) for r in range(2)}
        rec["foot_min_height_cm"] = {f"r{r}": round(100 * float(toes[:, r, :, 2].min()), 2) for r in range(2)}
        # robot-robot clearance
        cl = np.inf
        for t in range(0, T, 3):
            P = [np.concatenate([(Tl[l][t, r, :3, :3] @ robot.points[l].T).T + Tl[l][t, r, :3, 3] for l in robot.links]) for r in range(2)]
            cl = min(cl, float(cKDTree(P[0]).query(P[1])[0].min()))
        rec["robot_robot_min_clearance_cm"] = round(100 * cl, 2)
        # T12-diag phases
        ph = phase50[np.clip(np.rint(np.arange(T) / 30 * 50).astype(int), 0, len(phase50) - 1)]
        phs = {}
        for r in range(2):
            best = H[:, r].max(-1)
            for p in ACTIVE + ("still",):
                m = ph == p
                phs[f"r{r}_{p}"] = dict(frames=int(m.sum()), median_cm=round(100 * float(np.median(best[m])), 2), within_1p5cm=round(float((best[m] >= -0.015).mean()), 3),
                                        penetrating=round(float((best[m] > 0).mean()), 3))
        rec["phases_closer_hand"] = phs
        rec["generation"] = dict(finite=bool(np.isfinite(q).all()), object_track_equals_current=None)
        summary[name] = rec
        per_frame[name] = dict(H=H, phase=ph)
        print(name, json.dumps({k: rec[k] for k in ("contact_retention", "penetration", "foot_slip", "robot_robot_min_clearance_cm")}), flush=True)
    # object tracks identical?
    objs = {s.split("=", 1)[0]: np.load(s.split("=", 1)[1])["object_qpos"] for s in a.ref}
    base = next(iter(objs.values()))
    for k, v in objs.items():
        summary[k]["generation"]["object_track_equals_current"] = float(np.abs(v - base).max())
    (a.out / "metrics.json").write_text(json.dumps(dict(rules=__doc__, object_urdf=str(OBJ_URDF.relative_to(ROOT)), object_urdf_sha256=sha(OBJ_URDF),
                                                       physx_dump_scene_sha256=robot.dump_scene_sha256, refs=summary), indent=1) + "\n")
    names = list(per_frame)
    T = min(len(v["H"]) for v in per_frame.values())
    with (a.out / "per_frame_palm_distance.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_30hz", "t_s", "phase"] + [f"{n}_r{r}_{s}_cm" for n in names for r in range(2) for s in ("left", "right")])
        for t in range(T):
            w.writerow([t, round(t / 30, 4), per_frame[names[0]]["phase"][t]] + [round(100 * float(per_frame[n]["H"][t, r, h]), 2) for n in names for r in range(2) for h in range(2)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
