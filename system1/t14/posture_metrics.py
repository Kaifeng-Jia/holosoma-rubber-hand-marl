#!/usr/bin/env python3
"""T14 section 11 item 4: posture of a single-robot solve compared with its (scaled) person, in the person's contact frames.

The person = the solve's own human input (saved human_joints: the person scaled by s and grounded, in the solve world).
Contact frames = the frames inside the solve's anchor contact windows (run_record anchor_term; any hand of that person).
  * torso tilt: angle between the pelvis -> neck line and the vertical. Robot: pelvis link origin -> back-of-neck point
    (torso_link (0.003, -0.0005, 0.290) m, the T13 neck point); person: Pelvis -> Neck joint;
  * pelvis height: robot pelvis link origin z; person Pelvis joint z (scaled; both worlds have the floor at z = 0);
  * elbow angle (0 = straight): angle between upper arm and forearm. Robot: shoulder_pitch_link -> elbow_link -> rubber_hand_link
    origins; person: Shoulder -> Elbow -> Wrist joints;
  * elbow position error: |(robot elbow - person elbow) - horizontal part of (robot pelvis - person pelvis)| (the global
    horizontal placement of the robot is removed; heights are compared as they are);
  * foot slide: robot toe = the front-most vertex of the ankle_roll_link PhysX shapes, horizontal speed (30 Hz differences);
    support frames by two rules: the solver's own (extract_foot_sticking_sequence_velocity: person toe moves <= 0.01 m per frame,
    i.e. <= 0.3 m/s) and T14 section 4 (person toe speed < 0.01 m/s); per rule the robot toe speed (mean, p95, max, share
    > 0.01 m/s) and the person's toe speed p95; plus the solver's per-frame foot xy / height deviation diagnostics.
Means are over contact frames; "max diff" = largest |robot - person| over them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import contact_stats as cs  # noqa: E402

NECK_POINT_TORSO = np.array([0.003, -0.0005, 0.290])


class FullFK:
    def __init__(self):
        self.r = cs.Robot("physx")
        shapes = cs.dump_shapes("table", "/World/envs/env_0/Robot/")[0]
        self.toe = {}
        for side in ("left", "right"):
            V = np.concatenate([m.vertices for b, m in shapes if b == f"{side}_ankle_roll_link"])
            self.toe[side] = V[np.argmax(V[:, 0])]

    def fk(self, q):
        """q [T, 36] wxyz -> {link: [T, 4, 4]} for every link."""
        r = self.r
        dof = {n: q[:, 7 + i] for i, n in enumerate(r.dofs)}
        T = {r.root: cs.pose_mat(q[:, :3], q[:, [4, 5, 6, 3]])}
        for j in r.joints:
            M = T[j.parent] @ np.broadcast_to(np.eye(4) if j.origin is None else j.origin, (len(q), 4, 4))
            if j.type in ("revolute", "continuous"):
                M = M @ cs.axis_angle(j.axis, dof[j.name])
            T[j.child] = M
        return T


def _ang(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.einsum("...i,...i->...", a, b), -1, 1)))


def analyse(qpos_npz, run_record, ffk, names, toe_names):
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
    z = np.load(qpos_npz)
    q, H = z["qpos"][:, :36].astype(np.float64), z["human_joints"].astype(np.float64)
    rec = json.loads(Path(run_record).read_text())
    T = len(q)
    f = np.zeros(T, bool)
    for w in (rec.get("anchor_term") or {}).get("windows", []):
        f[w["start"]:w["stop"]] = True
    hi = {n: names.index(n) for n in names}
    L = ffk.fk(q)
    p = lambda link: L[link][:, :3, 3]
    up = np.array([0.0, 0.0, 1.0])
    neck_r = L["torso_link"][:, :3, :3] @ NECK_POINT_TORSO + p("torso_link")
    tilt_r = _ang(neck_r - p("pelvis"), up)
    tilt_h = _ang(H[:, hi["Neck"]] - H[:, hi["Pelvis"]], up)
    pel_r, pel_h = p("pelvis")[:, 2], H[:, hi["Pelvis"], 2]
    stat = lambda r, h, scale=1.0: dict(robot_mean=round(scale * float(r[f].mean()), 2), person_mean=round(scale * float(h[f].mean()), 2),
                                        max_abs_diff=round(scale * float(np.abs(r - h)[f].max()), 2)) if f.any() else None
    out = dict(contact_frames=int(f.sum()), torso_tilt_deg=stat(tilt_r, tilt_h), pelvis_height_cm=stat(pel_r, pel_h, 100.0))
    hz = lambda x: x * np.array([1.0, 1.0, 0.0])
    for side, S in (("left", "L"), ("right", "R")):
        sh, el, wr = p(f"{side}_shoulder_pitch_link"), p(f"{side}_elbow_link"), p(f"{side}_rubber_hand_link")
        ang_r = _ang(el - sh, wr - el)
        ang_h = _ang(H[:, hi[f"{S}_Elbow"]] - H[:, hi[f"{S}_Shoulder"]], H[:, hi[f"{S}_Wrist"]] - H[:, hi[f"{S}_Elbow"]])
        out[f"elbow_angle_{side}_deg"] = stat(ang_r, ang_h)
        d = np.linalg.norm((el - H[:, hi[f"{S}_Elbow"]]) - hz(p("pelvis") - H[:, hi["Pelvis"]]), axis=1)
        out[f"elbow_position_error_{side}_cm"] = dict(mean=round(100 * float(d[f].mean()), 2), p95=round(100 * float(np.percentile(d[f], 95)), 2)) if f.any() else None
    stick = extract_foot_sticking_sequence_velocity(H, names, toe_names)  # keys are always L_Toe / R_Toe; threshold 0.01 m per frame
    for (side, key, tn) in zip(("left", "right"), ("L_Toe", "R_Toe"), toe_names):
        toe = np.einsum("tij,j->ti", L[f"{side}_ankle_roll_link"][:, :3, :3], ffk.toe[side]) + p(f"{side}_ankle_roll_link")
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(toe[:, :2], axis=0), axis=1) * 30.0
        vh = np.zeros(T)
        vh[1:] = np.linalg.norm(np.diff(H[:, hi[tn], :2], axis=0), axis=1) * 30.0
        sup_solver = np.array([bool(s_[key]) for s_ in stick])[:T]
        sup_t14 = vh < 0.01
        sup_solver[0] = sup_t14[0] = False
        res = {}
        for nm, sp in (("solver_rule_0p01m_per_frame", sup_solver), ("t14_rule_person_toe_below_0p01mps", sup_t14)):
            res[nm] = dict(support_frames=int(sp.sum()), robot_toe_speed_mean_mps=round(float(v[sp].mean()), 4) if sp.any() else None,
                           robot_toe_speed_p95_mps=round(float(np.percentile(v[sp], 95)), 4) if sp.any() else None,
                           robot_toe_speed_max_mps=round(float(v[sp].max()), 4) if sp.any() else None,
                           person_toe_speed_p95_mps=round(float(np.percentile(vh[sp], 95)), 4) if sp.any() else None,
                           robot_share_above_0p01mps=round(float((v[sp] > 0.01).mean()), 4) if sp.any() else None)
        out[f"foot_slide_{side}"] = res
    dest = next(Path(qpos_npz).parent.glob("*_new.npz"), None)
    if dest is not None:
        dz = np.load(dest)
        out["solver_foot_xy_deviation_max_cm"] = round(100 * float(dz["foot_xy_deviation_max_m"].max()), 3)
        out["solver_foot_height_deviation_max_cm"] = round(100 * float(dz["foot_height_deviation_max_m"].max()), 3)
    return out


def main():
    import argparse
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS, SMPLX_DEMO_JOINTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", required=True, help="name=<solve dir> (CORE4D: the level dir with person1 / person2; OMOMO: single_<task>_<V>)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ffk = FullFK()
    res = {}
    for spec in a.ref:
        name, d = spec.split("=", 1)
        d = Path(d)
        if (d / "person1").exists():
            res[name] = {f"r{p - 1}": analyse(d / f"person{p}/person{p}_new_qpos.npz", d / f"person{p}/run_record.json", ffk,
                                              list(SMPLX_DEMO_JOINTS) + ["_"] * 64, ["L_Foot", "R_Foot"]) for p in (1, 2)}
        else:
            task = d.name.split("_")[1]
            res[name] = {"r0": analyse(d / f"{task}_new_qpos.npz", d / "run_record.json", ffk, list(SMPLH_DEMO_JOINTS) + ["_"] * 64, ["L_Toe", "R_Toe"])}
        print(name, json.dumps(res[name]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(a.out.read_text())["refs"] if a.out.exists() else {}
    a.out.write_text(json.dumps(dict(rules=__doc__, refs=old | res), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
