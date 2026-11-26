#!/usr/bin/env python3
"""T14 step 9g item 2 (Kevin 2026-11-26): per clip, two quantities of the mirror construction, report only (the construction is
not changed):
  * leg stretch in the walking segment: the partner's two-segment leg IK (frontback.leg_ik) overextension, i.e. how much the
    mirrored hip-to-reversed-ankle distance exceeds thigh + shank (the demonstration's own bone lengths); max per leg and over
    both legs, overextended frames (frontback diag9g leg_ik, walk segment);
  * single-frame foot displacement at the start of the reversed walk: the partner person's feet (ankles and toes, solver scale,
    the largest of the four points) between consecutive frames; "start" = the first 15 frames (0.5 s at 30 fps) of the partner's
    walking segment, steps t -> t+1 for t in [w0, w0 + 15); also the largest step over the whole walking segment (and its frame
    offset from w0), over the whole clip, the demonstration's largest step over its walking segment, and the source time covered
    by the partner's largest start step (tau(t+1) - tau(t), source frames);
  * the solved partner robot at level 1.0 when it was solved: the same step on its ankle-roll links (solver scale) over the same
    start frames (MuJoCo forward kinematics on the solve's own scene).
The construction is rebuilt exactly as step9_gates.py does (load_role0, walking_segment, frontback.build at m = 1.0 with the
driver's --fb-turn / --fb-legs) and checked against the level's gates_m1p00.json (leg IK diagnostics equal). Read only.

    hsretargeting_s1/bin/python -B system1/t14/objects/step9g_feet_ik_table.py --logdir logs_system1/t14/step9g --out <csv>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
START = 15


def robot_feet(level):
    import mujoco
    rec = json.loads((level / "r1/run_record.json").read_text())
    z = np.load(level / f"r1/{rec['tag']}_new_qpos.npz")["qpos"]
    m = mujoco.MjModel.from_xml_path(rec["scene"])
    d = mujoco.MjData(m)
    b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    P = np.zeros((len(z), 2, 3))
    for t in range(len(z)):
        d.qpos[:] = z[t, :m.nq]
        mujoco.mj_kinematics(m, d)
        P[t] = d.xpos[b]
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--driver", default="driver.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L = a.logdir if a.logdir.is_absolute() else ROOT / a.logdir
    import frontback as fb
    import omomo_motion as om
    import partner as pt
    from step9_diag import load_role0
    drv = json.loads((L / a.driver).read_text())
    turn, legs = drv["fb_args"][1], drv["fb_args"][3]
    assert legs == "mirror", drv["fb_args"]
    rows = []

    def mx(*v):  # a segment with no frames (e.g. no lift before the walk) has max_excess_m None
        v = [x for x in v if x is not None]
        return max(v) if v else None
    for clip, sm in drv["summary"].items():
        fr = tuple(int(x) for x in sm["frames"].split(":"))
        h0, names, nom, size, s = load_role0(sm["object"], sm["sequence"], fr)
        walk = fb.walking_segment(om.load_sequences({sm["object"]})[sm["sequence"]], *fr)
        C = fb.build(h0, names, pt.mj_to_mat(nom), size, 1.0, turn=turn, legs=legs, walk=walk)
        g = json.loads((L / clip / "gates_m1p00.json").read_text())
        ik, ikg = C["diag9g"]["leg_ik"], g["diag"]["diag9g"]["leg_ik"]
        assert json.loads(json.dumps(ik)) == ikg, clip
        w0, w1 = walk["w0"], walk["w1"]
        ia = [names.index(n) for n in ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")]
        fs = np.linalg.norm(np.diff(C["human1"][:, ia], axis=0), axis=2).max(1)  # fs[t]: frame t -> t + 1
        f0 = np.linalg.norm(np.diff(h0[:, ia], axis=0), axis=2).max(1)
        st = np.arange(w0, min(w0 + START, w1 - 1))
        wk = np.arange(w0, w1 - 1)
        ts = int(st[fs[st].argmax()])
        tw = int(wk[fs[wk].argmax()])
        tau = np.asarray(C["tau"], float)
        row = dict(clip=clip, sequence=sm["sequence"], object=sm["object"], frames=sm["frames"], walk=f"{w0}:{w1}", walk_frames=w1 - w0, smpl_scale=round(s, 4),
                   leg_stretch_walk_max_m=mx(ik["L"]["walk"]["max_excess_m"], ik["R"]["walk"]["max_excess_m"]),
                   leg_stretch_walk_L_m=ik["L"]["walk"]["max_excess_m"], leg_stretch_walk_R_m=ik["R"]["walk"]["max_excess_m"],
                   overextended_walk_frames_L=ik["L"]["walk"]["overextended_frames"], overextended_walk_frames_R=ik["R"]["walk"]["overextended_frames"],
                   leg_stretch_lift_max_m=mx(ik["L"]["lift"]["max_excess_m"], ik["R"]["lift"]["max_excess_m"]),
                   leg_stretch_put_down_max_m=mx(ik["L"]["put_down"]["max_excess_m"], ik["R"]["put_down"]["max_excess_m"]),
                   foot_step_walk_start_max_m=round(float(fs[ts]), 4), foot_step_walk_start_frame_offset=ts - w0,
                   foot_step_walk_start_source_frames=round(float(tau[ts + 1] - tau[ts]), 3) if len(tau) > ts + 1 else None,
                   foot_step_walk_max_m=round(float(fs[tw]), 4), foot_step_walk_max_frame_offset=tw - w0,
                   foot_step_clip_max_m=round(float(fs.max()), 4), foot_step_clip_max_frame=int(fs.argmax()),
                   demo_foot_step_walk_max_m=round(float(f0[wk].max()), 4),
                   source_frames_per_frame_max=C["diag9g"]["tau_walk"]["speed_ratio_max"])
        lv = L / clip / "m1p00"
        if (lv / "r1/run_record.json").exists():
            P = robot_feet(lv)
            rs = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1)
            row.update(robot_r1_1p00_foot_step_walk_start_max_m=round(float(rs[st[st < len(rs)]].max()), 4),
                       robot_r1_1p00_foot_step_walk_max_m=round(float(rs[wk[wk < len(rs)]].max()), 4))
        else:
            row.update(robot_r1_1p00_foot_step_walk_start_max_m=None, robot_r1_1p00_foot_step_walk_max_m=None)
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
