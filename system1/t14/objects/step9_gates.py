#!/usr/bin/env python3
"""T14 step 9 (Kevin 2026-11-25 00:30): pre-solve gates and diagnostics of the front-back construction (frontback.py) of one
class-a clip at the given levels. CPU, nothing is solved.

Per level m (target = the box lengthened along the walking axis to m x 2 w):
  * V0 gravity: relative_gravity_deviation_deg of S (the 180-deg map at frame 0) <= 5 deg (partner.gravity_deg);
  * main motion (Kevin: the screen's class-a threshold, pelvis net displacement > 1.0 m): horizontal pelvis net displacement
    over the clip of role 0's person (the demonstration) and of role 1's person (the reversed legs), divided by s (person's
    real scale), each > screen_omomo.RULES a_pelvis_net_m = 1.0 m. The solved robots' pelvis is judged after solving
    (step9_eval.py);
  * diagnostic (reported, not a gate): reversed-leg pelvis - object-attached pelvis (frontback.diag), the walking axis and its
    angle to the box axis, the partner's initial pelvis distance; step 9c also the legs-to-own-box distance (frontback.leg_box)
    and the walking segment.
Step 9c: --fb-turn world --fb-legs walk (frontback.py options; defaults = step 9b).
Writes <out>/gates_m<tag>.json per level and <out>/diag.json.

    hsretargeting_s1/bin/python -B system1/t14/objects/step9_gates.py --object smallbox --sequence sub4_smallbox_047 --frames 36:172 \
        --levels 1.0,1.25,1.5 --out logs_system1/t14/step9b/<tag>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
A_PELVIS_NET_M = 1.0  # screen_omomo.RULES["a_pelvis_net_m"]


def level_tag(m):
    return f"m{m:.2f}".replace(".", "p")


def gates(h0, names, O, size, s, m, turn="box", legs="all", walk=None):
    import frontback as fb
    import partner as pt
    C = fb.build(h0, names, O, size, m, turn=turn, legs=legs, walk=walk)
    ip = names.index("Pelvis")
    p0, p1 = h0[:, ip], C["human1"][:, ip]
    net0 = float(np.linalg.norm(p0[-1, :2] - p0[0, :2])) / s
    net1 = float(np.linalg.norm(p1[-1, :2] - p1[0, :2])) / s
    g = pt.gravity_deg(C["S"])
    mm = dict(rule="pelvis net horizontal displacement / s > 1.0 m (screen class a), persons of both roles", limit_m=A_PELVIS_NET_M,
              person_r0_m=round(net0, 4), person_r1_m=round(net1, 4), person_r0_solver_m=round(net0 * s, 4), person_r1_solver_m=round(net1 * s, 4),
              limit_solver_m=round(A_PELVIS_NET_M * s, 4), passed=bool(net0 > A_PELVIS_NET_M and net1 > A_PELVIS_NET_M))
    tilt = [float(np.degrees(np.arccos(np.clip((C["P"][t, :3, :3] @ C["A"][t, :3, :3].T)[2, 2], -1, 1)))) for t in range(len(O))]
    diag = dict(axis=C["k"], sigma=C["sigma"], walk_axis_angle_deg=round(C["walk_axis_angle_deg"], 2), L_m=round(C["Lm"], 4), **fb.diag(C["D"], s),
                partner_initial_pelvis_distance_m=round(float(np.linalg.norm(p1[0, :2] - p0[0, :2])), 4),
                partner_min_pelvis_distance_m=round(float(np.linalg.norm(p1[:, :2] - p0[:, :2], axis=1).min()), 4),
                shared_vs_demo_rotation_max_deg=round(max(tilt), 2), leg_offset_m=np.round(C["leg_offset"], 4).tolist(), turn=turn, legs=legs, walk=walk,
                leg_box=fb.leg_box(h0, names, O, size, C)[0], box_tilt_frame0_deg=round(float(np.degrees(np.arccos(np.clip(O[0][2, 2], -1, 1)))), 3))
    return dict(m=m, gravity_deg=round(g, 4), gravity_passed=g <= pt.V0_GRAVITY_DEG + 1e-10, main_motion=mm,
                passed=bool(g <= pt.V0_GRAVITY_DEG + 1e-10 and mm["passed"]), improper=False, S=np.round(C["S"], 6).tolist(),
                candidate=C["cand"], diag=diag)


def main():
    from step9_diag import load_role0
    import partner as pt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--frames", required=True, help="start:stop (stop exclusive), the screen's carry episode")
    ap.add_argument("--levels", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fb-turn", choices=("box", "world"), default="box")
    ap.add_argument("--fb-legs", choices=("all", "walk", "walk_scopy"), default="all")
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(":"))
    h0, names, nom, size, s = load_role0(a.object, a.sequence, fr)
    walk = None
    if a.fb_legs in ("walk", "walk_scopy"):
        import frontback as fb
        import omomo_motion as om
        walk = fb.walking_segment(om.load_sequences({a.object})[a.sequence], *fr)
    O = pt.mj_to_mat(nom)
    a.out.mkdir(parents=True, exist_ok=True)
    out = dict(object=a.object, sequence=a.sequence, frames=list(fr), smpl_scale=s, box_size_solver_m=np.round(size, 4).tolist(), levels={})
    for m in (float(x) for x in a.levels.split(",")):
        g = gates(h0, names, O, size, s, m, a.fb_turn, a.fb_legs, walk)
        (a.out / f"gates_{level_tag(m)}.json").write_text(json.dumps(g, indent=1) + "\n")
        out["levels"][f"{m:.2f}"] = {k: g[k] for k in ("gravity_deg", "main_motion", "passed", "diag")}
    prev = json.loads((a.out / "diag.json").read_text()) if (a.out / "diag.json").exists() else None
    if prev:
        prev["levels"].update(out["levels"])
        out["levels"] = dict(sorted(prev["levels"].items(), key=lambda kv: float(kv[0])))
    (a.out / "diag.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
