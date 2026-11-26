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
Step 9c: --fb-turn world --fb-legs walk (frontback.py options; defaults = step 9b); 9d walk_scopy, 9e full_scopy, 9f reverse_warp,
9g mirror (diag9g + prediag_9g).
Every gates file also holds r1_source (frontback.build): per partner frame the source frame and the rigid map of its legs.
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


def prediag_9f(C, h0, names, O, size, s):
    """step 9f pre-solve diagnostics (Kevin 2026-11-25 17:10 item 6), solver scale (m) unless noted:
    * time scale of each segment (partner frames / source frames);
    * partner palms vs their contact points on the shared target: contact point = the demonstration palm's position in its own
      box at tau(t), carried to the far end of the shared target (P(t) M1); frames where that demonstration palm is within 5 cm of
      its box (palm-point proxy of the contact flag); per segment: frames, mean / max distance; also the partner palm's signed
      distance to the shared target box;
    * partner feet (ankles, toes) vs the shared target box (size L x w x h at P(t)) and vs its own box: smallest signed distance
      per segment and frames inside;
    * pelvis jumps: largest frame-to-frame partner pelvis displacement (and frame), next to the demonstration's; the steps at the
      segment boundaries w0, w1."""
    import frontback as fb
    T = len(O)
    tau = C["tau"]
    w0, w1 = int(C["walk"]["w0"]), int(C["walk"]["w1"])
    ph = lambda t: "lift" if t < w0 else ("walk" if t < w1 else "put_down")  # noqa: E731
    tsize = np.array(size, float)
    tsize[C["k"]] = C["Lm"]
    far = C["P"] @ C["M1"]
    Ow = fb.poses_at(O, tau)
    hw = fb.lerp_at(h0, tau)
    out = dict(segments=C["warp_segments"], offset_d_m=C["offset_d"], tau_first_last=[round(float(tau[0]), 3), round(float(tau[-1]), 3)],
               own_box_vs_far_end_m=dict(mean=round(float(C["own_vs_far_end_m"].mean()), 4), max=round(float(C["own_vs_far_end_m"].max()), 4)))
    hands = {}
    for hn in ("L_Palm", "R_Palm"):
        j = names.index(hn)
        dist, sd_t, flag = np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, bool)
        for t in range(T):
            flag[t] = fb.box_sd(hw[t, j][None], Ow[t], size)[0] <= 0.05
            loc = (hw[t, j] - Ow[t][:3, 3]) @ Ow[t][:3, :3]  # demonstration palm in its own box at tau
            cp = far[t][:3, :3] @ loc + far[t][:3, 3]
            dist[t] = np.linalg.norm(C["human1"][t, j] - cp)
            sd_t[t] = fb.box_sd(C["human1"][t, j][None], C["P"][t], tsize)[0]
        seg = {}
        for p_ in ("lift", "walk", "put_down"):
            sel = [t for t in range(T) if ph(t) == p_ and flag[t]]
            seg[p_] = dict(frames=len(sel), dist_mean=round(float(dist[sel].mean()), 4) if sel else None, dist_max=round(float(dist[sel].max()), 4) if sel else None,
                           sd_to_shared_box_min=round(float(sd_t[sel].min()), 4) if sel else None, sd_to_shared_box_max=round(float(sd_t[sel].max()), 4) if sel else None)
        hands[hn] = seg
    out["palms_vs_contact_points"] = hands
    fj = [names.index(n) for n in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")]
    sd_sh = np.array([fb.box_sd(C["human1"][t, fj], C["P"][t], tsize).min() for t in range(T)])
    sd_own = np.array([fb.box_sd(C["human1"][t, fj], C["own1"][t], size).min() for t in range(T)])
    feet = {}
    for p_ in ("lift", "walk", "put_down"):
        sel = [t for t in range(T) if ph(t) == p_]
        feet[p_] = dict(frames=len(sel), min_to_shared_box=round(float(sd_sh[sel].min()), 4) if sel else None, inside_shared=int((sd_sh[sel] < 0).sum()) if sel else 0,
                        min_to_own_box=round(float(sd_own[sel].min()), 4) if sel else None, inside_own=int((sd_own[sel] < 0).sum()) if sel else 0)
    out["feet_vs_boxes"] = feet
    ip = names.index("Pelvis")
    st1 = np.linalg.norm(np.diff(C["human1"][:, ip], axis=0), axis=1)
    st0 = np.linalg.norm(np.diff(h0[:, ip], axis=0), axis=1)
    out["pelvis_steps"] = dict(partner_max_m=round(float(st1.max()), 4), partner_max_at_frame=int(st1.argmax()) + 1, demo_max_m=round(float(st0.max()), 4),
                               at_w0=None if not 0 < w0 < T else round(float(st1[w0 - 1]), 4), at_w1=None if not 0 < w1 < T else round(float(st1[w1 - 1]), 4),
                               partner_max_person_scale_m=round(float(st1.max()) / s, 4))
    return out


def prediag_9g(C, h0, names, O, size, s):
    """step 9g pre-solve diagnostics (same quantities as step 9f's, for the mirror construction), solver scale (m):
    * palms vs their contact points, both roles: contact point = the demonstration palm's position in its own (real) box at the
      same frame, carried to the role's end of the shared target (P(t) M0 for role 0, P(t) M1 for role 1; the partner's left palm
      <- the demonstration's right palm); frames where that demonstration palm is within 5 cm of its box; per segment: frames,
      mean / max distance, the palm's signed distance to the shared target box;
    * partner feet (ankles, toes) vs the shared target box and vs its own box: smallest signed distance per segment, frames inside;
    * pelvis steps: largest frame-to-frame partner pelvis displacement, next to the demonstration's, and the steps at w0 / w1.
    The construction's own record (plane, box deviation from the plane, distance playback, leg IK) is diag9g."""
    import frontback as fb
    T = len(O)
    w0, w1 = int(C["walk"]["w0"]), int(C["walk"]["w1"])
    ph = lambda t: "lift" if t < w0 else ("walk" if t < w1 else "put_down")  # noqa: E731
    tsize = np.array(size, float)
    tsize[C["k"]] = C["Lm"]
    out = {}
    pal = {}
    for role, M, person in ((0, C["M0"], h0), (1, C["M1"], C["human1"])):
        ends = C["P"] @ M
        for hn in ("L_Palm", "R_Palm"):
            src = names.index(hn) if role == 0 else names.index(("R_" if hn.startswith("L_") else "L_") + hn[2:])
            j = names.index(hn)
            dist, sd_t, flag = np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, bool)
            for t in range(T):
                flag[t] = fb.box_sd(h0[t, src][None], O[t], size)[0] <= 0.05
                loc = (h0[t, src] - O[t][:3, 3]) @ O[t][:3, :3]
                cp = ends[t][:3, :3] @ loc + ends[t][:3, 3]
                dist[t] = np.linalg.norm(person[t, j] - cp)
                sd_t[t] = fb.box_sd(person[t, j][None], C["P"][t], tsize)[0]
            seg = {}
            for p_ in ("lift", "walk", "put_down"):
                sel = [t for t in range(T) if ph(t) == p_ and flag[t]]
                seg[p_] = dict(frames=len(sel), dist_mean=round(float(dist[sel].mean()), 4) if sel else None, dist_max=round(float(dist[sel].max()), 4) if sel else None,
                               sd_to_shared_box_min=round(float(sd_t[sel].min()), 4) if sel else None, sd_to_shared_box_max=round(float(sd_t[sel].max()), 4) if sel else None)
            pal[f"r{role}_{hn}"] = seg
    out["palms_vs_contact_points"] = pal
    fj = [names.index(n) for n in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")]
    sd_sh = np.array([fb.box_sd(C["human1"][t, fj], C["P"][t], tsize).min() for t in range(T)])
    sd_own = np.array([fb.box_sd(C["human1"][t, fj], C["own1"][t], size).min() for t in range(T)])
    feet = {}
    for p_ in ("lift", "walk", "put_down"):
        sel = [t for t in range(T) if ph(t) == p_]
        feet[p_] = dict(frames=len(sel), min_to_shared_box=round(float(sd_sh[sel].min()), 4) if sel else None, inside_shared=int((sd_sh[sel] < 0).sum()) if sel else 0,
                        min_to_own_box=round(float(sd_own[sel].min()), 4) if sel else None, inside_own=int((sd_own[sel] < 0).sum()) if sel else 0)
    out["feet_vs_boxes"] = feet
    ip = names.index("Pelvis")
    st1 = np.linalg.norm(np.diff(C["human1"][:, ip], axis=0), axis=1)
    st0 = np.linalg.norm(np.diff(h0[:, ip], axis=0), axis=1)
    out["pelvis_steps"] = dict(partner_max_m=round(float(st1.max()), 4), partner_max_at_frame=int(st1.argmax()) + 1, demo_max_m=round(float(st0.max()), 4),
                               at_w0=None if not 0 < w0 < T else round(float(st1[w0 - 1]), 4), at_w1=None if not 0 < w1 < T else round(float(st1[w1 - 1]), 4))
    ia = [names.index(n) for n in ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")]
    fs = np.linalg.norm(np.diff(C["human1"][:, ia], axis=0), axis=2).max(1)
    out["feet_steps"] = dict(partner_max_m=round(float(fs.max()), 4), at_w0=None if not 0 < w0 < T else round(float(fs[w0 - 1]), 4),
                             at_w1=None if not 0 < w1 < T else round(float(fs[w1 - 1]), 4))
    return out


def gates(h0, names, O, size, s, m, turn="box", legs="all", walk=None, walk_distance="pelvis"):
    import frontback as fb
    import partner as pt
    C = fb.build(h0, names, O, size, m, turn=turn, legs=legs, walk=walk, walk_distance=walk_distance)
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
    if legs == "reverse_warp":
        diag["prediag_9f"] = prediag_9f(C, h0, names, O, size, s)
    if legs == "mirror":
        diag["diag9g"] = C["diag9g"]
        diag["prediag_9g"] = prediag_9g(C, h0, names, O, size, s)
    rs = C["r1_source"]  # per partner frame: source frame and rigid map of the legs (step9_eval.py --r1-source construction)
    return dict(m=m, gravity_deg=round(g, 4), gravity_passed=g <= pt.V0_GRAVITY_DEG + 1e-10, main_motion=mm,
                passed=bool(g <= pt.V0_GRAVITY_DEG + 1e-10 and mm["passed"]), improper=bool(np.linalg.det(C["S"][:3, :3]) < 0), S=np.round(C["S"], 6).tolist(),
                candidate=C["cand"], diag=diag,
                r1_source=dict(frame=[int(x) if float(x).is_integer() else round(float(x), 6) for x in rs["frame"]], R=np.asarray(rs["R"]).tolist(),
                               t=np.asarray(rs["t"]).tolist(), swap_lr=bool(rs.get("swap_lr", False))))


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
    ap.add_argument("--fb-legs", choices=("all", "walk", "walk_scopy", "full_scopy", "reverse_warp", "mirror"), default="all")
    ap.add_argument("--fb-walk-distance", choices=("pelvis", "pelvis_feet"), default="pelvis",
                    help="--fb-legs mirror: walked distance of the time alignment (frontback.py; step 9g second fix: pelvis_feet)")
    a = ap.parse_args()
    fr = tuple(int(x) for x in a.frames.split(":"))
    h0, names, nom, size, s = load_role0(a.object, a.sequence, fr)
    walk = None
    if a.fb_legs in ("walk", "walk_scopy", "full_scopy", "reverse_warp", "mirror"):
        import frontback as fb
        import omomo_motion as om
        walk = fb.walking_segment(om.load_sequences({a.object})[a.sequence], *fr)
    O = pt.mj_to_mat(nom)
    a.out.mkdir(parents=True, exist_ok=True)
    out = dict(object=a.object, sequence=a.sequence, frames=list(fr), smpl_scale=s, box_size_solver_m=np.round(size, 4).tolist(), levels={})
    for m in (float(x) for x in a.levels.split(",")):
        g = gates(h0, names, O, size, s, m, a.fb_turn, a.fb_legs, walk, a.fb_walk_distance)
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
