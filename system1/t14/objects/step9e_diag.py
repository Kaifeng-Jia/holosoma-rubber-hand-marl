#!/usr/bin/env python3
"""T14 step 9e: where (lift / walk / put down) and which body parts make a front-back level fail. Diagnostic only (read
only, CPU); the verdict itself is step9_eval.py's eval.json and is not changed here.

Phases from the level's gates file (walk = [w0, w1) of the demonstration; partner frames use the same clip frame):
lift = [0, w0), walk = [w0, w1), put down = [w1, T). For each robot, with the verdict's own quantities (shared_eval.evaluate):
  * non-hand penetration: additional proxy penetration per frame = max(0, -non-hand proxy clearance of the robot vs the shared
    target) - the source robot's own penetration vs the source box at the same frame (V0 round 2); frames above the 0.01 m
    limit per phase, the deepest value per phase, and the robot links of the CollisionProxy witnesses on those frames;
  * two-robot clearance: frames with cross-robot proxy clearance < -0.01 m per phase, the link pairs of the witnesses;
  * contact retention: condition frames (person's hand within 0.10 m of its box and the object moving > 0.03 m/s) on which
    the robot hand's exact signed distance is < -0.02 m, per phase and hand;
  * support drift (round-6 windows, step9_eval --r1-source as in eval.json): per foot, the windows whose added drift exceeds
    0.01 m, their frames and phase.
Writes one json per level (default <level dir>/diag9e.json).

    sonic_s1/bin/python -B system1/t14/objects/step9e_diag.py --level-dir <dir with r0/ r1/ eval.json> --source <source solve>
        --gates <gates_m<tag>.json> [--out <json>]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared_eval as se  # noqa: E402  (import paths)
import step8_rejudge as rj  # noqa: E402

PHASES = ("lift", "walk", "put_down")


def phase_of(t, w0, w1):
    return "lift" if t < w0 else ("walk" if t < w1 else "put_down")


def ranges(idx):
    idx = sorted(int(i) for i in idx)
    out = []
    for i in idx:
        if out and i == out[-1][1]:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def by_phase(frames, w0, w1, values=None):
    out = {}
    for ph in PHASES:
        sel = [k for k, t in enumerate(frames) if phase_of(t, w0, w1) == ph]
        d = dict(frames=len(sel), ranges=ranges([frames[k] for k in sel]))
        if values is not None and sel:
            d["deepest_m"] = round(float(max(values[k] for k in sel)), 4)
        out[ph] = d
    return out


def main():
    from reference_geometry_math import BoxUnion
    from reference_geometry_round2_checks import CollisionProxy
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
    import support
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--gates", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    C = se.ctx()
    fk = C["fk"]
    d = a.level_dir
    gates = json.loads(a.gates.read_text())
    walk = gates["diag"]["walk"]
    w0, w1 = int(walk["w0"]), int(walk["w1"])
    ev_json = json.loads((d / "eval.json").read_text())
    r1_mode = "construction" if "construction" in str((ev_json.get("support_drift_round6") or {}).get("r1", {}).get("source", "")) else "reversed"
    recs = [json.loads((d / f"r{r}/run_record.json").read_text()) for r in (0, 1)]
    zs = [np.load(d / f"r{r}/{recs[r]['tag']}_new_qpos.npz") for r in (0, 1)]
    q = [z["qpos"].astype(np.float64)[:, :36] for z in zs]
    obj = zs[0]["object_qpos_mj"].astype(np.float64)
    T = min(len(q[0]), len(q[1]))
    q, obj = [x[:T] for x in q], obj[:T]
    srec = json.loads((a.source / "run_record.json").read_text())
    sq = np.load(a.source / f"{srec['tag']}_new_qpos.npz")["qpos"].astype(np.float64)[:T]
    src_engine = CollisionProxy(fk, BoxUnion(srec["target"]["urdf"]))
    src_coll, _ = src_engine.evaluate([fk.poses(sq[:, :36], list(fk.meshes))], sq[:, 36:43])
    src_pen = np.maximum(0, -src_coll["nonhand_object_clearance_m"][:, 0])
    eng = CollisionProxy(fk, BoxUnion(recs[0]["target"]["urdf"]))
    poses = [fk.poses(x, list(fk.meshes)) for x in q]
    coll, wit = eng.evaluate(poses, obj)
    lname = lambda g: eng.names[g % eng.n]  # noqa: E731
    res = dict(level_dir=str(d), T=T, walk=[w0, w1], applicable=ev_json["applicable"], failed=ev_json["failed"], source_penetration_max_m=round(float(src_pen.max()), 4),
               robots={}, rules=__doc__)
    # two-robot clearance
    pair = coll["robot_robot_clearance_m"]
    bad = [int(t) for t in np.flatnonzero(pair < -0.01)]
    pairs = {ph: Counter() for ph in PHASES}
    for w in wit:
        if w["kind"] == "robot_robot" and w["frame"] in bad:
            g = w["geom_indices"]
            pairs[phase_of(w["frame"], w0, w1)][f"r0 {lname(g[0])} / r1 {lname(g[1])}"] += 1
    rr = by_phase(bad, w0, w1, values=[-pair[t] for t in bad])
    for ph in PHASES:
        rr[ph]["link_pairs"] = pairs[ph].most_common(4)
    res["robot_robot"] = dict(limit_m=-0.01, min_m=round(float(pair.min()), 4), phases=rr)
    # contact retention (exact signed distance of the robot hands)
    import eval_core4d_hulls as eh
    evd = eh.ObjectRobotDistance(eh.urdf_box_meshes(recs[0]["target"]["urdf"]), C["robot"], depth="exact")
    Hd, _ = evd.frames(np.stack(q, 1), obj)
    hc = [np.load(d / f"r{r}/human_contact.npz") for r in (0, 1)]
    # support drift windows
    names = list(SMPLH_DEMO_JOINTS)
    il, ir = names.index("L_Toe"), names.index("R_Toe")
    sfeet = se.feet(fk, sq[:, :36])
    for r in (0, 1):
        out = {}
        add = np.maximum(0, -coll["nonhand_object_clearance_m"][:, r]) - src_pen
        badn = [int(t) for t in np.flatnonzero(add > 0.01)]
        links = {ph: Counter() for ph in PHASES}
        for w in wit:
            if w["kind"] == "nonhand_object" and w.get("agent") == r and w["frame"] in badn:
                links[phase_of(w["frame"], w0, w1)][lname(w["geom_indices"][0])] += 1
        nh = by_phase(badn, w0, w1, values=[add[t] for t in badn])
        for ph in PHASES:
            nh[ph]["links"] = links[ph].most_common(4)
        out["nonhand"] = dict(limit_m=0.01, value_m=round(float(add.max()), 4), phases=nh)
        h_sd, spd = hc[r]["h_sd"][:T], hc[r]["object_speed"][:T]
        cond = (h_sd >= -0.10) & (spd[:, None] > 0.03)
        lost = cond & (Hd[:, r] < -0.02)
        ret = {}
        for h, side in enumerate(("left", "right")):
            fr = [int(t) for t in np.flatnonzero(lost[:, h])]
            ret[side] = dict(condition_frames=int(cond[:, h].sum()), lost_frames=len(fr), phases=by_phase(fr, w0, w1),
                             condition_by_phase={ph: int(sum(1 for t in np.flatnonzero(cond[:, h]) if phase_of(t, w0, w1) == ph)) for ph in PHASES})
        out["retention"] = dict(value=round(float((~lost)[cond].mean()), 4) if cond.any() else None, limit=0.99, hands=ret)
        z = zs[r]
        flags = support.support_flags(z["human_joints"][:, [il, ir]])
        rf = se.feet(fk, q[r])
        dr = {}
        for k, side in enumerate(("left", "right")):
            src0 = sfeet[f"{side}_foot_position_m"][:T]
            if r == 1 and r1_mode == "construction":
                rs = gates["r1_source"]
                fr = np.asarray(rs["frame"], float)
                if np.allclose(fr, np.round(fr)):
                    srcf = src0[np.round(fr).astype(int)]
                else:  # step 9f: fractional source times, linear interpolation as step9_eval.py
                    i0 = np.clip(np.floor(fr).astype(int), 0, len(src0) - 1)
                    i1 = np.clip(i0 + 1, 0, len(src0) - 1)
                    f_ = (fr - i0)[:, None]
                    srcf = src0[i0] * (1 - f_) + src0[i1] * f_
                src = np.einsum("tij,tj->ti", np.asarray(rs["R"]), srcf) + np.asarray(rs["t"])
            else:
                src = src0[::-1] if r == 1 else src0
            wins = []
            for s0, s1 in rj.segments(flags[:T, k]):
                rd = np.linalg.norm(rf[f"{side}_foot_position_m"][s0:s1, :2] - rf[f"{side}_foot_position_m"][s0, :2], axis=1)
                sd = np.linalg.norm(src[s0:s1, :2] - src[s0, :2], axis=1)
                v = float((rd - sd).max())
                if v > rj.LIMIT:
                    wins.append(dict(frames=[int(s0), int(s1)], value_m=round(v, 4), phase=sorted({phase_of(t, w0, w1) for t in range(s0, s1)})))
            dr[side] = wins
        out["support_drift"] = dict(source=r1_mode if r == 1 else "same frame", limit_m=rj.LIMIT, windows_over_limit=dr)
        res["robots"][f"r{r}"] = out
    o = a.out or (d / "diag9e.json")
    o.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(dict(level=str(d), failed=res["failed"], rr={ph: v["frames"] for ph, v in rr.items()},
                          nonhand={r: {ph: (v["nonhand"]["phases"][ph]["frames"], v["nonhand"]["phases"][ph]["links"][:2]) for ph in PHASES} for r, v in res["robots"].items()})), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
