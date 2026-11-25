#!/usr/bin/env python3
"""T14 step 9: verdict of one front-back level (two robots solved by role_solve.py --construction frontback). Read only, CPU.

Same verdict as step 8 (shared_eval.evaluate: V0 gravity on S, non-hand penetration, two-robot clearance, section-4 contact
retention >= 0.99; Section 4's penetration criterion has no baseline and is not used), with two items as Kevin set them:
  * support drift (Kevin 2026-11-24 18:40 item 1, the step-8 re-judge): V0 quantity and limit (step8_rejudge.added_drift, 0.01 m),
    windows = each role's own person's support segments of round 6 (support.py on the solve's saved human joints). Source robot
    of role 0 = the source solve at the same frame; of role 1 = the source solve played backwards in time (the partner's legs
    are the demonstration's legs reversed and turned about the vertical: a rigid motion, the same feet). The step-8 window
    values are kept as information (old_window_*). --r1-source construction (step 9e): role 1's source at frame t = the
    source solve at the construction's source frame, moved by the same rigid map (gates r1_source), so that it matches
    constructions that reverse only the walking segment (the whole-clip-reversed value is kept as information).
  * main motion (Kevin 2026-11-25: the screen's class-a threshold): pelvis net horizontal displacement over the clip / s > 1.0 m,
    for the two persons (pre-solve gate, step9_gates.py) and for the two solved robots (root position of each solve); in step
    9c's words the robots' displacement (solver scale) > 1.0 m x s, the same test; both scales are recorded.
Writes the evaluation json and (optionally) the pair npz in the step-7 format.

    sonic_s1/bin/python -B system1/t14/objects/step9_eval.py --cand-dir <level dir with r0/ r1/> --source <source solve> \
        --gates <gates_m<tag>.json> --out <eval.json> [--pair-out <pair_fps30.npz>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared_eval as se  # noqa: E402  (sets the import paths)
import step8_rejudge as rj  # noqa: E402

ORDER = ["V0 gravity", "V0 support drift r0", "V0 support drift r1", "V0 non-hand penetration r0", "V0 non-hand penetration r1",
         "V0 robot-robot clearance", "section-4 contact retention >= 0.99", "main motion"]
A_PELVIS_NET_M = 1.0


def evaluate(cand_dir, source_dir, gates, r1_source="reversed"):
    import support
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
    res = se.evaluate(cand_dir, source_dir, gates)
    fk = se.ctx()["fk"]
    srec = json.loads((source_dir / "run_record.json").read_text())
    sq = np.load(source_dir / f"{srec['tag']}_new_qpos.npz")["qpos"][:, :36].astype(np.float64)
    sfeet = se.feet(fk, sq)
    names = list(SMPLH_DEMO_JOINTS)
    il, ir = names.index("L_Toe"), names.index("R_Toe")
    drift, robots_net = {}, {}
    failed = [x for x in res["failed"] if not x.startswith("V0 support drift") and x != "main motion"]
    for r in (0, 1):
        rec = json.loads((cand_dir / f"r{r}/run_record.json").read_text())
        z = np.load(cand_dir / f"r{r}/{rec['tag']}_new_qpos.npz")
        q = z["qpos"][:, :36].astype(np.float64)
        T = len(q)
        assert len(sq) == T, (len(sq), T)
        flags = support.support_flags(z["human_joints"][:, [il, ir]])
        rf = se.feet(fk, q)
        vals, win = [], {}
        for k, side in enumerate(("left", "right")):
            src0 = sfeet[f"{side}_foot_position_m"][:T]
            src = src0[::-1] if r == 1 else src0
            alt = None
            if r == 1 and r1_source == "construction":  # the source frame and rigid map that built the partner's legs at each frame
                rs = gates["r1_source"]
                fr, R_, t_ = np.asarray(rs["frame"], float), np.asarray(rs["R"]), np.asarray(rs["t"])
                assert len(fr) == T, (len(fr), T)
                alt = rj.added_drift(rf[f"{side}_foot_position_m"], src, flags[:T, k])[0]  # the earlier (whole clip reversed) source
                if np.allclose(fr, np.round(fr)):
                    srcf = src0[np.round(fr).astype(int)]
                else:  # step 9f: fractional source times (time-aligned reversal): linear interpolation of the source feet
                    i0 = np.clip(np.floor(fr).astype(int), 0, len(src0) - 1)
                    i1 = np.clip(i0 + 1, 0, len(src0) - 1)
                    f_ = (fr - i0)[:, None]
                    srcf = src0[i0] * (1 - f_) + src0[i1] * f_
                src = np.einsum("tij,tj->ti", R_, srcf) + t_
            v, nw, nf = rj.added_drift(rf[f"{side}_foot_position_m"], src, flags[:T, k])
            vals.append(v)
            win[side] = dict(windows=nw, frames=nf, value_m=None if v is None else round(v, 5))
            if alt is not None:
                win[side]["reversed_source_value_m"] = round(alt, 5)
        fin = [v for v in vals if v is not None]
        val = max(fin) if fin else None
        ok = val is None or val <= rj.LIMIT + 1e-10
        drift[f"r{r}"] = dict(value_m=None if val is None else round(val, 5), limit=rj.LIMIT, passed=bool(ok), per_foot=win,
                              old_window_value_m=res["V0"][f"support_drift_r{r}"]["value_m"],
                              source=("source solve, whole clip reversed" if r == 1 and r1_source == "reversed" else
                                      "source solve at the construction's source frame, moved by its rigid map (gates r1_source)" if r == 1 else
                                      "source solve, same frame"))
        if not ok:
            failed.append(f"V0 support drift r{r}")
        robots_net[f"r{r}"] = round(float(np.linalg.norm(q[-1, :2] - q[0, :2])) / float(rec["smpl_scale"]), 4)
        robots_net[f"solver_r{r}"] = round(float(np.linalg.norm(q[-1, :2] - q[0, :2])), 4)
        s_ = float(rec["smpl_scale"])
    mm = dict(gates["main_motion"], robot_r0_m=robots_net["r0"], robot_r1_m=robots_net["r1"], robot_r0_solver_m=robots_net["solver_r0"],
              robot_r1_solver_m=robots_net["solver_r1"], robot_limit_solver_m=round(A_PELVIS_NET_M * s_, 4), smpl_scale=s_)
    mm["robots_passed"] = bool(robots_net["r0"] > A_PELVIS_NET_M and robots_net["r1"] > A_PELVIS_NET_M)
    mm["persons_passed"] = bool(gates["main_motion"]["passed"])
    mm["passed"] = mm["robots_passed"] and mm["persons_passed"]
    if not mm["passed"]:
        failed.append("main motion")
    failed = sorted(set(failed), key=lambda x: ORDER.index(x) if x in ORDER else 99)
    res.update(applicable=not failed, failed=failed, support_drift_round6=drift, main_motion=mm, pelvis_mismatch=gates.get("diag"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--gates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair-out", type=Path, default=None)
    ap.add_argument("--r1-source", choices=("reversed", "construction"), default="reversed",
                    help="support-drift source of the partner: 'reversed' = the source solve played backwards over the whole clip "
                         "(steps 9b-9d); 'construction' = at each frame the source frame and rigid map that built the partner's legs "
                         "(gates r1_source; step 9e: walking segment reversed, lift / put down S copies at the same frame)")
    a = ap.parse_args()
    res = evaluate(a.cand_dir, a.source, json.loads(a.gates.read_text()), a.r1_source)
    if a.pair_out is not None:
        p = se.write_pair(a.cand_dir, a.pair_out)
        res.update(pair=str(p), pair_sha256=se.sha(p))
    res.update(rules=__doc__, shared_eval_rules=se.__doc__, command=" ".join(sys.argv))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps(dict(applicable=res["applicable"], failed=res["failed"], retention=res["contact_retention"], rr=res["robot_robot_min_distance_cm"],
                          intersect=res["robot_robot_intersect_frames"], drift={k: v["value_m"] for k, v in res["support_drift_round6"].items()},
                          main=res["main_motion"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
