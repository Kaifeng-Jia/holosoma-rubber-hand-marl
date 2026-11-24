#!/usr/bin/env python3
"""T14 step 9 part A (Kevin 2026-11-24 18:40, step-8 decision 1): re-judge every solved step-8 candidate with the support-drift
windows of round 6. Recompute only, no new solve.

Unchanged: the V0 support-drift quantity and its frozen limit (reference_geometry_round2_checks: per foot, the horizontal
displacement of the ankle_roll_link origin from the start of each support window, robot minus source robot, frame by frame;
largest value <= V0 additional_support_drift_m = 0.01 m) and every other item of the step-8 verdict (shared_eval.py).
Changed: the windows. Old (step 8) = the source robot's V0 support windows (foot bottom <= 4 cm, foot speed <= 0.15 m/s,
>= 0.1 s). New = the person's support segments of round 6 (support.py: per foot, toe speed < 0.01 m/s after Gaussian smoothing,
sigma 8 frames; the definition the support lock uses), of each role's own person (role 1 = the S-mapped person; for a mirrored
S its left foot is the source's right foot), read from the role solve's saved human joints.
Source robot of role 1 = the source robot moved by S (drift unchanged; mirrored S swaps the feet).

    sonic_s1/bin/python -B system1/t14/objects/step8_rejudge.py --out results_system1/T14_retargeting/step9/step8_rejudge.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "scripts", ROOT / "system1/t14", ROOT / "system1/t12v2", ROOT / "src/holosoma_retargeting", HERE):
    sys.path.insert(0, str(p))
L = ROOT / "logs_system1/t14/step8"
TAGS = ("smallboxb", "largeboxb", "suitcaseb", "smallboxc", "largeboxc", "suitcasec", "push", "pull")
LIMIT = 0.01  # V0 additional_support_drift_m


def segments(flag):
    f = np.concatenate([[False], np.asarray(flag, bool), [False]])
    e = np.flatnonzero(np.diff(f.astype(int)))
    return list(zip(e[0::2], e[1::2]))


def added_drift(robot_pos, source_pos, flag):
    """V0 quantity in the given windows: max over window frames of (robot drift - source drift), drift = horizontal distance
    from the window's first frame."""
    best, nwin, nfr = -np.inf, 0, 0
    for a, b in segments(flag):
        d = np.linalg.norm(robot_pos[a:b, :2] - robot_pos[a, :2], axis=1)
        d0 = np.linalg.norm(source_pos[a:b, :2] - source_pos[a, :2], axis=1)
        best = max(best, float((d - d0).max()))
        nwin += 1
        nfr += b - a
    return (None if not np.isfinite(best) else best), nwin, nfr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import support
    from reference_geometry_pilot import RobotFK
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
    import shared_eval as se
    fk = RobotFK()
    names = list(SMPLH_DEMO_JOINTS)
    il, ir = names.index("L_Toe"), names.index("R_Toe")
    rows, seq_summary = [], {}
    for tag in TAGS:
        cs = json.loads((L / f"cands/{tag}/candidates.json").read_text())
        src = Path(cs["source"])
        srec = json.loads((src / "run_record.json").read_text())
        sq = np.load(src / f"{srec.get('tag', srec.get('task'))}_new_qpos.npz")["qpos"][:, :36].astype(np.float64)
        sfeet = se.feet(fk, sq)
        old_app, new_app = [], []
        for c in cs["candidates"]:
            d = L / f"cands/{tag}/c{c['index']:02d}/m1p00"
            ef = d / "eval.json"
            if not ef.exists():
                continue
            e = json.loads(ef.read_text())
            row = dict(sequence=tag, candidate=f"c{c['index']:02d}", id=c["id"], old_applicable=e["applicable"], old_failed=" | ".join(e["failed"]))
            new_fail = [x for x in e["failed"] if not x.startswith("V0 support drift")]
            for r in (0, 1):
                rec = json.loads((d / f"r{r}/run_record.json").read_text())
                z = np.load(d / f"r{r}/{rec['tag']}_new_qpos.npz")
                T = len(z["qpos"])
                q = z["qpos"][:, :36].astype(np.float64)
                hj = z["human_joints"]
                flags = support.support_flags(hj[:, [il, ir]])  # [T, 2] left, right of this role's person
                rf = se.feet(fk, q)
                swap = bool(rec.get("improper")) and r == 1
                vals = []
                for k, side in enumerate(("left", "right")):
                    sside = ("right", "left")[k] if swap else side
                    v, nw, nf = added_drift(rf[f"{side}_foot_position_m"], sfeet[f"{sside}_foot_position_m"][:T], flags[:T, k])
                    vals.append(v)
                    row[f"new_windows_r{r}_{side}"] = nw
                    row[f"new_window_frames_r{r}_{side}"] = nf
                fin = [v for v in vals if v is not None]
                new = max(fin) if fin else None
                old = e["V0"][f"support_drift_r{r}"]["value_m"]
                row[f"old_support_drift_r{r}_m"] = old
                row[f"new_support_drift_r{r}_m"] = None if new is None else round(new, 5)
                row[f"round6_toe_drift_max_r{r}_cm"] = max((e["foot_slide"][f"r{r}"][f"foot_slide_{s}"]["drift_max_cm"] or 0) for s in ("left", "right"))
                if new is not None and new > LIMIT + 1e-10:
                    new_fail.append(f"V0 support drift r{r}")
            order = ["V0 gravity", "V0 support drift r0", "V0 support drift r1", "V0 non-hand penetration r0", "V0 non-hand penetration r1",
                     "V0 robot-robot clearance", "section-4 contact retention >= 0.99", "main motion"]
            new_fail = sorted(set(new_fail), key=lambda x: order.index(x) if x in order else 99)
            row.update(new_applicable=not new_fail, new_failed=" | ".join(new_fail))
            rows.append(row)
            (old_app if e["applicable"] else []).append(c["index"])
            (new_app if not new_fail else []).append(c["index"])
        seq_summary[tag] = dict(solved=sum(1 for r in rows if r["sequence"] == tag), old_applicable=old_app, new_applicable=new_app)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    a.out.with_suffix(".json").write_text(json.dumps(dict(rules=__doc__, summary=seq_summary), indent=1, ensure_ascii=False) + "\n")
    for tag, s in seq_summary.items():
        print(tag, s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
