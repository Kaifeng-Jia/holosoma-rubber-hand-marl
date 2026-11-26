#!/usr/bin/env python3
"""T14 step 9g second fix, item 3 (Kevin 2026-11-26, received 10:29): before (step 9g, walked distance = the pelvis) vs after
(step 9g2, walked distance = mean of the pelvis and both ankles) on the 22 debug clips, both judged with penetration and two-robot
clearance on the actual collision shapes (verdict C, rejudge_clearance_exact.py). Per clip:
  * the largest single-frame foot displacement over the first 15 frames of the reversed walk, person side (partner person's
    ankles and toes, solver scale) and robot side (solved partner robot r1 at level 1.0, ankle-roll links), and the
    demonstration's own largest step over its walk (step9g_feet_ik_table.py);
  * the largest leg stretch in the walking segment (hip-to-ankle distance minus thigh + shank, both legs) and the overextended
    walking frames per leg;
  * the applicable reported levels and the failed items per reported level.
Totals: applicable clips and reported levels before / after. Read only; writes compare.csv and compare.json.

    python3 system1/t14/objects/step9g2_compare.py --before results_system1/T14_retargeting/step9g --after results_system1/T14_retargeting/step9g2
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHORT = {"V0 gravity": "重力", "V0 support drift r0": "漂移 r0", "V0 support drift r1": "漂移 r1", "V0 non-hand penetration r0": "非手 r0",
         "V0 non-hand penetration r1": "非手 r1", "V0 robot-robot clearance": "两机间隙", "section-4 contact retention >= 0.99": "保持率",
         "main motion": "主运动"}
FEET = ["foot_step_walk_start_max_m", "foot_step_walk_start_frame_offset", "robot_r1_1p00_foot_step_walk_start_max_m", "leg_stretch_walk_max_m",
        "overextended_walk_frames_L", "overextended_walk_frames_R", "foot_step_walk_max_m", "source_frames_per_frame_max"]


def fl(levels):
    return " ".join(f"{v:.2f}" for v in levels) or "无"


def load(R, stem):
    rows = list(csv.DictReader(open(R / "clearance_exact" / f"{stem}.csv")))
    fk = {r["clip"]: r for r in csv.DictReader(open(R / "feet_ik.csv"))}
    return rows, fk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--before-stem", default="step9g_all22_clearance_exact")
    ap.add_argument("--after-stem", default="step9g2_clearance_exact")
    ap.add_argument("--before-clearance", type=Path, default=ROOT / "results_system1/T14_retargeting/step9g2",
                    help="directory whose clearance_exact/ holds the before table (step 9g's re-judge lives in step9g2/)")
    a = ap.parse_args()
    B = a.before if a.before.is_absolute() else ROOT / a.before
    A = a.after if a.after.is_absolute() else ROOT / a.after
    bc = a.before_clearance if a.before_clearance.is_absolute() else ROOT / a.before_clearance
    rb = list(csv.DictReader(open(bc / "clearance_exact" / f"{a.before_stem}.csv")))
    fb = {r["clip"]: r for r in csv.DictReader(open(B / "feet_ik.csv"))}
    ra, fa = load(A, a.after_stem)
    out = []
    tot = {}
    for clip in fb:
        row = dict(clip=clip, sequence=fb[clip]["sequence"], object=fb[clip]["object"], walk=fb[clip]["walk"],
                   demo_foot_step_walk_max_m=fb[clip]["demo_foot_step_walk_max_m"])
        for k in FEET:
            row[f"{k}_before"] = fb[clip][k]
            row[f"{k}_after"] = fa[clip][k] if clip in fa else None
        for lab, rows in (("before", rb), ("after", ra)):
            rr = [r for r in rows if r["clip"] == clip and r["reported"] == "True"]
            row[f"reported_levels_evaluated_{lab}"] = fl([float(r["level"]) for r in rr])
            row[f"applicable_levels_{lab}"] = fl([float(r["level"]) for r in rr if r["applicable_C"] == "True"])
            row[f"failed_{lab}"] = " ‖ ".join(f"{r['level']}: " + ("适用" if not r["failed_C"] else "、".join(SHORT.get(x, x) for x in r["failed_C"].split(" | ")))
                                             for r in rr)
            t = tot.setdefault(lab, dict(clips=0, levels=0, reported_evaluated=0))
            t["clips"] += any(r["applicable_C"] == "True" for r in rr)
            t["levels"] += sum(r["applicable_C"] == "True" for r in rr)
            t["reported_evaluated"] += len(rr)
        out.append(row)
    for lab in ("before", "after"):
        tot[lab]["applicable_clip_names"] = [r["clip"] for r in out if r[f"applicable_levels_{lab}"] != "无"]

    def rng(k):
        v = [float(r[k]) for r in out if r[k] not in (None, "")]
        return [min(v), max(v)] if v else None
    tot["ranges_m"] = {f"{k}_{lab}": rng(f"{k}_{lab}") for k in ("foot_step_walk_start_max_m", "robot_r1_1p00_foot_step_walk_start_max_m", "leg_stretch_walk_max_m")
                       for lab in ("before", "after")}
    with open(A / "compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (A / "compare.json").write_text(json.dumps(dict(rules=__doc__, totals=tot), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(tot, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
