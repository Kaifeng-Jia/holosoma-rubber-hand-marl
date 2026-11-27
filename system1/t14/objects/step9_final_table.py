#!/usr/bin/env python3
"""T14 final test after step 9g2 (Kevin 2026-11-26): per-clip result table of the frozen 9g2 construction on the plasticbox and
smalltable straight clips, verdict C (non-hand penetration and two-robot clearance on the actual collision shapes, as the 22 debug
clips). Read only; inputs are the package's own tables: clearance_exact/step9_final_clearance_exact.csv (verdict C per level),
nonhand_exact/*.csv (exact non-hand penetration), evals/<clip>_<m>.json (contact retention, support drift as judged, main motion),
feet_ik.csv (foot steps and leg stretch, same definitions as step 9g2's compare.csv) and class_a_turns.csv (the clip list).
Writes clips.csv (one row per clip), levels.csv (one row per evaluated level) and totals.json.

    python3 system1/t14/objects/step9_final_table.py --dir results_system1/T14_retargeting/step9_final
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHORT = {"V0 gravity": "重力", "V0 support drift r0": "漂移 r0", "V0 support drift r1": "漂移 r1", "V0 non-hand penetration r0": "非手 r0",
         "V0 non-hand penetration r1": "非手 r1", "V0 robot-robot clearance": "两机间隙", "section-4 contact retention >= 0.99": "保持率",
         "main motion": "主运动"}
FEET = ["frames", "walk", "smpl_scale", "foot_step_walk_start_max_m", "robot_r1_1p00_foot_step_walk_start_max_m", "demo_foot_step_walk_max_m",
        "leg_stretch_walk_max_m", "overextended_walk_frames_L", "overextended_walk_frames_R"]


def fl(levels):
    return " ".join(f"{v:.2f}" for v in levels) or "无"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "results_system1/T14_retargeting/step9_final")
    ap.add_argument("--stem", default="step9_final_clearance_exact")
    a = ap.parse_args()
    D = a.dir if a.dir.is_absolute() else ROOT / a.dir
    cr = list(csv.DictReader(open(D / "clearance_exact" / f"{a.stem}.csv")))
    nh = {(r["clip"], r["level"]): r for p in sorted((D / "nonhand_exact").glob("*.csv")) for r in csv.DictReader(open(p))}
    fk = {r["clip"]: r for r in csv.DictReader(open(D / "feet_ik.csv"))}
    turns = [r for r in csv.DictReader(open(D / "class_a_turns.csv")) if r["straight"] == "True"]
    levels = []
    for r in cr:
        m = "m" + r["level"].replace(".", "p")
        e = json.loads((D / "evals" / f"{r['clip']}_{m}.json").read_text())
        n = nh[(r["clip"], r["level"])]
        levels.append(dict(
            clip=r["clip"], level=r["level"], reported=r["reported"], applicable_C=r["applicable_C"],
            failed_C="、".join(SHORT.get(x, x) for x in r["failed_C"].split(" | ")) if r["failed_C"] else "",
            contact_retention_r0=e["contact_retention"]["r0"], contact_retention_r1=e["contact_retention"]["r1"],
            support_drift_r0_m=e["support_drift_round6"]["r0"]["value_m"], support_drift_r1_m=e["support_drift_round6"]["r1"]["value_m"],
            nonhand_exact_r0_m=n["new_exact_nonhand_r0_m"], nonhand_exact_link_r0=n["new_exact_link_r0"],
            nonhand_exact_r1_m=n["new_exact_nonhand_r1_m"], nonhand_exact_link_r1=n["new_exact_link_r1"],
            clearance_exact_min_distance_m=r["new_exact_min_distance_m"], clearance_exact_intersect_frames=r["exact_intersect_frames"],
            main_motion_passed=e["main_motion"]["passed"], gravity_deg=e["V0"]["gravity"]["value_deg"],
            eval_failed_own_verdict=" | ".join(e["failed"])))
    clips = []
    for t in turns:
        c = t["sequence"]
        rr = [x for x in levels if x["clip"] == c and x["reported"] == "True"]
        row = dict(clip=c, object=t["object"], clip_frames_30fps=f"{t['start']}:{t['stop']}", heading_vs_walk_max_abs_deg=t["heading_vs_walk_max_abs_deg"],
                   pelvis_net_m=t["pelvis_net_m"], grip=t["grip"],
                   levels_evaluated=fl([float(x["level"]) for x in levels if x["clip"] == c]),
                   reported_levels=fl([float(x["level"]) for x in rr]),
                   applicable_levels_C=fl([float(x["level"]) for x in rr if x["applicable_C"] == "True"]),
                   failed_C=" ‖ ".join(f"{x['level']}: " + (x["failed_C"] or "适用") for x in rr),
                   retention_min=min(min(float(x["contact_retention_r0"]), float(x["contact_retention_r1"])) for x in rr),
                   support_drift_max_m=max(max(float(x["support_drift_r0_m"]), float(x["support_drift_r1_m"])) for x in rr),
                   nonhand_exact_max_m=max(max(float(x["nonhand_exact_r0_m"]), float(x["nonhand_exact_r1_m"])) for x in rr),
                   clearance_exact_min_m=min(float(x["clearance_exact_min_distance_m"]) for x in rr))
        for k in FEET:
            row[k] = fk[c][k]
        clips.append(row)
    tot = dict(clips=len(clips), objects=sorted({c["object"] for c in clips}),
               applicable_clips=[c["clip"] for c in clips if c["applicable_levels_C"] != "无"],
               applicable_levels=sum(x["applicable_C"] == "True" for x in levels if x["reported"] == "True"),
               reported_levels=sum(x["reported"] == "True" for x in levels), evaluated_levels=len(levels),
               failed_item_counts_reported={})
    for x in levels:
        if x["reported"] == "True" and x["failed_C"]:
            for it in x["failed_C"].split("、"):
                tot["failed_item_counts_reported"][it] = tot["failed_item_counts_reported"].get(it, 0) + 1
    for name, rows in (("clips.csv", clips), ("levels.csv", levels)):
        with open(D / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (D / "totals.json").write_text(json.dumps(dict(rules=__doc__, totals=tot), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(tot, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
