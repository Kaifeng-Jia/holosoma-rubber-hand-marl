#!/usr/bin/env python3
"""T14 step 9g item 2 (Kevin 2026-11-26): one row per clip of the 22-clip run, both verdicts side by side and the two report-only
construction quantities. Reads the re-judge table (rejudge_nonhand_exact.py, step9g_all22_nonhand_exact.csv / .json) and
feet_ik.csv (step9g_feet_ik_table.py); writes clips_both.csv. Read only.

Columns: reported levels (a, 1.0, 1.25, 1.5, first free level above 1.5, as the driver), the ones without a solve, applicable
reported levels under the old measure (per-link boxes) and under the new one (actual collision shapes), the failed items per
reported level under each measure, then from feet_ik.csv the largest leg stretch in the walking segment (both legs, m; > 0 = the
target hip-to-ankle distance exceeds thigh + shank), overextended walking frames per leg, and the largest single-frame foot
displacement over the first 15 frames of the reversed walk (m, its frame offset from w0, the source frames it covers), the
largest over the whole walk, the demonstration's largest over its walk, and the solved partner robot's at level 1.0.

    python3 system1/t14/objects/step9g_clips_both.py --results results_system1/T14_retargeting/step9g
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHORT = {"V0 gravity": "重力", "V0 support drift r0": "漂移 r0", "V0 support drift r1": "漂移 r1", "V0 non-hand penetration r0": "非手 r0",
         "V0 non-hand penetration r1": "非手 r1", "V0 robot-robot clearance": "两机间隙", "section-4 contact retention >= 0.99": "保持率",
         "main motion": "主运动"}


def fl(levels):
    return " ".join(f"{v:.2f}" for v in levels) or "无"


def items(rows, key):
    return " ‖ ".join(f"{r['level']}: " + ("适用" if not r[key] else "、".join(SHORT.get(x, x) for x in r[key].split(" | "))) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()
    R = a.results if a.results.is_absolute() else ROOT / a.results
    summ = json.loads((R / "nonhand_exact/step9g_all22_nonhand_exact.json").read_text())["summary"]
    lv = list(csv.DictReader(open(R / "nonhand_exact/step9g_all22_nonhand_exact.csv")))
    fk = {r["clip"]: r for r in csv.DictReader(open(R / "feet_ik.csv"))}
    out = []
    for clip, s in summ.items():
        rr = [r for r in lv if r["clip"] == clip and r["reported"] == "True"]
        f = fk[clip]
        out.append(dict(clip=clip, sequence=s["sequence"], object=f["object"], frames=f["frames"], walk=f["walk"],
                        reported_levels=fl(s["report_levels"]), unsolved_reported_levels=fl(s["unsolved_report_levels"]),
                        applicable_old_measure=fl(s["applicable_old"]), applicable_new_measure=fl(s["applicable_new"]),
                        failed_old_measure=items(rr, "old_failed"), failed_new_measure=items(rr, "new_failed"),
                        leg_stretch_walk_max_m=f["leg_stretch_walk_max_m"], overextended_walk_frames_L=f["overextended_walk_frames_L"],
                        overextended_walk_frames_R=f["overextended_walk_frames_R"], foot_step_walk_start_max_m=f["foot_step_walk_start_max_m"],
                        foot_step_walk_start_frame_offset=f["foot_step_walk_start_frame_offset"],
                        foot_step_walk_start_source_frames=f["foot_step_walk_start_source_frames"], foot_step_walk_max_m=f["foot_step_walk_max_m"],
                        demo_foot_step_walk_max_m=f["demo_foot_step_walk_max_m"],
                        robot_r1_1p00_foot_step_walk_start_max_m=f["robot_r1_1p00_foot_step_walk_start_max_m"]))
    with open(R / "clips_both.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
