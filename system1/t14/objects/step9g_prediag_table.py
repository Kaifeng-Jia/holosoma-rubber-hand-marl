#!/usr/bin/env python3
"""T14 step 9g: table of the pre-solve diagnostics of the mirror construction (step9_gates.py diag9g + prediag_9g) per clip at
the levels 1.0 / 1.25 / 1.5: plane (initial position, travel), demonstration box vs the plane along the walking direction per
segment, distance playback (first / last source frame, largest source frames per partner frame), leg IK (overextended frames,
largest excess, mirrored hip vs reversed hip, knee vs hint) per segment and leg, palms vs contact points (both roles), feet vs
boxes, pelvis and foot steps. Read only.

    python3 system1/t14/objects/step9g_prediag_table.py --logdir logs_system1/t14/step9g --out results_system1/T14_retargeting/step9g/prediag.csv
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PH = ("lift", "walk", "put_down")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L = a.logdir if a.logdir.is_absolute() else ROOT / a.logdir
    rows = []
    for clip in json.loads((L / "driver.json").read_text())["summary"]:
        for m in ("1p00", "1p25", "1p50"):
            f = L / clip / f"gates_m{m}.json"
            if not f.exists():
                continue
            g = json.loads(f.read_text())
            dg, pg = g["diag"]["diag9g"], g["diag"]["prediag_9g"]
            row = dict(clip=clip, level=m.replace("p", "."), walk=f"{g['diag']['walk']['w0']}:{g['diag']['walk']['w1']}", L_m=g["diag"]["L_m"],
                       walk_dir_vs_clip_net_deg=dg["walk_dir_vs_clip_net_deg"], plane_travel_m=dg["plane_travel_m"],
                       demo_box_vs_level_rotation_max_deg=dg["demo_box_vs_level_box_deg"]["rotation_max"], demo_box_tilt_max_deg=dg["demo_box_vs_level_box_deg"]["tilt_max"],
                       tau_first=dg["tau_walk"]["first"], tau_last=dg["tau_walk"]["last"], source_frames_per_frame_max=dg["tau_walk"]["speed_ratio_max"])
            for p in PH:
                x = dg["attached_vs_plane_along_e_m"].get(p)
                row[f"demo_box_vs_plane_{p}_max_abs_m"] = None if x is None else x["max_abs"]
            for sd in ("L", "R"):
                ik = dg["leg_ik"][sd]
                for p in PH:
                    x = ik[p]
                    row[f"ik_{sd}_{p}_overextended_frames"] = x["overextended_frames"]
                    row[f"ik_{sd}_{p}_max_excess_m"] = x["max_excess_m"]
                    row[f"ik_{sd}_{p}_hip_vs_reversed_hip_max_m"] = x["hip_vs_reversed_hip_max_m"]
                    row[f"ik_{sd}_{p}_knee_vs_hint_max_m"] = x["knee_vs_hint_max_m"]
            for hn, v in pg["palms_vs_contact_points"].items():
                for p, x in v.items():
                    row[f"{hn}_{p}_contact_frames"] = x["frames"]
                    row[f"{hn}_{p}_dist_mean_m"] = x["dist_mean"]
                    row[f"{hn}_{p}_dist_max_m"] = x["dist_max"]
            for p, x in pg["feet_vs_boxes"].items():
                row[f"feet_{p}_min_to_shared_box_m"] = x["min_to_shared_box"]
                row[f"feet_{p}_frames_inside_shared"] = x["inside_shared"]
            ps, fs = pg["pelvis_steps"], pg["feet_steps"]
            row.update(pelvis_step_max_partner_m=ps["partner_max_m"], pelvis_step_max_demo_m=ps["demo_max_m"], pelvis_step_at_w0_m=ps["at_w0"],
                       pelvis_step_at_w1_m=ps["at_w1"], foot_step_max_partner_m=fs["partner_max_m"], foot_step_at_w0_m=fs["at_w0"], foot_step_at_w1_m=fs["at_w1"],
                       gravity_deg=g["gravity_deg"], main_motion_passed=g["main_motion"]["passed"])
            rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows")


if __name__ == "__main__":
    raise SystemExit(main())
