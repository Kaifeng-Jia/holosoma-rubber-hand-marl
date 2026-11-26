#!/usr/bin/env python3
"""T14 step 9f: table of the pre-solve diagnostics (step9_gates.py prediag_9f; Kevin 2026-11-25 17:10 item 6) of every clip
at the levels 1.0 / 1.25 / 1.5: segment time scales, palms vs contact points, feet vs boxes, pelvis steps. Read only.

    python3 system1/t14/objects/step9f_prediag_table.py --logdir logs_system1/t14/step9f --out results_system1/T14_retargeting/step9f/prediag.csv
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


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
            d = g["diag"]["prediag_9f"]
            sc = {x["segment"].split(" <-")[0].replace(" ", "_"): x for x in d["segments"]}
            row = dict(clip=clip, level=m.replace("p", "."), walk=f"{g['diag']['walk']['w0']}:{g['diag']['walk']['w1']}",
                       scale_lift=sc["lift"]["scale"], scale_walk=sc["walk"]["scale"], scale_put_down=sc["put_down"]["scale"],
                       frames_lift=f"{sc['lift']['partner_frames']}<-{sc['lift']['source_frames']}",
                       frames_put_down=f"{sc['put_down']['partner_frames']}<-{sc['put_down']['source_frames']}",
                       offset_d_m=" ".join(f"{x:.3f}" for x in d["offset_d_m"]),
                       own_box_vs_far_end_mean_m=d["own_box_vs_far_end_m"]["mean"], own_box_vs_far_end_max_m=d["own_box_vs_far_end_m"]["max"])
            for hn, v in d["palms_vs_contact_points"].items():
                for p, x in v.items():
                    row[f"{hn}_{p}_contact_frames"] = x["frames"]
                    row[f"{hn}_{p}_dist_mean_m"] = x["dist_mean"]
                    row[f"{hn}_{p}_dist_max_m"] = x["dist_max"]
            for p, x in d["feet_vs_boxes"].items():
                row[f"feet_{p}_min_to_shared_box_m"] = x["min_to_shared_box"]
                row[f"feet_{p}_frames_inside_shared"] = x["inside_shared"]
                row[f"feet_{p}_min_to_own_box_m"] = x["min_to_own_box"]
            ps = d["pelvis_steps"]
            row.update(pelvis_step_max_partner_m=ps["partner_max_m"], pelvis_step_max_demo_m=ps["demo_max_m"], pelvis_step_at_w0_m=ps["at_w0"],
                       pelvis_step_at_w1_m=ps["at_w1"], gravity_deg=g["gravity_deg"], main_motion_passed=g["main_motion"]["passed"])
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
