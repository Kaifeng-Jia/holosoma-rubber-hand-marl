#!/usr/bin/env python3
"""T14 step 9: does a class-a carry episode walk straight? Information only (nothing is solved or chosen from it).

Per class-a episode of the screen (screen/episodes.csv; smallbox, largebox, suitcase), on the OMOMO original release
(omomo_motion.py quantities, 30 Hz):
  * walking direction = the pelvis net horizontal displacement over the episode (the screen's definition);
  * heading(t) = horizontal (left hip - right hip) x up (the screen's definition); largest |heading - walking direction| over the
    episode frames, and that angle at the episode's first frame (lift);
  * box yaw change = the object's rotation about the vertical from the first to the last episode frame (unwrapped);
  * "straight" = |heading - walking direction| < 45 deg at every frame of the episode (the screen's class-a angle, applied per
    frame instead of to the mean heading).
Typical order = the screen's typical-pick distance (layer1_choice.ranked), OMOMO_new availability as in step 7.

    sonic_s1/bin/python -B system1/t14/objects/step9_turns.py --out results_system1/T14_retargeting/step9/class_a_turns.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import layer1_choice as lc  # noqa: E402
import omomo_motion as om  # noqa: E402

OBJS = ("smallbox", "largebox", "suitcase")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    eps = list(csv.DictReader(open(ROOT / "results_system1/T14_retargeting/step7/screen/episodes.csv")))
    rank = {e["sequence"] + ":" + e["start"]: i for i, e in enumerate(lc.ranked(eps, "a"))}
    cls_a = [e for e in eps if e.get("class_a") == "True" and e["object"] in OBJS]
    seqs = om.load_sequences(set(OBJS))
    rows = []
    for e in cls_a:
        v = seqs[e["sequence"]]
        s, t = int(e["start"]), int(e["stop"])
        J = om.smplx_joints(v)[s:t]
        hd = om.heading(J)[:, :2]
        wd = J[-1, 0, :2] - J[0, 0, :2]
        wd = wd / np.linalg.norm(wd)
        ang = np.degrees(np.arctan2(hd[:, 1], hd[:, 0]) - np.arctan2(wd[1], wd[0]))
        ang = (ang + 180) % 360 - 180
        R = np.asarray(v["obj_rot"], float)[s:t]
        ok = np.isfinite(R).all(axis=(1, 2))
        yaw = np.degrees(np.unwrap(np.arctan2(R[ok, 1, 0], R[ok, 0, 0])))
        key = e["sequence"] + ":" + e["start"]
        rows.append(dict(object=e["object"], sequence=e["sequence"], start=s, stop=t, typical_rank_in_class_a=rank.get(key),
                         in_omomo_new=os.path.exists(f"/home/kevin/holosoma/OMOMO_new/{e['sequence']}.pt"),
                         heading_vs_walk_at_lift_deg=round(float(ang[0]), 1), heading_vs_walk_max_abs_deg=round(float(np.abs(ang).max()), 1),
                         box_yaw_change_deg=round(float(yaw[-1] - yaw[0]), 1) if len(yaw) else None, straight=bool(np.abs(ang).max() < 45.0),
                         pelvis_net_m=e["pelvis_net_m"], steps=e["steps"], grip=e["grip"]))
    rows.sort(key=lambda r: (r["object"], r["typical_rank_in_class_a"] if r["typical_rank_in_class_a"] is not None else 10 ** 6))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for o in OBJS:
        r = [x for x in rows if x["object"] == o]
        st = [x for x in r if x["straight"]]
        first = next((x for x in st if x["in_omomo_new"] and x["typical_rank_in_class_a"] is not None), None)
        summ[o] = dict(class_a_episodes=len(r), straight=len(st), most_typical_straight_in_omomo_new=None if first is None else f"{first['sequence']} {first['start']}-{first['stop'] - 1}")
    a.out.with_suffix(".json").write_text(json.dumps(dict(rules=__doc__, summary=summ), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
