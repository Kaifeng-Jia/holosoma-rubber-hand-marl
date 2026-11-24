#!/usr/bin/env python3
"""T14 step 7, layer 1 choice (Kevin 2026-11-24 part 3 item 2): for classes b and c, one most typical sequence per object, at
most 3 objects, objects that a box approximates well first. Rules fixed before any solve:

  * box approximation order: box_assets.quality "box_surface_within_2cm_share" (share of the oriented box's surface within
    2 cm of the captured mesh) of the object's class-c pick sequence, descending; the first 3 objects that have a class member
    available in OMOMO_new (the retargeting pipeline's person input is OMOMO_new; a sequence missing there cannot be solved)
    are used for both classes;
  * most typical per object = the screen's typical-pick distance (screen_omomo.pick: class median, IQR-normalised features,
    both-hand contact share >= 0.8 (c) / 0.5 (b)), smallest first, among sequences present in OMOMO_new.

    sonic_s1/bin/python -B system1/t14/objects/layer1_choice.py --screen results_system1/T14_retargeting/step7/screen --out <json>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_assets as ba  # noqa: E402
import omomo_motion as om  # noqa: E402
import screen_omomo as so  # noqa: E402

OMOMO_NEW = Path("/home/kevin/holosoma/OMOMO_new")


def ranked(episodes, cls):
    cand = [e for e in episodes if e.get(f"class_{cls}") == "True" and float(e["both_hands_share"]) >= so.R_["pick_both_hands_share"][cls]]
    feats = so.R_["pick_features"][cls]
    X = np.array([[float(e[f]) for f in feats] for e in cand])
    med = np.median(X, 0)
    iqr = np.subtract(*np.percentile(X, [75, 25], 0))
    iqr[iqr <= 0] = 1.0
    d = np.linalg.norm((X - med) / iqr, axis=1)
    return [dict(cand[i], typical_distance=round(float(d[i]), 4)) for i in np.argsort(d, kind="stable")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    eps = list(csv.DictReader(open(a.screen / "episodes.csv")))
    sc = json.loads((a.screen / "screen.json").read_text())
    cpick = {p["object"]: p["sequence"] for p in sc["most_typical_per_object"]["c"]}
    seqs = om.load_sequences(set(cpick))
    quality = {}
    for obj, s in cpick.items():
        bf = ba.box_frame(obj, seqs[s])
        quality[obj] = dict(sequence=s, extents_m=np.round(bf["extents_m"], 4).tolist(), **ba.quality(obj, bf))
    order = sorted(quality, key=lambda o: -quality[o]["box_surface_within_2cm_share"])
    rank = {cls: ranked(eps, cls) for cls in "bc"}
    avail = lambda s: (OMOMO_NEW / f"{s}.pt").exists()
    objects = [o for o in order if all(any(e["object"] == o and avail(e["sequence"]) for e in rank[c]) for c in "bc")][:3]
    choice = {}
    for cls in "bc":
        choice[cls] = []
        for o in objects:
            per = [e for e in rank[cls] if e["object"] == o]
            skipped = []
            for e in per:
                if avail(e["sequence"]):
                    choice[cls].append(dict(e, skipped_not_in_omomo_new=skipped))
                    break
                skipped.append(e["sequence"])
    rec = dict(rules=__doc__, box_quality=quality, object_order=order, objects=objects, choice=choice)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    for cls in "bc":
        for e in choice[cls]:
            print(cls, e["object"], e["sequence"], e["start"], e["stop"], e["typical_distance"], e["left_face"], e["right_face"], e.get("push_pull", ""), "skipped", e["skipped_not_in_omomo_new"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
