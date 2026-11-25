#!/usr/bin/env python3
"""T14 step 9e: size of the partner person's step at the lift -> walk (w0) and walk -> put-down (w1) boundaries of the frozen
construction (frontback --fb-legs full_scopy), next to step 9d's (walk_scopy) and to the demonstration's own largest
frame-to-frame motion. Diagnostic only (read only, CPU, nothing solved).

Per clip and level: max over joints of |x(t) - x(t-1)| at t = w0 and t = w1 for the partner's upper body (every joint except
Pelvis, hips, knees, ankles, toes; palms included), its legs (four foot points and the pelvis), and its own box (Gp O: origin
translation and rotation angle); the demonstration's largest per-frame joint displacement over the clip for scale.

    hsretargeting_s1/bin/python -B system1/t14/objects/step9e_jump.py --out <json> [--levels 1.0,1.25,1.5]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
TURNS = ROOT / "results_system1/T14_retargeting/step9/class_a_turns.csv"


def main():
    from scipy.spatial.transform import Rotation as R
    from step9_diag import load_role0
    import frontback as fb
    import omomo_motion as om
    import partner as pt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels", default="1.0,1.25,1.5")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(TURNS)) if r["straight"] == "True"]
    out = {}
    for r in rows:
        fr = (int(r["start"]), int(r["stop"]))
        h0, names, nom, size, s = load_role0(r["object"], r["sequence"], fr)
        O = pt.mj_to_mat(nom)
        walk = fb.walking_segment(om.load_sequences({r["object"]})[r["sequence"]], *fr)
        il = [names.index(n) for n in fb.LOWER]
        iu = [i for i in range(len(names)) if i not in il]
        feet = [names.index(n) for n in ("Pelvis", "L_Ankle", "L_Toe", "R_Ankle", "R_Toe")]
        demo_step = float(np.linalg.norm(np.diff(h0, axis=0), axis=-1).max())
        rec = dict(object=r["object"], frames=list(fr), T=len(O), walk=[walk["w0"], walk["w1"]], demo_max_joint_step_m=round(demo_step, 4), levels={})
        for m in (float(x) for x in a.levels.split(",")):
            lv = {}
            for legs in ("walk_scopy", "full_scopy"):
                C = fb.build(h0, names, O, size, m, turn="world", legs=legs, walk=walk)
                H = C["human1"]
                box = C["Gp"] @ O
                b = {}
                for key, t in (("w0", walk["w0"]), ("w1", walk["w1"])):
                    if not (0 < t < len(O)):
                        b[key] = None
                        continue
                    dj = np.linalg.norm(H[t] - H[t - 1], axis=-1)
                    b[key] = dict(frame=int(t), upper_body_max_m=round(float(dj[iu].max()), 4), upper_body_joint=names[iu[int(dj[iu].argmax())]],
                                  feet_pelvis_max_m=round(float(dj[feet].max()), 4),
                                  own_box_translation_m=round(float(np.linalg.norm(box[t][:3, 3] - box[t - 1][:3, 3])), 4),
                                  own_box_rotation_deg=round(float(np.degrees(np.linalg.norm(R.from_matrix(box[t - 1][:3, :3].T @ box[t][:3, :3]).as_rotvec()))), 2))
                lv[legs] = b
            rec["levels"][f"{m:.2f}"] = lv
        tag = r["sequence"]
        out[tag] = rec
        f = rec["levels"]["1.00"]["full_scopy"]
        print(tag, "walk", rec["walk"], "demo step", rec["demo_max_joint_step_m"], "| 9e w0", f["w0"] and (f["w0"]["upper_body_max_m"], f["w0"]["own_box_translation_m"]),
              "w1", f["w1"] and (f["w1"]["upper_body_max_m"], f["w1"]["own_box_translation_m"]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(rules=__doc__, clips=out), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
