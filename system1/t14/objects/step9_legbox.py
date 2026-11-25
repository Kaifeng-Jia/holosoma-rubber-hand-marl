#!/usr/bin/env python3
"""T14 step 9b: why do partner (role 1) first-stage solves fail at the end of the clip? Diagnostic only, nothing is solved.

Per clip at level 1.0 (frontback.build in the solver world): per frame, |D| = reversed-leg pelvis - object-attached pelvis
(horizontal), and the smallest signed distance from the person's leg joints (knees, ankles, toes) to the person's own box
(oriented box of the source size): role 1 = the partner person vs its own box G(t) O(t); role 0 = the demonstration vs O(t).
Negative = inside the box. Writes <out> (json).

    hsretargeting_s1/bin/python -B system1/t14/objects/step9_legbox.py --out results_system1/T14_retargeting/step9b/legbox.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CLIPS = (("smallbox", "sub4_smallbox_047", (36, 172)), ("largebox", "sub7_largebox_045", (60, 185)), ("suitcase", "sub8_suitcase_006", (65, 193)))
LEGS = ("L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe")


def box_sd(p, M, size):
    q = (p - M[:3, 3]) @ M[:3, :3]
    d = np.abs(q) - np.asarray(size) / 2
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


def main():
    import frontback as fb
    import partner as pt
    from step9_diag import load_role0
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for name, seq, fr in CLIPS:
        h0, names, nom, size, s = load_role0(name, seq, fr)
        O = pt.mj_to_mat(nom)
        C = fb.build(h0, names, O, size, 1.0)
        il = [names.index(n) for n in LEGS]
        T = len(O)
        D = np.linalg.norm(C["D"], axis=1)
        sd1 = np.array([box_sd(C["human1"][t, il], C["G"][t] @ O[t], size).min() for t in range(T)])
        sd0 = np.array([box_sd(h0[t, il], O[t], size).min() for t in range(T)])
        out[seq] = dict(object=name, frames=list(fr), T=T, pelvis_mismatch_m=np.round(D, 4).tolist(), partner_legs_to_own_box_m=np.round(sd1, 4).tolist(),
                        demo_legs_to_own_box_m=np.round(sd0, 4).tolist(), last5=dict(pelvis_mismatch_m=np.round(D[-5:], 3).tolist(),
                                                                                    partner_legs_to_box_m=np.round(sd1[-5:], 3).tolist(), demo_legs_to_box_m=np.round(sd0[-5:], 3).tolist()),
                        partner_min_m=round(float(sd1.min()), 4), partner_min_frame=int(sd1.argmin()), demo_min_m=round(float(sd0.min()), 4),
                        demo_min_frame=int(sd0.argmin()), frames_partner_inside=int((sd1 < 0).sum()))
        print(seq, out[seq]["last5"], "partner min", out[seq]["partner_min_m"], "at", out[seq]["partner_min_frame"], "inside", out[seq]["frames_partner_inside"],
              "demo min", out[seq]["demo_min_m"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(rules=__doc__, clips=out), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
