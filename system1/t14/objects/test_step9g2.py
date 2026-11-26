#!/usr/bin/env python3
"""Smoke tests, T14 step 9g second fix (Kevin 2026-11-26, received 10:29):
  * item 2, frontback walk_distance: the default ("pelvis") rebuilds step 9g's stored gates exactly (old behaviour unchanged);
    "pelvis_feet" keeps tau's end frames, is monotone, meets x(tau) = x(w0) + x(w1-1) - x(t) on the pelvis-and-ankles mean,
    and changes only the partner's ankles, toes and knees (upper body, pelvis, hips, shared target, plane, p_f unchanged);
  * item 1, rejudge_clearance_exact: the exact verdict is the eval's two-robot minimum distance against -1 cm, B keeps A's
    clearance item, and C differs from B in that item only.
Runs in hsretargeting_s1 (OMOMO loader, as step9_gates.py), no pytest needed:

    hsretargeting_s1/bin/python -B system1/t14/objects/test_step9g2.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
L9G = ROOT / "logs_system1/t14/step9g"
CLIPS = ("sub8_smallbox_045", "suitcasea")


def setup(clip):
    import frontback as fb
    import omomo_motion as om
    from step9_diag import load_role0
    sm = json.loads((L9G / "driver.json").read_text())["summary"][clip]
    fr = tuple(int(x) for x in sm["frames"].split(":"))
    h0, names, nom, size, s = load_role0(sm["object"], sm["sequence"], fr)
    walk = fb.walking_segment(om.load_sequences({sm["object"]})[sm["sequence"]], *fr)
    return fb, h0, names, nom, size, s, walk, fr


def test_default_unchanged():
    import step9_gates as sg
    import partner as pt
    for clip in CLIPS:
        fb, h0, names, nom, size, s, walk, fr = setup(clip)
        for m in (1.0, 1.5):
            g = sg.gates(h0, names, pt.mj_to_mat(nom), size, s, m, "world", "mirror", walk)
            old = json.loads((L9G / clip / f"gates_{sg.level_tag(m)}.json").read_text())
            assert json.loads(json.dumps(g)) == old, (clip, m)
            g2 = sg.gates(h0, names, pt.mj_to_mat(nom), size, s, m, "world", "mirror", walk, "pelvis")
            assert json.loads(json.dumps(g2)) == old, (clip, m)
    print("default (pelvis) = step 9g gates, 2 clips x 2 levels")


def test_pelvis_feet():
    import partner as pt
    for clip in CLIPS:
        fb, h0, names, nom, size, s, walk, fr = setup(clip)
        O = pt.mj_to_mat(nom)
        A = fb.build(h0, names, O, size, 1.0, turn="world", legs="mirror", walk=walk)
        B = fb.build(h0, names, O, size, 1.0, turn="world", legs="mirror", walk=walk, walk_distance="pelvis_feet")
        w0, w1 = walk["w0"], walk["w1"]
        tw = B["tau"][w0:w1]
        assert abs(tw[0] - (w1 - 1)) < 1e-6 and abs(tw[-1] - w0) < 1e-6
        assert np.all(np.diff(tw) <= 1e-9)
        e = np.r_[np.asarray(B["diag9g"]["walk_dir"]), 0.0]
        x = fb.walked_x(h0, names, e, "pelvis_feet")
        xs = np.maximum.accumulate(x[w0:w1])
        xt = np.interp(tw, np.arange(w0, w1), xs)  # x at the fractional source time
        assert np.abs(xt - (xs[0] + xs[-1] - xs)).max() < 1e-6, np.abs(xt - (xs[0] + xs[-1] - xs)).max()
        legs = {names.index(f"{sd}_{j}") for sd in "LR" for j in ("Ankle", "Toe", "Knee")}
        other = [i for i in range(len(names)) if i not in legs]
        assert np.abs(A["human1"][:, other] - B["human1"][:, other]).max() == 0.0
        for k in ("P", "plane", "Sm", "Ob", "own1", "M0", "M1"):
            assert np.array_equal(A[k], B[k]), k
        assert A["diag9g"]["plane_fixed_legs"] == B["diag9g"]["plane_fixed_legs"]
        assert not np.array_equal(A["tau"], B["tau"])
        ia = [names.index(n) for n in ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")]
        st = lambda C: float(np.linalg.norm(np.diff(C["human1"][w0:w0 + 16, ia], axis=0), axis=2).max())  # noqa: E731
        print(clip, f"walk-start foot step {100 * st(A):.1f} -> {100 * st(B):.1f} cm, walked {B['diag9g']['walked_distance_m']} m")


def test_rejudge_clearance():
    R = ROOT / "results_system1/T14_retargeting/step9g2/clearance_exact"
    for stem in ("step8b_clearance_exact", "step9f_clearance_exact", "step9g_all22_clearance_exact"):
        for r in csv.DictReader(open(R / f"{stem}.csv")):
            e = json.loads((ROOT / r["eval"]).read_text())
            ok = e["robot_robot_min_distance_cm"] >= -1.0 - 1e-8
            assert (r["new_exact_clearance_passed"] == "True") == ok
            assert (r["old_proxy_clearance_passed"] == "True") == e["V0"]["robot_robot_clearance"]["passed"]
            fb_ = {x for x in r["failed_B"].split(" | ") if x} - {"V0 robot-robot clearance"}
            fc_ = {x for x in r["failed_C"].split(" | ") if x} - {"V0 robot-robot clearance"}
            assert fb_ == fc_
            assert ("V0 robot-robot clearance" in r["failed_C"]) == (not ok)
    print("clearance re-judge tables consistent with the eval jsons")


if __name__ == "__main__":
    test_rejudge_clearance()
    test_default_unchanged()
    test_pelvis_feet()
    print("ALL PASS")
