#!/usr/bin/env python3
"""Smoke tests of nonhand_exact.py (T14 step 9g item 1). Uses the step-9g debug level largeboxa 1.0 (logs_system1/t14/step9g).

    sonic_s1/bin/python -B system1/t14/objects/test_nonhand_exact.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import nonhand_exact as nx  # noqa: E402

LEVEL = ROOT / "logs_system1/t14/step9g/largeboxa/m1p00"
SOURCE = ROOT / "logs_system1/t14/step9b/src/largeboxa"


def level_pair():
    recs = [json.loads((LEVEL / f"r{r}/run_record.json").read_text()) for r in (0, 1)]
    zs = [np.load(LEVEL / f"r{r}/{recs[r]['tag']}_new_qpos.npz") for r in (0, 1)]
    T = min(len(z["qpos"]) for z in zs)
    return recs[0], np.stack([z["qpos"][:T, :36].astype(float) for z in zs], 1), zs[0]["object_qpos_mj"][:T].astype(float)


def test_matches_solver_scene():
    """same value as the independent computation on the solver's own scene file (scene_sim.xml, step9g_constraint_links)."""
    import step9g_constraint_links as cl
    m = nx.measure(LEVEL, SOURCE)
    ref = cl.solver_depth(LEVEL)["robots"]
    for r in (0, 1):
        assert abs(m[f"r{r}"]["max_depth_m"] - ref[f"r{r}"]["solver_shape_max_penetration_m"]) <= 2e-4, (r, m[f"r{r}"], ref[f"r{r}"])
    assert m["r1"]["link"] == ref["r1"]["body"] == "left_hip_roll_link" and m["r1"]["frame"] == ref["r1"]["frame"]


def test_box_in_pelvis_and_far_away():
    rec, pair, obj = level_pair()
    ev = nx.distance_model(rec["target"]["urdf"])
    p, o = pair[:1].copy(), obj[:1].copy()
    o[0, :3] = p[0, 0, :3]  # box centre at robot 0's root
    D, W = nx.nonhand_depth(ev, p, o)
    assert D[0, 0] > 0.05 and W[0, 0] is not None and "rubber_hand" not in W[0, 0], (D, W)
    o[0, :3] = p[0, 0, :3] + np.array([0, 0, 50.0])
    D, W = nx.nonhand_depth(ev, p, o)
    assert np.allclose(D[0], -ev.margin) and W[0, 0] is None and W[0, 1] is None


def test_hand_is_excluded():
    """a 2 cm cube at the centre of robot 0's right rubber-hand shape: penetrates the hand only, so the non-hand depth stays
    negative; with the hand filter switched off the same pose gives a positive depth."""
    import contact_stats as cs
    rec, pair, obj = level_pair()
    p = pair[:1].copy()
    with tempfile.TemporaryDirectory() as d:
        u = Path(d) / "cube.urdf"
        u.write_text('<robot name="cube"><link name="cube_link"><collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><box size="0.02 0.02 0.02"/>'
                     '</geometry></collision></link></robot>')
        ev = nx.distance_model(u)
        far = np.array([[0, 0, 50.0, 1, 0, 0, 0]])
        nx.nonhand_depth(ev, p, far)  # places the robot shapes
        g = [i for i, n in ev.geom_link.items() if n == "right_rubber_hand_link" and ev.geom_robot[i] == 0]
        c = ev.d.geom_xpos[g].mean(0)
        o = np.array([[*c, 1, 0, 0, 0]])
        D, _ = nx.nonhand_depth(ev, p, o)
        assert D[0, 0] < 0, D
        hands = cs.HANDS
        try:
            cs.HANDS = ()
            D2, W2 = nx.nonhand_depth(ev, p, o)
        finally:
            cs.HANDS = hands
        assert D2[0, 0] > 0 and W2[0, 0] == "right_rubber_hand_link", (D2, W2)


def test_replace_nonhand():
    import rejudge_nonhand_exact as rj
    ok, bad = dict(passed=True), dict(passed=False)
    old = ["V0 non-hand penetration r0", "V0 non-hand penetration r1", "V0 robot-robot clearance"]
    assert rj.replace_nonhand(old, dict(r0=ok, r1=ok)) == ["V0 robot-robot clearance"]
    assert rj.replace_nonhand(["main motion"], dict(r0=ok, r1=bad)) == ["V0 non-hand penetration r1", "main motion"]
    assert rj.replace_nonhand([], dict(r0=ok, r1=ok)) == []


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
