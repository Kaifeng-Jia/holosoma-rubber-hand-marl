"""Smoke tests for T14 step 9 (frontback.py, step8_rejudge.added_drift). Run: sonic_s1/bin/python -B system1/t14/objects/test_step9.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontback as fb  # noqa: E402
import step8_rejudge as rj  # noqa: E402

NAMES = list(fb.LOWER) + ["Torso", "L_Wrist", "R_Wrist"]


def straight_carry(T=61, turn_deg=0.0):
    """person walks 1 m along +x with a time-symmetric profile carrying a 0.3 m box 0.4 m ahead; optional yaw turn of the box."""
    tt = np.linspace(0, 1, T)
    x = 0.5 - 0.5 * np.cos(np.pi * tt)  # symmetric in time
    H = np.zeros((T, len(NAMES), 3))
    for j, n in enumerate(NAMES):
        y = 0.1 if n.startswith("L_") else (-0.1 if n.startswith("R_") else 0.0)
        z = 0.05 if "Toe" in n or "Ankle" in n else 0.9
        H[:, j] = np.stack([x, np.full(T, y), np.full(T, z)], 1)
    H[:, NAMES.index("L_Wrist"), 0] += 0.25
    H[:, NAMES.index("R_Wrist"), 0] += 0.25
    O = np.tile(np.eye(4), (T, 1, 1))
    for t in range(T):
        c, s = np.cos(np.radians(turn_deg * tt[t])), np.sin(np.radians(turn_deg * tt[t]))
        O[t, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        O[t, :3, 3] = [x[t] + 0.4, 0.0, 0.8]
    return H, O


def test_straight_symmetric_walk_is_consistent():
    H, O = straight_carry()
    C = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0)
    assert C["k"] == 0 and C["sigma"] == 1.0 and np.isclose(C["Lm"], 0.6)
    assert np.linalg.norm(C["D"], axis=1).max() < 1e-9  # reversed legs coincide with the object-attached pelvis
    pel1 = C["human1"][:, 0]
    assert np.isclose(pel1[-1, 0] - pel1[0, 0], 1.0)  # the partner moves forward with the box (walking backwards)
    assert np.allclose(C["P"][:, :3, 3], C["A"][:, :3, 3])  # translation = the demonstration's
    w = C["human1"][0, NAMES.index("L_Wrist")]
    # target centre at x = 0.4 + 0.15; the wrist (x = 0.25, y = 0.1) turned 180 deg about it -> (0.85, -0.1): the far end
    assert np.allclose(w[:2], [0.85, -0.1], atol=1e-9)


def test_turn_breaks_consistency():
    H, O = straight_carry(turn_deg=180.0)
    C = fb.build(H, NAMES, O, np.array([0.3, 0.4, 0.3]), 1.0)
    assert np.linalg.norm(C["D"], axis=1).max() > 0.3


def test_added_drift():
    T = 20
    rob, src = np.zeros((T, 3)), np.zeros((T, 3))
    rob[5:10, 0] = np.linspace(0, 0.03, 5)
    flag = np.zeros(T, bool)
    flag[5:10] = True
    v, nw, nf = rj.added_drift(rob, src, flag)
    assert np.isclose(v, 0.03) and nw == 1 and nf == 5
    assert rj.added_drift(rob, src, np.zeros(T, bool))[0] is None


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
