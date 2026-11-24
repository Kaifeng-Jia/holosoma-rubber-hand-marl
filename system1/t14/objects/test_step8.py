"""Smoke tests for T14 step 8 (partner.py). Run: sonic_s1/bin/python -B system1/t14/objects/test_step8.py (or pytest).
role_solve.py / shared_eval.py are smoke-run on real data (12 frames, README)."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partner as pt  # noqa: E402


def traj(T, d=np.zeros(3), tilt_deg=0.0, yaw0=0.0):
    O = np.tile(np.eye(4), (T, 1, 1))
    for t in range(T):
        f = t / (T - 1)
        O[t, :3, :3] = Rotation.from_euler("zy", [yaw0, tilt_deg * f], degrees=True).as_matrix()
        O[t, :3, 3] = np.array([1.0, 2.0, 0.2]) + f * d
    return O


def cand(Q0, c0, Q1, c1, axis=0, L1=0.6):
    return dict(Q0=np.asarray(Q0).tolist(), c0=list(c0), Q1=np.asarray(Q1).tolist(), c1=list(c1), axis=axis, L1=L1)


I, YAW = np.eye(3), np.diag([-1.0, -1.0, 1.0])


def test_identity():
    O = traj(5, d=np.array([0.3, 0, 0]))
    C = pt.construction(O, cand(I, [0, 0, 0], I, [0, 0, 0]), 0, 0.6, 0.6)
    assert np.allclose(C["S"], np.eye(4)) and np.allclose(C["P"], O)


def test_translation_keeps_motion_and_feet():
    O = traj(5, d=np.array([0.6, 0, 0]), yaw0=30)
    C = pt.construction(O, cand(I, [-0.15, 0, 0], I, [0.15, 0, 0]), 0, 0.6, 0.6)
    assert pt.gravity_deg(C["S"]) < 1e-9 and abs(C["S"][2, 3]) < 1e-12  # copy stays on the floor
    assert np.isclose(np.linalg.norm(C["S"][:3, 3]), 0.3)  # S moves the source by the region spacing
    assert np.allclose(np.diff(C["P"][:, :3, 3], axis=0), np.diff(O[:, :3, 3], axis=0))  # pure translation: same motion
    mm = pt.main_motion(C["P"], pt.box_corners([0.6, 0.3, 0.2]), "b", 1.0)
    assert mm["passed"] and np.isclose(mm["travel_m"], 0.6)


def test_facing_cancels_horizontal_and_tilt():
    O = traj(9, d=np.array([0.6, 0, 0.3]), tilt_deg=20)
    C = pt.construction(O, cand(I, [-0.15, 0, 0], YAW, [0.15, 0, 0]), 0, 0.6, 0.6)
    assert pt.gravity_deg(C["S"]) < 1e-9
    dp = C["P"][-1, :3, 3] - C["P"][0, :3, 3]
    # pulls toward each one cancel; the lift stays (minus the lever 0.15 m x sin 20 deg of each role's tilt about its own object origin)
    assert abs(dp[0]) < 1e-9 and abs(dp[1]) < 1e-9 and np.isclose(dp[2], 0.3 - 0.15 * np.sin(np.radians(20)), atol=1e-3)
    up = C["P"][-1, :3, :3] @ np.array([0, 0, 1.0])
    assert np.isclose(up[2], 1.0)  # one-sided tilts cancel
    mm = pt.main_motion(C["P"], pt.box_corners([0.6, 0.3, 0.2]), "b", 1.0)
    assert not mm["passed"]


def test_gravity_upside_down():
    C = pt.construction(traj(3), cand(I, [0, 0, 0], np.diag([1.0, -1.0, -1.0]), [0, 0, 0]), 0, 0.6, 0.6)
    assert np.isclose(pt.gravity_deg(C["S"]), 180.0)


def test_level_maps():
    Q0, c0, Q1, c1 = pt.region_maps(cand(I, [-0.15, 0, 0], I, [0.15, 0, 0]), 0, 0.6, 0.9)
    assert np.isclose(c1[0] - c0[0], 0.3 + 0.3)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
