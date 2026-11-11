"""Smoke test for system1/t02/object_analysis.py (synthetic data, no files).

    hssim_s1/bin/python -B system1/tests/test_object_analysis.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system1.t02 import object_analysis as oa  # noqa: E402


def quat_xyzw(axis, a):
    v = dict(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1])[axis]
    return np.array([*(np.sin(a / 2) * np.array(v)), np.cos(a / 2)])


def qmul(p, q):  # xyzw
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return np.array([pw * qx + px * qw + py * qz - pz * qy, pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw, pw * qw - px * qx - py * qy - pz * qz])


def main():
    t = np.linspace(0, 1, 11)[:, None]
    ref = np.array([1.0, 2.0, 0.4]) + t * np.array([0.3, 0.4, 0.0])           # moves 0.5 m
    # object follows exactly -> distance 0.5, ratio 1
    m = oa.push_at(ref.copy(), ref, 10)
    assert abs(m["push_distance_m"] - 0.5) < 1e-12 and abs(m["push_ratio"] - 1) < 1e-12, m
    # object moves half way, plus a perpendicular component that must not count
    obj = ref[0] + t * np.array([0.15, 0.2, 0.0]) + t * np.array([-0.4, 0.3, 0.0])
    m = oa.push_at(obj, ref, 10)
    assert abs(m["push_distance_m"] - 0.25) < 1e-12 and abs(m["push_ratio"] - 0.5) < 1e-12, m
    # opposite direction -> negative
    m = oa.push_at(ref[0] - (ref - ref[0]), ref, 10)
    assert abs(m["push_ratio"] + 1) < 1e-12, m
    # static object -> 0; reference displacement below 5 cm -> ratio undefined
    m = oa.push_at(np.repeat(ref[:1], 11, 0), ref, 10)
    assert m["push_distance_m"] == 0 and m["push_ratio"] == 0, m
    m = oa.push_at(ref.copy(), ref, 0)
    assert m["push_distance_m"] is None and m["push_ratio"] is None, m
    m = oa.push_at(ref.copy(), ref, 0 + 0)  # k = 0 -> zero reference displacement
    ref_small = ref[0] + t * np.array([0.03, 0.0, 0.0])
    m = oa.push_at(ref_small.copy(), ref_small, 10)
    assert abs(m["push_distance_m"] - 0.03) < 1e-12 and m["push_ratio"] is None, m

    # pitch: pure pitch, heading invariance, and FK = quaternion product
    for a in (-0.4, 0.0, 0.3, 0.9):
        assert abs(oa.pitch(oa.rot("y", np.array(a))) - a) < 1e-12
        R = oa.rot("z", np.array(1.3)) @ oa.rot("y", np.array(a))
        assert abs(oa.pitch(R) - a) < 1e-12
    rng = np.random.default_rng(0)
    for _ in range(20):
        qp = rng.normal(size=4)
        qp /= np.linalg.norm(qp)
        w = rng.uniform(-0.5, 0.5, size=3)
        q = qmul(qmul(qmul(qp, quat_xyzw("z", w[0])), quat_xyzw("x", w[1])), quat_xyzw("y", w[2]))
        assert np.abs(oa.torso_fk(qp, w) - oa.qmat(q)).max() < 1e-12
    assert oa.first_above(np.array([0.0, 0.1, 0.3, 0.2]), 0.25) == 2 and oa.first_above(np.zeros(3), 0.25) is None
    print("test_object_analysis OK")


if __name__ == "__main__":
    main()
