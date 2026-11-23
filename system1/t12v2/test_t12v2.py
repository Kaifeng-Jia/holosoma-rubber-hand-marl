"""Smoke tests for T12-v2 (CPU, no simulation): forward kinematics vs yourdfpy, the union-of-boxes signed distance, the
executed-row rule and the window statistics.

    OMP_NUM_THREADS=1 hssim_s1/bin/python -B -m pytest -q -p no:cacheprovider system1/t12v2/test_t12v2.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_stats as cs  # noqa: E402


@pytest.fixture(scope="module")
def robot():
    return cs.Robot()


def test_fk_matches_yourdfpy(robot):
    rng = np.random.default_rng(0)
    for _ in range(3):
        q = rng.uniform(-0.5, 0.5, len(robot.dofs))
        pos, quat = rng.normal(0, 1, 3), rng.normal(0, 1, 4)
        quat /= np.linalg.norm(quat)
        T = robot.fk(pos[None], quat[None], q[None])
        robot.u.update_cfg(dict(zip(robot.dofs, q)))
        for link in ("left_rubber_hand_link", "right_ankle_roll_link", "head_link"):
            ref = cs.pose_mat(pos, quat) @ robot.u.get_transform(link)
            assert np.allclose(T[link][0], ref, atol=1e-9)


def test_hand_points_follow_t12_rule(robot):
    for h in cs.HANDS:
        assert 1000 <= len(robot.points[h]) <= 1600


def test_box_sdf():
    obj = cs.Object.__new__(cs.Object)
    obj.kind = "boxes"
    obj.box_c = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    obj.box_h = np.array([[0.5, 0.5, 0.5], [0.5, 0.1, 0.1]])
    p = np.array([[0, 0, 0], [0, 0, 0.6], [2.0, 0, 0], [0.75, 0.0, 0.0], [0.0, 0.6, 0.6]])
    s = obj.sdf(p)
    assert np.allclose(s, [0.5, -0.1, -0.5, 0.1, -np.sqrt(0.02)])
    m = trimesh.creation.box(extents=(1, 1, 1))
    q = np.random.default_rng(1).uniform(-1, 1, (200, 3))
    single = cs.Object.__new__(cs.Object)
    single.kind, single.box_c, single.box_h = "boxes", np.zeros((1, 3)), np.full((1, 3), 0.5)
    assert np.allclose(single.sdf(q), trimesh.proximity.signed_distance(m, q), atol=1e-6)


def test_executed_rows():
    assert cs.executed_rows(dict(steps=687, completed_reference=True), 688) == 687
    assert cs.executed_rows(dict(steps=347, completed_reference=False), 348) == 348


def test_window_stats():
    rb = SimpleNamespace(links=list(cs.HANDS) + [x for g in cs.GROUPS.values() for x in g])
    F, L = 10, len(rb.links)
    D = np.full((F, 2, L), -1.0)
    D[2:6, 0, 0] = 0.0      # robot 0 left hand rows 2..5
    D[4:8, 1, 1] = -0.01    # robot 1 right hand rows 4..7 (within 1.5 cm)
    D[5, 1, rb.links.index("torso_link")] = 0.0
    D[6, 0, rb.links.index("left_wrist_yaw_link")] = 0.0
    st = cs.window_stats(D, rb, 2, 8)  # rows 2..7
    assert st["window_rows"] == 6
    assert st["frac_r0_hand"] == pytest.approx(4 / 6) and st["frac_r1_hand"] == pytest.approx(4 / 6)
    assert st["frac_both_hands"] == pytest.approx(2 / 6)
    assert st["frac_r1_nonhand"] == pytest.approx(1 / 6) and st["frac_r1_torso"] == pytest.approx(1 / 6)
    assert st["frac_r0_wrist"] == pytest.approx(1 / 6) and st["frac_r0_nonhand_excl_wrist"] == 0
    assert cs.window_stats(D, rb, 9, 8) == dict(window_rows=0)
