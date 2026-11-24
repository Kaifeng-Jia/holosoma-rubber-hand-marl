"""Smoke / switch-off tests for the T14 round-6 foot rule (support.py) and the stance lock (palm_retargeter.enable_stance_lock).
Run with the hsretargeting interpreter: python -m pytest -q system1/t14/test_support_stance.py

1. support_flags: a toe that stands still is support, a toe moving at 0.1 m/s is not; frame 0 is never support.
2. stance reference: one anchor per constrained foot link and per support segment (the solver holds 5 sphere links per foot,
   each at its own segment-start position); constant within a segment; a new segment gets a new anchor.
3. switch off: without enable_stance_lock the retargeter has no foot_sticking_reference attribute (base code = previous frame).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_support_flags():
    import support
    T = 120
    toe = np.zeros((T, 2, 3))
    toe[:, 1, 0] = np.arange(T) * 0.1 / support.FPS  # right toe moves at 0.1 m/s
    toe[60:, 0, 0] = (np.arange(T - 60)) * 0.1 / support.FPS  # left toe still, then moves
    f = support.support_flags(toe)
    assert not f[0].any()
    assert f[1:40, 0].all() and not f[80:, 0].any()
    assert not f[:, 1].any()
    seq = support.as_solver_sequence(f)
    assert seq[10] == {"L_Toe": True, "R_Toe": False}


def test_stance_reference_per_link_and_segment():
    from palm_retargeter import PalmInteractionMeshRetargeter
    r = object.__new__(PalmInteractionMeshRetargeter)  # only the stance-lock state is exercised
    flags = np.zeros((10, 2), bool)
    flags[1:4, 0] = True
    flags[6:9, 0] = True
    r.enable_stance_lock(flags)
    a1 = r.foot_sticking_reference("left_foot_sphere_1_link", 1, np.array([0.0, 0.0, 0.0]))
    a2 = r.foot_sticking_reference("left_foot_sphere_2_link", 1, np.array([0.05, 0.0, 0.0]))
    assert not np.allclose(a1, a2)  # each link keeps its own anchor
    b1 = r.foot_sticking_reference("left_foot_sphere_1_link", 3, np.array([0.02, 0.0, 0.0]))
    assert np.array_equal(a1, b1)  # same segment: anchor fixed
    c1 = r.foot_sticking_reference("left_foot_sphere_1_link", 6, np.array([0.3, 0.0, 0.0]))
    assert np.allclose(c1, [0.3, 0.0, 0.0])  # new segment: new anchor


def test_switch_off():
    from test_palm_retargeter import _make
    from palm_retargeter import PalmInteractionMeshRetargeter
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    assert not hasattr(_make(PalmInteractionMeshRetargeter, palm_offsets=None), "foot_sticking_reference")
    assert not hasattr(_make(InteractionMeshRetargeter), "foot_sticking_reference")
