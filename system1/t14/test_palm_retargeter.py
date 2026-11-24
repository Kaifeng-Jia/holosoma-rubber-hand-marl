"""Smoke / switch-off tests for system1/t14/palm_retargeter.py (run with the hsretargeting interpreter).

1. palm_offsets=None: _calc_manipulator_jacobians returns exactly (bit for bit) what InteractionMeshRetargeter returns.
2. palm_offsets given: non-palm keys are bit-identical to the base class; the palm key's position is the link pose applied
   to the offset, and its Jacobian matches a finite difference of that position.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _make(cls, **extra):
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    from holosoma_retargeting.config_types.retargeter import RetargeterConfig
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.task import TaskConfig
    from holosoma_retargeting.examples.robot_retarget import build_retargeter_kwargs_from_config, create_task_constants
    rc = RobotConfig(robot_type="g1", robot_urdf_file=str(ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"))
    const = create_task_constants(rc, MotionDataConfig(data_format="smplx", robot_type="g1"), TaskConfig(object_name="ground"), "robot_only")
    if extra.get("palm_offsets"):
        const.DEMO_JOINTS = list(const.DEMO_JOINTS) + ["L_Palm"]
        const.JOINTS_MAPPING = dict(const.JOINTS_MAPPING) | {"L_Palm": "left_rubber_hand_link"}
    kw = build_retargeter_kwargs_from_config(RetargeterConfig(visualize=False, debug=False), const, None, "robot_only")
    return cls(**kw, **extra)


def _q(r, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros(r.robot_model.nq)
    q[2] = 0.8
    q[3] = 1.0
    lo, hi = r.robot_model.jnt_range[1:, 0], r.robot_model.jnt_range[1:, 1]
    q[7:7 + len(lo)] = rng.uniform(lo, hi)
    return q


def test_switch_off_identical():
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from palm_retargeter import PalmInteractionMeshRetargeter
    base = _make(InteractionMeshRetargeter)
    new = _make(PalmInteractionMeshRetargeter, palm_offsets=None)
    for seed in range(3):
        q = _q(base, seed)
        a = base._calc_manipulator_jacobians(q, base.laplacian_match_links)
        b = new._calc_manipulator_jacobians(q, new.laplacian_match_links)
        for k in a[0]:
            assert np.array_equal(a[0][k], b[0][k]) and np.array_equal(a[1][k], b[1][k]), k


def test_palm_point_position_and_jacobian():
    import mujoco
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from palm_retargeter import PalmInteractionMeshRetargeter
    off = np.array([0.063785, -0.042639, 0.008036])
    r = _make(PalmInteractionMeshRetargeter, palm_offsets={"L_Palm": off})
    base = _make(InteractionMeshRetargeter)
    q = _q(r, 7)
    J, p, _ = r._calc_manipulator_jacobians(q, r.laplacian_match_links)
    Jb, pb, _ = base._calc_manipulator_jacobians(q, base.laplacian_match_links)
    for k in pb:
        assert np.array_equal(p[k], pb[k]) and np.array_equal(J[k], Jb[k]), k
    bid = mujoco.mj_name2id(r.robot_model, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand_link")
    r.robot_data.qpos[:] = q
    mujoco.mj_forward(r.robot_model, r.robot_data)
    expect = r.robot_data.xpos[bid] + r.robot_data.xmat[bid].reshape(3, 3) @ off
    assert np.allclose(p["L_Palm"], expect, atol=1e-12)
    # finite difference along a few joint coordinates (indices into q_a = q[q_a_indices])
    for j in range(7, len(r.q_a_indices)):  # every hinge column (round 3: mujoco 4.0 zeroed them through an enum test)
        qi = r.q_a_indices[j]
        eps = 1e-6
        qp, qm = q.copy(), q.copy()
        qp[qi] += eps
        qm[qi] -= eps
        pp = r._calc_manipulator_jacobians(qp, r.laplacian_match_links)[1]["L_Palm"]
        pm = r._calc_manipulator_jacobians(qm, r.laplacian_match_links)[1]["L_Palm"]
        assert np.allclose((pp - pm) / (2 * eps), J["L_Palm"][:, j], atol=1e-5), j
