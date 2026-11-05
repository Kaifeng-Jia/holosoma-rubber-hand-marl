"""Bounded real-G1 checks for the optional collision-aware PT palm stage."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import (
    ElasticConstraintConfig,
    HandOrientationConfig,
    PlanBPalmContactConfig,
    PTFullArmOrientationConfig,
    PTPalmCollisionConfig,
    PTWristDominantSurfaceConfig,
    PTWristOrientationConfig,
    RetargeterConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import create_task_constants
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.pt_palm_collision import _contact_manifold_rows


PACKAGE = Path(__file__).resolve().parents[2] / "src/holosoma_retargeting/holosoma_retargeting"


@pytest.fixture
def constants(monkeypatch):
    # The production constructor resolves some model assets relative to package CWD.
    monkeypatch.chdir(PACKAGE)
    return create_task_constants(
        RobotConfig(robot_type="g1"),
        MotionDataConfig(data_format="smplh", robot_type="g1"),
        TaskConfig(object_name="largetable"),
        "object_interaction",
    )


@pytest.fixture
def retargeter(constants):
    return InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path=constants.OBJECT_URDF_FILE,
        activate_foot_sticking=False,
        pt_palm_collision=PTPalmCollisionConfig(enable=True, max_iterations=20),
    )


def _synthetic_qpos(retargeter):
    """Reachable arm pose and remote table; no external demonstration dependency."""
    model = retargeter.robot_model
    q = model.qpos0.copy()
    q[:3] = [0.0, 0.0, 1.2]
    q[3:7] = Rotation.from_euler("xyz", [0.12, -0.09, 0.43]).as_quat()[[3, 0, 1, 2]]
    q[-7:-4] = [3.0, 0.0, 0.4]
    q[-4:] = Rotation.from_euler("x", np.pi / 2).as_quat()[[3, 0, 1, 2]]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for suffix, angle in (
            ("shoulder_pitch", 0.15), ("shoulder_roll", sign * 0.08),
            ("shoulder_yaw", sign * 0.04), ("elbow", 0.65),
            ("wrist_roll", sign * 0.05), ("wrist_pitch", -0.08), ("wrist_yaw", sign * 0.03),
        ):
            joint = model.joint(f"{side}_{suffix}_joint")
            q[int(joint.qposadr[0])] = angle
    return q


def _palm_targets(retargeter, qpos):
    targets = np.empty((len(qpos), 2, 3, 3))
    for frame, q in enumerate(qpos):
        for hand, spec in enumerate(retargeter._hand_orientation_specs):
            rotation, _, _ = retargeter._calc_body_orientation_linearization(q, int(spec["body_id"]))
            targets[frame, hand] = rotation @ np.asarray(spec["palm_basis"])
    return targets


def _contact_manifold_fixture(*, reverse=False, mutate_cache=False):
    """Two contact witnesses with opposite rotational distance derivatives."""
    contacts = [
        SimpleNamespace(
            geom1=2 if reverse else 1, geom2=1 if reverse else 2,
            dist=distance, pos=np.array([x, 0.0, 0.0]),
            frame=np.array([0.0, 0.0, -1.0 if reverse else 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        for x, distance in ((-1.0, -0.001), (1.0, -0.002))
    ]
    # Unselected contact must not contribute a constraint row.
    contacts.append(SimpleNamespace(geom1=0, geom2=2))
    calls = []

    def jacobian(body, point, *, input_world):
        assert input_world is True
        calls.append((body, point.copy()))
        if mutate_cache:
            # FK/Jacobian helpers may refresh MuJoCo's shared contact storage.
            for contact in contacts[:2]:
                contact.pos[:] = 99.0
                contact.frame[:] = 0.0
                contact.dist = 99.0
        result = np.zeros((3, 4))
        if body == 10:  # movable hand; table (body 20) is fixed
            result[2, 1] = 1.0
            result[2, 3] = -point[0]
        return result

    return SimpleNamespace(
        robot_model=SimpleNamespace(geom_bodyid=np.array([0, 10, 20])),
        robot_data=SimpleNamespace(ncon=len(contacts), contact=contacts),
        _calc_contact_jacobian_from_point=jacobian,
    ), calls


def test_contact_manifold_keeps_two_witnesses_for_one_geometry_pair():
    retargeter, calls = _contact_manifold_fixture()
    rows = _contact_manifold_rows(retargeter, [(1, 2)], np.array([1, 3]))
    assert len(rows) == 2
    assert len(calls) == 4
    # Same geom pair, but rotation separates one witness and penetrates the other.
    np.testing.assert_array_equal(np.stack([row for row, _ in rows]), [[-1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_array_equal([distance for _, distance in rows], [-0.001, -0.002])


def test_contact_manifold_preserves_ordered_normal_and_snapshots_before_fk_refresh():
    retargeter, calls = _contact_manifold_fixture(reverse=True, mutate_cache=True)
    # Sorted broad-phase key is opposite the actual table-to-hand contact order.
    rows = _contact_manifold_rows(retargeter, [(1, 2)], np.array([1, 3]))
    assert len(rows) == 2
    np.testing.assert_array_equal(np.stack([row for row, _ in rows]), [[-1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_array_equal([distance for _, distance in rows], [-0.001, -0.002])
    assert [body for body, _ in calls] == [20, 10, 20, 10]
    np.testing.assert_array_equal(
        np.stack([point for _, point in calls]),
        [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    assert np.all(retargeter.robot_data.contact[1].pos == 99.0)


def test_palm_collision_is_disabled_by_default():
    assert not PTPalmCollisionConfig().enable
    config = RetargeterConfig()
    assert not config.pt_palm_collision.enable
    enabled = replace(config, pt_palm_collision=replace(config.pt_palm_collision, enable=True))
    assert enabled.pt_palm_collision.enable
    assert not config.pt_palm_collision.enable
    assert not enabled.pt_wrist_orientation.enable
    assert not enabled.plan_b_palm_contact.enable


@pytest.mark.parametrize("name,config", [
    ("hand_orientation", HandOrientationConfig(enable=True)),
    ("plan_b_palm_contact", PlanBPalmContactConfig(enable=True)),
    ("pt_wrist_orientation", PTWristOrientationConfig(enable=True)),
    ("pt_full_arm_orientation", PTFullArmOrientationConfig(enable=True)),
    ("pt_wrist_dominant_surface", PTWristDominantSurfaceConfig(enable=True)),
])
def test_palm_collision_cannot_be_combined_with_another_hand_mode(constants, name, config):
    with pytest.raises(ValueError, match="mutually exclusive"):
        InteractionMeshRetargeter(
            task_constants=constants,
            object_urdf_path=constants.OBJECT_URDF_FILE,
            pt_palm_collision=PTPalmCollisionConfig(enable=True),
            **{name: config},
        )


@pytest.mark.parametrize("name,value,match", [
    ("enable", False, "disabled"),
    ("orientation_weight", 0.0, "orientation_weight"),
    ("hand_position_weight", np.nan, "hand_position_weight"),
    ("arm_prior_weight", -1.0, "arm_prior_weight"),
    ("correction_temporal_weight", -0.1, "correction_temporal_weight"),
    ("clearance", -0.1, "clearance"),
    ("clearance", 0.1, "collision detection threshold"),
    ("validation_tolerance", np.inf, "validation_tolerance"),
    ("max_iterations", 0, "positive integer"),
    ("max_iterations", True, "positive integer"),
])
def test_invalid_configuration_is_rejected_before_solving(retargeter, name, value, match):
    qpos = _synthetic_qpos(retargeter)[None]
    targets = _palm_targets(retargeter, qpos)
    retargeter.pt_palm_collision = replace(retargeter.pt_palm_collision, **{name: value})
    with pytest.raises(ValueError, match=match):
        retargeter.apply_pt_palm_collision_postprocess(qpos, targets)


@pytest.mark.parametrize("name,value,match", [
    ("activate_obj_non_penetration", False, "requires collisions and joint limits"),
    ("activate_joint_limits", False, "requires collisions and joint limits"),
    ("elastic_constraints", ElasticConstraintConfig(enable=True), "does not use elastic collision slack"),
    ("step_size", 0.0, "step_size"),
])
def test_required_hard_constraints_cannot_be_disabled(retargeter, name, value, match):
    qpos = _synthetic_qpos(retargeter)[None]
    targets = _palm_targets(retargeter, qpos)
    setattr(retargeter, name, value)
    with pytest.raises(ValueError, match=match):
        retargeter.apply_pt_palm_collision_postprocess(qpos, targets)


def test_bad_qpos_and_palm_arrays_are_rejected(retargeter):
    qpos = _synthetic_qpos(retargeter)[None]
    targets = _palm_targets(retargeter, qpos)
    bad_q = qpos.copy()
    bad_q[0, 0] = np.nan
    for candidate in (qpos[:, :-1], qpos[0], qpos[:0], bad_q):
        with pytest.raises(ValueError, match="finite nonempty qpos"):
            retargeter.apply_pt_palm_collision_postprocess(candidate, targets)
    bad_palms = targets.copy()
    bad_palms[0, 0, 0, 0] = np.nan
    for candidate in (targets[:, :1], targets[0], bad_palms):
        with pytest.raises(ValueError, match="finite palm rotations"):
            retargeter.apply_pt_palm_collision_postprocess(qpos, candidate)
    reflection = targets.copy()
    reflection[:, 0, :, 0] *= -1
    for candidate in (np.zeros_like(targets), reflection):
        with pytest.raises(ValueError, match="proper rotation matrices"):
            retargeter.apply_pt_palm_collision_postprocess(qpos, candidate)


def test_enabled_mode_maps_anatomical_targets_to_actual_hand_rotations(retargeter):
    qpos = _synthetic_qpos(retargeter)[None]
    targets = _palm_targets(retargeter, qpos)
    mapped = retargeter._map_pt_palm_orientations_to_robot_links(targets, len(qpos))
    for hand, spec in enumerate(retargeter._hand_orientation_specs):
        rotation, _, _ = retargeter._calc_body_orientation_linearization(qpos[0], int(spec["body_id"]))
        np.testing.assert_allclose(mapped[0, hand], rotation, atol=1e-12)


@pytest.mark.parametrize("hand", [0, 1])
def test_world_frame_arm_jacobians_match_central_differences(retargeter, hand):
    q = _synthetic_qpos(retargeter)
    spec = retargeter._hand_orientation_specs[hand]
    body_id = int(spec["body_id"])
    local_point = np.array([0.025, -0.007, 0.012])
    _, jp, _, jr = retargeter._calc_body_point_linearization(q, body_id, local_point)
    indices = np.concatenate([s["pt_full_arm_qpos_indices"] for s in retargeter._hand_orientation_specs])
    eps = 1e-6
    for index in indices:
        qp, qm = q.copy(), q.copy()
        qp[index] += eps
        qm[index] -= eps
        pp, _, rp, _ = retargeter._calc_body_point_linearization(qp, body_id, local_point)
        pm, _, rm, _ = retargeter._calc_body_point_linearization(qm, body_id, local_point)
        local_index = int(np.flatnonzero(retargeter.q_a_indices == index)[0])
        np.testing.assert_allclose(jp[:, local_index], (pp - pm) / (2 * eps), atol=2e-7, rtol=2e-6)
        world_angular_derivative = Rotation.from_matrix(rp @ rm.T).as_rotvec() / (2 * eps)
        np.testing.assert_allclose(jr[:, local_index], world_angular_derivative, atol=2e-7, rtol=2e-6)


def test_synthetic_targets_improve_with_frozen_nonarm_and_physical_limits(retargeter):
    base = np.repeat(_synthetic_qpos(retargeter)[None], 2, axis=0)
    desired = base.copy()
    for hand, spec in enumerate(retargeter._hand_orientation_specs):
        wrists = np.asarray(spec["pt_full_arm_qpos_indices"])[4:]
        desired[0, wrists] += np.array([0.10, -0.08, 0.14]) * (-1 if hand else 1)
        desired[1, wrists] += np.array([0.12, -0.06, 0.16]) * (-1 if hand else 1)
    targets = _palm_targets(retargeter, desired)
    mapped = retargeter._map_pt_palm_orientations_to_robot_links(targets, len(base))
    before = np.empty((2, 2))
    for frame in range(2):
        for hand, spec in enumerate(retargeter._hand_orientation_specs):
            rotation, _, _ = retargeter._calc_body_orientation_linearization(base[frame], int(spec["body_id"]))
            before[frame, hand] = np.degrees(Rotation.from_matrix(mapped[frame, hand] @ rotation.T).magnitude())
    original = base.copy()
    result, diagnostics = retargeter.apply_pt_palm_collision_postprocess(base, targets)
    np.testing.assert_array_equal(base, original)
    indices = diagnostics["arm_qpos_indices"]
    fixed = np.setdiff1d(np.arange(retargeter.nq), indices)
    np.testing.assert_array_equal(result[:, fixed], base[:, fixed])
    assert np.all(diagnostics["orientation_errors_deg"] < before * 0.5)
    assert np.any(np.abs(result[:, indices] - base[:, indices]) > 1e-3)
    np.testing.assert_allclose(diagnostics["arm_correction_rad"], result[:, indices] - base[:, indices])
    assert np.isfinite(result).all()
    assert np.all(diagnostics["minimum_arm_collision_distance_m"] >= (
        retargeter.pt_palm_collision.clearance - retargeter.pt_palm_collision.validation_tolerance
    ))
    assert np.all((diagnostics["sqp_iterations"] >= 1) & (diagnostics["sqp_iterations"] <= 20))
    for spec in retargeter._hand_orientation_specs:
        idx = np.asarray(spec["pt_full_arm_qpos_indices"])
        assert np.all(result[:, idx] >= spec["pt_full_arm_lower_limits"] - 1e-6)
        assert np.all(result[:, idx] <= spec["pt_full_arm_upper_limits"] + 1e-6)
        local = [int(np.flatnonzero(retargeter.q_a_indices == i)[0]) for i in idx[:4]]
        assert np.all(result[:, idx[:4]] >= retargeter.q_a_lb[local] - 1e-6)
        assert np.all(result[:, idx[:4]] <= retargeter.q_a_ub[local] + 1e-6)
    # This is a kinematic solver fixture, not a balance/contact-force validation.
    assert result.shape == (2, 43)


def test_real_a1_penetrating_palm_target_is_refined_without_moving_lower_body(retargeter):
    """Frame 12 is an actual old-A1 hand/table intersection, not a distant obstacle.

    The tracked base and A1 assets share their lower body and object pose. We ask
    for A1's palm directions, but permit an orientation residual to preserve the
    configured nonlinear collision tolerance. Frozen floor contacts are outside
    this arm-only refinement's scope.
    """
    asset_dir = Path(__file__).resolve().parents[2] / (
        "data/retargeted/rubber_hand_largetable_v1/sub6_largetable_033/a1"
    )
    with np.load(asset_dir / "sub6_largetable_033_fixed_object_base.npz") as data:
        base = data["qpos"][12:13].copy()
    with np.load(asset_dir / "sub6_largetable_033_fixed_object.npz") as data:
        old_a1 = data["qpos"][12:13].copy()
    # Exercise the production budget and weights, not the shorter unit fixture.
    retargeter.pt_palm_collision = PTPalmCollisionConfig(enable=True)
    assert retargeter.pt_palm_collision.max_iterations == 60
    model, state = retargeter.robot_model, retargeter.robot_data
    object_geoms = [
        g for g in retargeter._object_geom_ids
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    hand_geoms = [model.geom(f"{side}_rubber_hand_link").id for side in ("left", "right")]
    shoulder_bodies = {model.body(f"{side}_shoulder_pitch_link").id for side in ("left", "right")}
    arm_geoms = []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = int(model.geom_bodyid[geom])
        while body:
            if body in shoulder_bodies:
                arm_geoms.append(geom)
                break
            body = int(model.body_parentid[body])
    assert set(hand_geoms).issubset(arm_geoms)

    def object_distances(q, geoms):
        # Query MuJoCo directly, independently of the SQP pair/Jacobian helper.
        state.qpos[:] = q
        mujoco.mj_forward(model, state)
        return np.array([
            min(
                mujoco.mj_geomDistance(model, state, geom, obj, 1.0, None)
                for obj in object_geoms
                if ((model.geom_contype[geom] & model.geom_conaffinity[obj])
                    or (model.geom_contype[obj] & model.geom_conaffinity[geom]))
            )
            for geom in geoms
        ])

    assert np.min(object_distances(old_a1[0], hand_geoms)) < -0.08
    targets = _palm_targets(retargeter, old_a1)
    base_palms = _palm_targets(retargeter, base)
    initial_errors = np.degrees(Rotation.from_matrix(
        targets[0] @ base_palms[0].transpose(0, 2, 1)
    ).magnitude())
    original = base.copy()
    result, diagnostics = retargeter.apply_pt_palm_collision_postprocess(base, targets)
    tolerance = retargeter.pt_palm_collision.validation_tolerance
    clearance = retargeter.pt_palm_collision.clearance
    assert np.min(object_distances(result[0], arm_geoms)) >= clearance - tolerance - 1e-8
    actual_hand_distances = object_distances(result[0], hand_geoms)
    np.testing.assert_allclose(diagnostics["hand_object_distance_m"][0], actual_hand_distances, atol=1e-9)
    assert np.all(diagnostics["orientation_errors_deg"][0] < initial_errors * 0.5)
    assert np.all((diagnostics["sqp_iterations"] >= 1) & (diagnostics["sqp_iterations"] <= 60))
    fixed = np.setdiff1d(np.arange(retargeter.nq), diagnostics["arm_qpos_indices"])
    np.testing.assert_array_equal(result[:, fixed], original[:, fixed])
    np.testing.assert_array_equal(base, original)
