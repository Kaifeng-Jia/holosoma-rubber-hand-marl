from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.config_types.retargeter import (
    HandOrientationConfig,
    PlanBPalmContactConfig,
    PTWristOrientationConfig,
)
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.examples.robot_retarget import validate_config
from holosoma_retargeting.src.interaction_mesh_retargeter import (
    InteractionMeshRetargeter,
)


def _model() -> mujoco.MjModel:
    repo_root = Path(__file__).resolve().parents[2]
    xml_path = (
        repo_root
        / "src"
        / "holosoma_retargeting"
        / "holosoma_retargeting"
        / "models"
        / "g1"
        / "g1_29dof_w_largetable.xml"
    )
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _plan_b_geometry_retargeter() -> InteractionMeshRetargeter:
    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = _model()
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.has_dynamic_object = True
    retargeter.q_a_indices = np.arange(36)
    retargeter.plan_b_palm_contact = PlanBPalmContactConfig(enable=True)
    retargeter._hand_orientation_specs = [
        {"side": "left"},
        {"side": "right"},
    ]
    retargeter._init_plan_b_tabletop()
    return retargeter


def test_plan_b_and_a1_are_mutually_exclusive():
    cfg = RetargetingConfig(fixed_object_size_adaptation=True)
    cfg.retargeter = replace(
        cfg.retargeter,
        plan_b_palm_contact=PlanBPalmContactConfig(enable=True),
        pt_wrist_orientation=PTWristOrientationConfig(enable=True),
    )

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        validate_config(cfg)


def test_plan_b_table_targets_are_symmetric_and_face_inward():
    retargeter = _plan_b_geometry_retargeter()
    object_poses = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    weights = np.ones((1, 2))

    positions, normals, twists = retargeter._compute_plan_b_palm_targets(
        object_poses,
        weights,
    )

    np.testing.assert_allclose(positions[0, 0, 0], 0.12)
    np.testing.assert_allclose(positions[0, 1, 0], -0.12)
    np.testing.assert_allclose(positions[0, :, 1], 0.036275)
    np.testing.assert_allclose(positions[0, :, 2], -0.2619764)
    np.testing.assert_allclose(normals[0], [[0.0, 0.0, 1.0]] * 2)
    np.testing.assert_allclose(
        twists[0],
        [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
    )


def test_plan_b_contact_point_lies_on_palm_mesh_support_plane():
    retargeter = _plan_b_geometry_retargeter()
    palm_normals = {
        "left": np.array([-0.07513681, -0.99540367, -0.05938011]),
        "right": np.array([-0.07514936, 0.99540878, -0.05927846]),
    }

    for side, normal in palm_normals.items():
        normal /= np.linalg.norm(normal)
        point = retargeter._derive_palm_contact_point(side, normal)
        geom_id = mujoco.mj_name2id(
            retargeter.robot_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{side}_rubber_hand_link",
        )
        mesh_id = int(retargeter.robot_model.geom_dataid[geom_id])
        start = int(retargeter.robot_model.mesh_vertadr[mesh_id])
        count = int(retargeter.robot_model.mesh_vertnum[mesh_id])
        vertices = retargeter.robot_model.mesh_vert[start : start + count]
        geom_quat = retargeter.robot_model.geom_quat[geom_id]
        geom_rotation = Rotation.from_quat(
            [geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]]
        ).as_matrix()
        vertices = (
            vertices @ geom_rotation.T
            + retargeter.robot_model.geom_pos[geom_id]
        )

        np.testing.assert_allclose(point @ normal, np.max(vertices @ normal))


def test_plan_b_initializes_each_contact_point_from_its_own_mesh():
    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = _model()
    retargeter.has_dynamic_object = True
    retargeter.hand_orientation = HandOrientationConfig()
    retargeter.plan_b_palm_contact = PlanBPalmContactConfig(enable=True)
    retargeter.pt_wrist_orientation = PTWristOrientationConfig()
    retargeter.demo_joints = ["L_Wrist", "R_Wrist"]
    retargeter.laplacian_match_links = {
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    }
    retargeter.q_a_indices = np.arange(36)
    retargeter.q_a_lb = np.full(36, -np.inf)
    retargeter.q_a_ub = np.full(36, np.inf)

    retargeter._init_hand_orientation()

    for spec in retargeter._hand_orientation_specs:
        side = str(spec["side"])
        expected = retargeter._derive_palm_contact_point(
            side,
            np.asarray(spec["palm_normal"], dtype=float),
        )
        np.testing.assert_allclose(spec["palm_contact_point"], expected)


def test_plan_b_phase_order_is_retract_orient_approach():
    orientation = np.zeros((12, 1))
    orientation[4:8, 0] = [0.5, 1.0, 1.0, 0.5]

    position_task, approach = (
        InteractionMeshRetargeter._compute_plan_b_phase_weights(
            orientation,
            delay_frames=2,
        )
    )

    np.testing.assert_allclose(
        position_task[:, 0],
        [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        approach[:, 0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    )
