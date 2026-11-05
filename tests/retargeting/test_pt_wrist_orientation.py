from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from holosoma_retargeting.config_types.retargeter import (
    PTFullArmOrientationConfig,
    PTWristDominantSurfaceConfig,
    PTWristOrientationConfig,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (
    InteractionMeshRetargeter,
    _select_pt_wrist_orientation_indices,
)
from holosoma_retargeting.src.utils import (
    build_pt_wrist_palm_orientation_targets,
    load_intermimic_wrist_quaternions,
    normalize_quaternion_sequences_xyzw,
)


def test_load_intermimic_wrist_quaternions_uses_global_xyzw_slots(tmp_path):
    data = torch.zeros((3, 591), dtype=torch.float32)
    global_quaternions = data[:, 383:591].reshape(3, 52, 4)
    global_quaternions[:, 17] = torch.tensor([0.0, 0.0, 0.0, 2.0])
    global_quaternions[:, 36] = torch.tensor([0.0, 0.0, 2.0, 0.0])
    global_quaternions[1, 17] *= -1.0
    global_quaternions[2, 36] *= -1.0
    pt_path = tmp_path / "motion.pt"
    torch.save(data, pt_path)

    wrists = load_intermimic_wrist_quaternions(pt_path)

    assert wrists.shape == (3, 2, 4)
    np.testing.assert_allclose(wrists[:, 0], np.array([[0.0, 0.0, 0.0, 1.0]] * 3))
    np.testing.assert_allclose(wrists[:, 1], np.array([[0.0, 0.0, 1.0, 0.0]] * 3))


def test_pt_wrist_targets_recover_left_and_right_anatomical_palm_frames():
    joint_names = [
        "L_Wrist",
        "L_Index1",
        "L_Middle1",
        "L_Pinky1",
        "R_Wrist",
        "R_Index1",
        "R_Middle1",
        "R_Pinky1",
    ]
    num_frames = 12
    angles = np.linspace(0.0, 0.8, num_frames)
    palm_rotations = Rotation.from_euler(
        "zyx",
        np.column_stack([angles, -0.25 * angles, 0.15 * angles]),
    )
    palm_matrices = palm_rotations.as_matrix()

    human_joints = np.zeros((num_frames, len(joint_names), 3), dtype=float)
    for side in ("L", "R"):
        wrist_idx = joint_names.index(f"{side}_Wrist")
        index_idx = joint_names.index(f"{side}_Index1")
        middle_idx = joint_names.index(f"{side}_Middle1")
        pinky_idx = joint_names.index(f"{side}_Pinky1")
        wrist = np.array([0.0, 0.25 if side == "L" else -0.25, 1.0])

        forward = palm_matrices[:, :, 0]
        across = palm_matrices[:, :, 1]
        human_joints[:, wrist_idx] = wrist
        human_joints[:, middle_idx] = wrist + 0.12 * forward
        if side == "L":
            human_joints[:, index_idx] = wrist + 0.10 * forward + 0.035 * across
            human_joints[:, pinky_idx] = wrist + 0.10 * forward - 0.035 * across
        else:
            human_joints[:, index_idx] = wrist + 0.10 * forward - 0.035 * across
            human_joints[:, pinky_idx] = wrist + 0.10 * forward + 0.035 * across

    wrist_to_palm = (
        Rotation.from_euler("xyz", [0.4, -0.2, 0.1]),
        Rotation.from_euler("xyz", [-0.3, 0.25, -0.15]),
    )
    wrist_quaternions = np.stack(
        [
            (palm_rotations * wrist_to_palm[0].inv()).as_quat(),
            (palm_rotations * wrist_to_palm[1].inv()).as_quat(),
        ],
        axis=1,
    )
    wrist_quaternions[4:, 0] *= -1.0
    wrist_quaternions[7:, 1] *= -1.0
    wrist_quaternions *= 2.0

    targets, calibration_errors = build_pt_wrist_palm_orientation_targets(
        human_joints,
        wrist_quaternions,
        joint_names,
    )

    expected = np.repeat(palm_matrices[:, None], 2, axis=1)
    target_error = (
        Rotation.from_matrix(targets.reshape(-1, 3, 3))
        * Rotation.from_matrix(expected.reshape(-1, 3, 3)).inv()
    ).magnitude()
    assert np.degrees(target_error).max() < 1e-6
    assert calibration_errors.max() < 1e-6


def test_normalize_quaternion_sequences_rejects_zero_norm():
    with np.testing.assert_raises(ValueError):
        normalize_quaternion_sequences_xyzw(np.zeros((2, 2, 4)))


def test_pt_orientation_uses_only_wrist_joint_indices():
    np.testing.assert_array_equal(
        _select_pt_wrist_orientation_indices(np.arange(7)),
        np.array([4, 5, 6]),
    )


def test_pt_orientation_rejects_an_incomplete_arm_group():
    with np.testing.assert_raises(ValueError):
        _select_pt_wrist_orientation_indices(np.arange(6))


def test_wrist_dominant_surface_is_mutually_exclusive_with_existing_modes():
    constants = SimpleNamespace(
        ROBOT_URDF_FILE="unused.urdf",
        OBJECT_NAME="unused",
        FOOT_STICKING_LINKS=[],
        DEMO_JOINTS=[],
        JOINTS_MAPPING={},
    )

    with np.testing.assert_raises_regex(ValueError, "mutually exclusive"):
        InteractionMeshRetargeter(
            task_constants=constants,
            object_urdf_path="unused.urdf",
            pt_full_arm_orientation=PTFullArmOrientationConfig(enable=True),
            pt_wrist_dominant_surface=PTWristDominantSurfaceConfig(enable=True),
        )


def test_a1_postprocess_changes_only_wrist_qpos_and_matches_targets():
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
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = data
    retargeter.nq = model.nq
    retargeter.pt_wrist_orientation = PTWristOrientationConfig(enable=True)
    retargeter._hand_orientation_specs = []

    wrist_qpos_indices = []
    for side in ("left", "right"):
        indices = []
        lower = []
        upper = []
        for suffix in ("wrist_roll", "wrist_pitch", "wrist_yaw"):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_{suffix}_joint",
            )
            indices.append(int(model.jnt_qposadr[joint_id]))
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
        wrist_qpos_indices.extend(indices)
        retargeter._hand_orientation_specs.append(
            {
                "side": side,
                "body_id": mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{side}_rubber_hand_link",
                ),
                "palm_basis": np.eye(3),
                "pt_wrist_qpos_indices": np.asarray(indices),
                "pt_wrist_lower_limits": np.asarray(lower),
                "pt_wrist_upper_limits": np.asarray(upper),
            }
        )

    num_frames = 5
    qpos = np.repeat(model.qpos0[None, :], num_frames, axis=0)
    desired = qpos.copy()
    left_values = np.column_stack(
        [
            np.linspace(0.0, 0.3, num_frames),
            np.linspace(0.0, -0.2, num_frames),
            np.linspace(0.0, 0.1, num_frames),
        ]
    )
    right_values = np.column_stack(
        [
            np.linspace(0.0, -0.25, num_frames),
            np.linspace(0.0, 0.15, num_frames),
            np.linspace(0.0, -0.1, num_frames),
        ]
    )
    desired[:, retargeter._hand_orientation_specs[0]["pt_wrist_qpos_indices"]] = left_values
    desired[:, retargeter._hand_orientation_specs[1]["pt_wrist_qpos_indices"]] = right_values

    palm_targets = np.empty((num_frames, 2, 3, 3))
    for frame_idx in range(num_frames):
        data.qpos[:] = desired[frame_idx]
        mujoco.mj_forward(model, data)
        for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
            palm_targets[frame_idx, hand_idx] = data.xmat[int(spec["body_id"])].reshape(3, 3)

    result, errors_deg = retargeter.apply_pt_wrist_orientation_postprocess(
        qpos,
        palm_targets,
    )

    non_wrist_indices = np.setdiff1d(np.arange(model.nq), np.asarray(wrist_qpos_indices))
    np.testing.assert_array_equal(result[:, non_wrist_indices], qpos[:, non_wrist_indices])
    assert errors_deg.max() < 1e-8


def test_full_arm_postprocess_is_bounded_and_changes_only_the_two_arms():
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
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = data
    retargeter.nq = model.nq
    retargeter.pt_wrist_orientation = PTWristOrientationConfig()
    retargeter.pt_full_arm_orientation = PTFullArmOrientationConfig(
        enable=True,
        correction_temporal_weight=1.0,
    )
    retargeter._hand_orientation_specs = []

    all_arm_indices: list[int] = []
    arm_suffixes = (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
    for side in ("left", "right"):
        indices = []
        lower = []
        upper = []
        for suffix in arm_suffixes:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_{suffix}_joint",
            )
            indices.append(int(model.jnt_qposadr[joint_id]))
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
        all_arm_indices.extend(indices)
        retargeter._hand_orientation_specs.append(
            {
                "side": side,
                "body_id": mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{side}_rubber_hand_link",
                ),
                "palm_basis": np.eye(3),
                "pt_full_arm_qpos_indices": np.asarray(indices),
                "pt_full_arm_lower_limits": np.asarray(lower),
                "pt_full_arm_upper_limits": np.asarray(upper),
            }
        )

    num_frames = 3
    baseline = np.repeat(model.qpos0[None, :], num_frames, axis=0)
    # Give the immutable object coordinates distinctive values so accidental
    # full-qpos writes cannot pass unnoticed.
    baseline[:, -7:] = np.array([0.3, -0.2, 0.7, 1.0, 0.0, 0.0, 0.0])
    desired = baseline.copy()
    for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
        wrist_indices = np.asarray(spec["pt_full_arm_qpos_indices"])[-3:]
        sign = 1.0 if hand_idx == 0 else -1.0
        desired[:, wrist_indices] = sign * np.array([0.12, -0.08, 0.10])

    palm_targets = np.empty((num_frames, 2, 3, 3), dtype=float)
    baseline_errors = np.empty((num_frames, 2), dtype=float)
    for frame_idx in range(num_frames):
        data.qpos[:] = desired[frame_idx]
        mujoco.mj_forward(model, data)
        for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
            palm_targets[frame_idx, hand_idx] = data.xmat[
                int(spec["body_id"])
            ].reshape(3, 3)

        data.qpos[:] = baseline[frame_idx]
        mujoco.mj_forward(model, data)
        for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
            current = data.xmat[int(spec["body_id"])].reshape(3, 3)
            baseline_errors[frame_idx, hand_idx] = np.degrees(
                Rotation.from_matrix(
                    palm_targets[frame_idx, hand_idx] @ current.T
                ).magnitude()
            )

    result, metrics = retargeter.apply_pt_full_arm_orientation_postprocess(
        baseline,
        palm_targets,
    )

    arm_indices = np.asarray(all_arm_indices, dtype=int)
    non_arm_indices = np.setdiff1d(np.arange(model.nq), arm_indices)
    np.testing.assert_array_equal(result[:, non_arm_indices], baseline[:, non_arm_indices])
    assert np.isfinite(result).all()
    assert metrics["orientation_errors_deg"].shape == (num_frames, 2)
    assert metrics["hand_position_errors_m"].shape == (num_frames, 2)
    assert metrics["arm_corrections_rad"].shape == (num_frames, 2, 7)
    assert metrics["arm_steps_rad"].shape == (num_frames, 2, 7)
    assert metrics["correction_steps_rad"].shape == (num_frames, 2, 7)
    assert metrics["solver_success"].all()
    assert np.max(metrics["orientation_errors_deg"]) < np.max(baseline_errors)
    assert np.max(metrics["hand_position_errors_m"]) < 0.01
    np.testing.assert_allclose(
        metrics["correction_steps_rad"][1:],
        np.diff(metrics["arm_corrections_rad"], axis=0),
    )

    for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
        indices = np.asarray(spec["pt_full_arm_qpos_indices"], dtype=int)
        lower = np.asarray(spec["pt_full_arm_lower_limits"])
        upper = np.asarray(spec["pt_full_arm_upper_limits"])
        assert np.all(result[:, indices] >= lower - 1.0e-8)
        assert np.all(result[:, indices] <= upper + 1.0e-8)


def test_wrist_dominant_surface_preserves_support_point_and_non_arm_qpos():
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
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = data
    retargeter.nq = model.nq
    retargeter.pt_wrist_orientation = PTWristOrientationConfig()
    retargeter.pt_full_arm_orientation = PTFullArmOrientationConfig()
    retargeter.pt_wrist_dominant_surface = PTWristDominantSurfaceConfig(
        enable=True
    )
    retargeter._hand_orientation_specs = []

    palm_normals = {
        "left": np.array([-0.07513681, -0.99540367, -0.05938011]),
        "right": np.array([-0.07514936, 0.99540878, -0.05927846]),
    }
    arm_suffixes = (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
    all_arm_indices: list[int] = []
    for side in ("left", "right"):
        indices = []
        lower = []
        upper = []
        for suffix in arm_suffixes:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_{suffix}_joint",
            )
            indices.append(int(model.jnt_qposadr[joint_id]))
            lower.append(float(model.jnt_range[joint_id, 0]))
            upper.append(float(model.jnt_range[joint_id, 1]))
        all_arm_indices.extend(indices)
        palm_normal = palm_normals[side]
        palm_normal /= np.linalg.norm(palm_normal)
        finger = np.array([1.0, 0.0, 0.0])
        finger -= np.dot(finger, palm_normal) * palm_normal
        finger /= np.linalg.norm(finger)
        across = np.cross(palm_normal, finger)
        across /= np.linalg.norm(across)
        retargeter._hand_orientation_specs.append(
            {
                "side": side,
                "body_id": mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{side}_rubber_hand_link",
                ),
                "palm_normal": palm_normal,
                "palm_basis": np.column_stack([finger, across, palm_normal]),
                "palm_contact_point": retargeter._derive_palm_contact_point(
                    side,
                    palm_normal,
                ),
                "pt_full_arm_qpos_indices": np.asarray(indices),
                "pt_full_arm_lower_limits": np.asarray(lower),
                "pt_full_arm_upper_limits": np.asarray(upper),
            }
        )

    num_frames = 3
    baseline = np.repeat(model.qpos0[None, :], num_frames, axis=0)
    baseline[:, -7:] = np.array([0.2, -0.1, 0.7, 1.0, 0.0, 0.0, 0.0])
    desired = baseline.copy()
    for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
        wrist_indices = np.asarray(spec["pt_full_arm_qpos_indices"])[-3:]
        sign = 1.0 if hand_idx == 0 else -1.0
        desired[:, wrist_indices] = sign * np.array([0.15, -0.10, 0.12])

    palm_targets = np.empty((num_frames, 2, 3, 3), dtype=float)
    desired_link_rotations = np.empty_like(palm_targets)
    baseline_normal_errors = np.empty((num_frames, 2), dtype=float)
    for frame_idx in range(num_frames):
        data.qpos[:] = desired[frame_idx]
        mujoco.mj_forward(model, data)
        for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
            desired_link_rotations[frame_idx, hand_idx] = data.xmat[
                int(spec["body_id"])
            ].reshape(3, 3)
            palm_targets[frame_idx, hand_idx] = (
                desired_link_rotations[frame_idx, hand_idx]
                @ np.asarray(spec["palm_basis"])
            )

        data.qpos[:] = baseline[frame_idx]
        mujoco.mj_forward(model, data)
        for hand_idx, spec in enumerate(retargeter._hand_orientation_specs):
            body_id = int(spec["body_id"])
            normal = np.asarray(spec["palm_normal"])
            current_normal = data.xmat[body_id].reshape(3, 3) @ normal
            target_normal = desired_link_rotations[frame_idx, hand_idx] @ normal
            baseline_normal_errors[frame_idx, hand_idx] = np.degrees(
                np.arccos(
                    np.clip(np.dot(current_normal, target_normal), -1.0, 1.0)
                )
            )

    result, metrics = (
        retargeter.apply_pt_wrist_dominant_surface_postprocess(
            baseline,
            palm_targets,
        )
    )

    arm_indices = np.asarray(all_arm_indices, dtype=int)
    non_arm_indices = np.setdiff1d(np.arange(model.nq), arm_indices)
    np.testing.assert_array_equal(result[:, non_arm_indices], baseline[:, non_arm_indices])
    assert metrics["palm_normal_errors_deg"].shape == (num_frames, 2)
    assert metrics["finger_direction_errors_deg"].shape == (num_frames, 2)
    assert metrics["surface_position_errors_m"].shape == (num_frames, 2)
    assert metrics["link_origin_errors_m"].shape == (num_frames, 2)
    assert metrics["arm_corrections_rad"].shape == (num_frames, 2, 7)
    assert metrics["solver_success"].all()
    assert np.max(metrics["palm_normal_errors_deg"]) < np.max(
        baseline_normal_errors
    )
    assert np.max(metrics["surface_position_errors_m"]) < 0.01
    assert np.max(np.abs(metrics["arm_corrections_rad"][:, :, :4])) < np.max(
        np.abs(metrics["arm_corrections_rad"][:, :, 4:])
    )
    np.testing.assert_allclose(
        metrics["correction_steps_rad"][1:],
        np.diff(metrics["arm_corrections_rad"], axis=0),
    )
