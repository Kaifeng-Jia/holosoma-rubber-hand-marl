from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from holosoma_retargeting.config_types.retargeter import PTWristOrientationConfig
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
