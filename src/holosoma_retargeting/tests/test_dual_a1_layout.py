"""Tests for the shared dual-agent A1 layout geometry."""

from __future__ import annotations

import numpy as np
import pytest

from holosoma_retargeting.dual_a1_layout import (
    g1_sagittal_joint_map,
    lateral_offset_trajectory,
    matrix_to_quaternion_wxyz,
    mirror_g1_robot_joint_positions_about_table,
    object_centered_lateral_offset_trajectory,
    quaternion_wxyz_to_matrix,
    robot_velocities_from_joint_positions,
    shifted_robot_motion_channels,
    shifted_robot_positions,
    table_local_x_in_world,
)


G1_TEST_JOINT_NAMES = np.asarray(
    [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "waist_yaw_joint",
        "waist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]
)


def test_identity_table_axis_is_world_x() -> None:
    axis = table_local_x_in_world(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1.0e-12)


def test_quarter_turn_about_world_z_rotates_table_axis() -> None:
    half_angle = np.pi / 4.0
    quat_wxyz = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
    axis = table_local_x_in_world(quat_wxyz)
    np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1.0e-12)


def test_shifted_roots_preserve_midpoint_and_spacing() -> None:
    root = np.array([1.2, -0.5, 0.83])
    first, second = shifted_robot_positions(root, np.array([2.0, 0.0, 0.0, 0.0]), 0.8)
    np.testing.assert_allclose(0.5 * (first + second), root, atol=1.0e-12)
    assert np.linalg.norm(second - first) == pytest.approx(0.8)


def test_rejects_invalid_spacing_and_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="positive"):
        shifted_robot_positions(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0.0)
    with pytest.raises(ValueError, match="zero-norm"):
        table_local_x_in_world(np.zeros(4))


def test_lateral_offsets_are_symmetric_and_have_consistent_velocity() -> None:
    angles = np.linspace(0.0, 0.2, 5)
    quaternions = np.stack(
        (
            np.cos(0.5 * angles),
            np.zeros_like(angles),
            np.zeros_like(angles),
            np.sin(0.5 * angles),
        ),
        axis=-1,
    )
    left_pos, left_vel = lateral_offset_trajectory(quaternions, 0.8, -1, 50)
    right_pos, right_vel = lateral_offset_trajectory(quaternions, 0.8, 1, 50)
    np.testing.assert_allclose(left_pos, -right_pos, atol=1.0e-12)
    np.testing.assert_allclose(left_vel, -right_vel, atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.norm(left_pos, axis=-1), 0.4, atol=1.0e-12)


def test_object_centered_offsets_remove_only_mean_local_x_bias() -> None:
    frames = 5
    object_pos = np.zeros((frames, 3))
    root_pos = np.zeros((frames, 3))
    root_pos[:, 0] = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))

    left_offset, _, source_bias = object_centered_lateral_offset_trajectory(
        root_pos, object_pos, quaternions, 0.8, -1, 50
    )
    right_offset, _, _ = object_centered_lateral_offset_trajectory(
        root_pos, object_pos, quaternions, 0.8, 1, 50
    )
    left_local_x = root_pos[:, 0] + left_offset[:, 0]
    right_local_x = root_pos[:, 0] + right_offset[:, 0]

    assert source_bias == pytest.approx(0.04)
    assert np.mean(left_local_x) == pytest.approx(-0.4)
    assert np.mean(right_local_x) == pytest.approx(0.4)
    np.testing.assert_allclose(right_local_x - left_local_x, 0.8, atol=1.0e-12)
    np.testing.assert_allclose(np.diff(left_local_x), np.diff(root_pos[:, 0]), atol=1.0e-12)


def test_shifted_motion_updates_every_robot_translation_channel_only() -> None:
    frame_count = 4
    joint_pos = np.zeros((frame_count, 36))
    joint_vel = np.zeros((frame_count, 35))
    body_pos = np.zeros((frame_count, 3, 3))
    body_velocity = np.zeros_like(body_pos)
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (frame_count, 1))
    shifted = shifted_robot_motion_channels(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_lin_vel_w=body_velocity,
        object_quat_wxyz=quaternions,
        lateral_spacing=0.8,
        observer_side=-1,
        fps=50,
    )
    expected_offset = np.tile([-0.4, 0.0, 0.0], (frame_count, 1))
    np.testing.assert_allclose(shifted["joint_pos"][:, :3], expected_offset)
    expected_body_offset = np.repeat(expected_offset[:, None, :], body_pos.shape[1], axis=1)
    np.testing.assert_allclose(shifted["body_pos_w"], expected_body_offset)
    np.testing.assert_allclose(shifted["joint_vel"], 0.0)
    np.testing.assert_allclose(shifted["body_lin_vel_w"], 0.0)
    np.testing.assert_allclose(shifted["joint_pos"][:, 3:], joint_pos[:, 3:])


def test_quaternion_matrix_round_trip_is_sign_aware() -> None:
    quaternion = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.9238795325, 0.0, 0.0, 0.3826834324],
        ]
    )
    recovered = matrix_to_quaternion_wxyz(quaternion_wxyz_to_matrix(quaternion))
    np.testing.assert_allclose(np.abs(np.sum(recovered * quaternion, axis=-1)), 1.0, atol=1.0e-9)


def test_g1_joint_map_swaps_sides_and_uses_configured_signs() -> None:
    index_map, sign_mask = g1_sagittal_joint_map(G1_TEST_JOINT_NAMES)
    values = np.arange(len(G1_TEST_JOINT_NAMES), dtype=np.float64)
    mirrored = values[index_map] * sign_mask
    expected = np.asarray([2.0, -3.0, 0.0, -1.0, -4.0, 5.0, -7.0, -6.0])
    np.testing.assert_allclose(mirrored, expected)
    np.testing.assert_allclose(mirrored[index_map] * sign_mask, values)


def test_robot_mirror_about_table_is_an_involution_and_flips_local_x() -> None:
    frames = 4
    table_pos = np.tile([0.1, -0.2, 0.4], (frames, 1))
    angles = np.linspace(0.0, 0.3, frames)
    table_quat = np.stack(
        (np.cos(angles / 2.0), np.zeros(frames), np.zeros(frames), np.sin(angles / 2.0)), axis=-1
    )
    table_x = table_local_x_in_world(table_quat)
    root_pos = table_pos + 0.4 * table_x + np.asarray([0.0, 0.3, 0.5])
    root_quat = np.tile([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], (frames, 1))
    dof_pos = np.arange(frames * len(G1_TEST_JOINT_NAMES), dtype=np.float64).reshape(frames, -1) / 100.0
    joint_pos = np.concatenate((root_pos, root_quat, dof_pos), axis=-1)

    mirrored = mirror_g1_robot_joint_positions_about_table(
        joint_pos, G1_TEST_JOINT_NAMES, table_pos, table_quat
    )
    recovered = mirror_g1_robot_joint_positions_about_table(
        mirrored, G1_TEST_JOINT_NAMES, table_pos, table_quat
    )
    source_local_x = np.sum((joint_pos[:, :3] - table_pos) * table_x, axis=-1)
    mirrored_local_x = np.sum((mirrored[:, :3] - table_pos) * table_x, axis=-1)
    np.testing.assert_allclose(mirrored_local_x, -source_local_x, atol=1.0e-10)
    np.testing.assert_allclose(recovered[:, :3], joint_pos[:, :3], atol=1.0e-10)
    np.testing.assert_allclose(
        np.abs(np.sum(recovered[:, 3:7] * joint_pos[:, 3:7], axis=-1)), 1.0, atol=1.0e-10
    )
    np.testing.assert_allclose(recovered[:, 7:], joint_pos[:, 7:], atol=1.0e-10)
    mirrored_rotation = quaternion_wxyz_to_matrix(mirrored[:, 3:7])
    np.testing.assert_allclose(np.linalg.det(mirrored_rotation), 1.0, atol=1.0e-10)


def test_velocity_reconstruction_matches_stationary_pose() -> None:
    joint_pos = np.zeros((5, 15))
    joint_pos[:, 3] = 1.0
    velocities = robot_velocities_from_joint_positions(joint_pos, fps=50)
    np.testing.assert_allclose(velocities, 0.0, atol=1.0e-12)
