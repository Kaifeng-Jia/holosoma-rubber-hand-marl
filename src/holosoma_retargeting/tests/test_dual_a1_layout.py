"""Tests for the shared dual-agent A1 layout geometry."""

from __future__ import annotations

import numpy as np
import pytest

from holosoma_retargeting.dual_a1_layout import (
    lateral_offset_trajectory,
    shifted_robot_motion_channels,
    shifted_robot_positions,
    table_local_x_in_world,
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
