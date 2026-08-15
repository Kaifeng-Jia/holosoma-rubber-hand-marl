"""Tests for the shared dual-agent A1 layout geometry."""

from __future__ import annotations

import numpy as np
import pytest

from holosoma_retargeting.dual_a1_layout import shifted_robot_positions, table_local_x_in_world


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
