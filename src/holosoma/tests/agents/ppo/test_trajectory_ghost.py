"""Unit tests for the trajectory-consistent Stage 1B ghost geometry."""

from __future__ import annotations

import math

import pytest
import torch

from holosoma.envs.wbt.wbt_marl_compat_manager import (
    _finite_difference_by_clip,
    _table_local_x_in_world_xyzw,
)


def test_table_local_x_xyzw_rotates_with_yaw() -> None:
    half_angle = math.pi / 4.0
    quaternion_xyzw = torch.tensor([[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]])
    axis = _table_local_x_in_world_xyzw(quaternion_xyzw)
    torch.testing.assert_close(axis, torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-6, rtol=0.0)


def test_finite_difference_does_not_cross_motion_boundaries() -> None:
    values = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [100.0, 0.0],
            [102.0, 0.0],
        ]
    )
    derivative = _finite_difference_by_clip(
        values,
        torch.tensor([0, 3]),
        torch.tensor([3, 5]),
        dt=1.0,
    )
    torch.testing.assert_close(derivative[:, 0], torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]))


def test_finite_difference_uses_second_order_clip_endpoints() -> None:
    values = torch.tensor([[0.0], [1.0], [4.0], [9.0]])
    derivative = _finite_difference_by_clip(values, torch.tensor([0]), torch.tensor([4]), dt=1.0)
    torch.testing.assert_close(derivative[:, 0], torch.tensor([0.0, 2.0, 4.0, 6.0]))


def test_ghost_geometry_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[N, 4\]"):
        _table_local_x_in_world_xyzw(torch.zeros(4))
    with pytest.raises(ValueError, match="positive"):
        _finite_difference_by_clip(torch.zeros((2, 2)), torch.tensor([0]), torch.tensor([2]), dt=0.0)
