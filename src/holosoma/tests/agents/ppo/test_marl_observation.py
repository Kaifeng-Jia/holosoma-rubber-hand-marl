"""Tests for explicit teammate observation buffers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.observation.terms.marl import (
    teammate_relative_position_b,
    teammate_relative_velocity_b,
)


def test_reads_explicit_teammate_buffers() -> None:
    position = torch.tensor([[0.0, 0.8], [0.1, -0.8]])
    velocity = torch.tensor([[0.2, 0.0], [-0.2, 0.0]])
    env = SimpleNamespace(
        num_envs=2,
        teammate_relative_position_b=position,
        teammate_relative_velocity_b=velocity,
    )
    assert teammate_relative_position_b(env) is position
    assert teammate_relative_velocity_b(env) is velocity


def test_missing_buffer_fails_loudly() -> None:
    env = SimpleNamespace(num_envs=1)
    with pytest.raises(AttributeError, match="GhostTeammateWholeBodyTrackingManager"):
        teammate_relative_position_b(env)


def test_wrong_shape_fails_loudly() -> None:
    env = SimpleNamespace(num_envs=2, teammate_relative_position_b=torch.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        teammate_relative_position_b(env)
