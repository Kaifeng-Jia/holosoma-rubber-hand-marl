"""Observation terms for homogeneous multi-agent compatibility stages."""

from __future__ import annotations

from typing import Any

import torch


def _validated_teammate_buffer(env: Any, attribute: str) -> torch.Tensor:
    if not hasattr(env, attribute):
        raise AttributeError(
            f"{type(env).__name__} does not expose required teammate buffer {attribute!r}; "
            "use GhostTeammateWholeBodyTrackingManager or a real multi-agent environment"
        )
    value = getattr(env, attribute)
    expected_shape = (env.num_envs, 2)
    if not isinstance(value, torch.Tensor) or value.shape != expected_shape:
        actual = getattr(value, "shape", type(value).__name__)
        raise ValueError(f"Expected {attribute} shape {expected_shape}, got {actual}")
    return value


def teammate_relative_position_b(env: Any) -> torch.Tensor:
    """Return teammate planar position in the observing robot's yaw frame."""
    return _validated_teammate_buffer(env, "teammate_relative_position_b")


def teammate_relative_velocity_b(env: Any) -> torch.Tensor:
    """Return teammate planar velocity in the observing robot's yaw frame."""
    return _validated_teammate_buffer(env, "teammate_relative_velocity_b")
