"""Shared physical episode termination terms for Demo 3."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.command.terms.demo3_tug import Demo3TugMotionCommand
from holosoma.utils.rotations import quat_rotate_inverse


def _command(env: Any) -> Demo3TugMotionCommand:
    command = env.command_manager.get_state("paired_motion_command")
    if not isinstance(command, Demo3TugMotionCommand):
        raise TypeError(f"Expected Demo3TugMotionCommand, got {type(command)}")
    return command


def reference_horizon_reached(env: Any) -> torch.Tensor:
    """End after one complete 317-frame Pull prior; no command looping."""
    frames = _command(env).reference.num_frames
    return env.episode_length_buf >= frames


def any_robot_clearly_fallen(
    env: Any,
    *,
    minimum_ref_body_height: float,
    maximum_gravity_z: float,
) -> torch.Tensor:
    """Terminate only an unmistakable fall, never table/reference deviation."""
    command = _command(env)
    position = env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index]
    orientation = env.simulator.agent_rigid_body_rot[:, :, command.ref_body_index]
    gravity_w = torch.zeros_like(position)
    gravity_w[..., 2] = -1.0
    gravity_b = quat_rotate_inverse(
        orientation.flatten(0, 1), gravity_w.flatten(0, 1), w_last=True
    ).reshape_as(position)
    fallen_by_height = position[..., 2] < float(minimum_ref_body_height)
    fallen_by_tilt = gravity_b[..., 2] > float(maximum_gravity_z)
    return torch.any(fallen_by_height | fallen_by_tilt, dim=1)


__all__ = ["any_robot_clearly_fallen", "reference_horizon_reached"]
