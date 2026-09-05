"""Task success and broad physical-safety terminations for Demo 4."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.command.terms.demo4_rotate import Demo4RotateMotionCommand
from holosoma.utils.rotations import quat_rotate_inverse


def _command(env: Any) -> Demo4RotateMotionCommand:
    command = env.command_manager.get_state("paired_motion_command")
    if not isinstance(command, Demo4RotateMotionCommand):
        raise TypeError(f"Expected Demo4RotateMotionCommand, got {type(command)}")
    return command


def yaw_goal_reached(env: Any) -> torch.Tensor:
    """Succeed at +90 degrees only while the table remains physically upright."""
    return _command(env).goal_reached_safely


def reference_horizon_reached(env: Any) -> torch.Tensor:
    """Timeout after one non-looping 316-frame Pull prior."""
    return env.episode_length_buf >= _command(env).reference.num_frames


def any_robot_clearly_fallen(
    env: Any,
    *,
    minimum_ref_body_height: float,
    maximum_gravity_z: float,
) -> torch.Tensor:
    """End only an unmistakable robot fall, not motion-reference deviation."""
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


def table_physical_safety_exceeded(
    env: Any,
    *,
    maximum_tilt_degrees: float,
    maximum_xy_drift_m: float,
    minimum_height_m: float,
    maximum_height_m: float,
) -> torch.Tensor:
    """Broad runaway/tip guard that does not constrain the desired yaw motion."""
    command = _command(env)
    maximum_tilt = torch.deg2rad(
        torch.tensor(maximum_tilt_degrees, device=env.device)
    )
    position = command.simulator_object_pos_w
    bad_tilt = command.table_tilt_radians > maximum_tilt
    bad_planar_position = command.table_xy_drift > float(maximum_xy_drift_m)
    bad_height = (position[:, 2] < float(minimum_height_m)) | (
        position[:, 2] > float(maximum_height_m)
    )
    return bad_tilt | bad_planar_position | bad_height


__all__ = [
    "any_robot_clearly_fallen",
    "reference_horizon_reached",
    "table_physical_safety_exceeded",
    "yaw_goal_reached",
]
