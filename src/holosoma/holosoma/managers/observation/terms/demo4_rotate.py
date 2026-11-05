"""Actual table observations for cooperative Demo 4 rotation."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.observation.terms import marl
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, yaw_quat


def _actual_table_state_w(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    command = marl._paired_motion_command(env)
    position = command.simulator_object_pos_w
    orientation = command.simulator_object_quat_w
    linear_velocity = env.simulator.all_root_states[
        command.object_indices_in_simulator
    ][:, 7:10]
    expected = (env.num_envs, 3)
    if position.shape != expected or linear_velocity.shape != expected:
        raise ValueError(
            "Demo 4 table position and linear velocity must have shape "
            f"{expected}, got {tuple(position.shape)} and {tuple(linear_velocity.shape)}"
        )
    if orientation.shape != (env.num_envs, 4):
        raise ValueError(
            "Demo 4 table orientation must have shape "
            f"({env.num_envs}, 4), got {tuple(orientation.shape)}"
        )
    return position, orientation, linear_velocity


def _observer_headings(env: Any) -> torch.Tensor:
    root_states = env.simulator.agent_root_states
    expected = (env.num_envs, env.num_agents, 13)
    if env.num_agents != 2 or root_states.shape != expected:
        raise ValueError(
            f"Demo 4 requires root states shaped {expected}, got {tuple(root_states.shape)}"
        )
    return yaw_quat(root_states[..., 3:7], w_last=True)


def table_relative_position_b(env: Any) -> torch.Tensor:
    """Actual table planar position in each robot's heading frame."""
    table_position, _, _ = _actual_table_state_w(env)
    relative_w = table_position[:, None, :] - env.simulator.agent_root_states[..., :3]
    relative_b = quat_rotate_inverse(
        _observer_headings(env).flatten(0, 1),
        relative_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    return relative_b[..., :2]


def table_linear_velocity_b(env: Any) -> torch.Tensor:
    """Actual table planar linear velocity in each robot's heading frame."""
    _, _, table_velocity = _actual_table_state_w(env)
    velocity_w = table_velocity[:, None, :].expand(-1, env.num_agents, -1)
    velocity_b = quat_rotate_inverse(
        _observer_headings(env).flatten(0, 1),
        velocity_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    return velocity_b[..., :2]


def table_relative_yaw_sin_cos(env: Any) -> torch.Tensor:
    """Actual table heading relative to each robot, without yaw rate."""
    _, table_orientation, _ = _actual_table_state_w(env)
    table_local_x_w = quaternion_to_matrix(table_orientation, w_last=True)[..., :, 0]
    table_local_x_w = table_local_x_w[:, None].expand(-1, env.num_agents, -1)
    table_local_x_b = quat_rotate_inverse(
        _observer_headings(env).flatten(0, 1),
        table_local_x_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    planar = table_local_x_b[..., :2]
    planar = planar / torch.linalg.vector_norm(
        planar, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    return torch.stack((planar[..., 1], planar[..., 0]), dim=-1)


__all__ = [
    "table_linear_velocity_b",
    "table_relative_position_b",
    "table_relative_yaw_sin_cos",
]
