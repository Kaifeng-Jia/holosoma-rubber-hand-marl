"""Actual small-table state observations for the CORE4D paired demo."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.observation.terms import marl
from holosoma.utils.rotations import quat_rotate_inverse, yaw_quat


def _actual_table_and_agent_state_w(
    env: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return table position/velocity and both agent root states in world frame."""
    command = marl._paired_motion_command(env)
    table_position = command.simulator_object_pos_w
    table_velocity = env.simulator.all_root_states[
        command.object_indices_in_simulator
    ][:, 7:10]
    agent_root_states = env.simulator.agent_root_states

    if env.num_agents != 2:
        raise ValueError(
            f"CORE4D small-table observations require exactly two agents, got {env.num_agents}"
        )
    expected_table = (env.num_envs, 3)
    expected_agents = (env.num_envs, env.num_agents, 13)
    if table_position.shape != expected_table or table_velocity.shape != expected_table:
        raise ValueError(
            "CORE4D small-table position and linear velocity must have shape "
            f"{expected_table}, got {tuple(table_position.shape)} and "
            f"{tuple(table_velocity.shape)}"
        )
    if agent_root_states.shape != expected_agents:
        raise ValueError(
            "CORE4D small-table observations require agent root states shaped "
            f"{expected_agents}, got {tuple(agent_root_states.shape)}"
        )
    return table_position, table_velocity, agent_root_states


def _observer_headings(agent_root_states: torch.Tensor) -> torch.Tensor:
    """Return each robot's yaw-only quaternion in ``xyzw`` convention."""
    return yaw_quat(agent_root_states[..., 3:7], w_last=True)


def table_relative_position_b(env: Any) -> torch.Tensor:
    """Return table-minus-robot XYZ position in each robot's heading frame."""
    table_position, _, agent_root_states = _actual_table_and_agent_state_w(env)
    relative_position_w = table_position[:, None, :] - agent_root_states[..., :3]
    relative_position_b = quat_rotate_inverse(
        _observer_headings(agent_root_states).flatten(0, 1),
        relative_position_w.flatten(0, 1),
        w_last=True,
    )
    return relative_position_b.reshape(env.num_envs, env.num_agents, 3)


def table_relative_linear_velocity_b(env: Any) -> torch.Tensor:
    """Return table-minus-robot XYZ linear velocity in each heading frame."""
    _, table_velocity, agent_root_states = _actual_table_and_agent_state_w(env)
    relative_velocity_w = table_velocity[:, None, :] - agent_root_states[..., 7:10]
    relative_velocity_b = quat_rotate_inverse(
        _observer_headings(agent_root_states).flatten(0, 1),
        relative_velocity_w.flatten(0, 1),
        w_last=True,
    )
    return relative_velocity_b.reshape(env.num_envs, env.num_agents, 3)


__all__ = [
    "table_relative_linear_velocity_b",
    "table_relative_position_b",
]
