"""Ego-ordered centralized observations for competitive Demo 3."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.observation.terms import marl
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, yaw_quat


DEMO3_CRITIC_OBS_DIM = 527


def _actual_table_state_w(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    command = marl._paired_motion_command(env)
    position = command.simulator_object_pos_w
    orientation = command.simulator_object_quat_w
    linear_velocity = env.simulator.all_root_states[
        command.object_indices_in_simulator
    ][:, 7:10]
    expected = {
        "position": (env.num_envs, 3),
        "orientation": (env.num_envs, 4),
        "linear_velocity": (env.num_envs, 3),
    }
    for name, value in {
        "position": position,
        "orientation": orientation,
        "linear_velocity": linear_velocity,
    }.items():
        if value.shape != expected[name]:
            raise ValueError(
                f"Demo 3 table {name} must have shape {expected[name]}, "
                f"got {tuple(value.shape)}"
            )
    return position, orientation, linear_velocity


def _observer_headings(env: Any) -> torch.Tensor:
    root_states = env.simulator.agent_root_states
    expected = (env.num_envs, env.num_agents, 13)
    if root_states.shape != expected or env.num_agents != 2:
        raise ValueError(
            f"Demo 3 table observation requires root states shaped {expected}, "
            f"got {tuple(root_states.shape)}"
        )
    return yaw_quat(root_states[..., 3:7], w_last=True)


def table_relative_position_b(env: Any) -> torch.Tensor:
    """Return actual table planar position in each observer's heading frame."""
    table_position, _, _ = _actual_table_state_w(env)
    relative_position_w = table_position[:, None, :] - env.simulator.agent_root_states[..., :3]
    relative_position_b = quat_rotate_inverse(
        _observer_headings(env).flatten(0, 1),
        relative_position_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    return relative_position_b[..., :2]


def table_linear_velocity_b(env: Any) -> torch.Tensor:
    """Return actual table world linear velocity in each observer heading frame."""
    _, _, table_velocity = _actual_table_state_w(env)
    velocity_w = table_velocity[:, None, :].expand(-1, env.num_agents, -1)
    velocity_b = quat_rotate_inverse(
        _observer_headings(env).flatten(0, 1),
        velocity_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    return velocity_b[..., :2]


def table_relative_yaw_sin_cos(env: Any) -> torch.Tensor:
    """Return table heading relative to each observer as sine and cosine."""
    _, table_orientation, _ = _actual_table_state_w(env)
    table_local_x_w = quaternion_to_matrix(table_orientation, w_last=True)[..., :, 0]
    table_local_x_w = table_local_x_w[:, None, :].expand(-1, env.num_agents, -1)
    table_local_x_b = quat_rotate_inverse(
        _observer_headings(env).flatten(0, 1),
        table_local_x_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    planar_heading = table_local_x_b[..., :2]
    planar_heading = planar_heading / torch.linalg.vector_norm(
        planar_heading,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    return torch.stack((planar_heading[..., 1], planar_heading[..., 0]), dim=-1)


def _paired_base_lin_vel_b(env: Any) -> torch.Tensor:
    root = env.simulator.agent_root_states
    value = quat_rotate_inverse(
        root[..., 3:7].flatten(0, 1),
        root[..., 7:10].flatten(0, 1),
        w_last=True,
    )
    return value.reshape(env.num_envs, env.num_agents, 3)


def _paired_body_state_b(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    position, quaternion = marl._centralized_agent_body_transform_b(env)
    orientation = marl.quaternion_to_matrix(quaternion, w_last=True)
    return position.flatten(2), orientation[..., :2].flatten(2)


def _per_agent_state(env: Any) -> torch.Tensor:
    ref_position, _ = marl._paired_motion_ref_transform_b(env)
    body_position, body_orientation = _paired_body_state_b(env)
    blocks = (
        marl.paired_actions(env),
        marl.paired_base_ang_vel(env),
        _paired_base_lin_vel_b(env),
        body_orientation,
        body_position,
        marl.paired_dof_pos(env),
        marl.paired_dof_vel(env),
        marl.paired_motion_ref_ori_b(env),
        ref_position,
    )
    state = torch.cat(blocks, dim=-1)
    if state.shape != (env.num_envs, env.num_agents, 228):
        raise ValueError(
            "Demo 3 per-agent critic state must have shape "
            f"({env.num_envs}, {env.num_agents}, 228), got {tuple(state.shape)}"
        )
    return state


def ego_ordered_critic_obs(env: Any) -> torch.Tensor:
    """Return ``[E, 2, 527]`` global state ordered as ego then opponent.

    Both agents see the complete physical state, but each row begins with the
    evaluating agent.  This lets one shared critic predict two different
    competitive returns without introducing role-specific parameters.
    """
    if env.num_agents != 2:
        raise ValueError(f"Demo 3 requires exactly two agents, got {env.num_agents}")
    opponent = torch.tensor([1, 0], dtype=torch.long, device=env.device)
    per_agent = _per_agent_state(env)
    ordered_agents = torch.cat((per_agent, per_agent[:, opponent]), dim=-1)

    command = marl.paired_motion_command(env)
    table = torch.cat(
        (
            table_relative_position_b(env),
            table_linear_velocity_b(env),
            table_relative_yaw_sin_cos(env),
        ),
        dim=-1,
    )
    ordered_table = torch.cat((table, table[:, opponent]), dim=-1)
    phase = marl.centralized_shared_phase(env)[:, None].expand(-1, env.num_agents, -1)
    result = torch.cat((ordered_agents, command, ordered_table, phase), dim=-1)
    expected = (env.num_envs, env.num_agents, DEMO3_CRITIC_OBS_DIM)
    if result.shape != expected:
        raise ValueError(f"Demo 3 critic observation must have shape {expected}, got {tuple(result.shape)}")
    return result


__all__ = [
    "DEMO3_CRITIC_OBS_DIM",
    "ego_ordered_critic_obs",
    "table_linear_velocity_b",
    "table_relative_position_b",
    "table_relative_yaw_sin_cos",
]
