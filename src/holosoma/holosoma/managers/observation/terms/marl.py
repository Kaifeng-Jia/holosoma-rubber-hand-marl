"""Observation terms for homogeneous multi-agent compatibility stages."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.utils.rotations import (
    quat_rotate_inverse,
    quaternion_to_matrix,
    subtract_frame_transforms,
    yaw_quat,
)


def _paired_motion_command(env: Any) -> Any:
    command = env.command_manager.get_state("paired_motion_command")
    if command is None:
        raise RuntimeError("Plan 5 observation requires paired_motion_command")
    return command


def paired_motion_command(env: Any) -> torch.Tensor:
    """Return per-agent reference joint position and velocity ``[E, A, 58]``."""
    return _paired_motion_command(env).command


def paired_motion_ref_ori_b(env: Any) -> torch.Tensor:
    """Return each reference-body orientation in its physical robot frame."""
    command = _paired_motion_command(env)
    actual_pos = env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index]
    actual_quat = env.simulator.agent_rigid_body_rot[:, :, command.ref_body_index]
    reference_pos = command.agent_ref_pos_w
    reference_quat = command.agent_ref_quat_w
    _, relative_quat = subtract_frame_transforms(
        actual_pos.flatten(0, 1),
        actual_quat.flatten(0, 1),
        reference_pos.flatten(0, 1),
        reference_quat.flatten(0, 1),
    )
    matrix = quaternion_to_matrix(relative_quat, w_last=True)
    return matrix[..., :2].reshape(env.num_envs, env.num_agents, 6)


def paired_base_ang_vel(env: Any) -> torch.Tensor:
    """Return each robot base angular velocity in its full base frame."""
    root_states = env.simulator.agent_root_states
    values = quat_rotate_inverse(
        root_states[..., 3:7].flatten(0, 1),
        root_states[..., 10:13].flatten(0, 1),
        w_last=True,
    )
    return values.reshape(env.num_envs, env.num_agents, 3)


def paired_dof_pos(env: Any) -> torch.Tensor:
    """Return per-agent joint positions relative to the shared default pose."""
    return env.simulator.agent_dof_pos - env.default_dof_pos[:, None, :]


def paired_dof_vel(env: Any) -> torch.Tensor:
    """Return per-agent joint velocities."""
    return env.simulator.agent_dof_vel


def paired_actions(env: Any) -> torch.Tensor:
    """Return the last action with an explicit agent axis."""
    return env.action_manager.action.reshape(env.num_envs, env.num_agents, env.num_dof)


def _real_teammate_planar_state_b(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    root_states = env.simulator.agent_root_states
    if root_states.shape != (env.num_envs, env.num_agents, 13) or env.num_agents != 2:
        raise ValueError(
            "Plan 5 teammate state requires root states shaped "
            f"[{env.num_envs}, 2, 13], got {tuple(root_states.shape)}"
        )

    teammate_ids = torch.tensor([1, 0], device=root_states.device)
    relative_position_w = root_states[:, teammate_ids, :3] - root_states[:, :, :3]
    relative_velocity_w = root_states[:, teammate_ids, 7:10] - root_states[:, :, 7:10]
    observer_heading = yaw_quat(root_states[:, :, 3:7], w_last=True).flatten(0, 1)

    relative_position_b = quat_rotate_inverse(
        observer_heading,
        relative_position_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    relative_velocity_b = quat_rotate_inverse(
        observer_heading,
        relative_velocity_w.flatten(0, 1),
        w_last=True,
    ).reshape(env.num_envs, env.num_agents, 3)
    return relative_position_b[..., :2], relative_velocity_b[..., :2]


def real_teammate_relative_position_b(env: Any) -> torch.Tensor:
    """Return the other physical robot's planar position in observer heading frame."""
    return _real_teammate_planar_state_b(env)[0]


def real_teammate_relative_velocity_b(env: Any) -> torch.Tensor:
    """Return the other physical robot's planar velocity in observer heading frame."""
    return _real_teammate_planar_state_b(env)[1]


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
