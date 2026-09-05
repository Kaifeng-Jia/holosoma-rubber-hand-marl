"""Per-agent Pull priors and physical table progress for Demo 3."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.demo3_tug import Demo3TugMotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude


def _command(env: Any) -> Demo3TugMotionCommand:
    command = env.command_manager.get_state("paired_motion_command")
    if not isinstance(command, Demo3TugMotionCommand):
        raise TypeError(f"Expected Demo3TugMotionCommand, got {type(command)}")
    return command


def motion_global_ref_position_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index]
    error = torch.sum(torch.square(command.agent_ref_pos_w - actual), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_rot[:, :, command.ref_body_index]
    error = quat_error_magnitude(command.agent_ref_quat_w, actual) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = torch.sum(
        torch.square(command.agent_body_pos_relative_w - command.simulator_agent_body_pos_w),
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = quat_error_magnitude(
        command.agent_body_quat_relative_w,
        command.simulator_agent_body_quat_w,
    ) ** 2
    return torch.exp(-error.mean(dim=-1) / sigma**2)


def motion_global_body_lin_vel(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_vel[:, :, command.tracked_body_indexes]
    error = torch.sum(torch.square(command.agent_body_lin_vel_w - actual), dim=-1)
    return torch.exp(-error.mean(dim=-1) / sigma**2)


def motion_global_body_ang_vel(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_ang_vel[:, :, command.tracked_body_indexes]
    error = torch.sum(torch.square(command.agent_body_ang_vel_w - actual), dim=-1)
    return torch.exp(-error.mean(dim=-1) / sigma**2)


def penalty_action_rate(env: Any) -> torch.Tensor:
    shape = (env.num_envs, env.num_agents, env.num_dof)
    actions = env.action_manager.action.reshape(shape)
    previous = env.action_manager.prev_action.reshape(shape)
    return torch.sum(torch.square(previous - actions), dim=-1)


def limits_dof_pos(env: Any, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    limits = env.simulator.hard_dof_pos_limits
    midpoint = (limits[:, 0] + limits[:, 1]) / 2
    width = limits[:, 1] - limits[:, 0]
    lower = midpoint - 0.5 * width * soft_dof_pos_limit
    upper = midpoint + 0.5 * width * soft_dof_pos_limit
    positions = env.simulator.agent_dof_pos
    excess = -(positions - lower).clip(max=0.0)
    excess += (positions - upper).clip(min=0.0)
    return torch.sum(excess, dim=-1)


def signed_table_progress_velocity(env: Any) -> torch.Tensor:
    """Actual table velocity toward each agent's fixed, opposite Pull axis."""
    command = _command(env)
    velocity_xy = command.simulator_object_lin_vel_w[:, :2]
    return torch.sum(velocity_xy[:, None] * command.agent_pull_axis_w, dim=-1)


class JointAccelerationPenalty(RewardTermBase):
    """Optional 50 Hz per-agent acceleration diagnostic/penalty."""

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        shape = (env.num_envs, env.num_agents, env.num_dof)
        self._previous_velocity = torch.zeros(shape, device=env.device)
        self._valid_previous = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        velocity = env.simulator.agent_dof_vel
        acceleration = (velocity - self._previous_velocity) / float(env.dt)
        value = torch.sum(torch.square(acceleration), dim=-1)
        value = torch.where(self._valid_previous[:, None], value, torch.zeros_like(value))
        self._previous_velocity.copy_(velocity)
        self._valid_previous.fill_(True)
        return value

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._previous_velocity.zero_()
            self._valid_previous.zero_()
        else:
            self._previous_velocity[env_ids] = 0.0
            self._valid_previous[env_ids] = False


__all__ = [
    "JointAccelerationPenalty",
    "limits_dof_pos",
    "motion_global_body_ang_vel",
    "motion_global_body_lin_vel",
    "motion_global_ref_orientation_error_exp",
    "motion_global_ref_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "penalty_action_rate",
    "signed_table_progress_velocity",
]
