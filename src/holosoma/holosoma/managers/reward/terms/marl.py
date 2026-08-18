"""Plan 5 homogeneous multi-agent reward terms."""

from __future__ import annotations

import re
from typing import Any

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.marl import PairedA1MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude


def _command(env: Any) -> PairedA1MotionCommand:
    command = env.command_manager.get_state("paired_motion_command")
    if not isinstance(command, PairedA1MotionCommand):
        raise TypeError(f"Expected PairedA1MotionCommand, got {type(command)}")
    return command


def motion_global_ref_position_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index]
    error = torch.sum(torch.square(command.agent_ref_pos_w - actual), dim=-1)
    return torch.exp(-error / sigma**2).mean(dim=1)


def motion_global_ref_orientation_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_rot[:, :, command.ref_body_index]
    error = quat_error_magnitude(command.agent_ref_quat_w, actual) ** 2
    return torch.exp(-error / sigma**2).mean(dim=1)


def motion_relative_body_position_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = torch.sum(
        torch.square(command.agent_body_pos_relative_w - command.simulator_agent_body_pos_w),
        dim=-1,
    )
    return torch.exp(-error.mean(dim=-1) / sigma**2).mean(dim=1)


def motion_relative_body_orientation_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = quat_error_magnitude(
        command.agent_body_quat_relative_w,
        command.simulator_agent_body_quat_w,
    ) ** 2
    return torch.exp(-error.mean(dim=-1) / sigma**2).mean(dim=1)


def motion_global_body_lin_vel(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_vel[:, :, command.tracked_body_indexes]
    error = torch.sum(torch.square(command.agent_body_lin_vel_w - actual), dim=-1)
    return torch.exp(-error.mean(dim=-1) / sigma**2).mean(dim=1)


def motion_global_body_ang_vel(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    actual = env.simulator.agent_rigid_body_ang_vel[:, :, command.tracked_body_indexes]
    error = torch.sum(torch.square(command.agent_body_ang_vel_w - actual), dim=-1)
    return torch.exp(-error.mean(dim=-1) / sigma**2).mean(dim=1)


def penalty_action_rate(env: Any) -> torch.Tensor:
    shape = (env.num_envs, env.num_agents, env.num_dof)
    actions = env.action_manager.action.reshape(shape)
    previous = env.action_manager.prev_action.reshape(shape)
    return torch.sum(torch.square(previous - actions), dim=-1).mean(dim=1)


def limits_dof_pos(env: Any, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    limits = env.simulator.hard_dof_pos_limits
    midpoint = (limits[:, 0] + limits[:, 1]) / 2
    width = limits[:, 1] - limits[:, 0]
    lower = midpoint - 0.5 * width * soft_dof_pos_limit
    upper = midpoint + 0.5 * width * soft_dof_pos_limit
    positions = env.simulator.agent_dof_pos
    excess = -(positions - lower).clip(max=0.0)
    excess += (positions - upper).clip(min=0.0)
    return torch.sum(excess, dim=-1).mean(dim=1)


def object_global_ref_position_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = torch.sum(torch.square(command.object_pos_w - command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = quat_error_magnitude(command.object_quat_w, command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


class UndesiredContacts(RewardTermBase):
    """Average the original per-robot incidental-contact penalty over agents."""

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        pattern = cfg.params.get("undesired_contacts_body_names", "")
        body_names = env.simulator.body_names
        indexes = [index for index, name in enumerate(body_names) if re.match(pattern, name)]
        self.indexes = torch.tensor(indexes, dtype=torch.long, device=env.device)
        self.threshold = float(cfg.params.get("threshold", 1.0))

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        history = env.simulator.agent_contact_forces_history
        forces = history[:, :, :, self.indexes]
        contacted = torch.amax(torch.linalg.vector_norm(forces, dim=-1), dim=2) > self.threshold
        return contacted.sum(dim=-1).float().mean(dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        return
