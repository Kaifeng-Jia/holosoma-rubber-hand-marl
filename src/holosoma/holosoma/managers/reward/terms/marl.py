"""Plan 5 homogeneous multi-agent reward terms."""

from __future__ import annotations

import math
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


class JointAccelerationPenalty(RewardTermBase):
    """Penalize control-step joint acceleration, averaged over both agents.

    The simulator exposes the latest physical joint velocities after the full
    control decimation.  Keeping the previous 50 Hz sample here therefore
    measures the acceleration seen by the policy, rather than a single 200 Hz
    physics substep.  The first sample after every reset is deliberately zero
    because no valid predecessor exists in the new episode.
    """

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        shape = (env.num_envs, env.num_agents, env.num_dof)
        self._previous_velocity = torch.zeros(shape, device=env.device)
        self._valid_previous = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_value = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        velocity = env.simulator.agent_dof_vel
        if velocity.shape != self._previous_velocity.shape:
            raise ValueError(
                "Plan 5 joint velocity shape changed: "
                f"expected {tuple(self._previous_velocity.shape)}, got {tuple(velocity.shape)}"
            )
        control_dt = float(env.dt)
        if control_dt <= 0.0:
            raise ValueError(f"Control dt must be positive, got {control_dt}")

        acceleration = (velocity - self._previous_velocity) / control_dt
        value = torch.sum(torch.square(acceleration), dim=-1).mean(dim=1)
        value = torch.where(self._valid_previous, value, torch.zeros_like(value))

        self._previous_velocity.copy_(velocity)
        self._valid_previous.fill_(True)
        self._last_value.copy_(value)
        return value

    def snapshot(self) -> torch.Tensor:
        """Return the last computed value without advancing reward state."""
        return self._last_value.clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._previous_velocity.zero_()
            self._valid_previous.zero_()
            self._last_value.zero_()
            return
        self._previous_velocity[env_ids] = 0.0
        self._valid_previous[env_ids] = False
        self._last_value[env_ids] = 0.0


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


def object_global_ref_position_error_exp(
    env: Any, sigma: float, object_z_error_weight: float = 1.0,
) -> torch.Tensor:
    """Track world position with optional weighting of squared vertical error.

    The default follows the original arithmetic exactly for existing demos.
    This is not a height bonus: overshooting reference height is penalized too.
    """
    if (
        type(object_z_error_weight) not in (int, float)
        or not math.isfinite(object_z_error_weight)
        or object_z_error_weight <= 0.0
    ):
        raise ValueError("object_z_error_weight must be positive and finite")
    command = _command(env)
    squared_error = torch.square(command.object_pos_w - command.simulator_object_pos_w)
    if object_z_error_weight != 1.0:
        error = torch.sum(squared_error[..., :2], dim=-1) + object_z_error_weight * squared_error[..., 2]
    else:
        error = torch.sum(squared_error, dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: Any, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = quat_error_magnitude(command.object_quat_w, command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


class ObjectHeightErrorPenalty(RewardTermBase):
    """Unbounded squared world-height error for the shared object's origin.

    Return one positive raw cost per environment, (z_actual - z_ref)^2 / scale^2.
    RewardManager supplies the negative weight and control dt exactly once.
    Both positions use the existing command's world-frame actor origins, not
    the object's COM or a contact-conditioned height. Command lookup is lazy
    because the reward manager is constructed before the command manager.
    """

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        scale_m = cfg.params.get("scale_m", 0.05)
        if type(scale_m) not in (int, float) or not math.isfinite(scale_m) or scale_m <= 0.0:
            raise ValueError("object_height_penalty_scale must be positive and finite")
        self.scale_m = float(scale_m)
        self.last_error_z_m = torch.zeros(env.num_envs, device=env.device)
        self.last_raw_penalty = torch.zeros(env.num_envs, device=env.device)
        self._signed_error_sum = torch.zeros((), device=env.device)
        self._absolute_error_sum = torch.zeros((), device=env.device)
        self._squared_error_sum = torch.zeros((), device=env.device)
        self._penalty_sum = torch.zeros((), device=env.device)
        self._penalty_max = torch.zeros((), device=env.device)
        self._sample_count = 0

    @torch.no_grad()
    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        command = _command(env)
        error_z = command.simulator_object_pos_w[..., 2] - command.object_pos_w[..., 2]
        if error_z.shape != (env.num_envs,):
            raise ValueError("Object height penalty requires one shared object per environment")
        penalty = torch.square(error_z / self.scale_m)
        self.last_error_z_m.copy_(error_z)
        self.last_raw_penalty.copy_(penalty)
        self._signed_error_sum.add_(error_z.sum())
        self._absolute_error_sum.add_(error_z.abs().sum())
        self._squared_error_sum.add_(error_z.square().sum())
        self._penalty_sum.add_(penalty.sum())
        self._penalty_max.copy_(torch.maximum(self._penalty_max, penalty.max()))
        self._sample_count += env.num_envs
        return penalty

    def get_iteration_diagnostics(self, reset: bool = True) -> dict[str, torch.Tensor]:
        """Read all reward-evaluated samples, independently of episode resets.

        Signed error is actual minus reference; RMSE is sqrt(mean(error^2)),
        not mean(abs(error)). Scalars stay on device until the logger reads
        them. Reading diagnostics never recomputes or advances reward state.
        """
        if self._sample_count == 0:
            return {}
        result = {
            "Height/error_z_signed_mean_m": self._signed_error_sum / self._sample_count,
            "Height/error_z_abs_mean_m": self._absolute_error_sum / self._sample_count,
            "Height/error_z_rmse_m": (self._squared_error_sum / self._sample_count).sqrt(),
            "Height/raw_penalty_mean": self._penalty_sum / self._sample_count,
            "Height/raw_penalty_max": self._penalty_max.clone(),
            "Height/sample_count": self._penalty_sum.new_tensor(self._sample_count),
        }
        if reset:
            self._signed_error_sum.zero_()
            self._absolute_error_sum.zero_()
            self._squared_error_sum.zero_()
            self._penalty_sum.zero_()
            self._penalty_max.zero_()
            self._sample_count = 0
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Episode resets clear snapshots, never the iteration-wide statistics.
        if env_ids is None:
            self.last_error_z_m.zero_()
            self.last_raw_penalty.zero_()
        else:
            self.last_error_z_m[env_ids] = 0.0
            self.last_raw_penalty[env_ids] = 0.0


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
