"""Stateful cooperative yaw rewards for Demo 4 rotation."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.demo4_rotate import Demo4RotateMotionCommand
from holosoma.managers.reward.base import RewardTermBase


def _command(env: Any) -> Demo4RotateMotionCommand:
    command = env.command_manager.get_state("paired_motion_command")
    if not isinstance(command, Demo4RotateMotionCommand):
        raise TypeError(f"Expected Demo4RotateMotionCommand, got {type(command)}")
    return command


def _control_dt(env: Any) -> float:
    value = float(env.dt)
    if value <= 0.0:
        raise ValueError(f"Demo 4 control dt must be positive, got {value}")
    return value


def _safe_goal_mask(env: Any) -> torch.Tensor:
    termination_manager = getattr(env, "termination_manager", None)
    results = getattr(termination_manager, "term_results", {})
    resolved = results.get("yaw_goal_success")
    if isinstance(resolved, torch.Tensor):
        return resolved
    return _command(env).goal_reached_safely


def _unsafe_terminal_mask(env: Any) -> torch.Tensor | None:
    termination_manager = getattr(env, "termination_manager", None)
    results = getattr(termination_manager, "term_results", {})
    robot_fall = results.get("clear_robot_fall")
    table_safety = results.get("table_physical_safety")
    if isinstance(robot_fall, torch.Tensor) and isinstance(
        table_safety, torch.Tensor
    ):
        return robot_fall | table_safety
    return None


class SignedYawPotentialDelta(RewardTermBase):
    """Reward signed changes in clipped normalized yaw progress.

    RewardManager multiplies the returned derivative by ``dt``.  A weight of
    10 therefore gives exactly 10 integrated reward units for progress from
    zero to the +90-degree goal, independent of control frequency.
    """

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)
        self._last_delta = torch.zeros_like(self._previous_potential)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        potential = _command(env).normalized_yaw_progress
        delta = potential - self._previous_potential
        self._previous_potential.copy_(potential.detach())
        unsafe = _unsafe_terminal_mask(env)
        if unsafe is not None:
            delta = torch.where(unsafe, torch.zeros_like(delta), delta)
        self._last_delta.copy_(delta.detach())
        return delta / _control_dt(env)

    def snapshot(self) -> torch.Tensor:
        return self._last_delta.clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._previous_potential.zero_()
            self._last_delta.zero_()
            return
        self._previous_potential[env_ids] = 0.0
        self._last_delta[env_ids] = 0.0


class FirstYawGoalBonus(RewardTermBase):
    """Emit one dt-normalized impulse when the table first reaches +90 degrees."""

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        self._awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        reached = _safe_goal_mask(env)
        first = reached & ~self._awarded
        self._awarded |= reached
        return first.to(dtype=torch.float32) / _control_dt(env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._awarded.zero_()
        else:
            self._awarded[env_ids] = False


__all__ = ["FirstYawGoalBonus", "SignedYawPotentialDelta"]
