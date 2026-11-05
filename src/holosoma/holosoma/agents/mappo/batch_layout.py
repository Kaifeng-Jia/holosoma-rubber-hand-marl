"""Tensor-shape contract for a homogeneous shared-actor team."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HomogeneousAgentBatchLayout:
    """Validate and reshape per-environment, per-agent tensors.

    Physical environments keep an explicit agent axis. The shared actor sees
    that axis folded into the batch axis, so every row remains one 29-DoF G1
    action rather than a role-bound 58-dimensional joint action.
    """

    num_agents: int = 2
    actor_obs_dim: int = 158
    action_dim: int = 29

    def __post_init__(self) -> None:
        for field_name in ("num_agents", "actor_obs_dim", "action_dim"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

    def _flatten(self, values: torch.Tensor, feature_dim: int, name: str) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (self.num_agents, feature_dim):
            raise ValueError(
                f"{name} must have shape [num_envs, {self.num_agents}, {feature_dim}], "
                f"got {tuple(values.shape)}"
            )
        return values.reshape(values.shape[0] * self.num_agents, feature_dim)

    def _unflatten(
        self,
        values: torch.Tensor,
        *,
        num_envs: int,
        feature_dim: int,
        name: str,
    ) -> torch.Tensor:
        expected = (num_envs * self.num_agents, feature_dim)
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(values.shape)}")
        return values.reshape(num_envs, self.num_agents, feature_dim)

    def flatten_actor_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Return shared-actor input shaped ``[num_envs * num_agents, 158]``."""
        return self._flatten(observations, self.actor_obs_dim, "actor observations")

    def flatten_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Return per-agent actions shaped ``[num_envs * num_agents, 29]``."""
        return self._flatten(actions, self.action_dim, "actions")

    def unflatten_actions(self, actions: torch.Tensor, *, num_envs: int) -> torch.Tensor:
        """Restore shared-actor output to ``[num_envs, num_agents, 29]``."""
        return self._unflatten(
            actions,
            num_envs=num_envs,
            feature_dim=self.action_dim,
            name="shared-actor actions",
        )

    def expand_team_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Repeat one team value/state per environment across the agent axis."""
        if values.ndim < 1:
            raise ValueError("team tensor must have a leading environment dimension")
        return values.unsqueeze(1).expand(-1, self.num_agents, *values.shape[1:])
