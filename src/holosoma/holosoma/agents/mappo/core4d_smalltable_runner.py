"""Shared-Actor and centralized-Critic routing for CORE4D small-table MAPPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from holosoma.agents.mappo.core4d_smalltable_initialization import (
    CORE4D_SMALLTABLE_ACTION_DIM,
    CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
    CORE4D_SMALLTABLE_CRITIC_OBS_DIM,
    CORE4D_SMALLTABLE_NUM_AGENTS,
)
from holosoma.agents.mappo.initialization import Plan5ModelBundle


CORE4D_SMALLTABLE_ACTOR_GROUPS = (
    ("actor_obs", 154),
    ("teammate_obs", 4),
    ("table_obs", 6),
)


@dataclass
class Core4DSmallTablePolicyDecision:
    actions: torch.Tensor
    values: torch.Tensor
    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    normalized_actor_observations: torch.Tensor
    normalized_critic_observations: torch.Tensor
    action_log_probs: torch.Tensor | None = None
    action_means: torch.Tensor | None = None
    action_sigmas: torch.Tensor | None = None


class Core4DSmallTablePolicyRunner:
    """Call one 164-D Actor once per robot and one 527-D Critic per team."""

    def __init__(self, models: Plan5ModelBundle) -> None:
        self.models = models

    @staticmethod
    def _unflatten(values: torch.Tensor, num_envs: int, width: int) -> torch.Tensor:
        expected = (num_envs * CORE4D_SMALLTABLE_NUM_AGENTS, width)
        if values.shape != expected:
            raise ValueError(f"Expected flattened shape {expected}, got {tuple(values.shape)}")
        return values.reshape(num_envs, CORE4D_SMALLTABLE_NUM_AGENTS, width)

    def _prepare(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_actor_normalizer: bool,
        update_critic_normalizer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        first = observations.get("actor_obs")
        if not isinstance(first, torch.Tensor) or first.ndim != 3:
            raise ValueError("actor_obs must have shape [num_envs, 2, 154]")
        num_envs = first.shape[0]
        parts = []
        for name, width in CORE4D_SMALLTABLE_ACTOR_GROUPS:
            value = observations.get(name)
            expected = (num_envs, CORE4D_SMALLTABLE_NUM_AGENTS, width)
            if not isinstance(value, torch.Tensor) or value.shape != expected:
                shape = getattr(value, "shape", None)
                raise ValueError(f"{name} must have shape {expected}, got {shape}")
            parts.append(value)
        actor_observations = torch.cat(parts, dim=-1)
        flat_actor = actor_observations.reshape(
            num_envs * CORE4D_SMALLTABLE_NUM_AGENTS,
            CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
        )
        normalized_actor = self.models.actor_obs_normalizer(
            flat_actor,
            update=update_actor_normalizer,
        )

        critic_observations = observations.get("critic_obs")
        expected_critic = (num_envs, CORE4D_SMALLTABLE_CRITIC_OBS_DIM)
        if (
            not isinstance(critic_observations, torch.Tensor)
            or critic_observations.shape != expected_critic
        ):
            shape = getattr(critic_observations, "shape", None)
            raise ValueError(f"critic_obs must have shape {expected_critic}, got {shape}")
        normalized_critic = self.models.critic_obs_normalizer(
            critic_observations,
            update=update_critic_normalizer,
        )
        return (
            actor_observations,
            normalized_actor,
            critic_observations,
            normalized_critic,
            num_envs,
        )

    def _team_values(self, normalized_critic: torch.Tensor) -> torch.Tensor:
        values = self.models.critic.evaluate({"critic_obs": normalized_critic})
        expected = (normalized_critic.shape[0], 1)
        if values.shape != expected:
            raise ValueError(f"Team Critic must return {expected}, got {tuple(values.shape)}")
        return values

    @torch.no_grad()
    def decide(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        evaluate_critic: bool = True,
    ) -> Core4DSmallTablePolicyDecision:
        prepared = self._prepare(
            observations,
            update_actor_normalizer=False,
            update_critic_normalizer=False,
        )
        actor_obs, normalized_actor, critic_obs, normalized_critic, num_envs = prepared
        flat_actions = self.models.actor.act_inference({"actor_obs": normalized_actor})
        values = (
            self._team_values(normalized_critic)
            if evaluate_critic
            else torch.zeros(num_envs, 1, device=flat_actions.device)
        )
        return Core4DSmallTablePolicyDecision(
            actions=self._unflatten(flat_actions, num_envs, CORE4D_SMALLTABLE_ACTION_DIM),
            values=values,
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor.reshape(
                num_envs,
                CORE4D_SMALLTABLE_NUM_AGENTS,
                CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
            ),
            normalized_critic_observations=normalized_critic,
        )

    @torch.no_grad()
    def sample(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Core4DSmallTablePolicyDecision:
        prepared = self._prepare(
            observations,
            update_actor_normalizer=True,
            update_critic_normalizer=update_critic_normalizer,
        )
        actor_obs, normalized_actor, critic_obs, normalized_critic, num_envs = prepared
        flat_actions = self.models.actor.act({"actor_obs": normalized_actor})
        flat_log_probs = self.models.actor.get_actions_log_prob(flat_actions).unsqueeze(-1)
        return Core4DSmallTablePolicyDecision(
            actions=self._unflatten(flat_actions, num_envs, CORE4D_SMALLTABLE_ACTION_DIM),
            values=self._team_values(normalized_critic),
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor.reshape(
                num_envs,
                CORE4D_SMALLTABLE_NUM_AGENTS,
                CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
            ),
            normalized_critic_observations=normalized_critic,
            action_log_probs=self._unflatten(flat_log_probs, num_envs, 1),
            action_means=self._unflatten(
                self.models.actor.action_mean,
                num_envs,
                CORE4D_SMALLTABLE_ACTION_DIM,
            ),
            action_sigmas=self._unflatten(
                self.models.actor.action_std,
                num_envs,
                CORE4D_SMALLTABLE_ACTION_DIM,
            ),
        )

    @torch.no_grad()
    def step_environment(
        self,
        env: Any,
        observations: dict[str, torch.Tensor],
    ) -> tuple[Core4DSmallTablePolicyDecision, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        decision = self.decide(observations)
        next_observations, rewards, dones, extras = env.step(
            {"actions": decision.actions}
        )
        return decision, next_observations, rewards, dones, extras


__all__ = [
    "CORE4D_SMALLTABLE_ACTOR_GROUPS",
    "Core4DSmallTablePolicyDecision",
    "Core4DSmallTablePolicyRunner",
]
