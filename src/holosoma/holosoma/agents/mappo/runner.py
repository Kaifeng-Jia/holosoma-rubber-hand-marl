"""Online shared-policy inference boundary for Plan 5 environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from holosoma.agents.mappo.batch_layout import HomogeneousAgentBatchLayout
from holosoma.agents.mappo.initialization import Plan5ModelBundle


@dataclass
class Plan5PolicyDecision:
    actions: torch.Tensor
    values: torch.Tensor
    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    normalized_actor_observations: torch.Tensor
    normalized_critic_observations: torch.Tensor
    action_log_probs: torch.Tensor | None = None
    action_means: torch.Tensor | None = None
    action_sigmas: torch.Tensor | None = None


@dataclass
class Plan5EnvironmentTransition:
    decision: Plan5PolicyDecision
    observations: dict[str, torch.Tensor]
    rewards: torch.Tensor
    dones: torch.Tensor
    extras: dict[str, Any]


class Plan5PolicyRunner:
    """Run one shared actor per agent and one centralized critic per team."""

    def __init__(
        self,
        models: Plan5ModelBundle,
        *,
        layout: HomogeneousAgentBatchLayout | None = None,
    ) -> None:
        self.models = models
        self.layout = layout or HomogeneousAgentBatchLayout()

    def _prepare_observations(
        self,
        observations: dict[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        actor_obs = observations["actor_obs"]
        teammate_obs = observations["teammate_obs"]
        critic_obs = observations["critic_obs"]
        num_envs = actor_obs.shape[0]

        expected_actor = (num_envs, self.layout.num_agents, 154)
        expected_teammate = (num_envs, self.layout.num_agents, 4)
        if actor_obs.shape != expected_actor:
            raise ValueError(f"actor_obs must have shape {expected_actor}, got {tuple(actor_obs.shape)}")
        if teammate_obs.shape != expected_teammate:
            raise ValueError(
                f"teammate_obs must have shape {expected_teammate}, got {tuple(teammate_obs.shape)}"
            )
        if critic_obs.shape != (num_envs, 527):
            raise ValueError(f"critic_obs must have shape ({num_envs}, 527), got {tuple(critic_obs.shape)}")

        combined_actor_obs = torch.cat((actor_obs, teammate_obs), dim=-1)
        flat_actor_obs = self.layout.flatten_actor_observations(combined_actor_obs)
        normalized_actor_obs = self.models.actor_obs_normalizer(flat_actor_obs)

        normalized_critic_obs = self.models.critic_obs_normalizer(
            critic_obs,
            update=update_critic_normalizer,
        )
        return (
            combined_actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            num_envs,
        )

    @torch.no_grad()
    def decide(
        self,
        observations: dict[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Plan5PolicyDecision:
        (
            combined_actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            num_envs,
        ) = self._prepare_observations(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        flat_actions = self.models.actor.act_inference({"actor_obs": normalized_actor_obs})
        actions = self.layout.unflatten_actions(flat_actions, num_envs=num_envs)
        values = self.models.critic.evaluate({"critic_obs": normalized_critic_obs})
        if values.shape != (num_envs, 1):
            raise ValueError(f"critic values must have shape ({num_envs}, 1), got {tuple(values.shape)}")

        return Plan5PolicyDecision(
            actions=actions,
            values=values,
            actor_observations=combined_actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor_obs.reshape(
                num_envs,
                self.layout.num_agents,
                self.layout.actor_obs_dim,
            ),
            normalized_critic_observations=normalized_critic_obs,
        )

    @torch.no_grad()
    def sample(
        self,
        observations: dict[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Plan5PolicyDecision:
        """Sample one stochastic action per agent for PPO data collection."""
        (
            combined_actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            num_envs,
        ) = self._prepare_observations(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        flat_actions = self.models.actor.act({"actor_obs": normalized_actor_obs})
        flat_log_probs = self.models.actor.get_actions_log_prob(flat_actions)
        flat_means = self.models.actor.action_mean
        flat_sigmas = self.models.actor.action_std
        values = self.models.critic.evaluate({"critic_obs": normalized_critic_obs})
        if values.shape != (num_envs, 1):
            raise ValueError(f"critic values must have shape ({num_envs}, 1), got {tuple(values.shape)}")

        return Plan5PolicyDecision(
            actions=self.layout.unflatten_actions(flat_actions, num_envs=num_envs),
            values=values,
            actor_observations=combined_actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor_obs.reshape(
                num_envs,
                self.layout.num_agents,
                self.layout.actor_obs_dim,
            ),
            normalized_critic_observations=normalized_critic_obs,
            action_log_probs=flat_log_probs.reshape(num_envs, self.layout.num_agents, 1),
            action_means=self.layout.unflatten_actions(flat_means, num_envs=num_envs),
            action_sigmas=self.layout.unflatten_actions(flat_sigmas, num_envs=num_envs),
        )

    @torch.no_grad()
    def step_environment(
        self,
        env: Any,
        observations: dict[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Plan5EnvironmentTransition:
        decision = self.decide(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        next_observations, rewards, dones, extras = env.step({"actions": decision.actions})
        return Plan5EnvironmentTransition(
            decision=decision,
            observations=next_observations,
            rewards=rewards,
            dones=dones,
            extras=extras,
        )
