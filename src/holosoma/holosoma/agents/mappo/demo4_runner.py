"""Shared-Actor and team-Critic routing for cooperative Demo 4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from holosoma.agents.mappo.demo4_initialization import (
    DEMO4_ACTION_DIM,
    DEMO4_ACTOR_OBS_DIM,
    DEMO4_CRITIC_OBS_DIM,
    DEMO4_NUM_AGENTS,
)
from holosoma.agents.mappo.initialization import Plan5ModelBundle


DEMO4_ACTOR_OBSERVATION_LAYOUT = (
    ("actor_obs", 154),
    ("teammate_obs", 4),
    ("table_obs", 6),
)


@dataclass
class Demo4PolicyDecision:
    """One synchronized team decision with one action row per robot."""

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
class Demo4EnvironmentTransition:
    decision: Demo4PolicyDecision
    observations: dict[str, torch.Tensor]
    rewards: torch.Tensor
    dones: torch.Tensor
    extras: dict[str, Any]


class Demo4PolicyRunner:
    """Apply one shared Actor independently and one Critic once per team.

    Actor input is folded from ``[E, 2, 164]`` to ``[E * 2, 164]``.  The
    centralized Critic deliberately keeps its physical-environment batch and
    receives exactly ``[E, 527]`` rather than one ego row per robot.
    """

    def __init__(
        self,
        models: Plan5ModelBundle,
        *,
        actor_observation_dims: Mapping[str, int] | None = None,
        critic_observation_key: str = "critic_obs",
    ) -> None:
        self.models = models
        self.actor_observation_dims = dict(
            DEMO4_ACTOR_OBSERVATION_LAYOUT
            if actor_observation_dims is None
            else actor_observation_dims
        )
        actual_layout = tuple(self.actor_observation_dims.items())
        if actual_layout != DEMO4_ACTOR_OBSERVATION_LAYOUT:
            raise ValueError(
                "Demo 4 Actor observation layout must be exactly "
                f"{DEMO4_ACTOR_OBSERVATION_LAYOUT}, got {actual_layout}"
            )
        if not critic_observation_key:
            raise ValueError("Demo 4 Critic observation key must be non-empty")
        self.actor_observation_keys = tuple(self.actor_observation_dims)
        self.critic_observation_key = critic_observation_key

    @staticmethod
    def _flatten_agents(values: torch.Tensor, feature_dim: int, name: str) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (DEMO4_NUM_AGENTS, feature_dim):
            raise ValueError(
                f"{name} must have shape [num_envs, {DEMO4_NUM_AGENTS}, {feature_dim}], "
                f"got {tuple(values.shape)}"
            )
        return values.reshape(values.shape[0] * DEMO4_NUM_AGENTS, feature_dim)

    @staticmethod
    def _unflatten_agents(
        values: torch.Tensor,
        *,
        num_envs: int,
        feature_dim: int,
        name: str,
    ) -> torch.Tensor:
        expected = (num_envs * DEMO4_NUM_AGENTS, feature_dim)
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(values.shape)}")
        return values.reshape(num_envs, DEMO4_NUM_AGENTS, feature_dim)

    def _prepare(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_critic_normalizer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        missing = [
            key
            for key in (*self.actor_observation_keys, self.critic_observation_key)
            if key not in observations
        ]
        if missing:
            raise ValueError(f"Missing required Demo 4 observation groups: {missing}")

        first = observations[self.actor_observation_keys[0]]
        if first.ndim < 1:
            raise ValueError("Demo 4 Actor observations require an environment axis")
        num_envs = first.shape[0]
        parts = []
        for key, feature_dim in self.actor_observation_dims.items():
            value = observations[key]
            expected = (num_envs, DEMO4_NUM_AGENTS, feature_dim)
            if value.shape != expected:
                raise ValueError(f"{key} must have shape {expected}, got {tuple(value.shape)}")
            parts.append(value)
        combined_actor_obs = torch.cat(parts, dim=-1)
        flat_actor_obs = self._flatten_agents(
            combined_actor_obs,
            DEMO4_ACTOR_OBS_DIM,
            "Demo 4 Actor observations",
        )
        normalized_actor_obs = self.models.actor_obs_normalizer(
            flat_actor_obs,
            update=False,
        )

        critic_obs = observations[self.critic_observation_key]
        expected_critic = (num_envs, DEMO4_CRITIC_OBS_DIM)
        if critic_obs.shape != expected_critic:
            raise ValueError(
                f"{self.critic_observation_key} must have shape {expected_critic}, "
                f"got {tuple(critic_obs.shape)}"
            )
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

    def _evaluate_team(self, normalized_critic_obs: torch.Tensor) -> torch.Tensor:
        values = self.models.critic.evaluate({"critic_obs": normalized_critic_obs})
        expected = (normalized_critic_obs.shape[0], 1)
        if values.shape != expected:
            raise ValueError(f"Demo 4 team values must have shape {expected}, got {values.shape}")
        return values

    @torch.no_grad()
    def decide(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Demo4PolicyDecision:
        (
            combined_actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            num_envs,
        ) = self._prepare(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        flat_actions = self.models.actor.act_inference({"actor_obs": normalized_actor_obs})
        actions = self._unflatten_agents(
            flat_actions,
            num_envs=num_envs,
            feature_dim=DEMO4_ACTION_DIM,
            name="Demo 4 Actor actions",
        )
        return Demo4PolicyDecision(
            actions=actions,
            values=self._evaluate_team(normalized_critic_obs),
            actor_observations=combined_actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor_obs.reshape(
                num_envs,
                DEMO4_NUM_AGENTS,
                DEMO4_ACTOR_OBS_DIM,
            ),
            normalized_critic_observations=normalized_critic_obs,
        )

    @torch.no_grad()
    def sample(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Demo4PolicyDecision:
        """Sample two independent action rows from the same Actor parameters."""

        (
            combined_actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            num_envs,
        ) = self._prepare(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        flat_actions = self.models.actor.act({"actor_obs": normalized_actor_obs})
        flat_log_probs = self.models.actor.get_actions_log_prob(flat_actions).unsqueeze(-1)
        return Demo4PolicyDecision(
            actions=self._unflatten_agents(
                flat_actions,
                num_envs=num_envs,
                feature_dim=DEMO4_ACTION_DIM,
                name="Demo 4 sampled actions",
            ),
            values=self._evaluate_team(normalized_critic_obs),
            actor_observations=combined_actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor_obs.reshape(
                num_envs,
                DEMO4_NUM_AGENTS,
                DEMO4_ACTOR_OBS_DIM,
            ),
            normalized_critic_observations=normalized_critic_obs,
            action_log_probs=self._unflatten_agents(
                flat_log_probs,
                num_envs=num_envs,
                feature_dim=1,
                name="Demo 4 action log probabilities",
            ),
            action_means=self._unflatten_agents(
                self.models.actor.action_mean,
                num_envs=num_envs,
                feature_dim=DEMO4_ACTION_DIM,
                name="Demo 4 action means",
            ),
            action_sigmas=self._unflatten_agents(
                self.models.actor.action_std,
                num_envs=num_envs,
                feature_dim=DEMO4_ACTION_DIM,
                name="Demo 4 action sigmas",
            ),
        )

    @torch.no_grad()
    def step_environment(
        self,
        env: Any,
        observations: dict[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Demo4EnvironmentTransition:
        decision = self.decide(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        next_observations, rewards, dones, extras = env.step(
            {"actions": decision.actions}
        )
        return Demo4EnvironmentTransition(
            decision=decision,
            observations=next_observations,
            rewards=rewards,
            dones=dones,
            extras=extras,
        )


__all__ = [
    "DEMO4_ACTOR_OBSERVATION_LAYOUT",
    "Demo4EnvironmentTransition",
    "Demo4PolicyDecision",
    "Demo4PolicyRunner",
]
