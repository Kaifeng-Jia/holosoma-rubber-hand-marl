"""Demo 3 shared-policy inference with one ego-first value per agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from holosoma.agents.mappo.demo3_initialization import (
    DEMO3_ACTION_DIM,
    DEMO3_ACTOR_OBS_DIM,
    DEMO3_CRITIC_OBS_DIM,
)
from holosoma.agents.mappo.initialization import Plan5ModelBundle


DEMO3_NUM_AGENTS = 2
DEMO3_ACTOR_OBSERVATION_LAYOUT = (
    ("actor_obs", 154),
    ("teammate_obs", 4),
    ("table_obs", 6),
)


@dataclass
class Demo3PolicyDecision:
    """One synchronized decision for both competitive agents."""

    actions: torch.Tensor
    values: torch.Tensor
    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    normalized_actor_observations: torch.Tensor
    normalized_critic_observations: torch.Tensor
    action_log_probs: torch.Tensor | None = None
    action_means: torch.Tensor | None = None
    action_sigmas: torch.Tensor | None = None


class Demo3PolicyRunner:
    """Apply one Actor and one ego-conditioned Critic to both agents.

    Both networks are parameter shared.  Agent identity is represented only by
    the observation row: actor observations are local, while each centralized
    critic row is ordered as ``[ego, opponent, shared state]`` by the environment.
    The explicit agent axis is folded into the batch axis for each network call
    and restored immediately afterwards.
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
            DEMO3_ACTOR_OBSERVATION_LAYOUT
            if actor_observation_dims is None
            else actor_observation_dims
        )
        actual_actor_layout = tuple(self.actor_observation_dims.items())
        if actual_actor_layout != DEMO3_ACTOR_OBSERVATION_LAYOUT:
            raise ValueError(
                "Demo 3 Actor observation layout must be exactly "
                f"{DEMO3_ACTOR_OBSERVATION_LAYOUT}, got {actual_actor_layout}"
            )
        if not critic_observation_key:
            raise ValueError("Demo 3 critic observation key must be non-empty")
        self.actor_obs_keys = tuple(self.actor_observation_dims)
        self.critic_observation_key = critic_observation_key

    @staticmethod
    def _flatten_agents(values: torch.Tensor, feature_dim: int, name: str) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (DEMO3_NUM_AGENTS, feature_dim):
            raise ValueError(
                f"{name} must have shape [num_envs, {DEMO3_NUM_AGENTS}, {feature_dim}], "
                f"got {tuple(values.shape)}"
            )
        return values.reshape(values.shape[0] * DEMO3_NUM_AGENTS, feature_dim)

    @staticmethod
    def _unflatten_agents(
        values: torch.Tensor,
        *,
        num_envs: int,
        feature_dim: int,
        name: str,
    ) -> torch.Tensor:
        expected = (num_envs * DEMO3_NUM_AGENTS, feature_dim)
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(values.shape)}")
        return values.reshape(num_envs, DEMO3_NUM_AGENTS, feature_dim)

    def _prepare_actor_observations(
        self,
        observations: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        missing = [key for key in self.actor_obs_keys if key not in observations]
        if missing:
            raise ValueError(f"Missing required Demo 3 Actor observation groups: {missing}")

        first = observations[self.actor_obs_keys[0]]
        if first.ndim < 1:
            raise ValueError("Demo 3 Actor observation must have an environment axis")
        num_envs = first.shape[0]
        parts = []
        for key, dimension in self.actor_observation_dims.items():
            value = observations[key]
            expected = (num_envs, DEMO3_NUM_AGENTS, dimension)
            if value.shape != expected:
                raise ValueError(f"{key} must have shape {expected}, got {tuple(value.shape)}")
            parts.append(value)

        combined = torch.cat(parts, dim=-1)
        flat = self._flatten_agents(combined, DEMO3_ACTOR_OBS_DIM, "Demo 3 Actor observations")
        normalized = self.models.actor_obs_normalizer(flat, update=False)
        if normalized.shape != flat.shape:
            raise ValueError(
                "Demo 3 Actor normalizer changed shape: "
                f"expected {tuple(flat.shape)}, got {tuple(normalized.shape)}"
            )
        return combined, normalized, num_envs

    def evaluate_critic_observations(
        self,
        critic_observations: torch.Tensor,
        *,
        update_normalizer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-agent values and normalized ego-first critic rows."""
        flat = self._flatten_agents(
            critic_observations,
            DEMO3_CRITIC_OBS_DIM,
            "Demo 3 ego-first critic observations",
        )
        normalized_flat = self.models.critic_obs_normalizer(
            flat,
            update=update_normalizer,
        )
        if normalized_flat.shape != flat.shape:
            raise ValueError(
                "Demo 3 critic normalizer changed shape: "
                f"expected {tuple(flat.shape)}, got {tuple(normalized_flat.shape)}"
            )
        flat_values = self.models.critic.evaluate({"critic_obs": normalized_flat})
        num_envs = critic_observations.shape[0]
        values = self._unflatten_agents(
            flat_values,
            num_envs=num_envs,
            feature_dim=1,
            name="Demo 3 critic values",
        )
        normalized = normalized_flat.reshape(
            num_envs,
            DEMO3_NUM_AGENTS,
            DEMO3_CRITIC_OBS_DIM,
        )
        return values, normalized

    def _prepare(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_critic_normalizer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.critic_observation_key not in observations:
            raise ValueError(
                f"Missing required Demo 3 critic observation group: {self.critic_observation_key!r}"
            )
        combined_actor_obs, normalized_actor_obs, num_envs = self._prepare_actor_observations(
            observations
        )
        critic_obs = observations[self.critic_observation_key]
        expected_critic = (num_envs, DEMO3_NUM_AGENTS, DEMO3_CRITIC_OBS_DIM)
        if critic_obs.shape != expected_critic:
            raise ValueError(
                f"{self.critic_observation_key} must have shape {expected_critic}, "
                f"got {tuple(critic_obs.shape)}"
            )
        values, normalized_critic_obs = self.evaluate_critic_observations(
            critic_obs,
            update_normalizer=update_critic_normalizer,
        )
        return (
            combined_actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            values,
            num_envs,
        )

    @torch.no_grad()
    def decide(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Demo3PolicyDecision:
        (
            actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            values,
            num_envs,
        ) = self._prepare(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        flat_actions = self.models.actor.act_inference({"actor_obs": normalized_actor_obs})
        actions = self._unflatten_agents(
            flat_actions,
            num_envs=num_envs,
            feature_dim=DEMO3_ACTION_DIM,
            name="Demo 3 Actor actions",
        )
        return Demo3PolicyDecision(
            actions=actions,
            values=values,
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor_obs.reshape(
                num_envs,
                DEMO3_NUM_AGENTS,
                DEMO3_ACTOR_OBS_DIM,
            ),
            normalized_critic_observations=normalized_critic_obs,
        )

    @torch.no_grad()
    def sample(
        self,
        observations: Mapping[str, torch.Tensor],
        *,
        update_critic_normalizer: bool = True,
    ) -> Demo3PolicyDecision:
        """Sample one independent action row per agent from the shared Actor."""
        (
            actor_obs,
            normalized_actor_obs,
            critic_obs,
            normalized_critic_obs,
            values,
            num_envs,
        ) = self._prepare(
            observations,
            update_critic_normalizer=update_critic_normalizer,
        )
        flat_actions = self.models.actor.act({"actor_obs": normalized_actor_obs})
        flat_log_probs = self.models.actor.get_actions_log_prob(flat_actions).unsqueeze(-1)
        actions = self._unflatten_agents(
            flat_actions,
            num_envs=num_envs,
            feature_dim=DEMO3_ACTION_DIM,
            name="Demo 3 sampled actions",
        )
        action_log_probs = self._unflatten_agents(
            flat_log_probs,
            num_envs=num_envs,
            feature_dim=1,
            name="Demo 3 action log probabilities",
        )
        action_means = self._unflatten_agents(
            self.models.actor.action_mean,
            num_envs=num_envs,
            feature_dim=DEMO3_ACTION_DIM,
            name="Demo 3 action means",
        )
        action_sigmas = self._unflatten_agents(
            self.models.actor.action_std,
            num_envs=num_envs,
            feature_dim=DEMO3_ACTION_DIM,
            name="Demo 3 action sigmas",
        )
        return Demo3PolicyDecision(
            actions=actions,
            values=values,
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            normalized_actor_observations=normalized_actor_obs.reshape(
                num_envs,
                DEMO3_NUM_AGENTS,
                DEMO3_ACTOR_OBS_DIM,
            ),
            normalized_critic_observations=normalized_critic_obs,
            action_log_probs=action_log_probs,
            action_means=action_means,
            action_sigmas=action_sigmas,
        )


__all__ = [
    "DEMO3_NUM_AGENTS",
    "Demo3PolicyDecision",
    "Demo3PolicyRunner",
]
