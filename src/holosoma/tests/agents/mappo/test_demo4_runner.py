"""CPU tensor-routing tests for the cooperative Demo 4 policy boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from holosoma.agents.mappo.demo4_runner import Demo4PolicyRunner


class _CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.tensor(0.0))

    def forward(self, value: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        if update:
            self.count.add_(value.shape[0])
        return value


class _RecordingActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input = None
        self.distribution = None

    def _mean(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_input = state["actor_obs"].clone()
        return state["actor_obs"][:, :29]

    def act_inference(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._mean(state)

    def act(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        mean = self._mean(state)
        self.distribution = Normal(mean, torch.ones_like(mean))
        return mean

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev


class _RecordingCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input = None

    def evaluate(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_input = state["critic_obs"].clone()
        return state["critic_obs"][:, :1]


def _runner() -> tuple[Demo4PolicyRunner, SimpleNamespace]:
    models = SimpleNamespace(
        actor=_RecordingActor(),
        critic=_RecordingCritic(),
        actor_obs_normalizer=_CountingIdentity(),
        critic_obs_normalizer=_CountingIdentity(),
    )
    return Demo4PolicyRunner(models), models


def _observations(num_envs: int = 3) -> dict[str, torch.Tensor]:
    actor_obs = torch.zeros(num_envs, 2, 154)
    critic_obs = torch.zeros(num_envs, 527)
    row_ids = torch.arange(num_envs * 2, dtype=torch.float).reshape(num_envs, 2)
    actor_obs[..., 0] = row_ids
    critic_obs[..., 0] = 100.0 + torch.arange(num_envs, dtype=torch.float)
    return {
        "actor_obs": actor_obs,
        "teammate_obs": torch.ones(num_envs, 2, 4),
        "table_obs": torch.full((num_envs, 2, 6), 2.0),
        "critic_obs": critic_obs,
    }


def test_demo4_routes_two_actor_rows_and_one_team_critic_row_per_environment() -> None:
    runner, models = _runner()

    decision = runner.sample(_observations())

    assert models.actor.last_input.shape == (6, 164)
    assert models.critic.last_input.shape == (3, 527)
    assert decision.actor_observations.shape == (3, 2, 164)
    assert decision.critic_observations.shape == (3, 527)
    assert decision.actions.shape == (3, 2, 29)
    assert decision.values.shape == (3, 1)
    assert decision.action_log_probs.shape == (3, 2, 1)
    assert decision.action_means.shape == (3, 2, 29)
    assert decision.action_sigmas.shape == (3, 2, 29)
    torch.testing.assert_close(
        decision.actions[..., 0],
        torch.arange(6, dtype=torch.float).reshape(3, 2),
    )
    torch.testing.assert_close(
        decision.values[..., 0],
        100.0 + torch.arange(3, dtype=torch.float),
    )
    assert models.actor_obs_normalizer.count.item() == 0
    assert models.critic_obs_normalizer.count.item() == 3


def test_demo4_actor_group_order_and_team_critic_shape_fail_closed() -> None:
    _, models = _runner()
    with pytest.raises(ValueError, match="layout must be exactly"):
        Demo4PolicyRunner(
            models,
            actor_observation_dims={
                "actor_obs": 154,
                "table_obs": 6,
                "teammate_obs": 4,
            },
        )

    runner = Demo4PolicyRunner(models)
    observations = _observations()
    observations["critic_obs"] = torch.zeros(3, 2, 527)
    with pytest.raises(ValueError, match="critic_obs must have shape"):
        runner.decide(observations)
