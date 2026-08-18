"""Shared actor and centralized critic online routing tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle
from holosoma.agents.mappo.runner import Plan5PolicyRunner
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


REPO_ROOT = Path(__file__).resolve().parents[5]
CHECKPOINT = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"


class FakeEnvironment:
    def __init__(self, observations):
        self.observations = observations
        self.received_actions = None

    def step(self, actor_state):
        self.received_actions = actor_state["actions"].clone()
        num_envs = self.received_actions.shape[0]
        return (
            self.observations,
            torch.zeros(num_envs),
            torch.zeros(num_envs, dtype=torch.long),
            {"source": "fake"},
        )


def _observations(num_envs: int = 3):
    return {
        "actor_obs": torch.zeros(num_envs, 2, 154),
        "teammate_obs": torch.zeros(num_envs, 2, 4),
        "critic_obs": torch.zeros(num_envs, 527),
    }


def test_shared_actor_and_team_critic_route_one_environment_step() -> None:
    models = initialize_plan5_model_bundle(
        CHECKPOINT,
        g1_29dof_wbt_w_object.algo.config,
        device="cpu",
    )
    runner = Plan5PolicyRunner(models)
    observations = _observations()
    env = FakeEnvironment(observations)

    transition = runner.step_environment(env, observations)

    assert transition.decision.actor_observations.shape == (3, 2, 158)
    assert transition.decision.critic_observations.shape == (3, 527)
    assert transition.decision.actions.shape == (3, 2, 29)
    assert transition.decision.values.shape == (3, 1)
    torch.testing.assert_close(env.received_actions, transition.decision.actions)
    assert transition.extras == {"source": "fake"}
    assert models.critic_obs_normalizer.count.item() == 3


def test_runner_shape_contract_fails_closed() -> None:
    models = initialize_plan5_model_bundle(
        CHECKPOINT,
        g1_29dof_wbt_w_object.algo.config,
        device="cpu",
    )
    runner = Plan5PolicyRunner(models)
    observations = _observations()
    observations["actor_obs"] = torch.zeros(3, 154)

    with pytest.raises(ValueError, match="actor_obs must have shape"):
        runner.decide(observations)
