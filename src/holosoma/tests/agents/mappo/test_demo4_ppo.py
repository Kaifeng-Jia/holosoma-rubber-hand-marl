"""CPU tests for Demo 4 cooperative team-reward MAPPO."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from holosoma.agents.mappo.demo4_checkpoint import (
    DEMO4_CHECKPOINT_INTERVAL,
    DEMO4_MAPPO_CHECKPOINT_VERSION,
    DEMO4_STATIC_RUNTIME_SHA256,
)
from holosoma.agents.mappo.demo4_ppo import Demo4PPO


class _FrozenIdentity(nn.Module):
    def forward(self, value: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        return value


class _CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.tensor(0.0))

    def forward(self, value: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        if update:
            self.count.add_(value.shape[0])
        return value


class _TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(164, 29, bias=False)
        self.log_std = nn.Parameter(torch.full((29,), -0.5))
        self.distribution = None

    def _mean(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(state["actor_obs"])

    def act_inference(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._mean(state)

    def act(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        mean = self._mean(state)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(-1)

    def reset(self, dones: torch.Tensor) -> None:
        del dones


class _TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(527, 1, bias=False)

    def evaluate(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(state["critic_obs"])

    def reset(self, dones: torch.Tensor) -> None:
        del dones


def _config(**overrides) -> SimpleNamespace:
    values = {
        "num_steps_per_env": 2,
        "actor_learning_rate": 1.0e-3,
        "critic_learning_rate": 1.0e-3,
        "min_actor_learning_rate": None,
        "max_actor_learning_rate": None,
        "min_critic_learning_rate": None,
        "max_critic_learning_rate": None,
        "gamma": 0.9,
        "lam": 0.8,
        "num_mini_batches": 1,
        "num_learning_epochs": 1,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "value_loss_coef": 1.0,
        "max_grad_norm": 1.0,
        "schedule": "fixed",
        "desired_kl": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _models() -> SimpleNamespace:
    actor = _TinyActor()
    critic = _TinyCritic()
    return SimpleNamespace(
        actor=actor,
        critic=critic,
        actor_obs_normalizer=_FrozenIdentity(),
        critic_obs_normalizer=_CountingIdentity(),
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1.0e-3),
        source_iteration=8050,
        source_sha256="demo4-test-source",
    )


def _observations(step: int = 0, num_envs: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(400 + step)
    return {
        "actor_obs": torch.randn(num_envs, 2, 154, generator=generator) * 0.1,
        "teammate_obs": torch.randn(num_envs, 2, 4, generator=generator) * 0.1,
        "table_obs": torch.randn(num_envs, 2, 6, generator=generator) * 0.1,
        "critic_obs": torch.randn(num_envs, 527, generator=generator) * 0.1,
    }


class _TeamEnvironment:
    def __init__(self, num_envs: int = 3) -> None:
        self.num_envs = num_envs
        self.step_index = 0
        self.action_shapes = []

    def step(self, state: dict[str, torch.Tensor]):
        actions = state["actions"]
        self.action_shapes.append(tuple(actions.shape))
        reward = 0.2 + actions.mean(dim=(1, 2))
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        self.step_index += 1
        return (
            _observations(self.step_index, self.num_envs),
            reward,
            done,
            {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool)},
        )


class _DiagnosticTeamEnvironment(_TeamEnvironment):
    def step(self, state: dict[str, torch.Tensor]):
        observations, reward, _, extras = super().step(state)
        if self.step_index == 1:
            done = torch.tensor([True, True, False])
            yaw = torch.tensor([1.5, -0.2, 0.4])
            termination_terms = {
                "yaw_goal_success": torch.tensor([True, False, False]),
                "clear_robot_fall": torch.tensor([False, True, False]),
                "table_physical_safety": torch.zeros(3, dtype=torch.bool),
                "reference_horizon": torch.zeros(3, dtype=torch.bool),
            }
        else:
            done = torch.tensor([False, True, True])
            yaw = torch.tensor([0.1, 0.3, 0.8])
            termination_terms = {
                "yaw_goal_success": torch.zeros(3, dtype=torch.bool),
                "clear_robot_fall": torch.zeros(3, dtype=torch.bool),
                "table_physical_safety": torch.tensor([False, True, False]),
                "reference_horizon": torch.tensor([False, False, True]),
            }
        extras["termination_terms"] = termination_terms
        extras["to_log"] = {"rotate/yaw_progress_rad": yaw}
        return observations, reward, done, extras


def test_demo4_collects_one_team_reward_and_gae_stream() -> None:
    torch.manual_seed(721)
    learner = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")
    env = _TeamEnvironment()

    learner.collect_rollout(env, _observations())

    assert env.action_shapes == [(3, 2, 29), (3, 2, 29)]
    assert learner.storage.agent("actor_obs").shape == (2, 3, 2, 164)
    assert learner.storage.agent("actions").shape == (2, 3, 2, 29)
    assert learner.storage.team("critic_obs").shape == (2, 3, 527)
    assert learner.storage.team("rewards").shape == (2, 3, 1)
    assert learner.storage.team("advantages").shape == (2, 3, 1)
    assert learner.models.critic_obs_normalizer.count.item() == 2 * 3


def test_demo4_collects_reset_safe_rollout_diagnostics_from_step_extras() -> None:
    torch.manual_seed(721)
    learner = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")

    learner.collect_rollout(_DiagnosticTeamEnvironment(), _observations())

    diagnostics = learner.last_rollout_diagnostics
    assert diagnostics["termination_counts"] == {
        "yaw_goal_success": 1,
        "clear_robot_fall": 1,
        "table_physical_safety": 1,
        "reference_horizon": 1,
    }
    assert diagnostics["completed_episodes"] == 4
    assert diagnostics["success_count"] == 1
    assert diagnostics["success_rate"] == pytest.approx(0.25)
    assert diagnostics["yaw_sample_count"] == 6
    assert diagnostics["yaw_sample_mean_rad"] == pytest.approx(2.9 / 6.0)
    assert diagnostics["yaw_sample_min_rad"] == pytest.approx(-0.2)
    assert diagnostics["yaw_sample_max_rad"] == pytest.approx(1.5)
    assert diagnostics["terminal_yaw_sample_count"] == 4
    assert diagnostics["terminal_yaw_mean_rad"] == pytest.approx(0.6)
    assert diagnostics["terminal_yaw_min_rad"] == pytest.approx(-0.2)
    assert diagnostics["terminal_yaw_max_rad"] == pytest.approx(1.5)


def test_demo4_rollout_diagnostics_tolerate_missing_optional_extras() -> None:
    learner = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")

    learner.collect_rollout(_TeamEnvironment(), _observations())

    assert learner.last_rollout_diagnostics == {
        "termination_counts": {
            "yaw_goal_success": 0,
            "clear_robot_fall": 0,
            "table_physical_safety": 0,
            "reference_horizon": 0,
        },
        "completed_episodes": 0,
        "success_count": 0,
        "success_rate": 0.0,
        "yaw_sample_count": 0,
        "yaw_sample_mean_rad": 0.0,
        "yaw_sample_min_rad": 0.0,
        "yaw_sample_max_rad": 0.0,
        "terminal_yaw_sample_count": 0,
        "terminal_yaw_mean_rad": 0.0,
        "terminal_yaw_min_rad": 0.0,
        "terminal_yaw_max_rad": 0.0,
    }


def test_demo4_update_and_checkpoint_are_cooperative_and_isolated() -> None:
    torch.manual_seed(17)
    models = _models()
    learner = Demo4PPO(models, _config(), num_envs=3, device="cpu")
    actor_before = models.actor.linear.weight.detach().clone()
    critic_before = models.critic.linear.weight.detach().clone()

    learner.collect_rollout(_TeamEnvironment(), _observations())
    metrics = learner.update()
    state = learner.training_state_dict(iteration=1000)

    assert not torch.equal(actor_before, models.actor.linear.weight)
    assert not torch.equal(critic_before, models.critic.linear.weight)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.__dict__.values())
    assert set(state).isdisjoint({"plan5_mappo", "demo3_mappo"})
    metadata = state["demo4_mappo"]
    assert metadata["version"] == DEMO4_MAPPO_CHECKPOINT_VERSION
    assert metadata["checkpoint_interval"] == DEMO4_CHECKPOINT_INTERVAL
    assert metadata["actor_obs_dim"] == 164
    assert metadata["critic_obs_dim"] == 527
    assert metadata["return_model"].startswith("one_team_reward")
    assert metadata["static_runtime_sha256"] == DEMO4_STATIC_RUNTIME_SHA256
    assert metadata["reference_frames"] == 316
    assert metadata["reference_fps"] == 50
    assert metadata["target_yaw_degrees"] == 90.0
    assert metadata["object_mass_kg"] == 20.0
    assert metadata["object_material_static_dynamic_restitution"] == [0.5, 0.5, 0.0]
    assert metadata["yaw_progress_reward_weight"] == 10.0
    assert metadata["success_bonus_weight"] == 5.0
    assert metadata["robot_asset"].endswith("rubberhand.urdf")
    assert metadata["rubber_hand_collision"] is True
    assert metadata["table_obs_contains_yaw_rate"] is False

    restored = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")
    assert restored.load_training_state_dict(state) == 1000
    for key, value in learner.models.actor.state_dict().items():
        torch.testing.assert_close(value, restored.models.actor.state_dict()[key])
    for key, value in learner.models.critic.state_dict().items():
        torch.testing.assert_close(value, restored.models.critic.state_dict()[key])

    contaminated = dict(state)
    contaminated["plan5_mappo"] = {}
    with pytest.raises(ValueError, match="cross-demo metadata"):
        restored.load_training_state_dict(contaminated)

    incompatible = copy.deepcopy(state)
    incompatible["demo4_mappo"]["static_runtime_sha256"] = "wrong-runtime"
    with pytest.raises(ValueError, match="checkpoint metadata mismatch"):
        restored.load_training_state_dict(incompatible)


def test_demo4_resume_keeps_checkpoint_learning_rates_without_override() -> None:
    learner = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")
    learner.actor_learning_rate = 2.5e-4
    learner.critic_learning_rate = 7.5e-4
    for group in learner.models.actor_optimizer.param_groups:
        group["lr"] = learner.actor_learning_rate
    for group in learner.models.critic_optimizer.param_groups:
        group["lr"] = learner.critic_learning_rate
    state = learner.training_state_dict(iteration=1200)

    restored = Demo4PPO(
        _models(),
        _config(actor_learning_rate=9.0e-3, critic_learning_rate=8.0e-3),
        num_envs=3,
        device="cpu",
    )
    assert restored.load_training_state_dict(state) == 1200
    assert restored.actor_learning_rate == pytest.approx(2.5e-4)
    assert restored.critic_learning_rate == pytest.approx(7.5e-4)
    assert all(
        group["lr"] == pytest.approx(2.5e-4)
        for group in restored.models.actor_optimizer.param_groups
    )
    assert all(
        group["lr"] == pytest.approx(7.5e-4)
        for group in restored.models.critic_optimizer.param_groups
    )


def test_demo4_resume_applies_explicit_learning_rate_overrides() -> None:
    source = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")
    state = source.training_state_dict(iteration=2400)
    restored = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")

    assert restored.load_training_state_dict(
        state,
        actor_learning_rate=1.25e-4,
        critic_learning_rate=3.5e-4,
    ) == 2400
    assert restored.actor_learning_rate == pytest.approx(1.25e-4)
    assert restored.critic_learning_rate == pytest.approx(3.5e-4)
    assert all(
        group["lr"] == pytest.approx(1.25e-4)
        for group in restored.models.actor_optimizer.param_groups
    )
    assert all(
        group["lr"] == pytest.approx(3.5e-4)
        for group in restored.models.critic_optimizer.param_groups
    )


@pytest.mark.parametrize("override", [0.0, -1.0, float("nan"), float("inf")])
def test_demo4_resume_rejects_invalid_learning_rate_override(override: float) -> None:
    source = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")
    state = source.training_state_dict(iteration=1)
    restored = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")

    with pytest.raises(ValueError, match="finite and positive"):
        restored.load_training_state_dict(
            state,
            actor_learning_rate=override,
        )


def test_demo4_rejects_per_agent_reward_shape() -> None:
    class _PerAgentRewardEnvironment(_TeamEnvironment):
        def step(self, state: dict[str, torch.Tensor]):
            observations, reward, done, extras = super().step(state)
            return observations, reward[:, None].expand(-1, 2), done, extras

    learner = Demo4PPO(_models(), _config(), num_envs=3, device="cpu")
    with pytest.raises(ValueError, match="Team scalar must have shape"):
        learner.collect_rollout(_PerAgentRewardEnvironment(), _observations())
