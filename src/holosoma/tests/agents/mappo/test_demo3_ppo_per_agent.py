"""CPU tests for Demo 3's isolated per-agent MAPPO update path."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from holosoma.agents.mappo.demo3_ppo import Demo3PPO


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
        self.reset_masks = []

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
        self.reset_masks.append(dones.clone())


class _TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(527, 1, bias=False)
        self.reset_masks = []
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[0, 0] = 1.0

    def evaluate(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.linear(state["critic_obs"])

    def reset(self, dones: torch.Tensor) -> None:
        self.reset_masks.append(dones.clone())


def _config(**overrides) -> SimpleNamespace:
    values = {
        "num_steps_per_env": 2,
        "actor_learning_rate": 1.0e-3,
        "critic_learning_rate": 1.0e-3,
        "min_actor_learning_rate": None,
        "max_actor_learning_rate": None,
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
        actor_obs_normalizer=_CountingIdentity(),
        critic_obs_normalizer=_CountingIdentity(),
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1.0e-3),
        source_sha256="demo3-test-source",
    )


def _observations(step: int) -> dict[str, torch.Tensor]:
    actor_obs = torch.zeros(3, 2, 154)
    critic_obs = torch.zeros(3, 2, 527)
    row_ids = 10.0 * step + torch.arange(6, dtype=torch.float).reshape(3, 2)
    actor_obs[..., 0] = row_ids
    critic_obs[..., 0] = row_ids
    return {
        "actor_obs": actor_obs,
        "teammate_obs": torch.ones(3, 2, 4),
        "table_obs": torch.ones(3, 2, 6),
        "critic_obs": critic_obs,
    }


class _FakeEnvironment:
    def __init__(self, *, timeout: bool = False, team_reward: bool = False) -> None:
        self.step_index = 0
        self.timeout = timeout
        self.team_reward = team_reward
        self.received_action_shapes = []

    def step(self, state: dict[str, torch.Tensor]):
        self.received_action_shapes.append(tuple(state["actions"].shape))
        rewards = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        ) + self.step_index
        if self.team_reward:
            rewards = rewards[:, 0]
        dones = torch.zeros(3, dtype=torch.bool)
        extras = {}
        if self.timeout and self.step_index == 0:
            dones = torch.tensor([False, True, True])
            extras["time_outs"] = torch.tensor([False, False, True])
            final_observations = _observations(50)
            final_observations["critic_obs"][2, 0, 0] = 7.0
            final_observations["critic_obs"][2, 1, 0] = 9.0
            extras["final_observations"] = final_observations
        next_observations = _observations(self.step_index + 1)
        self.step_index += 1
        return next_observations, rewards, dones, extras


def test_collect_rollout_keeps_per_agent_rewards_and_timeout_bootstrap() -> None:
    torch.manual_seed(7)
    models = _models()
    algorithm = Demo3PPO(models, _config(), num_envs=3, device="cpu")
    env = _FakeEnvironment(timeout=True)

    algorithm.collect_rollout(env, _observations(0))

    assert algorithm.storage.agent("values").shape == (2, 3, 2, 1)
    assert algorithm.storage.agent("rewards").shape == (2, 3, 2, 1)
    assert algorithm.storage.agent("returns").shape == (2, 3, 2, 1)
    assert algorithm.storage.agent("advantages").shape == (2, 3, 2, 1)
    assert algorithm.storage.shared("dones").shape == (2, 3, 1)
    assert env.received_action_shapes == [(3, 2, 29), (3, 2, 29)]
    torch.testing.assert_close(
        algorithm.storage.agent("rewards")[0, 2, :, 0],
        torch.tensor([3.0 + 0.9 * 7.0, 30.0 + 0.9 * 9.0]),
    )
    assert models.critic_obs_normalizer.count.item() == 2 * 3 * 2
    assert models.actor_obs_normalizer.count.item() == 0
    torch.testing.assert_close(
        models.actor.reset_masks[0],
        torch.tensor([False, False, True, True, True, True]),
    )


def test_gae_broadcasts_shared_done_without_crossing_agent_streams() -> None:
    algorithm = Demo3PPO(
        _models(),
        _config(gamma=0.5, lam=1.0),
        num_envs=3,
        device="cpu",
    )
    values = torch.zeros(2, 3, 2, 1)
    rewards = torch.tensor(
        [
            [[[1.0], [10.0]], [[2.0], [20.0]], [[3.0], [30.0]]],
            [[[4.0], [40.0]], [[5.0], [50.0]], [[6.0], [60.0]]],
        ]
    )
    dones = torch.tensor(
        [[[False], [True], [False]], [[False], [False], [False]]]
    )
    last_values = torch.tensor(
        [[[2.0], [4.0]], [[6.0], [8.0]], [[10.0], [12.0]]]
    )

    returns, advantages = algorithm.compute_returns_and_advantages(
        last_values=last_values,
        values=values,
        dones=dones,
        rewards=rewards,
    )

    expected = torch.tensor(
        [
            [[[3.5], [31.0]], [[2.0], [20.0]], [[8.5], [63.0]]],
            [[[5.0], [42.0]], [[8.0], [54.0]], [[11.0], [66.0]]],
        ]
    )
    torch.testing.assert_close(returns, expected)
    assert advantages.shape == (2, 3, 2, 1)
    assert not torch.equal(advantages[..., 0, :], advantages[..., 1, :])


def test_demo3_rejects_team_reward_and_zero_rollout_length() -> None:
    with pytest.raises(ValueError, match="num_steps_per_env must be positive"):
        Demo3PPO(_models(), _config(), num_envs=3, num_steps_per_env=0)

    algorithm = Demo3PPO(_models(), _config(), num_envs=3)
    with pytest.raises(ValueError, match="Demo 3 rewards must have shape"):
        algorithm.collect_rollout(_FakeEnvironment(team_reward=True), _observations(0))


def test_update_uses_agent_batches_and_checkpoint_is_demo3_only() -> None:
    torch.manual_seed(11)
    models = _models()
    algorithm = Demo3PPO(models, _config(), num_envs=3)
    algorithm.collect_rollout(_FakeEnvironment(), _observations(0))
    assert not torch.equal(
        algorithm.storage.agent("advantages")[:, :, 0],
        algorithm.storage.agent("advantages")[:, :, 1],
    )
    actor_before = models.actor.linear.weight.detach().clone()
    critic_before = models.critic.linear.weight.detach().clone()

    metrics = algorithm.update()

    assert torch.isfinite(torch.tensor(list(metrics.__dict__.values()))).all()
    assert not torch.equal(models.actor.linear.weight, actor_before)
    assert not torch.equal(models.critic.linear.weight, critic_before)

    state = algorithm.training_state_dict(iteration=17)
    assert "demo3_mappo" in state
    assert "plan5_mappo" not in state
    assert state["demo3_mappo"]["reward_layout"] == "per_agent"
    assert algorithm.load_training_state_dict(state) == 17
    state["demo3_mappo"] = {**state["demo3_mappo"], "reward_layout": "team"}
    with pytest.raises(ValueError, match="checkpoint metadata mismatch"):
        algorithm.load_training_state_dict(state)
