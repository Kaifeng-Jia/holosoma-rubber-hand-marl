from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle
from holosoma.agents.mappo.ppo import Plan5PPO
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


REPO_ROOT = Path(__file__).resolve().parents[5]
CHECKPOINT = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"


def _observations(num_envs: int, step: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(900 + step)
    return {
        "actor_obs": torch.randn(num_envs, 2, 154, generator=generator) * 0.1,
        "teammate_obs": torch.empty(num_envs, 2, 4).uniform_(
            -0.5,
            0.5,
            generator=generator,
        ),
        "critic_obs": torch.randn(num_envs, 527, generator=generator) * 0.1,
    }


class FakeTeamEnvironment:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.step_count = 0

    def step(self, actor_state):
        actions = actor_state["actions"]
        assert actions.shape == (self.num_envs, 2, 29)
        rewards = torch.tensor([0.4, -0.2])[: self.num_envs]
        rewards = rewards + 0.01 * actions.mean(dim=(1, 2))
        self.step_count += 1
        return (
            _observations(self.num_envs, self.step_count),
            rewards,
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool)},
        )


def _learner(*, num_envs: int = 2, num_steps: int = 2) -> Plan5PPO:
    config = replace(
        g1_29dof_wbt_w_object.algo.config,
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    models = initialize_plan5_model_bundle(CHECKPOINT, config, device="cpu")
    return Plan5PPO(
        models,
        config,
        num_envs=num_envs,
        num_steps_per_env=num_steps,
        device="cpu",
    )


def test_team_gae_stops_at_joint_done() -> None:
    learner = _learner(num_envs=1, num_steps=2)
    values = torch.zeros(2, 1, 1)
    rewards = torch.ones(2, 1, 1)
    dones = torch.tensor([[[True]], [[False]]])

    returns, advantages = learner.compute_team_returns_and_advantages(
        last_values=torch.tensor([[2.0]]),
        values=values,
        rewards=rewards,
        dones=dones,
    )

    torch.testing.assert_close(returns[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(returns[1], torch.tensor([[1.0 + learner.config.gamma * 2.0]]))
    assert torch.isfinite(advantages).all()


def test_single_mappo_update_changes_actor_and_critic_without_unfreezing_actor_normalizer() -> None:
    torch.manual_seed(721)
    learner = _learner()
    environment = FakeTeamEnvironment(num_envs=2)
    actor_before = {key: value.clone() for key, value in learner.models.actor.state_dict().items()}
    critic_before = {key: value.clone() for key, value in learner.models.critic.state_dict().items()}
    normalizer_before = {
        key: value.clone() for key, value in learner.models.actor_obs_normalizer.state_dict().items()
    }

    learner.collect_rollout(environment, _observations(2))
    assert not learner.storage.team("timeouts").any()
    metrics = learner.update()

    assert any(
        not torch.equal(value, learner.models.actor.state_dict()[key])
        for key, value in actor_before.items()
    )
    assert any(
        not torch.equal(value, learner.models.critic.state_dict()[key])
        for key, value in critic_before.items()
    )
    for key, value in normalizer_before.items():
        torch.testing.assert_close(value, learner.models.actor_obs_normalizer.state_dict()[key])
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.__dict__.values())


def test_mappo_checkpoint_round_trip_and_metadata_fail_closed() -> None:
    learner = _learner()
    learner.collect_rollout(FakeTeamEnvironment(2), _observations(2))
    learner.update()
    state = learner.training_state_dict(iteration=1)

    restored = _learner()
    assert restored.load_training_state_dict(state) == 1
    for key, value in learner.models.actor.state_dict().items():
        torch.testing.assert_close(value, restored.models.actor.state_dict()[key])
    for key, value in learner.models.critic.state_dict().items():
        torch.testing.assert_close(value, restored.models.critic.state_dict()[key])

    bad_state = dict(state)
    bad_state["plan5_mappo"] = {**state["plan5_mappo"], "num_agents": 3}
    with pytest.raises(ValueError, match="metadata mismatch"):
        restored.load_training_state_dict(bad_state)


def test_critic_only_update_preserves_actor_and_updates_critic() -> None:
    torch.manual_seed(721)
    learner = _learner()
    actor_before = {key: value.clone() for key, value in learner.models.actor.state_dict().items()}
    critic_before = {key: value.clone() for key, value in learner.models.critic.state_dict().items()}
    actor_lr_before = learner.models.actor_optimizer.param_groups[0]["lr"]
    critic_lr_before = learner.models.critic_optimizer.param_groups[0]["lr"]

    learner.collect_rollout(FakeTeamEnvironment(2), _observations(2))
    metrics = learner.update(update_actor=False)

    for key, value in actor_before.items():
        torch.testing.assert_close(value, learner.models.actor.state_dict()[key], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(value, learner.models.critic.state_dict()[key])
        for key, value in critic_before.items()
    )
    assert metrics.surrogate_loss == 0.0
    assert metrics.entropy == 0.0
    assert metrics.kl == 0.0
    assert metrics.actor_grad_norm == 0.0
    assert learner.models.actor_optimizer.param_groups[0]["lr"] == actor_lr_before
    assert learner.models.critic_optimizer.param_groups[0]["lr"] == critic_lr_before


@pytest.mark.parametrize(
    ("kl", "actor_lr_factor"),
    [(0.03, 1.0 / 1.5), (0.001, 1.5)],
)
def test_adaptive_policy_kl_changes_only_actor_learning_rate(
    kl: float,
    actor_lr_factor: float,
) -> None:
    learner = _learner()
    actor_lr_before = learner.actor_learning_rate
    critic_lr_before = learner.critic_learning_rate

    learner._update_actor_learning_rate(torch.tensor(kl))

    assert learner.actor_learning_rate == pytest.approx(actor_lr_before * actor_lr_factor)
    assert learner.models.actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        actor_lr_before * actor_lr_factor
    )
    assert learner.critic_learning_rate == critic_lr_before
    assert learner.models.critic_optimizer.param_groups[0]["lr"] == critic_lr_before


def test_fixed_schedule_keeps_actor_and_critic_learning_rates_during_update() -> None:
    learner = _learner()
    learner.config = replace(learner.config, schedule="fixed")
    actor_lr_before = learner.actor_learning_rate
    critic_lr_before = learner.critic_learning_rate

    learner.collect_rollout(FakeTeamEnvironment(2), _observations(2))
    learner.update(teammate_input_only=True)

    assert learner.actor_learning_rate == actor_lr_before
    assert learner.critic_learning_rate == critic_lr_before
    assert learner.models.actor_optimizer.param_groups[0]["lr"] == actor_lr_before
    assert learner.models.critic_optimizer.param_groups[0]["lr"] == critic_lr_before


def test_teammate_input_only_update_changes_only_four_new_input_columns() -> None:
    torch.manual_seed(721)
    learner = _learner()
    actor_before = {key: value.clone() for key, value in learner.models.actor.state_dict().items()}
    critic_before = {key: value.clone() for key, value in learner.models.critic.state_dict().items()}

    learner.collect_rollout(FakeTeamEnvironment(2), _observations(2))
    learner.update(teammate_input_only=True)

    first_weight = "actor_module.module.0.weight"
    actor_after = learner.models.actor.state_dict()
    torch.testing.assert_close(
        actor_after[first_weight][:, :154],
        actor_before[first_weight][:, :154],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        actor_after[first_weight][:, 154:158],
        actor_before[first_weight][:, 154:158],
    )
    for key, value in actor_before.items():
        if key != first_weight:
            torch.testing.assert_close(actor_after[key], value, rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(value, learner.models.critic.state_dict()[key])
        for key, value in critic_before.items()
    )

    zero_teammate = _observations(2)
    zero_teammate["teammate_obs"].zero_()
    flat_observations = torch.cat(
        (zero_teammate["actor_obs"], zero_teammate["teammate_obs"]),
        dim=-1,
    ).reshape(-1, 158)
    flat_normalized = learner.models.actor_obs_normalizer(flat_observations, update=False)
    with torch.no_grad():
        learner.models.actor.act({"actor_obs": flat_normalized})
        action_after = learner.models.actor.action_mean.clone()
        learner.models.actor.load_state_dict(actor_before, strict=True)
        learner.models.actor.act({"actor_obs": flat_normalized})
        action_before = learner.models.actor.action_mean.clone()
    torch.testing.assert_close(action_after, action_before, rtol=0.0, atol=0.0)


def test_teammate_input_only_requires_actor_update() -> None:
    learner = _learner()
    with pytest.raises(ValueError, match="requires update_actor=True"):
        learner.update(update_actor=False, teammate_input_only=True)
