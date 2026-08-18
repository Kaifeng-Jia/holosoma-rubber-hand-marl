from __future__ import annotations

import pytest
import torch

from holosoma.agents.mappo import MultiAgentRolloutStorage


def _storage() -> MultiAgentRolloutStorage:
    storage = MultiAgentRolloutStorage(
        num_envs=2,
        num_agents=2,
        num_transitions_per_env=2,
    )
    storage.register_agent("actor_obs", (3,))
    storage.register_agent("actions", (2,))
    storage.register_team("critic_obs", (4,))
    storage.register_team("rewards", (1,))
    return storage


def _add_step(storage: MultiAgentRolloutStorage, step: int) -> None:
    actor_obs = torch.zeros((2, 2, 3))
    actions = torch.zeros((2, 2, 2))
    critic_obs = torch.zeros((2, 4))
    rewards = torch.zeros((2, 1))
    for env in range(2):
        sample_id = float(step * 10 + env)
        critic_obs[env] = sample_id
        rewards[env] = sample_id
        for agent in range(2):
            actor_obs[env, agent] = torch.tensor([sample_id, float(agent), 1.0])
            actions[env, agent] = torch.tensor([sample_id, float(agent)])
    storage.add(
        agent={"actor_obs": actor_obs, "actions": actions},
        team={"critic_obs": critic_obs, "rewards": rewards},
    )


def test_storage_separates_agent_and_team_axes() -> None:
    storage = _storage()
    _add_step(storage, 0)

    assert storage.agent("actor_obs").shape == (2, 2, 2, 3)
    assert storage.team("critic_obs").shape == (2, 2, 4)
    torch.testing.assert_close(storage.agent("actor_obs")[0, 1, 0], torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(storage.agent("actor_obs")[0, 1, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_minibatch_preserves_team_to_agent_pairing() -> None:
    storage = _storage()
    _add_step(storage, 0)
    _add_step(storage, 1)

    minibatch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    actor_ids = minibatch["agent"]["actor_obs"][:, 0].reshape(-1, 2)
    actor_roles = minibatch["agent"]["actor_obs"][:, 1].reshape(-1, 2)
    team_ids = minibatch["team"]["rewards"]

    torch.testing.assert_close(actor_ids[:, 0:1], team_ids)
    torch.testing.assert_close(actor_ids[:, 1:2], team_ids)
    torch.testing.assert_close(actor_roles, torch.tensor([[0.0, 1.0]]).expand(4, -1))


def test_add_rejects_missing_keys_and_wrong_shapes() -> None:
    storage = _storage()
    valid_agent = {
        "actor_obs": torch.zeros((2, 2, 3)),
        "actions": torch.zeros((2, 2, 2)),
    }
    valid_team = {
        "critic_obs": torch.zeros((2, 4)),
        "rewards": torch.zeros((2, 1)),
    }

    with pytest.raises(ValueError, match="keys mismatch"):
        storage.add(agent={"actor_obs": valid_agent["actor_obs"]}, team=valid_team)
    with pytest.raises(ValueError, match="must have shape"):
        storage.add(
            agent={**valid_agent, "actions": torch.zeros((2, 1, 2))},
            team=valid_team,
        )


def test_rollout_must_be_full_before_sampling_and_clear_allows_reuse() -> None:
    storage = _storage()
    _add_step(storage, 0)
    with pytest.raises(RuntimeError, match="before rollout is full"):
        next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))

    _add_step(storage, 1)
    storage.clear()
    assert storage.step == 0
    _add_step(storage, 2)
    assert storage.step == 1


def test_storage_rejects_invalid_sizes_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="num_agents must be positive"):
        MultiAgentRolloutStorage(
            num_envs=1,
            num_agents=0,
            num_transitions_per_env=1,
        )

    storage = _storage()
    with pytest.raises(ValueError, match="already registered"):
        storage.register_team("actor_obs", (1,))


def test_deferred_team_buffer_is_written_after_rollout() -> None:
    storage = _storage()
    storage.register_team("advantages", (1,), deferred=True)
    _add_step(storage, 0)
    _add_step(storage, 1)
    advantages = torch.arange(4, dtype=torch.float).reshape(2, 2, 1)

    storage.set_team("advantages", advantages)

    torch.testing.assert_close(storage.team("advantages"), advantages)
    next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    with pytest.raises(ValueError, match="not registered as deferred"):
        storage.set_team("rewards", advantages)


def test_unwritten_deferred_team_buffer_blocks_sampling() -> None:
    storage = _storage()
    storage.register_team("advantages", (1,), deferred=True)
    _add_step(storage, 0)
    _add_step(storage, 1)

    with pytest.raises(RuntimeError, match="Deferred team buffers were not written"):
        next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
