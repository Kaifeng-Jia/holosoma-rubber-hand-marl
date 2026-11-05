"""CPU tests for Demo 3's explicit per-agent rollout layout."""

import pytest
import torch

from holosoma.agents.mappo.demo3_storage import Demo3RolloutStorage


def _add_step(storage: Demo3RolloutStorage, step: int) -> None:
    row_ids = 100.0 * step + 10.0 * torch.arange(3).unsqueeze(1) + torch.arange(2)
    actor_obs = torch.zeros(3, 2, 164)
    critic_obs = torch.zeros(3, 2, 527)
    actions = torch.zeros(3, 2, 29)
    actor_obs[..., 0] = row_ids
    critic_obs[..., 0] = row_ids
    actions[..., 0] = row_ids
    scalar = row_ids.unsqueeze(-1)
    storage.add(
        agent={
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "actions": actions,
            "rewards": scalar,
            "values": scalar + 1.0,
            "actions_log_prob": scalar + 2.0,
            "action_mean": actions + 3.0,
            "action_sigma": actions + 4.0,
        },
        shared={
            "dones": torch.tensor([[False], [step == 0], [False]]),
            "timeouts": torch.zeros(3, 1, dtype=torch.bool),
        },
    )


def test_demo3_storage_preserves_agent_targets_in_minibatches() -> None:
    storage = Demo3RolloutStorage(num_envs=3, num_transitions_per_env=2)
    _add_step(storage, 0)
    _add_step(storage, 1)
    returns = storage.agent("rewards") + 5.0
    advantages = storage.agent("rewards") + 6.0
    storage.set_agent("returns", returns)
    storage.set_agent("advantages", advantages)

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    actor_ids = batch["agent"]["actor_obs"][:, 0].reshape(-1, 2)
    critic_ids = batch["agent"]["critic_obs"][:, 0].reshape(-1, 2)
    reward_ids = batch["agent"]["rewards"].reshape(-1, 2)
    advantage_ids = batch["agent"]["advantages"].reshape(-1, 2) - 6.0

    torch.testing.assert_close(actor_ids, critic_ids)
    torch.testing.assert_close(actor_ids, reward_ids)
    torch.testing.assert_close(actor_ids, advantage_ids)
    torch.testing.assert_close(actor_ids[:, 1] - actor_ids[:, 0], torch.ones(6))
    assert batch["shared"]["dones"].shape == (6, 1)
    assert storage.agent("returns").shape == (2, 3, 2, 1)


def test_demo3_storage_fails_closed_for_old_team_shapes_and_deferred_targets() -> None:
    storage = Demo3RolloutStorage(num_envs=3, num_transitions_per_env=1)
    with pytest.raises(ValueError, match="rewards.*must have shape"):
        storage.add(
            agent={
                "actor_obs": torch.zeros(3, 2, 164),
                "critic_obs": torch.zeros(3, 2, 527),
                "actions": torch.zeros(3, 2, 29),
                "rewards": torch.zeros(3, 1),
                "values": torch.zeros(3, 2, 1),
                "actions_log_prob": torch.zeros(3, 2, 1),
                "action_mean": torch.zeros(3, 2, 29),
                "action_sigma": torch.ones(3, 2, 29),
            },
            shared={
                "dones": torch.zeros(3, 1, dtype=torch.bool),
                "timeouts": torch.zeros(3, 1, dtype=torch.bool),
            },
        )

    _add_step(storage, 0)
    with pytest.raises(RuntimeError, match="Deferred Demo 3 agent buffers"):
        next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
