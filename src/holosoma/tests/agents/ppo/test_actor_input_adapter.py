"""Tests for lossless first-layer actor input adaptation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.std = nn.Parameter(torch.ones(2))
        self.actor_module = _ActorModule()


class _ActorModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_indices_dict = {"actor_obs": slice(0, 3), "teammate_obs": slice(3, 5)}
        self.module = nn.Sequential(nn.Linear(5, 4), nn.ELU(), nn.Linear(4, 2))


def _ppo(actor: nn.Module, groups: list[str] | None, weight_decay: float = 0.0) -> PPO:
    instance = object.__new__(PPO)
    instance.actor = actor
    instance.config = SimpleNamespace(
        actor_input_adapter_groups=groups,
        actor_optimizer=SimpleNamespace(weight_decay=weight_decay),
    )
    return instance


def test_adapter_updates_only_selected_first_layer_columns() -> None:
    actor = _Actor()
    ppo = _ppo(actor, ["teammate_obs"])
    ppo._configure_actor_input_adapter()

    first_linear = actor.actor_module.module[0]
    original = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    optimizer = torch.optim.AdamW((p for p in actor.parameters() if p.requires_grad), lr=0.1, weight_decay=0.0)
    optimizer.zero_grad()
    actor.actor_module.module(torch.ones((1, 5))).sum().backward()
    optimizer.step()

    assert torch.count_nonzero(first_linear.weight[:, 3:] - original["actor_module.module.0.weight"][:, 3:])
    torch.testing.assert_close(first_linear.weight[:, :3], original["actor_module.module.0.weight"][:, :3])
    torch.testing.assert_close(first_linear.bias, original["actor_module.module.0.bias"])
    torch.testing.assert_close(actor.actor_module.module[2].weight, original["actor_module.module.2.weight"])
    torch.testing.assert_close(actor.std, original["std"])


def test_adapter_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="Unknown actor input adapter groups"):
        _ppo(_Actor(), ["missing"])._configure_actor_input_adapter()


def test_adapter_requires_zero_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay=0"):
        _ppo(_Actor(), ["teammate_obs"], weight_decay=0.001)._configure_actor_input_adapter()
