"""Tests for lossless Stage-1A PPO checkpoint expansion."""

from __future__ import annotations

import copy

import pytest
import torch

from holosoma.agents.ppo.checkpoint_compat import (
    ACTOR_FIRST_WEIGHT,
    OPTIMIZER_KEYS,
    TeammateObservationExpansion,
    expand_ppo_checkpoint_for_teammate_obs,
    validate_lossless_expansion,
)


def _checkpoint() -> dict:
    generator = torch.Generator().manual_seed(7)
    return {
        "actor_model_state_dict": {
            "std": torch.rand(29, generator=generator),
            ACTOR_FIRST_WEIGHT: torch.rand((8, 154), generator=generator),
            "actor_module.module.0.bias": torch.rand(8, generator=generator),
        },
        "critic_model_state_dict": {
            "critic_module.module.0.weight": torch.rand((8, 298), generator=generator),
            "critic_module.module.0.bias": torch.rand(8, generator=generator),
        },
        "actor_obs_normalizer_state_dict": {
            "_mean": torch.rand((1, 154), generator=generator),
            "_var": torch.rand((1, 154), generator=generator),
            "_std": torch.rand((1, 154), generator=generator),
            "count": torch.tensor(123456, dtype=torch.long),
        },
        "actor_optimizer_state_dict": {"state": {1: "old"}},
        "critic_optimizer_state_dict": {"state": {2: "old"}},
        "experiment_config": {
            "env_class": "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
            "algo": {
                "config": {
                    "load_optimizer": True,
                    "module_dict": {"actor": {"input_dim": ["actor_obs"]}},
                }
            },
            "observation": {"groups": {"actor_obs": {}, "critic_obs": {}}},
        },
    }


def test_expansion_is_lossless_and_does_not_mutate_source() -> None:
    source = _checkpoint()
    frozen_source = copy.deepcopy(source)
    converted = expand_ppo_checkpoint_for_teammate_obs(source, source_sha256="abc")

    validate_lossless_expansion(source, converted)
    validate_lossless_expansion(frozen_source, converted)
    assert source.keys() == frozen_source.keys()
    assert converted["actor_model_state_dict"][ACTOR_FIRST_WEIGHT].shape == (8, 158)
    assert torch.count_nonzero(converted["actor_model_state_dict"][ACTOR_FIRST_WEIGHT][:, 154:]) == 0
    for key in OPTIMIZER_KEYS:
        assert key not in converted
    assert converted["experiment_config"]["algo"]["config"]["load_optimizer"] is False
    assert converted["experiment_config"]["algo"]["config"]["module_dict"]["actor"][
        "input_dim"
    ] == ["actor_obs", "teammate_obs"]


def test_new_normalizer_dimensions_use_fixed_identity_statistics() -> None:
    source = _checkpoint()
    converted = expand_ppo_checkpoint_for_teammate_obs(source, source_sha256="abc")
    normalizer = converted["actor_obs_normalizer_state_dict"]

    torch.testing.assert_close(normalizer["_mean"][:, 154:], torch.zeros((1, 4)))
    torch.testing.assert_close(normalizer["_var"][:, 154:], torch.ones((1, 4)))
    torch.testing.assert_close(normalizer["_std"][:, 154:], torch.ones((1, 4)))
    assert normalizer["count"].item() == 123456


def test_rejects_wrong_source_actor_dimension() -> None:
    source = _checkpoint()
    source["actor_model_state_dict"][ACTOR_FIRST_WEIGHT] = torch.zeros((8, 153))
    with pytest.raises(ValueError, match="Actor input dimension 154"):
        expand_ppo_checkpoint_for_teammate_obs(source, source_sha256="abc")


def test_expansion_contract_dimensions() -> None:
    spec = TeammateObservationExpansion()
    assert spec.source_dim == 154
    assert spec.teammate_dim == 4
    assert spec.target_dim == 158
    assert spec.critic_dim == 298
