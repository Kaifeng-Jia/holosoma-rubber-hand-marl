"""Warm-start boundaries for the Plan 5 actor and fresh centralized critic."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from holosoma.agents.mappo.initialization import (
    PLAN5_ACTOR_CHECKPOINT_SHA256,
    PLAN5_ACTOR_OBS_DIM,
    PLAN5_CRITIC_OBS_DIM,
    initialize_plan5_model_bundle,
)
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


REPO_ROOT = Path(__file__).resolve().parents[5]
CHECKPOINT = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"


def _first_linear_weight(module: torch.nn.Module) -> torch.Tensor:
    return next(layer.weight for layer in module.modules() if isinstance(layer, torch.nn.Linear))


def test_plan5_initialization_loads_only_actor_side_state() -> None:
    torch.manual_seed(7)
    bundle = initialize_plan5_model_bundle(
        CHECKPOINT,
        g1_29dof_wbt_w_object.algo.config,
        device="cpu",
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    assert bundle.source_sha256 == PLAN5_ACTOR_CHECKPOINT_SHA256
    assert bundle.source_iteration == 7999
    assert _first_linear_weight(bundle.actor).shape[1] == PLAN5_ACTOR_OBS_DIM
    torch.testing.assert_close(
        _first_linear_weight(bundle.actor)[:, 154:],
        torch.zeros(512, 4),
    )
    assert _first_linear_weight(bundle.critic).shape[1] == PLAN5_CRITIC_OBS_DIM
    assert _first_linear_weight(bundle.critic).shape != _first_linear_weight(bundle.actor).shape
    assert all(parameter.requires_grad for parameter in bundle.actor.parameters())
    assert all(parameter.requires_grad for parameter in bundle.critic.parameters())
    assert bundle.actor_optimizer.state == {}
    assert bundle.critic_optimizer.state == {}

    actor_state = bundle.actor.state_dict()
    for name, expected in checkpoint["actor_model_state_dict"].items():
        torch.testing.assert_close(actor_state[name], expected)
    assert g1_29dof_wbt_w_object.algo.config.module_dict.actor.input_dim == ["actor_obs"]

    actor_input = bundle.actor_obs_normalizer(torch.zeros(4, PLAN5_ACTOR_OBS_DIM))
    actor_output = bundle.actor.act_inference({"actor_obs": actor_input})
    critic_input = bundle.critic_obs_normalizer(torch.zeros(4, PLAN5_CRITIC_OBS_DIM))
    critic_output = bundle.critic.evaluate({"critic_obs": critic_input})
    assert actor_output.shape == (4, 29)
    assert critic_output.shape == (4, 1)
    assert torch.isfinite(actor_output).all()
    assert torch.isfinite(critic_output).all()


def test_actor_normalizer_is_frozen_and_critic_normalizer_is_fresh() -> None:
    bundle = initialize_plan5_model_bundle(
        CHECKPOINT,
        g1_29dof_wbt_w_object.algo.config,
        device="cpu",
    )
    actor_count = bundle.actor_obs_normalizer.count.clone()
    actor_mean = bundle.actor_obs_normalizer._mean.clone()

    bundle.actor_obs_normalizer.train()
    _ = bundle.actor_obs_normalizer(torch.randn(32, PLAN5_ACTOR_OBS_DIM), update=True)

    assert not bundle.actor_obs_normalizer.training
    torch.testing.assert_close(bundle.actor_obs_normalizer.count, actor_count)
    torch.testing.assert_close(bundle.actor_obs_normalizer._mean, actor_mean)
    assert bundle.critic_obs_normalizer.training
    assert bundle.critic_obs_normalizer.count.item() == 0
    torch.testing.assert_close(
        bundle.critic_obs_normalizer._mean,
        torch.zeros(1, PLAN5_CRITIC_OBS_DIM),
    )


def test_frozen_a1_actor_is_invariant_to_random_teammate_channels() -> None:
    torch.manual_seed(721)
    bundle = initialize_plan5_model_bundle(
        CHECKPOINT,
        g1_29dof_wbt_w_object.algo.config,
        device="cpu",
    )
    base_observation = torch.randn(257, 154)
    zero_teammate = torch.zeros(257, 4)
    random_teammate = torch.empty(257, 4).uniform_(-1.0, 1.0)

    def action(teammate: torch.Tensor) -> torch.Tensor:
        observation = torch.cat((base_observation, teammate), dim=-1)
        normalized = bundle.actor_obs_normalizer(observation, update=False)
        return bundle.actor.act_inference({"actor_obs": normalized})

    torch.testing.assert_close(
        action(zero_teammate),
        action(random_teammate),
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_hash_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        initialize_plan5_model_bundle(
            CHECKPOINT,
            g1_29dof_wbt_w_object.algo.config,
            device="cpu",
            expected_sha256="0" * 64,
        )
