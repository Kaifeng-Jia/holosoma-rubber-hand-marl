"""Demo 4 accepted-Pull warm-start tests."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from holosoma.agents.mappo.demo4_initialization import (
    DEMO4_ACTOR_OBS_DIM,
    DEMO4_CRITIC_OBS_DIM,
    DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
    DEMO4_SOURCE_PULL_ITERATION,
    initialize_demo4_model_bundle,
)
from holosoma.agents.mappo.initialization import FrozenEmpiricalNormalization
from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


REPO_ROOT = Path(__file__).resolve().parents[5]
PULL_08050 = (
    REPO_ROOT
    / "logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt"
)


def _first_linear_weight(module: torch.nn.Module) -> torch.Tensor:
    return next(layer.weight for layer in module.modules() if isinstance(layer, torch.nn.Linear))


def test_demo4_initialization_preserves_pull_actor_and_uses_fresh_training_state() -> None:
    config = g1_29dof_wbt_w_object.algo.config
    source_checkpoint = torch.load(PULL_08050, map_location="cpu", weights_only=False)
    actor_config = replace(
        copy.deepcopy(config.module_dict.actor),
        input_dim=["actor_obs", "teammate_obs"],
    )
    source_actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": 154, "teammate_obs": 4},
        module_config=actor_config,
        num_actions=29,
        init_noise_std=config.init_noise_std,
        device="cpu",
        history_length={"actor_obs": 1, "teammate_obs": 1},
    )
    source_actor.load_state_dict(source_checkpoint["actor_model_state_dict"], strict=True)
    source_normalizer = FrozenEmpiricalNormalization(158, device="cpu")
    source_normalizer.load_state_dict(
        source_checkpoint["actor_obs_normalizer_state_dict"],
        strict=True,
    )
    demo4_bundle = initialize_demo4_model_bundle(PULL_08050, config, device="cpu")

    assert demo4_bundle.source_iteration == DEMO4_SOURCE_PULL_ITERATION
    assert demo4_bundle.source_sha256 == DEMO4_SOURCE_PULL_CHECKPOINT_SHA256
    assert _first_linear_weight(demo4_bundle.actor).shape[1] == DEMO4_ACTOR_OBS_DIM
    assert _first_linear_weight(demo4_bundle.critic).shape[1] == DEMO4_CRITIC_OBS_DIM
    torch.testing.assert_close(
        _first_linear_weight(demo4_bundle.actor)[:, 158:],
        torch.zeros(512, 6),
        rtol=0.0,
        atol=0.0,
    )
    assert demo4_bundle.actor_optimizer.state == {}
    assert demo4_bundle.critic_optimizer.state == {}
    assert demo4_bundle.critic_obs_normalizer.count.item() == 0

    generator = torch.Generator().manual_seed(721)
    source_obs = torch.randn(41, 158, generator=generator)
    table_obs = torch.randn(41, 6, generator=generator)
    source_action = source_actor.act_inference(
        {
            "actor_obs": source_normalizer(
                source_obs,
                update=False,
            )
        }
    )
    demo4_action = demo4_bundle.actor.act_inference(
        {
            "actor_obs": demo4_bundle.actor_obs_normalizer(
                torch.cat((source_obs, table_obs), dim=-1),
                update=False,
            )
        }
    )
    torch.testing.assert_close(demo4_action, source_action, rtol=0.0, atol=0.0)


def test_demo4_initialization_fails_closed_on_source_hash() -> None:
    with pytest.raises(ValueError, match="source Pull checkpoint SHA256 mismatch"):
        initialize_demo4_model_bundle(
            PULL_08050,
            g1_29dof_wbt_w_object.algo.config,
            expected_sha256="0" * 64,
            device="cpu",
        )
