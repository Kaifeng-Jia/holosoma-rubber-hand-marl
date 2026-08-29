"""Lossless checkpoint and initialization contracts for Demo 3."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

from holosoma.agents.mappo.demo3_checkpoint import (
    Demo3TableObservationExpansion,
    expand_actor_checkpoint_for_demo3_table_obs,
    validate_demo3_lossless_expansion,
)
from holosoma.agents.mappo.demo3_initialization import initialize_demo3_model_bundle
from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle
from holosoma.agents.ppo.checkpoint_compat import ACTOR_FIRST_WEIGHT
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


REPO_ROOT = Path(__file__).resolve().parents[5]
PULL_CHECKPOINT = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt"
PULL_CHECKPOINT_SHA256 = "f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_pull_actor_expands_without_mutating_source() -> None:
    source = torch.load(PULL_CHECKPOINT, map_location="cpu", weights_only=False)
    frozen_source = copy.deepcopy(source)
    converted = expand_actor_checkpoint_for_demo3_table_obs(
        source,
        source_sha256=PULL_CHECKPOINT_SHA256,
    )
    validate_demo3_lossless_expansion(source, converted)
    validate_demo3_lossless_expansion(frozen_source, converted)

    spec = Demo3TableObservationExpansion()
    assert source["actor_model_state_dict"][ACTOR_FIRST_WEIGHT].shape[1] == spec.source_dim
    assert converted["actor_model_state_dict"][ACTOR_FIRST_WEIGHT].shape[1] == spec.target_dim
    assert converted["demo3_compatibility"]["contains_yaw_rate"] is False
    assert converted["demo3_compatibility"]["table_order"][-2:] == [
        "relative_yaw_sin",
        "relative_yaw_cos",
    ]
    table_terms = converted["experiment_config"]["observation"]["groups"][
        "table_obs"
    ]["terms"]
    assert all(
        term["func"].startswith("holosoma.managers.observation.terms.demo3_tug:")
        for term in table_terms.values()
    )


def test_demo3_loader_preserves_pull_actions_and_builds_fresh_critic(tmp_path) -> None:
    source = torch.load(PULL_CHECKPOINT, map_location="cpu", weights_only=False)
    converted = expand_actor_checkpoint_for_demo3_table_obs(
        source,
        source_sha256=PULL_CHECKPOINT_SHA256,
    )
    converted_path = tmp_path / "pull_actor164.pt"
    torch.save(converted, converted_path)
    converted_sha256 = _sha256(converted_path)

    source_bundle = initialize_plan5_model_bundle(
        PULL_CHECKPOINT,
        g1_29dof_wbt_w_object.algo.config,
        expected_sha256=PULL_CHECKPOINT_SHA256,
        device="cpu",
    )
    demo3_bundle = initialize_demo3_model_bundle(
        converted_path,
        g1_29dof_wbt_w_object.algo.config,
        expected_sha256=converted_sha256,
        device="cpu",
    )
    generator = torch.Generator().manual_seed(721)
    source_obs = torch.randn((31, 158), generator=generator)
    table_obs = torch.randn((31, 6), generator=generator)
    source_action = source_bundle.actor.act_inference(
        {"actor_obs": source_bundle.actor_obs_normalizer(source_obs, update=False)}
    )
    demo3_action = demo3_bundle.actor.act_inference(
        {
            "actor_obs": demo3_bundle.actor_obs_normalizer(
                torch.cat((source_obs, table_obs), dim=-1),
                update=False,
            )
        }
    )

    torch.testing.assert_close(demo3_action, source_action, rtol=0.0, atol=0.0)
    first_actor_weight = next(
        layer.weight for layer in demo3_bundle.actor.modules() if isinstance(layer, torch.nn.Linear)
    )
    first_critic_weight = next(
        layer.weight for layer in demo3_bundle.critic.modules() if isinstance(layer, torch.nn.Linear)
    )
    assert first_actor_weight.shape[1] == 164
    assert first_critic_weight.shape[1] == 527
    assert demo3_bundle.critic_optimizer.state == {}
