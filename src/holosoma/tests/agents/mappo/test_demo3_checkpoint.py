"""Lossless checkpoint and initialization contracts for Demo 3."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from holosoma.agents.mappo.demo3_checkpoint import (
    DEMO3_RUNTIME_REFERENCE_SHA256,
    Demo3TableObservationExpansion,
    demo3_ppo_contract,
    demo3_training_contract,
    expand_actor_checkpoint_for_demo3_table_obs,
    is_demo3_periodic_checkpoint,
    validate_demo3_asset,
    validate_demo3_lossless_expansion,
)
from holosoma.agents.mappo.demo3_initialization import initialize_demo3_model_bundle
from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle
from holosoma.agents.ppo.checkpoint_compat import ACTOR_FIRST_WEIGHT
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


REPO_ROOT = Path(__file__).resolve().parents[5]
PULL_CHECKPOINT = REPO_ROOT / "logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt"
PULL_CHECKPOINT_SHA256 = "f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364"
DEMO3_RUNTIME_REFERENCE = REPO_ROOT / (
    "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "demo3_tug/sub3_010_diagonal_tug_runtime.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_demo3_formal_training_contract_and_checkpoint_cadence() -> None:
    contract = demo3_training_contract()

    assert contract["runtime_reference_sha256"] == DEMO3_RUNTIME_REFERENCE_SHA256
    assert contract["warm_start_iteration"] == 7999
    assert contract["reference_frames"] == 317
    assert contract["reference_fps"] == 50
    assert contract["object_mass_kg"] == 20.0
    assert contract["object_material_static_dynamic_restitution"] == [0.5, 0.5, 0.0]
    assert contract["signed_progress_reward_weight"] == 10.0
    assert contract["table_obs_contains_yaw_rate"] is False
    assert contract["critic_only_iterations"] == 50
    assert contract["full_actor_iterations"] == 8000
    assert contract["checkpoint_interval"] == 1000
    assert contract["randomization"].startswith("fixed_frame0")
    assert is_demo3_periodic_checkpoint(
        50,
        critic_only_iterations=50,
        checkpoint_interval=1000,
    )
    assert is_demo3_periodic_checkpoint(
        1050,
        critic_only_iterations=50,
        checkpoint_interval=1000,
    )
    assert not is_demo3_periodic_checkpoint(
        1000,
        critic_only_iterations=50,
        checkpoint_interval=1000,
    )


def test_demo3_ppo_contract_captures_update_semantics() -> None:
    config = g1_29dof_wbt_w_object.algo.config
    contract = demo3_ppo_contract(config, num_steps_per_env=24)

    assert contract["num_steps_per_env"] == 24
    assert contract["module_dict"]["actor"]["layer_config"]["activation"] == "ELU"
    assert contract["module_dict"]["actor"]["layer_config"]["hidden_dims"] == [
        512,
        256,
        128,
    ]
    assert contract["module_dict"]["critic"]["layer_config"]["activation"] == "ELU"
    assert contract["num_learning_epochs"] == 5
    assert contract["num_mini_batches"] == 4
    assert contract["gamma"] == 0.99
    assert contract["gae_lambda"] == 0.95
    assert contract["clip_param"] == 0.2
    assert contract["entropy_coef"] == 0.005
    assert contract["value_loss_coef"] == 1.0
    assert contract["max_grad_norm"] == 1.0
    assert contract["schedule"] == "adaptive"
    assert contract["desired_kl"] == 0.01
    assert contract["actor_optimizer"] == {
        "target": "torch.optim.AdamW",
        "weight_decay": 0.0,
    }
    assert contract["critic_optimizer"] == {
        "target": "torch.optim.AdamW",
        "weight_decay": 0.0,
    }


def test_demo3_runtime_hash_validation_fails_closed_on_tamper(tmp_path) -> None:
    assert validate_demo3_asset(
        DEMO3_RUNTIME_REFERENCE,
        expected_sha256=DEMO3_RUNTIME_REFERENCE_SHA256,
        label="runtime reference",
    ) == DEMO3_RUNTIME_REFERENCE_SHA256

    tampered = tmp_path / "runtime.npz"
    tampered.write_bytes(DEMO3_RUNTIME_REFERENCE.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_demo3_asset(
            tampered,
            expected_sha256=DEMO3_RUNTIME_REFERENCE_SHA256,
            label="runtime reference",
        )


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
