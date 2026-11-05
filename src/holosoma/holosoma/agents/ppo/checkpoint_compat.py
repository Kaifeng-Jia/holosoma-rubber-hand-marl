"""Lossless PPO checkpoint conversion for the Stage-1A teammate interface."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


ACTOR_FIRST_WEIGHT = "actor_module.module.0.weight"
CRITIC_FIRST_WEIGHT = "critic_module.module.0.weight"
NORMALIZER_VECTOR_KEYS = ("_mean", "_var", "_std")
OPTIMIZER_KEYS = ("actor_optimizer_state_dict", "critic_optimizer_state_dict")
COMPAT_ENV_CLASS = "holosoma.envs.wbt.wbt_marl_compat_manager.GhostTeammateWholeBodyTrackingManager"
POSITION_TERM = "holosoma.managers.observation.terms.marl:teammate_relative_position_b"
VELOCITY_TERM = "holosoma.managers.observation.terms.marl:teammate_relative_velocity_b"


@dataclass(frozen=True)
class TeammateObservationExpansion:
    """Contract for the frozen Stage-1A A1 conversion."""

    source_dim: int = 154
    teammate_dim: int = 4
    critic_dim: int = 298
    group_name: str = "teammate_obs"

    @property
    def target_dim(self) -> int:
        return self.source_dim + self.teammate_dim


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint field {key!r} must be a dictionary")
    return value


def _validate_checkpoint(checkpoint: dict[str, Any], spec: TeammateObservationExpansion) -> None:
    actor = _require_mapping(checkpoint, "actor_model_state_dict")
    critic = _require_mapping(checkpoint, "critic_model_state_dict")
    normalizer = _require_mapping(checkpoint, "actor_obs_normalizer_state_dict")
    config = _require_mapping(checkpoint, "experiment_config")

    actor_weight = actor.get(ACTOR_FIRST_WEIGHT)
    if not isinstance(actor_weight, torch.Tensor) or actor_weight.ndim != 2:
        raise ValueError(f"Missing two-dimensional Actor tensor {ACTOR_FIRST_WEIGHT!r}")
    if actor_weight.shape[1] != spec.source_dim:
        raise ValueError(
            f"Expected Actor input dimension {spec.source_dim}, got {actor_weight.shape[1]}"
        )

    critic_weight = critic.get(CRITIC_FIRST_WEIGHT)
    if not isinstance(critic_weight, torch.Tensor) or critic_weight.ndim != 2:
        raise ValueError(f"Missing two-dimensional Critic tensor {CRITIC_FIRST_WEIGHT!r}")
    if critic_weight.shape[1] != spec.critic_dim:
        raise ValueError(
            f"Expected Critic input dimension {spec.critic_dim}, got {critic_weight.shape[1]}"
        )

    for key in NORMALIZER_VECTOR_KEYS:
        value = normalizer.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (1, spec.source_dim):
            shape = getattr(value, "shape", None)
            raise ValueError(f"Expected normalizer {key} shape (1, {spec.source_dim}), got {shape}")
    count = normalizer.get("count")
    if not isinstance(count, torch.Tensor) or count.numel() != 1:
        raise ValueError("Expected scalar actor normalizer count")

    actor_inputs = config["algo"]["config"]["module_dict"]["actor"]["input_dim"]
    if actor_inputs != ["actor_obs"]:
        raise ValueError(f"Expected source Actor input groups ['actor_obs'], got {actor_inputs}")
    groups = config["observation"]["groups"]
    if spec.group_name in groups:
        raise ValueError(f"Source checkpoint already contains observation group {spec.group_name!r}")


def _teammate_group_config() -> dict[str, Any]:
    return {
        "concatenate": True,
        "enable_noise": False,
        "history_length": 1,
        "terms": {
            "relative_position_b": {
                "clip": [-1.0, 1.0],
                "func": POSITION_TERM,
                "noise": 0.0,
                "params": {},
                "scale": 1.0,
            },
            "relative_velocity_b": {
                "clip": [-1.0, 1.0],
                "func": VELOCITY_TERM,
                "noise": 0.0,
                "params": {},
                "scale": 1.0,
            },
        },
    }


def expand_ppo_checkpoint_for_teammate_obs(
    checkpoint: dict[str, Any],
    *,
    source_sha256: str,
    spec: TeammateObservationExpansion = TeammateObservationExpansion(),
) -> dict[str, Any]:
    """Return a losslessly expanded checkpoint without mutating the source."""
    _validate_checkpoint(checkpoint, spec)
    converted = copy.deepcopy(checkpoint)

    actor = converted["actor_model_state_dict"]
    source_weight = actor[ACTOR_FIRST_WEIGHT]
    expanded_weight = source_weight.new_zeros((source_weight.shape[0], spec.target_dim))
    expanded_weight[:, : spec.source_dim].copy_(source_weight)
    actor[ACTOR_FIRST_WEIGHT] = expanded_weight

    normalizer = converted["actor_obs_normalizer_state_dict"]
    for key in NORMALIZER_VECTOR_KEYS:
        source_value = normalizer[key]
        fill = 0.0 if key == "_mean" else 1.0
        expanded_value = source_value.new_full((1, spec.target_dim), fill)
        expanded_value[:, : spec.source_dim].copy_(source_value)
        normalizer[key] = expanded_value

    removed_optimizer_keys = []
    for key in OPTIMIZER_KEYS:
        if key in converted:
            converted.pop(key)
            removed_optimizer_keys.append(key)

    config = converted["experiment_config"]
    config["env_class"] = COMPAT_ENV_CLASS
    algo_config = config["algo"]["config"]
    algo_config["load_optimizer"] = False
    algo_config["module_dict"]["actor"]["input_dim"] = ["actor_obs", spec.group_name]
    config["observation"]["groups"][spec.group_name] = _teammate_group_config()

    converted["marl_compatibility"] = {
        "version": "a1_actor_obs_154_to_158_v1",
        "source_sha256": source_sha256,
        "source_actor_dim": spec.source_dim,
        "teammate_dim": spec.teammate_dim,
        "target_actor_dim": spec.target_dim,
        "critic_dim": spec.critic_dim,
        "teammate_order": [
            "relative_position_b_x",
            "relative_position_b_y",
            "relative_velocity_b_x",
            "relative_velocity_b_y",
        ],
        "fixed_input_units": ["m", "m", "m/s", "m/s"],
        "fixed_input_clip": [-1.0, 1.0],
        "new_actor_columns": "zeros",
        "new_normalizer_mean": 0.0,
        "new_normalizer_var_std": 1.0,
        "normalizer_count_preserved": True,
        "removed_optimizer_keys": removed_optimizer_keys,
    }
    return converted


def validate_lossless_expansion(
    source: dict[str, Any],
    converted: dict[str, Any],
    spec: TeammateObservationExpansion = TeammateObservationExpansion(),
) -> None:
    """Raise if any frozen tensor changed or any new column is nonzero."""
    source_actor = source["actor_model_state_dict"]
    converted_actor = converted["actor_model_state_dict"]
    if source_actor.keys() != converted_actor.keys():
        raise ValueError("Actor state-dict keys changed during conversion")
    for key, source_value in source_actor.items():
        target_value = converted_actor[key]
        if key == ACTOR_FIRST_WEIGHT:
            torch.testing.assert_close(
                target_value[:, : spec.source_dim], source_value, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                target_value[:, spec.source_dim :],
                torch.zeros_like(target_value[:, spec.source_dim :]),
                rtol=0.0,
                atol=0.0,
            )
        else:
            torch.testing.assert_close(target_value, source_value, rtol=0.0, atol=0.0)

    for key, source_value in source["critic_model_state_dict"].items():
        torch.testing.assert_close(
            converted["critic_model_state_dict"][key], source_value, rtol=0.0, atol=0.0
        )

    source_normalizer = source["actor_obs_normalizer_state_dict"]
    converted_normalizer = converted["actor_obs_normalizer_state_dict"]
    for key in NORMALIZER_VECTOR_KEYS:
        torch.testing.assert_close(
            converted_normalizer[key][:, : spec.source_dim],
            source_normalizer[key],
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        converted_normalizer["count"], source_normalizer["count"], rtol=0.0, atol=0.0
    )
