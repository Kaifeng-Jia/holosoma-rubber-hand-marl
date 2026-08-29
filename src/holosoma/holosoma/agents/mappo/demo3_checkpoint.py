"""Lossless 158-to-164 Actor expansion for the competitive Demo 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from holosoma.agents.ppo.checkpoint_compat import (
    ACTOR_FIRST_WEIGHT,
    NORMALIZER_VECTOR_KEYS,
    OPTIMIZER_KEYS,
)


DEMO3_COMPATIBILITY_VERSION = "pull_actor_obs_158_to_164_table_v1"


@dataclass(frozen=True)
class Demo3TableObservationExpansion:
    source_dim: int = 158
    table_dim: int = 6
    group_name: str = "table_obs"

    @property
    def target_dim(self) -> int:
        return self.source_dim + self.table_dim


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint field {key!r} must be a dictionary")
    return value


def _validate_source(
    checkpoint: dict[str, Any],
    spec: Demo3TableObservationExpansion,
) -> None:
    actor = _require_mapping(checkpoint, "actor_model_state_dict")
    normalizer = _require_mapping(checkpoint, "actor_obs_normalizer_state_dict")
    config = _require_mapping(checkpoint, "experiment_config")
    actor_weight = actor.get(ACTOR_FIRST_WEIGHT)
    if not isinstance(actor_weight, torch.Tensor) or actor_weight.ndim != 2:
        raise ValueError(f"Missing two-dimensional Actor tensor {ACTOR_FIRST_WEIGHT!r}")
    if actor_weight.shape[1] != spec.source_dim:
        raise ValueError(
            f"Expected source Actor input dimension {spec.source_dim}, got {actor_weight.shape[1]}"
        )
    for key in NORMALIZER_VECTOR_KEYS:
        value = normalizer.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (1, spec.source_dim):
            shape = getattr(value, "shape", None)
            raise ValueError(
                f"Expected normalizer {key} shape (1, {spec.source_dim}), got {shape}"
            )
    count = normalizer.get("count")
    if not isinstance(count, torch.Tensor) or count.numel() != 1:
        raise ValueError("Expected scalar actor normalizer count")
    actor_inputs = config["algo"]["config"]["module_dict"]["actor"]["input_dim"]
    if actor_inputs != ["actor_obs", "teammate_obs"]:
        raise ValueError(
            "Expected source Actor input groups ['actor_obs', 'teammate_obs'], "
            f"got {actor_inputs}"
        )
    groups = config["observation"]["groups"]
    if spec.group_name in groups:
        raise ValueError(f"Source checkpoint already contains group {spec.group_name!r}")


def _table_group_config() -> dict[str, Any]:
    prefix = "holosoma.managers.observation.terms.demo3_tug"
    return {
        "concatenate": True,
        "enable_noise": False,
        "history_length": 1,
        "terms": {
            "position_b": {
                "clip": None,
                "func": f"{prefix}:table_relative_position_b",
                "noise": 0.0,
                "params": {},
                "scale": 1.0,
            },
            "velocity_b": {
                "clip": None,
                "func": f"{prefix}:table_linear_velocity_b",
                "noise": 0.0,
                "params": {},
                "scale": 1.0,
            },
            "yaw_sin_cos": {
                "clip": None,
                "func": f"{prefix}:table_relative_yaw_sin_cos",
                "noise": 0.0,
                "params": {},
                "scale": 1.0,
            },
        },
    }


def expand_actor_checkpoint_for_demo3_table_obs(
    checkpoint: dict[str, Any],
    *,
    source_sha256: str,
    spec: Demo3TableObservationExpansion = Demo3TableObservationExpansion(),
) -> dict[str, Any]:
    """Return a losslessly expanded Actor checkpoint without mutating its source."""
    _validate_source(checkpoint, spec)
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
    algo_config = config["algo"]["config"]
    algo_config["load_optimizer"] = False
    algo_config["module_dict"]["actor"]["input_dim"] = [
        "actor_obs",
        "teammate_obs",
        spec.group_name,
    ]
    config["observation"]["groups"][spec.group_name] = _table_group_config()
    converted["demo3_compatibility"] = {
        "version": DEMO3_COMPATIBILITY_VERSION,
        "source_sha256": source_sha256,
        "source_actor_dim": spec.source_dim,
        "table_dim": spec.table_dim,
        "target_actor_dim": spec.target_dim,
        "actor_group_order": ["actor_obs", "teammate_obs", spec.group_name],
        "table_order": [
            "relative_position_b_x",
            "relative_position_b_y",
            "linear_velocity_b_x",
            "linear_velocity_b_y",
            "relative_yaw_sin",
            "relative_yaw_cos",
        ],
        "contains_yaw_rate": False,
        "new_actor_columns": "zeros",
        "new_normalizer_mean": 0.0,
        "new_normalizer_var_std": 1.0,
        "normalizer_count_preserved": True,
        "removed_optimizer_keys": removed_optimizer_keys,
        "critic_state_policy": "preserved_in_file_but_ignored_and_reinitialized",
    }
    return converted


def validate_demo3_lossless_expansion(
    source: dict[str, Any],
    converted: dict[str, Any],
    spec: Demo3TableObservationExpansion = Demo3TableObservationExpansion(),
) -> None:
    """Raise if any source Actor state changed or any new input is non-neutral."""
    source_actor = source["actor_model_state_dict"]
    converted_actor = converted["actor_model_state_dict"]
    if source_actor.keys() != converted_actor.keys():
        raise ValueError("Actor state-dict keys changed during Demo 3 conversion")
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

    source_normalizer = source["actor_obs_normalizer_state_dict"]
    converted_normalizer = converted["actor_obs_normalizer_state_dict"]
    for key in NORMALIZER_VECTOR_KEYS:
        torch.testing.assert_close(
            converted_normalizer[key][:, : spec.source_dim],
            source_normalizer[key],
            rtol=0.0,
            atol=0.0,
        )
        expected_fill = 0.0 if key == "_mean" else 1.0
        torch.testing.assert_close(
            converted_normalizer[key][:, spec.source_dim :],
            torch.full_like(
                converted_normalizer[key][:, spec.source_dim :],
                expected_fill,
            ),
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        converted_normalizer["count"],
        source_normalizer["count"],
        rtol=0.0,
        atol=0.0,
    )


__all__ = [
    "DEMO3_COMPATIBILITY_VERSION",
    "Demo3TableObservationExpansion",
    "expand_actor_checkpoint_for_demo3_table_obs",
    "validate_demo3_lossless_expansion",
]
