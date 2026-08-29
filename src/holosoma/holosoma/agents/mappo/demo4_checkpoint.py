"""Lossless 158-to-164 Actor expansion for cooperative Demo 4 rotation."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from holosoma.agents.mappo.ppo import (
    PLAN5_MAPPO_CHECKPOINT_VERSION,
)
from holosoma.agents.ppo.checkpoint_compat import (
    ACTOR_FIRST_WEIGHT,
    NORMALIZER_VECTOR_KEYS,
)


DEMO4_ACTOR_COMPATIBILITY_VERSION = "demo4_pull_actor_158_to_164_table_v1"
DEMO4_MAPPO_CHECKPOINT_VERSION = "demo4_cooperative_rotate_team_mappo_v1"
DEMO4_CHECKPOINT_INTERVAL = 1000
DEMO4_STATIC_RUNTIME_SHA256 = (
    "3eb482e1bdb9072b50054c5501cda1d4646bef39ef754df55c03c281626199f3"
)
DEMO4_REFERENCE_FRAMES = 316
DEMO4_REFERENCE_FPS = 50
DEMO4_TARGET_YAW_DEGREES = 90.0
DEMO4_MAXIMUM_SUCCESS_TILT_DEGREES = 60.0
DEMO4_OBJECT_MASS_KG = 20.0
DEMO4_OBJECT_MATERIAL = (0.5, 0.5, 0.0)
DEMO4_YAW_PROGRESS_REWARD_WEIGHT = 10.0
DEMO4_SUCCESS_BONUS_WEIGHT = 5.0
DEMO4_ROBOT_ASSET = "main_mesh_collision_rubberhand.urdf"
DEMO4_TABLE_OBS_CONTAINS_YAW_RATE = False


def demo4_static_training_contract() -> dict[str, Any]:
    """Return the immutable task facts embedded in every Demo 4 checkpoint."""

    return {
        "static_runtime_sha256": DEMO4_STATIC_RUNTIME_SHA256,
        "reference_frames": DEMO4_REFERENCE_FRAMES,
        "reference_fps": DEMO4_REFERENCE_FPS,
        "target_yaw_degrees": DEMO4_TARGET_YAW_DEGREES,
        "maximum_success_tilt_degrees": DEMO4_MAXIMUM_SUCCESS_TILT_DEGREES,
        "yaw_progress_representation": "per_step_unwrapped_heading_accumulator",
        "unsafe_terminal_task_reward": False,
        "object_mass_kg": DEMO4_OBJECT_MASS_KG,
        "object_material_static_dynamic_restitution": list(DEMO4_OBJECT_MATERIAL),
        "yaw_progress_reward_weight": DEMO4_YAW_PROGRESS_REWARD_WEIGHT,
        "success_bonus_weight": DEMO4_SUCCESS_BONUS_WEIGHT,
        "robot_asset": DEMO4_ROBOT_ASSET,
        "rubber_hand_collision": True,
        "table_obs_contains_yaw_rate": DEMO4_TABLE_OBS_CONTAINS_YAW_RATE,
        "table_reference_role": "reset_and_schema_only_not_tracking_target",
    }


def validate_demo4_static_runtime(path: str | Path) -> str:
    """Fail closed unless ``path`` is the frozen static Demo 4 runtime NPZ."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Demo 4 static runtime does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != DEMO4_STATIC_RUNTIME_SHA256:
        raise ValueError(
            "Demo 4 static runtime SHA-256 mismatch: "
            f"expected {DEMO4_STATIC_RUNTIME_SHA256}, got {actual}"
        )
    return actual


@dataclass(frozen=True)
class Demo4TableObservationExpansion:
    """Frozen input layout for the Demo 4 shared Actor."""

    source_dim: int = 158
    table_dim: int = 6
    target_dim: int = 164
    group_name: str = "table_obs"

    def __post_init__(self) -> None:
        if self.source_dim + self.table_dim != self.target_dim:
            raise ValueError("Demo 4 target_dim must equal source_dim + table_dim")


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint field {key!r} must be a dictionary")
    return value


def _validate_plan5_pull_source(
    checkpoint: dict[str, Any],
    spec: Demo4TableObservationExpansion,
) -> None:
    metadata = _require_mapping(checkpoint, "plan5_mappo")
    required_metadata = {
        "version": PLAN5_MAPPO_CHECKPOINT_VERSION,
        "num_agents": 2,
        "actor_obs_dim": spec.source_dim,
        "critic_obs_dim": 527,
        "action_dim": 29,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Incompatible Plan 5 Pull checkpoint metadata: {mismatches}")

    actor = _require_mapping(checkpoint, "actor_model_state_dict")
    normalizer = _require_mapping(checkpoint, "actor_obs_normalizer_state_dict")
    actor_weight = actor.get(ACTOR_FIRST_WEIGHT)
    if not isinstance(actor_weight, torch.Tensor) or actor_weight.ndim != 2:
        raise ValueError(f"Missing two-dimensional Actor tensor {ACTOR_FIRST_WEIGHT!r}")
    if actor_weight.shape[1] != spec.source_dim:
        raise ValueError(
            f"Expected source Actor input dimension {spec.source_dim}, "
            f"got {actor_weight.shape[1]}"
        )
    for key in NORMALIZER_VECTOR_KEYS:
        value = normalizer.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (1, spec.source_dim):
            shape = getattr(value, "shape", None)
            raise ValueError(
                f"Expected Actor normalizer {key} shape (1, {spec.source_dim}), got {shape}"
            )
    count = normalizer.get("count")
    if not isinstance(count, torch.Tensor) or count.numel() != 1:
        raise ValueError("Expected scalar Actor normalizer count")


def expand_plan5_pull_actor_for_demo4(
    checkpoint: dict[str, Any],
    *,
    source_file_sha256: str,
    spec: Demo4TableObservationExpansion = Demo4TableObservationExpansion(),
) -> dict[str, Any]:
    """Return a minimal Demo 4 Actor artifact without mutating its source.

    The old centralized Critic, both optimizer states, and the old Critic
    normalizer are intentionally absent.  Demo 4 builds all of those fresh.
    """

    _validate_plan5_pull_source(checkpoint, spec)
    source_actor = checkpoint["actor_model_state_dict"]
    source_normalizer = checkpoint["actor_obs_normalizer_state_dict"]
    actor = copy.deepcopy(source_actor)
    normalizer = copy.deepcopy(source_normalizer)

    source_weight = source_actor[ACTOR_FIRST_WEIGHT]
    expanded_weight = source_weight.new_zeros((source_weight.shape[0], spec.target_dim))
    expanded_weight[:, : spec.source_dim].copy_(source_weight)
    actor[ACTOR_FIRST_WEIGHT] = expanded_weight

    for key in NORMALIZER_VECTOR_KEYS:
        source_value = source_normalizer[key]
        identity_fill = 0.0 if key == "_mean" else 1.0
        expanded_value = source_value.new_full((1, spec.target_dim), identity_fill)
        expanded_value[:, : spec.source_dim].copy_(source_value)
        normalizer[key] = expanded_value

    source_metadata = checkpoint["plan5_mappo"]
    return {
        "demo4_actor_compatibility": {
            "version": DEMO4_ACTOR_COMPATIBILITY_VERSION,
            "source_file_sha256": source_file_sha256,
            "source_iteration": int(checkpoint.get("iter", 0)),
            "source_plan5_version": source_metadata["version"],
            "source_plan5_actor_origin_sha256": source_metadata.get("source_sha256"),
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
            "critic_policy": "fresh_527_dim_team_critic",
            "optimizer_policy": "fresh_actor_and_critic_optimizers",
        },
        "actor_model_state_dict": actor,
        "actor_obs_normalizer_state_dict": normalizer,
        "iter": int(checkpoint.get("iter", 0)),
    }


def validate_demo4_lossless_expansion(
    source: dict[str, Any],
    converted: dict[str, Any],
    spec: Demo4TableObservationExpansion = Demo4TableObservationExpansion(),
) -> None:
    """Raise if the expansion changes any existing Actor behavior state."""

    _validate_plan5_pull_source(source, spec)
    expected_keys = {
        "demo4_actor_compatibility",
        "actor_model_state_dict",
        "actor_obs_normalizer_state_dict",
        "iter",
    }
    if set(converted) != expected_keys:
        raise ValueError(
            "Demo 4 Actor artifact must contain only isolated Actor state; "
            f"got {sorted(converted)}"
        )

    source_actor = source["actor_model_state_dict"]
    converted_actor = converted["actor_model_state_dict"]
    if source_actor.keys() != converted_actor.keys():
        raise ValueError("Actor state-dict keys changed during Demo 4 conversion")
    for key, source_value in source_actor.items():
        target_value = converted_actor[key]
        if key == ACTOR_FIRST_WEIGHT:
            torch.testing.assert_close(
                target_value[:, : spec.source_dim],
                source_value,
                rtol=0.0,
                atol=0.0,
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

    metadata = converted.get("demo4_actor_compatibility", {})
    if metadata.get("version") != DEMO4_ACTOR_COMPATIBILITY_VERSION:
        raise ValueError("Demo 4 Actor compatibility metadata is missing or invalid")
    if metadata.get("contains_yaw_rate") is not False:
        raise ValueError("Demo 4 Actor input must not contain table yaw rate")


def is_demo4_periodic_checkpoint(iteration: int) -> bool:
    """Return whether a positive iteration is on the frozen 1000-step cadence."""

    return iteration > 0 and iteration % DEMO4_CHECKPOINT_INTERVAL == 0


__all__ = [
    "DEMO4_ACTOR_COMPATIBILITY_VERSION",
    "DEMO4_CHECKPOINT_INTERVAL",
    "DEMO4_MAPPO_CHECKPOINT_VERSION",
    "DEMO4_OBJECT_MASS_KG",
    "DEMO4_OBJECT_MATERIAL",
    "DEMO4_REFERENCE_FPS",
    "DEMO4_REFERENCE_FRAMES",
    "DEMO4_ROBOT_ASSET",
    "DEMO4_STATIC_RUNTIME_SHA256",
    "DEMO4_SUCCESS_BONUS_WEIGHT",
    "DEMO4_TABLE_OBS_CONTAINS_YAW_RATE",
    "DEMO4_TARGET_YAW_DEGREES",
    "DEMO4_YAW_PROGRESS_REWARD_WEIGHT",
    "Demo4TableObservationExpansion",
    "demo4_static_training_contract",
    "expand_plan5_pull_actor_for_demo4",
    "is_demo4_periodic_checkpoint",
    "validate_demo4_static_runtime",
    "validate_demo4_lossless_expansion",
]
