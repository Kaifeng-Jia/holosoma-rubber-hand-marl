"""Lossless 158-to-164 Actor expansion for the competitive Demo 3."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

from holosoma.agents.ppo.checkpoint_compat import (
    ACTOR_FIRST_WEIGHT,
    NORMALIZER_VECTOR_KEYS,
    OPTIMIZER_KEYS,
)


DEMO3_COMPATIBILITY_VERSION = "pull_actor_obs_158_to_164_table_v1"
DEMO3_MAPPO_CHECKPOINT_VERSION = "demo3_ego_first_per_agent_mappo_v2"
DEMO3_CHECKPOINT_INTERVAL = 1000
DEMO3_DEFAULT_CRITIC_ONLY_ITERATIONS = 50
DEMO3_DEFAULT_FULL_ACTOR_ITERATIONS = 8000
DEMO3_WARM_START_SHA256 = (
    "048f952cad01d5fda42851502347ee626751dccab923905af303eb44807ef01b"
)
DEMO3_RUNTIME_REFERENCE_SHA256 = (
    "5baedb3f109402c521590701263facfa6b42649f4f18bdcefb5933d9b4a7caf5"
)
DEMO3_SQUARE_TABLE_URDF_SHA256 = (
    "386da8a4201a9365f7d4eef9e6ae2fb326c2777fac4998b72c4af6de693f9d1b"
)
DEMO3_ROBOT_URDF_SHA256 = (
    "7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25"
)
DEMO3_REFERENCE_FRAMES = 317
DEMO3_REFERENCE_FPS = 50
DEMO3_OBJECT_MASS_KG = 20.0
DEMO3_OBJECT_COM_M = (0.0, -0.012413473401914, 0.0)
DEMO3_OBJECT_INERTIA_KG_M2 = (
    0.821972006993735,
    0.0,
    0.0,
    0.0,
    1.206183371508506,
    0.0,
    0.0,
    0.0,
    0.821972006993735,
)
DEMO3_OBJECT_MATERIAL = (0.5, 0.5, 0.0)
DEMO3_SIGNED_PROGRESS_REWARD_WEIGHT = 10.0
DEMO3_ROBOT_ASSET = "main_mesh_collision_rubberhand.urdf"
DEMO3_TABLE_ASSET = "objects_squaretable_demo3_training.urdf"
DEMO3_TABLE_OBS_CONTAINS_YAW_RATE = False


def _sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Demo 3 required asset does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_demo3_asset(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> str:
    """Fail closed before Isaac startup if a frozen Demo 3 asset changed."""

    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Demo 3 {label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def demo3_ppo_contract(
    config: Any,
    *,
    num_steps_per_env: int,
) -> dict[str, Any]:
    """Serialize every non-LR PPO setting that changes Demo 3 updates.

    Optimizer learning rates are intentionally handled outside this immutable
    contract: a resume restores their checkpoint values, unless the operator
    explicitly supplies one of the two supported command-line overrides.
    """

    if num_steps_per_env < 1:
        raise ValueError("num_steps_per_env must be positive")

    def optimizer_contract(name: str) -> dict[str, Any]:
        optimizer = getattr(config, name)
        return {
            "target": str(optimizer._target_),
            "weight_decay": float(optimizer.weight_decay),
        }

    module_dict = config.module_dict
    if not is_dataclass(module_dict):
        raise TypeError("Demo 3 PPO module_dict must be a dataclass instance")
    # JSON round-trip canonicalizes tuples so checkpoint metadata and the
    # persisted run_config compare identically after a resume.
    module_contract = json.loads(json.dumps(asdict(module_dict), sort_keys=True))

    return {
        "module_dict": module_contract,
        "num_steps_per_env": int(num_steps_per_env),
        "num_learning_epochs": int(config.num_learning_epochs),
        "num_mini_batches": int(config.num_mini_batches),
        "clip_param": float(config.clip_param),
        "gamma": float(config.gamma),
        "gae_lambda": float(config.lam),
        "value_loss_coef": float(config.value_loss_coef),
        "entropy_coef": float(config.entropy_coef),
        "max_grad_norm": float(config.max_grad_norm),
        "schedule": str(config.schedule),
        "desired_kl": (
            None if config.desired_kl is None else float(config.desired_kl)
        ),
        "min_actor_learning_rate": (
            None
            if config.min_actor_learning_rate is None
            else float(config.min_actor_learning_rate)
        ),
        "max_actor_learning_rate": (
            None
            if config.max_actor_learning_rate is None
            else float(config.max_actor_learning_rate)
        ),
        "actor_optimizer": optimizer_contract("actor_optimizer"),
        "critic_optimizer": optimizer_contract("critic_optimizer"),
        "use_symmetry": bool(config.use_symmetry),
        "symmetry_actor_coef": float(config.symmetry_actor_coef),
        "symmetry_critic_coef": float(config.symmetry_critic_coef),
        "empirical_normalization": bool(config.empirical_normalization),
        "init_noise_std": float(config.init_noise_std),
        "learning_rate_resume_policy": (
            "restore_optimizer_state_unless_explicit_cli_override"
        ),
    }


def demo3_training_contract(
    *,
    critic_only_iterations: int = DEMO3_DEFAULT_CRITIC_ONLY_ITERATIONS,
    full_actor_iterations: int = DEMO3_DEFAULT_FULL_ACTOR_ITERATIONS,
    checkpoint_interval: int = DEMO3_CHECKPOINT_INTERVAL,
) -> dict[str, Any]:
    """Return immutable task facts plus the exact run-stage schedule."""

    if critic_only_iterations < 0:
        raise ValueError("critic_only_iterations must be non-negative")
    if full_actor_iterations < 1:
        raise ValueError("full_actor_iterations must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    return {
        "scenario": "competitive_square_table_diagonal_tug",
        "runtime_reference_sha256": DEMO3_RUNTIME_REFERENCE_SHA256,
        "square_table_urdf_sha256": DEMO3_SQUARE_TABLE_URDF_SHA256,
        "robot_urdf_sha256": DEMO3_ROBOT_URDF_SHA256,
        "warm_start_sha256": DEMO3_WARM_START_SHA256,
        "warm_start_iteration": 7999,
        "reference_frames": DEMO3_REFERENCE_FRAMES,
        "reference_fps": DEMO3_REFERENCE_FPS,
        "reference_start": "fixed_frame_zero",
        "reference_end": "clamp_without_looping",
        "table_reference_role": "reset_and_schema_only_not_tracking_target",
        "object_mass_kg": DEMO3_OBJECT_MASS_KG,
        "object_com_m": list(DEMO3_OBJECT_COM_M),
        "object_inertia_kg_m2_row_major": list(DEMO3_OBJECT_INERTIA_KG_M2),
        "object_material_static_dynamic_restitution": list(DEMO3_OBJECT_MATERIAL),
        "signed_progress_reward_weight": DEMO3_SIGNED_PROGRESS_REWARD_WEIGHT,
        "reward_weights": {
            "motion_global_ref_position_error_exp": 0.5,
            "motion_global_ref_orientation_error_exp": 0.5,
            "motion_relative_body_position_error_exp": 1.0,
            "motion_relative_body_orientation_error_exp": 1.0,
            "motion_global_body_lin_vel": 1.0,
            "motion_global_body_ang_vel": 1.0,
            "action_rate_l2": -0.1,
            "limits_dof_pos": -10.0,
            "signed_table_progress_velocity": DEMO3_SIGNED_PROGRESS_REWARD_WEIGHT,
        },
        "termination": {
            "reference_horizon_frames": DEMO3_REFERENCE_FRAMES,
            "minimum_ref_body_height_m": 0.25,
            "maximum_gravity_z": -0.2,
        },
        "robot_asset": DEMO3_ROBOT_ASSET,
        "table_asset": DEMO3_TABLE_ASSET,
        "rubber_hand_collision": True,
        "table_obs_contains_yaw_rate": DEMO3_TABLE_OBS_CONTAINS_YAW_RATE,
        "actor_parameter_sharing": "one_current_shared_actor_for_both_agents",
        "opponent_policy": "synchronous_current_shared_actor",
        "return_model": "per_agent_reward_value_return_advantage",
        "done_model": "shared_physical_episode",
        "randomization": "fixed_frame0_fixed_mass_fixed_material_no_opponent_randomization",
        "critic_only_iterations": int(critic_only_iterations),
        "full_actor_iterations": int(full_actor_iterations),
        "checkpoint_interval": int(checkpoint_interval),
    }


def is_demo3_periodic_checkpoint(
    iteration: int,
    *,
    critic_only_iterations: int,
    checkpoint_interval: int,
) -> bool:
    """Save the critic boundary and each full-Actor interval thereafter."""

    if iteration <= 0:
        return False
    if critic_only_iterations > 0 and iteration == critic_only_iterations:
        return True
    full_actor_iteration = iteration - critic_only_iterations
    return full_actor_iteration > 0 and full_actor_iteration % checkpoint_interval == 0


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
    "DEMO3_CHECKPOINT_INTERVAL",
    "DEMO3_COMPATIBILITY_VERSION",
    "DEMO3_DEFAULT_CRITIC_ONLY_ITERATIONS",
    "DEMO3_DEFAULT_FULL_ACTOR_ITERATIONS",
    "DEMO3_MAPPO_CHECKPOINT_VERSION",
    "DEMO3_OBJECT_COM_M",
    "DEMO3_OBJECT_INERTIA_KG_M2",
    "DEMO3_OBJECT_MASS_KG",
    "DEMO3_OBJECT_MATERIAL",
    "DEMO3_REFERENCE_FPS",
    "DEMO3_REFERENCE_FRAMES",
    "DEMO3_ROBOT_ASSET",
    "DEMO3_ROBOT_URDF_SHA256",
    "DEMO3_RUNTIME_REFERENCE_SHA256",
    "DEMO3_SIGNED_PROGRESS_REWARD_WEIGHT",
    "DEMO3_SQUARE_TABLE_URDF_SHA256",
    "DEMO3_TABLE_ASSET",
    "DEMO3_TABLE_OBS_CONTAINS_YAW_RATE",
    "DEMO3_WARM_START_SHA256",
    "Demo3TableObservationExpansion",
    "demo3_ppo_contract",
    "demo3_training_contract",
    "expand_actor_checkpoint_for_demo3_table_obs",
    "is_demo3_periodic_checkpoint",
    "validate_demo3_asset",
    "validate_demo3_lossless_expansion",
]
