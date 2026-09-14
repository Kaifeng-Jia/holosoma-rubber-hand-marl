"""Fresh team-reward MAPPO specialization for the CORE4D small-table demo."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from holosoma.agents.mappo.batch_layout import HomogeneousAgentBatchLayout
from holosoma.agents.mappo.core4d_smalltable_initialization import (
    CORE4D_SMALLTABLE_ACTION_DIM,
    CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
    CORE4D_SMALLTABLE_CRITIC_OBS_DIM,
    CORE4D_SMALLTABLE_INITIALIZATION,
    CORE4D_SMALLTABLE_NUM_AGENTS,
)
from holosoma.agents.mappo.core4d_smalltable_runner import (
    Core4DSmallTablePolicyRunner,
)
from holosoma.agents.mappo.initialization import Plan5ModelBundle
from holosoma.agents.mappo.ppo import Plan5PPO
from holosoma.config_types.algo import PPOConfig
from holosoma.config_values.marl.g1.core4d_smalltable_reward import validate_object_z_error_weight


CORE4D_SMALLTABLE_MAPPO_VERSION = "core4d_smalltable_shared_actor_mappo_158_v1"
CORE4D_PAIR_MAPPO_VERSION = "core4d_pair_shared_actor_mappo_158_v1"
CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256 = (
    "582e76693f877c61b0b09ab3b584922f330aeb85cae6035f6b7cd2ae729ee153"
)
CORE4D_SMALLTABLE_OBJECT_URDF_SHA256 = (
    "d4f166913ee6464dae1155428bdfe5169f63be94fbc8869535710bb6b672fc60"
)
CORE4D_SMALLTABLE_TRAINING_ROBOT_URDF_SHA256 = (
    "7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25"
)
CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256 = (
    "6d55d7235c49456dd2793cf5ecafe6abd2420800504eb664e7ac0027efcba584"
)
CORE4D_SMALLTABLE_PHYSICS_CONTRACT = {
    "object_mass_kg": 20.0,
    "material_static_dynamic_restitution": (0.5, 0.5, 0.0),
    "physics_hz": 200,
    "control_hz": 50,
}
CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION = "plan5_tracking_with_object_v1"
CORE4D_SMALLTABLE_TERMINATION_CONTRACT_VERSION = "nonloop_joint_tracking_v1"
CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION = (
    "plan5_tracking_with_object_plus_interaction_mesh_v1"
)
CORE4D_SMALLTABLE_INTERACTION_CONTRACT_VERSION = "core4d_smalltable_interaction_mesh_v1"
CORE4D_OBJECT_POSITION_TRACKING_CONTRACT_VERSION = "world_xyz_squared_error_weighting_v1"
_INTERACTION_FIELDS = {
    "version",
    "reference_file_sha256",
    "runtime_reference_sha256",
    "object_urdf_sha256",
    "training_robot_urdf_sha256",
    "object_points_sha256",
    "requested_object_points",
    "actual_object_points",
    "num_body_points",
    "sigma",
    "weight",
}

_EXPERIMENT_HASH_FIELDS = (
    "source_pair_sha256",
    "runtime_reference_sha256",
    "object_urdf_sha256",
    "training_promotion_sha256",
)
_EXPERIMENT_FIELDS = {
    "experiment_id", "object_name", "reference_frames", "reference_fps",
    "physics_contract", *_EXPERIMENT_HASH_FIELDS,
}
_EXPERIMENT_PHYSICS_FIELDS = {
    "object_mass_kg", "material_static_dynamic_restitution", "physics_hz",
    "control_hz", "object_collider_type",
}


def _validated_experiment_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a descriptor-selected contract, never infer one from a checkpoint.

    Asset identity and reviewed values belong to the calling descriptor. Here we
    validate their schema and consistency, then require exact equality on resume.
    This avoids duplicating each new object's constants inside the PPO learner.
    """
    if not isinstance(contract, Mapping) or set(contract) != _EXPERIMENT_FIELDS:
        raise ValueError("CORE4D experiment contract has missing or unexpected fields")
    result = copy.deepcopy(dict(contract))
    for key in ("experiment_id", "object_name"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"CORE4D experiment {key} must be a non-empty string")
    for key in _EXPERIMENT_HASH_FIELDS:
        if not isinstance(result[key], str) or re.fullmatch(r"[0-9a-f]{64}", result[key]) is None:
            raise ValueError(f"CORE4D experiment {key} must be a lowercase SHA-256 digest")
    for key, minimum in (("reference_frames", 3), ("reference_fps", 1)):
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(f"CORE4D experiment {key} must be an integer >= {minimum}")
    physics = result["physics_contract"]
    if not isinstance(physics, Mapping) or set(physics) != _EXPERIMENT_PHYSICS_FIELDS:
        raise ValueError("CORE4D experiment physics contract has missing or unexpected fields")
    physics = dict(physics)
    mass = physics["object_mass_kg"]
    if type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0:
        raise ValueError("CORE4D experiment object_mass_kg must be positive and finite")
    physics["object_mass_kg"] = float(mass)
    material = physics["material_static_dynamic_restitution"]
    if (
        not isinstance(material, (tuple, list))
        or len(material) != 3
        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in material)
        or material[2] > 1
    ):
        raise ValueError("CORE4D experiment material must be finite non-negative friction and restitution in [0, 1]")
    physics["material_static_dynamic_restitution"] = tuple(float(value) for value in material)
    for key in ("physics_hz", "control_hz"):
        if type(physics[key]) is not int or physics[key] < 1:
            raise ValueError(f"CORE4D experiment {key} must be a positive integer")
    if physics["control_hz"] != result["reference_fps"] or physics["physics_hz"] % physics["control_hz"]:
        raise ValueError("CORE4D experiment reference/control FPS and physics decimation must agree")
    if physics["object_collider_type"] not in ("convex_hull", "convex_decomposition"):
        raise ValueError("CORE4D experiment object_collider_type must be convex_hull or convex_decomposition")
    result["physics_contract"] = physics
    return result


def _validated_interaction_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a complete, reviewed contract; never accept a free-form variant tag."""
    if not isinstance(contract, Mapping) or set(contract) != _INTERACTION_FIELDS:
        raise ValueError("CORE4D interaction contract has missing or unexpected fields")
    result = dict(contract)
    if result["version"] != CORE4D_SMALLTABLE_INTERACTION_CONTRACT_VERSION:
        raise ValueError("CORE4D interaction contract has an unsupported version")
    for key in (
        "reference_file_sha256",
        "runtime_reference_sha256",
        "object_urdf_sha256",
        "training_robot_urdf_sha256",
        "object_points_sha256",
    ):
        digest = result[key]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"CORE4D interaction {key} must be a lowercase SHA-256 digest")
    for key, expected in (
        ("runtime_reference_sha256", CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256),
        ("object_urdf_sha256", CORE4D_SMALLTABLE_OBJECT_URDF_SHA256),
        ("training_robot_urdf_sha256", CORE4D_SMALLTABLE_TRAINING_ROBOT_URDF_SHA256),
    ):
        if result[key] != expected:
            raise ValueError(f"CORE4D interaction {key} differs from the frozen baseline")
    for key, expected in (
        ("requested_object_points", 100),
        ("actual_object_points", 85),
        ("num_body_points", 19),
    ):
        if type(result[key]) is not int or result[key] != expected:
            raise ValueError(f"CORE4D interaction {key} must be {expected}")
    for key, expected in (("sigma", 0.06), ("weight", 1.0)):
        if type(result[key]) not in (int, float) or result[key] != expected:
            raise ValueError(f"CORE4D interaction {key} must be {expected}")
        result[key] = float(result[key])
    return result


def build_core4d_interaction_contract(reference_file: str | Path) -> dict[str, Any]:
    """Bind the reviewed graph asset and frozen training geometry to a checkpoint.

    This loads only numeric/JSON data. It does not import a retargeting solver,
    regenerate a graph, alter any model or start a simulator.
    """
    import numpy as np

    path = Path(reference_file).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        if not isinstance(metadata, dict):
            raise ValueError("CORE4D interaction asset metadata must be a JSON object")
        if metadata.get("version") != CORE4D_SMALLTABLE_INTERACTION_CONTRACT_VERSION:
            raise ValueError("CORE4D interaction asset has an unsupported version")
        points = np.asarray(archive["object_points"])
        if points.shape != (85, 3) or points.dtype != np.dtype("float64"):
            raise ValueError("CORE4D interaction asset requires 85 float64 object points")
        if not np.isfinite(points).all():
            raise ValueError("CORE4D interaction object points must be finite")
        points_digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
        if metadata.get("object_points_sha256") != points_digest:
            raise ValueError("CORE4D interaction object point SHA-256 mismatch")
        for key, expected in (("num_frames", 687), ("fps", 50), ("num_agents", 2), ("seed", 42)):
            if metadata.get(key) != expected:
                raise ValueError(f"CORE4D interaction asset {key} must be {expected}")
        contract = {
            key: metadata.get(key)
            for key in _INTERACTION_FIELDS.difference({"reference_file_sha256", "sigma", "weight"})
        }
    contract.update(
        reference_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sigma=0.06,
        weight=1.0,
    )
    return _validated_interaction_contract(contract)


def core4d_object_position_tracking_contract(object_z_error_weight: float) -> dict[str, Any] | None:
    """Keep legacy metadata absent at z=1; explicitly bind all new position semantics."""
    value = validate_object_z_error_weight(object_z_error_weight)
    if value == 1.0:
        return None
    return {
        "version": CORE4D_OBJECT_POSITION_TRACKING_CONTRACT_VERSION,
        "frame": "world",
        "squared_error_weights_xyz": (1.0, 1.0, value),
        "sigma_m": 0.3,
        "reward_weight": 1.0,
    }


def _checkpoint_object_z_error_weight(metadata: Mapping[str, Any]) -> float:
    if "object_position_tracking" not in metadata:
        return 1.0
    contract = metadata["object_position_tracking"]
    if not isinstance(contract, Mapping):
        raise ValueError("CORE4D object_position_tracking must be a position contract")
    weights = contract.get("squared_error_weights_xyz")
    if not isinstance(weights, tuple) or len(weights) != 3:
        raise ValueError("CORE4D object_position_tracking requires three squared-error weights")
    for weight in weights:
        validate_object_z_error_weight(weight)
    for key in ("sigma_m", "reward_weight"):
        validate_object_z_error_weight(contract.get(key))
    value = validate_object_z_error_weight(weights[2])
    expected = core4d_object_position_tracking_contract(value)
    if expected is None or contract != expected:
        raise ValueError("CORE4D object_position_tracking contract mismatch")
    return value


def expected_core4d_smalltable_checkpoint_metadata(
    *, interaction_contract: Mapping[str, Any] | None = None,
    experiment_contract: Mapping[str, Any] | None = None,
    object_z_error_weight: float = 1.0,
) -> dict[str, Any]:
    """Preserve the baseline dictionary exactly unless a reviewed variant is supplied."""
    if experiment_contract is not None and interaction_contract is not None:
        raise ValueError("CORE4D custom experiments do not support interaction_mesh")
    position_contract = core4d_object_position_tracking_contract(object_z_error_weight)
    if experiment_contract is not None and position_contract is not None:
        raise ValueError("Non-default object_z_error_weight is restricted to smalltable")
    metadata = {
        "version": CORE4D_SMALLTABLE_MAPPO_VERSION,
        "num_agents": CORE4D_SMALLTABLE_NUM_AGENTS,
        "actor_obs_dim": CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
        "critic_obs_dim": CORE4D_SMALLTABLE_CRITIC_OBS_DIM,
        "action_dim": CORE4D_SMALLTABLE_ACTION_DIM,
        "initialization": CORE4D_SMALLTABLE_INITIALIZATION,
        "runtime_reference_sha256": CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
        "object_urdf_sha256": CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
        "training_promotion_sha256": CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
        "physics_contract": dict(CORE4D_SMALLTABLE_PHYSICS_CONTRACT),
        "reward_contract": CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION,
        "termination_contract": CORE4D_SMALLTABLE_TERMINATION_CONTRACT_VERSION,
    }
    if interaction_contract is not None:
        metadata["reward_contract"] = CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION
        metadata["interaction_mesh"] = _validated_interaction_contract(interaction_contract)
    if experiment_contract is not None:
        contract = _validated_experiment_contract(experiment_contract)
        metadata["version"] = CORE4D_PAIR_MAPPO_VERSION
        for key in ("runtime_reference_sha256", "object_urdf_sha256", "training_promotion_sha256"):
            metadata[key] = contract[key]
        metadata["physics_contract"] = copy.deepcopy(contract["physics_contract"])
        metadata["experiment_contract"] = contract
    if position_contract is not None:
        metadata["object_position_tracking"] = position_contract
    return metadata


def validate_core4d_smalltable_checkpoint(
    state: Mapping[str, Any], *, experiment_contract: Mapping[str, Any] | None = None,
) -> int:
    metadata = state.get("core4d_smalltable_mappo")
    if isinstance(metadata, Mapping):
        if "experiment_contract" in metadata and experiment_contract is None:
            raise ValueError("CORE4D custom checkpoint requires an explicit experiment_contract")
        if experiment_contract is not None and (
            "interaction_mesh" in metadata
            or metadata.get("reward_contract") == CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION
        ):
            raise ValueError("CORE4D custom experiments do not support interaction_mesh")
    interaction_contract = None
    if isinstance(metadata, Mapping) and (
        metadata.get("reward_contract") == CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION
    ):
        interaction_contract = _validated_interaction_contract(metadata.get("interaction_mesh"))
    expected = expected_core4d_smalltable_checkpoint_metadata(
        interaction_contract=interaction_contract,
        experiment_contract=experiment_contract,
        object_z_error_weight=(
            _checkpoint_object_z_error_weight(metadata) if isinstance(metadata, Mapping) else 1.0
        ),
    )
    if metadata != expected:
        raise ValueError(
            "CORE4D small-table checkpoint metadata mismatch: "
            f"expected {expected!r}, got {metadata!r}"
        )
    contaminants = sorted({"demo3_mappo", "demo4_mappo", "plan5_mappo"}.intersection(state))
    if contaminants:
        raise ValueError(f"CORE4D checkpoint contains cross-demo metadata: {contaminants}")
    required = {
        "actor_model_state_dict",
        "critic_model_state_dict",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
        "actor_obs_normalizer_state_dict",
        "critic_obs_normalizer_state_dict",
        "iter",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"CORE4D checkpoint is incomplete: {missing}")
    iteration = state["iter"]
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"CORE4D checkpoint iteration must be non-negative, got {iteration!r}")
    return iteration


class Core4DSmallTablePPO(Plan5PPO):
    """Reuse the proven team GAE/PPO objective with the 158-D Actor layout."""

    def __init__(
        self,
        models: Plan5ModelBundle,
        config: PPOConfig,
        *,
        num_envs: int,
        num_steps_per_env: int | None = None,
        device: str = "cpu",
        interaction_contract: Mapping[str, Any] | None = None,
        experiment_contract: Mapping[str, Any] | None = None,
        object_z_error_weight: float = 1.0,
    ) -> None:
        # Freeze a value copy before constructing models/storage. Resume must match it.
        self._checkpoint_metadata = expected_core4d_smalltable_checkpoint_metadata(
            interaction_contract=interaction_contract,
            experiment_contract=experiment_contract,
            object_z_error_weight=object_z_error_weight,
        )
        super().__init__(
            models,
            config,
            num_envs=num_envs,
            num_steps_per_env=num_steps_per_env,
            layout=HomogeneousAgentBatchLayout(
                num_agents=CORE4D_SMALLTABLE_NUM_AGENTS,
                actor_obs_dim=CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
                action_dim=CORE4D_SMALLTABLE_ACTION_DIM,
            ),
            device=device,
        )
        self.runner = Core4DSmallTablePolicyRunner(models)

    def training_state_dict(self, *, iteration: int) -> dict[str, Any]:
        state = super().training_state_dict(iteration=iteration)
        state.pop("plan5_mappo", None)
        state["core4d_smalltable_mappo"] = copy.deepcopy(self._checkpoint_metadata)
        return state

    def _validate_training_state(self, state: dict[str, Any]) -> None:
        validate_core4d_smalltable_checkpoint(
            state, experiment_contract=self._checkpoint_metadata.get("experiment_contract"),
        )
        if state["core4d_smalltable_mappo"] != self._checkpoint_metadata:
            raise ValueError(
                "CORE4D small-table resume reward contract mismatch: "
                "baseline, interaction variant, position tracking and reference hashes must match the learner"
            )


__all__ = [
    "CORE4D_PAIR_MAPPO_VERSION",
    "CORE4D_OBJECT_POSITION_TRACKING_CONTRACT_VERSION",
    "CORE4D_SMALLTABLE_MAPPO_VERSION",
    "CORE4D_SMALLTABLE_INTERACTION_CONTRACT_VERSION",
    "CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION",
    "CORE4D_SMALLTABLE_OBJECT_URDF_SHA256",
    "CORE4D_SMALLTABLE_PHYSICS_CONTRACT",
    "CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION",
    "CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256",
    "CORE4D_SMALLTABLE_TRAINING_ROBOT_URDF_SHA256",
    "CORE4D_SMALLTABLE_TERMINATION_CONTRACT_VERSION",
    "CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256",
    "Core4DSmallTablePPO",
    "build_core4d_interaction_contract",
    "core4d_object_position_tracking_contract",
    "expected_core4d_smalltable_checkpoint_metadata",
    "validate_core4d_smalltable_checkpoint",
]
