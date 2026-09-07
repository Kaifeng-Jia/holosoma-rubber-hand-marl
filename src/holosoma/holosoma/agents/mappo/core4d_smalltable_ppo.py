"""Fresh team-reward MAPPO specialization for the CORE4D small-table demo."""

from __future__ import annotations

import hashlib
import json
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


CORE4D_SMALLTABLE_MAPPO_VERSION = "core4d_smalltable_shared_actor_mappo_158_v1"
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


def expected_core4d_smalltable_checkpoint_metadata(
    *, interaction_contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Preserve the baseline dictionary exactly unless a reviewed variant is supplied."""
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
    return metadata


def validate_core4d_smalltable_checkpoint(state: Mapping[str, Any]) -> int:
    metadata = state.get("core4d_smalltable_mappo")
    interaction_contract = None
    if isinstance(metadata, Mapping) and (
        metadata.get("reward_contract") == CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION
    ):
        interaction_contract = _validated_interaction_contract(metadata.get("interaction_mesh"))
    expected = expected_core4d_smalltable_checkpoint_metadata(
        interaction_contract=interaction_contract
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
    ) -> None:
        # Freeze a value copy before constructing models/storage. Resume must match it.
        self._checkpoint_metadata = expected_core4d_smalltable_checkpoint_metadata(
            interaction_contract=interaction_contract
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
        state["core4d_smalltable_mappo"] = (
            expected_core4d_smalltable_checkpoint_metadata(
                interaction_contract=self._checkpoint_metadata.get("interaction_mesh")
            )
        )
        return state

    def _validate_training_state(self, state: dict[str, Any]) -> None:
        validate_core4d_smalltable_checkpoint(state)
        if state["core4d_smalltable_mappo"] != self._checkpoint_metadata:
            raise ValueError(
                "CORE4D small-table resume reward contract mismatch: "
                "baseline, interaction variant, and reference hashes must match the learner"
            )


__all__ = [
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
    "expected_core4d_smalltable_checkpoint_metadata",
    "validate_core4d_smalltable_checkpoint",
]
