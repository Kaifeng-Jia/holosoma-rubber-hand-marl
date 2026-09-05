"""Fresh team-reward MAPPO specialization for the CORE4D small-table demo."""

from __future__ import annotations

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


CORE4D_SMALLTABLE_MAPPO_VERSION = "core4d_smalltable_shared_actor_mappo_v2"
CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256 = (
    "582e76693f877c61b0b09ab3b584922f330aeb85cae6035f6b7cd2ae729ee153"
)
CORE4D_SMALLTABLE_OBJECT_URDF_SHA256 = (
    "d4f166913ee6464dae1155428bdfe5169f63be94fbc8869535710bb6b672fc60"
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


def expected_core4d_smalltable_checkpoint_metadata() -> dict[str, Any]:
    return {
        "version": CORE4D_SMALLTABLE_MAPPO_VERSION,
        "num_agents": CORE4D_SMALLTABLE_NUM_AGENTS,
        "actor_obs_dim": CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
        "critic_obs_dim": CORE4D_SMALLTABLE_CRITIC_OBS_DIM,
        "action_dim": CORE4D_SMALLTABLE_ACTION_DIM,
        "initialization": CORE4D_SMALLTABLE_INITIALIZATION,
        "runtime_reference_sha256": CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
        "object_urdf_sha256": CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
        "training_promotion_sha256": CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
        "physics_contract": CORE4D_SMALLTABLE_PHYSICS_CONTRACT,
        "reward_contract": CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION,
        "termination_contract": CORE4D_SMALLTABLE_TERMINATION_CONTRACT_VERSION,
    }


def validate_core4d_smalltable_checkpoint(state: Mapping[str, Any]) -> int:
    metadata = state.get("core4d_smalltable_mappo")
    expected = expected_core4d_smalltable_checkpoint_metadata()
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
    """Reuse the proven team GAE/PPO objective with the 164-D Actor layout."""

    def __init__(
        self,
        models: Plan5ModelBundle,
        config: PPOConfig,
        *,
        num_envs: int,
        num_steps_per_env: int | None = None,
        device: str = "cpu",
    ) -> None:
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
            expected_core4d_smalltable_checkpoint_metadata()
        )
        return state

    def _validate_training_state(self, state: dict[str, Any]) -> None:
        validate_core4d_smalltable_checkpoint(state)


__all__ = [
    "CORE4D_SMALLTABLE_MAPPO_VERSION",
    "CORE4D_SMALLTABLE_OBJECT_URDF_SHA256",
    "CORE4D_SMALLTABLE_PHYSICS_CONTRACT",
    "CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION",
    "CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256",
    "CORE4D_SMALLTABLE_TERMINATION_CONTRACT_VERSION",
    "CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256",
    "Core4DSmallTablePPO",
    "expected_core4d_smalltable_checkpoint_metadata",
    "validate_core4d_smalltable_checkpoint",
]
