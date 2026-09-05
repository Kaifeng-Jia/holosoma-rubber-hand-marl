"""Training and deterministic-evaluation observations for the CORE4D demo."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.marl.g1.observation import (
    plan5_actor_obs,
    plan5_centralized_critic_obs,
    plan5_teammate_obs,
)


CORE4D_SMALLTABLE_TABLE_OBS_DIM = 6
CORE4D_SMALLTABLE_ACTOR_OBS_DIM = 164
CORE4D_SMALLTABLE_CRITIC_OBS_DIM = 527

core4d_smalltable_table_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "position_b": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.core4d_smalltable:"
                "table_relative_position_b"
            )
        ),
        "velocity_b": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.core4d_smalltable:"
                "table_relative_linear_velocity_b"
            )
        ),
    },
)

g1_29dof_core4d_smalltable_observation = ObservationManagerCfg(
    groups={
        "actor_obs": plan5_actor_obs,
        "teammate_obs": plan5_teammate_obs,
        "table_obs": core4d_smalltable_table_obs,
        "critic_obs": plan5_centralized_critic_obs,
    }
)

g1_29dof_core4d_smalltable_evaluation_observation = replace(
    g1_29dof_core4d_smalltable_observation,
    groups={
        **g1_29dof_core4d_smalltable_observation.groups,
        "actor_obs": replace(plan5_actor_obs, enable_noise=False),
    },
)

__all__ = [
    "CORE4D_SMALLTABLE_ACTOR_OBS_DIM",
    "CORE4D_SMALLTABLE_CRITIC_OBS_DIM",
    "CORE4D_SMALLTABLE_TABLE_OBS_DIM",
    "core4d_smalltable_table_obs",
    "g1_29dof_core4d_smalltable_evaluation_observation",
    "g1_29dof_core4d_smalltable_observation",
]
