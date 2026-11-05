"""Plan-5-compatible observations for the CORE4D small-table demo."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_values.marl.g1.observation import (
    plan5_actor_obs,
    plan5_centralized_critic_obs,
    plan5_teammate_obs,
)


CORE4D_SMALLTABLE_CRITIC_OBS_DIM = 527

g1_29dof_core4d_smalltable_observation = ObservationManagerCfg(
    groups={
        "actor_obs": plan5_actor_obs,
        "teammate_obs": plan5_teammate_obs,
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
    "CORE4D_SMALLTABLE_CRITIC_OBS_DIM",
    "g1_29dof_core4d_smalltable_evaluation_observation",
    "g1_29dof_core4d_smalltable_observation",
]
