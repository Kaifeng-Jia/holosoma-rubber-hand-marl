"""Actor observation preset for the square-table competitive Demo 3."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.marl.g1.observation import (
    plan5_actor_obs,
    plan5_teammate_obs,
)


DEMO3_TABLE_OBS_DIM = 6
DEMO3_ACTOR_OBS_DIM = 164
DEMO3_CRITIC_OBS_DIM = 527

demo3_table_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "position_b": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.demo3_tug:"
                "table_relative_position_b"
            )
        ),
        "velocity_b": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.demo3_tug:"
                "table_linear_velocity_b"
            )
        ),
        "yaw_sin_cos": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.demo3_tug:"
                "table_relative_yaw_sin_cos"
            )
        ),
    },
)

demo3_ego_critic_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "ego_ordered_global_state": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.demo3_tug:"
                "ego_ordered_critic_obs"
            )
        )
    },
)

g1_29dof_demo3_observation = ObservationManagerCfg(
    groups={
        "actor_obs": plan5_actor_obs,
        "teammate_obs": plan5_teammate_obs,
        "table_obs": demo3_table_obs,
        "critic_obs": demo3_ego_critic_obs,
    }
)

__all__ = [
    "DEMO3_ACTOR_OBS_DIM",
    "DEMO3_CRITIC_OBS_DIM",
    "DEMO3_TABLE_OBS_DIM",
    "demo3_ego_critic_obs",
    "demo3_table_obs",
    "g1_29dof_demo3_observation",
]
