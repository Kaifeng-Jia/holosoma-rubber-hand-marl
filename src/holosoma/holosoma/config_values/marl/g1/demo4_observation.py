"""Actor and team-critic observation preset for Demo 4 rotation."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_values.marl.g1.observation import (
    plan5_actor_obs,
    plan5_centralized_critic_obs,
    plan5_teammate_obs,
)


DEMO4_TABLE_OBS_DIM = 6
DEMO4_ACTOR_OBS_DIM = 164
DEMO4_CRITIC_OBS_DIM = 527

demo4_table_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "position_b": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.demo4_rotate:"
                "table_relative_position_b"
            )
        ),
        "velocity_b": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.demo4_rotate:"
                "table_linear_velocity_b"
            )
        ),
        "yaw_sin_cos": ObsTermCfg(
            func=(
                "holosoma.managers.observation.terms.demo4_rotate:"
                "table_relative_yaw_sin_cos"
            )
        ),
    },
)

g1_29dof_demo4_rotate_observation = ObservationManagerCfg(
    groups={
        "actor_obs": plan5_actor_obs,
        "teammate_obs": plan5_teammate_obs,
        "table_obs": demo4_table_obs,
        "critic_obs": plan5_centralized_critic_obs,
    }
)

__all__ = [
    "DEMO4_ACTOR_OBS_DIM",
    "DEMO4_CRITIC_OBS_DIM",
    "DEMO4_TABLE_OBS_DIM",
    "demo4_table_obs",
    "g1_29dof_demo4_rotate_observation",
]
