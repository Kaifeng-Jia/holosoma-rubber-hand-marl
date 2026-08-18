"""Per-agent observation preset for the first Plan 5 Push A1 environment."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    teammate_obs_marl_compat,
)


_actor_term_functions = {
    "motion_command": "paired_motion_command",
    "motion_ref_ori_b": "paired_motion_ref_ori_b",
    "base_ang_vel": "paired_base_ang_vel",
    "dof_pos": "paired_dof_pos",
    "dof_vel": "paired_dof_vel",
    "actions": "paired_actions",
}

plan5_actor_obs = replace(
    actor_obs_shared,
    terms={
        name: replace(
            term,
            func=f"holosoma.managers.observation.terms.marl:{_actor_term_functions[name]}",
        )
        for name, term in actor_obs_shared.terms.items()
    },
)

_teammate_term_functions = {
    "relative_position_b": "real_teammate_relative_position_b",
    "relative_velocity_b": "real_teammate_relative_velocity_b",
}

plan5_teammate_obs = replace(
    teammate_obs_marl_compat,
    terms={
        name: replace(
            term,
            func=f"holosoma.managers.observation.terms.marl:{_teammate_term_functions[name]}",
        )
        for name, term in teammate_obs_marl_compat.terms.items()
    },
)

g1_29dof_plan5_actor_observation = ObservationManagerCfg(
    groups={
        "actor_obs": plan5_actor_obs,
        "teammate_obs": plan5_teammate_obs,
    }
)

__all__ = ["g1_29dof_plan5_actor_observation"]
