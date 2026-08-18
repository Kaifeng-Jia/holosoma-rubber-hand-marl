"""Per-agent observation preset for the first Plan 5 Push A1 environment."""

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
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

plan5_centralized_critic_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "agent_actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_actions"
        ),
        "agent_base_ang_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_base_ang_vel_b"
        ),
        "agent_base_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_base_lin_vel_b"
        ),
        "agent_body_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_body_ori_b"
        ),
        "agent_body_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_body_pos_b"
        ),
        "agent_dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_dof_pos"
        ),
        "agent_dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_dof_vel"
        ),
        "agent_motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_motion_ref_ori_b"
        ),
        "agent_motion_ref_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_agent_motion_ref_pos_b"
        ),
        "shared_motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_shared_motion_command"
        ),
        "shared_object_tracking": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_shared_object_tracking"
        ),
        "shared_phase": ObsTermCfg(
            func="holosoma.managers.observation.terms.marl:centralized_shared_phase"
        ),
    },
)

g1_29dof_plan5_observation = ObservationManagerCfg(
    groups={
        "actor_obs": plan5_actor_obs,
        "teammate_obs": plan5_teammate_obs,
        "critic_obs": plan5_centralized_critic_obs,
    }
)

__all__ = ["g1_29dof_plan5_observation", "plan5_centralized_critic_obs"]
