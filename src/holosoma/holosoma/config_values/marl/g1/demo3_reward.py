"""Provisional reward preset for isolated competitive Demo 3."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


_TERMS = "holosoma.managers.reward.terms.demo3_tug"

g1_29dof_demo3_tug_reward = RewardManagerCfg(
    terms={
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func=f"{_TERMS}:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func=f"{_TERMS}:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func=f"{_TERMS}:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func=f"{_TERMS}:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        "motion_global_body_lin_vel": RewardTermCfg(
            func=f"{_TERMS}:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func=f"{_TERMS}:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func=f"{_TERMS}:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func=f"{_TERMS}:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        # This is the only antisymmetric term.  Its numerical weight remains a
        # formal-training confirmation item; smoke tests only validate signs.
        "signed_table_progress_velocity": RewardTermCfg(
            func=f"{_TERMS}:signed_table_progress_velocity",
            weight=10.0,
        ),
    }
)

__all__ = ["g1_29dof_demo3_tug_reward"]
