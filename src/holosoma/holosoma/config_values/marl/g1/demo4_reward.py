"""Cooperative team reward preset for Demo 4 table rotation."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


_WBT_TERMS = "holosoma.managers.reward.terms.marl"
_ROTATE_TERMS = "holosoma.managers.reward.terms.demo4_rotate"

g1_29dof_demo4_rotate_reward = RewardManagerCfg(
    terms={
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func=f"{_WBT_TERMS}:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func=f"{_WBT_TERMS}:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func=f"{_WBT_TERMS}:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func=f"{_WBT_TERMS}:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        "motion_global_body_lin_vel": RewardTermCfg(
            func=f"{_WBT_TERMS}:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func=f"{_WBT_TERMS}:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func=f"{_WBT_TERMS}:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func=f"{_WBT_TERMS}:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "signed_yaw_potential_delta": RewardTermCfg(
            func=f"{_ROTATE_TERMS}:SignedYawPotentialDelta",
            weight=10.0,
        ),
        "first_yaw_goal_bonus": RewardTermCfg(
            func=f"{_ROTATE_TERMS}:FirstYawGoalBonus",
            weight=5.0,
        ),
    }
)

__all__ = ["g1_29dof_demo4_rotate_reward"]
