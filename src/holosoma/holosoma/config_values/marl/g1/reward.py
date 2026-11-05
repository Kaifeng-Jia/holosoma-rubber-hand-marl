"""Reward preset for the Plan 5 cooperative Push A1 baseline."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


_TERMS = "holosoma.managers.reward.terms.marl"

g1_29dof_plan5_push_reward = RewardManagerCfg(
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
        "undesired_contacts": RewardTermCfg(
            func=f"{_TERMS}:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_rubber_hand_link$)(?!right_rubber_hand_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
        "object_global_ref_position_error_exp": RewardTermCfg(
            func=f"{_TERMS}:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func=f"{_TERMS}:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_plan5_push_smooth_reward = RewardManagerCfg(
    terms={
        **g1_29dof_plan5_push_reward.terms,
        "joint_acceleration_l2": RewardTermCfg(
            func=f"{_TERMS}:JointAccelerationPenalty",
            weight=-5.0e-9,
        ),
    }
)

__all__ = ["g1_29dof_plan5_push_reward", "g1_29dof_plan5_push_smooth_reward"]
