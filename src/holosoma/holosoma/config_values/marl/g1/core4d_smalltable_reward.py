"""Frozen Plan-5-style tracking reward for the CORE4D small-table demo."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


_TERMS = "holosoma.managers.reward.terms.marl"

g1_29dof_core4d_smalltable_reward = RewardManagerCfg(
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

__all__ = ["g1_29dof_core4d_smalltable_reward"]
