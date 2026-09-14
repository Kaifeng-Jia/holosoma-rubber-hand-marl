"""Original tracking reward and an explicit additive interaction variant."""

import copy
import math
from dataclasses import replace

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

CORE4D_INTERACTION_WEIGHT = 1.0
CORE4D_INTERACTION_SIGMA = 0.06


def validate_object_z_error_weight(value: float) -> float:
    """Reject invalid experiment coefficients before constructing a simulator."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0.0:
        raise ValueError("object_z_error_weight must be positive and finite")
    return float(value)


def with_object_z_error_weight(base_reward: RewardManagerCfg, value: float) -> RewardManagerCfg:
    """Copy the preset and change only the coefficient of squared world-z error."""
    value = validate_object_z_error_weight(value)
    reward = copy.deepcopy(base_reward)
    name = "object_global_ref_position_error_exp"
    term = reward.terms[name]
    if term.func != f"{_TERMS}:{name}" or term.params.get("sigma") != 0.3 or term.weight != 1.0:
        raise ValueError("Vertical tracking requires the unchanged sigma=0.3, weight=1 position term")
    params = dict(term.params)
    params.pop("object_z_error_weight", None)
    if value != 1.0:
        params["object_z_error_weight"] = value
    reward.terms[name] = replace(term, params=params)
    return reward


def with_interaction_reward_term(base_reward: RewardManagerCfg, reference_file: str) -> RewardManagerCfg:
    """Copy a reward preset without mutating any of its existing terms."""
    if not str(reference_file).strip():
        raise ValueError("An explicit precomputed interaction reference is required")
    if "interaction_mesh" in base_reward.terms:
        raise ValueError("The interaction reward is already enabled")
    return replace(
        base_reward,
        terms={
            **base_reward.terms,
            "interaction_mesh": RewardTermCfg(
                func="holosoma.managers.reward.terms.interaction_mesh:InteractionMeshReward",
                params={"reference_file": str(reference_file), "sigma": CORE4D_INTERACTION_SIGMA},
                weight=CORE4D_INTERACTION_WEIGHT,
            ),
        },
    )


__all__ = [
    "g1_29dof_core4d_smalltable_reward",
    "CORE4D_INTERACTION_WEIGHT",
    "CORE4D_INTERACTION_SIGMA",
    "with_interaction_reward_term",
    "validate_object_z_error_weight",
    "with_object_z_error_weight",
]
