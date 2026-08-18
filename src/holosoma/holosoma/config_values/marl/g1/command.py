"""Command preset for the first Plan 5 cooperative Push A1 environment."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig

motion_config = MotionConfig(
    motion_file=(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/"
        "rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz"
    ),
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=False,
)

_term = CommandTermCfg(
    func="holosoma.managers.command.terms.marl:PairedA1MotionCommand",
    params={"motion_config": motion_config, "lateral_spacing_m": 0.8},
)

g1_29dof_paired_a1_command = CommandManagerCfg(
    setup_terms={"paired_motion_command": _term},
    reset_terms={"paired_motion_command": _term},
    step_terms={"paired_motion_command": _term},
)

__all__ = ["g1_29dof_paired_a1_command"]
