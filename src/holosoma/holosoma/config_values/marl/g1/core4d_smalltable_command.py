"""Command preset for the CORE4D paired small-table demo."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig


CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE = (
    "holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/"
    "core4d_pair_runtime_fps50.npz"
)
CORE4D_SMALLTABLE_REFERENCE_FPS = 50
CORE4D_SMALLTABLE_REFERENCE_FRAMES = 687

core4d_smalltable_motion_config = MotionConfig(
    motion_file=CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE,
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

_core4d_smalltable_term = CommandTermCfg(
    func=(
        "holosoma.managers.command.terms.core4d_smalltable:"
        "Core4DSmallTableMotionCommand"
    ),
    params={
        "motion_config": core4d_smalltable_motion_config,
        "paired_reference_file": CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE,
    },
)

g1_29dof_core4d_smalltable_command = CommandManagerCfg(
    setup_terms={"paired_motion_command": _core4d_smalltable_term},
    reset_terms={"paired_motion_command": _core4d_smalltable_term},
    step_terms={"paired_motion_command": _core4d_smalltable_term},
)

__all__ = [
    "CORE4D_SMALLTABLE_REFERENCE_FPS",
    "CORE4D_SMALLTABLE_REFERENCE_FRAMES",
    "CORE4D_SMALLTABLE_RUNTIME_REFERENCE_FILE",
    "core4d_smalltable_motion_config",
    "g1_29dof_core4d_smalltable_command",
]
