"""Command preset for the isolated cooperative Kick experiment."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig
from holosoma.config_values.marl.g1.command import motion_config


PLAN5_KICK_RUNTIME_REFERENCE_FILE = (
    "holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "rubber_hand_largetable_v1/kick/"
    "plan5_attempt09_dual_kick_mirrored_runtime.npz"
)

kick_motion_config = MotionConfig(
    motion_file=(
        "holosoma/data/motions/g1_29dof/whole_body_tracking/"
        "rubber_hand_largetable_v1/kick/"
        "sub16_largetable_028_kick_mj_fps50_w_obj.npz"
    ),
    body_names_to_track=list(motion_config.body_names_to_track),
    body_name_ref=list(motion_config.body_name_ref),
    use_adaptive_timesteps_sampler=False,
)

_kick_term = CommandTermCfg(
    func="holosoma.managers.command.terms.marl:PairedA1MotionCommand",
    params={
        "motion_config": kick_motion_config,
        "paired_reference_file": PLAN5_KICK_RUNTIME_REFERENCE_FILE,
    },
)

g1_29dof_paired_kick_command = CommandManagerCfg(
    setup_terms={"paired_motion_command": _kick_term},
    reset_terms={"paired_motion_command": _kick_term},
    step_terms={"paired_motion_command": _kick_term},
)

__all__ = [
    "PLAN5_KICK_RUNTIME_REFERENCE_FILE",
    "g1_29dof_paired_kick_command",
    "kick_motion_config",
]
