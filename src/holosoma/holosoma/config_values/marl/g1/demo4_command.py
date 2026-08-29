"""Command preset for cooperative rectangular-table rotation."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg
from holosoma.config_values.marl.g1.command import pull_motion_config


DEMO4_ROTATE_RUNTIME_REFERENCE_FILE = (
    "holosoma/data/motions/g1_29dof/whole_body_tracking/demo4_rotate/"
    "demo4_pull_pull_static_table_runtime.npz"
)

demo4_pull_motion_config = replace(
    pull_motion_config,
    start_at_timestep_zero_prob=1.0,
)

_demo4_rotate_term = CommandTermCfg(
    func=(
        "holosoma.managers.command.terms.demo4_rotate:"
        "Demo4RotateMotionCommand"
    ),
    params={
        "motion_config": demo4_pull_motion_config,
        "paired_reference_file": DEMO4_ROTATE_RUNTIME_REFERENCE_FILE,
        "goal_yaw_degrees": 90.0,
        "maximum_success_tilt_degrees": 60.0,
    },
)

g1_29dof_demo4_rotate_command = CommandManagerCfg(
    setup_terms={"paired_motion_command": _demo4_rotate_term},
    reset_terms={"paired_motion_command": _demo4_rotate_term},
    step_terms={"paired_motion_command": _demo4_rotate_term},
)

__all__ = [
    "DEMO4_ROTATE_RUNTIME_REFERENCE_FILE",
    "demo4_pull_motion_config",
    "g1_29dof_demo4_rotate_command",
]
