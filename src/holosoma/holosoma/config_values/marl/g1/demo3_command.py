"""Command preset for the isolated square-table Demo 3 tug task."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg
from holosoma.config_values.marl.g1.command import pull_motion_config


DEMO3_TUG_RUNTIME_REFERENCE_FILE = (
    "holosoma/data/motions/g1_29dof/whole_body_tracking/demo3_tug/"
    "sub3_010_diagonal_tug_runtime.npz"
)

demo3_pull_motion_config = replace(
    pull_motion_config,
    start_at_timestep_zero_prob=1.0,
)

_demo3_tug_term = CommandTermCfg(
    func="holosoma.managers.command.terms.demo3_tug:Demo3TugMotionCommand",
    params={
        "motion_config": demo3_pull_motion_config,
        "paired_reference_file": DEMO3_TUG_RUNTIME_REFERENCE_FILE,
    },
)

g1_29dof_demo3_tug_command = CommandManagerCfg(
    setup_terms={"paired_motion_command": _demo3_tug_term},
    reset_terms={"paired_motion_command": _demo3_tug_term},
    step_terms={"paired_motion_command": _demo3_tug_term},
)

__all__ = [
    "DEMO3_TUG_RUNTIME_REFERENCE_FILE",
    "demo3_pull_motion_config",
    "g1_29dof_demo3_tug_command",
]
