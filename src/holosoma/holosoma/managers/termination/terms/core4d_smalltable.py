"""Termination terms specific to the CORE4D small-table clip."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.command.terms.core4d_smalltable import (
    Core4DSmallTableMotionCommand,
)


def reference_horizon_reached(env: Any) -> torch.Tensor:
    """End the episode at the final frame instead of looping the clip."""
    command = env.command_manager.get_state("paired_motion_command")
    if not isinstance(command, Core4DSmallTableMotionCommand):
        raise TypeError(
            "CORE4D small-table horizon requires Core4DSmallTableMotionCommand, "
            f"got {type(command)}"
        )
    return command.time_steps >= command.reference.num_frames - 1


__all__ = ["reference_horizon_reached"]
