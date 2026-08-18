"""Single-robot WBT environment with a stable teammate-observation interface."""

from __future__ import annotations

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


class GhostTeammateWholeBodyTrackingManager(WholeBodyTrackingManager):
    """Reserve four teammate channels without changing the frozen WBT policy.

    The compatibility environment intentionally initializes the relative
    planar position and velocity to zero. Plan 5 will replace these buffers
    with bounded randomized values during single-agent compatibility training
    and real teammate/opponent measurements in the multi-agent environment,
    while preserving the same 158-dimensional actor schema.
    """

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.teammate_relative_position_b = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.teammate_relative_velocity_b = torch.zeros_like(
            self.teammate_relative_position_b
        )
