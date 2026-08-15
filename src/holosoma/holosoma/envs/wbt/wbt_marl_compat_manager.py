"""Single-robot WBT environment with an explicit ghost-teammate interface."""

from __future__ import annotations

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


class GhostTeammateWholeBodyTrackingManager(WholeBodyTrackingManager):
    """Expose a four-value teammate buffer without changing standard WBT."""

    def _init_buffers(self) -> None:
        super()._init_buffers()
        self.teammate_relative_position_b = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.teammate_relative_velocity_b = torch.zeros_like(self.teammate_relative_position_b)
