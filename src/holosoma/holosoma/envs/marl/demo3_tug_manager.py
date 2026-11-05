"""Isolated two-agent competitive tug-of-war environment."""

from __future__ import annotations

import torch

from holosoma.envs.marl.plan5_push_manager import Plan5PushManager
from holosoma.managers.reward.demo3_manager import Demo3AgentRewardManager


class Demo3TugManager(Plan5PushManager):
    """Two physical G1s, per-agent rewards, and one shared square table."""

    def __init__(self, tyro_config, *, device):
        self._demo3_reward_cfg = tyro_config.reward
        super().__init__(tyro_config, device=device)
        # BaseTask constructs the generic [E] manager.  Replace it only inside
        # this Demo 3 subclass with the explicit [E, A] contract.
        self.reward_manager = Demo3AgentRewardManager(
            self._demo3_reward_cfg, self, self.device
        )

    def _init_buffers(self):
        super()._init_buffers()
        self.rew_buf = torch.zeros(
            self.num_envs,
            self.num_agents,
            dtype=torch.float,
            device=self.device,
        )

    def _update_log_dict(self):
        command = self.command_manager.get_state("paired_motion_command")
        if command is not None:
            command.update_metrics()
            self.log_dict.update(command.metrics)


__all__ = ["Demo3TugManager"]
