"""System1 T13 v4 D-Table-PA: the CORE4D bucket-reward environment with per-robot rewards (rew_buf [E, A])."""

from __future__ import annotations

import torch

from holosoma.envs.marl.core4d_bucket_manager import Core4DBucketManager
from holosoma.managers.reward.per_agent_team_manager import PerAgentTeamRewardManager


class Core4DBucketPerAgentManager(Core4DBucketManager):
    def __init__(self, tyro_config, *, device):
        self._per_agent_reward_cfg = tyro_config.reward
        super().__init__(tyro_config, device=device)
        self.reward_manager = PerAgentTeamRewardManager(self._per_agent_reward_cfg, self, self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self.rew_buf = torch.zeros(self.num_envs, self.num_agents, dtype=torch.float, device=self.device)


__all__ = ["Core4DBucketPerAgentManager"]
