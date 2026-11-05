"""Per-agent reward aggregation for isolated competitive Demo 3."""

from __future__ import annotations

import torch

from holosoma.managers.reward.manager import RewardManager


class Demo3AgentRewardManager(RewardManager):
    """Aggregate reward terms shaped ``[num_envs, num_agents]``."""

    def __init__(self, cfg, env, device: str):
        if cfg.only_positive_rewards:
            raise ValueError("Competitive Demo 3 rewards must not be clipped to positive")
        super().__init__(cfg, env, device)
        shape = (env.num_envs, env.num_agents)
        self._reward_buf = torch.zeros(shape, dtype=torch.float, device=device)
        self._episode_sums = {
            name: torch.zeros(shape, dtype=torch.float, device=device)
            for name in self._term_names
        }
        self._episode_sums_raw = {
            name: torch.zeros(shape, dtype=torch.float, device=device)
            for name in self._term_names
        }

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf.zero_()
        expected = (self.env.num_envs, self.env.num_agents)
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_name in self._term_instances:
                raw = self._term_instances[term_name](self.env, **term_cfg.params)
            else:
                raw = self._term_funcs[term_name](self.env, **term_cfg.params)
            if raw.shape != expected:
                raise ValueError(
                    f"Demo 3 reward term {term_name!r} must return {expected}, "
                    f"got {tuple(raw.shape)}"
                )
            scaled = raw * term_cfg.weight * dt
            self._reward_buf.add_(scaled)
            self._episode_sums[term_name].add_(scaled)
            self._episode_sums_raw[term_name].add_(raw)
        return self._reward_buf


__all__ = ["Demo3AgentRewardManager"]
