"""System1 T13 v4 D-Table-PA: per-robot reward aggregation for the cooperative pair (switch --per-agent-reward).

Terms return [E, A] (per-robot tracking / contact / penalty terms) or [E] for the explicitly shared object terms, which
count for both robots (T13 v4 section 2 item 5: own tracking and contact terms per robot, object terms shared). The
compute order, weights and dt scaling are those of RewardManager."""

from __future__ import annotations

import torch

from holosoma.managers.reward.manager import RewardManager

SHARED_TERMS = frozenset({"bucket_interaction", "object_global_ref_position_error_exp", "object_global_ref_orientation_error_exp",
                          "interaction_mesh", "object_height_error_penalty"})


class PerAgentTeamRewardManager(RewardManager):
    def __init__(self, cfg, env, device: str):
        super().__init__(cfg, env, device)
        shape = (env.num_envs, env.num_agents)
        self._reward_buf = torch.zeros(shape, dtype=torch.float, device=device)
        self._episode_sums = {name: torch.zeros(shape, dtype=torch.float, device=device) for name in self._term_names}
        self._episode_sums_raw = {name: torch.zeros(shape, dtype=torch.float, device=device) for name in self._term_names}

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf.zero_()
        E, A = self.env.num_envs, self.env.num_agents
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_name in self._term_instances:
                raw = self._term_instances[term_name](self.env, **term_cfg.params)
            else:
                raw = self._term_funcs[term_name](self.env, **term_cfg.params)
            if term_name in SHARED_TERMS:
                if raw.shape != (E,):
                    raise ValueError(f"shared reward term {term_name!r} must return [E], got {tuple(raw.shape)}")
                raw = raw[:, None].expand(E, A)
            elif raw.shape != (E, A):
                raise ValueError(f"per-agent reward term {term_name!r} must return [E, A], got {tuple(raw.shape)}")
            scaled = raw * term_cfg.weight * dt
            self._reward_buf.add_(scaled)
            self._episode_sums[term_name].add_(scaled)
            self._episode_sums_raw[term_name].add_(raw)
        return self._reward_buf


__all__ = ["PerAgentTeamRewardManager", "SHARED_TERMS"]
