"""T13 v4 section 2 item 5 (D-Table-PA, switch --per-agent-reward; default off = the team implementation, unchanged).

Each robot's own tracking and contact terms count only for it, the object terms for both (PerAgentTeamRewardManager); the
centralised critic (same 585-D input) outputs one value per robot; GAE runs per robot with the shared done / timeout flags;
the actor rows (env-major, robots adjacent, as the team code's repeat_interleave layout) take their own robot's advantage.
Everything else (PPO objective, clipping, KL schedule, normalisers, minibatching) is the team implementation (the
demo3 per_agent_reward_value_return_advantage scheme on the cooperative CORE4D code path)."""
from __future__ import annotations

import copy
from dataclasses import replace

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from holosoma.agents.mappo.core4d_smalltable_ppo import Core4DSmallTablePPO
from holosoma.agents.mappo.core4d_smalltable_runner import Core4DSmallTablePolicyRunner

PER_AGENT_KEYS = ("rewards", "values", "returns", "advantages")
NUM_AGENTS = 2


class Core4DPerAgentRunner(Core4DSmallTablePolicyRunner):
    def _team_values(self, normalized_critic: torch.Tensor) -> torch.Tensor:
        values = self.models.critic.evaluate({"critic_obs": normalized_critic})
        expected = (normalized_critic.shape[0], NUM_AGENTS)
        if values.shape != expected:
            raise ValueError(f"Per-agent critic must return {expected}, got {tuple(values.shape)}")
        return values


class Core4DPerAgentPPO(Core4DSmallTablePPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runner = Core4DPerAgentRunner(self.models)
        self._checkpoint_metadata["system1_t13v4_return_model"] = "per_agent_reward_value_return_advantage"

    def _register_storage(self) -> None:
        orig = self.storage.register_team

        def reg(name, shape, *a, **k):
            if name in PER_AGENT_KEYS:
                shape = (NUM_AGENTS,)
            return orig(name, shape, *a, **k)
        self.storage.register_team = reg
        try:
            super()._register_storage()
        finally:
            self.storage.register_team = orig

    @staticmethod
    def _column(values: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 2 or values.shape[1] not in (1, NUM_AGENTS):
            raise ValueError(f"Per-agent scalars must have shape [E], [E, 1] or [E, {NUM_AGENTS}], got {tuple(values.shape)}")
        return values.to(dtype=dtype) if dtype is not None else values

    def compute_team_returns_and_advantages(self, *, last_values, values, dones, rewards):
        if values.shape != rewards.shape or dones.shape != values.shape[:-1] + (1,):
            raise ValueError("Per-agent values / rewards [T, E, A] and dones [T, E, 1] expected")
        advantage = torch.zeros_like(last_values)
        returns = torch.zeros_like(values)
        for step in reversed(range(values.shape[0])):
            next_values = last_values if step == values.shape[0] - 1 else values[step + 1]
            not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + self.config.gamma * not_terminal * next_values - values[step]
            advantage = delta + self.config.gamma * self.config.lam * not_terminal * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        # demo3 per_agent_reward_value_return_advantage: one normalisation over both robots, population std
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
        return returns, advantages

    def _update_minibatch(self, minibatch, *, update_actor: bool, teammate_input_only: bool):
        if teammate_input_only:
            raise ValueError("per-agent PPO: teammate_input_only is not supported")
        agent, team = minibatch["agent"], minibatch["team"]
        if update_actor:
            actor_advantages = team["advantages"].reshape(-1, 1)   # [B, A] -> [B * A, 1], env-major like the actor rows
            self.models.actor.act({"actor_obs": agent["actor_obs"]})
            action_log_probs = self.models.actor.get_actions_log_prob(agent["actions"]).unsqueeze(-1)
            ratio = torch.exp(action_log_probs - agent["actions_log_prob"])
            surrogate = -actor_advantages * ratio
            surrogate_clipped = -actor_advantages * torch.clamp(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            entropy = self.models.actor.entropy.mean()
            actor_loss = surrogate_loss - self.config.entropy_coef * entropy
            with torch.no_grad():
                kl = kl_divergence(Normal(agent["action_mean"], agent["action_sigma"]),
                                   Normal(self.models.actor.action_mean, self.models.actor.action_std)).sum(-1).mean()
                if self.config.schedule == "adaptive" and self.config.desired_kl is not None:
                    self._update_actor_learning_rate(kl)
            self.models.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.models.actor.parameters(), self.config.max_grad_norm)
            self.models.actor_optimizer.step()
        else:
            self.models.actor_optimizer.zero_grad()
            zero = torch.zeros((), device=team["critic_obs"].device)
            surrogate_loss = entropy = kl = actor_grad_norm = zero
        values = self.models.critic.evaluate({"critic_obs": team["critic_obs"]})
        value_clipped = team["values"] + (values - team["values"]).clamp(-self.config.clip_param, self.config.clip_param)
        value_loss = torch.max((values - team["returns"]).pow(2), (value_clipped - team["returns"]).pow(2)).mean()
        self.models.critic_optimizer.zero_grad()
        (self.config.value_loss_coef * value_loss).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.models.critic.parameters(), self.config.max_grad_norm)
        self.models.critic_optimizer.step()
        return {"surrogate_loss": surrogate_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item(), "kl": kl.item(),
                "actor_grad_norm": float(actor_grad_norm), "critic_grad_norm": float(critic_grad_norm)}


def per_agent_ppo_config(ppo_config):
    """The critic outputs one value per robot."""
    module_dict = copy.deepcopy(ppo_config.module_dict)
    module_dict = replace(module_dict, critic=replace(module_dict.critic, output_dim=[NUM_AGENTS]))
    return replace(ppo_config, module_dict=module_dict)


def with_per_agent_config(config):
    """Environment / reward-term swaps of the per-agent mode (CORE4D bucket path)."""
    from holosoma.config_types.reward import RewardTermCfg  # noqa: F401
    from holosoma.managers.reward.terms.marl_per_agent import PER_AGENT
    if config.env_class != "holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager":
        raise ValueError(f"per-agent reward: only the CORE4D bucket environment is supported, got {config.env_class}")
    terms = {}
    for name, t in config.reward.terms.items():
        mod, fn = t.func.split(":")
        if mod == "holosoma.managers.reward.terms.marl" and fn in PER_AGENT:
            t = replace(t, func=f"holosoma.managers.reward.terms.marl_per_agent:{PER_AGENT[fn]}")
        elif name == "contact_v1":
            t = replace(t, params={**t.params, "per_agent": True})
        terms[name] = t
    return replace(config, env_class="holosoma.envs.marl.core4d_bucket_per_agent_manager.Core4DBucketPerAgentManager",
                   reward=replace(config.reward, terms=terms))


__all__ = ["Core4DPerAgentPPO", "Core4DPerAgentRunner", "per_agent_ppo_config", "with_per_agent_config"]
