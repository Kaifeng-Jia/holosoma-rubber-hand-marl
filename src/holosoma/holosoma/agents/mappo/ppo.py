"""Minimal parameter-sharing MAPPO learning loop for Plan 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from holosoma.agents.mappo.batch_layout import HomogeneousAgentBatchLayout
from holosoma.agents.mappo.initialization import Plan5ModelBundle
from holosoma.agents.mappo.runner import Plan5PolicyRunner
from holosoma.agents.mappo.storage import MultiAgentRolloutStorage
from holosoma.config_types.algo import PPOConfig


PLAN5_MAPPO_CHECKPOINT_VERSION = "plan5_shared_actor_mappo_v1"
PLAN5_ACTOR_FIRST_WEIGHT = "actor_module.module.0.weight"
PLAN5_LEGACY_ACTOR_OBS_DIM = 154


@dataclass(frozen=True)
class Plan5PPOUpdateMetrics:
    surrogate_loss: float
    value_loss: float
    entropy: float
    kl: float
    actor_grad_norm: float
    critic_grad_norm: float


class Plan5PPO:
    """Collect team rollouts and apply the original WBT PPO objective.

    The actor receives one sample per agent. The critic, reward, done, return,
    and advantage remain one sample per physical environment. During the actor
    update, each team's advantage is repeated once for each homogeneous agent.
    """

    def __init__(
        self,
        models: Plan5ModelBundle,
        config: PPOConfig,
        *,
        num_envs: int,
        num_steps_per_env: int | None = None,
        layout: HomogeneousAgentBatchLayout | None = None,
        device: str = "cpu",
    ) -> None:
        self.models = models
        self.config = config
        self.layout = layout or HomogeneousAgentBatchLayout()
        self.device = device
        self.num_steps_per_env = num_steps_per_env or config.num_steps_per_env
        self.runner = Plan5PolicyRunner(models, layout=self.layout)
        self.actor_learning_rate = config.actor_learning_rate
        self.critic_learning_rate = config.critic_learning_rate
        self.min_actor_learning_rate = config.min_actor_learning_rate or min(
            self.actor_learning_rate,
            1.0e-5,
        )
        self.max_actor_learning_rate = config.max_actor_learning_rate or max(
            self.actor_learning_rate,
            1.0e-2,
        )
        self.min_critic_learning_rate = config.min_critic_learning_rate or min(
            self.critic_learning_rate,
            1.0e-5,
        )
        self.max_critic_learning_rate = config.max_critic_learning_rate or max(
            self.critic_learning_rate,
            1.0e-2,
        )
        self.storage = MultiAgentRolloutStorage(
            num_envs=num_envs,
            num_agents=self.layout.num_agents,
            num_transitions_per_env=self.num_steps_per_env,
            device=device,
        )
        self._register_storage()

    def _register_storage(self) -> None:
        self.storage.register_agent("actor_obs", (self.layout.actor_obs_dim,))
        self.storage.register_agent("actions", (self.layout.action_dim,))
        self.storage.register_agent("actions_log_prob", (1,))
        self.storage.register_agent("action_mean", (self.layout.action_dim,))
        self.storage.register_agent("action_sigma", (self.layout.action_dim,))
        self.storage.register_team("critic_obs", (527,))
        self.storage.register_team("rewards", (1,))
        self.storage.register_team("dones", (1,), dtype=torch.bool)
        self.storage.register_team("timeouts", (1,), dtype=torch.bool)
        self.storage.register_team("values", (1,))
        self.storage.register_team("returns", (1,), deferred=True)
        self.storage.register_team("advantages", (1,), deferred=True)

    @staticmethod
    def _column(values: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(f"Team scalar must have shape [num_envs] or [num_envs, 1], got {values.shape}")
        return values.to(dtype=dtype) if dtype is not None else values

    @torch.no_grad()
    def collect_rollout(
        self,
        env: Any,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Collect one synchronized team rollout and compute team GAE."""
        self.storage.clear()
        for _ in range(self.num_steps_per_env):
            decision = self.runner.sample(observations, update_critic_normalizer=True)
            if (
                decision.action_log_probs is None
                or decision.action_means is None
                or decision.action_sigmas is None
            ):
                raise RuntimeError("Stochastic decision is missing PPO distribution statistics")

            next_observations, rewards, dones, extras = env.step({"actions": decision.actions})
            rewards = self._column(rewards).to(self.device)
            dones = self._column(dones, dtype=torch.bool).to(self.device)
            time_outs = extras.get("time_outs")
            timeout_mask = self._column(
                torch.as_tensor(
                    time_outs if time_outs is not None else torch.zeros_like(dones),
                    device=self.device,
                ),
                dtype=torch.bool,
            )
            if timeout_mask.any():
                final_observations = extras.get("final_observations")
                if not isinstance(final_observations, dict) or "critic_obs" not in final_observations:
                    raise RuntimeError("Timeout bootstrap requires final_observations['critic_obs']")
                final_critic_obs = self.models.critic_obs_normalizer(
                    final_observations["critic_obs"],
                    update=False,
                )
                final_values = self.models.critic.evaluate({"critic_obs": final_critic_obs})
                rewards = rewards + self.config.gamma * final_values * timeout_mask

            self.storage.add(
                agent={
                    "actor_obs": decision.normalized_actor_observations,
                    "actions": decision.actions,
                    "actions_log_prob": decision.action_log_probs,
                    "action_mean": decision.action_means,
                    "action_sigma": decision.action_sigmas,
                },
                team={
                    "critic_obs": decision.normalized_critic_observations,
                    "rewards": rewards,
                    "dones": dones,
                    "timeouts": timeout_mask,
                    "values": decision.values,
                },
            )
            self.models.actor.reset(dones.squeeze(-1))
            self.models.critic.reset(dones.squeeze(-1))
            observations = next_observations

        last_critic_obs = self.models.critic_obs_normalizer(
            observations["critic_obs"],
            update=False,
        )
        last_values = self.models.critic.evaluate({"critic_obs": last_critic_obs})
        returns, advantages = self.compute_team_returns_and_advantages(
            last_values=last_values,
            values=self.storage.team("values"),
            dones=self.storage.team("dones"),
            rewards=self.storage.team("rewards"),
        )
        self.storage.set_team("returns", returns)
        self.storage.set_team("advantages", advantages)
        return observations

    def compute_team_returns_and_advantages(
        self,
        *,
        last_values: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one GAE stream per physical environment."""
        if values.shape != rewards.shape or values.shape != dones.shape:
            raise ValueError("Team values, rewards, and dones must have identical shapes")
        if last_values.shape != values.shape[1:]:
            raise ValueError(
                f"last_values must have shape {tuple(values.shape[1:])}, got {tuple(last_values.shape)}"
            )
        advantage = torch.zeros_like(last_values)
        returns = torch.zeros_like(values)
        for step in reversed(range(values.shape[0])):
            next_values = last_values if step == values.shape[0] - 1 else values[step + 1]
            not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + self.config.gamma * not_terminal * next_values - values[step]
            advantage = delta + self.config.gamma * self.config.lam * not_terminal * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        return returns, advantages

    def update(
        self,
        *,
        update_actor: bool = True,
        teammate_input_only: bool = False,
    ) -> Plan5PPOUpdateMetrics:
        """Apply PPO epochs to one full synchronized rollout."""
        if teammate_input_only and not update_actor:
            raise ValueError("teammate_input_only requires update_actor=True")
        totals = {
            "surrogate_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
        }
        updates = 0
        for minibatch in self.storage.mini_batch_generator(
            num_mini_batches=self.config.num_mini_batches,
            num_epochs=self.config.num_learning_epochs,
        ):
            metrics = self._update_minibatch(
                minibatch,
                update_actor=update_actor,
                teammate_input_only=teammate_input_only,
            )
            for key, value in metrics.items():
                totals[key] += value
            updates += 1
        if updates == 0:
            raise RuntimeError("PPO update produced no mini-batches")
        self.storage.clear()
        return Plan5PPOUpdateMetrics(**{key: value / updates for key, value in totals.items()})

    def _update_minibatch(
        self,
        minibatch: dict[str, dict[str, torch.Tensor]],
        *,
        update_actor: bool,
        teammate_input_only: bool,
    ) -> dict[str, float]:
        agent = minibatch["agent"]
        team = minibatch["team"]
        if update_actor:
            frozen_actor_parameters = (
                self._snapshot_frozen_actor_parameters() if teammate_input_only else None
            )
            team_advantages = team["advantages"]
            actor_advantages = team_advantages.repeat_interleave(self.layout.num_agents, dim=0)

            self.models.actor.act({"actor_obs": agent["actor_obs"]})
            action_log_probs = self.models.actor.get_actions_log_prob(agent["actions"]).unsqueeze(-1)
            ratio = torch.exp(action_log_probs - agent["actions_log_prob"])
            surrogate = -actor_advantages * ratio
            surrogate_clipped = -actor_advantages * torch.clamp(
                ratio,
                1.0 - self.config.clip_param,
                1.0 + self.config.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            entropy = self.models.actor.entropy.mean()
            actor_loss = surrogate_loss - self.config.entropy_coef * entropy

            with torch.no_grad():
                old_dist = Normal(agent["action_mean"], agent["action_sigma"])
                new_dist = Normal(self.models.actor.action_mean, self.models.actor.action_std)
                kl = kl_divergence(old_dist, new_dist).sum(-1).mean()
                if self.config.schedule == "adaptive" and self.config.desired_kl is not None:
                    self._update_actor_learning_rate(kl)

            self.models.actor_optimizer.zero_grad()
            actor_loss.backward()
            if teammate_input_only:
                self._mask_actor_gradients_to_teammate_inputs()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.models.actor.parameters(),
                self.config.max_grad_norm,
            )
            self.models.actor_optimizer.step()
            if frozen_actor_parameters is not None:
                self._restore_frozen_actor_parameters(frozen_actor_parameters)
        else:
            self.models.actor_optimizer.zero_grad()
            zero = torch.zeros((), device=team["critic_obs"].device)
            surrogate_loss = zero
            entropy = zero
            kl = zero
            actor_grad_norm = zero

        values = self.models.critic.evaluate({"critic_obs": team["critic_obs"]})
        value_clipped = team["values"] + (values - team["values"]).clamp(
            -self.config.clip_param,
            self.config.clip_param,
        )
        value_losses = (values - team["returns"]).pow(2)
        value_losses_clipped = (value_clipped - team["returns"]).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
        critic_loss = self.config.value_loss_coef * value_loss

        self.models.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.models.critic.parameters(),
            self.config.max_grad_norm,
        )
        self.models.critic_optimizer.step()

        return {
            "surrogate_loss": surrogate_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
        }

    def _snapshot_frozen_actor_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self.models.actor.named_parameters()
        }

    def _mask_actor_gradients_to_teammate_inputs(self) -> None:
        first_weight_found = False
        for name, parameter in self.models.actor.named_parameters():
            if name == PLAN5_ACTOR_FIRST_WEIGHT:
                if parameter.shape[1] != self.layout.actor_obs_dim:
                    raise RuntimeError(
                        f"Unexpected Plan 5 actor first-layer shape: {tuple(parameter.shape)}"
                    )
                if parameter.grad is None:
                    raise RuntimeError("Plan 5 actor first-layer gradient is missing")
                parameter.grad[:, :PLAN5_LEGACY_ACTOR_OBS_DIM] = 0.0
                first_weight_found = True
            else:
                parameter.grad = None
        if not first_weight_found:
            raise RuntimeError(f"Missing Plan 5 actor parameter {PLAN5_ACTOR_FIRST_WEIGHT!r}")

    @torch.no_grad()
    def _restore_frozen_actor_parameters(
        self,
        snapshot: dict[str, torch.Tensor],
    ) -> None:
        for name, parameter in self.models.actor.named_parameters():
            if name == PLAN5_ACTOR_FIRST_WEIGHT:
                parameter[:, :PLAN5_LEGACY_ACTOR_OBS_DIM].copy_(
                    snapshot[name][:, :PLAN5_LEGACY_ACTOR_OBS_DIM]
                )
            else:
                parameter.copy_(snapshot[name])

    def _update_actor_learning_rate(self, kl: torch.Tensor) -> None:
        """Adapt only the policy optimizer to policy KL.

        The centralized critic has an independent optimizer and objective. Its
        learning rate must not be throttled by divergence between old and new
        actor distributions.
        """
        desired_kl = self.config.desired_kl
        if desired_kl is None:
            return
        if kl > desired_kl * 2.0:
            self.actor_learning_rate = max(
                self.min_actor_learning_rate,
                self.actor_learning_rate / 1.5,
            )
        elif 0.0 < kl < desired_kl / 2.0:
            self.actor_learning_rate = min(
                self.max_actor_learning_rate,
                self.actor_learning_rate * 1.5,
            )
        for group in self.models.actor_optimizer.param_groups:
            group["lr"] = self.actor_learning_rate

    def training_state_dict(self, *, iteration: int) -> dict[str, Any]:
        """Return a resumable MAPPO checkpoint with an explicit compatibility contract."""
        return {
            "plan5_mappo": {
                "version": PLAN5_MAPPO_CHECKPOINT_VERSION,
                "num_agents": self.layout.num_agents,
                "actor_obs_dim": self.layout.actor_obs_dim,
                "critic_obs_dim": 527,
                "action_dim": self.layout.action_dim,
                "source_sha256": self.models.source_sha256,
            },
            "actor_model_state_dict": self.models.actor.state_dict(),
            "critic_model_state_dict": self.models.critic.state_dict(),
            "actor_optimizer_state_dict": self.models.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.models.critic_optimizer.state_dict(),
            "actor_obs_normalizer_state_dict": self.models.actor_obs_normalizer.state_dict(),
            "critic_obs_normalizer_state_dict": self.models.critic_obs_normalizer.state_dict(),
            "iter": iteration,
        }

    def _validate_training_state(self, state: dict[str, Any]) -> None:
        metadata = state.get("plan5_mappo", {})
        expected = {
            "version": PLAN5_MAPPO_CHECKPOINT_VERSION,
            "num_agents": self.layout.num_agents,
            "actor_obs_dim": self.layout.actor_obs_dim,
            "critic_obs_dim": 527,
            "action_dim": self.layout.action_dim,
            "source_sha256": self.models.source_sha256,
        }
        if metadata != expected:
            raise ValueError(f"Plan 5 MAPPO checkpoint metadata mismatch: {metadata!r}")

    def load_training_state_dict(self, state: dict[str, Any]) -> int:
        """Strictly restore a checkpoint created by :meth:`training_state_dict`."""
        self._validate_training_state(state)
        self.models.actor.load_state_dict(state["actor_model_state_dict"], strict=True)
        self.models.critic.load_state_dict(state["critic_model_state_dict"], strict=True)
        self.models.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        self.models.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        self.models.actor_obs_normalizer.load_state_dict(
            state["actor_obs_normalizer_state_dict"],
            strict=True,
        )
        self.models.critic_obs_normalizer.load_state_dict(
            state["critic_obs_normalizer_state_dict"],
            strict=True,
        )
        self.actor_learning_rate = self.models.actor_optimizer.param_groups[0]["lr"]
        self.critic_learning_rate = self.models.critic_optimizer.param_groups[0]["lr"]
        return int(state["iter"])

    def load_fine_tune_state_dict(self, state: dict[str, Any]) -> int:
        """Restore learned models and normalizers while keeping fresh optimizers.

        Fine-tuning changes the reward objective.  Reusing the old Adam moments
        would carry update momentum from the pre-smoothing objective, so this
        path intentionally leaves both newly constructed optimizers untouched.
        """
        self._validate_training_state(state)
        self.models.actor.load_state_dict(state["actor_model_state_dict"], strict=True)
        self.models.critic.load_state_dict(state["critic_model_state_dict"], strict=True)
        self.models.actor_obs_normalizer.load_state_dict(
            state["actor_obs_normalizer_state_dict"],
            strict=True,
        )
        self.models.critic_obs_normalizer.load_state_dict(
            state["critic_obs_normalizer_state_dict"],
            strict=True,
        )
        return int(state["iter"])
