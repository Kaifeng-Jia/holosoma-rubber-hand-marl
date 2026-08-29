"""Per-agent MAPPO update path for competitive Demo 3 tug-of-war."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from holosoma.agents.mappo.demo3_checkpoint import (
    DEMO3_MAPPO_CHECKPOINT_VERSION,
    DEMO3_WARM_START_SHA256,
    demo3_ppo_contract,
    demo3_training_contract,
)
from holosoma.agents.mappo.demo3_initialization import (
    DEMO3_ACTION_DIM,
    DEMO3_ACTOR_OBS_DIM,
    DEMO3_CRITIC_OBS_DIM,
)
from holosoma.agents.mappo.demo3_runner import DEMO3_NUM_AGENTS, Demo3PolicyRunner
from holosoma.agents.mappo.demo3_storage import Demo3RolloutStorage
from holosoma.agents.mappo.initialization import Plan5ModelBundle
from holosoma.config_types.algo import PPOConfig


_DEMO3_TERMINATION_KEYS = ("clear_robot_fall", "reference_horizon")
_DEMO3_LOG_SERIES = {
    "progress_agent_a": "tug/progress_agent_a_m",
    "progress_agent_b": "tug/progress_agent_b_m",
    "table_displacement": "tug/table_displacement_m",
}


def _empty_rollout_diagnostics() -> dict[str, Any]:
    result: dict[str, Any] = {
        "termination_counts": {name: 0 for name in _DEMO3_TERMINATION_KEYS},
        "completed_episodes": 0,
    }
    for prefix in _DEMO3_LOG_SERIES:
        result.update(
            {
                f"{prefix}_sample_count": 0,
                f"{prefix}_sample_mean_m": 0.0,
                f"{prefix}_sample_min_m": 0.0,
                f"{prefix}_sample_max_m": 0.0,
                f"{prefix}_terminal_count": 0,
                f"{prefix}_terminal_mean_m": 0.0,
                f"{prefix}_terminal_min_m": 0.0,
                f"{prefix}_terminal_max_m": 0.0,
            }
        )
    return result


class _ScalarSeriesAccumulator:
    def __init__(self) -> None:
        self.count: torch.Tensor | None = None
        self.total: torch.Tensor | None = None
        self.minimum: torch.Tensor | None = None
        self.maximum: torch.Tensor | None = None
        self.terminal_count: torch.Tensor | None = None
        self.terminal_total: torch.Tensor | None = None
        self.terminal_minimum: torch.Tensor | None = None
        self.terminal_maximum: torch.Tensor | None = None

    @staticmethod
    def _add(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        return value if current is None else current + value

    def observe(self, values: torch.Tensor, done: torch.Tensor) -> None:
        self.count = self._add(
            self.count,
            torch.as_tensor(values.numel(), device=values.device),
        )
        self.total = self._add(self.total, values.sum())
        step_minimum = values.min()
        self.minimum = (
            step_minimum
            if self.minimum is None
            else torch.minimum(self.minimum, step_minimum)
        )
        step_maximum = values.max()
        self.maximum = (
            step_maximum
            if self.maximum is None
            else torch.maximum(self.maximum, step_maximum)
        )

        terminal = values[done]
        if terminal.numel() == 0:
            return
        self.terminal_count = self._add(
            self.terminal_count,
            torch.as_tensor(terminal.numel(), device=values.device),
        )
        self.terminal_total = self._add(self.terminal_total, terminal.sum())
        step_terminal_minimum = terminal.min()
        self.terminal_minimum = (
            step_terminal_minimum
            if self.terminal_minimum is None
            else torch.minimum(self.terminal_minimum, step_terminal_minimum)
        )
        step_terminal_maximum = terminal.max()
        self.terminal_maximum = (
            step_terminal_maximum
            if self.terminal_maximum is None
            else torch.maximum(self.terminal_maximum, step_terminal_maximum)
        )

    @staticmethod
    def _integer(value: torch.Tensor | None) -> int:
        return 0 if value is None else int(value.item())

    @staticmethod
    def _floating(value: torch.Tensor | None) -> float:
        return 0.0 if value is None else float(value.item())

    def finalize(self, prefix: str) -> dict[str, int | float]:
        count = self._integer(self.count)
        terminal_count = self._integer(self.terminal_count)
        return {
            f"{prefix}_sample_count": count,
            f"{prefix}_sample_mean_m": (
                self._floating(self.total) / count if count > 0 else 0.0
            ),
            f"{prefix}_sample_min_m": self._floating(self.minimum),
            f"{prefix}_sample_max_m": self._floating(self.maximum),
            f"{prefix}_terminal_count": terminal_count,
            f"{prefix}_terminal_mean_m": (
                self._floating(self.terminal_total) / terminal_count
                if terminal_count > 0
                else 0.0
            ),
            f"{prefix}_terminal_min_m": self._floating(self.terminal_minimum),
            f"{prefix}_terminal_max_m": self._floating(self.terminal_maximum),
        }


class _RolloutDiagnosticsAccumulator:
    """Read reset-safe competition metrics from each actual environment step."""

    def __init__(self) -> None:
        self.termination_counts: dict[str, torch.Tensor] = {}
        self.completed_episodes: torch.Tensor | None = None
        self.series = {
            prefix: _ScalarSeriesAccumulator() for prefix in _DEMO3_LOG_SERIES
        }

    @staticmethod
    def _vector(
        values: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        label: str,
        expected_numel: int,
    ) -> torch.Tensor:
        vector = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
        if vector.numel() != expected_numel:
            raise ValueError(
                f"Demo 3 {label} must contain one value per environment; "
                f"expected {expected_numel}, got {vector.numel()}"
            )
        return vector.detach()

    @staticmethod
    def _add(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        return value if current is None else current + value

    def observe(self, dones: torch.Tensor, extras: dict[str, Any]) -> None:
        done = torch.as_tensor(dones).reshape(-1).to(dtype=torch.bool)
        device = done.device
        num_envs = done.numel()
        self.completed_episodes = self._add(
            self.completed_episodes,
            done.count_nonzero(),
        )

        termination_terms = extras.get("termination_terms", {})
        if not isinstance(termination_terms, dict):
            raise TypeError("extras['termination_terms'] must be a dictionary")
        for name, values in termination_terms.items():
            mask = self._vector(
                values,
                device=device,
                dtype=torch.bool,
                label=f"termination term {name!r}",
                expected_numel=num_envs,
            )
            self.termination_counts[name] = self._add(
                self.termination_counts.get(name),
                mask.count_nonzero(),
            )

        to_log = extras.get("to_log", {})
        if not isinstance(to_log, dict):
            raise TypeError("extras['to_log'] must be a dictionary")
        for prefix, log_key in _DEMO3_LOG_SERIES.items():
            if log_key not in to_log:
                continue
            values = self._vector(
                to_log[log_key],
                device=device,
                dtype=torch.float32,
                label=log_key,
                expected_numel=num_envs,
            )
            self.series[prefix].observe(values, done)

    @staticmethod
    def _integer(value: torch.Tensor | None) -> int:
        return 0 if value is None else int(value.item())

    def finalize(self) -> dict[str, Any]:
        termination_counts = {
            name: self._integer(self.termination_counts.get(name))
            for name in _DEMO3_TERMINATION_KEYS
        }
        termination_counts.update(
            {
                name: self._integer(value)
                for name, value in self.termination_counts.items()
                if name not in termination_counts
            }
        )
        result: dict[str, Any] = {
            "termination_counts": termination_counts,
            "completed_episodes": self._integer(self.completed_episodes),
        }
        for prefix, accumulator in self.series.items():
            result.update(accumulator.finalize(prefix))
        return result


@dataclass(frozen=True)
class Demo3PPOUpdateMetrics:
    surrogate_loss: float
    value_loss: float
    entropy: float
    kl: float
    actor_grad_norm: float
    critic_grad_norm: float


class Demo3PPO:
    """Train one shared Actor and one shared ego-first Critic.

    Unlike the cooperative Plan 5 learner, every policy sample keeps its own
    reward, value, return, and advantage.  The physical episode termination is
    still shared because both agents inhabit one simulator environment.
    """

    def __init__(
        self,
        models: Plan5ModelBundle,
        config: PPOConfig,
        *,
        num_envs: int,
        num_steps_per_env: int | None = None,
        device: str = "cpu",
        training_contract: dict[str, Any] | None = None,
    ) -> None:
        self.models = models
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.num_steps_per_env = (
            config.num_steps_per_env if num_steps_per_env is None else num_steps_per_env
        )
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.num_steps_per_env <= 0:
            raise ValueError("num_steps_per_env must be positive")
        self.training_contract = dict(
            demo3_training_contract()
            if training_contract is None
            else training_contract
        )
        expected_contract = demo3_training_contract(
            critic_only_iterations=int(
                self.training_contract.get("critic_only_iterations", -1)
            ),
            full_actor_iterations=int(
                self.training_contract.get("full_actor_iterations", -1)
            ),
            checkpoint_interval=int(
                self.training_contract.get("checkpoint_interval", -1)
            ),
        )
        if self.training_contract != expected_contract:
            raise ValueError("Demo 3 training contract is incomplete or incompatible")
        self.ppo_contract = demo3_ppo_contract(
            config,
            num_steps_per_env=self.num_steps_per_env,
        )
        if models.source_sha256 != DEMO3_WARM_START_SHA256:
            raise ValueError(
                "Demo 3 must initialize from the frozen Pull07999 Actor artifact: "
                f"expected {DEMO3_WARM_START_SHA256}, got {models.source_sha256}"
            )
        if int(models.source_iteration) != int(
            self.training_contract["warm_start_iteration"]
        ):
            raise ValueError(
                "Demo 3 warm-start iteration mismatch: "
                f"expected {self.training_contract['warm_start_iteration']}, "
                f"got {models.source_iteration}"
            )

        self.runner = Demo3PolicyRunner(models)
        self.storage = Demo3RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            device=device,
        )
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
        self.last_rollout_diagnostics = _empty_rollout_diagnostics()

    @staticmethod
    def _agent_column(values: torch.Tensor, *, num_envs: int) -> torch.Tensor:
        if values.shape == (num_envs, DEMO3_NUM_AGENTS):
            values = values.unsqueeze(-1)
        expected = (num_envs, DEMO3_NUM_AGENTS, 1)
        if values.shape != expected:
            raise ValueError(
                "Demo 3 rewards must have shape "
                f"({num_envs}, {DEMO3_NUM_AGENTS}) or {expected}, got {tuple(values.shape)}"
            )
        return values

    @staticmethod
    def _shared_column(
        values: torch.Tensor,
        *,
        num_envs: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if values.shape == (num_envs,):
            values = values.unsqueeze(-1)
        expected = (num_envs, 1)
        if values.shape != expected:
            raise ValueError(
                f"Demo 3 shared scalar must have shape ({num_envs},) or {expected}, "
                f"got {tuple(values.shape)}"
            )
        return values.to(dtype=dtype)

    @torch.no_grad()
    def collect_rollout(
        self,
        env: Any,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Collect a synchronized rollout and compute one GAE stream per agent."""
        self.storage.clear()
        diagnostics = _RolloutDiagnosticsAccumulator()
        self.last_rollout_diagnostics = _empty_rollout_diagnostics()
        for _ in range(self.num_steps_per_env):
            decision = self.runner.sample(
                observations,
                update_critic_normalizer=True,
            )
            if (
                decision.action_log_probs is None
                or decision.action_means is None
                or decision.action_sigmas is None
            ):
                raise RuntimeError("Demo 3 stochastic decision is missing PPO statistics")

            next_observations, rewards, dones, extras = env.step(
                {"actions": decision.actions}
            )
            diagnostics.observe(dones, extras)
            rewards = self._agent_column(
                torch.as_tensor(rewards, device=self.device),
                num_envs=self.num_envs,
            ).to(dtype=torch.float)
            dones = self._shared_column(
                torch.as_tensor(dones, device=self.device),
                num_envs=self.num_envs,
                dtype=torch.bool,
            )
            raw_timeouts = extras.get("time_outs")
            if raw_timeouts is None:
                raw_timeouts = torch.zeros(
                    self.num_envs,
                    device=self.device,
                    dtype=torch.bool,
                )
            timeouts = self._shared_column(
                torch.as_tensor(raw_timeouts, device=self.device),
                num_envs=self.num_envs,
                dtype=torch.bool,
            )

            if timeouts.any():
                final_observations = extras.get("final_observations")
                if not isinstance(final_observations, dict):
                    raise RuntimeError("Demo 3 timeout bootstrap requires final_observations")
                final_critic_obs = final_observations.get(self.runner.critic_observation_key)
                if not isinstance(final_critic_obs, torch.Tensor):
                    raise RuntimeError(
                        "Demo 3 timeout bootstrap requires final_observations['critic_obs']"
                    )
                final_values, _ = self.runner.evaluate_critic_observations(
                    final_critic_obs,
                    update_normalizer=False,
                )
                rewards = rewards + (
                    self.config.gamma
                    * final_values
                    * timeouts.unsqueeze(1).to(dtype=final_values.dtype)
                )

            self.storage.add(
                agent={
                    "actor_obs": decision.normalized_actor_observations,
                    "critic_obs": decision.normalized_critic_observations,
                    "actions": decision.actions,
                    "rewards": rewards,
                    "values": decision.values,
                    "actions_log_prob": decision.action_log_probs,
                    "action_mean": decision.action_means,
                    "action_sigma": decision.action_sigmas,
                },
                shared={
                    "dones": dones,
                    "timeouts": timeouts,
                },
            )

            shared_done = dones.squeeze(-1)
            agent_done = shared_done.unsqueeze(1).expand(-1, DEMO3_NUM_AGENTS).reshape(-1)
            self.models.actor.reset(agent_done)
            self.models.critic.reset(agent_done)
            observations = next_observations

        last_critic_obs = observations.get(self.runner.critic_observation_key)
        if not isinstance(last_critic_obs, torch.Tensor):
            raise RuntimeError("Demo 3 final observations are missing critic_obs")
        last_values, _ = self.runner.evaluate_critic_observations(
            last_critic_obs,
            update_normalizer=False,
        )
        returns, advantages = self.compute_returns_and_advantages(
            last_values=last_values,
            values=self.storage.agent("values"),
            dones=self.storage.shared("dones"),
            rewards=self.storage.agent("rewards"),
        )
        self.storage.set_agent("returns", returns)
        self.storage.set_agent("advantages", advantages)
        self.last_rollout_diagnostics = diagnostics.finalize()
        return observations

    def compute_returns_and_advantages(
        self,
        *,
        last_values: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE per agent while broadcasting a joint terminal mask."""
        expected_agent_shape = (
            self.num_steps_per_env,
            self.num_envs,
            DEMO3_NUM_AGENTS,
            1,
        )
        expected_shared_shape = (self.num_steps_per_env, self.num_envs, 1)
        expected_last_shape = (self.num_envs, DEMO3_NUM_AGENTS, 1)
        for name, tensor in (("values", values), ("rewards", rewards)):
            if tensor.shape != expected_agent_shape:
                raise ValueError(
                    f"Demo 3 {name} must have shape {expected_agent_shape}, "
                    f"got {tuple(tensor.shape)}"
                )
        if dones.shape != expected_shared_shape:
            raise ValueError(
                f"Demo 3 dones must have shape {expected_shared_shape}, got {tuple(dones.shape)}"
            )
        if last_values.shape != expected_last_shape:
            raise ValueError(
                f"Demo 3 last_values must have shape {expected_last_shape}, "
                f"got {tuple(last_values.shape)}"
            )

        advantage = torch.zeros_like(last_values)
        returns = torch.zeros_like(values)
        for step in reversed(range(self.num_steps_per_env)):
            next_values = last_values if step == self.num_steps_per_env - 1 else values[step + 1]
            not_terminal = 1.0 - dones[step].float().unsqueeze(1)
            delta = (
                rewards[step]
                + self.config.gamma * not_terminal * next_values
                - values[step]
            )
            advantage = (
                delta
                + self.config.gamma
                * self.config.lam
                * not_terminal
                * advantage
            )
            returns[step] = advantage + values[step]

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1.0e-8
        )
        return returns, advantages

    def update(self, *, update_actor: bool = True) -> Demo3PPOUpdateMetrics:
        """Apply PPO epochs using each Actor row's own advantage."""
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
            metrics = self._update_minibatch(minibatch, update_actor=update_actor)
            for key, value in metrics.items():
                totals[key] += value
            updates += 1
        if updates == 0:
            raise RuntimeError("Demo 3 PPO update produced no mini-batches")
        self.storage.clear()
        return Demo3PPOUpdateMetrics(
            **{key: value / updates for key, value in totals.items()}
        )

    def _update_minibatch(
        self,
        minibatch: dict[str, dict[str, torch.Tensor]],
        *,
        update_actor: bool,
    ) -> dict[str, float]:
        agent = minibatch["agent"]
        if update_actor:
            self.models.actor.act({"actor_obs": agent["actor_obs"]})
            action_log_probs = self.models.actor.get_actions_log_prob(
                agent["actions"]
            ).unsqueeze(-1)
            ratio = torch.exp(action_log_probs - agent["actions_log_prob"])
            # Each row is already one agent sample; no team-advantage repetition.
            surrogate = -agent["advantages"] * ratio
            surrogate_clipped = -agent["advantages"] * torch.clamp(
                ratio,
                1.0 - self.config.clip_param,
                1.0 + self.config.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            entropy = self.models.actor.entropy.mean()
            actor_loss = surrogate_loss - self.config.entropy_coef * entropy

            with torch.no_grad():
                old_dist = Normal(agent["action_mean"], agent["action_sigma"])
                new_dist = Normal(
                    self.models.actor.action_mean,
                    self.models.actor.action_std,
                )
                kl = kl_divergence(old_dist, new_dist).sum(-1).mean()
                if self.config.schedule == "adaptive" and self.config.desired_kl is not None:
                    self._update_actor_learning_rate(kl)

            self.models.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.models.actor.parameters(),
                self.config.max_grad_norm,
            )
            self.models.actor_optimizer.step()
        else:
            self.models.actor_optimizer.zero_grad()
            zero = torch.zeros((), device=agent["critic_obs"].device)
            surrogate_loss = zero
            entropy = zero
            kl = zero
            actor_grad_norm = zero

        values = self.models.critic.evaluate({"critic_obs": agent["critic_obs"]})
        value_clipped = agent["values"] + (values - agent["values"]).clamp(
            -self.config.clip_param,
            self.config.clip_param,
        )
        value_losses = (values - agent["returns"]).pow(2)
        value_losses_clipped = (value_clipped - agent["returns"]).pow(2)
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

    def _update_actor_learning_rate(self, kl: torch.Tensor) -> None:
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

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "version": DEMO3_MAPPO_CHECKPOINT_VERSION,
            "num_agents": DEMO3_NUM_AGENTS,
            "actor_obs_dim": DEMO3_ACTOR_OBS_DIM,
            "actor_obs_groups": ["actor_obs", "teammate_obs", "table_obs"],
            "critic_obs_dim": DEMO3_CRITIC_OBS_DIM,
            "action_dim": DEMO3_ACTION_DIM,
            "critic_layout": "ego_first_per_agent",
            "reward_layout": "per_agent",
            "done_layout": "shared_environment",
            "source_iteration": self.models.source_iteration,
            "source_sha256": self.models.source_sha256,
            "ppo_contract": self.ppo_contract,
            **self.training_contract,
        }

    def training_state_dict(self, *, iteration: int) -> dict[str, Any]:
        """Return a resumable Demo 3-only checkpoint with the full contract."""
        return {
            "demo3_mappo": self._checkpoint_metadata(),
            "actor_model_state_dict": self.models.actor.state_dict(),
            "critic_model_state_dict": self.models.critic.state_dict(),
            "actor_optimizer_state_dict": self.models.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.models.critic_optimizer.state_dict(),
            "actor_obs_normalizer_state_dict": self.models.actor_obs_normalizer.state_dict(),
            "critic_obs_normalizer_state_dict": self.models.critic_obs_normalizer.state_dict(),
            "iter": iteration,
        }

    def load_training_state_dict(
        self,
        state: dict[str, Any],
        *,
        actor_learning_rate: float | None = None,
        critic_learning_rate: float | None = None,
    ) -> int:
        """Restore exactly, then apply only explicitly requested LR overrides."""

        for name, value in {
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
        }.items():
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} override must be finite and positive")
        contaminants = sorted({"plan5_mappo", "demo4_mappo"}.intersection(state))
        if contaminants:
            raise ValueError(
                f"Demo 3 checkpoint contains cross-demo metadata: {contaminants}"
            )
        metadata = state.get("demo3_mappo", {})
        expected = self._checkpoint_metadata()
        if metadata != expected:
            raise ValueError(f"Demo 3 MAPPO checkpoint metadata mismatch: {metadata!r}")
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
        if actor_learning_rate is not None:
            self.actor_learning_rate = float(actor_learning_rate)
            for group in self.models.actor_optimizer.param_groups:
                group["lr"] = self.actor_learning_rate
        if critic_learning_rate is not None:
            self.critic_learning_rate = float(critic_learning_rate)
            for group in self.models.critic_optimizer.param_groups:
                group["lr"] = self.critic_learning_rate
        return int(state["iter"])


__all__ = [
    "DEMO3_MAPPO_CHECKPOINT_VERSION",
    "Demo3PPO",
    "Demo3PPOUpdateMetrics",
]
