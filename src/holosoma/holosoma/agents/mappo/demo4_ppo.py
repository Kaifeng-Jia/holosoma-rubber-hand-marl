"""Cooperative team-reward MAPPO for Demo 4 table rotation."""

from __future__ import annotations

import math
from typing import Any

import torch

from holosoma.agents.mappo.batch_layout import HomogeneousAgentBatchLayout
from holosoma.agents.mappo.demo4_checkpoint import (
    DEMO4_CHECKPOINT_INTERVAL,
    DEMO4_MAPPO_CHECKPOINT_VERSION,
    demo4_static_training_contract,
)
from holosoma.agents.mappo.demo4_initialization import (
    DEMO4_ACTION_DIM,
    DEMO4_ACTOR_OBS_DIM,
    DEMO4_CRITIC_OBS_DIM,
    DEMO4_NUM_AGENTS,
)
from holosoma.agents.mappo.demo4_runner import Demo4PolicyRunner
from holosoma.agents.mappo.initialization import Plan5ModelBundle
from holosoma.agents.mappo.ppo import Plan5PPO, Plan5PPOUpdateMetrics
from holosoma.config_types.algo import PPOConfig


Demo4PPOUpdateMetrics = Plan5PPOUpdateMetrics


_DEMO4_TERMINATION_KEYS = (
    "yaw_goal_success",
    "clear_robot_fall",
    "table_physical_safety",
    "reference_horizon",
)
_DEMO4_YAW_LOG_KEY = "rotate/yaw_progress_rad"


def _empty_rollout_diagnostics() -> dict[str, Any]:
    return {
        "termination_counts": {name: 0 for name in _DEMO4_TERMINATION_KEYS},
        "completed_episodes": 0,
        "success_count": 0,
        "success_rate": 0.0,
        "yaw_sample_count": 0,
        "yaw_sample_mean_rad": 0.0,
        "yaw_sample_min_rad": 0.0,
        "yaw_sample_max_rad": 0.0,
        "terminal_yaw_sample_count": 0,
        "terminal_yaw_mean_rad": 0.0,
        "terminal_yaw_min_rad": 0.0,
        "terminal_yaw_max_rad": 0.0,
    }


class _RolloutDiagnosticsAccumulator:
    """Accumulate reset-safe diagnostics directly from real step extras."""

    def __init__(self) -> None:
        self.termination_counts: dict[str, torch.Tensor] = {}
        self.completed_episodes: torch.Tensor | None = None
        self.yaw_sample_count: torch.Tensor | None = None
        self.yaw_sample_sum: torch.Tensor | None = None
        self.yaw_sample_min: torch.Tensor | None = None
        self.yaw_sample_max: torch.Tensor | None = None
        self.terminal_yaw_sample_count: torch.Tensor | None = None
        self.terminal_yaw_sum: torch.Tensor | None = None
        self.terminal_yaw_min: torch.Tensor | None = None
        self.terminal_yaw_max: torch.Tensor | None = None

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
                f"Demo 4 {label} must contain one value per environment; "
                f"expected {expected_numel}, got {vector.numel()}"
            )
        return vector.detach()

    @staticmethod
    def _add(
        current: torch.Tensor | None,
        increment: torch.Tensor,
    ) -> torch.Tensor:
        return increment if current is None else current + increment

    def observe(
        self,
        dones: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        done_vector = torch.as_tensor(dones).reshape(-1).to(dtype=torch.bool)
        device = done_vector.device
        num_envs = done_vector.numel()
        self.completed_episodes = self._add(
            self.completed_episodes,
            done_vector.count_nonzero(),
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
        if _DEMO4_YAW_LOG_KEY not in to_log:
            return
        yaw = self._vector(
            to_log[_DEMO4_YAW_LOG_KEY],
            device=device,
            dtype=torch.float32,
            label=_DEMO4_YAW_LOG_KEY,
            expected_numel=num_envs,
        )
        self.yaw_sample_count = self._add(
            self.yaw_sample_count,
            torch.as_tensor(yaw.numel(), device=device),
        )
        self.yaw_sample_sum = self._add(self.yaw_sample_sum, yaw.sum())
        step_min = yaw.min()
        self.yaw_sample_min = (
            step_min
            if self.yaw_sample_min is None
            else torch.minimum(self.yaw_sample_min, step_min)
        )
        step_max = yaw.max()
        self.yaw_sample_max = (
            step_max
            if self.yaw_sample_max is None
            else torch.maximum(self.yaw_sample_max, step_max)
        )

        terminal_yaw = yaw[done_vector]
        if terminal_yaw.numel() == 0:
            return
        self.terminal_yaw_sample_count = self._add(
            self.terminal_yaw_sample_count,
            torch.as_tensor(terminal_yaw.numel(), device=device),
        )
        self.terminal_yaw_sum = self._add(
            self.terminal_yaw_sum,
            terminal_yaw.sum(),
        )
        step_terminal_min = terminal_yaw.min()
        self.terminal_yaw_min = (
            step_terminal_min
            if self.terminal_yaw_min is None
            else torch.minimum(self.terminal_yaw_min, step_terminal_min)
        )
        step_terminal_max = terminal_yaw.max()
        self.terminal_yaw_max = (
            step_terminal_max
            if self.terminal_yaw_max is None
            else torch.maximum(self.terminal_yaw_max, step_terminal_max)
        )

    @staticmethod
    def _integer(value: torch.Tensor | None) -> int:
        return 0 if value is None else int(value.item())

    @staticmethod
    def _floating(value: torch.Tensor | None) -> float:
        return 0.0 if value is None else float(value.item())

    def finalize(self) -> dict[str, Any]:
        termination_counts = {
            name: self._integer(self.termination_counts.get(name))
            for name in _DEMO4_TERMINATION_KEYS
        }
        termination_counts.update(
            {
                name: self._integer(value)
                for name, value in self.termination_counts.items()
                if name not in termination_counts
            }
        )
        completed_episodes = self._integer(self.completed_episodes)
        success_count = termination_counts["yaw_goal_success"]
        yaw_sample_count = self._integer(self.yaw_sample_count)
        terminal_yaw_sample_count = self._integer(
            self.terminal_yaw_sample_count
        )
        return {
            "termination_counts": termination_counts,
            "completed_episodes": completed_episodes,
            "success_count": success_count,
            "success_rate": (
                success_count / completed_episodes
                if completed_episodes > 0
                else 0.0
            ),
            "yaw_sample_count": yaw_sample_count,
            "yaw_sample_mean_rad": (
                self._floating(self.yaw_sample_sum) / yaw_sample_count
                if yaw_sample_count > 0
                else 0.0
            ),
            "yaw_sample_min_rad": self._floating(self.yaw_sample_min),
            "yaw_sample_max_rad": self._floating(self.yaw_sample_max),
            "terminal_yaw_sample_count": terminal_yaw_sample_count,
            "terminal_yaw_mean_rad": (
                self._floating(self.terminal_yaw_sum) / terminal_yaw_sample_count
                if terminal_yaw_sample_count > 0
                else 0.0
            ),
            "terminal_yaw_min_rad": self._floating(self.terminal_yaw_min),
            "terminal_yaw_max_rad": self._floating(self.terminal_yaw_max),
        }


class _StepDiagnosticsProxy:
    """Delegate the environment while observing each real step result."""

    def __init__(self, env: Any, accumulator: _RolloutDiagnosticsAccumulator) -> None:
        self._env = env
        self._accumulator = accumulator

    def step(self, actor_state: dict[str, torch.Tensor]):
        result = self._env.step(actor_state)
        _, _, dones, extras = result
        self._accumulator.observe(dones, extras)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class Demo4PPO(Plan5PPO):
    """Thin cooperative wrapper around Plan 5's team storage and team GAE.

    This class intentionally does not inherit the competitive Demo 3 update.
    Each environment supplies one team reward, one team value, and one GAE
    stream.  That team advantage is repeated for both rows of the shared Actor.
    """

    def __init__(
        self,
        models: Plan5ModelBundle,
        config: PPOConfig,
        *,
        num_envs: int,
        num_steps_per_env: int | None = None,
        device: str = "cpu",
    ) -> None:
        layout = HomogeneousAgentBatchLayout(
            num_agents=DEMO4_NUM_AGENTS,
            actor_obs_dim=DEMO4_ACTOR_OBS_DIM,
            action_dim=DEMO4_ACTION_DIM,
        )
        super().__init__(
            models,
            config,
            num_envs=num_envs,
            num_steps_per_env=num_steps_per_env,
            layout=layout,
            device=device,
        )
        self.runner = Demo4PolicyRunner(models)
        self.last_rollout_diagnostics = _empty_rollout_diagnostics()

    @torch.no_grad()
    def collect_rollout(
        self,
        env: Any,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Collect via Plan 5 PPO and retain diagnostics from every real step."""

        accumulator = _RolloutDiagnosticsAccumulator()
        self.last_rollout_diagnostics = _empty_rollout_diagnostics()
        observations = super().collect_rollout(
            _StepDiagnosticsProxy(env, accumulator),
            observations,
        )
        self.last_rollout_diagnostics = accumulator.finalize()
        return observations

    def update(self, *, update_actor: bool = True) -> Demo4PPOUpdateMetrics:
        """Apply the cooperative update; Demo 4 has no column-only mode."""

        return super().update(
            update_actor=update_actor,
            teammate_input_only=False,
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "version": DEMO4_MAPPO_CHECKPOINT_VERSION,
            "scenario": "cooperative_rectangular_table_rotate_90deg",
            "return_model": "one_team_reward_value_return_advantage_per_environment",
            "num_agents": DEMO4_NUM_AGENTS,
            "actor_obs_dim": DEMO4_ACTOR_OBS_DIM,
            "actor_obs_groups": ["actor_obs", "teammate_obs", "table_obs"],
            "critic_obs_dim": DEMO4_CRITIC_OBS_DIM,
            "action_dim_per_agent": DEMO4_ACTION_DIM,
            "checkpoint_interval": DEMO4_CHECKPOINT_INTERVAL,
            "source_iteration": self.models.source_iteration,
            "source_sha256": self.models.source_sha256,
            **demo4_static_training_contract(),
        }

    def training_state_dict(self, *, iteration: int) -> dict[str, Any]:
        """Return a resumable checkpoint carrying only Demo 4 metadata."""

        return {
            "demo4_mappo": self._metadata(),
            "actor_model_state_dict": self.models.actor.state_dict(),
            "critic_model_state_dict": self.models.critic.state_dict(),
            "actor_optimizer_state_dict": self.models.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.models.critic_optimizer.state_dict(),
            "actor_obs_normalizer_state_dict": (
                self.models.actor_obs_normalizer.state_dict()
            ),
            "critic_obs_normalizer_state_dict": (
                self.models.critic_obs_normalizer.state_dict()
            ),
            "iter": int(iteration),
        }

    def _validate_training_state(self, state: dict[str, Any]) -> None:
        metadata = state.get("demo4_mappo")
        if metadata != self._metadata():
            raise ValueError(
                "Demo 4 cooperative MAPPO checkpoint metadata mismatch: "
                f"{metadata!r}"
            )
        if "plan5_mappo" in state or "demo3_mappo" in state:
            raise ValueError("Demo 4 checkpoint contains cross-demo metadata")

    def load_training_state_dict(
        self,
        state: dict[str, Any],
        *,
        actor_learning_rate: float | None = None,
        critic_learning_rate: float | None = None,
    ) -> int:
        """Restore training state and apply explicit post-restore LR overrides.

        Optimizer state dictionaries contain their own learning rates.  An
        override therefore has to be applied after the optimizer restore; if
        no override is requested, the exact checkpoint rate remains active.
        """

        overrides = {
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
        }
        for name, value in overrides.items():
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} override must be finite and positive")

        iteration = super().load_training_state_dict(state)
        if actor_learning_rate is not None:
            self.actor_learning_rate = float(actor_learning_rate)
            for group in self.models.actor_optimizer.param_groups:
                group["lr"] = self.actor_learning_rate
        if critic_learning_rate is not None:
            self.critic_learning_rate = float(critic_learning_rate)
            for group in self.models.critic_optimizer.param_groups:
                group["lr"] = self.critic_learning_rate
        return iteration


__all__ = ["Demo4PPO", "Demo4PPOUpdateMetrics"]
