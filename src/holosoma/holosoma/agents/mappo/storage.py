"""Rollout storage with separate agent and team tensor contracts."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor


class MultiAgentRolloutStorage:
    """Store local-policy samples without duplicating centralized team state.

    Agent buffers use ``[time, env, agent, ...]``. Team buffers use
    ``[time, env, ...]``. Mini-batches sample complete physical environment
    transitions, then flatten only the selected agent axis for actor updates.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        num_agents: int,
        num_transitions_per_env: int,
        device: str = "cpu",
    ) -> None:
        for name, value in (
            ("num_envs", num_envs),
            ("num_agents", num_agents),
            ("num_transitions_per_env", num_transitions_per_env),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        self.step = 0
        self._agent_buffers: dict[str, Tensor] = {}
        self._team_buffers: dict[str, Tensor] = {}
        self._deferred_team_keys: set[str] = set()
        self._written_deferred_team_keys: set[str] = set()

    def register_agent(
        self,
        key: str,
        shape: tuple[int, ...] | list[int] = (),
        dtype: torch.dtype = torch.float,
    ) -> None:
        """Register a value stored separately for every agent."""
        self._register(key, shape, dtype, is_agent=True, deferred=False)

    def register_team(
        self,
        key: str,
        shape: tuple[int, ...] | list[int] = (),
        dtype: torch.dtype = torch.float,
        *,
        deferred: bool = False,
    ) -> None:
        """Register one shared value per physical environment."""
        self._register(key, shape, dtype, is_agent=False, deferred=deferred)

    def _register(
        self,
        key: str,
        shape: tuple[int, ...] | list[int],
        dtype: torch.dtype,
        *,
        is_agent: bool,
        deferred: bool,
    ) -> None:
        if not isinstance(shape, (tuple, list)):
            raise ValueError("shape must be a tuple or list")
        if key in self._agent_buffers or key in self._team_buffers:
            raise ValueError(f"Key {key!r} is already registered")
        prefix = (
            self.num_transitions_per_env,
            self.num_envs,
            self.num_agents,
        ) if is_agent else (self.num_transitions_per_env, self.num_envs)
        buffers = self._agent_buffers if is_agent else self._team_buffers
        buffers[key] = torch.zeros((*prefix, *shape), dtype=dtype, device=self.device)
        if deferred:
            self._deferred_team_keys.add(key)

    def add(self, *, agent: dict[str, Tensor], team: dict[str, Tensor]) -> None:
        """Append one synchronized multi-agent transition."""
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError(
                f"Buffer overflow: step {self.step} >= {self.num_transitions_per_env}"
            )
        self._validate_keys(agent, self._agent_buffers, set(), "agent")
        self._validate_keys(team, self._team_buffers, self._deferred_team_keys, "team")
        for key, value in agent.items():
            self._copy_step(key, value, self._agent_buffers, "agent")
        for key, value in team.items():
            self._copy_step(key, value, self._team_buffers, "team")
        self.step += 1

    @staticmethod
    def _validate_keys(
        values: dict[str, Tensor],
        buffers: dict[str, Tensor],
        deferred_keys: set[str],
        kind: str,
    ) -> None:
        unknown = sorted(set(values).difference(buffers))
        missing = sorted(set(buffers).difference(deferred_keys).difference(values))
        supplied_deferred = sorted(set(values).intersection(deferred_keys))
        if unknown or missing or supplied_deferred:
            raise ValueError(
                f"{kind} data keys mismatch: missing={missing}, unknown={unknown}, "
                f"deferred_supplied={supplied_deferred}"
            )

    def _copy_step(
        self,
        key: str,
        value: Tensor,
        buffers: dict[str, Tensor],
        kind: str,
    ) -> None:
        expected = buffers[key][self.step].shape
        if value.shape != expected:
            raise ValueError(
                f"{kind} value {key!r} must have shape {tuple(expected)}, got {tuple(value.shape)}"
            )
        if value.requires_grad:
            raise ValueError(f"Cannot store tensor requiring gradients for {kind} key {key!r}")
        buffers[key][self.step].copy_(value)

    def agent(self, key: str) -> Tensor:
        return self._agent_buffers[key]

    def team(self, key: str) -> Tensor:
        return self._team_buffers[key]

    def set_team(self, key: str, value: Tensor) -> None:
        """Write a full deferred team buffer after the rollout is complete."""
        if key not in self._deferred_team_keys:
            raise ValueError(f"Team key {key!r} is not registered as deferred")
        expected = self._team_buffers[key].shape
        if value.shape != expected:
            raise ValueError(
                f"Deferred team value {key!r} must have shape {tuple(expected)}, "
                f"got {tuple(value.shape)}"
            )
        if value.requires_grad:
            raise ValueError(f"Cannot store tensor requiring gradients for deferred team key {key!r}")
        self._team_buffers[key].copy_(value)
        self._written_deferred_team_keys.add(key)

    def clear(self) -> None:
        """Reset the write cursor while retaining allocated tensors."""
        self.step = 0
        self._written_deferred_team_keys.clear()

    def mini_batch_generator(
        self,
        *,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Iterator[dict[str, dict[str, Tensor]]]:
        """Yield paired actor/team mini-batches.

        Returned agent tensors have shape ``[batch_env_steps * num_agents, ...]``;
        returned team tensors have shape ``[batch_env_steps, ...]``.
        """
        if self.step != self.num_transitions_per_env:
            raise RuntimeError(
                "Cannot generate mini-batches before rollout is full: "
                f"{self.step}/{self.num_transitions_per_env} transitions"
            )
        unwritten = sorted(self._deferred_team_keys.difference(self._written_deferred_team_keys))
        if unwritten:
            raise RuntimeError(f"Deferred team buffers were not written: {unwritten}")
        batch_size = self.num_transitions_per_env * self.num_envs
        if num_mini_batches <= 0 or num_mini_batches > batch_size:
            raise ValueError(f"num_mini_batches must be in [1, {batch_size}]")
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        mini_batch_size = batch_size // num_mini_batches
        usable_size = mini_batch_size * num_mini_batches
        flat_agent = {
            key: value.flatten(0, 1) for key, value in self._agent_buffers.items()
        }
        flat_team = {
            key: value.flatten(0, 1) for key, value in self._team_buffers.items()
        }

        for _ in range(num_epochs):
            permutation = torch.randperm(batch_size, device=self.device)[:usable_size]
            for batch_index in range(num_mini_batches):
                start = batch_index * mini_batch_size
                end = start + mini_batch_size
                indices = permutation[start:end]
                yield {
                    "agent": {
                        key: value[indices].flatten(0, 1)
                        for key, value in flat_agent.items()
                    },
                    "team": {key: value[indices] for key, value in flat_team.items()},
                }
