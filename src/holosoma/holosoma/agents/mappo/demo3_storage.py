"""Rollout storage for Demo 3 per-agent rewards and ego-first values."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor

from holosoma.agents.mappo.demo3_initialization import (
    DEMO3_ACTION_DIM,
    DEMO3_ACTOR_OBS_DIM,
    DEMO3_CRITIC_OBS_DIM,
)
from holosoma.agents.mappo.demo3_runner import DEMO3_NUM_AGENTS


class Demo3RolloutStorage:
    """Keep all learning targets on the agent axis.

    Agent buffers use ``[time, env, agent, ...]``.  Only episode termination
    state is shared and stored as ``[time, env, ...]``.  Mini-batches sample
    complete physical environment transitions before folding the selected agent
    axis into the learning batch, preserving Actor/Critic/advantage alignment.
    """

    _DEFERRED_AGENT_KEYS = frozenset({"returns", "advantages"})

    def __init__(
        self,
        *,
        num_envs: int,
        num_transitions_per_env: int,
        device: str = "cpu",
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if num_transitions_per_env <= 0:
            raise ValueError("num_transitions_per_env must be positive")
        self.num_envs = num_envs
        self.num_agents = DEMO3_NUM_AGENTS
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        self.step = 0
        self._written_deferred_agent_keys: set[str] = set()

        agent_prefix = (num_transitions_per_env, num_envs, self.num_agents)
        shared_prefix = (num_transitions_per_env, num_envs)
        self._agent_buffers = {
            "actor_obs": torch.zeros(
                (*agent_prefix, DEMO3_ACTOR_OBS_DIM), device=device, dtype=torch.float
            ),
            "critic_obs": torch.zeros(
                (*agent_prefix, DEMO3_CRITIC_OBS_DIM), device=device, dtype=torch.float
            ),
            "actions": torch.zeros(
                (*agent_prefix, DEMO3_ACTION_DIM), device=device, dtype=torch.float
            ),
            "rewards": torch.zeros((*agent_prefix, 1), device=device, dtype=torch.float),
            "values": torch.zeros((*agent_prefix, 1), device=device, dtype=torch.float),
            "returns": torch.zeros((*agent_prefix, 1), device=device, dtype=torch.float),
            "advantages": torch.zeros((*agent_prefix, 1), device=device, dtype=torch.float),
            "actions_log_prob": torch.zeros(
                (*agent_prefix, 1), device=device, dtype=torch.float
            ),
            "action_mean": torch.zeros(
                (*agent_prefix, DEMO3_ACTION_DIM), device=device, dtype=torch.float
            ),
            "action_sigma": torch.zeros(
                (*agent_prefix, DEMO3_ACTION_DIM), device=device, dtype=torch.float
            ),
        }
        self._shared_buffers = {
            "dones": torch.zeros((*shared_prefix, 1), device=device, dtype=torch.bool),
            "timeouts": torch.zeros((*shared_prefix, 1), device=device, dtype=torch.bool),
        }

    def add(self, *, agent: dict[str, Tensor], shared: dict[str, Tensor]) -> None:
        """Append one synchronized competitive transition."""
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError(
                f"Buffer overflow: step {self.step} >= {self.num_transitions_per_env}"
            )
        expected_agent_keys = set(self._agent_buffers).difference(self._DEFERRED_AGENT_KEYS)
        self._validate_keys(agent, expected_agent_keys, "agent")
        self._validate_keys(shared, set(self._shared_buffers), "shared")
        for key, value in agent.items():
            self._copy_step(key, value, self._agent_buffers, "agent")
        for key, value in shared.items():
            self._copy_step(key, value, self._shared_buffers, "shared")
        self.step += 1

    @staticmethod
    def _validate_keys(values: dict[str, Tensor], expected: set[str], kind: str) -> None:
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"Demo 3 {kind} data keys mismatch: "
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
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
                f"Demo 3 {kind} value {key!r} must have shape {tuple(expected)}, "
                f"got {tuple(value.shape)}"
            )
        if value.requires_grad:
            raise ValueError(
                f"Cannot store a tensor requiring gradients for Demo 3 {kind} key {key!r}"
            )
        buffers[key][self.step].copy_(value)

    def set_agent(self, key: str, value: Tensor) -> None:
        """Write a deferred full-rollout per-agent target."""
        if key not in self._DEFERRED_AGENT_KEYS:
            raise ValueError(f"Demo 3 agent key {key!r} is not deferred")
        expected = self._agent_buffers[key].shape
        if value.shape != expected:
            raise ValueError(
                f"Deferred Demo 3 agent value {key!r} must have shape {tuple(expected)}, "
                f"got {tuple(value.shape)}"
            )
        if value.requires_grad:
            raise ValueError(f"Cannot store a deferred tensor requiring gradients for {key!r}")
        self._agent_buffers[key].copy_(value)
        self._written_deferred_agent_keys.add(key)

    def agent(self, key: str) -> Tensor:
        return self._agent_buffers[key]

    def shared(self, key: str) -> Tensor:
        return self._shared_buffers[key]

    def clear(self) -> None:
        self.step = 0
        self._written_deferred_agent_keys.clear()

    def mini_batch_generator(
        self,
        *,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Iterator[dict[str, dict[str, Tensor]]]:
        """Yield aligned flattened agent samples and shared done state."""
        if self.step != self.num_transitions_per_env:
            raise RuntimeError(
                "Cannot generate Demo 3 mini-batches before rollout is full: "
                f"{self.step}/{self.num_transitions_per_env} transitions"
            )
        unwritten = self._DEFERRED_AGENT_KEYS.difference(self._written_deferred_agent_keys)
        if unwritten:
            raise RuntimeError(
                f"Deferred Demo 3 agent buffers were not written: {sorted(unwritten)}"
            )
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
        flat_shared = {
            key: value.flatten(0, 1) for key, value in self._shared_buffers.items()
        }
        for _ in range(num_epochs):
            permutation = torch.randperm(batch_size, device=self.device)[:usable_size]
            for batch_index in range(num_mini_batches):
                start = batch_index * mini_batch_size
                indices = permutation[start : start + mini_batch_size]
                yield {
                    "agent": {
                        key: value[indices].flatten(0, 1)
                        for key, value in flat_agent.items()
                    },
                    "shared": {key: value[indices] for key, value in flat_shared.items()},
                }


__all__ = ["Demo3RolloutStorage"]
