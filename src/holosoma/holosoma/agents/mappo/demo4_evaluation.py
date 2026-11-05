"""Pure helpers for deterministic Demo 4 checkpoint evaluation.

The runtime entry point lives in :mod:`scripts.evaluate_demo4_rotate`.  This
module deliberately contains no Isaac Sim imports so checkpoint and recording
contracts can be tested on CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

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
    DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
    DEMO4_SOURCE_PULL_ITERATION,
)


DEMO4_TERMINATION_NAMES = (
    "yaw_goal_success",
    "clear_robot_fall",
    "table_physical_safety",
    "reference_horizon",
)
DEMO4_ACTOR_GROUPS = (
    ("actor_obs", 154),
    ("teammate_obs", 4),
    ("table_obs", 6),
)
VISER_FIVE_CHANNEL_SHAPES = {
    "root_pos": (DEMO4_NUM_AGENTS, 3),
    "root_quat_xyzw": (DEMO4_NUM_AGENTS, 4),
    "dof_pos": (DEMO4_NUM_AGENTS, DEMO4_ACTION_DIM),
    "object_pos_w": (3,),
    "object_quat_xyzw": (4,),
}


def expected_demo4_metadata() -> dict[str, Any]:
    """Return the exact metadata accepted by the isolated evaluator."""

    return {
        "version": DEMO4_MAPPO_CHECKPOINT_VERSION,
        "scenario": "cooperative_rectangular_table_rotate_90deg",
        "return_model": "one_team_reward_value_return_advantage_per_environment",
        "num_agents": DEMO4_NUM_AGENTS,
        "actor_obs_dim": DEMO4_ACTOR_OBS_DIM,
        "actor_obs_groups": [name for name, _ in DEMO4_ACTOR_GROUPS],
        "critic_obs_dim": DEMO4_CRITIC_OBS_DIM,
        "action_dim_per_agent": DEMO4_ACTION_DIM,
        "checkpoint_interval": DEMO4_CHECKPOINT_INTERVAL,
        "source_iteration": DEMO4_SOURCE_PULL_ITERATION,
        "source_sha256": DEMO4_SOURCE_PULL_CHECKPOINT_SHA256,
        **demo4_static_training_contract(),
    }


def validate_demo4_evaluation_checkpoint(state: Mapping[str, Any]) -> int:
    """Fail closed unless ``state`` is an uncontaminated Demo 4 checkpoint."""

    metadata = state.get("demo4_mappo")
    if metadata != expected_demo4_metadata():
        raise ValueError(
            "Demo 4 evaluation checkpoint metadata mismatch: "
            f"{metadata!r}"
        )
    contaminants = sorted({"plan5_mappo", "demo3_mappo"}.intersection(state))
    if contaminants:
        raise ValueError(
            f"Demo 4 evaluation checkpoint contains cross-demo metadata: {contaminants}"
        )
    required = {
        "actor_model_state_dict",
        "actor_obs_normalizer_state_dict",
        "iter",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Demo 4 evaluation checkpoint is incomplete: {missing}")
    iteration = state["iter"]
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"Demo 4 checkpoint iteration must be non-negative, got {iteration!r}")
    return iteration


@torch.no_grad()
def deterministic_demo4_actions(
    models: Any,
    observations: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run only the shared Actor, once for each robot, without the Critic."""

    first = observations.get(DEMO4_ACTOR_GROUPS[0][0])
    if not isinstance(first, torch.Tensor) or first.ndim != 3:
        raise ValueError("Demo 4 actor_obs must have shape [num_envs, 2, 154]")
    num_envs = first.shape[0]
    parts = []
    for name, width in DEMO4_ACTOR_GROUPS:
        value = observations.get(name)
        expected = (num_envs, DEMO4_NUM_AGENTS, width)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            shape = getattr(value, "shape", None)
            raise ValueError(f"{name} must have shape {expected}, got {shape}")
        parts.append(value)
    combined = torch.cat(parts, dim=-1)
    flat = combined.reshape(num_envs * DEMO4_NUM_AGENTS, DEMO4_ACTOR_OBS_DIM)
    normalized = models.actor_obs_normalizer(flat, update=False)
    flat_actions = models.actor.act_inference({"actor_obs": normalized})
    expected_actions = (num_envs * DEMO4_NUM_AGENTS, DEMO4_ACTION_DIM)
    if tuple(flat_actions.shape) != expected_actions:
        raise ValueError(
            f"Demo 4 Actor actions must have shape {expected_actions}, "
            f"got {tuple(flat_actions.shape)}"
        )
    if not torch.isfinite(flat_actions).all():
        raise ValueError("Demo 4 Actor produced non-finite deterministic actions")
    return flat_actions.reshape(num_envs, DEMO4_NUM_AGENTS, DEMO4_ACTION_DIM)


def validate_viser_five_channels(channels: Mapping[str, np.ndarray]) -> int:
    """Validate the exact five-channel contract consumed by ViSER."""

    missing = sorted(set(VISER_FIVE_CHANNEL_SHAPES).difference(channels))
    if missing:
        raise ValueError(f"ViSER recording is missing channels: {missing}")
    frame_count = int(np.asarray(channels["root_pos"]).shape[0])
    if frame_count < 1:
        raise ValueError("ViSER recording must contain at least one frame")
    for name, trailing_shape in VISER_FIVE_CHANNEL_SHAPES.items():
        values = np.asarray(channels[name])
        expected = (frame_count, *trailing_shape)
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("root_quat_xyzw", "object_quat_xyzw"):
        norms = np.linalg.norm(np.asarray(channels[name]), axis=-1)
        if not np.allclose(norms, 1.0, atol=1.0e-3, rtol=0.0):
            raise ValueError(f"{name} contains non-unit quaternions")
    return frame_count


def save_viser_episode(
    path: str | Path,
    channels: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Save one representative episode in the canonical ViSER layout."""

    validate_viser_five_channels(channels)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: np.asarray(channels[name]) for name in VISER_FIVE_CHANNEL_SHAPES
    }
    payload["_metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    np.savez_compressed(output, **payload)
    return output


@dataclass(frozen=True)
class Demo4EpisodeResult:
    episode: int
    steps: int
    success: bool
    fall: bool
    table_safety: bool
    timeout: bool
    final_unwrapped_yaw_rad: float
    max_unwrapped_yaw_rad: float
    reward_sum: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "steps": self.steps,
            "success": self.success,
            "fall": self.fall,
            "table_safety": self.table_safety,
            "timeout": self.timeout,
            "final_unwrapped_yaw_rad": self.final_unwrapped_yaw_rad,
            "final_unwrapped_yaw_deg": float(np.degrees(self.final_unwrapped_yaw_rad)),
            "max_unwrapped_yaw_rad": self.max_unwrapped_yaw_rad,
            "max_unwrapped_yaw_deg": float(np.degrees(self.max_unwrapped_yaw_rad)),
            "reward_sum": self.reward_sum,
        }


def representative_episode_index(results: list[Demo4EpisodeResult]) -> int:
    """Prefer success, then the greatest physical yaw progress."""

    if not results:
        raise ValueError("At least one Demo 4 episode result is required")
    return max(
        range(len(results)),
        key=lambda index: (
            results[index].success,
            results[index].max_unwrapped_yaw_rad,
            results[index].final_unwrapped_yaw_rad,
            -results[index].episode,
        ),
    )


__all__ = [
    "DEMO4_ACTOR_GROUPS",
    "DEMO4_TERMINATION_NAMES",
    "Demo4EpisodeResult",
    "deterministic_demo4_actions",
    "expected_demo4_metadata",
    "representative_episode_index",
    "save_viser_episode",
    "validate_demo4_evaluation_checkpoint",
    "validate_viser_five_channels",
]
