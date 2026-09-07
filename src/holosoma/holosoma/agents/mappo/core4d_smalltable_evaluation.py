"""Pure Actor-only evaluation helpers for the CORE4D small-table demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from holosoma.agents.mappo.core4d_smalltable_initialization import (
    CORE4D_SMALLTABLE_ACTION_DIM,
    CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
    CORE4D_SMALLTABLE_NUM_AGENTS,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (
    CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION,
    validate_core4d_smalltable_checkpoint,
)
from holosoma.agents.mappo.core4d_smalltable_runner import (
    CORE4D_SMALLTABLE_ACTOR_GROUPS,
)


def core4d_smalltable_evaluation_reward_metadata(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Label the fixed baseline evaluation score separately from the training reward.

    Evaluation runs the same Actor, physics, observations and termination regardless
    of the training reward. It deliberately does not instantiate a training-only graph
    reward or need its precomputed asset merely to replay a checkpoint.
    """
    validate_core4d_smalltable_checkpoint(state)
    training_contract = state["core4d_smalltable_mappo"]["reward_contract"]
    is_interaction = (
        training_contract == CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION
    )
    metadata = {
        "training_reward_variant": "interaction_mesh" if is_interaction else "baseline",
        "training_reward_contract": training_contract,
        "evaluation_reward_contract": CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION,
        "evaluation_reward_matches_training": not is_interaction,
        "evaluation_reward_note": (
            "reward_sum uses the unchanged baseline reward and excludes interaction_mesh; "
            "shared Actor-only inference, physics and termination are unchanged"
        ),
    }
    if is_interaction:
        metadata["interaction_mesh_training_contract"] = dict(
            state["core4d_smalltable_mappo"]["interaction_mesh"]
        )
    return metadata


@torch.no_grad()
def deterministic_core4d_smalltable_actions(
    models: Any,
    observations: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run the shared Actor twice without reading or evaluating Critic state."""
    first = observations.get("actor_obs")
    if not isinstance(first, torch.Tensor) or first.ndim != 3:
        raise ValueError("actor_obs must have shape [num_envs, 2, 154]")
    num_envs = first.shape[0]
    parts = []
    for name, width in CORE4D_SMALLTABLE_ACTOR_GROUPS:
        value = observations.get(name)
        expected = (num_envs, CORE4D_SMALLTABLE_NUM_AGENTS, width)
        if not isinstance(value, torch.Tensor) or value.shape != expected:
            shape = getattr(value, "shape", None)
            raise ValueError(f"{name} must have shape {expected}, got {shape}")
        parts.append(value)
    flat = torch.cat(parts, dim=-1).reshape(
        num_envs * CORE4D_SMALLTABLE_NUM_AGENTS,
        CORE4D_SMALLTABLE_ACTOR_OBS_DIM,
    )
    normalized = models.actor_obs_normalizer(flat, update=False)
    flat_actions = models.actor.act_inference({"actor_obs": normalized})
    expected_actions = (
        num_envs * CORE4D_SMALLTABLE_NUM_AGENTS,
        CORE4D_SMALLTABLE_ACTION_DIM,
    )
    if flat_actions.shape != expected_actions or not torch.isfinite(flat_actions).all():
        raise ValueError(
            "CORE4D Actor returned invalid actions: "
            f"shape={tuple(flat_actions.shape)}, expected={expected_actions}"
        )
    return flat_actions.reshape(
        num_envs,
        CORE4D_SMALLTABLE_NUM_AGENTS,
        CORE4D_SMALLTABLE_ACTION_DIM,
    )


def save_core4d_smalltable_viser(
    path: str | Path,
    channels: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Write the five actual-state channels consumed by the paired ViSER tool."""
    shapes = {
        "root_pos": (2, 3),
        "root_quat_xyzw": (2, 4),
        "dof_pos": (2, 29),
        "object_pos_w": (3,),
        "object_quat_xyzw": (4,),
    }
    frame_count = int(np.asarray(channels["root_pos"]).shape[0])
    payload: dict[str, np.ndarray] = {}
    for name, trailing in shapes.items():
        values = np.asarray(channels[name])
        expected = (frame_count, *trailing)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite with shape {expected}, got {values.shape}")
        payload[name] = values
    payload["_metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


__all__ = [
    "core4d_smalltable_evaluation_reward_metadata",
    "deterministic_core4d_smalltable_actions",
    "save_core4d_smalltable_viser",
    "validate_core4d_smalltable_checkpoint",
]
