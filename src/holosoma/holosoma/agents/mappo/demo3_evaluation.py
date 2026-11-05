"""Pure helpers for deterministic Demo 3 checkpoint evaluation.

The Isaac Sim entry point lives in :mod:`scripts.evaluate_demo3_tug`.  This
module deliberately contains no Isaac Sim imports so its checkpoint, asset,
winner, and recording contracts can be tested on CPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from holosoma.agents.mappo.demo3_checkpoint import (
    DEMO3_CHECKPOINT_INTERVAL,
    DEMO3_DEFAULT_CRITIC_ONLY_ITERATIONS,
    DEMO3_DEFAULT_FULL_ACTOR_ITERATIONS,
    DEMO3_MAPPO_CHECKPOINT_VERSION,
    DEMO3_REFERENCE_FPS,
    DEMO3_REFERENCE_FRAMES,
    DEMO3_ROBOT_URDF_SHA256,
    DEMO3_RUNTIME_REFERENCE_SHA256,
    DEMO3_SQUARE_TABLE_URDF_SHA256,
    DEMO3_WARM_START_SHA256,
    demo3_ppo_contract,
    demo3_training_contract,
)
from holosoma.agents.mappo.demo3_initialization import (
    DEMO3_ACTION_DIM,
    DEMO3_ACTOR_OBS_DIM,
    DEMO3_CRITIC_OBS_DIM,
)
from holosoma.agents.mappo.demo3_runner import (
    DEMO3_ACTOR_OBSERVATION_LAYOUT,
    DEMO3_NUM_AGENTS,
)
from holosoma.config_values.marl.g1.demo3_experiment import (
    g1_29dof_demo3_tug_baseline,
)


DEMO3_WIN_THRESHOLD_M = 0.05
DEMO3_RUNTIME_REFERENCE_RELATIVE_PATH = (
    "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "demo3_tug/sub3_010_diagonal_tug_runtime.npz"
)
DEMO3_ROBOT_URDF_RELATIVE_PATH = (
    "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
)
DEMO3_TABLE_URDF_RELATIVE_PATH = (
    "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "objects_squaretable_demo3_training.urdf"
)
DEMO3_WARM_START_RELATIVE_PATH = (
    "logs/Demo3Tug/checkpoints/model_07999_actor164_table_neutral.pt"
)
DEMO3_TERMINATION_NAMES = ("clear_robot_fall", "reference_horizon")
VISER_FIVE_CHANNEL_SHAPES = {
    "root_pos": (DEMO3_NUM_AGENTS, 3),
    "root_quat_xyzw": (DEMO3_NUM_AGENTS, 4),
    "dof_pos": (DEMO3_NUM_AGENTS, DEMO3_ACTION_DIM),
    "object_pos_w": (3,),
    "object_quat_xyzw": (4,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_demo3_metadata(
    *,
    critic_only_iterations: int = DEMO3_DEFAULT_CRITIC_ONLY_ITERATIONS,
    full_actor_iterations: int = DEMO3_DEFAULT_FULL_ACTOR_ITERATIONS,
    checkpoint_interval: int = DEMO3_CHECKPOINT_INTERVAL,
    num_steps_per_env: int = 24,
) -> dict[str, Any]:
    """Return the exact current Demo 3 MAPPO checkpoint contract."""

    training_contract = demo3_training_contract(
        critic_only_iterations=critic_only_iterations,
        full_actor_iterations=full_actor_iterations,
        checkpoint_interval=checkpoint_interval,
    )
    ppo_contract = demo3_ppo_contract(
        g1_29dof_demo3_tug_baseline.algo.config,
        num_steps_per_env=num_steps_per_env,
    )
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
        "source_iteration": training_contract["warm_start_iteration"],
        "source_sha256": DEMO3_WARM_START_SHA256,
        "ppo_contract": ppo_contract,
        **training_contract,
    }


def validate_demo3_evaluation_checkpoint(state: Mapping[str, Any]) -> int:
    """Fail closed unless ``state`` is an uncontaminated Demo 3 checkpoint."""

    metadata = state.get("demo3_mappo")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Demo 3 evaluation checkpoint metadata mismatch: " f"{metadata!r}"
        )
    schedule_values: dict[str, int] = {}
    for name in (
        "critic_only_iterations",
        "full_actor_iterations",
        "checkpoint_interval",
    ):
        value = metadata.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"Demo 3 evaluation checkpoint metadata {name!r} must be an integer"
            )
        schedule_values[name] = value
    ppo_contract = metadata.get("ppo_contract")
    if not isinstance(ppo_contract, Mapping):
        raise ValueError("Demo 3 evaluation checkpoint ppo_contract must be a mapping")
    num_steps_per_env = ppo_contract.get("num_steps_per_env")
    if (
        not isinstance(num_steps_per_env, int)
        or isinstance(num_steps_per_env, bool)
        or num_steps_per_env < 1
    ):
        raise ValueError(
            "Demo 3 evaluation checkpoint ppo_contract num_steps_per_env "
            "must be a positive integer"
        )
    try:
        expected_metadata = expected_demo3_metadata(
            **schedule_values,
            num_steps_per_env=num_steps_per_env,
        )
    except ValueError as error:
        raise ValueError(
            f"Demo 3 evaluation checkpoint training schedule is invalid: {error}"
        ) from error
    if dict(metadata) != expected_metadata:
        raise ValueError(
            "Demo 3 evaluation checkpoint metadata mismatch: " f"{metadata!r}"
        )
    contaminants = sorted({"plan5_mappo", "demo4_mappo"}.intersection(state))
    if contaminants:
        raise ValueError(
            f"Demo 3 evaluation checkpoint contains cross-demo metadata: {contaminants}"
        )
    required = {
        "actor_model_state_dict",
        "actor_obs_normalizer_state_dict",
        "iter",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Demo 3 evaluation checkpoint is incomplete: {missing}")
    iteration = state["iter"]
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError(
            f"Demo 3 checkpoint iteration must be non-negative, got {iteration!r}"
        )
    return iteration


def validate_demo3_evaluation_assets(
    repo_root: str | Path,
    *,
    warm_start_path: str | Path | None = None,
) -> dict[str, str]:
    """Validate the frozen runtime/reference/URDF/warm-start byte contract."""

    root = Path(repo_root).expanduser().resolve()
    paths = {
        "runtime_reference": root / DEMO3_RUNTIME_REFERENCE_RELATIVE_PATH,
        "rubber_hand_robot_urdf": root / DEMO3_ROBOT_URDF_RELATIVE_PATH,
        "square_table_urdf": root / DEMO3_TABLE_URDF_RELATIVE_PATH,
        "actor164_warm_start": (
            Path(warm_start_path).expanduser().resolve()
            if warm_start_path is not None
            else root / DEMO3_WARM_START_RELATIVE_PATH
        ),
    }
    expected_hashes = {
        "runtime_reference": DEMO3_RUNTIME_REFERENCE_SHA256,
        "rubber_hand_robot_urdf": DEMO3_ROBOT_URDF_SHA256,
        "square_table_urdf": DEMO3_SQUARE_TABLE_URDF_SHA256,
        "actor164_warm_start": DEMO3_WARM_START_SHA256,
    }
    resolved: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Demo 3 required asset is missing ({name}): {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hashes[name]:
            raise ValueError(
                f"Demo 3 {name} SHA256 mismatch: expected {expected_hashes[name]}, "
                f"got {actual_hash}"
            )
        resolved[name] = str(path)

    runtime = paths["runtime_reference"]
    with np.load(runtime, allow_pickle=False) as data:
        expected_shapes = {
            "agent_joint_pos": (DEMO3_REFERENCE_FRAMES, DEMO3_NUM_AGENTS, DEMO3_ACTION_DIM),
            "agent_joint_vel": (DEMO3_REFERENCE_FRAMES, DEMO3_NUM_AGENTS, DEMO3_ACTION_DIM),
            "agent_body_pos_w": (DEMO3_REFERENCE_FRAMES, DEMO3_NUM_AGENTS, 51, 3),
            "agent_body_quat_w": (DEMO3_REFERENCE_FRAMES, DEMO3_NUM_AGENTS, 51, 4),
            "object_pos_w": (DEMO3_REFERENCE_FRAMES, 3),
            "object_quat_w": (DEMO3_REFERENCE_FRAMES, 4),
        }
        for name, expected_shape in expected_shapes.items():
            if name not in data or data[name].shape != expected_shape:
                actual = None if name not in data else data[name].shape
                raise ValueError(
                    f"Demo 3 runtime {name} shape mismatch: expected "
                    f"{expected_shape}, got {actual}"
                )
        if int(data["fps"]) != DEMO3_REFERENCE_FPS:
            raise ValueError(
                f"Demo 3 runtime FPS must be {DEMO3_REFERENCE_FPS}, got {int(data['fps'])}"
            )
        provenance = json.loads(str(data["provenance"]))
        expected_provenance = {
            "demo": "Demo3Tug",
            "opponent_transform": "world_z_half_turn",
            "table_channel_role": "reset_and_schema_only_not_tracking_target",
        }
        for name, expected_value in expected_provenance.items():
            if provenance.get(name) != expected_value:
                raise ValueError(
                    f"Demo 3 runtime provenance {name!r} mismatch: "
                    f"expected {expected_value!r}, got {provenance.get(name)!r}"
                )
    return resolved


@torch.no_grad()
def deterministic_demo3_actions(
    models: Any,
    observations: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Run only the shared Actor, once per competitor; never call the Critic."""

    first = observations.get(DEMO3_ACTOR_OBSERVATION_LAYOUT[0][0])
    if not isinstance(first, torch.Tensor) or first.ndim != 3:
        raise ValueError("Demo 3 actor_obs must have shape [num_envs, 2, 154]")
    num_envs = first.shape[0]
    parts = []
    for name, width in DEMO3_ACTOR_OBSERVATION_LAYOUT:
        value = observations.get(name)
        expected = (num_envs, DEMO3_NUM_AGENTS, width)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            shape = getattr(value, "shape", None)
            raise ValueError(f"{name} must have shape {expected}, got {shape}")
        parts.append(value)
    combined = torch.cat(parts, dim=-1)
    flat = combined.reshape(num_envs * DEMO3_NUM_AGENTS, DEMO3_ACTOR_OBS_DIM)
    normalized = models.actor_obs_normalizer(flat, update=False)
    flat_actions = models.actor.act_inference({"actor_obs": normalized})
    expected_actions = (num_envs * DEMO3_NUM_AGENTS, DEMO3_ACTION_DIM)
    if tuple(flat_actions.shape) != expected_actions:
        raise ValueError(
            f"Demo 3 Actor actions must have shape {expected_actions}, "
            f"got {tuple(flat_actions.shape)}"
        )
    if not torch.isfinite(flat_actions).all():
        raise ValueError("Demo 3 Actor produced non-finite deterministic actions")
    return flat_actions.reshape(num_envs, DEMO3_NUM_AGENTS, DEMO3_ACTION_DIM)


def classify_demo3_winner(
    signed_displacement_m: float,
    *,
    threshold_m: float = DEMO3_WIN_THRESHOLD_M,
) -> str:
    """Classify using strict ±threshold bounds along Agent A's initial Pull axis."""

    if not np.isfinite(signed_displacement_m):
        raise ValueError("Demo 3 signed displacement must be finite")
    if threshold_m <= 0.0 or not np.isfinite(threshold_m):
        raise ValueError("Demo 3 win threshold must be finite and positive")
    if signed_displacement_m > threshold_m:
        return "agent_a"
    if signed_displacement_m < -threshold_m:
        return "agent_b"
    return "draw"


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
class Demo3EpisodeResult:
    episode: int
    steps: int
    winner: str
    signed_displacement_m: float
    planar_displacement_xy_m: tuple[float, float]
    fall: bool
    timeout: bool
    reward_sum_agent_a: float
    reward_sum_agent_b: float

    def __post_init__(self) -> None:
        if self.winner not in {"agent_a", "agent_b", "draw"}:
            raise ValueError(f"Invalid Demo 3 winner: {self.winner!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "steps": self.steps,
            "winner": self.winner,
            "signed_displacement_m": self.signed_displacement_m,
            "planar_displacement_xy_m": list(self.planar_displacement_xy_m),
            "fall": self.fall,
            "timeout": self.timeout,
            "reward_sum_agent_a": self.reward_sum_agent_a,
            "reward_sum_agent_b": self.reward_sum_agent_b,
        }


def representative_episode_indices(
    results: list[Demo3EpisodeResult],
) -> dict[str, int]:
    """Select strongest A/B wins plus the largest absolute displacement.

    The returned values are list indices, not episode identifiers.  A category
    is omitted when no episode has that outcome.  Ties prefer the earliest
    episode for reproducibility.
    """

    if not results:
        raise ValueError("At least one Demo 3 episode result is required")
    selections: dict[str, int] = {
        "max_absolute_displacement": max(
            range(len(results)),
            key=lambda index: (
                abs(results[index].signed_displacement_m),
                -results[index].episode,
            ),
        )
    }
    agent_a = [index for index, result in enumerate(results) if result.winner == "agent_a"]
    if agent_a:
        selections["agent_a_win"] = max(
            agent_a,
            key=lambda index: (results[index].signed_displacement_m, -results[index].episode),
        )
    agent_b = [index for index, result in enumerate(results) if result.winner == "agent_b"]
    if agent_b:
        selections["agent_b_win"] = min(
            agent_b,
            key=lambda index: (results[index].signed_displacement_m, results[index].episode),
        )
    return selections


__all__ = [
    "DEMO3_ACTOR_OBSERVATION_LAYOUT",
    "DEMO3_REFERENCE_FPS",
    "DEMO3_REFERENCE_FRAMES",
    "DEMO3_RUNTIME_REFERENCE_RELATIVE_PATH",
    "DEMO3_TERMINATION_NAMES",
    "DEMO3_WARM_START_RELATIVE_PATH",
    "DEMO3_WARM_START_SHA256",
    "DEMO3_WIN_THRESHOLD_M",
    "VISER_FIVE_CHANNEL_SHAPES",
    "Demo3EpisodeResult",
    "classify_demo3_winner",
    "deterministic_demo3_actions",
    "expected_demo3_metadata",
    "representative_episode_indices",
    "save_viser_episode",
    "validate_demo3_evaluation_assets",
    "validate_demo3_evaluation_checkpoint",
    "validate_viser_five_channels",
]
