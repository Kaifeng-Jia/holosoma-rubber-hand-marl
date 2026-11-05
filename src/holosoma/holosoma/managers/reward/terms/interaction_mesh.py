"""Frozen-reference interaction-mesh reward for the paired small-table task.

Only physical state is read here: this does not add Actor/Critic observations,
write a reference into physics, estimate contact force, or reconstruct a graph.
The offline artifact stores Q = L_body.T @ W @ L_body for the agreed grouped
Laplacian error; object-row contributions are included in Q.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.path import resolve_data_file_path


ARTIFACT_VERSION = "core4d_smalltable_interaction_mesh_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply unit XYZW quaternions, broadcasting over agents and landmarks."""
    xyz = quaternion[..., :3].expand_as(vector)
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., 3:] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def grouped_quadratic_error(
    actual_points_object: torch.Tensor,
    reference_points_object: torch.Tensor,
    q_matrices: torch.Tensor,
) -> torch.Tensor:
    """Return per-agent grouped squared error, in m², without exp/weight/dt.

    Q is validated as positive semidefinite on artifact load. Clamping here
    only removes a possible negative floating-point round-off near zero.
    """
    delta = actual_points_object - reference_points_object
    return (delta * torch.matmul(q_matrices, delta)).sum(dim=(-2, -1)).clamp_min(0.0)


class InteractionMeshReward(RewardTermBase):
    """Team reward exp(-mean_agent(grouped_error) / sigma²).

    The reward manager applies the configured weight and dt exactly once.
    Binding the motion command is deliberately lazy: RewardManager is created
    before CommandManager in the environment startup sequence.
    """

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        self.sigma = float(cfg.params.get("sigma", 0.06))
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Interaction mesh sigma must be finite and positive")
        self.reference_file = Path(resolve_data_file_path(str(cfg.params["reference_file"])))
        with np.load(self.reference_file, allow_pickle=False) as artifact:
            self.metadata = json.loads(str(artifact["metadata_json"].item()))
            reference = np.asarray(artifact["reference_points_object"], dtype=np.float32)
            matrices = np.asarray(artifact["q_matrices"], dtype=np.float32)
            offsets = np.asarray(artifact["point_offsets"], dtype=np.float32)
            object_points = np.asarray(artifact["object_points"], dtype=np.float32)
            self.point_body_names = [str(value) for value in artifact["point_body_names"].tolist()]
            self.point_names = [str(value) for value in artifact["point_names"].tolist()]
            self.fps = float(artifact["fps"].item())
        if self.metadata.get("version") != ARTIFACT_VERSION:
            raise ValueError("Unsupported interaction mesh artifact version")
        self.num_frames = len(reference)
        shapes = {
            "reference_points_object": (reference, (self.num_frames, 2, 19, 3)),
            "q_matrices": (matrices, (self.num_frames, 2, 19, 19)),
            "point_offsets": (offsets, (19, 3)),
            "object_points": (object_points, (85, 3)),
        }
        for name, (value, expected_shape) in shapes.items():
            if value.shape != expected_shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected finite {expected_shape}, got {value.shape}")
        if self.num_frames < 1 or len(self.point_body_names) != 19 or len(set(self.point_names)) != 19:
            raise ValueError("Interaction artifact requires nonempty frames and 19 uniquely named points")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Interaction artifact fps must be finite and positive")
        for key, value in {
            "num_frames": self.num_frames, "num_agents": 2,
            "num_body_points": 19, "actual_object_points": 85, "fps": self.fps,
        }.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"Interaction artifact metadata does not match {key}={value}")
        if not np.allclose(matrices, matrices.swapaxes(-1, -2), rtol=1e-5, atol=1e-7):
            raise ValueError("Interaction Q matrices must be symmetric")
        if np.linalg.eigvalsh(matrices.astype(np.float64)).min() < -1e-6:
            raise ValueError("Interaction Q matrices must be positive semidefinite")
        source_hash = self.metadata.get("runtime_reference_sha256", "")
        if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("Interaction artifact requires source reference SHA256")

        self.reference_points_object = torch.as_tensor(reference, device=env.device)
        self.q_matrices = torch.as_tensor(matrices, device=env.device)
        self.point_offsets = torch.as_tensor(offsets, device=env.device)
        self.last_error_m2 = torch.zeros(env.num_envs, 2, device=env.device)
        self.last_raw_reward = torch.zeros(env.num_envs, device=env.device)
        self._command = None
        self._body_indexes = None
        self._error_sum = torch.zeros(2, device=env.device)
        self._reward_sum = torch.zeros((), device=env.device)
        self._reward_min = torch.full((), float("inf"), device=env.device)
        self._reward_max = torch.full((), -float("inf"), device=env.device)
        self._sample_count = 0

    def _bind(self, env: Any) -> Any:
        if self._command is not None:
            return self._command
        command = env.command_manager.get_state("paired_motion_command")
        reference = command.reference
        if reference.num_frames != self.num_frames or not math.isclose(float(reference.fps), self.fps):
            raise ValueError("Interaction artifact frame count/fps differs from active motion reference")
        if not math.isclose(float(env.dt) * self.fps, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Interaction reference fps must match the command control-step rate")
        paths = reference.motion_files
        if len(paths) != 1 or _sha256(Path(resolve_data_file_path(str(paths[0])))) != self.metadata["runtime_reference_sha256"]:
            raise ValueError("Interaction artifact SHA256 differs from active motion reference")
        body_names = list(env.simulator._body_list)
        if len(set(body_names)) != len(body_names):
            raise ValueError("Simulator body names must be unique")
        missing = sorted(set(self.point_body_names) - set(body_names))
        if missing:
            raise ValueError(f"Interaction landmarks refer to unavailable simulator bodies: {missing}")
        expected_shape = (env.num_envs, 2, len(body_names), 3)
        if tuple(env.simulator.agent_rigid_body_pos.shape) != expected_shape:
            raise ValueError("Interaction landmarks expect two agents in simulator body-list order")
        if tuple(env.simulator.agent_rigid_body_rot.shape) != (*expected_shape[:-1], 4):
            raise ValueError("Interaction landmarks require XYZW rotations for every simulator body")
        if command.time_steps.shape != (env.num_envs,) or command.time_steps.dtype != torch.long:
            raise ValueError("Interaction command phase must be a long tensor [num_envs]")
        self._body_indexes = torch.tensor(
            [body_names.index(name) for name in self.point_body_names],
            dtype=torch.long, device=env.device,
        )
        self._command = command
        return command

    def landmarks_world(self, env: Any) -> torch.Tensor:
        """Read physical landmark positions [environment, agent, point, xyz]."""
        self._bind(env)
        positions = env.simulator.agent_rigid_body_pos[:, :, self._body_indexes]
        rotations = env.simulator.agent_rigid_body_rot[:, :, self._body_indexes]
        offsets = self.point_offsets[None, None].expand_as(positions)
        return positions + _rotate_xyzw(rotations, offsets)

    def points_object(self, env: Any) -> torch.Tensor:
        """Express real landmarks in the actual object's actor-origin frame.

        Subtract the actual object origin, including its environment offset;
        do not use COM, reference object pose, torso alignment, or yaw only.
        """
        command = self._bind(env)
        relative = self.landmarks_world(env) - command.simulator_object_pos_w[:, None, None, :]
        quaternion = command.simulator_object_quat_w[:, None, None, :]
        conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)
        return _rotate_xyzw(conjugate, relative)

    @torch.no_grad()
    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        command = self._bind(env)
        phase = command.time_steps
        error = grouped_quadratic_error(
            self.points_object(env), self.reference_points_object[phase], self.q_matrices[phase],
        )
        reward = torch.exp(-error.mean(dim=1) / self.sigma**2)
        self.last_error_m2.copy_(error)
        self.last_raw_reward.copy_(reward)
        self._error_sum.add_(error.sum(dim=0))
        self._reward_sum.add_(reward.sum())
        self._reward_min.copy_(torch.minimum(self._reward_min, reward.min()))
        self._reward_max.copy_(torch.maximum(self._reward_max, reward.max()))
        self._sample_count += env.num_envs
        return reward

    def get_iteration_diagnostics(self, reset: bool = True) -> dict[str, torch.Tensor]:
        """Aggregate all evaluated states since the previous drain, on device.

        Environment resets do not erase this accumulation. RMS is the square
        root of mean squared error, not the mean of per-state square roots.
        No CPU synchronization takes place until the caller logs the tensors.
        """
        if self._sample_count == 0:
            return {}
        means = self._error_sum / self._sample_count
        result = {
            "Interaction/error_agent0_m2": means[0].clone(),
            "Interaction/error_agent1_m2": means[1].clone(),
            "Interaction/error_team_m2": means.mean(),
            "Interaction/rms_agent0_m": means[0].sqrt(),
            "Interaction/rms_agent1_m": means[1].sqrt(),
            "Interaction/rms_team_m": means.mean().sqrt(),
            "Interaction/raw_reward_mean": self._reward_sum / self._sample_count,
            "Interaction/raw_reward_min": self._reward_min.clone(),
            "Interaction/raw_reward_max": self._reward_max.clone(),
            "Interaction/sample_count": self._error_sum.new_tensor(self._sample_count),
        }
        if reset:
            self._error_sum.zero_()
            self._reward_sum.zero_()
            self._reward_min.fill_(float("inf"))
            self._reward_max.fill_(-float("inf"))
            self._sample_count = 0
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.last_error_m2.zero_()
            self.last_raw_reward.zero_()
        else:
            self.last_error_m2[env_ids] = 0.0
            self.last_raw_reward[env_ids] = 0.0


__all__ = ["ARTIFACT_VERSION", "InteractionMeshReward", "grouped_quadratic_error"]
