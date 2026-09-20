"""Opt-in bucket tracking block with an optional reference-weighted contact gate.

Filtered forces are normal contact forces on the object, not frictional force,
grasp support, or a measurement of each agent's share of the object's weight.
The manager applies the outer weight and control dt exactly once.
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
from holosoma.config_values.marl.g1.core4d_bucket_contract import BUCKET_VARIANTS, SMALLTABLE_REWARD_VARIANTS, bucket_block_weights
from holosoma.managers.reward.base import RewardTermBase
from holosoma.managers.reward.terms.interaction_vectors import weighted_body_object_vector_error
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_error_magnitude


ARTIFACT_VERSION = "core4d_bucket_vectors_v1"
POSITION_SIGMA_M = 0.3
ORIENTATION_SIGMA_RAD = 0.4
HEIGHT_SIGMA_M = 0.10
VECTOR_SIGMA_M = 0.04
VECTOR_WEIGHT = bucket_block_weights("A")[3]
DISTANCE_FLOOR_M = 0.10
CONTACT_THRESHOLD_N = 1.0
GATE_ERROR_SCALE = 2.0
HAND_FILTER_PATHS = tuple(
    f"/World/envs/env_.*/{robot}/{hand}_rubber_hand_link"
    for robot in ("Robot", "Robot_1") for hand in ("left", "right")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., :3].expand_as(vector)
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., 3:] * cross + torch.cross(xyz, cross, dim=-1)


def contact_gate(
    reference_weights: torch.Tensor, contacts: torch.Tensor, variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gate and confidence-weighted missing-contact error per env.

    Artifact loading validates confidence weights in [0, 1]. The denominator
    is max(1, sum(alpha)): weak confidence is not amplified to a full-strength
    requirement. Zero target rows have zero error/unit gate, even for B.
    Extra contacts are not penalized by this missing-contact-only definition.
    """
    if variant not in BUCKET_VARIANTS:
        raise ValueError(f"Bucket variant must be one of {BUCKET_VARIANTS}")
    if reference_weights.ndim != 2 or reference_weights.shape[-1] != 2 or contacts.shape != reference_weights.shape:
        raise ValueError("Contact weights and contacts must have matching [N, 2] shapes")
    denominator = reference_weights.sum(dim=-1)
    missing = (reference_weights * (~contacts.bool()).to(reference_weights.dtype)).sum(dim=-1)
    error = missing / denominator.clamp_min(1.0)
    gate = 0.5 + 0.5 * torch.exp(-GATE_ERROR_SCALE * error) if variant == "B" else torch.ones_like(error)
    return gate, error


class BucketInteractionReward(RewardTermBase):
    """Return the variant-weighted position/orientation/height/relation block.

    A_no_rel/A_no_height remove only that reward contribution, not diagnostics.
    Every variant reads the same current-step four-hand filtered force sensor.
    There is no contact history, smoothing, hysteresis, or policy observation.
    Contact sensor reset/fresh-sample validity is owned by the environment.
    """

    def __init__(self, cfg: RewardTermCfg, env: Any):
        super().__init__(cfg, env)
        self.variant = cfg.params.get("variant", "A")
        self.block_weights = bucket_block_weights(self.variant)
        if int(env.num_agents) != 2:
            raise ValueError("Bucket reward requires two agents")
        raw_path = cfg.params.get("reference_file")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("Bucket reward requires an explicit reference_file")
        self.reference_file = Path(resolve_data_file_path(raw_path))
        self.reference_sha256 = _sha256(self.reference_file)
        expected_hash = cfg.params.get("reference_sha256")
        if expected_hash is not None and expected_hash != self.reference_sha256:
            raise ValueError("Bucket artifact SHA256 does not match the configured hash")
        with np.load(self.reference_file, allow_pickle=False) as artifact:
            self.metadata = json.loads(str(artifact["metadata_json"].item()))
            if not isinstance(self.metadata, dict) or self.metadata.get("version") != ARTIFACT_VERSION:
                raise ValueError("Unsupported bucket interaction artifact version")
            if self.metadata.get("contact_target") == "not_used_for_smalltable_A" and self.variant not in SMALLTABLE_REWARD_VARIANTS:
                raise ValueError("Table transfer supports only A and the ungated both-removed control")
            arrays = {
                name: np.asarray(artifact[name], dtype=np.float32)
                for name in (
                    "reference_body_points_world", "reference_object_points_world", "object_points",
                    "point_offsets", "body_point_weights", "reference_contact_weights",
                )
            }
            body_names = np.asarray(artifact["point_body_names"])
            point_names = np.asarray(artifact["point_names"])
            self.fps = float(artifact["fps"].item())
        if _sha256(self.reference_file) != self.reference_sha256:
            raise ValueError("Bucket artifact changed while being loaded")
        reference_body = arrays["reference_body_points_world"]
        objects = arrays["object_points"]
        if reference_body.ndim != 4 or reference_body.shape[0] < 1:
            raise ValueError("Bucket body reference must have nonempty [T, 2, 19, 3] shape")
        if objects.ndim != 2 or objects.shape[0] < 1 or objects.shape[1] != 3:
            raise ValueError("Bucket object points must have nonempty [K, 3] shape")
        self.num_frames = reference_body.shape[0]
        self.num_object_points = objects.shape[0]
        shapes = {
            "reference_body_points_world": (self.num_frames, 2, 19, 3),
            "reference_object_points_world": (self.num_frames, self.num_object_points, 3),
            "object_points": (self.num_object_points, 3),
            "point_offsets": (19, 3),
            "body_point_weights": (19,),
            "reference_contact_weights": (self.num_frames, 2),
        }
        for name, shape in shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"Invalid {name}: expected finite {shape}")
        if body_names.shape != (19,) or point_names.shape != (19,) or body_names.dtype.kind not in "US" or point_names.dtype.kind not in "US":
            raise ValueError("Bucket landmark body/point names must be string arrays [19]")
        self.point_body_names = [str(value) for value in body_names.tolist()]
        self.point_names = [str(value) for value in point_names.tolist()]
        if len(set(self.point_names)) != 19 or any(not name for name in self.point_names + self.point_body_names):
            raise ValueError("Bucket requires 19 unique nonempty point names and nonempty body names")
        prior = arrays["body_point_weights"]
        if (prior < 0).any() or prior.sum() <= 0:
            raise ValueError("Bucket body_point_weights must be nonnegative with positive total")
        if ((arrays["reference_contact_weights"] < 0) | (arrays["reference_contact_weights"] > 1)).any():
            raise ValueError("Bucket reference_contact_weights must lie in [0, 1]")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Bucket artifact fps must be finite and positive")
        source_hash = self.metadata.get("runtime_reference_sha256", "")
        if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise ValueError("Bucket artifact requires runtime_reference_sha256")
        for name, expected in (
            ("num_frames", self.num_frames), ("num_agents", 2), ("num_body_points", 19),
            ("actual_object_points", self.num_object_points), ("fps", self.fps),
        ):
            if name in self.metadata and self.metadata[name] != expected:
                raise ValueError(f"Bucket metadata {name} disagrees with artifact arrays")
        for name, value in arrays.items():
            setattr(self, name, torch.as_tensor(value, device=env.device))
        self._command = None
        self._body_indexes = None
        self._sensor = None
        self.last_error_m2 = torch.zeros(env.num_envs, 2, device=env.device)
        self.last_contacts = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.last_raw_reward = torch.zeros(env.num_envs, device=env.device)
        self.last_gate = torch.ones(env.num_envs, device=env.device)
        self.last_components = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name in (
                "position", "orientation", "height", "relative", "gate", "contact_error",
                "height_abs_error_m", "height_signed_error_m", "contact_agent0", "contact_agent1",
                "vector_error_agent0_m2", "vector_error_agent1_m2", "raw_ungated", "raw_reward",
            )
        }
        self._sums = {name: torch.zeros((), device=env.device) for name in self.last_components}
        self._reward_min = torch.full((), float("inf"), device=env.device)
        self._reward_max = torch.full((), -float("inf"), device=env.device)
        self._sample_count = 0

    def _bind(self, env: Any) -> Any:
        if self._command is not None:
            return self._command
        command = env.command_manager.get_state("paired_motion_command")
        if command is None:
            raise ValueError("Bucket reward requires paired_motion_command")
        reference = command.reference
        if reference.num_frames != self.num_frames or not math.isclose(float(reference.fps), self.fps):
            raise ValueError("Bucket artifact frame count/fps differs from runtime reference")
        if not math.isclose(float(env.dt) * self.fps, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Bucket artifact fps must equal the control rate")
        paths = reference.motion_files
        if len(paths) != 1 or _sha256(Path(resolve_data_file_path(str(paths[0])))) != self.metadata["runtime_reference_sha256"]:
            raise ValueError("Bucket artifact runtime reference SHA256 mismatch")
        if _sha256(self.reference_file) != self.reference_sha256:
            raise ValueError("Bucket artifact changed before binding")
        names = list(env.simulator._body_list)
        if len(set(names)) != len(names):
            raise ValueError("Simulator body names must be unique")
        missing = sorted(set(self.point_body_names) - set(names))
        if missing:
            raise ValueError(f"Bucket landmarks refer to unavailable bodies: {missing}")
        body_shape = (env.num_envs, 2, len(names))
        if env.simulator.agent_rigid_body_pos.shape != (*body_shape, 3) or env.simulator.agent_rigid_body_rot.shape != (*body_shape, 4):
            raise ValueError("Bucket requires [N, 2, B, xyz/xyzw] simulator body states")
        if command.time_steps.shape != (env.num_envs,) or command.time_steps.dtype != torch.long:
            raise ValueError("Bucket command phase must be long [N]")
        for name, width in (
            ("object_pos_w", 3), ("simulator_object_pos_w", 3),
            ("object_quat_w", 4), ("simulator_object_quat_w", 4),
        ):
            if getattr(command, name).shape != (env.num_envs, width):
                raise ValueError(f"Bucket command {name} has an invalid shape")
        sensor = getattr(env.simulator, "object_hand_contact_sensor", None)
        if sensor is None:
            raise ValueError("All bucket variants require the same object_hand_contact_sensor")
        if tuple(sensor.cfg.filter_prim_paths_expr) != HAND_FILTER_PATHS:
            raise ValueError("Bucket hand filters must be Robot L/R then Robot_1 L/R, explicit rigid-body paths")
        forces = sensor.data.force_matrix_w
        if forces is None or forces.shape != (env.num_envs, 1, 4, 3):
            raise ValueError("Bucket filtered normal force matrix must have shape [N, 1, 4, 3]")
        self._body_indexes = torch.tensor([names.index(name) for name in self.point_body_names], device=env.device)
        self._sensor = sensor
        self._command = command
        return command

    def landmarks_world(self, env: Any) -> torch.Tensor:
        self._bind(env)
        positions = env.simulator.agent_rigid_body_pos[:, :, self._body_indexes]
        rotations = env.simulator.agent_rigid_body_rot[:, :, self._body_indexes]
        return positions + _rotate_xyzw(rotations, self.point_offsets[None, None].expand_as(positions))

    @torch.no_grad()
    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        command = self._bind(env)
        phase = command.time_steps
        actual_objects = _rotate_xyzw(
            command.simulator_object_quat_w[:, None, :],
            self.object_points[None].expand(env.num_envs, -1, -1),
        ) + command.simulator_object_pos_w[:, None, :]
        # Each side may include a different common origin translation: it
        # cancels within body-object vectors. The world axes must still match.
        error, _ = weighted_body_object_vector_error(
            self.landmarks_world(env), self.reference_body_points_world[phase],
            actual_objects[:, None], self.reference_object_points_world[phase, None],
            body_point_weights=self.body_point_weights, distance_floor_m=DISTANCE_FLOOR_M,
        )
        delta = command.simulator_object_pos_w - command.object_pos_w
        squared = delta.square()
        position = torch.exp(-(squared[:, 0] + squared[:, 1] + 2.0 * squared[:, 2]) / POSITION_SIGMA_M**2)
        orientation = torch.exp(-quat_error_magnitude(command.object_quat_w, command.simulator_object_quat_w).square() / ORIENTATION_SIGMA_RAD**2)
        height = torch.exp(-squared[:, 2] / HEIGHT_SIGMA_M**2)
        relative = torch.exp(-error.mean(dim=1) / VECTOR_SIGMA_M**2)
        forces = self._sensor.data.force_matrix_w
        if forces is None or forces.shape != (env.num_envs, 1, 4, 3):
            raise ValueError("Bucket filtered normal force matrix must retain shape [N, 1, 4, 3]")
        # Norm each hand independently before any agent aggregation, avoiding
        # cancellation of opposing left/right clamping forces.
        contacts = torch.linalg.vector_norm(forces[:, 0].reshape(env.num_envs, 2, 2, 3), dim=-1).gt(CONTACT_THRESHOLD_N).any(dim=-1)
        episode_steps = getattr(env, "episode_length_buf", None)
        if episode_steps is not None:
            if episode_steps.shape != (env.num_envs,):
                raise ValueError("Bucket episode_length_buf must have shape [N]")
            contacts &= episode_steps.gt(0)[:, None]
        gate, contact_error = contact_gate(self.reference_contact_weights[phase], contacts, self.variant)
        position_weight, orientation_weight, height_weight, relation_weight = self.block_weights
        ungated = (position_weight * position + orientation_weight * orientation
                   + height_weight * height + relation_weight * relative)
        reward = ungated * gate
        values = {
            "position": position, "orientation": orientation, "height": height, "relative": relative,
            "gate": gate, "contact_error": contact_error, "height_abs_error_m": delta[:, 2].abs(),
            "height_signed_error_m": delta[:, 2], "contact_agent0": contacts[:, 0].float(),
            "contact_agent1": contacts[:, 1].float(), "vector_error_agent0_m2": error[:, 0],
            "vector_error_agent1_m2": error[:, 1], "raw_ungated": ungated, "raw_reward": reward,
        }
        self.last_error_m2.copy_(error)
        self.last_contacts.copy_(contacts)
        self.last_raw_reward.copy_(reward)
        self.last_gate.copy_(gate)
        for name, value in values.items():
            self.last_components[name].copy_(value)
            self._sums[name].add_(value.sum())
        self._reward_min.copy_(torch.minimum(self._reward_min, reward.min()))
        self._reward_max.copy_(torch.maximum(self._reward_max, reward.max()))
        self._sample_count += env.num_envs
        return reward

    def get_iteration_diagnostics(self, reset: bool = True) -> dict[str, torch.Tensor]:
        if self._sample_count == 0:
            return {}
        result = {f"Bucket/{name}_mean": value / self._sample_count for name, value in self._sums.items()}
        result.update({
            "Bucket/sample_count": self._reward_min.new_tensor(self._sample_count),
            "Bucket/raw_reward_min": self._reward_min.clone(),
            "Bucket/raw_reward_max": self._reward_max.clone(),
        })
        if reset:
            for value in self._sums.values():
                value.zero_()
            self._reward_min.fill_(float("inf"))
            self._reward_max.fill_(-float("inf"))
            self._sample_count = 0
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        selection = slice(None) if env_ids is None else env_ids
        self.last_error_m2[selection] = 0
        self.last_contacts[selection] = False
        self.last_raw_reward[selection] = 0
        self.last_gate[selection] = 1
        for value in self.last_components.values():
            value[selection] = 0
        # Iteration sums deliberately survive episode resets. Sensor freshness
        # is reset by the environment, not by this read-only reward term.


__all__ = ["ARTIFACT_VERSION", "HAND_FILTER_PATHS", "BucketInteractionReward", "contact_gate"]
