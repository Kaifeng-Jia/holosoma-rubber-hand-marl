"""Build a complete Plan 5 Pull runtime reference with MuJoCo FK.

The canonical dual Pull artifact intentionally contains only two robot qpos
trajectories and one shared object qpos trajectory.  This module expands that
compact, reviewed geometry into the full body-state contract required by WBT:

* per-agent joint positions and velocities;
* per-agent body poses and world-frame spatial velocities; and
* one shared object pose and world-frame velocity trajectory.

The fixed rubber-hand G1 MuJoCo model is the source of truth for forward
kinematics and name ordering.  Quaternions are WXYZ at every file boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mujoco  # type: ignore[import-not-found]
import numpy as np

from holosoma_retargeting.dual_pull_reference import (
    DualPullReference,
    G1_29DOF_JOINT_NAMES,
    ROBOT_QPOS_DIM,
)


DEFAULT_RUBBER_HAND_G1_XML = Path(__file__).resolve().parent / "models" / "g1" / "g1_29dof.xml"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_REFERENCE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path_label(path: str | Path) -> str:
    """Record a repository-relative path, or just a basename for external inputs."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _continuous_unit_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Normalize a quaternion time series and remove representation flips."""
    result = np.asarray(quaternions, dtype=np.float64).copy()
    if result.ndim < 2 or result.shape[-1] != 4:
        raise ValueError(f"Expected quaternion time series [T, ..., 4], got {result.shape}")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise ValueError("Quaternion time series contains a non-finite or zero-norm value")
    result /= norms
    flat = result.reshape(result.shape[0], -1, 4)
    for frame in range(1, len(flat)):
        flip = np.sum(flat[frame - 1] * flat[frame], axis=-1) < 0.0
        flat[frame, flip] *= -1.0
    return result


def _quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(first, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quaternion_to_rotation_vector_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _continuous_unit_quaternions_wxyz(np.asarray(quaternion)[None])[0]
    quaternion = np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector_norm = np.linalg.norm(quaternion[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quaternion[..., 0], -1.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1.0e-12,
    )
    return quaternion[..., 1:] * scale[..., None]


def _pose_world_velocities(pose_qpos_wxyz: np.ndarray, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate free-body poses into world linear and angular velocity."""
    pose = np.asarray(pose_qpos_wxyz, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected pose trajectory [T, 7], got {pose.shape}")
    if len(pose) < 2:
        raise ValueError("At least two frames are required to compute pose velocities")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")

    dt = 1.0 / float(fps)
    edge_order = 2 if len(pose) > 2 else 1
    linear_velocity = np.gradient(pose[:, :3], dt, axis=0, edge_order=edge_order)
    quaternion = _continuous_unit_quaternions_wxyz(pose[:, 3:7])

    def relative_rotation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        # q_second * inverse(q_first) is a world-frame relative rotation for
        # local-to-world pose quaternions.
        inverse_first = first * np.array([1.0, -1.0, -1.0, -1.0])
        return _quaternion_multiply_wxyz(second, inverse_first)

    angular_velocity = np.empty((len(pose), 3), dtype=np.float64)
    angular_velocity[0] = _quaternion_to_rotation_vector_wxyz(
        relative_rotation(quaternion[0], quaternion[1])
    ) / dt
    angular_velocity[-1] = _quaternion_to_rotation_vector_wxyz(
        relative_rotation(quaternion[-2], quaternion[-1])
    ) / dt
    if len(pose) > 2:
        angular_velocity[1:-1] = _quaternion_to_rotation_vector_wxyz(
            relative_rotation(quaternion[:-2], quaternion[2:])
        ) / (2.0 * dt)
    return linear_velocity, angular_velocity


def _model_names(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    joint_names = np.asarray(
        [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(1, model.njnt)
        ]
    )
    body_names = np.asarray(
        [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
            for body_id in range(model.nbody)
        ]
    )
    if joint_names.shape != (29,) or np.any(joint_names == None):  # noqa: E711
        raise ValueError(f"Expected 29 named G1 joints in the MuJoCo model, got {joint_names.tolist()}")
    required_rubber_bodies = {"left_rubber_hand_link", "right_rubber_hand_link"}
    if not required_rubber_bodies.issubset(set(body_names.tolist())):
        raise ValueError("MuJoCo model is not the fixed rubber-hand G1 model")
    return joint_names.astype(str), body_names.astype(str)


def _reorder_robot_qpos_for_model(
    robot_qpos: np.ndarray,
    source_joint_names: np.ndarray,
    model_joint_names: np.ndarray,
) -> np.ndarray:
    source_names = [str(name) for name in np.asarray(source_joint_names).tolist()]
    if len(source_names) != 29 or len(source_names) != len(set(source_names)):
        raise ValueError("Source reference must contain 29 unique joint names")
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in model_joint_names.tolist() if name not in source_index]
    if missing:
        raise ValueError(f"Source reference is missing model joints: {missing}")
    order = np.asarray([source_index[name] for name in model_joint_names.tolist()], dtype=np.int64)
    reordered = np.concatenate((robot_qpos[:, :7], robot_qpos[:, 7:][:, order]), axis=-1)
    reordered[:, 3:7] = _continuous_unit_quaternions_wxyz(reordered[:, 3:7])
    return reordered


def _generalized_velocities_from_qpos(
    model: mujoco.MjModel,
    model_qpos: np.ndarray,
    fps: int,
) -> np.ndarray:
    """Use MuJoCo's manifold-aware position difference at every frame."""
    if len(model_qpos) < 2:
        raise ValueError("At least two frames are required to compute robot velocities")
    dt = 1.0 / float(fps)
    qvel = np.empty((len(model_qpos), model.nv), dtype=np.float64)
    mujoco.mj_differentiatePos(model, qvel[0], dt, model_qpos[0], model_qpos[1])
    mujoco.mj_differentiatePos(model, qvel[-1], dt, model_qpos[-2], model_qpos[-1])
    for frame in range(1, len(model_qpos) - 1):
        mujoco.mj_differentiatePos(
            model,
            qvel[frame],
            2.0 * dt,
            model_qpos[frame - 1],
            model_qpos[frame + 1],
        )
    return qvel


def _forward_kinematics_and_velocity(
    model: mujoco.MjModel,
    model_qpos: np.ndarray,
    model_qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = len(model_qpos)
    body_pos = np.empty((frames, model.nbody, 3), dtype=np.float64)
    body_quat = np.empty((frames, model.nbody, 4), dtype=np.float64)
    body_lin_vel = np.empty((frames, model.nbody, 3), dtype=np.float64)
    body_ang_vel = np.empty((frames, model.nbody, 3), dtype=np.float64)
    data = mujoco.MjData(model)
    spatial_velocity = np.empty(6, dtype=np.float64)

    for frame in range(frames):
        data.qpos[:] = model_qpos[frame]
        data.qvel[:] = model_qvel[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos
        body_quat[frame] = data.xquat
        for body_id in range(model.nbody):
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                spatial_velocity,
                0,
            )
            body_ang_vel[frame, body_id] = spatial_velocity[:3]
            body_lin_vel[frame, body_id] = spatial_velocity[3:]

    body_quat = _continuous_unit_quaternions_wxyz(body_quat)
    return body_pos, body_quat, body_lin_vel, body_ang_vel


@dataclass(frozen=True)
class DualPullRuntimeReference:
    """Complete explicit paired-reference artifact for offline/runtime handoff."""

    agent_joint_pos: np.ndarray
    agent_joint_vel: np.ndarray
    agent_body_pos_w: np.ndarray
    agent_body_quat_wxyz: np.ndarray
    agent_body_lin_vel_w: np.ndarray
    agent_body_ang_vel_w: np.ndarray
    object_pos_w: np.ndarray
    object_quat_wxyz: np.ndarray
    object_lin_vel_w: np.ndarray
    object_ang_vel_w: np.ndarray
    fps: int
    joint_names: np.ndarray
    body_names: np.ndarray
    provenance: Mapping[str, object]

    def save(self, path: str | Path) -> Path:
        """Save the explicit paired contract without pickle-dependent fields."""
        _validate_runtime_reference(self)
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            agent_joint_pos=self.agent_joint_pos,
            agent_joint_vel=self.agent_joint_vel,
            agent_body_pos_w=self.agent_body_pos_w,
            # Runtime follows the existing WBT key names; provenance records
            # that every ``*_quat_w`` value is WXYZ on disk.
            agent_body_quat_w=self.agent_body_quat_wxyz,
            agent_body_lin_vel_w=self.agent_body_lin_vel_w,
            agent_body_ang_vel_w=self.agent_body_ang_vel_w,
            object_pos_w=self.object_pos_w,
            object_quat_w=self.object_quat_wxyz,
            object_lin_vel_w=self.object_lin_vel_w,
            object_ang_vel_w=self.object_ang_vel_w,
            fps=np.asarray(self.fps, dtype=np.int64),
            joint_names=np.asarray(self.joint_names, dtype=str),
            body_names=np.asarray(self.body_names, dtype=str),
            provenance=np.asarray(json.dumps(dict(self.provenance), sort_keys=True)),
        )
        return output


def _validate_runtime_reference(reference: DualPullRuntimeReference) -> None:
    frames = reference.agent_joint_pos.shape[0]
    bodies = len(reference.body_names)
    expected_shapes = {
        "agent_joint_pos": (frames, 2, 29),
        "agent_joint_vel": (frames, 2, 29),
        "agent_body_pos_w": (frames, 2, bodies, 3),
        "agent_body_quat_wxyz": (frames, 2, bodies, 4),
        "agent_body_lin_vel_w": (frames, 2, bodies, 3),
        "agent_body_ang_vel_w": (frames, 2, bodies, 3),
        "object_pos_w": (frames, 3),
        "object_quat_wxyz": (frames, 4),
        "object_lin_vel_w": (frames, 3),
        "object_ang_vel_w": (frames, 3),
    }
    for name, expected in expected_shapes.items():
        value = np.asarray(getattr(reference, name))
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains a non-finite value")
    if frames < 2:
        raise ValueError("Runtime reference must contain at least two frames")
    if reference.fps <= 0:
        raise ValueError(f"FPS must be positive, got {reference.fps}")
    if np.asarray(reference.joint_names).shape != (29,):
        raise ValueError(f"joint_names must have shape (29,), got {np.asarray(reference.joint_names).shape}")
    for name in ("agent_body_quat_wxyz", "object_quat_wxyz"):
        norms = np.linalg.norm(np.asarray(getattr(reference, name)), axis=-1)
        if not np.allclose(norms, 1.0, atol=1.0e-10):
            raise ValueError(f"{name} contains a non-unit quaternion")


def build_dual_pull_runtime_reference(
    reference: DualPullReference,
    *,
    source_joint_names: np.ndarray = G1_29DOF_JOINT_NAMES,
    model_path: str | Path = DEFAULT_RUBBER_HAND_G1_XML,
    provenance: Mapping[str, object] | None = None,
) -> DualPullRuntimeReference:
    """Expand canonical A/B qpos into the complete paired WBT contract."""
    if reference.robot_a_qpos.shape != reference.robot_b_qpos.shape:
        raise ValueError("Robot A and B qpos trajectories must have identical shapes")
    if reference.robot_a_qpos.ndim != 2 or reference.robot_a_qpos.shape[1] != ROBOT_QPOS_DIM:
        raise ValueError(f"Expected robot qpos [T, {ROBOT_QPOS_DIM}], got {reference.robot_a_qpos.shape}")
    frames = len(reference.robot_a_qpos)
    if reference.shared_object_qpos.shape != (frames, 7):
        raise ValueError(f"Expected shared object qpos [{frames}, 7], got {reference.shared_object_qpos.shape}")
    if frames < 2 or not isinstance(reference.fps, (int, np.integer)) or reference.fps <= 0:
        raise ValueError("Reference must contain at least two frames and a positive integer FPS")

    resolved_model_path = Path(model_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(resolved_model_path))
    if model.nq != 36 or model.nv != 35 or model.njnt != 30:
        raise ValueError(
            f"Expected floating-base 29-DoF G1 model (nq=36,nv=35,njnt=30), got "
            f"nq={model.nq},nv={model.nv},njnt={model.njnt}"
        )
    joint_names, body_names = _model_names(model)

    robot_qpos = np.stack((reference.robot_a_qpos, reference.robot_b_qpos), axis=1)
    model_qpos = np.stack(
        [
            _reorder_robot_qpos_for_model(robot_qpos[:, agent], source_joint_names, joint_names)
            for agent in range(2)
        ],
        axis=1,
    )
    model_qvel = np.stack(
        [_generalized_velocities_from_qpos(model, model_qpos[:, agent], reference.fps) for agent in range(2)],
        axis=1,
    )

    body_channels = [
        _forward_kinematics_and_velocity(model, model_qpos[:, agent], model_qvel[:, agent])
        for agent in range(2)
    ]
    agent_body_pos_w = np.stack([channels[0] for channels in body_channels], axis=1)
    agent_body_quat_wxyz = np.stack([channels[1] for channels in body_channels], axis=1)
    agent_body_lin_vel_w = np.stack([channels[2] for channels in body_channels], axis=1)
    agent_body_ang_vel_w = np.stack([channels[3] for channels in body_channels], axis=1)

    object_qpos = np.asarray(reference.shared_object_qpos, dtype=np.float64).copy()
    object_qpos[:, 3:7] = _continuous_unit_quaternions_wxyz(object_qpos[:, 3:7])
    object_lin_vel_w, object_ang_vel_w = _pose_world_velocities(object_qpos, reference.fps)
    provenance_payload: dict[str, object] = {
        "generator": "holosoma_retargeting.dual_pull_runtime_reference",
        "schema_version": RUNTIME_REFERENCE_SCHEMA_VERSION,
        "quaternion_convention": "wxyz",
        "velocity_frame": "world",
        "model_path": _portable_path_label(resolved_model_path),
        "model_sha256": sha256_file(resolved_model_path),
        "lateral_axis": "table_local_z",
        "mirror_plane": "table_local_xy",
        "lateral_leg_offset_m": float(reference.lateral_leg_offset_m),
    }
    if provenance is not None:
        provenance_payload.update(dict(provenance))

    result = DualPullRuntimeReference(
        agent_joint_pos=model_qpos[:, :, 7:],
        agent_joint_vel=model_qvel[:, :, 6:],
        agent_body_pos_w=agent_body_pos_w,
        agent_body_quat_wxyz=agent_body_quat_wxyz,
        agent_body_lin_vel_w=agent_body_lin_vel_w,
        agent_body_ang_vel_w=agent_body_ang_vel_w,
        object_pos_w=object_qpos[:, :3],
        object_quat_wxyz=object_qpos[:, 3:7],
        object_lin_vel_w=object_lin_vel_w,
        object_ang_vel_w=object_ang_vel_w,
        fps=int(reference.fps),
        joint_names=joint_names,
        body_names=body_names,
        provenance=provenance_payload,
    )
    _validate_runtime_reference(result)
    return result


def _load_dual_pull_qpos_reference(path: Path) -> DualPullReference:
    with np.load(path, allow_pickle=False) as data:
        required = {"robot_a_qpos", "robot_b_qpos", "shared_object_qpos", "fps", "joint_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Canonical dual Pull file is missing required channels: {sorted(missing)}")
        robot_a_qpos = np.asarray(data["robot_a_qpos"], dtype=np.float64)
        robot_b_qpos = np.asarray(data["robot_b_qpos"], dtype=np.float64)
        shared_object_qpos = np.asarray(data["shared_object_qpos"], dtype=np.float64)
        fps_value = np.asarray(data["fps"])
        joint_names = np.asarray(data["joint_names"]).astype(str)
        lateral_leg_offset = (
            float(np.asarray(data["lateral_leg_offset_m"]).reshape(()))
            if "lateral_leg_offset_m" in data.files
            else 0.0
        )
    if fps_value.size != 1:
        raise ValueError(f"Canonical dual Pull FPS must be scalar, got {fps_value.shape}")
    fps_float = float(fps_value.reshape(()))
    if not np.isfinite(fps_float) or not fps_float.is_integer() or fps_float <= 0.0:
        raise ValueError(f"Canonical dual Pull FPS must be a positive integer, got {fps_float!r}")
    if not np.array_equal(joint_names, G1_29DOF_JOINT_NAMES.astype(str)):
        raise ValueError("Canonical dual Pull joint_names do not match the approved G1 29-DoF contract")
    return DualPullReference(
        robot_a_qpos=robot_a_qpos,
        robot_b_qpos=robot_b_qpos,
        shared_object_qpos=shared_object_qpos,
        fps=int(fps_float),
        lateral_leg_offset_m=lateral_leg_offset,
    )


def build_dual_pull_runtime_reference_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_RUBBER_HAND_G1_XML,
) -> DualPullRuntimeReference:
    """Expand an approved canonical qpos NPZ and save its full runtime NPZ."""
    source = Path(source_path).expanduser().resolve()
    reference = _load_dual_pull_qpos_reference(source)
    result = build_dual_pull_runtime_reference(
        reference,
        model_path=model_path,
        provenance={
            "source_path": _portable_path_label(source),
            "source_sha256": sha256_file(source),
        },
    )
    result.save(output_path)
    return result


__all__ = [
    "DEFAULT_RUBBER_HAND_G1_XML",
    "DualPullRuntimeReference",
    "RUNTIME_REFERENCE_SCHEMA_VERSION",
    "build_dual_pull_runtime_reference",
    "build_dual_pull_runtime_reference_file",
    "sha256_file",
]
