"""Build the robot-motion prior for Demo 4 cooperative table rotation.

The accepted Pull 08050 physical replay supplies agent A's complete 29-DoF
motion. Agent B is a strict 180-degree world-Z copy of A around the table
centre. The rectangular table is upright and has one constant pose in every
frame: it is only a reset/schema carrier, never a motion-tracking target.

The target rotation direction is inferred from the source replay's measured
object-contact yaw angular impulse. That scalar is provenance for the later
MARL task; it does not animate the table or rotate either robot reference.
During simulation, all post-reset table motion must arise from physical
contact and the global yaw-progress reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from holosoma_retargeting.dual_a1_layout import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)
ROBOT_DOF = 29
ROBOT_QPOS_DIM = 7 + ROBOT_DOF
MOMENT_EPS = 1.0e-8
CONTACT_IMPULSE_EPS = 1.0e-6
RECTANGULAR_TABLE_ORIGIN_HEIGHT_M = 0.368
WORLD_Z_HALF_TURN = np.diag([-1.0, -1.0, 1.0])
EXPECTED_ROBOT_URDF = "@holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
EXPECTED_OBJECT_URDF = (
    "holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "objects_widetable_plan5_pull_training.urdf"
)


def _continuous_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a quaternion trajectory and remove sign flips."""
    result = np.asarray(quaternion, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"Expected quaternion trajectory [T, 4], got {result.shape}")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise ValueError("Quaternion trajectory contains a non-finite or zero norm")
    result /= norms
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] *= -1.0
    return result


def _constant_upright_table(source_object_qpos: np.ndarray) -> np.ndarray:
    """Return the source initial centre/heading as one fixed upright pose."""
    source_rotation = quaternion_wxyz_to_matrix(source_object_qpos[:, 3:7])
    heading = source_rotation[0, :2, 0]
    heading_norm = np.linalg.norm(heading)
    if not np.isfinite(heading_norm) or heading_norm <= 1.0e-8:
        raise ValueError("Table local X axis has a degenerate world-XY projection")

    local_x = np.array([heading[0] / heading_norm, heading[1] / heading_norm, 0.0])
    local_y = np.array([0.0, 0.0, 1.0])
    local_z = np.cross(local_x, local_y)
    upright_rotation = np.stack((local_x, local_y, local_z), axis=-1)
    upright_quaternion = matrix_to_quaternion_wxyz(upright_rotation)

    pose = np.empty((len(source_object_qpos), 7), dtype=np.float64)
    pose[:, :2] = source_object_qpos[0, :2]
    pose[:, 2] = RECTANGULAR_TABLE_ORIGIN_HEIGHT_M
    pose[:, 3:7] = upright_quaternion
    return pose


def _turn_robot_about_static_table(
    robot_qpos: np.ndarray,
    object_qpos: np.ndarray,
) -> np.ndarray:
    """Create agent B as a strict world-Z half-turn of agent A."""
    turned_position = object_qpos[:, :3] + np.einsum(
        "ij,tj->ti",
        WORLD_Z_HALF_TURN,
        robot_qpos[:, :3] - object_qpos[:, :3],
    )
    robot_rotation = quaternion_wxyz_to_matrix(robot_qpos[:, 3:7])
    turned_rotation = np.einsum("ij,tjk->tik", WORLD_Z_HALF_TURN, robot_rotation)
    turned_quaternion = _continuous_quaternion_wxyz(
        matrix_to_quaternion_wxyz(turned_rotation)
    )
    return np.concatenate(
        (turned_position, turned_quaternion, robot_qpos[:, 7:]), axis=-1
    )


def _planar_pull_moment_proxy(
    robot_a_qpos: np.ndarray,
    robot_b_qpos: np.ndarray,
    object_qpos: np.ndarray,
) -> tuple[float, float]:
    """Return each agent's signed root-displacement moment about the table."""
    object_xy = object_qpos[:, :2]
    moments: list[float] = []
    for robot in (robot_a_qpos, robot_b_qpos):
        relative_xy = robot[:, :2] - object_xy
        lever = relative_xy[0]
        displacement = relative_xy[-1] - relative_xy[0]
        moments.append(float(lever[0] * displacement[1] - lever[1] * displacement[0]))
    return moments[0], moments[1]


def _inferred_rotation_sign(moments: tuple[float, float]) -> float:
    if abs(moments[0]) <= MOMENT_EPS or abs(moments[1]) <= MOMENT_EPS:
        raise ValueError(f"Pull moment proxy is degenerate: {moments}")
    if np.sign(moments[0]) != np.sign(moments[1]):
        raise ValueError(f"Diagonal Pull moment proxies must have the same sign: {moments}")
    return float(np.sign(moments[0] + moments[1]))


def _contact_yaw_angular_impulse(
    contact_position_w: np.ndarray,
    contact_force_on_object_w: np.ndarray,
    object_position_w: np.ndarray,
    *,
    agent_index: int,
    agents: int,
    dt: float,
) -> float:
    """Integrate one agent's object-contact yaw moment in world coordinates."""
    expected_prefix = (len(object_position_w), 1)
    if (
        contact_position_w.ndim != 4
        or contact_position_w.shape[:2] != expected_prefix
        or contact_position_w.shape[-1] != 3
    ):
        raise ValueError(
            "object_robot_contact_pos_w must have shape [T, 1, filters, 3], "
            f"got {contact_position_w.shape}"
        )
    if contact_force_on_object_w.shape != contact_position_w.shape:
        raise ValueError(
            "Object contact force/position channels must have matching shapes, got "
            f"{contact_force_on_object_w.shape} and {contact_position_w.shape}"
        )
    filters = contact_position_w.shape[2]
    if agents <= 0 or filters % agents != 0:
        raise ValueError(
            f"Object contact filters ({filters}) must divide evenly across {agents} agents"
        )
    filters_per_agent = filters // agents
    start = agent_index * filters_per_agent
    stop = start + filters_per_agent
    position = contact_position_w[:, 0, start:stop]
    force = contact_force_on_object_w[:, 0, start:stop]
    valid = np.isfinite(position).all(axis=-1) & np.isfinite(force).all(axis=-1)
    lever = np.where(valid[..., None], position - object_position_w[:, None], 0.0)
    force = np.where(valid[..., None], force, 0.0)
    yaw_moment = np.cross(lever, force)[..., 2].sum(axis=-1)
    return float(yaw_moment.sum() * dt)


@dataclass(frozen=True)
class RotateTableReference:
    """Two Pull motion priors plus one constant reset-only table pose."""

    robot_a_qpos: np.ndarray
    robot_b_qpos: np.ndarray
    object_qpos: np.ndarray
    fps: int
    joint_names: np.ndarray
    signed_target_yaw_degrees: float
    pull_moment_proxy: tuple[float, float]
    rotation_direction_source: str
    source_contact_yaw_angular_impulse_nms: float | None

    def save_canonical_rollout(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> Path:
        """Save the five-channel static-table rollout consumed by ViSER."""
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, object] = {
            "generator": "holosoma_retargeting.rotate_table_reference",
            "schema_version": 2,
            "source_quaternion_convention": "wxyz",
            "canonical_quaternion_convention": "xyzw",
            "layout": "diagonal_pull_pull_force_couple",
            "agent_a_source": "accepted_pull08050_physical_replay_agent0",
            "agent_b_source": "world_z_half_turn_copy_of_agent_a",
            "joint_contract": "identical_29dof_pull_trajectory",
            "expected_robot_urdf": EXPECTED_ROBOT_URDF,
            "expected_object_urdf": EXPECTED_OBJECT_URDF,
            "signed_target_yaw_degrees": self.signed_target_yaw_degrees,
            "rotation_direction_source": self.rotation_direction_source,
            "pull_moment_proxy": list(self.pull_moment_proxy),
            "source_contact_yaw_angular_impulse_nms": (
                self.source_contact_yaw_angular_impulse_nms
            ),
            "table_channel_role": "reset_and_schema_only_not_tracking_target",
            "table_preview": "constant_upright_pose_no_animation",
            "post_reset_table_motion": "physics_contact_only",
        }
        if provenance is not None:
            metadata.update(dict(provenance))
        root_wxyz = np.stack(
            (self.robot_a_qpos[:, 3:7], self.robot_b_qpos[:, 3:7]), axis=1
        )
        np.savez_compressed(
            output,
            root_pos=np.stack(
                (self.robot_a_qpos[:, :3], self.robot_b_qpos[:, :3]), axis=1
            ),
            root_quat_xyzw=root_wxyz[..., [1, 2, 3, 0]],
            dof_pos=np.stack(
                (self.robot_a_qpos[:, 7:], self.robot_b_qpos[:, 7:]), axis=1
            ),
            object_pos_w=self.object_qpos[:, :3],
            object_quat_xyzw=self.object_qpos[:, [4, 5, 6, 3]],
            fps=np.asarray(self.fps, dtype=np.int64),
            joint_names=self.joint_names,
            provenance=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return output


def synthesize_rotate_table_reference(
    robot_qpos: np.ndarray,
    object_qpos: np.ndarray,
    *,
    fps: int,
    joint_names: np.ndarray,
    target_yaw_degrees: float = 90.0,
    rotation_direction: float | None = None,
    rotation_direction_source: str = "root_displacement_proxy",
    source_contact_yaw_angular_impulse_nms: float | None = None,
) -> RotateTableReference:
    """Build two robot priors around one constant upright table pose."""
    robot = np.asarray(robot_qpos, dtype=np.float64)
    source_object = np.asarray(object_qpos, dtype=np.float64)
    names = np.asarray(joint_names)
    frames = len(source_object)
    if robot.shape != (frames, ROBOT_QPOS_DIM):
        raise ValueError(
            f"Expected robot trajectory ({frames}, {ROBOT_QPOS_DIM}), got {robot.shape}"
        )
    if source_object.shape != (frames, 7):
        raise ValueError(f"Expected object trajectory ({frames}, 7), got {source_object.shape}")
    if names.shape != (ROBOT_DOF,):
        raise ValueError(f"Expected {ROBOT_DOF} joint names, got {names.shape}")
    if frames < 2 or not all(
        np.isfinite(values).all() for values in (robot, source_object)
    ):
        raise ValueError("Reference trajectories must contain at least two finite frames")
    if not isinstance(fps, (int, np.integer)) or int(fps) <= 0:
        raise ValueError(f"FPS must be a positive integer, got {fps!r}")

    target_magnitude = float(target_yaw_degrees)
    if (
        not np.isfinite(target_magnitude)
        or target_magnitude <= 0.0
        or target_magnitude > 180.0
    ):
        raise ValueError("target_yaw_degrees must be finite and in (0, 180]")
    if not rotation_direction_source:
        raise ValueError("rotation_direction_source must be non-empty")
    if source_contact_yaw_angular_impulse_nms is not None and not np.isfinite(
        source_contact_yaw_angular_impulse_nms
    ):
        raise ValueError("source_contact_yaw_angular_impulse_nms must be finite")

    agent_a = robot.copy()
    agent_a[:, 3:7] = _continuous_quaternion_wxyz(agent_a[:, 3:7])
    static_object = _constant_upright_table(source_object)
    agent_b = _turn_robot_about_static_table(agent_a, static_object)
    moments = _planar_pull_moment_proxy(agent_a, agent_b, static_object)
    proxy_direction = _inferred_rotation_sign(moments)
    if rotation_direction is None:
        direction = proxy_direction
    else:
        direction_value = float(rotation_direction)
        if not np.isfinite(direction_value) or abs(direction_value) <= MOMENT_EPS:
            raise ValueError("rotation_direction must be finite and non-zero")
        direction = float(np.sign(direction_value))

    return RotateTableReference(
        robot_a_qpos=agent_a,
        robot_b_qpos=agent_b,
        object_qpos=static_object,
        fps=int(fps),
        joint_names=names.copy(),
        signed_target_yaw_degrees=direction * target_magnitude,
        pull_moment_proxy=moments,
        rotation_direction_source=rotation_direction_source,
        source_contact_yaw_angular_impulse_nms=(
            source_contact_yaw_angular_impulse_nms
        ),
    )


def load_physical_pull_rollout_and_synthesize_rotate_table_reference(
    path: str | Path,
    target_yaw_degrees: float = 90.0,
    *,
    agent_index: int = 0,
    rotation_direction: float | None = None,
) -> RotateTableReference:
    """Build Demo 4 from the accepted ground-consistent Pull physical replay."""
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        required = {
            "root_pos",
            "root_quat_xyzw",
            "dof_pos",
            "object_pos_w",
            "object_quat_xyzw",
            "_metadata_json",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Physical Pull rollout is missing channels: {missing}")
        root_position = np.asarray(data["root_pos"], dtype=np.float64)
        root_quaternion_xyzw = np.asarray(data["root_quat_xyzw"], dtype=np.float64)
        joint_position = np.asarray(data["dof_pos"], dtype=np.float64)
        object_position = np.asarray(data["object_pos_w"], dtype=np.float64)
        object_quaternion_xyzw = np.asarray(
            data["object_quat_xyzw"], dtype=np.float64
        )
        has_contact_position = "object_robot_contact_pos_w" in data.files
        has_contact_force = "object_robot_contact_force_matrix_w" in data.files
        if has_contact_position != has_contact_force:
            raise ValueError(
                "Physical Pull rollout must contain both object contact position "
                "and force channels"
            )
        contact_position_w = (
            np.asarray(data["object_robot_contact_pos_w"], dtype=np.float64)
            if has_contact_position
            else None
        )
        contact_force_on_object_w = (
            np.asarray(data["object_robot_contact_force_matrix_w"], dtype=np.float64)
            if has_contact_force
            else None
        )
        metadata_raw = str(np.asarray(data["_metadata_json"]).reshape(()).item())

    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError as error:
        raise ValueError("Physical Pull rollout metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("Physical Pull rollout metadata must be a JSON object")
    fps_value = metadata.get("fps")
    if (
        not isinstance(fps_value, (int, float))
        or not np.isfinite(fps_value)
        or float(fps_value) <= 0.0
        or not float(fps_value).is_integer()
    ):
        raise ValueError(f"Physical Pull rollout has invalid FPS metadata: {fps_value!r}")
    joint_names = np.asarray(metadata.get("dof_names", []))
    if joint_names.shape != (ROBOT_DOF,):
        raise ValueError(
            f"Physical Pull rollout must contain {ROBOT_DOF} dof_names in metadata"
        )

    if root_position.ndim != 3 or root_position.shape[-1] != 3:
        raise ValueError(f"root_pos must have shape [T, A, 3], got {root_position.shape}")
    frames, agents, _ = root_position.shape
    expected_agent_pose = (frames, agents, 4)
    expected_joints = (frames, agents, ROBOT_DOF)
    if root_quaternion_xyzw.shape != expected_agent_pose:
        raise ValueError(
            "root_quat_xyzw must have shape "
            f"{expected_agent_pose}, got {root_quaternion_xyzw.shape}"
        )
    if joint_position.shape != expected_joints:
        raise ValueError(
            f"dof_pos must have shape {expected_joints}, got {joint_position.shape}"
        )
    if object_position.shape != (frames, 3):
        raise ValueError(
            f"object_pos_w must have shape ({frames}, 3), got {object_position.shape}"
        )
    if object_quaternion_xyzw.shape != (frames, 4):
        raise ValueError(
            "object_quat_xyzw must have shape "
            f"({frames}, 4), got {object_quaternion_xyzw.shape}"
        )
    if not isinstance(agent_index, int) or not 0 <= agent_index < agents:
        raise ValueError(
            f"agent_index must select one of {agents} agents, got {agent_index!r}"
        )
    values = (
        root_position,
        root_quaternion_xyzw,
        joint_position,
        object_position,
        object_quaternion_xyzw,
    )
    if frames < 2 or not all(np.isfinite(value).all() for value in values):
        raise ValueError("Physical Pull rollout must contain at least two finite frames")

    root_quaternion_wxyz = root_quaternion_xyzw[:, agent_index, [3, 0, 1, 2]]
    object_quaternion_wxyz = object_quaternion_xyzw[:, [3, 0, 1, 2]]
    contact_yaw_impulse: float | None = None
    if rotation_direction is None:
        if contact_position_w is None or contact_force_on_object_w is None:
            raise ValueError(
                "Physical Pull rollout has no object-contact channels; provide an "
                "explicit rotation_direction"
            )
        contact_metadata = metadata.get("object_robot_contact")
        force_semantics = (
            contact_metadata.get("force_semantics", "")
            if isinstance(contact_metadata, dict)
            else ""
        )
        if (
            not isinstance(force_semantics, str)
            or "force on the object sensor body" not in force_semantics.lower()
        ):
            raise ValueError(
                "Object contact metadata must state that forces act on the object"
            )
        dt_value = metadata.get("dt")
        if (
            not isinstance(dt_value, (int, float))
            or not np.isfinite(dt_value)
            or float(dt_value) <= 0.0
        ):
            raise ValueError(f"Physical Pull rollout has invalid dt metadata: {dt_value!r}")
        metadata_agents = metadata.get("num_agents", agents)
        if not isinstance(metadata_agents, int) or metadata_agents != agents:
            raise ValueError(
                f"Contact metadata num_agents must equal pose agents ({agents})"
            )
        object_physics = metadata.get("object_physics")
        com_pose_b = (
            np.asarray(object_physics.get("com_pose_b"), dtype=np.float64)
            if isinstance(object_physics, dict) and "com_pose_b" in object_physics
            else np.asarray([], dtype=np.float64)
        )
        if com_pose_b.shape != (7,) or not np.isfinite(com_pose_b).all():
            raise ValueError("Object physics metadata must contain finite com_pose_b[7]")
        object_rotation = quaternion_wxyz_to_matrix(object_quaternion_wxyz)
        object_com_position_w = object_position + np.einsum(
            "tij,j->ti", object_rotation, com_pose_b[:3]
        )
        contact_yaw_impulse = _contact_yaw_angular_impulse(
            contact_position_w,
            contact_force_on_object_w,
            object_com_position_w,
            agent_index=agent_index,
            agents=agents,
            dt=float(dt_value),
        )
        if abs(contact_yaw_impulse) <= CONTACT_IMPULSE_EPS:
            raise ValueError(
                "Object contact yaw angular impulse is too small to infer a direction; "
                "provide an explicit rotation_direction"
            )
        resolved_rotation_direction = float(np.sign(contact_yaw_impulse))
        direction_source = "object_robot_contact_yaw_angular_impulse"
    else:
        resolved_rotation_direction = rotation_direction
        direction_source = "explicit_rotation_direction"

    robot_qpos = np.concatenate(
        (
            root_position[:, agent_index],
            root_quaternion_wxyz,
            joint_position[:, agent_index],
        ),
        axis=-1,
    )
    object_qpos = np.concatenate(
        (object_position, object_quaternion_wxyz), axis=-1
    )
    return synthesize_rotate_table_reference(
        robot_qpos,
        object_qpos,
        fps=int(fps_value),
        joint_names=joint_names,
        target_yaw_degrees=target_yaw_degrees,
        rotation_direction=resolved_rotation_direction,
        rotation_direction_source=direction_source,
        source_contact_yaw_angular_impulse_nms=contact_yaw_impulse,
    )


__all__ = [
    "EXPECTED_OBJECT_URDF",
    "EXPECTED_ROBOT_URDF",
    "RECTANGULAR_TABLE_ORIGIN_HEIGHT_M",
    "RotateTableReference",
    "load_physical_pull_rollout_and_synthesize_rotate_table_reference",
    "synthesize_rotate_table_reference",
]
