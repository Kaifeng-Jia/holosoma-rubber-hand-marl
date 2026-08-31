"""Convert one official CORE4D sequence into OmniRetarget source arrays.

This module is deliberately a data-boundary adapter.  It does not call or
modify the OmniRetarget optimizer.  Official CORE4D person files contain a
pickled dictionary inside NPZ; this is the only module that reads that legacy
representation.  Its saved canonical artifact contains numeric/string arrays
only and is always readable with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.config_types.data_type import SMPLX_DEMO_JOINTS


CORE4D_FPS = 15
CORE4D_FULL_JOINT_COUNT = 127

# CORE4D uses a right-handed Y-up world.  OmniRetarget's contact and foot
# utilities use a right-handed Z-up world.  Rx(+90 degrees) maps +Y to +Z.
CORE4D_TO_OMNI_ROTATION = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# First 22 joints in the official SMPL-X ordering used by CORE4D.
SMPLX_BODY_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)
WRIST_BODY_INDICES = np.asarray([20, 21], dtype=np.int64)


def _continuous_quaternion_xyzw(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        raise ValueError("Quaternion data contain a non-finite or zero-norm value")
    result /= norms
    sequences = result.reshape(result.shape[0], -1, 4)
    for frame in range(1, len(sequences)):
        flip = np.sum(sequences[frame - 1] * sequences[frame], axis=-1) < 0.0
        sequences[frame, flip] *= -1.0
    return result


def _validate_fps(fps: int) -> int:
    if not isinstance(fps, (int, np.integer)) or int(fps) <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps!r}")
    return int(fps)


def _load_official_person_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one trusted official CORE4D pickled-NPZ person dictionary."""
    if not path.is_file():
        raise FileNotFoundError(f"CORE4D person motion does not exist: {path}")
    with np.load(path, allow_pickle=True) as archive:  # official format requires pickle
        if set(archive.files) != {"arr_0"}:
            raise ValueError(f"Expected only arr_0 in {path.name}, got {archive.files}")
        container = np.asarray(archive["arr_0"], dtype=object)
        if container.shape != ():
            raise ValueError(f"Expected a scalar person dictionary in {path.name}, got {container.shape}")
        raw = container.item()
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a person dictionary in {path.name}")

    required_shapes = {
        "betas": (10,),
        "joints": (CORE4D_FULL_JOINT_COUNT, 3),
        "global_orient": (3,),
        "body_pose": (21, 3),
    }
    parsed: dict[str, np.ndarray] = {}
    frame_count: int | None = None
    for key, trailing_shape in required_shapes.items():
        if key not in raw:
            raise ValueError(f"{path.name} is missing required field {key!r}")
        value = np.asarray(raw[key], dtype=np.float64)
        if value.ndim != len(trailing_shape) + 1 or value.shape[1:] != trailing_shape:
            raise ValueError(
                f"{path.name}:{key} must have shape [T, {', '.join(map(str, trailing_shape))}], "
                f"got {value.shape}"
            )
        if frame_count is None:
            frame_count = len(value)
        elif len(value) != frame_count:
            raise ValueError(f"{path.name} has inconsistent frame counts")
        if not np.isfinite(value).all():
            raise ValueError(f"{path.name}:{key} contains non-finite values")
        parsed[key] = value
    if frame_count is None or frame_count < 1:
        raise ValueError(f"{path.name} contains no frames")
    if not np.allclose(parsed["betas"], parsed["betas"][0], atol=1.0e-7):
        raise ValueError(f"{path.name}:betas must remain constant across the sequence")
    return parsed


def _global_body_rotations(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
) -> np.ndarray:
    """Return the 22 SMPL-X body rotations in the CORE4D world frame."""
    local_axis_angle = np.concatenate((global_orient[:, None, :], body_pose), axis=1)
    local_rotation = Rotation.from_rotvec(local_axis_angle.reshape(-1, 3)).as_matrix()
    local_rotation = local_rotation.reshape(len(local_axis_angle), 22, 3, 3)
    global_rotation = np.empty_like(local_rotation)
    global_rotation[:, 0] = local_rotation[:, 0]
    for joint_index in range(1, 22):
        parent_index = int(SMPLX_BODY_PARENTS[joint_index])
        global_rotation[:, joint_index] = np.einsum(
            "tij,tjk->tik",
            global_rotation[:, parent_index],
            local_rotation[:, joint_index],
        )
    return global_rotation


def _convert_person(person: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_joints = np.einsum(
        "ij,tkj->tki",
        CORE4D_TO_OMNI_ROTATION,
        np.asarray(person["joints"], dtype=np.float64),
    )
    body_joints = full_joints[:, : len(SMPLX_DEMO_JOINTS)]
    source_global_rotation = _global_body_rotations(
        np.asarray(person["global_orient"], dtype=np.float64),
        np.asarray(person["body_pose"], dtype=np.float64),
    )
    target_wrist_rotation = np.einsum(
        "ij,thjk->thik",
        CORE4D_TO_OMNI_ROTATION,
        source_global_rotation[:, WRIST_BODY_INDICES],
    )
    wrist_quat_xyzw = Rotation.from_matrix(target_wrist_rotation.reshape(-1, 3, 3)).as_quat()
    wrist_quat_xyzw = _continuous_quaternion_xyzw(
        wrist_quat_xyzw.reshape(len(full_joints), 2, 4)
    )
    return body_joints, full_joints, wrist_quat_xyzw


def _convert_object_poses(object_transforms: np.ndarray) -> np.ndarray:
    source = np.asarray(object_transforms, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (4, 4) or len(source) < 1:
        raise ValueError(f"smooth_objposes.npy must have shape [T, 4, 4], got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("smooth_objposes.npy contains non-finite values")
    if not np.allclose(
        source[:, 3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1.0e-7,
    ):
        raise ValueError("CORE4D object transforms must be homogeneous matrices")
    source_rotation = source[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(
        np.einsum("tji,tjk->tik", source_rotation, source_rotation),
        identity,
        atol=1.0e-5,
    ) or not np.allclose(np.linalg.det(source_rotation), 1.0, atol=1.0e-5):
        raise ValueError("CORE4D object transforms contain an invalid rotation")

    coordinate_transform = np.eye(4, dtype=np.float64)
    coordinate_transform[:3, :3] = CORE4D_TO_OMNI_ROTATION
    target = np.einsum("ij,tjk->tik", coordinate_transform, source)
    quaternion_xyzw = Rotation.from_matrix(target[:, :3, :3]).as_quat()
    quaternion_xyzw = _continuous_quaternion_xyzw(quaternion_xyzw)
    return np.concatenate(
        (quaternion_xyzw[:, [3, 0, 1, 2]], target[:, :3, 3]),
        axis=-1,
    )


def _load_aligned_frame_ids(path: Path, expected_frames: int) -> np.ndarray:
    if not path.is_file():
        return np.arange(expected_frames, dtype=np.int64)[:, None]
    rows: list[list[int]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append([int(value) for value in line.split(",")])
        except ValueError as exc:
            raise ValueError(f"Invalid integer on {path.name}:{line_number}") from exc
    if len(rows) != expected_frames:
        raise ValueError(
            f"aligned_frame_ids has {len(rows)} frames, expected {expected_frames}"
        )
    widths = {len(row) for row in rows}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError("aligned_frame_ids must be a non-empty rectangular integer table")
    return np.asarray(rows, dtype=np.int64)


def _resolve_object_mesh(object_model_root: Path, object_name: str) -> Path:
    if len(object_name) <= 3:
        raise ValueError(f"CORE4D obj_name is too short to contain a category: {object_name!r}")
    category = object_name[:-3]
    mesh_path = (object_model_root / category / f"{object_name}_m.obj").resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"CORE4D object mesh does not exist: {mesh_path}")
    return mesh_path


@dataclass(frozen=True)
class Core4DPairSequence:
    """Canonical two-person, one-object source contract for two Omni solves."""

    human_joints: np.ndarray
    human_joints_full: np.ndarray
    betas: np.ndarray
    wrist_quat_xyzw: np.ndarray
    object_poses: np.ndarray
    fps: int
    joint_names: np.ndarray
    aligned_frame_ids: np.ndarray
    object_name: str
    object_mesh_path: str
    provenance: Mapping[str, object]

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            human_joints=self.human_joints,
            human_joints_full=self.human_joints_full,
            betas=self.betas,
            wrist_quat_xyzw=self.wrist_quat_xyzw,
            object_poses=self.object_poses,
            fps=np.asarray(self.fps, dtype=np.int64),
            joint_names=self.joint_names,
            aligned_frame_ids=self.aligned_frame_ids,
            object_name=np.asarray(self.object_name),
            object_mesh_path=np.asarray(self.object_mesh_path),
            human_count=np.asarray(2, dtype=np.int64),
            quaternion_convention=np.asarray("object=wxyz,wrist=xyzw"),
            coordinate_system=np.asarray("right_handed_z_up_meters"),
            provenance_json=np.asarray(json.dumps(dict(self.provenance), sort_keys=True)),
        )
        return output


def _validate_canonical(sequence: Core4DPairSequence) -> None:
    frames = len(sequence.human_joints)
    expected = {
        "human_joints": (frames, 2, len(SMPLX_DEMO_JOINTS), 3),
        "human_joints_full": (frames, 2, CORE4D_FULL_JOINT_COUNT, 3),
        "betas": (2, 10),
        "wrist_quat_xyzw": (frames, 2, 2, 4),
        "object_poses": (frames, 7),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(sequence, name))
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    if frames < 1:
        raise ValueError("CORE4D sequence contains no frames")
    if sequence.aligned_frame_ids.ndim != 2 or len(sequence.aligned_frame_ids) != frames:
        raise ValueError("aligned_frame_ids must be a two-dimensional table with one row per frame")
    if not np.array_equal(sequence.joint_names, np.asarray(SMPLX_DEMO_JOINTS)):
        raise ValueError("joint_names do not match OmniRetarget's SMPL-X body order")
    _validate_fps(sequence.fps)
    if not np.allclose(
        np.linalg.norm(sequence.wrist_quat_xyzw, axis=-1),
        1.0,
        atol=1.0e-7,
    ):
        raise ValueError("wrist_quat_xyzw contains a non-unit quaternion")
    if not np.allclose(
        np.linalg.norm(sequence.object_poses[:, :4], axis=-1),
        1.0,
        atol=1.0e-7,
    ):
        raise ValueError("object_poses contains a non-unit quaternion")


def load_core4d_sequence(
    sequence_dir: str | Path,
    object_model_root: str | Path,
    *,
    fps: int = CORE4D_FPS,
) -> Core4DPairSequence:
    """Load and normalize one trusted official CORE4D real-data sequence."""
    sequence_path = Path(sequence_dir).expanduser().resolve()
    model_root = Path(object_model_root).expanduser().resolve()
    if not sequence_path.is_dir():
        raise FileNotFoundError(f"CORE4D sequence directory does not exist: {sequence_path}")
    fps = _validate_fps(fps)

    people = [
        _load_official_person_npz(sequence_path / "person1_poses.npz"),
        _load_official_person_npz(sequence_path / "person2_poses.npz"),
    ]
    converted_people = [_convert_person(person) for person in people]
    frame_counts = {len(values[0]) for values in converted_people}

    object_matrix_path = sequence_path / "smooth_objposes.npy"
    if not object_matrix_path.is_file():
        raise FileNotFoundError(f"CORE4D object motion does not exist: {object_matrix_path}")
    object_poses = _convert_object_poses(np.load(object_matrix_path, allow_pickle=False))
    frame_counts.add(len(object_poses))
    if len(frame_counts) != 1:
        raise ValueError(
            "CORE4D person1, person2, and object motions must have identical frame counts, "
            f"got {sorted(frame_counts)}"
        )
    frames = next(iter(frame_counts))

    metadata_path = sequence_path / "object_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"CORE4D object metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("obj_name"), str):
        raise ValueError("object_metadata.json must contain a string obj_name")
    object_name = metadata["obj_name"]
    object_mesh_path = _resolve_object_mesh(model_root, object_name)
    alignment_path = sequence_path / "aligned_frame_ids.txt"
    aligned_frame_ids = _load_aligned_frame_ids(
        alignment_path,
        frames,
    )

    sequence = Core4DPairSequence(
        human_joints=np.stack([values[0] for values in converted_people], axis=1),
        human_joints_full=np.stack([values[1] for values in converted_people], axis=1),
        betas=np.stack([person["betas"][0] for person in people], axis=0),
        wrist_quat_xyzw=np.stack([values[2] for values in converted_people], axis=1),
        object_poses=object_poses,
        fps=fps,
        joint_names=np.asarray(SMPLX_DEMO_JOINTS),
        aligned_frame_ids=aligned_frame_ids,
        object_name=object_name,
        object_mesh_path=str(object_mesh_path),
        provenance={
            "adapter": "holosoma_retargeting.data_utils.core4d_adapter",
            "source_sequence": sequence_path.as_posix(),
            "source_format": "CORE4D_Real",
            "source_coordinate_system": "Y_up_meters; handedness_pending_sample_validation",
            "target_coordinate_system": "right_handed_z_up_meters",
            "coordinate_transform": "Rx(+90deg)",
            "coordinate_validation": "pending_first_real_sequence_visual_check",
            "object_mesh_local_frame": "preserved_from_CORE4D",
            "person_order": ["person1", "person2"],
            "aligned_frame_ids_present": alignment_path.is_file(),
            "aligned_frame_ids_semantics": "preserved_only; not interpreted by adapter",
        },
    )
    _validate_canonical(sequence)
    return sequence


def load_canonical_core4d_sequence(path: str | Path) -> Core4DPairSequence:
    """Load a safe adapter artifact without enabling pickle."""
    source = Path(path).expanduser().resolve()
    required = {
        "human_joints",
        "human_joints_full",
        "betas",
        "wrist_quat_xyzw",
        "object_poses",
        "fps",
        "joint_names",
        "aligned_frame_ids",
        "object_name",
        "object_mesh_path",
        "human_count",
        "provenance_json",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Canonical CORE4D artifact is missing fields: {sorted(missing)}")
        if int(np.asarray(archive["human_count"]).reshape(())) != 2:
            raise ValueError("Canonical CORE4D artifact must contain exactly two people")
        sequence = Core4DPairSequence(
            human_joints=np.asarray(archive["human_joints"], dtype=np.float64),
            human_joints_full=np.asarray(archive["human_joints_full"], dtype=np.float64),
            betas=np.asarray(archive["betas"], dtype=np.float64),
            wrist_quat_xyzw=np.asarray(archive["wrist_quat_xyzw"], dtype=np.float64),
            object_poses=np.asarray(archive["object_poses"], dtype=np.float64),
            fps=int(np.asarray(archive["fps"]).reshape(())),
            joint_names=np.asarray(archive["joint_names"], dtype=str),
            aligned_frame_ids=np.asarray(archive["aligned_frame_ids"], dtype=np.int64),
            object_name=str(np.asarray(archive["object_name"]).reshape(())),
            object_mesh_path=str(np.asarray(archive["object_mesh_path"]).reshape(())),
            provenance=json.loads(str(np.asarray(archive["provenance_json"]).reshape(()))),
        )
    _validate_canonical(sequence)
    return sequence


__all__ = [
    "CORE4D_FPS",
    "CORE4D_FULL_JOINT_COUNT",
    "CORE4D_TO_OMNI_ROTATION",
    "Core4DPairSequence",
    "load_canonical_core4d_sequence",
    "load_core4d_sequence",
]
