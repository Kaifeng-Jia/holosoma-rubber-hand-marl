"""Convert the accepted CORE4D compact pair into a 50 Hz WBT runtime reference.

The reviewed small-table artifact stores only floating-base qpos for two G1
robots and one shared object pose.  Runtime training additionally needs joint
velocities plus MuJoCo body poses and spatial velocities.  This module first
resamples the compact 30 Hz signal onto a genuinely uniform 50 Hz grid, then
delegates manifold-aware differentiation and forward kinematics to the
existing paired-reference implementation.

All quaternions are WXYZ at file boundaries.  Playback is explicitly
non-looping: the final 50 Hz sample is the last uniform sample that does not
exceed the source duration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from holosoma_retargeting.dual_pull_reference import G1_29DOF_JOINT_NAMES
from holosoma_retargeting.dual_pull_runtime_reference import (
    DEFAULT_RUBBER_HAND_G1_XML,
    DualPullRuntimeReference,
    build_dual_pull_runtime_reference,
    sha256_file,
)


ACCEPTED_CORE4D_SMALLTABLE_COMPACT_SHA256 = (
    "d9d17e96f5f0b73aa76a1bf32f1ec50cfb7fa7fe78a847dd890882add4bbfd05"
)
CORE4D_SMALLTABLE_SOURCE_FPS = 30
CORE4D_SMALLTABLE_RUNTIME_FPS = 50
CORE4D_PAIR_RUNTIME_SCHEMA_VERSION = 1
ROBOT_QPOS_DIM = 36
OBJECT_QPOS_DIM = 7

# The serialized runtime schema is shared by all paired WBT references.  This
# alias prevents small-table callers from depending on a Pull-specific name.
Core4DPairRuntimeReference = DualPullRuntimeReference


@dataclass(frozen=True)
class Core4DCompactPairReference:
    """Validated compact qpos and the small amount of metadata needed downstream."""

    robot_qpos: np.ndarray
    object_qpos: np.ndarray
    fps: int
    object_name: str | None = None
    shared_object_scale: float | None = None
    source_provenance: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ResampledCore4DPairReference:
    """Two robot qpos streams on one uniform runtime clock."""

    robot_qpos: np.ndarray
    object_qpos: np.ndarray
    source_fps: int
    target_fps: int
    source_frames: int
    target_frames: int
    source_duration_seconds: float
    sampled_duration_seconds: float
    omitted_source_tail_seconds: float


def _positive_integer_scalar(value: np.ndarray, *, name: str) -> int:
    raw = np.asarray(value)
    if raw.size != 1:
        raise ValueError(f"{name} must be scalar, got {raw.shape}")
    number = float(raw.reshape(()))
    integer = int(number)
    if not np.isfinite(number) or number <= 0.0 or number != float(integer):
        raise ValueError(f"{name} must be a positive integer, got {number!r}")
    return integer


def _continuous_unit_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
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


def _validate_compact_reference(reference: Core4DCompactPairReference) -> None:
    robot_qpos = np.asarray(reference.robot_qpos)
    object_qpos = np.asarray(reference.object_qpos)
    if robot_qpos.ndim != 3 or robot_qpos.shape[1:] != (2, ROBOT_QPOS_DIM):
        raise ValueError(
            f"robot_qpos must have shape [T, 2, {ROBOT_QPOS_DIM}], got {robot_qpos.shape}"
        )
    frames = len(robot_qpos)
    if frames < 2:
        raise ValueError("Compact CORE4D reference must contain at least two frames")
    if object_qpos.shape != (frames, OBJECT_QPOS_DIM):
        raise ValueError(
            f"object_qpos must have shape [{frames}, {OBJECT_QPOS_DIM}], got {object_qpos.shape}"
        )
    if not np.isfinite(robot_qpos).all() or not np.isfinite(object_qpos).all():
        raise ValueError("Compact CORE4D reference contains a non-finite value")
    _positive_integer_scalar(np.asarray(reference.fps), name="fps")
    _continuous_unit_quaternions_wxyz(robot_qpos[:, :, 3:7])
    _continuous_unit_quaternions_wxyz(object_qpos[:, 3:7])
    if reference.shared_object_scale is not None:
        scale = float(reference.shared_object_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"shared_object_scale must be positive and finite, got {scale!r}")


def _linear_resample(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    flat = source.reshape(len(source), -1)
    result = np.empty((len(target_times), flat.shape[1]), dtype=np.float64)
    for component in range(flat.shape[1]):
        result[:, component] = np.interp(target_times, source_times, flat[:, component])
    return result.reshape((len(target_times),) + source.shape[1:])


def _slerp_wxyz(
    quaternions: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    source = _continuous_unit_quaternions_wxyz(quaternions)
    flat = source.reshape(len(source), -1, 4)
    result = np.empty((len(target_times), flat.shape[1], 4), dtype=np.float64)
    for sequence_index in range(flat.shape[1]):
        source_xyzw = flat[:, sequence_index, [1, 2, 3, 0]]
        target_xyzw = Slerp(source_times, Rotation.from_quat(source_xyzw))(
            target_times
        ).as_quat()
        result[:, sequence_index] = target_xyzw[:, [3, 0, 1, 2]]
    output_shape = (len(target_times),) + source.shape[1:]
    return _continuous_unit_quaternions_wxyz(result.reshape(output_shape))


def resample_core4d_pair_reference(
    reference: Core4DCompactPairReference,
    *,
    target_fps: int = CORE4D_SMALLTABLE_RUNTIME_FPS,
) -> ResampledCore4DPairReference:
    """Resample compact qpos onto a uniform, non-looping runtime clock.

    For the accepted 413-frame 30 Hz reference, the source ends at
    13.733333... seconds.  A uniform 50 Hz clock therefore contains 687
    samples from 0.00 through 13.72 seconds and omits the final 13.333 ms.
    Inventing a non-uniform final interval would make runtime differentiation
    inconsistent with the advertised 50 Hz contract.
    """
    _validate_compact_reference(reference)
    target_fps = _positive_integer_scalar(np.asarray(target_fps), name="target_fps")
    source_fps = int(reference.fps)
    source_frames = len(reference.robot_qpos)
    source_duration = (source_frames - 1) / float(source_fps)
    target_intervals = int(np.floor(source_duration * target_fps + 1.0e-12))
    target_frames = target_intervals + 1
    source_times = np.arange(source_frames, dtype=np.float64) / float(source_fps)
    target_times = np.arange(target_frames, dtype=np.float64) / float(target_fps)

    robot = np.asarray(reference.robot_qpos, dtype=np.float64)
    obj = np.asarray(reference.object_qpos, dtype=np.float64)
    target_robot = np.empty((target_frames, 2, ROBOT_QPOS_DIM), dtype=np.float64)
    target_robot[:, :, :3] = _linear_resample(robot[:, :, :3], source_times, target_times)
    target_robot[:, :, 3:7] = _slerp_wxyz(robot[:, :, 3:7], source_times, target_times)
    target_robot[:, :, 7:] = _linear_resample(robot[:, :, 7:], source_times, target_times)
    target_object = np.empty((target_frames, OBJECT_QPOS_DIM), dtype=np.float64)
    target_object[:, :3] = _linear_resample(obj[:, :3], source_times, target_times)
    target_object[:, 3:7] = _slerp_wxyz(obj[:, 3:7], source_times, target_times)

    sampled_duration = target_times[-1]
    omitted_tail = source_duration - sampled_duration
    if omitted_tail < -1.0e-12 or omitted_tail >= 1.0 / target_fps + 1.0e-12:
        raise RuntimeError(f"Invalid uniform-grid tail duration: {omitted_tail}")
    return ResampledCore4DPairReference(
        robot_qpos=target_robot,
        object_qpos=target_object,
        source_fps=source_fps,
        target_fps=target_fps,
        source_frames=source_frames,
        target_frames=target_frames,
        source_duration_seconds=source_duration,
        sampled_duration_seconds=sampled_duration,
        omitted_source_tail_seconds=max(0.0, omitted_tail),
    )


def _read_optional_json_scalar(raw: np.ndarray, *, name: str) -> dict[str, object]:
    value = np.asarray(raw)
    if value.shape != ():
        raise ValueError(f"{name} must be scalar JSON, got {value.shape}")
    decoded = json.loads(str(value.item()))
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return decoded


def load_core4d_compact_pair_reference(path: str | Path) -> Core4DCompactPairReference:
    """Load the pickle-free small-table compact contract."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CORE4D compact pair does not exist: {source}")
    with np.load(source, allow_pickle=False) as data:
        required = {"robot_qpos", "object_qpos", "fps"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"CORE4D compact pair is missing fields: {sorted(missing)}")
        fps = _positive_integer_scalar(data["fps"], name="fps")
        object_name = str(np.asarray(data["object_name"]).item()) if "object_name" in data.files else None
        shared_scale = (
            float(np.asarray(data["shared_object_scale"]).reshape(()))
            if "shared_object_scale" in data.files
            else None
        )
        source_provenance = (
            _read_optional_json_scalar(data["provenance_json"], name="provenance_json")
            if "provenance_json" in data.files
            else None
        )
        result = Core4DCompactPairReference(
            robot_qpos=np.asarray(data["robot_qpos"], dtype=np.float64).copy(),
            object_qpos=np.asarray(data["object_qpos"], dtype=np.float64).copy(),
            fps=fps,
            object_name=object_name,
            shared_object_scale=shared_scale,
            source_provenance=source_provenance,
        )
    _validate_compact_reference(result)
    return result


def _validated_source_manifest(path: Path, source_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"CORE4D source manifest does not exist: {path}")
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("CORE4D source manifest must contain a JSON object")
    manifest_hash = decoded.get("pair_sha256")
    if manifest_hash != source_sha256:
        raise ValueError(
            "CORE4D source manifest pair_sha256 does not match the compact source: "
            f"{manifest_hash!r} != {source_sha256!r}"
        )
    return decoded


def build_core4d_pair_runtime_reference(
    reference: Core4DCompactPairReference,
    *,
    target_fps: int = CORE4D_SMALLTABLE_RUNTIME_FPS,
    model_path: str | Path = DEFAULT_RUBBER_HAND_G1_XML,
    provenance: Mapping[str, object] | None = None,
) -> Core4DPairRuntimeReference:
    """Resample qpos and reuse MuJoCo FK/differentiation for the runtime channels."""
    resampled = resample_core4d_pair_reference(reference, target_fps=target_fps)
    qpos_reference = SimpleNamespace(
        robot_a_qpos=resampled.robot_qpos[:, 0],
        robot_b_qpos=resampled.robot_qpos[:, 1],
        shared_object_qpos=resampled.object_qpos,
        fps=resampled.target_fps,
        # Required only by the shared FK builder; omitted from final provenance.
        lateral_leg_offset_m=0.0,
    )
    runtime = build_dual_pull_runtime_reference(
        qpos_reference,
        source_joint_names=G1_29DOF_JOINT_NAMES,
        model_path=model_path,
    )
    runtime_provenance = dict(runtime.provenance)
    for pull_specific_key in ("lateral_axis", "mirror_plane", "lateral_leg_offset_m"):
        runtime_provenance.pop(pull_specific_key, None)
    runtime_provenance.update(
        {
            "generator": "holosoma_retargeting.core4d_pair_runtime_reference",
            "schema_version": CORE4D_PAIR_RUNTIME_SCHEMA_VERSION,
            "source_contract": "core4d_smalltable_compact_pair_qpos_v1",
            "source_joint_order": "G1_29DOF_JOINT_NAMES",
            "source_quaternion_convention": "wxyz",
            "source_frames": resampled.source_frames,
            "source_fps": resampled.source_fps,
            "source_duration_seconds": resampled.source_duration_seconds,
            "target_frames": resampled.target_frames,
            "target_fps": resampled.target_fps,
            "sampled_duration_seconds": resampled.sampled_duration_seconds,
            "omitted_source_tail_seconds": resampled.omitted_source_tail_seconds,
            "translation_and_joint_interpolation": "linear",
            "quaternion_interpolation": "SLERP",
            "uniform_runtime_timestep_seconds": 1.0 / resampled.target_fps,
            "episode_loop": False,
            "terminal_behavior": "stop_at_last_reference_frame_then_environment_reset",
        }
    )
    if reference.object_name is not None:
        runtime_provenance["object_name"] = reference.object_name
    if reference.shared_object_scale is not None:
        runtime_provenance["shared_object_scale"] = reference.shared_object_scale
    if reference.source_provenance is not None:
        runtime_provenance["source_provenance_kind"] = reference.source_provenance.get("kind")
        runtime_provenance["source_robot_reference"] = reference.source_provenance.get(
            "robot_reference"
        )
    if provenance is not None:
        runtime_provenance.update(dict(provenance))
    result = replace(runtime, provenance=runtime_provenance)
    return result


def build_core4d_pair_runtime_reference_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_fps: int = CORE4D_SMALLTABLE_RUNTIME_FPS,
    model_path: str | Path = DEFAULT_RUBBER_HAND_G1_XML,
    expected_source_sha256: str | None = ACCEPTED_CORE4D_SMALLTABLE_COMPACT_SHA256,
    source_manifest_path: str | Path | None = None,
) -> Core4DPairRuntimeReference:
    """Validate, expand, and serialize one compact CORE4D pair artifact."""
    source = Path(source_path).expanduser().resolve()
    source_hash = sha256_file(source)
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError(
            "Accepted CORE4D small-table SHA256 mismatch: "
            f"expected {expected_source_sha256}, got {source_hash}"
        )
    manifest_provenance: dict[str, object] = {}
    if source_manifest_path is not None:
        source_manifest = Path(source_manifest_path).expanduser().resolve()
        _validated_source_manifest(source_manifest, source_hash)
        manifest_provenance = {
            "source_manifest_path": source_manifest.name,
            "source_manifest_sha256": sha256_file(source_manifest),
        }
    compact = load_core4d_compact_pair_reference(source)
    if compact.fps != CORE4D_SMALLTABLE_SOURCE_FPS:
        raise ValueError(
            f"Accepted CORE4D small-table source must be {CORE4D_SMALLTABLE_SOURCE_FPS} Hz, "
            f"got {compact.fps} Hz"
        )
    runtime = build_core4d_pair_runtime_reference(
        compact,
        target_fps=target_fps,
        model_path=model_path,
        provenance={
            "source_path": source.name,
            "source_sha256": source_hash,
            **manifest_provenance,
        },
    )
    runtime.save(output_path)
    return runtime


__all__ = [
    "ACCEPTED_CORE4D_SMALLTABLE_COMPACT_SHA256",
    "CORE4D_PAIR_RUNTIME_SCHEMA_VERSION",
    "CORE4D_SMALLTABLE_RUNTIME_FPS",
    "CORE4D_SMALLTABLE_SOURCE_FPS",
    "Core4DCompactPairReference",
    "Core4DPairRuntimeReference",
    "ResampledCore4DPairReference",
    "build_core4d_pair_runtime_reference",
    "build_core4d_pair_runtime_reference_file",
    "load_core4d_compact_pair_reference",
    "resample_core4d_pair_reference",
    "sha256_file",
]
