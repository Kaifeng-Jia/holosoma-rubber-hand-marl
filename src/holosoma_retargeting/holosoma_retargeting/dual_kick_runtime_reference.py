"""Expand the accepted mirrored Kick ViSER rollout into the WBT runtime contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from holosoma_retargeting.dual_kick_reference import G1_29DOF_JOINT_NAMES
from holosoma_retargeting.dual_pull_runtime_reference import (
    DEFAULT_RUBBER_HAND_G1_XML,
    DualPullRuntimeReference,
    build_dual_pull_runtime_reference,
    sha256_file,
)


ACCEPTED_MIRRORED_KICK_VISER_SHA256 = (
    "4daada28ae9d530969e7249d8fee4025c303eba65bb516ac787e23e797bc5d85"
)
KICK_RUNTIME_REFERENCE_SCHEMA_VERSION = 1
_CANONICAL_KEYS = {
    "root_pos",
    "root_quat_xyzw",
    "dof_pos",
    "object_pos_w",
    "object_quat_xyzw",
    "fps",
    "provenance",
}

# The serialized contract is common to all paired WBT references.  Keep this
# public Kick name so callers do not need to depend on Pull-specific symbols.
DualKickRuntimeReference = DualPullRuntimeReference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scalar_fps(raw: np.ndarray) -> int:
    values = np.asarray(raw).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Kick FPS must be scalar, got {np.asarray(raw).shape}")
    value = float(values[0])
    if not np.isfinite(value) or value <= 0.0 or not value.is_integer():
        raise ValueError(f"Kick FPS must be a positive integer, got {value!r}")
    return int(value)


def _read_provenance(raw: np.ndarray) -> dict[str, object]:
    values = np.asarray(raw)
    if values.shape != ():
        raise ValueError(f"Kick provenance must be scalar JSON, got {values.shape}")
    decoded = json.loads(str(values.item()))
    if not isinstance(decoded, dict):
        raise ValueError("Kick provenance must decode to an object")
    if decoded.get("layout") != "mirrored":
        raise ValueError("Kick runtime input must use the accepted mirrored layout")
    if decoded.get("agent_joint_mirroring") is not True:
        raise ValueError("Kick runtime input must record physical agent joint mirroring")
    return decoded


def _load_accepted_viser_rollout(path: Path) -> tuple[SimpleNamespace, dict[str, object]]:
    source_hash = _sha256(path)
    if source_hash != ACCEPTED_MIRRORED_KICK_VISER_SHA256:
        raise ValueError(
            "Accepted mirrored Kick ViSER SHA256 mismatch: "
            f"expected {ACCEPTED_MIRRORED_KICK_VISER_SHA256}, got {source_hash}"
        )
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if keys != _CANONICAL_KEYS:
            raise ValueError(
                "Accepted mirrored Kick ViSER file must contain exactly the seven canonical "
                f"channels {sorted(_CANONICAL_KEYS)}, got {sorted(keys)}"
            )
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)
        root_quat_xyzw = np.asarray(data["root_quat_xyzw"], dtype=np.float64)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
        object_pos = np.asarray(data["object_pos_w"], dtype=np.float64)
        object_quat_xyzw = np.asarray(data["object_quat_xyzw"], dtype=np.float64)
        fps = _read_scalar_fps(data["fps"])
        source_provenance = _read_provenance(data["provenance"])

    frames = root_pos.shape[0] if root_pos.ndim == 3 else -1
    expected_shapes = {
        "root_pos": (frames, 2, 3),
        "root_quat_xyzw": (frames, 2, 4),
        "dof_pos": (frames, 2, 29),
        "object_pos_w": (frames, 3),
        "object_quat_xyzw": (frames, 4),
    }
    channels = {
        "root_pos": root_pos,
        "root_quat_xyzw": root_quat_xyzw,
        "dof_pos": dof_pos,
        "object_pos_w": object_pos,
        "object_quat_xyzw": object_quat_xyzw,
    }
    if frames < 3:
        raise ValueError("Accepted mirrored Kick rollout must contain at least three frames")
    for name, expected in expected_shapes.items():
        if channels[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {channels[name].shape}")
        if not np.isfinite(channels[name]).all():
            raise ValueError(f"{name} contains a non-finite value")
    for name in ("root_quat_xyzw", "object_quat_xyzw"):
        norms = np.linalg.norm(channels[name], axis=-1)
        if not np.allclose(norms, 1.0, rtol=1.0e-4, atol=1.0e-4):
            raise ValueError(f"{name} must contain unit quaternions")

    root_quat_wxyz = root_quat_xyzw[..., [3, 0, 1, 2]]
    object_quat_wxyz = object_quat_xyzw[..., [3, 0, 1, 2]]
    qpos = np.concatenate((root_pos, root_quat_wxyz, dof_pos), axis=-1)
    shared_object_qpos = np.concatenate((object_pos, object_quat_wxyz), axis=-1)
    reference = SimpleNamespace(
        robot_a_qpos=qpos[:, 0],
        robot_b_qpos=qpos[:, 1],
        shared_object_qpos=shared_object_qpos,
        fps=fps,
        # Required only by the legacy generic FK implementation; removed from
        # the public Kick provenance below.
        lateral_leg_offset_m=0.0,
    )
    return reference, source_provenance


def build_dual_kick_runtime_reference_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_RUBBER_HAND_G1_XML,
) -> DualKickRuntimeReference:
    """Validate the accepted ViSER artifact and save a complete paired runtime NPZ."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Accepted mirrored Kick ViSER file does not exist: {source}")
    reference, source_provenance = _load_accepted_viser_rollout(source)
    result = build_dual_pull_runtime_reference(
        reference,
        source_joint_names=G1_29DOF_JOINT_NAMES,
        model_path=model_path,
    )
    provenance = dict(result.provenance)
    provenance.pop("lateral_leg_offset_m", None)
    provenance.update(
        {
            "generator": "holosoma_retargeting.dual_kick_runtime_reference",
            "schema_version": KICK_RUNTIME_REFERENCE_SCHEMA_VERSION,
            "source_path": source.name,
            "source_sha256": ACCEPTED_MIRRORED_KICK_VISER_SHA256,
            "source_contract": "accepted_mirrored_kick_viser_seven_channel_v1",
            "source_layout": "mirrored",
            "source_attempt": source_provenance.get("source_attempt"),
            "agent_joint_mirroring": True,
            "mirror_plane": "table_local_xy",
        }
    )
    result = replace(result, provenance=provenance)
    result.save(output_path)
    return result


__all__ = [
    "ACCEPTED_MIRRORED_KICK_VISER_SHA256",
    "DEFAULT_RUBBER_HAND_G1_XML",
    "DualKickRuntimeReference",
    "KICK_RUNTIME_REFERENCE_SCHEMA_VERSION",
    "build_dual_kick_runtime_reference_file",
    "sha256_file",
]
