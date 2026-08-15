#!/usr/bin/env python3
"""Prepare symmetric, translation-only A1 references for Stage 1B evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.dual_a1_layout import (  # noqa: E402
    lateral_offset_trajectory,
    mirror_g1_robot_joint_positions_about_table,
    robot_velocities_from_joint_positions,
    shifted_robot_motion_channels,
)


DEFAULT_SOURCE = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "rubber_hand_largetable_v1"
    / "a1"
    / "sub6_largetable_033_a1_mj_fps50_w_obj.npz"
)
DEFAULT_MUJOCO_XML = (
    REPO_ROOT
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "models"
    / "g1"
    / "g1_29dof_w_largetable.xml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw_frame_planar(vector_w: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(root_quat_wxyz, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.stack(
        (
            cosine * vector_w[:, 0] + sine * vector_w[:, 1],
            -sine * vector_w[:, 0] + cosine * vector_w[:, 1],
        ),
        axis=-1,
    )


def _load_source(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_lin_vel_w",
        "object_pos_w",
        "object_quat_w",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Source motion is missing required channels: {missing}")
    return arrays


def _forward_kinematics_channels(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    mujoco_xml: Path,
) -> dict[str, np.ndarray]:
    """Regenerate all MuJoCo body channels from internally consistent robot state."""
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "Mirrored reference generation requires MuJoCo; run this command in the hsretargeting environment"
        ) from error

    model = mujoco.MjModel.from_xml_path(str(mujoco_xml))
    data = mujoco.MjData(model)
    model_body_names = np.asarray(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(model.nbody)]
    )
    if not np.array_equal(model_body_names, source_arrays["body_names"]):
        raise ValueError("MuJoCo XML body order does not match the source motion")
    model_joint_names = np.asarray(
        [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
            if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
        ]
    )
    if not np.array_equal(model_joint_names, source_arrays["joint_names"]):
        raise ValueError("MuJoCo XML actuated-joint order does not match the source motion")

    frame_count = len(joint_pos)
    body_pos = np.empty((frame_count, model.nbody, 3), dtype=np.float64)
    body_quat = np.empty((frame_count, model.nbody, 4), dtype=np.float64)
    body_lin_vel = np.empty((frame_count, model.nbody, 3), dtype=np.float64)
    body_ang_vel = np.empty((frame_count, model.nbody, 3), dtype=np.float64)
    body_velocity = np.empty((model.nbody, 6), dtype=np.float64)
    for frame in range(frame_count):
        data.qpos[:] = np.concatenate(
            (joint_pos[frame], source_arrays["object_pos_w"][frame], source_arrays["object_quat_w"][frame])
        )
        data.qvel[:] = np.concatenate(
            (
                joint_vel[frame],
                source_arrays["object_lin_vel_w"][frame],
                source_arrays["object_ang_vel_w"][frame],
            )
        )
        mujoco.mj_forward(model, data)
        for body_index in range(model.nbody):
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_index,
                body_velocity[body_index],
                0,
            )
        body_pos[frame] = data.xpos
        body_quat[frame] = data.xquat
        body_ang_vel[frame] = body_velocity[:, :3]
        body_lin_vel[frame] = body_velocity[:, 3:]
    return {
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
    }


def _prepare_side(
    source_arrays: dict[str, np.ndarray],
    destination: Path,
    *,
    side: int,
    spacing: float,
    source_sha256: str,
    object_centered_mean: bool,
    mirror_from_positive_side: bool,
    mujoco_xml: Path,
) -> dict[str, object]:
    fps = int(np.asarray(source_arrays["fps"]).reshape(-1)[0])
    shift_side = 1 if mirror_from_positive_side else side
    shifted = shifted_robot_motion_channels(
        joint_pos=source_arrays["joint_pos"],
        joint_vel=source_arrays["joint_vel"],
        body_pos_w=source_arrays["body_pos_w"],
        body_lin_vel_w=source_arrays["body_lin_vel_w"],
        object_quat_wxyz=source_arrays["object_quat_w"],
        lateral_spacing=spacing,
        observer_side=shift_side,
        fps=fps,
        object_pos_w=source_arrays["object_pos_w"],
        object_centered_mean=object_centered_mean,
    )
    output_arrays = {name: np.array(value, copy=True) for name, value in source_arrays.items()}
    if mirror_from_positive_side:
        mirrored_joint_pos = mirror_g1_robot_joint_positions_about_table(
            shifted["joint_pos"],
            source_arrays["joint_names"],
            source_arrays["object_pos_w"],
            source_arrays["object_quat_w"],
        )
        mirrored_joint_vel = robot_velocities_from_joint_positions(mirrored_joint_pos, fps)
        output_arrays["joint_pos"] = mirrored_joint_pos.astype(source_arrays["joint_pos"].dtype, copy=False)
        output_arrays["joint_vel"] = mirrored_joint_vel.astype(source_arrays["joint_vel"].dtype, copy=False)
        for name, value in _forward_kinematics_channels(
            mirrored_joint_pos,
            mirrored_joint_vel,
            source_arrays,
            mujoco_xml,
        ).items():
            output_arrays[name] = value.astype(source_arrays[name].dtype, copy=False)
    else:
        for name in ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w"):
            output_arrays[name] = shifted[name].astype(source_arrays[name].dtype, copy=False)
    output_arrays["stage1b_observer_side"] = np.asarray(side, dtype=np.int8)
    output_arrays["stage1b_lateral_spacing_m"] = np.asarray(spacing, dtype=np.float64)
    output_arrays["stage1b_source_sha256"] = np.asarray(source_sha256)
    output_arrays["stage1b_object_centered_mean"] = np.asarray(object_centered_mean)
    output_arrays["stage1b_mirrored_from_positive_side"] = np.asarray(mirror_from_positive_side)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output_arrays)

    relative_position_w, relative_velocity_w = lateral_offset_trajectory(
        source_arrays["object_quat_w"],
        2.0 * spacing,
        -side,
        fps,
    )
    root_quat_wxyz = source_arrays["joint_pos"][:, 3:7]
    relative_position_b = _yaw_frame_planar(relative_position_w, root_quat_wxyz)
    relative_velocity_b = _yaw_frame_planar(relative_velocity_w, root_quat_wxyz)
    result = {
        "observer_side": side,
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "frames": int(len(source_arrays["joint_pos"])),
        "fps": fps,
        "mirrored_from_positive_side": mirror_from_positive_side,
        "nominal_reference_teammate_position_min_m": relative_position_b.min(axis=0).tolist(),
        "nominal_reference_teammate_position_max_m": relative_position_b.max(axis=0).tolist(),
        "nominal_reference_teammate_velocity_min_m_s": relative_velocity_b.min(axis=0).tolist(),
        "nominal_reference_teammate_velocity_max_m_s": relative_velocity_b.max(axis=0).tolist(),
    }
    if object_centered_mean:
        result["source_mean_table_local_x_m"] = float(shifted["source_mean_table_local_x_m"])
        result["target_mean_table_local_x_m"] = 0.5 * side * spacing
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/holosoma_stage1b_a1"))
    parser.add_argument("--lateral-spacing", type=float, default=0.8)
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MUJOCO_XML)
    parser.add_argument(
        "--object-centered-mean",
        action="store_true",
        help="Remove the source root's mean table-local-X bias before applying the side target.",
    )
    parser.add_argument(
        "--mirror-left-from-right",
        action="store_true",
        help="Mirror the stable positive-side A1 across the table plane to create the negative-side reference.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.lateral_spacing <= 0.0:
        raise ValueError("lateral-spacing must be positive")
    if args.mirror_left_from_right and not args.object_centered_mean:
        raise ValueError("--mirror-left-from-right requires --object-centered-mean")
    if args.mirror_left_from_right and not args.mujoco_xml.is_file():
        raise FileNotFoundError(args.mujoco_xml)
    source_sha256 = _sha256(args.source)
    source_arrays = _load_source(args.source)

    destinations = {
        -1: args.output_dir / "a1_stage1b_left.npz",
        1: args.output_dir / "a1_stage1b_right.npz",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing files without --overwrite: {existing}")

    sides = [
        _prepare_side(
            source_arrays,
            destinations[side],
            side=side,
            spacing=args.lateral_spacing,
            source_sha256=source_sha256,
            object_centered_mean=args.object_centered_mean,
            mirror_from_positive_side=args.mirror_left_from_right and side == -1,
            mujoco_xml=args.mujoco_xml,
        )
        for side in (-1, 1)
    ]
    manifest = {
        "schema": (
            "holosoma.stage1b_mirrored_a1.v3"
            if args.mirror_left_from_right
            else "holosoma.stage1b_shifted_a1.v2"
            if args.object_centered_mean
            else "holosoma.stage1b_shifted_a1.v1"
        ),
        "source": str(args.source.resolve()),
        "source_sha256": source_sha256,
        "lateral_spacing_m": args.lateral_spacing,
        "translation_only": not args.mirror_left_from_right,
        "object_centered_mean": args.object_centered_mean,
        "mirror_left_from_right": args.mirror_left_from_right,
        "mujoco_xml": str(args.mujoco_xml.resolve()) if args.mirror_left_from_right else None,
        "object_channels_unchanged": True,
        "sides": sides,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
