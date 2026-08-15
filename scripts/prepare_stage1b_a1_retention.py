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

from holosoma_retargeting.dual_a1_layout import shifted_robot_motion_channels  # noqa: E402


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
        "object_quat_w",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Source motion is missing required channels: {missing}")
    return arrays


def _prepare_side(
    source_arrays: dict[str, np.ndarray],
    destination: Path,
    *,
    side: int,
    spacing: float,
    source_sha256: str,
) -> dict[str, object]:
    fps = int(np.asarray(source_arrays["fps"]).reshape(-1)[0])
    shifted = shifted_robot_motion_channels(
        joint_pos=source_arrays["joint_pos"],
        joint_vel=source_arrays["joint_vel"],
        body_pos_w=source_arrays["body_pos_w"],
        body_lin_vel_w=source_arrays["body_lin_vel_w"],
        object_quat_wxyz=source_arrays["object_quat_w"],
        lateral_spacing=spacing,
        observer_side=side,
        fps=fps,
    )
    output_arrays = {name: np.array(value, copy=True) for name, value in source_arrays.items()}
    for name in ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w"):
        output_arrays[name] = shifted[name].astype(source_arrays[name].dtype, copy=False)
    output_arrays["stage1b_observer_side"] = np.asarray(side, dtype=np.int8)
    output_arrays["stage1b_lateral_spacing_m"] = np.asarray(spacing, dtype=np.float64)
    output_arrays["stage1b_source_sha256"] = np.asarray(source_sha256)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output_arrays)

    relative_position_w = -2.0 * shifted["lateral_offset_w"]
    relative_velocity_w = -2.0 * shifted["lateral_offset_velocity_w"]
    root_quat_wxyz = source_arrays["joint_pos"][:, 3:7]
    relative_position_b = _yaw_frame_planar(relative_position_w, root_quat_wxyz)
    relative_velocity_b = _yaw_frame_planar(relative_velocity_w, root_quat_wxyz)
    return {
        "observer_side": side,
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "frames": int(len(source_arrays["joint_pos"])),
        "fps": fps,
        "nominal_reference_teammate_position_min_m": relative_position_b.min(axis=0).tolist(),
        "nominal_reference_teammate_position_max_m": relative_position_b.max(axis=0).tolist(),
        "nominal_reference_teammate_velocity_min_m_s": relative_velocity_b.min(axis=0).tolist(),
        "nominal_reference_teammate_velocity_max_m_s": relative_velocity_b.max(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/holosoma_stage1b_a1"))
    parser.add_argument("--lateral-spacing", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.lateral_spacing <= 0.0:
        raise ValueError("lateral-spacing must be positive")
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
        )
        for side in (-1, 1)
    ]
    manifest = {
        "schema": "holosoma.stage1b_shifted_a1.v1",
        "source": str(args.source.resolve()),
        "source_sha256": source_sha256,
        "lateral_spacing_m": args.lateral_spacing,
        "translation_only": True,
        "object_channels_unchanged": True,
        "sides": sides,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
