#!/usr/bin/env python3
"""Generate Demo 4's two Pull priors with a completely static preview table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.dual_a1_layout import quaternion_wxyz_to_matrix  # noqa: E402
from holosoma_retargeting.rotate_table_reference import (  # noqa: E402
    load_physical_pull_rollout_and_synthesize_rotate_table_reference,
)


DEFAULT_SOURCE = (
    REPO_ROOT
    / "logs"
    / "Plan5Pull"
    / "eval_full8000_seed721"
    / "model_08050_object_centric.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "demo4_rotate"
    / "demo4_pull_pull_static_table_viser.npz"
)


def _target_yaw(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed > 180.0:
        raise argparse.ArgumentTypeError("must be finite and in (0, 180]")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _geometry_report(result) -> dict[str, object]:
    object_rotation = quaternion_wxyz_to_matrix(result.object_qpos[:, 3:7])
    roots = np.stack((result.robot_a_qpos[:, :3], result.robot_b_qpos[:, :3]), axis=1)
    local_roots = np.einsum(
        "tji,taj->tai",
        object_rotation,
        roots - result.object_qpos[:, None, :3],
    )
    local_displacement = local_roots[-1] - local_roots[0]
    quaternion_values = (
        result.robot_a_qpos[:, 3:7],
        result.robot_b_qpos[:, 3:7],
        result.object_qpos[:, 3:7],
    )
    return {
        "signed_target_yaw_degrees": result.signed_target_yaw_degrees,
        "table_position_frame_delta_max_m": float(
            np.max(np.abs(result.object_qpos[:, :3] - result.object_qpos[:1, :3]))
        ),
        "table_quaternion_frame_delta_max": float(
            np.max(np.abs(result.object_qpos[:, 3:7] - result.object_qpos[:1, 3:7]))
        ),
        "initial_robot_roots_table_local_m": local_roots[0].tolist(),
        "pull_root_displacement_table_local_m": local_displacement.tolist(),
        "pull_root_planar_displacement_sum_norm_m": float(
            np.linalg.norm(local_displacement[:, [0, 2]].sum(axis=0))
        ),
        "pull_moment_proxy": list(result.pull_moment_proxy),
        "rotation_direction_source": result.rotation_direction_source,
        "source_contact_yaw_angular_impulse_nms": (
            result.source_contact_yaw_angular_impulse_nms
        ),
        "contact_and_root_proxy_same_sign": bool(
            result.source_contact_yaw_angular_impulse_nms is None
            or np.sign(result.source_contact_yaw_angular_impulse_nms)
            == np.sign(sum(result.pull_moment_proxy))
        ),
        "joint_copy_max_abs_error": float(
            np.max(np.abs(result.robot_a_qpos[:, 7:] - result.robot_b_qpos[:, 7:]))
        ),
        "quaternion_norm_max_abs_error": float(
            max(
                np.max(np.abs(np.linalg.norm(values, axis=-1) - 1.0))
                for values in quaternion_values
            )
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Ground-consistent physical replay of the accepted Pull 08050 actor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Canonical five-channel ViSER NPZ to create.",
    )
    parser.add_argument(
        "--target-yaw-degrees",
        type=_target_yaw,
        default=90.0,
        help=(
            "Unsigned MARL target magnitude. It is metadata only; the preview table "
            "does not rotate."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_sha256 = _sha256(source)
    result = load_physical_pull_rollout_and_synthesize_rotate_table_reference(
        source,
        target_yaw_degrees=args.target_yaw_degrees,
    )
    result.save_canonical_rollout(
        output,
        provenance={
            "script": "scripts/synthesize_demo4_rotate_reference.py",
            "source": _portable_path(source),
            "source_sha256": source_sha256,
            "requested_target_yaw_magnitude_degrees": args.target_yaw_degrees,
            "table_animation": "disabled",
        },
    )
    report = {
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "output_sha256": _sha256(output),
        "frames": len(result.object_qpos),
        "fps": result.fps,
        **_geometry_report(result),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
