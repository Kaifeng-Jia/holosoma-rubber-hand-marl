#!/usr/bin/env python3
"""Build Demo 4's complete two-agent MuJoCo-FK runtime reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.dual_pull_reference import DualPullReference  # noqa: E402
from holosoma_retargeting.dual_pull_runtime_reference import (  # noqa: E402
    build_dual_pull_runtime_reference,
    sha256_file,
)
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
    / "demo4_pull_pull_static_table_runtime.npz"
)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _target_yaw(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed > 180.0:
        raise argparse.ArgumentTypeError("must be finite and in (0, 180]")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Accepted Pull 08050 physical replay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Complete PairedMotionReference-compatible runtime NPZ.",
    )
    parser.add_argument(
        "--target-yaw-degrees",
        type=_target_yaw,
        default=90.0,
        help="Metadata-only MARL target magnitude; no table animation is generated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_hash = sha256_file(source)
    reference = load_physical_pull_rollout_and_synthesize_rotate_table_reference(
        source,
        target_yaw_degrees=args.target_yaw_degrees,
    )

    qpos_reference = DualPullReference(
        robot_a_qpos=reference.robot_a_qpos,
        robot_b_qpos=reference.robot_b_qpos,
        shared_object_qpos=reference.object_qpos,
        fps=reference.fps,
        lateral_leg_offset_m=0.0,
    )
    runtime = build_dual_pull_runtime_reference(
        qpos_reference,
        source_joint_names=reference.joint_names,
        provenance={
            "demo": "Demo4Rotate",
            "source_path": _portable_path(source),
            "source_sha256": source_hash,
            "agent_a_source": "accepted_pull08050_physical_replay_agent0",
            "agent_b_source": "world_z_half_turn_copy_of_agent_a",
            "layout": "diagonal_pull_pull_force_couple",
            "signed_target_yaw_degrees": reference.signed_target_yaw_degrees,
            "rotation_direction_source": reference.rotation_direction_source,
            "source_contact_yaw_angular_impulse_nms": (
                reference.source_contact_yaw_angular_impulse_nms
            ),
            "table_channel_role": "reset_and_schema_only_not_tracking_target",
            "table_pose": "constant_upright_all_frames",
            "post_reset_table_motion": "physics_contact_only",
        },
    )
    runtime.save(output)

    report = {
        "source": str(source),
        "source_sha256": source_hash,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "frames": int(runtime.agent_joint_pos.shape[0]),
        "fps": int(runtime.fps),
        "bodies": int(runtime.agent_body_pos_w.shape[2]),
        "signed_target_yaw_degrees": reference.signed_target_yaw_degrees,
        "source_contact_yaw_angular_impulse_nms": (
            reference.source_contact_yaw_angular_impulse_nms
        ),
        "table_position_frame_delta_max_m": float(
            np.max(np.abs(runtime.object_pos_w - runtime.object_pos_w[:1]))
        ),
        "table_quaternion_frame_delta_max": float(
            np.max(np.abs(runtime.object_quat_wxyz - runtime.object_quat_wxyz[:1]))
        ),
        "table_linear_velocity_max_mps": float(
            np.max(np.abs(runtime.object_lin_vel_w))
        ),
        "table_angular_velocity_max_radps": float(
            np.max(np.abs(runtime.object_ang_vel_w))
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
