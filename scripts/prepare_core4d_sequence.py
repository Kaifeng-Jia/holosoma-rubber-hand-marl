#!/usr/bin/env python3
"""Convert one trusted official CORE4D sequence to the canonical safe NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.data_utils.core4d_adapter import (  # noqa: E402
    CORE4D_FPS,
    load_canonical_core4d_sequence,
    load_core4d_sequence,
    resample_core4d_pair_sequence,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Directory containing one CORE4D real human-object motion sequence.",
    )
    parser.add_argument(
        "--object-model-root",
        type=Path,
        required=True,
        help="CORE4D_Real/object_models directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Canonical numeric-only NPZ to create.",
    )
    parser.add_argument(
        "--fps",
        type=_positive_int,
        default=CORE4D_FPS,
        help=f"Input motion rate (official processing default: {CORE4D_FPS} FPS).",
    )
    parser.add_argument(
        "--output-fps",
        type=_positive_int,
        default=None,
        help=(
            "Optionally resample the complete paired sequence before saving. "
            "For example, 197 frames at 15 FPS become 393 frames at 30 FPS."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly permit replacement of an existing output artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sequence_dir = args.sequence_dir.expanduser().resolve()
    object_model_root = args.object_model_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing CORE4D artifact: {output}")

    source_sequence = load_core4d_sequence(
        sequence_dir,
        object_model_root,
        fps=args.fps,
    )
    sequence = (
        resample_core4d_pair_sequence(source_sequence, args.output_fps)
        if args.output_fps is not None
        else source_sequence
    )
    sequence.save(output)
    verified = load_canonical_core4d_sequence(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_frames": int(len(source_sequence.human_joints)),
                "source_fps": source_sequence.fps,
                "frames": int(len(verified.human_joints)),
                "fps": verified.fps,
                "duration_seconds": (
                    (len(verified.human_joints) - 1) / verified.fps
                    if len(verified.human_joints) > 1
                    else 0.0
                ),
                "resampled": args.output_fps is not None,
                "human_joints_shape": list(verified.human_joints.shape),
                "human_joints_full_shape": list(verified.human_joints_full.shape),
                "betas_shape": list(verified.betas.shape),
                "human_heights": verified.human_heights.tolist(),
                "height_method": verified.height_method,
                "wrist_quat_xyzw_shape": list(verified.wrist_quat_xyzw.shape),
                "object_poses_shape": list(verified.object_poses.shape),
                "object_name": verified.object_name,
                "object_mesh_path": verified.object_mesh_path,
                "pickle_free_output_verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
