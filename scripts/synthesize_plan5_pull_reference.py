#!/usr/bin/env python3
"""Synthesize the canonical two-agent Plan 5 Pull reference for ViSER."""

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
from holosoma_retargeting.dual_pull_reference import (  # noqa: E402
    synthesize_dual_pull_reference,
)


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative value")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path_label(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _load_attempt_qpos(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"qpos", "fps"}.difference(data.files)
        if missing:
            raise ValueError(f"Pull rollout is missing required channels: {sorted(missing)}")
        qpos = np.asarray(data["qpos"])
        fps_value = np.asarray(data["fps"])

    if fps_value.size != 1:
        raise ValueError(f"Pull rollout FPS must be scalar, got shape {fps_value.shape}")
    fps_float = float(fps_value.reshape(()))
    if not np.isfinite(fps_float) or not fps_float.is_integer() or fps_float <= 0.0:
        raise ValueError(f"Pull rollout FPS must be a positive finite integer, got {fps_float!r}")
    return qpos, int(fps_float)


def _range(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _geometry_report(result) -> dict[str, object]:
    shared_qpos = result.shared_object_qpos
    table_rotation = quaternion_wxyz_to_matrix(shared_qpos[:, 3:7])
    robot_a_local = np.einsum(
        "tji,tj->ti",
        table_rotation,
        result.robot_a_qpos[:, :3] - shared_qpos[:, :3],
    )
    robot_b_local = np.einsum(
        "tji,tj->ti",
        table_rotation,
        result.robot_b_qpos[:, :3] - shared_qpos[:, :3],
    )

    signed_spacing = robot_b_local[:, 2] - robot_a_local[:, 2]
    absolute_spacing = np.abs(signed_spacing)
    initial_table_rotation = table_rotation[0]
    shared_displacement_initial_table = np.einsum(
        "ji,tj->ti",
        initial_table_rotation,
        shared_qpos[:, :3] - shared_qpos[0, :3],
    )
    shared_lateral_drift = shared_displacement_initial_table[:, 2]
    return {
        "robot_a_table_local_z_m": _range(robot_a_local[:, 2]),
        "robot_b_table_local_z_m": _range(robot_b_local[:, 2]),
        "robot_spacing_table_local_z_m": {
            **_range(absolute_spacing),
            "mean": float(np.mean(absolute_spacing)),
            "signed_first": float(signed_spacing[0]),
        },
        "shared_table_lateral_drift_initial_local_z_m": {
            **_range(shared_lateral_drift),
            "max_abs": float(np.max(np.abs(shared_lateral_drift))),
            "final": float(shared_lateral_drift[-1]),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Single-robot attempt NPZ containing qpos [T,43] and scalar fps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Canonical ViSER rollout NPZ to create.",
    )
    parser.add_argument(
        "--qpos-output",
        type=Path,
        default=None,
        help="Optional explicit A/B/shared-object qpos NPZ for the runtime expansion step.",
    )
    parser.add_argument(
        "--lateral-leg-offset-m",
        type=_nonnegative_finite_float,
        required=True,
        help="Outward table-local-Z offset applied to each robot, in metres.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_sha256 = _sha256(source)
    qpos, fps = _load_attempt_qpos(source)
    result = synthesize_dual_pull_reference(
        qpos,
        fps=fps,
        lateral_leg_offset_m=args.lateral_leg_offset_m,
    )
    qpos_output = args.qpos_output.expanduser().resolve() if args.qpos_output is not None else None
    if qpos_output is not None:
        result.save(qpos_output)
    result.save_canonical_rollout(
        output,
        provenance={
            "generator": "scripts/synthesize_plan5_pull_reference.py",
            "source": _portable_path_label(source),
            "source_sha256": source_sha256,
            "lateral_leg_offset_m": args.lateral_leg_offset_m,
        },
    )

    report = {
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "output_sha256": _sha256(output),
        "frames": int(result.robot_a_qpos.shape[0]),
        "fps": result.fps,
        "lateral_leg_offset_m": args.lateral_leg_offset_m,
        **_geometry_report(result),
    }
    if qpos_output is not None:
        report["qpos_output"] = str(qpos_output)
        report["qpos_output_sha256"] = _sha256(qpos_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
