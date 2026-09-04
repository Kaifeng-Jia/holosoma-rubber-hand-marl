#!/usr/bin/env python3
"""Synthesize the canonical two-agent Plan 5 Kick reference for ViSER."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.dual_a1_layout import quaternion_wxyz_to_matrix  # noqa: E402
from holosoma_retargeting.dual_kick_reference import (  # noqa: E402
    DEFAULT_KICK_LAYOUT,
    KICK_LAYOUTS,
    synthesize_dual_kick_reference_file,
)


DEFAULT_SOURCE_ATTEMPT = 9
DEFAULT_SOURCE_CONTACT_LOCAL_Z_M = -0.23525
DEFAULT_TARGET_CONTACT_HALF_SPAN_M = 0.6742736


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive value")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
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


def _range(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _geometry_report(result: Any) -> dict[str, object]:
    shared_qpos = np.asarray(result.shared_object_qpos)
    table_rotation = quaternion_wxyz_to_matrix(shared_qpos[:, 3:7])
    robot_a_local = np.einsum(
        "tji,tj->ti",
        table_rotation,
        np.asarray(result.robot_a_qpos)[:, :3] - shared_qpos[:, :3],
    )
    robot_b_local = np.einsum(
        "tji,tj->ti",
        table_rotation,
        np.asarray(result.robot_b_qpos)[:, :3] - shared_qpos[:, :3],
    )
    signed_spacing = robot_b_local[:, 2] - robot_a_local[:, 2]
    initial_table_rotation = table_rotation[0]
    object_displacement_initial_local = np.einsum(
        "ji,tj->ti",
        initial_table_rotation,
        shared_qpos[:, :3] - shared_qpos[0, :3],
    )
    source_contact_local_z_m = float(
        getattr(result, "source_contact_local_z_m", DEFAULT_SOURCE_CONTACT_LOCAL_Z_M)
    )
    layout = str(getattr(result, "layout", DEFAULT_KICK_LAYOUT))
    target_contact_half_span_m = float(
        getattr(result, "target_contact_half_span_m", DEFAULT_TARGET_CONTACT_HALF_SPAN_M)
    )
    robot_a_lateral_offset_m = float(
        getattr(
            result,
            "robot_a_lateral_offset_m",
            -target_contact_half_span_m - source_contact_local_z_m,
        )
    )
    robot_b_lateral_offset_m = float(
        getattr(
            result,
            "robot_b_lateral_offset_m",
            (
                target_contact_half_span_m + source_contact_local_z_m
                if layout == "mirrored"
                else target_contact_half_span_m - source_contact_local_z_m
            ),
        )
    )
    return {
        "layout": layout,
        "source_contact_local_z_m": source_contact_local_z_m,
        "target_contact_half_span_m": target_contact_half_span_m,
        "robot_a_lateral_offset_m": robot_a_lateral_offset_m,
        "robot_b_lateral_offset_m": robot_b_lateral_offset_m,
        "target_robot_spacing_m": 2.0 * target_contact_half_span_m,
        "robot_a_table_local_z_m": _range(robot_a_local[:, 2]),
        "robot_b_table_local_z_m": _range(robot_b_local[:, 2]),
        "robot_spacing_table_local_z_m": {
            **_range(np.abs(signed_spacing)),
            "mean": float(np.mean(np.abs(signed_spacing))),
            "signed_first": float(signed_spacing[0]),
        },
        "shared_table_net_displacement_initial_local_m": (
            object_displacement_initial_local[-1].tolist()
        ),
        "shared_table_lateral_drift_initial_local_z_m": {
            **_range(object_displacement_initial_local[:, 2]),
            "max_abs": float(np.max(np.abs(object_displacement_initial_local[:, 2]))),
            "final": float(object_displacement_initial_local[-1, 2]),
        },
    }


def _attempt_report(result: Any, requested_attempt: int) -> dict[str, object]:
    provenance = dict(getattr(result, "provenance", {}) or {})
    report: dict[str, object] = {
        "index": int(getattr(result, "source_attempt", requested_attempt)),
        "indexing": "zero_based",
    }
    aliases = {
        "start": ("source_attempt_start", "attempt_start"),
        "end_exclusive": ("source_attempt_end_exclusive", "attempt_end_exclusive"),
        "end_inclusive": ("source_attempt_end_inclusive", "attempt_end_inclusive", "attempt_end"),
        "length_steps": ("source_attempt_length", "attempt_length", "attempt_length_steps"),
        "completed": ("source_attempt_completed", "attempt_completed"),
        "terminated": ("source_attempt_terminated", "attempt_terminated"),
    }
    for output_name, candidate_names in aliases.items():
        for candidate_name in candidate_names:
            if candidate_name in provenance:
                report[output_name] = provenance[candidate_name]
                break
    if "end_exclusive" in report and "end_inclusive" not in report:
        report["end_inclusive"] = int(report["end_exclusive"]) - 1
    return report


def _resolve_manifest_path(output: Path, manifest: Path | None) -> Path:
    if manifest is not None:
        return manifest.expanduser().resolve()
    return output.with_suffix(".manifest.json")


def _validate_paths(
    *,
    source: Path,
    output: Path,
    qpos_output: Path | None,
    manifest: Path,
    force: bool,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Kick evaluation source does not exist: {source}")
    targets = [output, manifest]
    if qpos_output is not None:
        targets.append(qpos_output)
    if len(set(targets)) != len(targets):
        raise ValueError("Canonical output, qpos output, and manifest must be different paths")
    if source in targets:
        raise ValueError("Source evaluation NPZ cannot also be an output path")
    if not force:
        existing = [path for path in targets if path.exists()]
        if existing:
            paths = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing Kick artifact(s): {paths}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Aggregate single-agent Kick physics-evaluation NPZ.",
    )
    parser.add_argument(
        "--attempt",
        type=_nonnegative_int,
        default=DEFAULT_SOURCE_ATTEMPT,
        help="Zero-based closed-attempt index to extract (default: 9).",
    )
    parser.add_argument(
        "--layout",
        choices=sorted(KICK_LAYOUTS),
        default=DEFAULT_KICK_LAYOUT,
        help=(
            "Second-agent construction: full physical mirror (default) or "
            "the rejected same-action comparison layout."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Canonical seven-channel ViSER rollout NPZ to create.",
    )
    parser.add_argument(
        "--qpos-output",
        type=Path,
        default=None,
        help="Optional explicit robot-A/robot-B/shared-object qpos contract.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON path (default: OUTPUT with .manifest.json suffix).",
    )
    parser.add_argument(
        "--source-contact-local-z-m",
        type=_finite_float,
        default=DEFAULT_SOURCE_CONTACT_LOCAL_Z_M,
        help=(
            "Source rollout contact-lane coordinate in table-local Z "
            f"(default: {DEFAULT_SOURCE_CONTACT_LOCAL_Z_M})."
        ),
    )
    parser.add_argument(
        "--target-contact-half-span-m",
        type=_positive_finite_float,
        default=DEFAULT_TARGET_CONTACT_HALF_SPAN_M,
        help=(
            "Target absolute table-local-Z contact-lane coordinate for each robot "
            f"(default: {DEFAULT_TARGET_CONTACT_HALF_SPAN_M})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly permit replacement of existing output artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    qpos_output = (
        args.qpos_output.expanduser().resolve() if args.qpos_output is not None else None
    )
    manifest_path = _resolve_manifest_path(output, args.manifest)
    _validate_paths(
        source=source,
        output=output,
        qpos_output=qpos_output,
        manifest=manifest_path,
        force=args.force,
    )

    source_sha256 = _sha256(source)
    result = synthesize_dual_kick_reference_file(
        source,
        source_attempt=args.attempt,
        layout=args.layout,
        source_contact_local_z_m=args.source_contact_local_z_m,
        target_contact_half_span_m=args.target_contact_half_span_m,
    )
    layout = str(result.layout)
    agent_joint_mirroring = layout == "mirrored"
    mirror_plane = "table_local_xy" if agent_joint_mirroring else None
    canonical_provenance = {
        "generator": "scripts/synthesize_plan5_kick_reference.py",
        "source": _portable_path_label(source),
        "source_sha256": source_sha256,
        "source_attempt": args.attempt,
        "source_attempt_indexing": "zero_based",
        "layout": layout,
        "agent_joint_mirroring": agent_joint_mirroring,
        "mirror_plane": mirror_plane,
        "source_contact_local_z_m": args.source_contact_local_z_m,
        "target_contact_half_span_m": args.target_contact_half_span_m,
        "robot_a_lateral_offset_m": float(result.robot_a_lateral_offset_m),
        "robot_b_lateral_offset_m": float(result.robot_b_lateral_offset_m),
    }
    result.save_canonical_rollout(output, provenance=canonical_provenance)
    if qpos_output is not None:
        result.save(qpos_output)

    attempt = _attempt_report(result, args.attempt)
    geometry = _geometry_report(result)
    artifacts: dict[str, object] = {
        "canonical_rollout": {
            "path": _portable_path_label(output),
            "sha256": _sha256(output),
        },
    }
    if qpos_output is not None:
        artifacts["qpos_contract"] = {
            "path": _portable_path_label(qpos_output),
            "sha256": _sha256(qpos_output),
        }
    manifest = {
        "schema_version": 1,
        "generator": "scripts/synthesize_plan5_kick_reference.py",
        "source": {
            "path": _portable_path_label(source),
            "sha256": source_sha256,
        },
        "attempt": attempt,
        "layout": layout,
        "agent_joint_mirroring": agent_joint_mirroring,
        "mirror_plane": mirror_plane,
        "frames": int(np.asarray(result.robot_a_qpos).shape[0]),
        "fps": int(result.fps),
        "source_contact_local_z_m": float(
            getattr(result, "source_contact_local_z_m", args.source_contact_local_z_m)
        ),
        "target_contact_half_span_m": float(
            getattr(result, "target_contact_half_span_m", args.target_contact_half_span_m)
        ),
        "quaternion_convention": {
            "qpos_contract": "wxyz",
            "canonical_rollout": "xyzw",
        },
        "artifacts": artifacts,
        "geometry": geometry,
        "provenance": canonical_provenance,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report = {
        **manifest,
        "source_resolved": str(source),
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
