#!/usr/bin/env python3
"""Generate the planar-neutral Demo 3 table-leg tug-of-war ViSER preview."""

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
from holosoma_retargeting.tug_of_war_reference import (  # noqa: E402
    load_paired_pull_and_synthesize_tug_of_war_reference,
    load_single_pull_and_synthesize_tug_of_war_reference,
)


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
    table_rotation = quaternion_wxyz_to_matrix(result.object_qpos[:, 3:7])
    robot_a_relative_w = result.robot_a_qpos[:, :3] - result.object_qpos[:, :3]
    robot_b_relative_w = result.robot_b_qpos[:, :3] - result.object_qpos[:, :3]
    robot_a_local = np.einsum(
        "tji,tj->ti",
        table_rotation,
        robot_a_relative_w,
    )
    robot_b_local = np.einsum(
        "tji,tj->ti",
        table_rotation,
        robot_b_relative_w,
    )
    table_heading = table_rotation[:, :2, 0]
    table_heading /= np.linalg.norm(table_heading, axis=-1, keepdims=True)
    return {
        "agent_a_local_x_m": [float(robot_a_local[:, 0].min()), float(robot_a_local[:, 0].max())],
        "agent_b_local_x_m": [float(robot_b_local[:, 0].min()), float(robot_b_local[:, 0].max())],
        "opponent_world_xy_symmetry_max_abs_error_m": float(
            np.max(np.abs(robot_a_relative_w[:, :2] + robot_b_relative_w[:, :2]))
        ),
        "opponent_root_z_max_abs_error_m": float(
            np.max(np.abs(result.robot_a_qpos[:, 2] - result.robot_b_qpos[:, 2]))
        ),
        "planar_table_translation_max_abs_m": float(
            np.max(np.abs(result.object_qpos[:, :2] - result.object_qpos[0, :2]))
        ),
        "planar_table_heading_max_abs_error": float(
            np.max(np.abs(table_heading - table_heading[0]))
        ),
        "retained_table_z_range_m": [
            float(result.object_qpos[:, 2].min()),
            float(result.object_qpos[:, 2].max()),
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pull source NPZ.")
    parser.add_argument(
        "--source-kind",
        choices=("single-pull", "paired-runtime"),
        required=True,
        help="Schema of the source NPZ.",
    )
    parser.add_argument("--output", type=Path, required=True, help="ViSER rollout NPZ to create.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_sha256 = _sha256(source)
    if args.source_kind == "single-pull":
        result = load_single_pull_and_synthesize_tug_of_war_reference(source)
    else:
        result = load_paired_pull_and_synthesize_tug_of_war_reference(source)
    result.save_canonical_rollout(
        output,
        provenance={
            "source": _portable_path(source),
            "source_kind": args.source_kind,
            "source_sha256": source_sha256,
        },
    )
    report = {
        "source": str(source),
        "source_kind": args.source_kind,
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
