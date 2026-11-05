#!/usr/bin/env python3
"""Build the full-body runtime reference for the isolated Demo 3 tug task."""

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
from holosoma_retargeting.tug_of_war_reference import (  # noqa: E402
    load_single_pull_and_synthesize_tug_of_war_reference,
)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tug = load_single_pull_and_synthesize_tug_of_war_reference(source)

    # The object channel is a schema/reset carrier only.  Demo 3 never tracks it
    # after reset; all subsequent table motion must come from physical contact.
    qpos_reference = DualPullReference(
        robot_a_qpos=tug.robot_a_qpos,
        robot_b_qpos=tug.robot_b_qpos,
        shared_object_qpos=tug.object_qpos,
        fps=tug.fps,
        lateral_leg_offset_m=0.0,
    )
    runtime = build_dual_pull_runtime_reference(
        qpos_reference,
        source_joint_names=tug.joint_names,
        provenance={
            "demo": "Demo3Tug",
            "source_path": _portable_path(source),
            "source_sha256": sha256_file(source),
            "opponent_transform": "world_z_half_turn",
            "leg_pair": "diagonally_opposite",
            "table_channel_role": "reset_and_schema_only_not_tracking_target",
            "table_planar_motion": "neutralized_xy_and_heading",
        },
    )
    runtime.save(output)

    with np.load(output, allow_pickle=False) as data:
        provenance = json.loads(str(np.asarray(data["provenance"]).reshape(())))
    report = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "frames": int(runtime.agent_joint_pos.shape[0]),
        "fps": int(runtime.fps),
        "agent_joint_pos_shape": list(runtime.agent_joint_pos.shape),
        "agent_body_pos_shape": list(runtime.agent_body_pos_w.shape),
        "provenance": provenance,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
