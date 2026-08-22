#!/usr/bin/env python3
"""Expand an approved dual Pull qpos artifact into the full WBT contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.dual_pull_runtime_reference import (  # noqa: E402
    DEFAULT_RUBBER_HAND_G1_XML,
    build_dual_pull_runtime_reference_file,
    sha256_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Approved NPZ with robot_a_qpos, robot_b_qpos, and shared_object_qpos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Full explicit paired runtime NPZ to create.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    result = build_dual_pull_runtime_reference_file(source, output)
    report = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "model": str(DEFAULT_RUBBER_HAND_G1_XML.resolve()),
        "model_sha256": sha256_file(DEFAULT_RUBBER_HAND_G1_XML),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "frames": int(result.agent_joint_pos.shape[0]),
        "agents": int(result.agent_joint_pos.shape[1]),
        "joints": int(result.agent_joint_pos.shape[2]),
        "bodies": int(result.agent_body_pos_w.shape[2]),
        "fps": result.fps,
        "quaternion_convention": "wxyz",
        "velocity_frame": "world",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
