#!/usr/bin/env python3
"""Build the full runtime reference from the accepted mirrored Kick ViSER artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.dual_kick_runtime_reference import (  # noqa: E402
    ACCEPTED_MIRRORED_KICK_VISER_SHA256,
    DEFAULT_RUBBER_HAND_G1_XML,
    build_dual_kick_runtime_reference_file,
    sha256_file,
)

DEFAULT_SOURCE = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "rubber_hand_largetable_v1/kick/plan5_attempt09_dual_kick_mirrored_viser.npz"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.with_name("plan5_attempt09_dual_kick_mirrored_runtime.npz")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError("Kick runtime output must not overwrite its accepted ViSER source")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kick runtime reference: {output}")
    result = build_dual_kick_runtime_reference_file(source, output)
    report = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "expected_source_sha256": ACCEPTED_MIRRORED_KICK_VISER_SHA256,
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
        "layout": "mirrored",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
