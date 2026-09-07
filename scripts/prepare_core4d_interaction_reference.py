#!/usr/bin/env python3
"""Precompute the reviewed small-table graph artifact; never run training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = next(path for path in (*Path(__file__).resolve().parents, Path.cwd()) if (path / "src/holosoma_retargeting").is_dir())
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))

import numpy as np

from holosoma_retargeting.interaction_mesh_training_reference import (
    compile_training_reference, sha256, validate_replay,
)


MOTIONS = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New NPZ path; existing artifacts are never overwritten")
    parser.add_argument("--reference", type=Path, default=MOTIONS / "core4d_smalltable/core4d_pair_runtime_fps50.npz")
    parser.add_argument("--object-urdf", type=Path, default=MOTIONS / "objects_core4d_desk001_small_training.urdf")
    parser.add_argument("--reference-robot-xml", type=Path, default=ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml")
    parser.add_argument("--training-robot-urdf", type=Path, default=ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf")
    parser.add_argument("--robot-config", type=Path, default=ROOT / "src/holosoma/holosoma/config_values/robot.py")
    parser.add_argument("--validation-rollout", type=Path, help="Optional existing complete replay; validation only, never affects reference artifact")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    sidecar = args.output.with_suffix(".diagnostics.json")
    if args.output.exists() or sidecar.exists():
        parser.error("Output or diagnostics already exists; choose a new path")
    arrays, diagnostics = compile_training_reference(
        args.reference, args.object_urdf, args.reference_robot_xml, args.training_robot_urdf, args.robot_config,
    )
    if args.validation_rollout is not None:
        diagnostics["historical_replay_validation"] = validate_replay(
            arrays, args.reference, args.validation_rollout, args.reference_robot_xml,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous artifacts even if the CLI is raced.
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    diagnostics["artifact_path"] = str(args.output.resolve())
    diagnostics["artifact_sha256"] = sha256(args.output)
    with sidecar.open("x") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "artifact": str(args.output), "diagnostics": str(sidecar),
        "shape": list(arrays["reference_points_object"].shape),
        "actual_object_points": len(arrays["object_points"]),
        "fixed_chain_max_error_m": diagnostics["fixed_chain_reference_equivalence_max_m"],
        "q_max_error_m2": diagnostics["q_vs_full_laplacian_max_abs_m2"],
        "artifact_sha256": diagnostics["artifact_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
