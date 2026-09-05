#!/usr/bin/env python3
"""Build the accepted CORE4D small-table pair's 50 Hz WBT runtime artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.core4d_pair_runtime_reference import (  # noqa: E402
    ACCEPTED_CORE4D_SMALLTABLE_COMPACT_SHA256,
    CORE4D_PAIR_RUNTIME_SCHEMA_VERSION,
    CORE4D_SMALLTABLE_RUNTIME_FPS,
    DEFAULT_RUBBER_HAND_G1_XML,
    build_core4d_pair_runtime_reference_file,
    sha256_file,
)


DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "core4d_smalltable"
)
DEFAULT_SOURCE = DEFAULT_DATA_DIR / "20231030_001_desk001_move_compact_fps30.npz"
DEFAULT_SOURCE_MANIFEST = DEFAULT_DATA_DIR / "source_manifest.json"
DEFAULT_TRAINING_PROMOTION = DEFAULT_DATA_DIR / "training_asset_manifest.json"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "core4d_pair_runtime_fps50.npz"
DEFAULT_OUTPUT_MANIFEST = DEFAULT_DATA_DIR / "core4d_pair_runtime_fps50.manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Source manifest; defaults to source_manifest.json beside --source when present.",
    )
    parser.add_argument(
        "--training-promotion",
        type=Path,
        default=None,
        help=(
            "Reviewed training-asset promotion; defaults to "
            "training_asset_manifest.json beside --source when present."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest; defaults to <output-stem>.manifest.json.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=CORE4D_SMALLTABLE_RUNTIME_FPS,
    )
    return parser.parse_args(argv)


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _default_output_manifest(output: Path) -> Path:
    return output.with_name(f"{output.stem}.manifest.json")


def _validate_training_promotion(
    path: Path,
    *,
    source_sha256: str,
    output_sha256: str,
    source_manifest_sha256: str | None,
) -> None:
    promotion = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "training_ready": True,
        "source_pair_sha256": source_sha256,
        "runtime_reference_sha256": output_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": promotion.get(key)}
        for key, value in expected.items()
        if promotion.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Training-asset promotion mismatch: {mismatches}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_manifest_candidate = source.with_name("source_manifest.json")
    source_manifest = (
        args.source_manifest.expanduser().resolve()
        if args.source_manifest is not None
        else source_manifest_candidate if source_manifest_candidate.is_file() else None
    )
    promotion_candidate = source.with_name("training_asset_manifest.json")
    training_promotion = (
        args.training_promotion.expanduser().resolve()
        if args.training_promotion is not None
        else promotion_candidate if promotion_candidate.is_file() else None
    )
    output_manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else _default_output_manifest(output)
    )

    if source == output:
        raise ValueError("Runtime output must not overwrite the compact CORE4D source")
    for path in (output, output_manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing runtime artifact: {path}")

    runtime = build_core4d_pair_runtime_reference_file(
        source,
        output,
        target_fps=args.target_fps,
        expected_source_sha256=ACCEPTED_CORE4D_SMALLTABLE_COMPACT_SHA256,
        source_manifest_path=source_manifest,
    )
    provenance = dict(runtime.provenance)
    source_sha256 = sha256_file(source)
    output_sha256 = sha256_file(output)
    source_manifest_sha256 = (
        sha256_file(source_manifest) if source_manifest is not None else None
    )
    if training_promotion is not None:
        _validate_training_promotion(
            training_promotion,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            source_manifest_sha256=source_manifest_sha256,
        )
    report = {
        "artifact_kind": "core4d_smalltable_pair_runtime_reference",
        "training_ready": training_promotion is not None,
        "schema_version": CORE4D_PAIR_RUNTIME_SCHEMA_VERSION,
        "source": _portable_path(source),
        "source_sha256": source_sha256,
        "expected_source_sha256": ACCEPTED_CORE4D_SMALLTABLE_COMPACT_SHA256,
        "source_manifest": _portable_path(source_manifest) if source_manifest is not None else None,
        "source_manifest_sha256": source_manifest_sha256,
        "training_promotion": (
            _portable_path(training_promotion)
            if training_promotion is not None
            else None
        ),
        "training_promotion_sha256": (
            sha256_file(training_promotion)
            if training_promotion is not None
            else None
        ),
        "model": _portable_path(DEFAULT_RUBBER_HAND_G1_XML),
        "model_sha256": sha256_file(DEFAULT_RUBBER_HAND_G1_XML),
        "output": _portable_path(output),
        "output_sha256": output_sha256,
        "frames": int(runtime.agent_joint_pos.shape[0]),
        "agents": int(runtime.agent_joint_pos.shape[1]),
        "joints": int(runtime.agent_joint_pos.shape[2]),
        "bodies": int(runtime.agent_body_pos_w.shape[2]),
        "fps": int(runtime.fps),
        "source_frames": int(provenance["source_frames"]),
        "source_fps": int(provenance["source_fps"]),
        "source_duration_seconds": float(provenance["source_duration_seconds"]),
        "sampled_duration_seconds": float(provenance["sampled_duration_seconds"]),
        "omitted_source_tail_seconds": float(provenance["omitted_source_tail_seconds"]),
        "uniform_runtime_timestep_seconds": float(
            provenance["uniform_runtime_timestep_seconds"]
        ),
        "quaternion_convention": "wxyz",
        "velocity_frame": "world",
        "episode_loop": False,
        "terminal_behavior": provenance["terminal_behavior"],
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "manifest": _portable_path(output_manifest)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
