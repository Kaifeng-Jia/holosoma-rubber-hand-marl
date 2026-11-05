#!/usr/bin/env python3
"""Expand a 158-D Pull Actor checkpoint to the 164-D Demo 3 interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))

from holosoma.agents.mappo.demo3_checkpoint import (  # noqa: E402
    Demo3TableObservationExpansion,
    expand_actor_checkpoint_for_demo3_table_obs,
    validate_demo3_lossless_expansion,
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


def _actor_mean(checkpoint: dict, actor_input: torch.Tensor) -> torch.Tensor:
    normalizer = checkpoint["actor_obs_normalizer_state_dict"]
    value = (actor_input - normalizer["_mean"]) / (normalizer["_std"] + 1.0e-2)
    state = checkpoint["actor_model_state_dict"]
    for index in (0, 2, 4):
        value = functional.elu(
            functional.linear(
                value,
                state[f"actor_module.module.{index}.weight"],
                state[f"actor_module.module.{index}.bias"],
            )
        )
    return functional.linear(
        value,
        state["actor_module.module.6.weight"],
        state["actor_module.module.6.bias"],
    )


def _output_equivalence(source: dict, converted: dict) -> float:
    spec = Demo3TableObservationExpansion()
    generator = torch.Generator(device="cpu").manual_seed(721)
    source_obs = torch.randn((257, spec.source_dim), generator=generator)
    table_obs = torch.empty((257, spec.table_dim)).uniform_(-2.0, 2.0, generator=generator)
    source_mean = _actor_mean(source, source_obs)
    converted_mean = _actor_mean(converted, torch.cat((source_obs, table_obs), dim=-1))
    return float(torch.max(torch.abs(source_mean - converted_mean)).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_hash = _sha256(args.source)
    if source_hash != args.expected_source_sha256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {args.expected_source_sha256}, got {source_hash}"
        )
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    converted = expand_actor_checkpoint_for_demo3_table_obs(
        source,
        source_sha256=source_hash,
    )
    validate_demo3_lossless_expansion(source, converted)
    max_abs_output_error = _output_equivalence(source, converted)
    if max_abs_output_error > 1.0e-6:
        raise ValueError(f"Converted Actor output error {max_abs_output_error} exceeds 1e-6")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    output_hash = _sha256(args.output)
    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    validate_demo3_lossless_expansion(source, reloaded)
    spec = Demo3TableObservationExpansion()
    manifest = {
        "version": converted["demo3_compatibility"]["version"],
        "source": _portable_path(args.source),
        "source_sha256": source_hash,
        "output": _portable_path(args.output),
        "output_sha256": output_hash,
        "actor_input_before": spec.source_dim,
        "actor_input_after": spec.target_dim,
        "table_observation_dim": spec.table_dim,
        "contains_yaw_rate": False,
        "max_abs_deterministic_output_error": max_abs_output_error,
        "output_tolerance": 1.0e-6,
        "new_actor_columns_zero": True,
        "new_normalizer_statistics_identity": True,
        "critic_reinitialized_by_demo3_loader": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
