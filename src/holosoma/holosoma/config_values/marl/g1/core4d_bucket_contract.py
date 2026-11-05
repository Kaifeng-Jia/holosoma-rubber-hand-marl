"""Portable identities for paired bucket A/B and single-term ablations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np


BUCKET_DATA_DIR = "holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1"
BUCKET_SOURCE_SHA256 = "ea1e73c80d98ac15602871a543bf1aa6ca221a7039fc8ba98b25b3b09cc094af"
BUCKET_REWARD_VERSION = "core4d_bucket_vectors_soft_contact_v1"
BUCKET_ARTIFACT_VERSION = "core4d_bucket_vectors_v1"
BUCKET_VARIANTS = ("A", "B", "A_no_rel", "A_no_height", "A_no_rel_no_height")
SMALLTABLE_REWARD_VARIANTS = ("A", "A_no_rel_no_height")


def bucket_block_weights(variant: str) -> tuple[float, float, float, float]:
    """Return position, rotation, height and relation weights for a fixed variant.

    Keep the A/B version and complete contract unchanged for existing checkpoints.
    The ablations have distinct variant/weight identities under the same schema.
    Only B enables the contact gate; all variants retain component diagnostics.
    """
    if variant not in BUCKET_VARIANTS:
        raise ValueError(f"Bucket reward variant must be one of {BUCKET_VARIANTS}")
    return (1.0, 1.0, 0.0 if variant in ("A_no_height", "A_no_rel_no_height") else 1.0,
            0.0 if variant in ("A_no_rel", "A_no_rel_no_height") else 2.0)


def bucket_reward_contract(variant: str, artifact_sha256: str, runtime_sha256: str,
                           robot_sha256: str, *, contact_target: str = "source_human_geometry_confidence_not_force_ground_truth") -> dict:
    if contact_target not in ("source_human_geometry_confidence_not_force_ground_truth", "not_used_for_smalltable_A"):
        raise ValueError("Unsupported contact target contract")
    if contact_target == "not_used_for_smalltable_A" and variant not in SMALLTABLE_REWARD_VARIANTS:
        raise ValueError("5kg table supports only A and the ungated both-removed control")
    block_weights = bucket_block_weights(variant)
    for value in (artifact_sha256, runtime_sha256, robot_sha256):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Bucket reward requires lowercase SHA256 identities")
    return {
        "version": BUCKET_REWARD_VERSION, "variant": variant,
        "artifact_sha256": artifact_sha256, "runtime_reference_sha256": runtime_sha256,
        "training_robot_urdf_sha256": robot_sha256,
        "position_sigma_m": 0.3, "position_error_weights_xyz": [1.0, 1.0, 2.0],
        "rotation_sigma_rad": 0.4, "height_sigma_m": 0.10,
        "relation_sigma_m": 0.04, "relation_distance_floor_m": 0.10,
        "block_weights_position_rotation_height_relation": list(block_weights),
        "contact_gate_floor": 0.5, "contact_gate_scale": 2.0,
        "contact_threshold_n": 1.0,
        "contact_error_normalization": "sum(alpha*(1-contact))/max(1,sum(alpha))",
        "contact_detection": "current_physics_sample_any_hand_per_agent_normal_force",
        "contact_target": contact_target,
        "reward_composition": "unchanged_body_and_regularizers_plus_gated_interaction_block",
    }


def validate_bucket_reward_contract(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Missing bucket reward contract")
    expected = bucket_reward_contract(
        value.get("variant"), value.get("artifact_sha256"),
        value.get("runtime_reference_sha256"), value.get("training_robot_urdf_sha256"),
        contact_target=value.get("contact_target"),
    )
    if dict(value) != expected:
        raise ValueError("Bucket reward contract differs from the A/B specification or declared ablation")
    return expected


def build_bucket_reward_contract(path: str | Path, variant: str, *, experiment=None) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    if metadata.get("version") != BUCKET_ARTIFACT_VERSION:
        raise ValueError("Unexpected bucket interaction artifact version")
    artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if experiment is not None:
        if experiment.experiment_id == "smalltable5kg_A" and (
            variant not in SMALLTABLE_REWARD_VARIANTS or metadata.get("contact_target") != "not_used_for_smalltable_A"
        ):
            raise ValueError("5kg table requires an approved ungated variant with no source contact targets")
        package_root = Path(__file__).resolve().parents[4]
        promotion = json.loads((package_root / experiment.training_promotion_file).read_text())
        expected = {
            "runtime_reference_sha256": experiment.runtime_reference_sha256,
            "source_pair_sha256": experiment.source_pair_sha256,
            "object_urdf_sha256": experiment.object_urdf_sha256,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Bucket artifact does not match the selected scene/reference")
        if artifact_hash != promotion.get("interaction_artifact_sha256"):
            raise ValueError("Bucket A/B must use the same artifact frozen in the asset manifest")
    return bucket_reward_contract(
        variant, artifact_hash,
        metadata["runtime_reference_sha256"], metadata["training_robot_urdf_sha256"],
        contact_target=("not_used_for_smalltable_A" if experiment is not None and
                        experiment.experiment_id == "smalltable5kg_A" else
                        "source_human_geometry_confidence_not_force_ground_truth"),
    )
