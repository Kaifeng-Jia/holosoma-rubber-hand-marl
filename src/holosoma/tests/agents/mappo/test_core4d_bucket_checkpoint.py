"""Bucket ablations preserve old identities and reject cross-variant resume."""

import copy
import io
from itertools import permutations

import pytest
import torch

from holosoma.agents.mappo.core4d_smalltable_ppo import (
    Core4DSmallTablePPO, expected_core4d_smalltable_checkpoint_metadata,
    validate_core4d_smalltable_checkpoint,
)
from holosoma.agents.mappo.core4d_smalltable_evaluation import core4d_smalltable_evaluation_reward_metadata
from holosoma.config_values.marl.g1.core4d_bucket_contract import (
    BUCKET_REWARD_VERSION,
    BUCKET_VARIANTS,
    bucket_block_weights,
    bucket_reward_contract,
)


def experiment():
    return {
        "experiment_id": "bucket003", "object_name": "bucket003",
        "source_pair_sha256": "1" * 64, "runtime_reference_sha256": "2" * 64,
        "object_urdf_sha256": "3" * 64, "training_promotion_sha256": "4" * 64,
        "reference_frames": 497, "reference_fps": 50,
        "physics_contract": {"object_mass_kg": 1., "material_static_dynamic_restitution": (.5, .5, 0.),
                             "physics_hz": 200, "control_hz": 50, "object_collider_type": "convex_decomposition"},
    }


def state(variant):
    value = {name: {} for name in (
        "actor_model_state_dict", "critic_model_state_dict", "actor_optimizer_state_dict",
        "critic_optimizer_state_dict", "actor_obs_normalizer_state_dict", "critic_obs_normalizer_state_dict",
    )}
    value.update(iter=2000, core4d_smalltable_mappo=expected_core4d_smalltable_checkpoint_metadata(
        experiment_contract=experiment(), bucket_contract=bucket_reward_contract(variant, "5"*64, "2"*64, "6"*64),
    ))
    return value


def legacy_bucket_contract(variant):
    """Frozen pre-ablation literal: do not derive fields from production helpers."""
    assert variant in ("A", "B")
    return {
        "version": "core4d_bucket_vectors_soft_contact_v1",
        "variant": variant,
        "artifact_sha256": "5" * 64,
        "runtime_reference_sha256": "2" * 64,
        "training_robot_urdf_sha256": "6" * 64,
        "position_sigma_m": 0.3,
        "position_error_weights_xyz": [1.0, 1.0, 2.0],
        "rotation_sigma_rad": 0.4,
        "height_sigma_m": 0.10,
        "relation_sigma_m": 0.04,
        "relation_distance_floor_m": 0.10,
        "block_weights_position_rotation_height_relation": [1.0, 1.0, 1.0, 2.0],
        "contact_gate_floor": 0.5,
        "contact_gate_scale": 2.0,
        "contact_threshold_n": 1.0,
        "contact_error_normalization": "sum(alpha*(1-contact))/max(1,sum(alpha))",
        "contact_detection": "current_physics_sample_any_hand_per_agent_normal_force",
        "contact_target": "source_human_geometry_confidence_not_force_ground_truth",
        "reward_composition": "unchanged_body_and_regularizers_plus_gated_interaction_block",
    }


def checkpoint_roundtrip(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    assert restored == value
    return restored


def test_bucket_variant_catalog_is_explicit_and_keeps_legacy_version():
    assert BUCKET_VARIANTS == ("A", "B", "A_no_rel", "A_no_height", "A_no_rel_no_height")
    assert BUCKET_REWARD_VERSION == "core4d_bucket_vectors_soft_contact_v1"


@pytest.mark.parametrize("variant, expected", [
    ("A", (1.0, 1.0, 1.0, 2.0)),
    ("B", (1.0, 1.0, 1.0, 2.0)),
    ("A_no_rel", (1.0, 1.0, 1.0, 0.0)),
    ("A_no_height", (1.0, 1.0, 0.0, 2.0)),
    ("A_no_rel_no_height", (1.0, 1.0, 0.0, 0.0)),
])
def test_bucket_variant_block_weights_are_frozen(variant, expected):
    assert bucket_block_weights(variant) == expected
    contract = bucket_reward_contract(variant, "5" * 64, "2" * 64, "6" * 64)
    assert contract["block_weights_position_rotation_height_relation"] == list(expected)


@pytest.mark.parametrize("variant", BUCKET_VARIANTS)
def test_bucket_roundtrip_and_actor_only_common_evaluation(variant):
    value = checkpoint_roundtrip(state(variant))
    assert validate_core4d_smalltable_checkpoint(value, experiment_contract=experiment()) == 2000
    report = core4d_smalltable_evaluation_reward_metadata(value, experiment_contract=experiment())
    assert report["training_reward_variant"] == f"bucket_{variant}"
    assert report["training_reward_contract"] == "core4d_bucket_vectors_soft_contact_v1"
    assert report["bucket_training_contract"] == value["core4d_smalltable_mappo"]["bucket_reward"]
    assert not report["evaluation_reward_matches_training"]
    assert report["evaluation_reward_contract"] == "plan5_tracking_with_object_v1"
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = value["core4d_smalltable_mappo"]
    learner._validate_training_state(value)


@pytest.mark.parametrize("variant", ("A", "B"))
def test_legacy_literal_contract_and_checkpoint_remain_compatible(variant):
    legacy = legacy_bucket_contract(variant)
    assert bucket_reward_contract(variant, "5" * 64, "2" * 64, "6" * 64) == legacy
    value = state(variant)
    value["core4d_smalltable_mappo"]["bucket_reward"] = legacy
    value["core4d_smalltable_mappo"]["reward_contract"] = "core4d_bucket_vectors_soft_contact_v1"
    restored = checkpoint_roundtrip(value)
    assert validate_core4d_smalltable_checkpoint(restored, experiment_contract=experiment()) == 2000
    report = core4d_smalltable_evaluation_reward_metadata(restored, experiment_contract=experiment())
    assert report["training_reward_variant"] == f"bucket_{variant}"
    assert report["bucket_training_contract"] == legacy
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = state(variant)["core4d_smalltable_mappo"]
    learner._validate_training_state(restored)


def test_a_and_b_only_differ_in_reward_variant_and_cannot_cross_resume():
    a, b = state("A"), state("B")
    altered = copy.deepcopy(a)
    altered["core4d_smalltable_mappo"]["bucket_reward"]["variant"] = "B"
    assert altered == b
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = a["core4d_smalltable_mappo"]
    learner._validate_training_state(a)
    with pytest.raises(ValueError, match="resume reward contract mismatch"):
        learner._validate_training_state(b)


@pytest.mark.parametrize("learner_variant, checkpoint_variant", list(permutations(BUCKET_VARIANTS, 2)))
def test_all_directed_cross_variant_resumes_are_rejected(learner_variant, checkpoint_variant):
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = state(learner_variant)["core4d_smalltable_mappo"]
    incoming = checkpoint_roundtrip(state(checkpoint_variant))
    assert validate_core4d_smalltable_checkpoint(incoming, experiment_contract=experiment()) == 2000
    with pytest.raises(ValueError, match="resume reward contract mismatch"):
        learner._validate_training_state(incoming)


@pytest.mark.parametrize("variant, disabled_index, original_weight", [
    ("A_no_rel", 3, 2.0),
    ("A_no_height", 2, 1.0),
    ("A_no_rel_no_height", 2, 1.0),
    ("A_no_rel_no_height", 3, 2.0),
])
@pytest.mark.parametrize("mutation", [
    "reenable_disabled_term", "remove_disabled_weight", "drop_block_weights",
    "change_variant_to_A", "drop_variant",
])
def test_ablation_contract_tampering_is_rejected(variant, disabled_index, original_weight, mutation):
    value = state(variant)
    contract = value["core4d_smalltable_mappo"]["bucket_reward"]
    key = "block_weights_position_rotation_height_relation"
    if mutation == "reenable_disabled_term":
        contract[key][disabled_index] = original_weight
    elif mutation == "remove_disabled_weight":
        del contract[key][disabled_index]
    elif mutation == "drop_block_weights":
        del contract[key]
    elif mutation == "change_variant_to_A":
        contract["variant"] = "A"
    else:
        del contract["variant"]
    with pytest.raises(ValueError):
        validate_core4d_smalltable_checkpoint(value, experiment_contract=experiment())
    with pytest.raises(ValueError):
        core4d_smalltable_evaluation_reward_metadata(value, experiment_contract=experiment())


def test_missing_or_modified_bucket_semantics_are_rejected():
    with pytest.raises(ValueError, match="explicit.*reward contract"):
        expected_core4d_smalltable_checkpoint_metadata(experiment_contract=experiment())
    value = state("B")
    value["core4d_smalltable_mappo"]["bucket_reward"]["height_sigma_m"] = .05
    with pytest.raises(ValueError, match="specification"):
        validate_core4d_smalltable_checkpoint(value, experiment_contract=experiment())
