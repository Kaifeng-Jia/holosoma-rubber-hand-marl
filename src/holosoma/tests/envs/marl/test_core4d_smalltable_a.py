"""5kg transfer preserves old data and reuses the exact A reward arithmetic."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from holosoma.agents.mappo.core4d_smalltable_ppo import expected_core4d_smalltable_checkpoint_metadata
from holosoma.config_values.marl.g1.core4d_pair_experiments import PACKAGE_ROOT, get_core4d_pair_experiment
from holosoma.config_values.marl.g1.core4d_bucket_contract import build_bucket_reward_contract
from holosoma.config_values.marl.g1.core4d_bucket_reward import with_bucket_reward
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_baseline, with_pair_experiment
from holosoma.managers.reward.terms.interaction_vectors import weighted_body_object_vector_error
import torch


def setup():
    old = get_core4d_pair_experiment("smalltable")
    new = get_core4d_pair_experiment("smalltable5kg_A")
    artifact = PACKAGE_ROOT / Path(new.training_promotion_file).parent / "interaction_vectors_v1.npz"
    return old, new, artifact


def test_reference_preserved_and_old_default_unchanged():
    old, new, _ = setup()
    assert old.object_mass_kg == 20 and old.checkpoint_contract is None
    assert new.object_mass_kg == 5 and new.reference_frames == 687
    assert old.runtime_reference_sha256 == new.runtime_reference_sha256
    assert (PACKAGE_ROOT / old.runtime_reference_file).read_bytes() == (PACKAGE_ROOT / new.runtime_reference_file).read_bytes()
    assert old.object_urdf_sha256 == "d4f166913ee6464dae1155428bdfe5169f63be94fbc8869535710bb6b672fc60"


def test_only_mass_and_inertia_changed():
    old, new, _ = setup()
    a, b = [ET.parse(PACKAGE_ROOT / e.object_urdf_file).getroot() for e in (old, new)]
    for root, mass in ((a, 20), (b, 5)):
        assert float(root.find("link/inertial/mass").get("value")) == mass
    for key, value in a.find("link/inertial/inertia").attrib.items():
        assert float(b.find("link/inertial/inertia").get(key)) == pytest.approx(float(value)*.25)
    assert a.find("link/inertial/origin").attrib == b.find("link/inertial/origin").attrib
    for root in (a,b):
        root.find("link").remove(root.find("link/inertial"))
        for element in root.iter():
            element.text = element.tail = None
    assert ET.tostring(a) == ET.tostring(b)


def test_a_contract_and_no_fake_contact_labels():
    _, exp, artifact = setup()
    contract = build_bucket_reward_contract(artifact, "A", experiment=exp)
    assert contract["block_weights_position_rotation_height_relation"] == [1.,1.,1.,2.]
    assert contract["height_sigma_m"] == .1 and contract["relation_sigma_m"] == .04
    assert contract["contact_target"] == "not_used_for_smalltable_A"
    metadata = expected_core4d_smalltable_checkpoint_metadata(experiment_contract=exp.checkpoint_contract, bucket_contract=contract)
    assert metadata["physics_contract"]["object_mass_kg"] == 5
    assert metadata["experiment_contract"]["experiment_id"] == "smalltable5kg_A"
    with np.load(artifact, allow_pickle=False) as z:
        assert z["reference_contact_weights"].shape == (687,2)
        assert not z["reference_contact_weights"].any()
        assert z["reference_body_points_world"].shape == (687,2,19,3)
        assert z["object_points"].shape == (85,3)
        body=torch.tensor(z["reference_body_points_world"][:2])
        obj=torch.tensor(z["reference_object_points_world"][:2,None])
        error,_=weighted_body_object_vector_error(body,body,obj,obj,body_point_weights=torch.tensor(z["body_point_weights"]),distance_floor_m=.1)
        assert torch.allclose(error,torch.zeros_like(error),atol=1e-10)


@pytest.mark.parametrize("variant",["B","A_no_rel","A_no_height"])
def test_unsupported_variants_rejected(variant):
    _, exp, artifact = setup()
    with pytest.raises(ValueError):
        build_bucket_reward_contract(artifact, variant, experiment=exp)


def test_legacy_reward_terms_and_observation_are_preserved():
    _, exp, artifact = setup()
    base = g1_29dof_core4d_smalltable_baseline
    table = with_bucket_reward(with_pair_experiment(base,exp),str(artifact),"A")
    assert table.observation == base.observation
    assert table.termination == base.termination
    assert table.algo == base.algo
    for name, term in base.reward.terms.items():
        if not name.startswith("object_global_ref_"):
            assert table.reward.terms[name] == term
    assert "interaction_mesh" not in table.reward.terms
    assert "object_height_error_penalty" not in table.reward.terms
    assert "bucket_interaction" not in base.reward.terms


def test_cannot_use_bucket_artifact_or_reward_contract_for_table():
    _, exp, artifact = setup()
    bucket = get_core4d_pair_experiment("bucket003")
    other = PACKAGE_ROOT / Path(bucket.training_promotion_file).parent / "interaction_vectors_v1.npz"
    with pytest.raises(ValueError):
        build_bucket_reward_contract(other,"A",experiment=exp)
    contract = build_bucket_reward_contract(artifact,"A",experiment=exp)
    contract["contact_target"] = "source_human_geometry_confidence_not_force_ground_truth"
    with pytest.raises(ValueError):
        expected_core4d_smalltable_checkpoint_metadata(experiment_contract=exp.checkpoint_contract,bucket_contract=contract)


def test_both_removed_contract_changes_only_variant_and_two_weights():
    _, exp, artifact = setup()
    original = build_bucket_reward_contract(artifact, "A", experiment=exp)
    control = build_bucket_reward_contract(artifact, "A_no_rel_no_height", experiment=exp)
    assert control["block_weights_position_rotation_height_relation"] == [1., 1., 0., 0.]
    assert {k for k in original if original[k] != control[k]} == {
        "variant", "block_weights_position_rotation_height_relation"}
    original_meta = expected_core4d_smalltable_checkpoint_metadata(experiment_contract=exp.checkpoint_contract, bucket_contract=original)
    control_meta = expected_core4d_smalltable_checkpoint_metadata(experiment_contract=exp.checkpoint_contract, bucket_contract=control)
    assert original_meta["physics_contract"] == control_meta["physics_contract"]


def test_both_removed_configuration_preserves_every_other_module():
    _, exp, artifact = setup()
    base = with_pair_experiment(g1_29dof_core4d_smalltable_baseline, exp)
    original = with_bucket_reward(base, str(artifact), "A")
    control = with_bucket_reward(base, str(artifact), "A_no_rel_no_height")
    from dataclasses import replace
    assert replace(control, reward=original.reward) == original
    assert original.reward.terms.keys() == control.reward.terms.keys()
    for name, term in original.reward.terms.items():
        other = control.reward.terms[name]
        if name == "bucket_interaction":
            assert other.params["variant"] == "A_no_rel_no_height"
            assert replace(other, params=term.params) == term
            assert {k for k in term.params if term.params[k] != other.params[k]} == {"variant"}
        else:
            assert other == term
