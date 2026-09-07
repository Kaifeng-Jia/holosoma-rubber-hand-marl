"""CPU-only compatibility checks for the optional interaction reward contract."""

import copy
import hashlib
import json

import numpy as np
import pytest

from holosoma.agents.mappo.core4d_smalltable_evaluation import (
    core4d_smalltable_evaluation_reward_metadata,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (
    CORE4D_SMALLTABLE_INTERACTION_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_MAPPO_VERSION,
    CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
    CORE4D_SMALLTABLE_PHYSICS_CONTRACT,
    CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
    CORE4D_SMALLTABLE_TERMINATION_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
    CORE4D_SMALLTABLE_TRAINING_ROBOT_URDF_SHA256,
    Core4DSmallTablePPO,
    build_core4d_interaction_contract,
    expected_core4d_smalltable_checkpoint_metadata,
    validate_core4d_smalltable_checkpoint,
)


def _contract():
    return {
        "version": CORE4D_SMALLTABLE_INTERACTION_CONTRACT_VERSION,
        "reference_file_sha256": "1" * 64,
        "runtime_reference_sha256": CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
        "object_urdf_sha256": CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
        "training_robot_urdf_sha256": CORE4D_SMALLTABLE_TRAINING_ROBOT_URDF_SHA256,
        "object_points_sha256": "2" * 64,
        "requested_object_points": 100,
        "actual_object_points": 85,
        "num_body_points": 19,
        "sigma": 0.06,
        "weight": 1.0,
    }


def _state(contract=None):
    return {
        "core4d_smalltable_mappo": expected_core4d_smalltable_checkpoint_metadata(
            interaction_contract=contract
        ),
        "actor_model_state_dict": {},
        "critic_model_state_dict": {},
        "actor_optimizer_state_dict": {},
        "critic_optimizer_state_dict": {},
        "actor_obs_normalizer_state_dict": {},
        "critic_obs_normalizer_state_dict": {},
        "iter": 12000,
    }


def _validation_only_learner(contract=None):
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = expected_core4d_smalltable_checkpoint_metadata(
        interaction_contract=contract
    )
    return learner


def test_baseline_metadata_is_exactly_the_existing_12000_format():
    assert expected_core4d_smalltable_checkpoint_metadata() == {
        "version": CORE4D_SMALLTABLE_MAPPO_VERSION,
        "num_agents": 2,
        "actor_obs_dim": 158,
        "critic_obs_dim": 527,
        "action_dim": 29,
        "initialization": "fresh_random_initialization_v1",
        "runtime_reference_sha256": CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
        "object_urdf_sha256": CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
        "training_promotion_sha256": CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
        "physics_contract": CORE4D_SMALLTABLE_PHYSICS_CONTRACT,
        "reward_contract": CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION,
        "termination_contract": CORE4D_SMALLTABLE_TERMINATION_CONTRACT_VERSION,
    }
    assert validate_core4d_smalltable_checkpoint(_state()) == 12000


def test_interaction_metadata_adds_only_the_two_reward_fields():
    baseline = expected_core4d_smalltable_checkpoint_metadata()
    new = expected_core4d_smalltable_checkpoint_metadata(interaction_contract=_contract())
    assert new.pop("interaction_mesh") == _contract()
    assert new.pop("reward_contract") == CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION
    baseline.pop("reward_contract")
    assert new == baseline
    assert validate_core4d_smalltable_checkpoint(_state(_contract())) == 12000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "unreviewed_version"),
        ("reference_file_sha256", "not_a_hash"),
        ("object_points_sha256", "X" * 64),
        ("runtime_reference_sha256", "a" * 64),
        ("object_urdf_sha256", "b" * 64),
        ("training_robot_urdf_sha256", "c" * 64),
        ("requested_object_points", 64),
        ("actual_object_points", 100),
        ("num_body_points", 15),
        ("sigma", 0.1),
        ("weight", 2.0),
        ("weight", True),
    ],
)
def test_unreviewed_contract_values_are_rejected(field, value):
    contract = _contract()
    contract[field] = value
    with pytest.raises(ValueError):
        expected_core4d_smalltable_checkpoint_metadata(interaction_contract=contract)


def test_missing_extra_or_cross_variant_metadata_is_rejected():
    for key in ("weight", "object_points_sha256"):
        contract = _contract()
        contract.pop(key)
        with pytest.raises(ValueError, match="fields"):
            expected_core4d_smalltable_checkpoint_metadata(interaction_contract=contract)
    contract = _contract()
    contract["unreviewed_contact_reward"] = True
    with pytest.raises(ValueError, match="fields"):
        expected_core4d_smalltable_checkpoint_metadata(interaction_contract=contract)
    state = _state()
    state["core4d_smalltable_mappo"]["interaction_mesh"] = _contract()
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_core4d_smalltable_checkpoint(state)
    state = _state(_contract())
    state["core4d_smalltable_mappo"].pop("interaction_mesh")
    with pytest.raises(ValueError):
        validate_core4d_smalltable_checkpoint(state)


def test_resume_requires_exact_variant_and_graph_asset():
    baseline = _validation_only_learner()
    interaction = _validation_only_learner(_contract())
    baseline._validate_training_state(_state())
    interaction._validate_training_state(_state(_contract()))
    with pytest.raises(ValueError, match="resume reward contract mismatch"):
        baseline._validate_training_state(_state(_contract()))
    with pytest.raises(ValueError, match="resume reward contract mismatch"):
        interaction._validate_training_state(_state())
    for field in ("object_points_sha256", "reference_file_sha256"):
        changed = _contract()
        changed[field] = "a" * 64
        with pytest.raises(ValueError, match="resume reward contract mismatch"):
            interaction._validate_training_state(_state(changed))


def test_contract_values_are_copied_and_do_not_mutate_baseline():
    contract = _contract()
    learner = _validation_only_learner(contract)
    contract["object_points_sha256"] = "f" * 64
    learner._validate_training_state(_state(_contract()))
    metadata = expected_core4d_smalltable_checkpoint_metadata()
    metadata["physics_contract"]["object_mass_kg"] = 999
    assert CORE4D_SMALLTABLE_PHYSICS_CONTRACT["object_mass_kg"] == 20.0


def test_interaction_learner_round_trip_keeps_network_and_reward_contract():
    from holosoma.agents.mappo.core4d_smalltable_initialization import (
        initialize_core4d_smalltable_model_bundle,
    )
    from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
        g1_29dof_core4d_smalltable_baseline,
    )

    config = g1_29dof_core4d_smalltable_baseline.algo.config
    contract = _contract()
    first = Core4DSmallTablePPO(
        initialize_core4d_smalltable_model_bundle(config, device="cpu"),
        config,
        num_envs=2,
        num_steps_per_env=2,
        interaction_contract=contract,
    )
    contract["weight"] = 999
    state = first.training_state_dict(iteration=3)
    assert state["core4d_smalltable_mappo"]["interaction_mesh"] == _contract()
    second = Core4DSmallTablePPO(
        initialize_core4d_smalltable_model_bundle(config, device="cpu"),
        config,
        num_envs=2,
        num_steps_per_env=2,
        interaction_contract=_contract(),
    )
    assert second.load_training_state_dict(state) == 3
    assert second.layout.actor_obs_dim == 158
    assert second.layout.action_dim == 29
    state["core4d_smalltable_mappo"]["interaction_mesh"]["weight"] = 999
    assert first.training_state_dict(iteration=4)["core4d_smalltable_mappo"]["interaction_mesh"] == _contract()


def _write_numeric_asset(path, *, tamper_points=False):
    points = np.arange(85 * 3, dtype=np.float64).reshape(85, 3) / 1000
    metadata = _contract()
    metadata.pop("reference_file_sha256")
    metadata.pop("sigma")
    metadata.pop("weight")
    metadata["object_points_sha256"] = hashlib.sha256(points.tobytes()).hexdigest()
    metadata.update(num_frames=687, fps=50, num_agents=2, seed=42)
    if tamper_points:
        points[0, 0] += 0.001
    np.savez_compressed(path, object_points=points, metadata_json=np.asarray(json.dumps(metadata)))
    return metadata


def test_builder_binds_numeric_asset_and_source_geometry_hashes(tmp_path):
    path = tmp_path / "reference.npz"
    metadata = _write_numeric_asset(path)
    contract = build_core4d_interaction_contract(path)
    assert contract["reference_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert contract["object_points_sha256"] == metadata["object_points_sha256"]
    assert contract["weight"] == 1.0
    assert contract["sigma"] == 0.06
    assert validate_core4d_smalltable_checkpoint(_state(contract)) == 12000


def test_builder_rejects_changed_points_without_matching_digest(tmp_path):
    path = tmp_path / "tampered.npz"
    _write_numeric_asset(path, tamper_points=True)
    with pytest.raises(ValueError, match="point SHA-256 mismatch"):
        build_core4d_interaction_contract(path)


@pytest.mark.parametrize("interaction", [False, True])
def test_actor_only_evaluation_labels_training_and_scoring_regimes(interaction):
    state = _state(_contract() if interaction else None)
    labels = core4d_smalltable_evaluation_reward_metadata(state)
    assert labels["training_reward_variant"] == ("interaction_mesh" if interaction else "baseline")
    assert labels["training_reward_contract"] == state["core4d_smalltable_mappo"]["reward_contract"]
    assert labels["evaluation_reward_contract"] == CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION
    assert labels["evaluation_reward_matches_training"] is (not interaction)
    if interaction:
        labels["interaction_mesh_training_contract"]["sigma"] = 99
        assert state["core4d_smalltable_mappo"]["interaction_mesh"]["sigma"] == 0.06


def test_cross_demo_and_incomplete_checks_still_apply_to_new_variant():
    state = _state(_contract())
    contaminated = copy.deepcopy(state)
    contaminated["plan5_mappo"] = {}
    with pytest.raises(ValueError, match="cross-demo metadata"):
        validate_core4d_smalltable_checkpoint(contaminated)
    del state["actor_optimizer_state_dict"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_core4d_smalltable_checkpoint(state)
