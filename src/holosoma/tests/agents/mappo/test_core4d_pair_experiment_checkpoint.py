"""CPU-only scene-contract checks without changing legacy small-table semantics."""

import copy

import pytest
import torch

from holosoma.agents.mappo.core4d_smalltable_evaluation import (
    core4d_smalltable_evaluation_reward_metadata,
)
from holosoma.agents.mappo.core4d_smalltable_initialization import (
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (
    CORE4D_PAIR_MAPPO_VERSION,
    CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_MAPPO_VERSION,
    CORE4D_SMALLTABLE_PHYSICS_CONTRACT,
    CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
    Core4DSmallTablePPO,
    expected_core4d_smalltable_checkpoint_metadata,
    validate_core4d_smalltable_checkpoint,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    g1_29dof_core4d_smalltable_baseline,
)


def _contract():
    return {
        "experiment_id": "chair021",
        "object_name": "chair021",
        "source_pair_sha256": "1" * 64,
        "runtime_reference_sha256": "2" * 64,
        "object_urdf_sha256": "3" * 64,
        "training_promotion_sha256": "4" * 64,
        "reference_frames": 391,
        "reference_fps": 50,
        "physics_contract": {
            "object_mass_kg": 5.0,
            "material_static_dynamic_restitution": (0.5, 0.5, 0.0),
            "physics_hz": 200,
            "control_hz": 50,
            "object_collider_type": "convex_decomposition",
        },
    }


def _state(contract=None):
    return {
        "core4d_smalltable_mappo": expected_core4d_smalltable_checkpoint_metadata(
            experiment_contract=contract,
        ),
        "actor_model_state_dict": {},
        "critic_model_state_dict": {},
        "actor_optimizer_state_dict": {},
        "critic_optimizer_state_dict": {},
        "actor_obs_normalizer_state_dict": {},
        "critic_obs_normalizer_state_dict": {},
        "iter": 2000,
    }


def _validation_only_learner(contract=None):
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = expected_core4d_smalltable_checkpoint_metadata(
        experiment_contract=contract,
    )
    return learner


def test_none_preserves_legacy_metadata_and_evaluation_labels():
    original = expected_core4d_smalltable_checkpoint_metadata()
    explicit_none = expected_core4d_smalltable_checkpoint_metadata(experiment_contract=None)
    assert explicit_none == original
    assert original["version"] == CORE4D_SMALLTABLE_MAPPO_VERSION
    assert original["runtime_reference_sha256"] == CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256
    assert original["physics_contract"] == CORE4D_SMALLTABLE_PHYSICS_CONTRACT
    assert "experiment_contract" not in original
    assert validate_core4d_smalltable_checkpoint(_state(), experiment_contract=None) == 2000
    assert core4d_smalltable_evaluation_reward_metadata(_state(), experiment_contract=None) == (
        core4d_smalltable_evaluation_reward_metadata(_state())
    )


def test_custom_metadata_binds_scene_but_preserves_network_and_objective():
    contract = _contract()
    metadata = expected_core4d_smalltable_checkpoint_metadata(experiment_contract=contract)
    assert metadata["version"] == CORE4D_PAIR_MAPPO_VERSION
    assert metadata["experiment_contract"] == contract
    for key in ("runtime_reference_sha256", "object_urdf_sha256", "training_promotion_sha256", "physics_contract"):
        assert metadata[key] == contract[key]
    assert metadata["num_agents"] == 2
    assert metadata["actor_obs_dim"] == 158
    assert metadata["critic_obs_dim"] == 527
    assert metadata["action_dim"] == 29
    assert metadata["initialization"] == "fresh_random_initialization_v1"
    assert metadata["reward_contract"] == CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION
    assert "interaction_mesh" not in metadata
    assert validate_core4d_smalltable_checkpoint(_state(contract), experiment_contract=contract) == 2000


def test_custom_learner_save_load_and_contract_deep_copy():
    config = g1_29dof_core4d_smalltable_baseline.algo.config
    contract = _contract()
    first = Core4DSmallTablePPO(
        initialize_core4d_smalltable_model_bundle(config, device="cpu"),
        config, num_envs=2, num_steps_per_env=2, device="cpu",
        experiment_contract=contract,
    )
    contract["physics_contract"]["object_mass_kg"] = 100
    contract["source_pair_sha256"] = "f" * 64
    state = first.training_state_dict(iteration=2000)
    assert state["core4d_smalltable_mappo"]["experiment_contract"] == _contract()
    assert state["core4d_smalltable_mappo"]["physics_contract"]["object_mass_kg"] == 5.0
    second = Core4DSmallTablePPO(
        initialize_core4d_smalltable_model_bundle(config, device="cpu"),
        config, num_envs=2, num_steps_per_env=2, device="cpu",
        experiment_contract=_contract(),
    )
    assert second.load_training_state_dict(state) == 2000
    for key, value in first.models.actor.state_dict().items():
        torch.testing.assert_close(value, second.models.actor.state_dict()[key])
    for key, value in first.models.critic.state_dict().items():
        torch.testing.assert_close(value, second.models.critic.state_dict()[key])
    state["core4d_smalltable_mappo"]["experiment_contract"]["physics_contract"]["object_mass_kg"] = 999
    state["core4d_smalltable_mappo"]["physics_contract"]["object_mass_kg"] = 888
    assert first.training_state_dict(iteration=4000)["core4d_smalltable_mappo"] == (
        expected_core4d_smalltable_checkpoint_metadata(experiment_contract=_contract())
    )


def test_chair_and_smalltable_resume_and_evaluation_mutually_reject():
    chair = _state(_contract())
    table = _state()
    with pytest.raises(ValueError, match="explicit experiment_contract"):
        validate_core4d_smalltable_checkpoint(chair)
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_core4d_smalltable_checkpoint(table, experiment_contract=_contract())
    with pytest.raises(ValueError):
        _validation_only_learner()._validate_training_state(chair)
    with pytest.raises(ValueError):
        _validation_only_learner(_contract())._validate_training_state(table)
    with pytest.raises(ValueError):
        core4d_smalltable_evaluation_reward_metadata(chair)
    with pytest.raises(ValueError):
        core4d_smalltable_evaluation_reward_metadata(table, experiment_contract=_contract())


@pytest.mark.parametrize("field", [
    "experiment_id", "object_name", "source_pair_sha256", "runtime_reference_sha256",
    "object_urdf_sha256", "training_promotion_sha256", "reference_frames",
])
def test_resume_rejects_a_different_explicit_scene_contract(field):
    different = _contract()
    if field.endswith("sha256"):
        different[field] = "a" * 64
    elif field == "reference_frames":
        different[field] = 400
    else:
        different[field] = "another_object"
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_core4d_smalltable_checkpoint(_state(_contract()), experiment_contract=different)
    with pytest.raises(ValueError, match="metadata mismatch"):
        _validation_only_learner(different)._validate_training_state(_state(_contract()))


def test_resume_rejects_different_mass_and_inconsistent_top_level_metadata():
    different = _contract()
    different["physics_contract"]["object_mass_kg"] = 6.0
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_core4d_smalltable_checkpoint(_state(_contract()), experiment_contract=different)
    for key, value in (("runtime_reference_sha256", "0" * 64), ("physics_contract", CORE4D_SMALLTABLE_PHYSICS_CONTRACT)):
        state = _state(_contract())
        state["core4d_smalltable_mappo"][key] = value
        with pytest.raises(ValueError, match="metadata mismatch"):
            validate_core4d_smalltable_checkpoint(state, experiment_contract=_contract())


@pytest.mark.parametrize("collider_type", ["convex_hull", "convex_decomposition"])
def test_supported_collider_types_are_valid_but_not_interchangeable_on_resume(collider_type):
    contract = _contract()
    contract["physics_contract"]["object_collider_type"] = collider_type
    state = _state(contract)
    assert validate_core4d_smalltable_checkpoint(state, experiment_contract=contract) == 2000
    other = copy.deepcopy(contract)
    other["physics_contract"]["object_collider_type"] = (
        "convex_decomposition" if collider_type == "convex_hull" else "convex_hull"
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_core4d_smalltable_checkpoint(state, experiment_contract=other)


def test_custom_experiment_does_not_enable_or_accept_an_interaction_graph():
    with pytest.raises(ValueError, match="do not support interaction_mesh"):
        expected_core4d_smalltable_checkpoint_metadata(
            experiment_contract=_contract(), interaction_contract={},
        )
    for change_reward in (False, True):
        state = _state(_contract())
        state["core4d_smalltable_mappo"]["interaction_mesh"] = {}
        if change_reward:
            state["core4d_smalltable_mappo"]["reward_contract"] = CORE4D_SMALLTABLE_INTERACTION_REWARD_CONTRACT_VERSION
        with pytest.raises(ValueError, match="do not support interaction_mesh"):
            validate_core4d_smalltable_checkpoint(state, experiment_contract=_contract())


def test_evaluation_labels_selected_chair_contract_without_changing_scoring():
    state = _state(_contract())
    labels = core4d_smalltable_evaluation_reward_metadata(state, experiment_contract=_contract())
    assert labels["experiment_id"] == "chair021"
    assert labels["object_name"] == "chair021"
    assert labels["experiment_contract"] == _contract()
    assert labels["training_reward_variant"] == "baseline"
    assert labels["evaluation_reward_contract"] == CORE4D_SMALLTABLE_REWARD_CONTRACT_VERSION
    assert labels["evaluation_reward_matches_training"] is True
    assert "interaction_mesh_training_contract" not in labels
    labels["experiment_contract"]["physics_contract"]["object_mass_kg"] = 999
    assert state["core4d_smalltable_mappo"]["experiment_contract"]["physics_contract"]["object_mass_kg"] == 5.0


@pytest.mark.parametrize(("field", "value"), [
    ("experiment_id", ""), ("object_name", 12),
    ("source_pair_sha256", "BAD"), ("runtime_reference_sha256", "Z" * 64),
    ("reference_frames", True), ("reference_frames", 2),
    ("reference_fps", 0), ("reference_fps", 30),
])
def test_experiment_contract_rejects_invalid_fields(field, value):
    contract = _contract()
    contract[field] = value
    with pytest.raises(ValueError):
        expected_core4d_smalltable_checkpoint_metadata(experiment_contract=contract)


@pytest.mark.parametrize(("field", "value"), [
    ("object_mass_kg", float("nan")), ("object_mass_kg", float("inf")),
    ("object_mass_kg", -1), ("object_mass_kg", True),
    ("material_static_dynamic_restitution", (0.5,)),
    ("material_static_dynamic_restitution", (0.5, 0.5, 2.0)),
    ("material_static_dynamic_restitution", (True, 0.5, 0.0)),
    ("physics_hz", 201), ("control_hz", 0),
    ("object_collider_type", "unknown"),
])
def test_experiment_physics_rejects_invalid_values(field, value):
    contract = _contract()
    contract["physics_contract"][field] = value
    with pytest.raises(ValueError):
        expected_core4d_smalltable_checkpoint_metadata(experiment_contract=contract)


def test_extra_missing_and_cross_demo_metadata_remain_rejected():
    for key in ("source_pair_sha256", "physics_contract"):
        contract = _contract()
        del contract[key]
        with pytest.raises(ValueError, match="fields"):
            expected_core4d_smalltable_checkpoint_metadata(experiment_contract=contract)
    contract = _contract()
    contract["free_form_reward"] = "extra"
    with pytest.raises(ValueError, match="fields"):
        expected_core4d_smalltable_checkpoint_metadata(experiment_contract=contract)
    contract = _contract()
    contract["physics_contract"]["hidden_force"] = 1
    with pytest.raises(ValueError, match="fields"):
        expected_core4d_smalltable_checkpoint_metadata(experiment_contract=contract)
    state = _state(_contract())
    state["plan5_mappo"] = {}
    with pytest.raises(ValueError, match="cross-demo metadata"):
        validate_core4d_smalltable_checkpoint(state, experiment_contract=_contract())
    state = _state(_contract())
    del state["actor_optimizer_state_dict"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_core4d_smalltable_checkpoint(state, experiment_contract=_contract())


def test_json_material_list_is_canonicalized_without_mutating_input():
    contract = _contract()
    contract["physics_contract"]["material_static_dynamic_restitution"] = [0.5, 0.5, 0]
    before = copy.deepcopy(contract)
    metadata = expected_core4d_smalltable_checkpoint_metadata(experiment_contract=contract)
    assert contract == before
    assert metadata["experiment_contract"] == _contract()
    metadata["physics_contract"]["object_mass_kg"] = 123
    assert metadata["experiment_contract"]["physics_contract"]["object_mass_kg"] == 5.0
