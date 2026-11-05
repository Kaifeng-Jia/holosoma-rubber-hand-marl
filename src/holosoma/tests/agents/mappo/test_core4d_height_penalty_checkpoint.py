"""CPU-only checkpoint/evaluation contracts for the optional root-height penalty."""

import copy

import pytest
import torch

from holosoma.agents.mappo.core4d_smalltable_evaluation import (
    core4d_smalltable_evaluation_reward_metadata,
    deterministic_core4d_smalltable_actions,
)
from holosoma.agents.mappo.core4d_smalltable_initialization import (
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (
    CORE4D_OBJECT_HEIGHT_PENALTY_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_INTERACTION_CONTRACT_VERSION,
    CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
    CORE4D_SMALLTABLE_TRAINING_ROBOT_URDF_SHA256,
    Core4DSmallTablePPO,
    core4d_object_height_penalty_contract,
    expected_core4d_smalltable_checkpoint_metadata,
    validate_core4d_smalltable_checkpoint,
)
from holosoma.agents.mappo.core4d_smalltable_runner import CORE4D_SMALLTABLE_ACTOR_GROUPS
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    g1_29dof_core4d_smalltable_baseline,
)


def _interaction_contract():
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


def _chair_contract():
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


def _state(**kwargs):
    return {
        "core4d_smalltable_mappo": expected_core4d_smalltable_checkpoint_metadata(**kwargs),
        "actor_model_state_dict": {},
        "critic_model_state_dict": {},
        "actor_optimizer_state_dict": {},
        "critic_optimizer_state_dict": {},
        "actor_obs_normalizer_state_dict": {},
        "critic_obs_normalizer_state_dict": {},
        "iter": 2000,
    }


def test_height_contract_names_the_signed_additive_world_root_objective():
    assert core4d_object_height_penalty_contract() is None
    assert core4d_object_height_penalty_contract(1.0, 0.05) == {
        "version": CORE4D_OBJECT_HEIGHT_PENALTY_CONTRACT_VERSION,
        "frame": "world",
        "position_source": "object_root",
        "reward_term": "object_height_error_penalty",
        "scale_m": 0.05,
        "weight": 1.0,
        "reward_weight": -1.0,
        "formula": "-weight * ((z - z_ref) / scale_m) ** 2",
    }


@pytest.mark.parametrize("interaction", [False, True])
@pytest.mark.parametrize("z_weight", [1.0, 2.0])
def test_disabled_contract_preserves_every_legacy_dictionary_and_label(interaction, z_weight):
    kwargs = {
        "interaction_contract": _interaction_contract() if interaction else None,
        "object_z_error_weight": z_weight,
    }
    legacy = _state(**kwargs)
    for scale in (0.05, 0.1):
        explicit = _state(**kwargs, object_height_penalty_weight=0.0, object_height_penalty_scale=scale)
        assert explicit == legacy
        assert "object_height_penalty" not in explicit["core4d_smalltable_mappo"]
        assert validate_core4d_smalltable_checkpoint(explicit) == 2000
        assert core4d_smalltable_evaluation_reward_metadata(explicit) == (
            core4d_smalltable_evaluation_reward_metadata(legacy)
        )


@pytest.mark.parametrize("interaction", [False, True])
@pytest.mark.parametrize("z_weight", [1.0, 2.0])
def test_enabled_contract_is_additive_to_baseline_interaction_and_wz2(interaction, z_weight):
    kwargs = {
        "interaction_contract": _interaction_contract() if interaction else None,
        "object_z_error_weight": z_weight,
    }
    old = expected_core4d_smalltable_checkpoint_metadata(**kwargs)
    state = _state(**kwargs, object_height_penalty_weight=1.0)
    new = copy.deepcopy(state["core4d_smalltable_mappo"])
    assert new.pop("object_height_penalty") == core4d_object_height_penalty_contract(1.0)
    assert new == old
    assert validate_core4d_smalltable_checkpoint(state) == 2000


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_invalid_weight_is_rejected_before_constructing_a_learner(value):
    with pytest.raises(ValueError):
        core4d_object_height_penalty_contract(value)
    with pytest.raises(ValueError):
        expected_core4d_smalltable_checkpoint_metadata(object_height_penalty_weight=value)
    with pytest.raises(ValueError):
        Core4DSmallTablePPO(None, None, num_envs=1, object_height_penalty_weight=value)


@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"), True, "0.05", None])
def test_invalid_scale_is_rejected_even_when_disabled(weight, value):
    with pytest.raises(ValueError):
        core4d_object_height_penalty_contract(weight, value)
    with pytest.raises(ValueError):
        expected_core4d_smalltable_checkpoint_metadata(
            object_height_penalty_weight=weight, object_height_penalty_scale=value,
        )
    with pytest.raises(ValueError):
        Core4DSmallTablePPO(
            None, None, num_envs=1,
            object_height_penalty_weight=weight, object_height_penalty_scale=value,
        )


@pytest.mark.parametrize(("field", "value"), [
    ("version", "other"), ("frame", "body"), ("position_source", "object_com"),
    ("reward_term", "object_global_ref_position_error_exp"),
    ("scale_m", 0), ("scale_m", True), ("scale_m", float("inf")),
    ("weight", 0), ("weight", -1), ("weight", True), ("weight", float("nan")),
    ("reward_weight", 1), ("reward_weight", -2), ("reward_weight", True),
    ("reward_weight", float("nan")), ("formula", "-abs(z-z_ref)"),
    ("unexpected_field", 1),
])
def test_checkpoint_rejects_tampered_height_semantics(field, value):
    state = _state(object_height_penalty_weight=1.0)
    state["core4d_smalltable_mappo"]["object_height_penalty"][field] = value
    with pytest.raises(ValueError):
        validate_core4d_smalltable_checkpoint(state)


def test_checkpoint_rejects_incomplete_or_disguised_disabled_contract():
    for field in core4d_object_height_penalty_contract(1.0):
        state = _state(object_height_penalty_weight=1.0)
        state["core4d_smalltable_mappo"]["object_height_penalty"].pop(field)
        with pytest.raises(ValueError):
            validate_core4d_smalltable_checkpoint(state)
    for contract in (None, [], {}, {"weight": 0.0, "scale_m": 0.05, "reward_weight": 0.0}):
        state = _state()
        state["core4d_smalltable_mappo"]["object_height_penalty"] = contract
        with pytest.raises(ValueError):
            validate_core4d_smalltable_checkpoint(state)


@pytest.mark.parametrize(("saved", "requested"), [
    ((0.0, 0.05), (1.0, 0.05)), ((1.0, 0.05), (0.0, 0.05)),
    ((1.0, 0.05), (2.0, 0.05)), ((1.0, 0.05), (1.0, 0.1)),
])
def test_resume_rejects_legacy_or_changed_height_objective(saved, requested):
    shared = {"interaction_contract": _interaction_contract(), "object_z_error_weight": 2.0}
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = expected_core4d_smalltable_checkpoint_metadata(
        **shared, object_height_penalty_weight=requested[0], object_height_penalty_scale=requested[1],
    )
    learner._validate_training_state(_state(
        **shared, object_height_penalty_weight=requested[0], object_height_penalty_scale=requested[1],
    ))
    with pytest.raises(ValueError, match="resume reward contract mismatch"):
        learner._validate_training_state(_state(
            **shared, object_height_penalty_weight=saved[0], object_height_penalty_scale=saved[1],
        ))


@pytest.mark.parametrize("interaction", [False, True])
@pytest.mark.parametrize("z_weight", [1.0, 2.0])
def test_evaluation_keeps_old_scoring_and_all_training_labels(interaction, z_weight):
    state = _state(
        interaction_contract=_interaction_contract() if interaction else None,
        object_z_error_weight=z_weight, object_height_penalty_weight=1.0,
    )
    labels = core4d_smalltable_evaluation_reward_metadata(state)
    assert labels["training_reward_variant"] == ("interaction_mesh" if interaction else "baseline")
    assert labels["evaluation_reward_matches_training"] is False
    assert labels["evaluation_object_height_penalty_weight"] == 0.0
    assert labels["object_height_penalty_training_contract"] == core4d_object_height_penalty_contract(1.0)
    assert "training includes" in labels["evaluation_reward_note"]
    assert "evaluation reward_sum excludes" in labels["evaluation_reward_note"]
    if interaction:
        assert labels["interaction_mesh_training_contract"] == _interaction_contract()
    if z_weight == 2.0:
        assert labels["evaluation_object_z_error_weight"] == 1.0
        assert "isotropic" in labels["evaluation_reward_note"]
        assert labels["object_position_tracking_training_contract"] == (
            state["core4d_smalltable_mappo"]["object_position_tracking"]
        )
    labels["object_height_penalty_training_contract"]["scale_m"] = 99
    assert state["core4d_smalltable_mappo"]["object_height_penalty"]["scale_m"] == 0.05


def test_chair_legacy_is_preserved_and_height_variant_is_rejected():
    contract = _chair_contract()
    legacy = _state(experiment_contract=contract)
    assert legacy == _state(experiment_contract=contract, object_height_penalty_weight=0.0)
    assert validate_core4d_smalltable_checkpoint(legacy, experiment_contract=contract) == 2000
    assert core4d_smalltable_evaluation_reward_metadata(
        legacy, experiment_contract=contract,
    )["evaluation_reward_matches_training"] is True
    with pytest.raises(ValueError, match="restricted to smalltable"):
        expected_core4d_smalltable_checkpoint_metadata(
            experiment_contract=contract, object_height_penalty_weight=1.0,
        )
    legacy["core4d_smalltable_mappo"]["object_height_penalty"] = core4d_object_height_penalty_contract(1.0)
    with pytest.raises(ValueError, match="restricted to smalltable"):
        validate_core4d_smalltable_checkpoint(legacy, experiment_contract=contract)


def test_cpu_checkpoint_roundtrip_preserves_networks_and_actor_only_actions(tmp_path):
    config = g1_29dof_core4d_smalltable_baseline.algo.config

    def learner():
        return Core4DSmallTablePPO(
            initialize_core4d_smalltable_model_bundle(config, device="cpu"),
            config, num_envs=2, num_steps_per_env=2, device="cpu",
            interaction_contract=_interaction_contract(), object_z_error_weight=2.0,
            object_height_penalty_weight=1.0, object_height_penalty_scale=0.05,
        )

    first, second = learner(), learner()
    state = first.training_state_dict(iteration=2000)
    path = tmp_path / "height_checkpoint.pt"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=True)
    assert second.load_training_state_dict(loaded) == 2000
    assert state["core4d_smalltable_mappo"]["object_height_penalty"] == core4d_object_height_penalty_contract(1.0)
    for network in ("actor", "critic"):
        for name, value in getattr(first.models, network).state_dict().items():
            torch.testing.assert_close(value, getattr(second.models, network).state_dict()[name])
    observations = {name: torch.randn(2, 2, width) for name, width in CORE4D_SMALLTABLE_ACTOR_GROUPS}
    torch.testing.assert_close(
        deterministic_core4d_smalltable_actions(first.models, observations),
        deterministic_core4d_smalltable_actions(second.models, observations),
    )
    loaded["core4d_smalltable_mappo"]["object_height_penalty"]["scale_m"] = 0.2
    assert first.training_state_dict(iteration=4000)["core4d_smalltable_mappo"] == (
        second.training_state_dict(iteration=4000)["core4d_smalltable_mappo"]
    )
