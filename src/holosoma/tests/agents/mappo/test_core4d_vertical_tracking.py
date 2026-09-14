"""CPU-only tests for the isolated small-table world-z tracking experiment."""

import ast
import copy
import inspect
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

from holosoma.agents.mappo.core4d_smalltable_evaluation import (
    core4d_smalltable_evaluation_reward_metadata,
)
from holosoma.agents.mappo.core4d_smalltable_initialization import (
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (
    Core4DSmallTablePPO,
    core4d_object_position_tracking_contract,
    expected_core4d_smalltable_checkpoint_metadata,
    validate_core4d_smalltable_checkpoint,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    g1_29dof_core4d_smalltable_baseline,
)
from holosoma.config_values.marl.g1.core4d_smalltable_reward import (
    g1_29dof_core4d_smalltable_reward,
    validate_object_z_error_weight,
    with_interaction_reward_term,
    with_object_z_error_weight,
)
from holosoma.managers.reward.terms import marl
from holosoma.utils import eval_utils


REPO_ROOT = Path(__file__).resolve().parents[5]


def _state(weight=1.0):
    return {
        "core4d_smalltable_mappo": expected_core4d_smalltable_checkpoint_metadata(
            object_z_error_weight=weight,
        ),
        "actor_model_state_dict": {},
        "critic_model_state_dict": {},
        "actor_optimizer_state_dict": {},
        "critic_optimizer_state_dict": {},
        "actor_obs_normalizer_state_dict": {},
        "critic_obs_normalizer_state_dict": {},
        "iter": 12000,
    }


def _reward(monkeypatch, error, **kwargs):
    error = torch.as_tensor(error, dtype=torch.float64)
    command = SimpleNamespace(object_pos_w=error, simulator_object_pos_w=torch.zeros_like(error))
    monkeypatch.setattr(marl, "_command", lambda env: command)
    return marl.object_global_ref_position_error_exp(None, sigma=0.3, **kwargs)


def test_default_and_explicit_one_are_exact_legacy_arithmetic(monkeypatch):
    error = torch.tensor([[0.01, -0.2, 0.06], [-0.7, 0.12, -0.4]], dtype=torch.float64)
    expected = torch.exp(-torch.square(error).sum(dim=-1) / 0.3**2)
    assert torch.equal(_reward(monkeypatch, error), expected)
    assert torch.equal(_reward(monkeypatch, error, object_z_error_weight=1.0), expected)


def test_two_changes_only_world_z_and_penalizes_height_overshoot(monkeypatch):
    error = [[0.06, 0, 0], [0, 0.06, 0], [0, 0, 0.06], [0, 0, -0.06], [0, 0, 0]]
    old = _reward(monkeypatch, error)
    new = _reward(monkeypatch, error, object_z_error_weight=2.0)
    assert torch.equal(new[:2], old[:2])
    assert new[2] < old[2]
    assert new[2] == new[3]
    assert new[4] == 1.0
    torch.testing.assert_close(new[2], torch.tensor(0.9231163463866358, dtype=torch.float64))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"), True, "2", None])
def test_invalid_coefficients_are_rejected_at_all_boundaries(monkeypatch, value):
    with pytest.raises(ValueError, match="positive and finite"):
        validate_object_z_error_weight(value)
    with pytest.raises(ValueError, match="positive and finite"):
        _reward(monkeypatch, [[0, 0, 0]], object_z_error_weight=value)
    with pytest.raises(ValueError, match="positive and finite"):
        with_object_z_error_weight(g1_29dof_core4d_smalltable_reward, value)
    with pytest.raises(ValueError, match="positive and finite"):
        expected_core4d_smalltable_checkpoint_metadata(object_z_error_weight=value)


def test_reward_helper_copies_without_mutating_baseline_or_interaction():
    original = copy.deepcopy(g1_29dof_core4d_smalltable_reward)
    graph = with_interaction_reward_term(g1_29dof_core4d_smalltable_reward, "/not/read/graph.npz")
    candidate = with_object_z_error_weight(graph, 2.0)
    name = "object_global_ref_position_error_exp"
    assert candidate.terms[name].params == {"sigma": 0.3, "object_z_error_weight": 2.0}
    assert candidate.terms[name].weight == 1.0
    assert graph.terms[name].params == {"sigma": 0.3}
    assert g1_29dof_core4d_smalltable_reward == original
    for key in graph.terms:
        if key != name:
            assert candidate.terms[key] == graph.terms[key]
    candidate.terms["interaction_mesh"].params["sigma"] = 99
    assert graph.terms["interaction_mesh"].params["sigma"] == 0.06
    assert with_object_z_error_weight(g1_29dof_core4d_smalltable_reward, 1) == original
    assert with_object_z_error_weight(with_object_z_error_weight(original, 2), 1) == original


def test_legacy_metadata_unchanged_and_nondefault_adds_only_explicit_contract():
    legacy = expected_core4d_smalltable_checkpoint_metadata()
    assert legacy == expected_core4d_smalltable_checkpoint_metadata(object_z_error_weight=1.0)
    assert "object_position_tracking" not in legacy
    assert core4d_object_position_tracking_contract(1.0) is None
    candidate = expected_core4d_smalltable_checkpoint_metadata(object_z_error_weight=2.0)
    assert candidate.pop("object_position_tracking") == {
        "version": "world_xyz_squared_error_weighting_v1",
        "frame": "world",
        "squared_error_weights_xyz": (1.0, 1.0, 2.0),
        "sigma_m": 0.3,
        "reward_weight": 1.0,
    }
    assert candidate == legacy
    assert validate_core4d_smalltable_checkpoint(_state()) == 12000
    assert validate_core4d_smalltable_checkpoint(_state(2.0)) == 12000


@pytest.mark.parametrize("field,value", [
    ("version", "other"), ("frame", "body"), ("sigma_m", 0.15), ("reward_weight", 2),
    ("squared_error_weights_xyz", (2.0, 1.0, 2.0)),
    ("squared_error_weights_xyz", (True, 1.0, 2.0)),
    ("squared_error_weights_xyz", (1.0, 1.0, 1.0)),
    ("squared_error_weights_xyz", (1.0, 1.0, float("nan"))),
    ("reward_weight", True),
])
def test_checkpoint_rejects_malformed_or_disguised_position_contract(field, value):
    state = _state(2.0)
    state["core4d_smalltable_mappo"]["object_position_tracking"][field] = value
    with pytest.raises(ValueError):
        validate_core4d_smalltable_checkpoint(state)


@pytest.mark.parametrize("saved,requested", [(1, 2), (2, 1), (2, 3)])
def test_resume_rejects_changed_position_objective(saved, requested):
    learner = object.__new__(Core4DSmallTablePPO)
    learner._checkpoint_metadata = expected_core4d_smalltable_checkpoint_metadata(
        object_z_error_weight=requested,
    )
    learner._validate_training_state(_state(requested))
    with pytest.raises(ValueError, match="resume reward contract mismatch"):
        learner._validate_training_state(_state(saved))


def test_cpu_save_load_roundtrip_preserves_vertical_contract():
    config = g1_29dof_core4d_smalltable_baseline.algo.config
    def learner():
        return Core4DSmallTablePPO(
            initialize_core4d_smalltable_model_bundle(config, device="cpu"),
            config, num_envs=2, num_steps_per_env=2, device="cpu", object_z_error_weight=2.0,
        )
    first, second = learner(), learner()
    state = first.training_state_dict(iteration=2000)
    assert second.load_training_state_dict(state) == 2000
    assert state["core4d_smalltable_mappo"]["object_position_tracking"] == (
        core4d_object_position_tracking_contract(2.0)
    )
    for name, value in first.models.actor.state_dict().items():
        torch.testing.assert_close(value, second.models.actor.state_dict()[name])


def test_actor_only_evaluation_accepts_new_checkpoint_and_labels_old_score():
    state = _state(2.0)
    labels = core4d_smalltable_evaluation_reward_metadata(state)
    assert labels["evaluation_reward_matches_training"] is False
    assert labels["evaluation_object_z_error_weight"] == 1.0
    assert "isotropic" in labels["evaluation_reward_note"]
    assert labels["object_position_tracking_training_contract"] == core4d_object_position_tracking_contract(2)
    labels["object_position_tracking_training_contract"]["sigma_m"] = 99
    assert state["core4d_smalltable_mappo"]["object_position_tracking"]["sigma_m"] == 0.3
    legacy = core4d_smalltable_evaluation_reward_metadata(_state())
    assert legacy["evaluation_reward_matches_training"] is True
    assert "object_position_tracking_training_contract" not in legacy


def test_custom_experiment_cannot_silently_inherit_vertical_variant():
    with pytest.raises(ValueError, match="restricted to smalltable"):
        expected_core4d_smalltable_checkpoint_metadata(
            experiment_contract={}, object_z_error_weight=2.0,
        )


class _BeforeSimulator(Exception):
    def __init__(self, namespace):
        self.namespace = namespace


def _cli_prefix(monkeypatch, args):
    def stop(config):
        raise _BeforeSimulator(dict(inspect.currentframe().f_back.f_globals))
    monkeypatch.setattr(eval_utils, "init_sim_imports", stop)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(sys, "argv", ["train_core4d_smalltable.py", *args])
    runpy.run_path(str(REPO_ROOT / "scripts/train_core4d_smalltable.py"), run_name="__main__")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_coefficient_before_simulator(monkeypatch, value):
    with pytest.raises(SystemExit) as caught:
        _cli_prefix(monkeypatch, ["--object-z-error-weight", value])
    assert caught.value.code == 2


def test_cli_rejects_chair_override_before_simulator(monkeypatch):
    with pytest.raises(SystemExit) as caught:
        _cli_prefix(monkeypatch, ["--experiment", "chair021", "--object-z-error-weight", "2"])
    assert caught.value.code == 2


def test_cli_applies_only_the_requested_smalltable_reward_change(monkeypatch):
    with pytest.raises(_BeforeSimulator) as caught:
        _cli_prefix(monkeypatch, ["--object-z-error-weight", "2"])
    namespace = caught.value.namespace
    config = namespace["CONFIG"]
    original = g1_29dof_core4d_smalltable_baseline
    assert config.robot is original.robot
    assert config.observation is original.observation
    assert config.algo is original.algo
    assert config.termination is original.termination
    assert config.reward == with_object_z_error_weight(original.reward, 2)
    assert "zweight2" in namespace["ARGS"].output_dir.name


def _atomic_save_function():
    """Load only the pure filesystem helper, never the simulation entrypoint."""
    path = REPO_ROOT / "scripts/train_core4d_smalltable.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_save_checkpoint_atomic")
    namespace = {"Path": Path, "torch": torch, "tempfile": tempfile, "os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_save_checkpoint_atomic"]


def test_atomic_checkpoint_save_exposes_only_complete_pt(tmp_path):
    destination = tmp_path / "model_02000.pt"
    _atomic_save_function()(destination, {"iter": 2000, "value": torch.arange(3)})
    loaded = torch.load(destination, weights_only=True)
    assert loaded["iter"] == 2000
    assert torch.equal(loaded["value"], torch.arange(3))
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_checkpoint_failure_preserves_previous_file(monkeypatch, tmp_path):
    destination = tmp_path / "model_02000.pt"
    save = _atomic_save_function()
    save(destination, {"iter": 2000})
    previous = destination.read_bytes()
    def fail(state, stream):
        stream.write(b"partial")
        raise RuntimeError("simulated serialization failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="serialization failure"):
        save(destination, {"iter": 4000})
    assert destination.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [destination]
