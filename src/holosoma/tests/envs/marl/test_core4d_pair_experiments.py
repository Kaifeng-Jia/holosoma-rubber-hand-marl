"""New data selection reuses the existing method without changing legacy defaults."""

from dataclasses import replace
import hashlib
import json

import pytest

from holosoma.config_values.marl.g1.core4d_pair_experiments import (
    PACKAGE_ROOT,
    get_core4d_pair_experiment,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    g1_29dof_core4d_smalltable_baseline,
    with_interaction_mesh_reward,
    with_pair_experiment,
)


def test_default_returns_unchanged_smalltable_config_and_contract():
    exp = get_core4d_pair_experiment()
    baseline = g1_29dof_core4d_smalltable_baseline
    assert exp.experiment_id == "smalltable"
    assert exp.checkpoint_contract is None
    assert exp.object_mass_kg == 20.0
    assert exp.reference_frames == 687
    assert exp.allow_interaction_mesh
    assert with_pair_experiment(baseline, exp) is baseline


def test_chair_replaces_only_data_physics_and_horizon():
    baseline = g1_29dof_core4d_smalltable_baseline
    exp = get_core4d_pair_experiment("chair021")
    chair = with_pair_experiment(baseline, exp)
    assert chair.env_class == baseline.env_class
    assert chair.reward == baseline.reward
    assert chair.termination == baseline.termination
    assert chair.observation == baseline.observation
    assert chair.algo == baseline.algo
    assert chair.action == baseline.action
    assert chair.robot.asset == baseline.robot.asset
    assert chair.training.project == "Core4DChair"
    assert chair.robot.object.collider_type == "convex_decomposition"
    assert baseline.robot.object.collider_type == "convex_hull"
    assert chair.robot.object.object_urdf_path.endswith("chair021_training.urdf")
    assert chair.simulator.config.sim.max_episode_length_s == pytest.approx(391 / 50)
    assert chair.simulator.config.sim.fps == 200
    assert chair.simulator.config.sim.control_decimation == 4
    for group in ("setup_terms", "reset_terms", "step_terms"):
        term = getattr(chair.command, group)["paired_motion_command"]
        old = getattr(baseline.command, group)["paired_motion_command"]
        assert term.func == old.func
        assert term.params["paired_reference_file"] == exp.runtime_reference_file
        assert term.params["motion_config"].motion_file == exp.runtime_reference_file
        assert old.params["paired_reference_file"] != exp.runtime_reference_file
        assert term.params["motion_config"].body_names_to_track == old.params["motion_config"].body_names_to_track


def test_chair_assets_and_manifest_are_bound_to_approved_source():
    exp = get_core4d_pair_experiment("chair021")
    for path, digest in (
        (exp.runtime_reference_file, exp.runtime_reference_sha256),
        (exp.object_urdf_file, exp.object_urdf_sha256),
        (exp.training_promotion_file, exp.training_promotion_sha256),
    ):
        assert hashlib.sha256((PACKAGE_ROOT / path).read_bytes()).hexdigest() == digest
    manifest = json.loads((PACKAGE_ROOT / exp.training_promotion_file).read_text())
    assert exp.object_mass_kg == manifest["object_mass_kg"] == 5.0
    assert exp.training_ready == manifest["training_ready"]
    assert exp.checkpoint_contract["experiment_id"] == "chair021"
    assert not exp.allow_interaction_mesh


def test_chair_rejects_old_table_graph_and_inconsistent_rates():
    exp = get_core4d_pair_experiment("chair021")
    base = g1_29dof_core4d_smalltable_baseline
    graph = with_interaction_mesh_reward(base, "old-table-only.npz")
    with pytest.raises(ValueError, match="interaction graph"):
        with_pair_experiment(graph, exp)
    with pytest.raises(ValueError, match="Reference FPS"):
        with_pair_experiment(base, replace(exp, reference_fps=30))
    with pytest.raises(ValueError, match="integer multiple"):
        with_pair_experiment(base, replace(exp, physics_hz=201))
    with pytest.raises(ValueError, match="Unknown CORE4D"):
        get_core4d_pair_experiment("typo")
