"""Exercise CLI/config selection without launching Isaac, an environment, or PPO."""

import ast
from dataclasses import replace
import inspect
from pathlib import Path
import runpy
import sys

import pytest

from holosoma.agents.mappo import core4d_smalltable_ppo
from holosoma.config_values.marl.g1 import core4d_pair_experiments
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    g1_29dof_core4d_smalltable_baseline,
    g1_29dof_core4d_smalltable_smoke,
)
from holosoma.config_values.marl.g1.core4d_smalltable_observation import (
    g1_29dof_core4d_smalltable_evaluation_observation,
)
from holosoma.utils import eval_utils


REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = (
    "train_core4d_smalltable.py",
    "evaluate_core4d_smalltable.py",
    "smoke_core4d_smalltable.py",
)


class _BeforeSimulator(Exception):
    def __init__(self, namespace):
        self.namespace = namespace


def _run_prefix(monkeypatch, script, args, experiment=None):
    """Stop at the simulator-import boundary; never execute main() or physics."""
    def stop(config):
        namespace = dict(inspect.currentframe().f_back.f_globals)
        assert namespace["CONFIG"] is config
        raise _BeforeSimulator(namespace)

    monkeypatch.setattr(eval_utils, "init_sim_imports", stop)
    monkeypatch.setenv("WORLD_SIZE", "1")
    if experiment is not None:
        def select(name):
            assert name == experiment.experiment_id
            return experiment
        monkeypatch.setattr(core4d_pair_experiments, "get_core4d_pair_experiment", select)
    required = (
        ["--checkpoint", "/not/read/model.pt", "--output-dir", "/not/written/evaluation"]
        if script.startswith("evaluate_") else []
    )
    monkeypatch.setattr(sys, "argv", [script, *required, *args])
    runpy.run_path(str(REPO_ROOT / "scripts" / script), run_name="__main__")


def _capture_prefix(monkeypatch, script, args, experiment=None):
    with pytest.raises(_BeforeSimulator) as caught:
        _run_prefix(monkeypatch, script, args, experiment)
    return caught.value.namespace


@pytest.fixture
def chair():
    # Exercise both readiness branches without modifying the reviewed manifest.
    return replace(core4d_pair_experiments.get_core4d_pair_experiment("chair021"), training_ready=True)


@pytest.mark.parametrize("script", SCRIPTS)
def test_default_smalltable_is_preserved(monkeypatch, script):
    state = _capture_prefix(monkeypatch, script, [])
    cfg = state["CONFIG"]
    old = (
        g1_29dof_core4d_smalltable_baseline
        if script.startswith("train_") else g1_29dof_core4d_smalltable_smoke
    )
    assert state["ARGS"].experiment == "smalltable"
    assert state["EXPERIMENT"].checkpoint_contract is None
    assert cfg.robot is old.robot
    assert cfg.command is old.command
    assert cfg.algo is old.algo
    assert cfg.reward is old.reward
    assert cfg.termination is old.termination
    expected_obs = (
        g1_29dof_core4d_smalltable_evaluation_observation
        if script.startswith("evaluate_") else old.observation
    )
    assert cfg.observation is expected_obs
    assert cfg.training.project == (
        "Core4DSmallTableEval" if script.startswith("evaluate_") else "Core4DSmallTable"
    )
    if script.startswith("train_"):
        assert state["ARGS"].output_dir == (
            REPO_ROOT / "logs/Core4DSmallTable/paired_reference_fresh_seed721_env2048"
        )
        assert state["ARGS"].iterations == 12000
        assert state["ARGS"].save_interval == 2000
    if script.startswith("smoke_"):
        assert state["ARGS"].ppo_update is False


@pytest.mark.parametrize("script", SCRIPTS)
def test_chair_selects_assets_without_changing_learning(monkeypatch, script, chair):
    state = _capture_prefix(monkeypatch, script, ["--experiment", "chair021"], chair)
    cfg = state["CONFIG"]
    old = g1_29dof_core4d_smalltable_baseline
    assert cfg.robot.object.object_urdf_path == chair.object_urdf_file
    assert cfg.robot.object.collider_type == "convex_decomposition"
    assert cfg.robot.asset is old.robot.asset
    assert cfg.algo is old.algo
    assert cfg.action is old.action
    assert cfg.reward is old.reward
    assert cfg.termination is old.termination
    assert cfg.training.project == ("Core4DChairEval" if script.startswith("evaluate_") else "Core4DChair")
    assert state["EXPERIMENT"].object_mass_kg == 5.0
    assert cfg.simulator.config.sim.fps == 200
    assert cfg.simulator.config.sim.control_decimation == 4
    assert cfg.simulator.config.sim.max_episode_length_s == 391 / 50
    for terms in (cfg.command.setup_terms, cfg.command.reset_terms, cfg.command.step_terms):
        params = terms["paired_motion_command"].params
        assert params["paired_reference_file"] == chair.runtime_reference_file
        assert params["motion_config"].motion_file == chair.runtime_reference_file
    assert cfg.randomization.setup_terms["set_object_rigid_body_material_startup"].params == {
        "static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0,
    }
    if script.startswith("train_"):
        assert state["ARGS"].output_dir.parent == REPO_ROOT / "logs/Core4DChair"
    if not script.startswith("smoke_"):
        for variable, relative in (
            ("RUNTIME_REFERENCE_PATH", chair.runtime_reference_file),
            ("OBJECT_URDF_PATH", chair.object_urdf_file),
            ("TRAINING_PROMOTION_PATH", chair.training_promotion_file),
        ):
            assert state[variable] == (REPO_ROOT / "src/holosoma" / relative).resolve()


@pytest.mark.parametrize("script", (SCRIPTS[0], SCRIPTS[2]))
def test_chair_graph_rejected_before_artifact_read_or_simulator(monkeypatch, capsys, script, chair):
    def never_read(*args, **kwargs):
        pytest.fail("Unsupported chair graph must be rejected before reading it")
    monkeypatch.setattr(core4d_smalltable_ppo, "build_core4d_interaction_contract", never_read)
    with pytest.raises(SystemExit) as caught:
        _run_prefix(monkeypatch, script, [
            "--experiment", "chair021", "--reward-variant", "interaction_mesh",
            "--interaction-reference", "/not/read/interaction.npz",
        ], chair)
    assert caught.value.code == 2
    assert "does not allow interaction_mesh" in capsys.readouterr().err


@pytest.mark.parametrize("script", (SCRIPTS[0], SCRIPTS[2]))
def test_legacy_smalltable_graph_still_opt_in(monkeypatch, script):
    contract = {"fixture": "sentinel"}
    monkeypatch.setattr(core4d_smalltable_ppo, "build_core4d_interaction_contract", lambda path: contract)
    state = _capture_prefix(monkeypatch, script, [
        "--reward-variant", "interaction_mesh", "--interaction-reference", "/not/read/interaction.npz",
    ])
    assert state["INTERACTION_CONTRACT"] is contract
    assert state["EXPERIMENT"].checkpoint_contract is None
    term = state["CONFIG"].reward.terms["interaction_mesh"]
    assert term.params == {"reference_file": "/not/read/interaction.npz", "sigma": 0.06}
    assert term.weight == 1.0


@pytest.mark.parametrize("script,args", [
    (SCRIPTS[0], []), (SCRIPTS[1], []), (SCRIPTS[2], ["--ppo-update"]),
])
def test_prepared_assets_cannot_start_training_or_evaluation(monkeypatch, capsys, script, args, chair):
    with pytest.raises(SystemExit) as caught:
        _run_prefix(monkeypatch, script, ["--experiment", "chair021", *args], replace(chair, training_ready=False))
    assert caught.value.code == 2
    assert "not training_ready" in capsys.readouterr().err


def test_prepared_assets_allow_physics_only_smoke(monkeypatch, chair):
    state = _capture_prefix(monkeypatch, SCRIPTS[2], ["--experiment", "chair021"], replace(chair, training_ready=False))
    assert state["ARGS"].ppo_update is False
    assert state["EXPERIMENT"].training_ready is False


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help_exposes_explicit_experiment(monkeypatch, capsys, script):
    with pytest.raises(SystemExit) as caught:
        _run_prefix(monkeypatch, script, ["--help"])
    assert caught.value.code == 0
    assert "--experiment {smalltable,chair021,bucket003,smalltable5kg_A}" in capsys.readouterr().out


@pytest.mark.parametrize("script", SCRIPTS)
def test_checkpoint_contract_wiring_and_syntax(script):
    source = (REPO_ROOT / "scripts" / script).read_text()
    tree = ast.parse(source)
    compile(tree, script, "exec")
    targets = (
        {"validate_core4d_smalltable_checkpoint", "core4d_smalltable_evaluation_reward_metadata"}
        if script.startswith("evaluate_") else {"Core4DSmallTablePPO"}
    )
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in targets:
            keyword = next(kw for kw in node.keywords if kw.arg == "experiment_contract")
            assert ast.unparse(keyword.value) == "EXPERIMENT.checkpoint_contract"
            found.add(node.func.id)
    assert found == targets


@pytest.mark.parametrize("script", (SCRIPTS[0], SCRIPTS[2]))
@pytest.mark.parametrize("variant,weights", [
    ("A", [1.0, 1.0, 1.0, 2.0]),
    ("B", [1.0, 1.0, 1.0, 2.0]),
    ("A_no_rel", [1.0, 1.0, 1.0, 0.0]),
    ("A_no_height", [1.0, 1.0, 0.0, 2.0]),
    ("A_no_rel_no_height", [1.0, 1.0, 0.0, 0.0]),
])
def test_bucket_cli_preserves_full_variant_and_frozen_scene(monkeypatch, script, variant, weights):
    from holosoma.config_values.marl.g1.core4d_bucket_contract import BUCKET_VARIANTS

    state = _capture_prefix(monkeypatch, script, [
        "--experiment", "bucket003", "--reward-variant", f"bucket_{variant}",
    ])
    assert state["BUCKET_REWARD_CHOICES"] == tuple(f"bucket_{item}" for item in BUCKET_VARIANTS)
    contract = state["BUCKET_CONTRACT"]
    assert contract["variant"] == variant
    assert contract["block_weights_position_rotation_height_relation"] == weights
    assert contract["position_error_weights_xyz"] == [1.0, 1.0, 2.0]
    cfg = state["CONFIG"]
    term = cfg.reward.terms["bucket_interaction"]
    assert term.params["variant"] == variant
    assert cfg.simulator.config.enable_object_hand_contact is True
    assert state["EXPERIMENT"].object_mass_kg == 1.0
    base = g1_29dof_core4d_smalltable_baseline
    for name, expected in base.reward.terms.items():
        if not name.startswith("object_global_ref_"):
            assert cfg.reward.terms[name] == expected
    assert cfg.algo is base.algo
    assert cfg.termination is base.termination
    if script.startswith("train_"):
        assert state["ARGS"].resume is None
        assert state["ARGS"].iterations == 12000
        assert state["ARGS"].num_envs == 2048
        assert state["ARGS"].save_interval == 2000


@pytest.mark.parametrize("script", (SCRIPTS[0], SCRIPTS[2]))
@pytest.mark.parametrize("variant", ("bucket_A_no_rel", "bucket_A_no_height", "bucket_A_no_rel_no_height"))
def test_bucket_ablations_cannot_silently_change_chair(monkeypatch, capsys, script, variant):
    with pytest.raises(SystemExit) as caught:
        _run_prefix(monkeypatch, script, ["--experiment", "chair021", "--reward-variant", variant])
    assert caught.value.code == 2
    assert "bucket003" in capsys.readouterr().err
