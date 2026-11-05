import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holosoma.envs.marl.plan5_push_manager import Plan5PushManager


class FakeActionManager:
    def __init__(self):
        self.received = None

    def process_actions(self, actions):
        self.received = actions.clone()


def make_uninitialized_env():
    env = object.__new__(Plan5PushManager)
    env.num_envs = 3
    env.num_agents = 2
    env.num_dof = 29
    env.action_manager = FakeActionManager()
    return env


def test_agent_actions_are_flattened_in_environment_major_order():
    env = make_uninitialized_env()
    actions = torch.arange(3 * 2 * 29, dtype=torch.float32).view(3, 2, 29)

    env._pre_physics_step(actions)

    torch.testing.assert_close(env.action_manager.received, actions.view(3, 58))
    torch.testing.assert_close(env.action_manager.received[1, :29], actions[1, 0])
    torch.testing.assert_close(env.action_manager.received[1, 29:], actions[1, 1])


def test_flat_actions_are_accepted_without_reordering():
    env = make_uninitialized_env()
    actions = torch.arange(3 * 58, dtype=torch.float32).view(3, 58)

    env._pre_physics_step(actions)

    torch.testing.assert_close(env.action_manager.received, actions)


def test_invalid_action_shape_fails_closed():
    env = make_uninitialized_env()

    with pytest.raises(ValueError, match="Plan 5 actions must have shape"):
        env._pre_physics_step(torch.zeros(3, 29))


def test_smoke_config_is_dual_rubber_hand_and_not_a_training_reward():
    from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_smoke

    cfg = g1_29dof_plan5_push_smoke
    assert cfg.simulator._target_.endswith(":DualRobotIsaacSim") or cfg.simulator._target_.endswith(
        ".DualRobotIsaacSim"
    )
    assert cfg.robot.asset.urdf_file.endswith("main_mesh_collision_rubberhand.urdf")
    assert "widetable_plan5_preflight.urdf" in cfg.robot.object.object_urdf_path
    assert cfg.action.terms["joint_control"].func.endswith(":DualJointPositionActionTerm")
    assert "paired_motion_command" in cfg.command.setup_terms
    assert list(cfg.observation.groups) == ["actor_obs", "teammate_obs", "critic_obs"]
    assert cfg.reward.terms == {}
    assert cfg.randomization.setup_terms == {}


def test_baseline_config_enables_original_reward_and_joint_termination():
    from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline

    cfg = g1_29dof_plan5_push_baseline
    assert len(cfg.reward.terms) == 11
    assert list(cfg.termination.terms) == ["timeout", "joint_bad_tracking"]
    assert cfg.termination.terms["joint_bad_tracking"].func.endswith(":JointBadTrackingZOnly")
    assert cfg.robot.asset.urdf_file.endswith("main_mesh_collision_rubberhand.urdf")
    assert cfg.robot.object.object_urdf_path.endswith("objects_widetable_plan5_training.urdf")
    assert list(cfg.randomization.setup_terms) == ["set_object_rigid_body_material_startup"]
    material = cfg.randomization.setup_terms["set_object_rigid_body_material_startup"]
    assert material.params == {
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
    }


def test_pull_command_uses_explicit_runtime_reference_without_push_spacing():
    from holosoma.config_values.marl.g1.command import g1_29dof_paired_pull_command

    term = g1_29dof_paired_pull_command.setup_terms["paired_motion_command"]
    assert term.params["motion_config"].motion_file.endswith(
        "sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz"
    )
    assert term.params["paired_reference_file"].endswith(
        "rubber_hand_largetable_v1/pull/plan5_attempt08_mirrored_pair_runtime.npz"
    )
    assert "lateral_spacing_m" not in term.params
    assert g1_29dof_paired_pull_command.reset_terms["paired_motion_command"] is term
    assert g1_29dof_paired_pull_command.step_terms["paired_motion_command"] is term


def test_pull_smoke_uses_formal_z_wide_table_and_fixed_material():
    from holosoma.config_values.marl.g1.experiment import (
        g1_29dof_plan5_pull_smoke,
        g1_29dof_plan5_push_smoke,
    )

    cfg = g1_29dof_plan5_pull_smoke
    assert cfg.simulator is g1_29dof_plan5_push_smoke.simulator
    assert cfg.robot.asset.urdf_file.endswith("main_mesh_collision_rubberhand.urdf")
    assert cfg.robot.object.object_urdf_path.endswith(
        "objects_widetable_plan5_pull_training.urdf"
    )
    assert cfg.reward.terms == {}
    assert list(cfg.termination.terms) == ["timeout"]
    material = cfg.randomization.setup_terms["set_object_rigid_body_material_startup"]
    assert material.params == {
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
    }


def test_pull_baseline_reuses_push_reward_termination_and_physics():
    from holosoma.config_values.marl.g1.experiment import (
        g1_29dof_plan5_pull_baseline,
        g1_29dof_plan5_push_baseline,
    )

    cfg = g1_29dof_plan5_pull_baseline
    assert cfg.reward is g1_29dof_plan5_push_baseline.reward
    assert cfg.termination is g1_29dof_plan5_push_baseline.termination
    assert cfg.randomization is g1_29dof_plan5_push_baseline.randomization
    assert cfg.robot.object.object_urdf_path.endswith(
        "objects_widetable_plan5_pull_training.urdf"
    )
    assert cfg.command.setup_terms["paired_motion_command"].params[
        "paired_reference_file"
    ].endswith("plan5_attempt08_mirrored_pair_runtime.npz")


def test_cuda_smoke_cli_has_skill_specific_checkpoint_and_spacing_contracts():
    repo_root = Path(__file__).resolve().parents[5]
    script = (repo_root / "scripts/smoke_plan5_environment.py").read_text()

    assert (
        'PARSER.add_argument("--skill", choices=("push", "pull", "kick"), default="push")'
        in script
    )
    assert "g1_29dof_plan5_push_smoke" in script
    assert "g1_29dof_plan5_push_baseline" in script
    assert "g1_29dof_plan5_pull_smoke" in script
    assert "g1_29dof_plan5_pull_baseline" in script
    assert "g1_29dof_plan5_kick_smoke" in script
    assert "g1_29dof_plan5_kick_baseline" in script
    assert "marl_compat_a1_v1/model_07999_actor158.pt" in script
    assert "marl_compat_pull_v1/model_07999_actor158.pt" in script
    assert "marl_compat_kick_v1/model_07999_actor158.pt" in script
    assert "f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364" in script
    assert "1555968f678c2b69fcd6f09c64d0a6252683eab902edd773a84acc205d0f5491" in script
    assert 'if ARGS.skill == "push":' in script
    assert "torch.full_like(lateral_spacing, 0.8)" in script
    assert "reset positions do not match the explicit paired reference" in script
    assert '"skill": ARGS.skill' in script
    assert '"config": CONFIG_LABEL' in script
    assert '"actor_checkpoint_sha256": models.source_sha256' in script


def test_training_cli_selects_independent_push_pull_and_kick_inputs_without_ddp():
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "scripts/train_plan5_push.py"
    script = script_path.read_text()

    assert (
        'PARSER.add_argument("--skill", choices=("push", "pull", "kick"), default="push")'
        in script
    )
    assert "g1_29dof_plan5_push_baseline" in script
    assert "g1_29dof_plan5_pull_baseline" in script
    assert "g1_29dof_plan5_kick_baseline" in script
    assert "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt" in script
    assert "logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt" in script
    assert "logs/WholeBodyTracking/marl_compat_kick_v1/model_07999_actor158.pt" in script
    assert "f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364" in script
    assert "1555968f678c2b69fcd6f09c64d0a6252683eab902edd773a84acc205d0f5491" in script
    assert 'command_term = CONFIG.command.setup_terms["paired_motion_command"]' in script
    assert '"motion_file": selected_motion_config.motion_file' in script
    assert '"paired_reference_file": command_params.get("paired_reference_file")' in script
    assert "from holosoma.config_values.marl.g1.command import motion_config" not in script
    assert '"distributed_data_parallel": False' in script
    assert "WORLD_SIZE must be 1" in script
    assert "3.95048914424628" in script
    assert "0.62774975216702" in script

    environment = dict(os.environ)
    environment["WORLD_SIZE"] = "1"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--skill {push,pull,kick}" in result.stdout


def test_pull_training_cli_rejects_push_only_smooth_mode_before_sim_start():
    repo_root = Path(__file__).resolve().parents[5]
    script_path = repo_root / "scripts/train_plan5_push.py"
    environment = dict(os.environ)
    environment["WORLD_SIZE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--skill",
            "pull",
            "--joint-acceleration-weight",
            "-0.1",
        ],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "smooth fine-tuning is Push-only" in result.stderr
