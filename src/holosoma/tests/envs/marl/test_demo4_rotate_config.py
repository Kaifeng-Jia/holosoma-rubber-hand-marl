"""Static configuration contract for the isolated Demo 4 task."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from holosoma.config_values.marl.g1.demo4_command import (
    DEMO4_ROTATE_RUNTIME_REFERENCE_FILE,
    g1_29dof_demo4_rotate_command,
)
from holosoma.config_values.marl.g1.demo4_experiment import (
    DEMO4_EPISODE_SECONDS,
    DEMO4_REFERENCE_FPS,
    DEMO4_REFERENCE_FRAMES,
    g1_29dof_demo4_rotate_baseline,
    g1_29dof_demo4_rotate_smoke,
)
from holosoma.config_values.marl.g1.demo4_reward import (
    g1_29dof_demo4_rotate_reward,
)
from holosoma.envs.marl.paired_motion_reference import PairedMotionReference
from holosoma.envs.marl.demo4_rotate_manager import Demo4RotateManager
from holosoma.envs.marl.plan5_push_manager import Plan5PushManager
from holosoma.utils.path import resolve_data_file_path


def test_demo4_uses_static_pull_runtime_and_fixed_positive_yaw_goal() -> None:
    term = g1_29dof_demo4_rotate_command.setup_terms["paired_motion_command"]
    assert term.params["paired_reference_file"] == DEMO4_ROTATE_RUNTIME_REFERENCE_FILE
    assert term.params["goal_yaw_degrees"] == 90.0
    assert term.params["maximum_success_tilt_degrees"] == 60.0
    assert term.params["motion_config"].start_at_timestep_zero_prob == 1.0
    assert DEMO4_ROTATE_RUNTIME_REFERENCE_FILE.endswith(
        "demo4_rotate/demo4_pull_pull_static_table_runtime.npz"
    )


def test_demo4_runtime_loads_as_316_frame_static_table_reference() -> None:
    resolved = resolve_data_file_path(DEMO4_ROTATE_RUNTIME_REFERENCE_FILE)
    with np.load(resolved, allow_pickle=False) as data:
        body_names = data["body_names"].astype(str).tolist()
        joint_names = data["joint_names"].astype(str).tolist()
    reference = PairedMotionReference(
        DEMO4_ROTATE_RUNTIME_REFERENCE_FILE,
        body_names,
        joint_names,
        device="cpu",
    )

    assert reference.num_frames == DEMO4_REFERENCE_FRAMES
    assert reference.fps == DEMO4_REFERENCE_FPS
    torch.testing.assert_close(
        reference.object_pos_w,
        reference.object_pos_w[:1].expand_as(reference.object_pos_w),
    )
    quaternion_alignment = torch.abs(
        torch.sum(reference.object_quat_w * reference.object_quat_w[:1], dim=-1)
    )
    torch.testing.assert_close(
        quaternion_alignment,
        torch.ones_like(quaternion_alignment),
    )
    torch.testing.assert_close(
        reference.object_lin_vel_w,
        torch.zeros_like(reference.object_lin_vel_w),
    )


def test_demo4_reward_has_only_motion_prior_regularization_and_yaw_task_terms() -> None:
    terms = g1_29dof_demo4_rotate_reward.terms
    assert list(terms) == [
        "motion_global_ref_position_error_exp",
        "motion_global_ref_orientation_error_exp",
        "motion_relative_body_position_error_exp",
        "motion_relative_body_orientation_error_exp",
        "motion_global_body_lin_vel",
        "motion_global_body_ang_vel",
        "action_rate_l2",
        "limits_dof_pos",
        "signed_yaw_potential_delta",
        "first_yaw_goal_bonus",
    ]
    assert terms["action_rate_l2"].weight == -0.1
    assert terms["limits_dof_pos"].weight == -10.0
    assert terms["signed_yaw_potential_delta"].weight == 10.0
    assert terms["first_yaw_goal_bonus"].weight == 5.0
    assert not any("object_global_ref" in name for name in terms)
    assert not any("contact" in name or "torso" in name for name in terms)


def test_demo4_experiment_preserves_reviewed_physics_and_dimensions() -> None:
    cfg = g1_29dof_demo4_rotate_baseline
    assert cfg.env_class.endswith("Demo4RotateManager")
    assert cfg.training.num_envs == 2048
    assert cfg.training.seed == 721
    assert cfg.algo.config.save_interval == 1000
    assert cfg.algo.config.module_dict.actor.input_dim == [
        "actor_obs",
        "teammate_obs",
        "table_obs",
    ]
    assert cfg.algo.config.module_dict.critic.input_dim == ["critic_obs"]
    assert cfg.robot.object.object_urdf_path.endswith(
        "objects_widetable_plan5_pull_training.urdf"
    )

    material = cfg.randomization.setup_terms[
        "set_object_rigid_body_material_startup"
    ].params
    assert material == {
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
    }
    assert DEMO4_REFERENCE_FRAMES == 316
    assert DEMO4_REFERENCE_FPS == 50
    assert DEMO4_EPISODE_SECONDS == 6.32
    assert g1_29dof_demo4_rotate_smoke.training.num_envs == 1


def test_demo4_non_timeout_terminals_take_priority_over_horizon() -> None:
    manager = object.__new__(Demo4RotateManager)
    results = {
        "yaw_goal_success": torch.tensor([True, True, False]),
        "reference_horizon": torch.tensor([True, True, True]),
        "clear_robot_fall": torch.tensor([False, True, False]),
        "table_physical_safety": torch.tensor([False, False, False]),
    }
    manager.termination_manager = SimpleNamespace(term_results=results)
    manager.reset_buf = torch.ones(3, dtype=torch.long)
    manager.time_out_buf = torch.ones(3, dtype=torch.bool)
    manager.extras = {
        "termination_terms": {
            name: value.clone() for name, value in results.items()
        }
    }

    with patch.object(Plan5PushManager, "_check_termination"):
        manager._check_termination()

    torch.testing.assert_close(
        manager.time_out_buf,
        torch.tensor([False, False, True]),
    )
    torch.testing.assert_close(
        manager.termination_manager.term_results["yaw_goal_success"],
        torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(
        manager.extras["termination_terms"]["yaw_goal_success"],
        torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(
        manager.termination_manager.term_results["reference_horizon"],
        torch.tensor([False, False, True]),
    )
    torch.testing.assert_close(
        manager.extras["termination_terms"]["reference_horizon"],
        torch.tensor([False, False, True]),
    )
