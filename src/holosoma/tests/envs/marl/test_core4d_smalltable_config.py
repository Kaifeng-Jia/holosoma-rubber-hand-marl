"""Static training contract for the isolated CORE4D small-table demo."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from holosoma.config_values.marl.g1.core4d_smalltable_command import (
    CORE4D_SMALLTABLE_REFERENCE_FPS,
    CORE4D_SMALLTABLE_REFERENCE_FRAMES,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    CORE4D_SMALLTABLE_EPISODE_SECONDS,
    g1_29dof_core4d_smalltable_baseline,
    g1_29dof_core4d_smalltable_smoke,
)
from holosoma.config_values.marl.g1.core4d_smalltable_reward import (
    g1_29dof_core4d_smalltable_reward,
)
from holosoma.envs.marl.core4d_smalltable_manager import Core4DSmallTableManager
from holosoma.envs.marl.plan5_push_manager import Plan5PushManager


def test_core4d_smalltable_uses_frozen_164d_shared_actor_contract() -> None:
    cfg = g1_29dof_core4d_smalltable_baseline
    assert cfg.env_class.endswith("Core4DSmallTableManager")
    assert cfg.training.num_envs == 2048
    assert cfg.training.seed == 721
    assert cfg.algo.config.save_interval == 1000
    assert cfg.algo.config.module_dict.actor.input_dim == [
        "actor_obs",
        "teammate_obs",
        "table_obs",
    ]
    assert cfg.algo.config.module_dict.critic.input_dim == ["critic_obs"]
    assert cfg.robot.asset.urdf_file.endswith("main_mesh_collision_rubberhand.urdf")
    assert "hemisphere" not in cfg.robot.asset.urdf_file.lower()
    assert cfg.robot.object.object_urdf_path.endswith(
        "objects_core4d_desk001_small_training.urdf"
    )
    assert cfg.simulator.config.sim.fps == 200
    assert cfg.simulator.config.sim.control_decimation == 4
    assert CORE4D_SMALLTABLE_REFERENCE_FRAMES == 687
    assert CORE4D_SMALLTABLE_REFERENCE_FPS == 50
    assert CORE4D_SMALLTABLE_EPISODE_SECONDS == 13.74
    assert g1_29dof_core4d_smalltable_smoke.training.num_envs == 1


def test_core4d_smalltable_reuses_plan5_tracking_reward_without_task_shaping() -> None:
    terms = g1_29dof_core4d_smalltable_reward.terms
    assert list(terms) == [
        "motion_global_ref_position_error_exp",
        "motion_global_ref_orientation_error_exp",
        "motion_relative_body_position_error_exp",
        "motion_relative_body_orientation_error_exp",
        "motion_global_body_lin_vel",
        "motion_global_body_ang_vel",
        "action_rate_l2",
        "limits_dof_pos",
        "undesired_contacts",
        "object_global_ref_position_error_exp",
        "object_global_ref_orientation_error_exp",
    ]
    assert not any("goal" in name or "yaw" in name for name in terms)


def test_core4d_smalltable_has_fixed_material_and_non_looping_termination() -> None:
    cfg = g1_29dof_core4d_smalltable_baseline
    material = cfg.randomization.setup_terms[
        "set_object_rigid_body_material_startup"
    ].params
    assert material == {
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
    }
    assert list(cfg.termination.terms) == ["reference_horizon", "joint_bad_tracking"]
    assert cfg.termination.terms["reference_horizon"].is_timeout is True


def test_physical_failure_takes_priority_over_simultaneous_reference_horizon() -> None:
    manager = object.__new__(Core4DSmallTableManager)
    results = {
        "reference_horizon": torch.tensor([True, True, False]),
        "joint_bad_tracking": torch.tensor([False, True, True]),
    }
    manager.termination_manager = SimpleNamespace(term_results=results)
    manager.time_out_buf = torch.tensor([True, True, False])
    manager.extras = {
        "termination_terms": {name: value.clone() for name, value in results.items()}
    }

    with patch.object(Plan5PushManager, "_check_termination"):
        manager._check_termination()

    torch.testing.assert_close(
        manager.time_out_buf,
        torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(
        manager.termination_manager.term_results["reference_horizon"],
        torch.tensor([True, False, False]),
    )
