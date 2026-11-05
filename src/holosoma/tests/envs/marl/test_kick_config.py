"""Static contracts for the isolated cooperative Kick configuration."""

from xml.etree import ElementTree

from holosoma.agents.mappo.initialization import (
    PLAN5_ACTION_DIM,
    PLAN5_ACTOR_OBS_DIM,
    PLAN5_CRITIC_OBS_DIM,
)
from holosoma.config_values.marl.g1.kick_command import (
    PLAN5_KICK_RUNTIME_REFERENCE_FILE,
    g1_29dof_paired_kick_command,
)
from holosoma.config_values.marl.g1.kick_experiment import (
    PLAN5_KICK_EPISODE_SECONDS,
    PLAN5_KICK_PROJECT,
    PLAN5_KICK_REFERENCE_FPS,
    PLAN5_KICK_REFERENCE_FRAMES,
    PLAN5_KICK_TABLE_URDF,
    PLAN5_KICK_TRAINING_NAME,
    g1_29dof_plan5_kick_baseline,
    g1_29dof_plan5_kick_smoke,
)
from holosoma.config_values.marl.g1.reward import g1_29dof_plan5_push_reward
from holosoma.config_values.marl.g1.termination import g1_29dof_plan5_push_termination
from holosoma.utils.path import resolve_data_file_path


def test_kick_command_points_to_the_accepted_mirrored_runtime() -> None:
    term = g1_29dof_paired_kick_command.setup_terms["paired_motion_command"]
    assert term.func.endswith(":PairedA1MotionCommand")
    assert term.params["paired_reference_file"] == PLAN5_KICK_RUNTIME_REFERENCE_FILE
    assert PLAN5_KICK_RUNTIME_REFERENCE_FILE.endswith(
        "kick/plan5_attempt09_dual_kick_mirrored_runtime.npz"
    )
    assert term.params["motion_config"].motion_file.endswith(
        "kick/sub16_largetable_028_kick_mj_fps50_w_obj.npz"
    )
    assert "lateral_spacing_m" not in term.params


def test_kick_baseline_is_isolated_but_reuses_plan5_learning_contract() -> None:
    cfg = g1_29dof_plan5_kick_baseline
    assert cfg.training.project == PLAN5_KICK_PROJECT
    assert cfg.training.name == PLAN5_KICK_TRAINING_NAME
    assert cfg.training.num_envs == 2048
    assert cfg.training.seed == 721
    assert cfg.command is g1_29dof_paired_kick_command
    assert cfg.reward is g1_29dof_plan5_push_reward
    assert cfg.termination is g1_29dof_plan5_push_termination
    assert cfg.robot.object.object_urdf_path == PLAN5_KICK_TABLE_URDF
    assert PLAN5_KICK_TABLE_URDF.endswith(
        "objects_widetable_plan5_pull_training.urdf"
    )
    assert PLAN5_ACTOR_OBS_DIM == 158
    assert PLAN5_CRITIC_OBS_DIM == 527
    assert PLAN5_ACTION_DIM == 29
    assert list(cfg.observation.groups) == ["actor_obs", "teammate_obs", "critic_obs"]


def test_kick_uses_frozen_physics_and_reference_horizon() -> None:
    cfg = g1_29dof_plan5_kick_baseline
    material = cfg.randomization.setup_terms[
        "set_object_rigid_body_material_startup"
    ].params
    assert material == {
        "static_friction": 0.5,
        "dynamic_friction": 0.5,
        "restitution": 0.0,
    }
    root = ElementTree.parse(resolve_data_file_path(PLAN5_KICK_TABLE_URDF)).getroot()
    inertial = root.find("./link/inertial")
    assert inertial is not None
    assert float(inertial.find("mass").attrib["value"]) == 20.0
    assert inertial.find("origin").attrib["xyz"] == "0 0.015111745244133 0"
    inertia = inertial.find("inertia").attrib
    assert [float(inertia[name]) for name in ("ixx", "iyy", "izz")] == [
        3.95048914424628,
        4.36041519206568,
        0.62774975216702,
    ]
    assert PLAN5_KICK_REFERENCE_FRAMES == 298
    assert PLAN5_KICK_REFERENCE_FPS == 50
    assert PLAN5_KICK_EPISODE_SECONDS == 5.96
    assert cfg.simulator.config.sim.max_episode_length_s == 5.96
    assert g1_29dof_plan5_kick_smoke.training.num_envs == 1
    assert list(g1_29dof_plan5_kick_smoke.termination.terms) == ["timeout"]


def test_kick_reward_remains_reference_guided_without_task_hacks() -> None:
    terms = g1_29dof_plan5_kick_baseline.reward.terms
    assert "object_global_ref_position_error_exp" in terms
    assert "object_global_ref_orientation_error_exp" in terms
    assert "action_rate_l2" in terms
