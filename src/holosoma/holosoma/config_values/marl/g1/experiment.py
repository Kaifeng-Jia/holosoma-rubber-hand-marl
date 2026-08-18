"""Experiment presets for Plan 5 cooperative Push A1."""

from dataclasses import replace

from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values import action, command, robot, simulator
from holosoma.config_values.marl.g1.observation import g1_29dof_plan5_observation
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


plan5_smoke_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        )
    }
)

g1_29dof_plan5_push_smoke = replace(
    g1_29dof_wbt_w_object,
    env_class="holosoma.envs.marl.plan5_push_manager.Plan5PushManager",
    training=replace(
        g1_29dof_wbt_w_object.training,
        project="Plan5Preflight",
        name="dual_rubberhand_a1_reset_smoke",
        num_envs=1,
        headless=True,
    ),
    simulator=replace(
        simulator.isaacsim_dual_robot,
        config=replace(
            simulator.isaacsim_dual_robot.config,
            scene=replace(simulator.isaacsim_dual_robot.config.scene, env_spacing=4.0),
            sim=replace(
                simulator.isaacsim_dual_robot.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path=(
                "holosoma/data/motions/g1_29dof/whole_body_tracking/"
                "objects_widetable_plan5_preflight.urdf"
            ),
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    observation=g1_29dof_plan5_observation,
    action=action.g1_29dof_dual_joint_pos,
    reward=RewardManagerCfg(),
    termination=plan5_smoke_termination,
    randomization=RandomizationManagerCfg(),
    command=command.g1_29dof_paired_a1_command,
    curriculum=CurriculumManagerCfg(),
    nightly=None,
)

__all__ = ["g1_29dof_plan5_push_smoke"]
