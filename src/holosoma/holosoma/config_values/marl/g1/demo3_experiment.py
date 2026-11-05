"""Fully isolated experiment presets for competitive Demo 3 tug-of-war."""

from dataclasses import replace

from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_values import action, robot, simulator
from holosoma.config_values.marl.g1.demo3_command import g1_29dof_demo3_tug_command
from holosoma.config_values.marl.g1.demo3_observation import g1_29dof_demo3_observation
from holosoma.config_values.marl.g1.demo3_reward import g1_29dof_demo3_tug_reward
from holosoma.config_values.marl.g1.demo3_termination import (
    g1_29dof_demo3_tug_termination,
)
from holosoma.config_values.marl.g1.randomization import (
    g1_29dof_plan5_fixed_object_material,
)
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


DEMO3_REFERENCE_FRAMES = 317
DEMO3_REFERENCE_FPS = 50
DEMO3_EPISODE_SECONDS = DEMO3_REFERENCE_FRAMES / DEMO3_REFERENCE_FPS

_demo3_algo = replace(
    g1_29dof_wbt_w_object.algo,
    config=replace(
        g1_29dof_wbt_w_object.algo.config,
        load_optimizer=False,
        save_interval=1000,
        module_dict=replace(
            g1_29dof_wbt_w_object.algo.config.module_dict,
            actor=replace(
                g1_29dof_wbt_w_object.algo.config.module_dict.actor,
                input_dim=["actor_obs", "teammate_obs", "table_obs"],
            ),
            critic=replace(
                g1_29dof_wbt_w_object.algo.config.module_dict.critic,
                input_dim=["critic_obs"],
            ),
        ),
    ),
)

g1_29dof_demo3_tug_smoke = replace(
    g1_29dof_wbt_w_object,
    env_class="holosoma.envs.marl.demo3_tug_manager.Demo3TugManager",
    training=replace(
        g1_29dof_wbt_w_object.training,
        project="Demo3Tug",
        name="square_table_diagonal_tug_smoke",
        num_envs=1,
        headless=True,
        seed=721,
    ),
    algo=_demo3_algo,
    simulator=replace(
        simulator.isaacsim_dual_robot,
        config=replace(
            simulator.isaacsim_dual_robot.config,
            scene=replace(simulator.isaacsim_dual_robot.config.scene, env_spacing=4.0),
            sim=replace(
                simulator.isaacsim_dual_robot.config.sim,
                max_episode_length_s=DEMO3_EPISODE_SECONDS,
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
                "objects_squaretable_demo3_training.urdf"
            ),
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    observation=g1_29dof_demo3_observation,
    action=action.g1_29dof_dual_joint_pos,
    reward=g1_29dof_demo3_tug_reward,
    termination=g1_29dof_demo3_tug_termination,
    randomization=g1_29dof_plan5_fixed_object_material,
    command=g1_29dof_demo3_tug_command,
    curriculum=CurriculumManagerCfg(),
    nightly=None,
)

g1_29dof_demo3_tug_baseline = replace(
    g1_29dof_demo3_tug_smoke,
    training=replace(
        g1_29dof_demo3_tug_smoke.training,
        name="square_table_diagonal_tug_mappo",
        num_envs=2048,
    ),
)

__all__ = [
    "DEMO3_EPISODE_SECONDS",
    "DEMO3_REFERENCE_FPS",
    "DEMO3_REFERENCE_FRAMES",
    "g1_29dof_demo3_tug_baseline",
    "g1_29dof_demo3_tug_smoke",
]
