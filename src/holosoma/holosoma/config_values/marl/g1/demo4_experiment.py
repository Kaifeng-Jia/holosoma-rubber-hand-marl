"""Isolated experiment presets for cooperative Demo 4 table rotation."""

from dataclasses import replace

from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_values import action, robot, simulator
from holosoma.config_values.marl.g1.demo4_command import (
    g1_29dof_demo4_rotate_command,
)
from holosoma.config_values.marl.g1.demo4_observation import (
    g1_29dof_demo4_rotate_observation,
)
from holosoma.config_values.marl.g1.demo4_reward import g1_29dof_demo4_rotate_reward
from holosoma.config_values.marl.g1.demo4_termination import (
    g1_29dof_demo4_rotate_termination,
)
from holosoma.config_values.marl.g1.randomization import (
    g1_29dof_plan5_fixed_object_material,
)
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


DEMO4_REFERENCE_FRAMES = 316
DEMO4_REFERENCE_FPS = 50
DEMO4_EPISODE_SECONDS = DEMO4_REFERENCE_FRAMES / DEMO4_REFERENCE_FPS

_demo4_algo = replace(
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

g1_29dof_demo4_rotate_smoke = replace(
    g1_29dof_wbt_w_object,
    env_class="holosoma.envs.marl.demo4_rotate_manager.Demo4RotateManager",
    training=replace(
        g1_29dof_wbt_w_object.training,
        project="Demo4Rotate",
        name="rectangular_diagonal_pull_rotate90_smoke",
        num_envs=1,
        headless=True,
        seed=721,
    ),
    algo=_demo4_algo,
    simulator=replace(
        simulator.isaacsim_dual_robot,
        config=replace(
            simulator.isaacsim_dual_robot.config,
            scene=replace(
                simulator.isaacsim_dual_robot.config.scene,
                env_spacing=4.0,
            ),
            sim=replace(
                simulator.isaacsim_dual_robot.config.sim,
                max_episode_length_s=DEMO4_EPISODE_SECONDS,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path=(
                "holosoma/data/motions/g1_29dof/whole_body_tracking/"
                "objects_widetable_plan5_pull_training.urdf"
            ),
        ),
        init_state=replace(
            robot.g1_29dof_w_object.init_state,
            pos=[0.0, 0.0, 0.76],
        ),
    ),
    observation=g1_29dof_demo4_rotate_observation,
    action=action.g1_29dof_dual_joint_pos,
    reward=g1_29dof_demo4_rotate_reward,
    termination=g1_29dof_demo4_rotate_termination,
    randomization=g1_29dof_plan5_fixed_object_material,
    command=g1_29dof_demo4_rotate_command,
    curriculum=CurriculumManagerCfg(),
    nightly=None,
)

g1_29dof_demo4_rotate_baseline = replace(
    g1_29dof_demo4_rotate_smoke,
    training=replace(
        g1_29dof_demo4_rotate_smoke.training,
        name="rectangular_diagonal_pull_rotate90_mappo",
        num_envs=2048,
    ),
)

__all__ = [
    "DEMO4_EPISODE_SECONDS",
    "DEMO4_REFERENCE_FPS",
    "DEMO4_REFERENCE_FRAMES",
    "g1_29dof_demo4_rotate_baseline",
    "g1_29dof_demo4_rotate_smoke",
]
