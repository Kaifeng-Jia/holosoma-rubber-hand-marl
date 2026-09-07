"""Isolated MAPPO experiment presets for the CORE4D small-table demo."""

from dataclasses import replace

from holosoma.config_types.curriculum import CurriculumManagerCfg
from holosoma.config_values import action, robot, simulator
from holosoma.config_values.marl.g1.core4d_smalltable_command import (
    CORE4D_SMALLTABLE_REFERENCE_FPS,
    CORE4D_SMALLTABLE_REFERENCE_FRAMES,
    g1_29dof_core4d_smalltable_command,
)
from holosoma.config_values.marl.g1.core4d_smalltable_observation import (
    g1_29dof_core4d_smalltable_observation,
)
from holosoma.config_values.marl.g1.core4d_smalltable_reward import (
    g1_29dof_core4d_smalltable_reward,
    with_interaction_reward_term,
)
from holosoma.config_values.marl.g1.core4d_smalltable_termination import (
    g1_29dof_core4d_smalltable_termination,
)
from holosoma.config_values.marl.g1.randomization import (
    g1_29dof_plan5_fixed_object_material,
)
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object


CORE4D_SMALLTABLE_OBJECT_URDF = (
    "holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "objects_core4d_desk001_small_training.urdf"
)
CORE4D_SMALLTABLE_EPISODE_SECONDS = (
    CORE4D_SMALLTABLE_REFERENCE_FRAMES / CORE4D_SMALLTABLE_REFERENCE_FPS
)

_core4d_smalltable_algo = replace(
    g1_29dof_wbt_w_object.algo,
    config=replace(
        g1_29dof_wbt_w_object.algo.config,
        load_optimizer=False,
        num_learning_iterations=12000,
        save_interval=2000,
        module_dict=replace(
            g1_29dof_wbt_w_object.algo.config.module_dict,
            actor=replace(
                g1_29dof_wbt_w_object.algo.config.module_dict.actor,
                input_dim=["actor_obs", "teammate_obs"],
            ),
            critic=replace(
                g1_29dof_wbt_w_object.algo.config.module_dict.critic,
                input_dim=["critic_obs"],
            ),
        ),
    ),
)

g1_29dof_core4d_smalltable_smoke = replace(
    g1_29dof_wbt_w_object,
    env_class="holosoma.envs.marl.core4d_smalltable_manager.Core4DSmallTableManager",
    training=replace(
        g1_29dof_wbt_w_object.training,
        project="Core4DSmallTable",
        name="paired_reference_smoke",
        num_envs=1,
        headless=True,
        seed=721,
    ),
    algo=_core4d_smalltable_algo,
    simulator=replace(
        simulator.isaacsim_dual_robot,
        config=replace(
            simulator.isaacsim_dual_robot.config,
            scene=replace(
                simulator.isaacsim_dual_robot.config.scene,
                env_spacing=5.0,
            ),
            sim=replace(
                simulator.isaacsim_dual_robot.config.sim,
                max_episode_length_s=CORE4D_SMALLTABLE_EPISODE_SECONDS,
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
            object_urdf_path=CORE4D_SMALLTABLE_OBJECT_URDF,
        ),
        init_state=replace(
            robot.g1_29dof_w_object.init_state,
            pos=[0.0, 0.0, 0.76],
        ),
    ),
    observation=g1_29dof_core4d_smalltable_observation,
    action=action.g1_29dof_dual_joint_pos,
    reward=g1_29dof_core4d_smalltable_reward,
    termination=g1_29dof_core4d_smalltable_termination,
    randomization=g1_29dof_plan5_fixed_object_material,
    command=g1_29dof_core4d_smalltable_command,
    curriculum=CurriculumManagerCfg(),
    nightly=None,
)

g1_29dof_core4d_smalltable_baseline = replace(
    g1_29dof_core4d_smalltable_smoke,
    training=replace(
        g1_29dof_core4d_smalltable_smoke.training,
        name="paired_reference_mappo_fresh",
        num_envs=2048,
    ),
)

def with_interaction_mesh_reward(config, reference_file: str):
    """Opt in to the approved relation term; keep all other settings intact."""
    return replace(config, reward=with_interaction_reward_term(config.reward, reference_file))


__all__ = [
    "CORE4D_SMALLTABLE_EPISODE_SECONDS",
    "CORE4D_SMALLTABLE_OBJECT_URDF",
    "g1_29dof_core4d_smalltable_baseline",
    "g1_29dof_core4d_smalltable_smoke",
    "with_interaction_mesh_reward",
]
