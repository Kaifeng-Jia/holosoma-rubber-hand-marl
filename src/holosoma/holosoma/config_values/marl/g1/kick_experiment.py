"""Isolated reference-guided cooperative Kick experiment presets."""

from dataclasses import replace

from holosoma.config_values import robot, simulator
from holosoma.config_values.marl.g1.experiment import (
    g1_29dof_plan5_push_smoke,
    plan5_smoke_termination,
)
from holosoma.config_values.marl.g1.kick_command import g1_29dof_paired_kick_command
from holosoma.config_values.marl.g1.randomization import (
    g1_29dof_plan5_fixed_object_material,
)
from holosoma.config_values.marl.g1.reward import g1_29dof_plan5_push_reward
from holosoma.config_values.marl.g1.termination import g1_29dof_plan5_push_termination


PLAN5_KICK_REFERENCE_FRAMES = 298
PLAN5_KICK_REFERENCE_FPS = 50
PLAN5_KICK_EPISODE_SECONDS = PLAN5_KICK_REFERENCE_FRAMES / PLAN5_KICK_REFERENCE_FPS
PLAN5_KICK_PROJECT = "Plan5Kick"
PLAN5_KICK_TRAINING_NAME = "mirrored_dual_kick_shared_reward_training"
PLAN5_KICK_TABLE_URDF = (
    "holosoma/data/motions/g1_29dof/whole_body_tracking/"
    "objects_widetable_plan5_pull_training.urdf"
)

_kick_simulator = replace(
    simulator.isaacsim_dual_robot,
    config=replace(
        simulator.isaacsim_dual_robot.config,
        scene=replace(simulator.isaacsim_dual_robot.config.scene, env_spacing=4.0),
        sim=replace(
            simulator.isaacsim_dual_robot.config.sim,
            max_episode_length_s=PLAN5_KICK_EPISODE_SECONDS,
        ),
    ),
)

_kick_robot = replace(
    robot.g1_29dof_w_object,
    asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
    object=replace(
        robot.g1_29dof_w_object.object,
        object_urdf_path=PLAN5_KICK_TABLE_URDF,
    ),
    init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
)

g1_29dof_plan5_kick_smoke = replace(
    g1_29dof_plan5_push_smoke,
    training=replace(
        g1_29dof_plan5_push_smoke.training,
        project=PLAN5_KICK_PROJECT,
        name="mirrored_dual_kick_reset_smoke",
        num_envs=1,
        headless=True,
        seed=721,
    ),
    simulator=_kick_simulator,
    robot=_kick_robot,
    reward=g1_29dof_plan5_push_reward,
    termination=plan5_smoke_termination,
    randomization=g1_29dof_plan5_fixed_object_material,
    command=g1_29dof_paired_kick_command,
)

g1_29dof_plan5_kick_baseline = replace(
    g1_29dof_plan5_kick_smoke,
    training=replace(
        g1_29dof_plan5_kick_smoke.training,
        name=PLAN5_KICK_TRAINING_NAME,
        num_envs=2048,
    ),
    termination=g1_29dof_plan5_push_termination,
)

__all__ = [
    "g1_29dof_plan5_kick_baseline",
    "g1_29dof_plan5_kick_smoke",
]
