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
    assert list(cfg.observation.groups) == ["actor_obs", "teammate_obs"]
    assert cfg.reward.terms == {}
    assert cfg.randomization.setup_terms == {}
