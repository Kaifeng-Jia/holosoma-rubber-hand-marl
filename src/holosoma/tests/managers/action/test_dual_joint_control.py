from types import SimpleNamespace

import torch

from holosoma.config_types.action import ActionTermCfg
from holosoma.managers.action.terms.marl import DualJointPositionActionTerm


class FakeSimulator:
    def __init__(self, num_envs: int, num_dof: int):
        self.agent_dof_pos = torch.zeros(num_envs, 2, num_dof)
        self.agent_dof_vel = torch.zeros_like(self.agent_dof_pos)
        self.applied_torques = None
        self.simulator_config = SimpleNamespace(sim=SimpleNamespace(control_decimation=2))

    def apply_agent_torques(self, torques: torch.Tensor) -> None:
        self.applied_torques = torques.clone()


def make_env(*, control_type: str = "T", clip_actions: bool = False):
    num_envs = 2
    num_dof = 3
    control = SimpleNamespace(
        clip_actions=clip_actions,
        action_clip_value=1.0,
        clip_torques=True,
        control_type=control_type,
        action_scale=2.0,
        action_scales_by_effort_limit_over_p_gain=False,
        stiffness={"joint": 10.0},
        damping={"joint": 1.0},
        integral={},
    )
    robot_config = SimpleNamespace(
        control=control,
        init_state=SimpleNamespace(default_joint_angles={f"joint_{i}": 0.0 for i in range(num_dof)}),
        dof_effort_limit_list=[100.0] * num_dof,
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        num_dof=num_dof,
        device="cpu",
        dof_names=[f"joint_{i}" for i in range(num_dof)],
        robot_config=robot_config,
        simulator=FakeSimulator(num_envs, num_dof),
        default_dof_pos=torch.zeros(num_envs, num_dof),
        torque_limits=torch.full((num_dof,), 5.0),
        sim_dt=0.01,
        log_dict={},
        _randomize_ctrl_delay=False,
        _pending_torque_rfi=(False, 0.0),
        randomization_manager=None,
    )
    return env


def make_term(env) -> DualJointPositionActionTerm:
    cfg = ActionTermCfg(func="unused")
    term = DualJointPositionActionTerm(cfg, env)
    term.setup()
    return term


def test_routes_agent_major_actions_without_cross_contamination():
    env = make_env(control_type="T")
    term = make_term(env)
    actions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3, 3.1, 3.2, 3.3],
        ]
    )

    term.process_actions(actions)

    torch.testing.assert_close(term.agent_actions[0, 0], actions[0, :3])
    torch.testing.assert_close(term.agent_actions[0, 1], actions[0, 3:])
    torch.testing.assert_close(term.agent_actions[1, 0], actions[1, :3])
    torch.testing.assert_close(term.agent_actions[1, 1], actions[1, 3:])


def test_applies_independent_torques_to_both_physical_robots():
    env = make_env(control_type="T")
    term = make_term(env)
    actions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            [-0.1, -0.2, -0.3, -1.1, -1.2, -1.3],
        ]
    )

    term.process_actions(actions)
    term.apply_actions()

    expected = actions.view(2, 2, 3) * 2.0
    torch.testing.assert_close(env.simulator.applied_torques, expected)
    torch.testing.assert_close(term.torques_substep[:, 0], expected)


def test_position_controller_uses_each_robots_own_state_and_clips():
    env = make_env(control_type="P", clip_actions=True)
    env.simulator.agent_dof_pos[0, 0] = torch.tensor([0.1, 0.2, 0.3])
    env.simulator.agent_dof_pos[0, 1] = torch.tensor([-0.1, -0.2, -0.3])
    env.simulator.agent_dof_vel[0, 0] = 0.5
    env.simulator.agent_dof_vel[0, 1] = -0.5
    term = make_term(env)

    term.process_actions(torch.tensor([[2.0, 2.0, 2.0, -2.0, -2.0, -2.0], [0.0] * 6]))
    term.apply_actions()

    expected_agent_0 = torch.tensor([5.0, 5.0, 5.0])
    expected_agent_1 = torch.tensor([-5.0, -5.0, -5.0])
    torch.testing.assert_close(env.simulator.applied_torques[0, 0], expected_agent_0)
    torch.testing.assert_close(env.simulator.applied_torques[0, 1], expected_agent_1)
    assert env.log_dict["action_clip_frac"].item() == 0.5


def test_registered_dual_action_config_is_opt_in():
    from holosoma.config_values.action import DEFAULTS

    assert DEFAULTS["g1_29dof_joint_pos"].terms["joint_control"].func.endswith(
        ":JointPositionActionTerm"
    )
    assert DEFAULTS["g1_29dof_dual_joint_pos"].terms["joint_control"].func.endswith(
        ":DualJointPositionActionTerm"
    )
