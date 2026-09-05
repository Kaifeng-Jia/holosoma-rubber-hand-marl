from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from holosoma.agents.callbacks.recording import EvalRecordingCallback
from holosoma.config_types.eval_callback import RecordingConfig
from tests.managers.command.test_paired_a1_command import make_command


class FakeActionManager:
    def __init__(self, num_envs: int, num_agents: int, num_dof: int, decimation: int):
        term = SimpleNamespace(
            _actions_after_delay=torch.zeros(num_envs, num_agents, num_dof),
            action_scales=torch.ones(num_dof),
            torques=torch.zeros(num_envs, num_agents, num_dof),
            torques_substep=torch.zeros(num_envs, decimation, num_agents, num_dof),
            dof_pos_substep=torch.zeros(num_envs, decimation, num_agents, num_dof),
            dof_vel_substep=torch.zeros(num_envs, decimation, num_agents, num_dof),
        )
        self._terms = [("joint_control", term)]

    def iter_terms(self):
        return iter(self._terms)


def make_recording_env(tmp_path):
    command, env = make_command(tmp_path)
    command.reset(None)
    sim = env.simulator
    sim.body_names = list(sim._body_list)
    sim.agent_root_states = torch.zeros(2, 2, 13)
    sim.agent_root_states[..., 3] = 1.0
    sim.agent_rigid_body_pos.copy_(command.agent_body_pos_w)
    sim.agent_rigid_body_rot.copy_(command.agent_body_quat_w)
    sim.agent_contact_forces = torch.zeros(2, 2, 2, 3)
    sim.agent_contact_forces_history = torch.zeros(2, 2, 3, 2, 3)
    sim.simulator_config = SimpleNamespace(
        sim=SimpleNamespace(control_decimation=4),
    )

    env.dt = 0.02
    env.sim_dt = 0.005
    env.num_agents = 2
    env.num_dof = 2
    env.episode_length_buf = torch.tensor([3, 4])
    env.default_dof_pos = torch.zeros(2, 2)
    env.command_manager = SimpleNamespace(
        get_state=lambda name: command if name == "paired_motion_command" else None
    )
    env.action_manager = FakeActionManager(2, 2, 2, 4)
    env.termination_manager = SimpleNamespace(
        active_terms=["timeout", "joint_bad_tracking"]
    )
    env.robot_config = SimpleNamespace(
        dof_effort_limit_list=[1.0, 1.0],
        dof_pos_lower_limit_list=[-1.0, -1.0],
        dof_pos_upper_limit_list=[1.0, 1.0],
        dof_vel_limit_list=[2.0, 2.0],
        asset=SimpleNamespace(asset_root=tmp_path, urdf_file="rubber_hand.urdf"),
    )
    return command, env


def test_paired_recording_writes_agent_major_channels_and_shared_object_once(tmp_path):
    _, env = make_recording_env(tmp_path)
    output = tmp_path / "plan5_recording.npz"
    training_loop = SimpleNamespace(
        device="cpu",
        actor_obs_keys=["actor_obs", "teammate_obs"],
        _unwrap_env=lambda: env,
    )
    callback = EvalRecordingCallback(
        RecordingConfig(enabled=True, output_path=str(output), env_id=0),
        training_loop,
    )
    callback.on_pre_evaluate_policy()
    actor_state = {
        "step": 0,
        "obs": {
            "actor_obs": torch.zeros(2, 2, 154),
            "teammate_obs": torch.zeros(2, 2, 4),
        },
        "actions": torch.zeros(2, 2, 2),
    }
    callback.on_pre_eval_env_step(actor_state)
    actor_state.update(
        rewards=torch.tensor([0.25, 0.5]),
        dones=torch.tensor([True, False]),
        extras={
            "time_outs": torch.tensor([False, False]),
            "termination_terms": {
                "timeout": torch.tensor([False, False]),
                "joint_bad_tracking": torch.tensor([True, False]),
            },
        },
    )
    callback.on_post_eval_env_step(actor_state)
    callback.on_post_evaluate_policy()

    with np.load(output) as data:
        assert data["policy_actor_obs"].shape == (1, 2, 158)
        assert data["ref_joint_pos"].shape == (1, 2, 2)
        assert data["pre_root_pos"].shape == (1, 2, 3)
        assert data["contact_forces_w"].shape == (1, 2, 2, 3)
        assert data["dof_pos"].shape == (1, 2, 2)
        assert data["torques_substep"].shape == (1, 2, 4, 2)
        assert data["ref_object_pos_w"].shape == (1, 3)
        assert data["object_pos_w"].shape == (1, 3)
        assert data["termination_term_joint_bad_tracking"].tolist() == [True]
        metadata = json.loads(data["_metadata_json"].item())
        assert metadata["agent_order"] == ["robot_0", "robot_1"]
        assert metadata["multi_agent_layout"] == "agent-major"
