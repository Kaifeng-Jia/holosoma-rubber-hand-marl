from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from holosoma.config_types.command import CommandTermCfg, MotionConfig
from holosoma.managers.command.terms.marl import PairedA1MotionCommand


def write_motion(path, frames: int = 5) -> None:
    joint_names = np.array(["joint_0", "joint_1"])
    body_names = np.array(["pelvis", "torso_link"])
    joint_pos = np.zeros((frames, 9), dtype=np.float32)
    joint_vel = np.zeros((frames, 8), dtype=np.float32)
    joint_pos[:, 7:] = np.arange(frames, dtype=np.float32)[:, None]
    joint_vel[:, 6:] = 0.5
    body_pos = np.zeros((frames, 2, 3), dtype=np.float32)
    body_pos[:, 0, 1] = 1.0
    body_pos[:, 1, 1] = 1.2
    body_quat = np.zeros((frames, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin_vel = np.zeros_like(body_pos)
    body_ang_vel = np.zeros_like(body_pos)
    object_pos = np.zeros((frames, 3), dtype=np.float32)
    object_pos[:, 1] = 2.0
    object_quat = np.zeros((frames, 4), dtype=np.float32)
    object_quat[:, 0] = 1.0

    np.savez(
        path,
        fps=np.array([50]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        object_pos_w=object_pos,
        object_quat_w=object_quat,
        object_lin_vel_w=np.zeros_like(object_pos),
        object_ang_vel_w=np.zeros_like(object_pos),
        joint_names=joint_names,
        body_names=body_names,
    )


class FakeSimulator:
    def __init__(self):
        self._body_list = ["pelvis", "torso_link"]
        self.dof_names = ["joint_0", "joint_1"]
        self.scene = SimpleNamespace(
            env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        )
        self.agent_dof_pos = torch.zeros(2, 2, 2)
        self.agent_dof_vel = torch.zeros_like(self.agent_dof_pos)
        self.agent_rigid_body_pos = torch.zeros(2, 2, 2, 3)
        self.agent_rigid_body_rot = torch.zeros(2, 2, 2, 4)
        self.all_root_states = torch.zeros(2, 13)
        self.root_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.dof_writes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.object_writes: list[tuple[list[str], torch.Tensor, torch.Tensor]] = []

    def get_actor_indices(self, name, env_ids=None):
        assert name == "object"
        return torch.tensor([0, 1])

    def set_agent_root_states(self, env_ids, root_states):
        self.root_writes.append((env_ids.clone(), root_states.clone()))

    def set_agent_dof_states(self, env_ids, dof_pos, dof_vel):
        self.dof_writes.append((env_ids.clone(), dof_pos.clone(), dof_vel.clone()))
        self.agent_dof_pos[env_ids] = dof_pos
        self.agent_dof_vel[env_ids] = dof_vel

    def set_actor_states(self, names, env_ids, states):
        self.object_writes.append((names, env_ids.clone(), states.clone()))
        self.all_root_states[env_ids] = states


def make_command(tmp_path, *, evaluating: bool = True):
    motion_path = tmp_path / "paired_test.npz"
    write_motion(motion_path)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        simulator=FakeSimulator(),
        is_evaluating=evaluating,
    )
    cfg = CommandTermCfg(
        func="unused",
        params={
            "motion_config": MotionConfig(
                motion_file=str(motion_path),
                body_name_ref=["torso_link"],
                body_names_to_track=["pelvis", "torso_link"],
            ),
            "lateral_spacing_m": 0.8,
        },
    )
    command = PairedA1MotionCommand(cfg, env)
    command.setup()
    return command, env


def test_joint_reset_uses_one_phase_for_two_robots_and_one_object(tmp_path):
    command, env = make_command(tmp_path)
    env_ids = torch.tensor([0, 1])

    command.reset(env_ids)

    assert command.time_steps.tolist() == [0, 0]
    assert len(env.simulator.root_writes) == 1
    _, roots = env.simulator.root_writes[0]
    torch.testing.assert_close(roots[0, :, 0], torch.tensor([-0.4, 0.4]))
    torch.testing.assert_close(roots[1, :, 0], torch.tensor([9.6, 10.4]))
    torch.testing.assert_close(roots[:, :, 1], torch.ones(2, 2))
    assert len(env.simulator.dof_writes) == 1
    assert len(env.simulator.object_writes) == 1
    names, written_env_ids, object_states = env.simulator.object_writes[0]
    assert names == ["object"]
    assert written_env_ids.tolist() == [0, 1]
    torch.testing.assert_close(object_states[:, :3], torch.tensor([[0.0, 2.0, 0.0], [10.0, 2.0, 0.0]]))


def test_shared_phase_advances_once_per_physical_environment(tmp_path):
    command, _ = make_command(tmp_path)
    command.reset(None)

    command.step()
    command.step()

    assert command.time_steps.tolist() == [2, 2]
    torch.testing.assert_close(command.agent_joint_pos[:, 0], command.agent_joint_pos[:, 1])
    torch.testing.assert_close(command.agent_joint_pos[:, 0], torch.full((2, 2), 2.0))


def test_partial_reset_keeps_other_environment_phase(tmp_path):
    command, env = make_command(tmp_path)
    command.reset(None)
    command.time_steps[:] = torch.tensor([3, 4])
    env.simulator.object_writes.clear()

    command.reset(torch.tensor([1]))

    assert command.time_steps.tolist() == [3, 0]
    assert len(env.simulator.object_writes) == 1
    assert env.simulator.object_writes[0][1].tolist() == [1]


def test_training_reset_samples_one_frame_per_environment_not_per_agent(tmp_path):
    torch.manual_seed(4)
    command, _ = make_command(tmp_path, evaluating=False)

    command.reset(None)

    assert command.time_steps.shape == (2,)
    assert torch.all(command.time_steps >= 0)
    assert torch.all(command.time_steps < command.reference.num_frames - 1)
    torch.testing.assert_close(command.agent_joint_pos[:, 0], command.agent_joint_pos[:, 1])
