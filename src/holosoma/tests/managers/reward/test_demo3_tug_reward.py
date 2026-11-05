"""Per-agent reward sign and aggregation tests for Demo 3."""

from types import SimpleNamespace

import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.command.terms.demo3_tug import Demo3TugMotionCommand
from holosoma.managers.reward.demo3_manager import Demo3AgentRewardManager
from holosoma.managers.reward.terms.demo3_tug import signed_table_progress_velocity


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        return self.command if name == "paired_motion_command" else None


def _env():
    command = object.__new__(Demo3TugMotionCommand)
    command.agent_pull_axis_w = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    command.object_indices_in_simulator = torch.tensor([0])
    simulator = SimpleNamespace(
        all_root_states=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        num_agents=2,
        device="cpu",
        max_episode_length_s=1.0,
        simulator=simulator,
        logger=None,
    )
    command.env = env
    env.command_manager = _CommandManager(command)
    return env


def test_same_table_velocity_rewards_competitors_with_opposite_signs() -> None:
    env = _env()
    raw = signed_table_progress_velocity(env)
    torch.testing.assert_close(raw, torch.tensor([[0.4, -0.4]]))

    cfg = RewardManagerCfg(
        terms={
            "progress": RewardTermCfg(
                func=(
                    "holosoma.managers.reward.terms.demo3_tug:"
                    "signed_table_progress_velocity"
                ),
                weight=2.0,
            )
        }
    )
    manager = Demo3AgentRewardManager(cfg, env, "cpu")
    reward = manager.compute(dt=0.02)
    torch.testing.assert_close(reward, torch.tensor([[0.016, -0.016]]))
