from types import SimpleNamespace

import torch

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.termination.manager import TerminationManager


def failure_term(_env):
    return torch.tensor([True, False])


def timeout_term(_env):
    return torch.tensor([False, True])


def test_base_task_copies_per_term_results_before_reset():
    env = SimpleNamespace(num_envs=2)
    cfg = TerminationManagerCfg(
        terms={
            "failure": TerminationTermCfg(
                func="tests.managers.termination.test_termination_snapshots:failure_term",
            ),
            "timeout": TerminationTermCfg(
                func="tests.managers.termination.test_termination_snapshots:timeout_term",
                is_timeout=True,
            ),
        }
    )
    manager = TerminationManager(cfg, env, "cpu")
    task = object.__new__(BaseTask)
    task.num_envs = 2
    task.termination_manager = manager
    task.reset_buf = torch.zeros(2, dtype=torch.long)
    task.time_out_buf = torch.zeros(2, dtype=torch.bool)
    task.extras = {}

    BaseTask._check_termination(task)
    snapshot = {name: value.clone() for name, value in task.extras["termination_terms"].items()}
    manager.reset(torch.tensor([0, 1]))

    torch.testing.assert_close(snapshot["failure"], torch.tensor([True, False]))
    torch.testing.assert_close(snapshot["timeout"], torch.tensor([False, True]))
    torch.testing.assert_close(task.reset_buf, torch.tensor([1, 1]))
    torch.testing.assert_close(task.time_out_buf, torch.tensor([False, True]))
    assert not manager.term_results["failure"].any()
    assert not manager.term_results["timeout"].any()
