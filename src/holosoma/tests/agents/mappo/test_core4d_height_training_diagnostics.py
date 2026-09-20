"""CPU-only checks for the opt-in rollout observer, without importing Isaac."""

import ast
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(scope="module")
def collect_with_diagnostics():
    """Compile only the pure helper, not the simulator-launching train module."""
    path = REPO_ROOT / "scripts/train_core4d_smalltable.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_collect_with_raw_reward_diagnostics"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[helper.name]


class FakeEnv:
    """Match BaseTask's increment-before-reset and reused reward-buffer behavior."""

    def __init__(self, transitions, initial_lengths, *, columns=False, fail_at=None):
        self.device = "cpu"
        self.transitions = transitions
        self.episode_length_buf = torch.tensor(initial_lengths, dtype=torch.long)
        self.reward_buffer = torch.zeros(len(initial_lengths))
        self.columns = columns
        self.fail_at = fail_at
        self.calls = 0
        self.actions = []
        self.results = []
        # Pin the bound method so identity can prove exact restoration, too.
        self.step = self.step

    def step(self, actions):
        if self.calls == self.fail_at:
            raise RuntimeError("environment step failed")
        transition = self.transitions[self.calls]
        self.calls += 1
        self.actions.append(actions)
        self.episode_length_buf += 1
        dones = torch.tensor(transition["dones"], dtype=torch.bool)
        self.episode_length_buf[dones] = 0
        self.reward_buffer.copy_(torch.tensor(transition["rewards"]))
        terminal_values = torch.tensor(transition.get("terminal_values", [7.0] * len(dones)))
        post_reset_values = terminal_values.clone()
        post_reset_values[dones] = -1000.0
        observations = {"critic_obs": post_reset_values[:, None]}
        extras = {"final_observations": {"critic_obs": terminal_values[:, None]}}
        if "timeouts" in transition:
            timeouts = torch.tensor(transition["timeouts"], dtype=torch.bool)
            extras["time_outs"] = timeouts[:, None] if self.columns else timeouts
        rewards = self.reward_buffer[:, None] if self.columns else self.reward_buffer
        result = (observations, rewards, dones[:, None] if self.columns else dones, extras)
        self.results.append(result)
        return result


class FakeStorage:
    def __init__(self):
        self.rewards = torch.empty((0, 0, 1))

    def team(self, name):
        assert name == "rewards"
        return self.rewards


class FakeLearner:
    """Use the real collector's out-of-place timeout bootstrap convention."""

    def __init__(self, steps, *, fail=None, wrong_count=False):
        self.steps = steps
        self.fail = fail
        self.wrong_count = wrong_count
        self.storage = FakeStorage()
        self.results = []
        self.raw_rewards = []
        self.initial_observations = None

    def collect_rollout(self, env, observations):
        self.initial_observations = observations
        if self.fail == "before":
            raise RuntimeError("learner failed before step")
        stored = []
        for _ in range(self.steps):
            result = env.step({"actions": torch.randn(len(env.episode_length_buf), 2)})
            self.results.append(result)
            observations, rewards, dones, extras = result
            self.raw_rewards.append(rewards.clone())
            if self.fail == "after":
                raise RuntimeError("learner failed after step")
            timeouts = extras.get("time_outs", torch.zeros_like(dones)).reshape(-1, 1)
            final_values = extras["final_observations"]["critic_obs"]
            stored.append(rewards.reshape(-1, 1) + 0.9 * final_values * timeouts)
        if stored:
            self.storage.rewards = torch.stack(stored)
        if self.wrong_count:
            self.storage.rewards = self.storage.rewards[:-1]
        return observations


def mixed_transitions():
    return [
        {
            "rewards": [-3.0, 2.0, -1.0],
            "dones": [True, True, False],
            "timeouts": [False, True, False],
            "terminal_values": [2.0, 10.0, 8.0],
        },
        {
            "rewards": [-6.0, -4.0, 1.0],
            "dones": [False, False, True],
            "timeouts": [False, False, False],
        },
    ]


@pytest.mark.parametrize("columns", [False, True])
def test_observer_preserves_rollout_rng_rewards_and_terminal_bootstrap(collect_with_diagnostics, columns):
    plain_env = FakeEnv(mixed_transitions(), [3, 9, 0], columns=columns)
    observed_env = FakeEnv(mixed_transitions(), [3, 9, 0], columns=columns)
    plain_learner, observed_learner = FakeLearner(2), FakeLearner(2)
    initial = {"critic_obs": torch.zeros(3, 1)}
    original_step = observed_env.step
    initial_rng = torch.random.get_rng_state()
    try:
        plain_output = plain_learner.collect_rollout(plain_env, initial)
        plain_rng_after = torch.random.get_rng_state()
        torch.random.set_rng_state(initial_rng)
        output, diagnostics = collect_with_diagnostics(observed_learner, observed_env, initial)
        assert torch.equal(torch.random.get_rng_state(), plain_rng_after)
    finally:
        torch.random.set_rng_state(initial_rng)

    assert observed_env.step is original_step
    assert observed_learner.initial_observations is initial
    assert output is observed_env.results[-1][0]
    assert torch.equal(output["critic_obs"], plain_output["critic_obs"])
    assert torch.equal(observed_env.episode_length_buf, plain_env.episode_length_buf)
    assert torch.equal(observed_learner.storage.rewards, plain_learner.storage.rewards)
    assert observed_learner.storage.rewards[0, 1, 0] == 11.0
    # A reset observation of -1000 must never replace the terminal value of 10.
    assert observed_env.results[0][0]["critic_obs"][1, 0] == -1000.0
    for index, result in enumerate(observed_learner.results):
        assert result is observed_env.results[index]
        assert torch.equal(observed_env.actions[index]["actions"], plain_env.actions[index]["actions"])
        assert torch.equal(observed_learner.raw_rewards[index], plain_learner.raw_rewards[index])
        assert torch.equal(
            observed_learner.raw_rewards[index].reshape(-1),
            torch.tensor(mixed_transitions()[index]["rewards"]),
        )

    assert diagnostics["Reward/env_raw_mean"].item() == pytest.approx(-11.0 / 6)
    assert diagnostics["Reward/env_raw_min"].item() == -6.0
    assert diagnostics["Reward/env_raw_max"].item() == 2.0
    assert diagnostics["Reward/timeout_bootstrap_mean"].item() == pytest.approx(9.0 / 6)
    assert diagnostics["Reward/raw_sample_count"].item() == 6
    assert diagnostics["Episode/reset_count"].item() == 3
    assert diagnostics["Episode/failure_count"].item() == 2
    # Pre-step lengths [3, 9, 0] yield ended lengths 4, 10, then 2, not reset zeros.
    assert diagnostics["Episode/reset_length_mean_steps"].item() == pytest.approx(16.0 / 3)
    assert diagnostics["Episode/failure_length_mean_steps"].item() == 3.0
    assert all(value.device.type == "cpu" and not value.requires_grad for value in diagnostics.values())


def test_missing_timeouts_count_done_as_failure_and_preserve_negative_maximum(collect_with_diagnostics):
    env = FakeEnv([{"rewards": [-2.0, -4.0], "dones": [True, False]}], [12, 8])
    learner = FakeLearner(1)
    _, diagnostics = collect_with_diagnostics(learner, env, {})
    assert diagnostics["Reward/env_raw_mean"].item() == -3.0
    assert diagnostics["Reward/env_raw_max"].item() == -2.0
    assert diagnostics["Reward/timeout_bootstrap_mean"].item() == 0.0
    assert diagnostics["Episode/reset_count"].item() == 1
    assert diagnostics["Episode/failure_count"].item() == 1
    assert diagnostics["Episode/failure_length_mean_steps"].item() == 13.0


def test_no_reset_means_zero_counts_and_finite_zero_length_means(collect_with_diagnostics):
    env = FakeEnv([{"rewards": [-2.0, 1.0], "dones": [False, False]}], [12, 8])
    _, diagnostics = collect_with_diagnostics(FakeLearner(1), env, {})
    for name in (
        "Episode/reset_count", "Episode/failure_count",
        "Episode/reset_length_mean_steps", "Episode/failure_length_mean_steps",
    ):
        assert diagnostics[name].item() == 0.0
    assert all(torch.isfinite(value).item() for value in diagnostics.values())


@pytest.mark.parametrize("failure", ["environment", "before", "after"])
def test_step_restored_when_environment_or_learner_raises(collect_with_diagnostics, failure):
    env = FakeEnv(mixed_transitions(), [3, 9, 0], fail_at=0 if failure == "environment" else None)
    learner = FakeLearner(2, fail=failure)
    original_step = env.step
    with pytest.raises(RuntimeError, match="failed"):
        collect_with_diagnostics(learner, env, {})
    assert env.step is original_step


def test_empty_collection_fails_after_restoring_step(collect_with_diagnostics):
    env = FakeEnv([], [0])
    original_step = env.step
    with pytest.raises(RuntimeError, match="No raw rewards"):
        collect_with_diagnostics(FakeLearner(0), env, {})
    assert env.step is original_step


def test_storage_sample_mismatch_fails_after_restoring_step(collect_with_diagnostics):
    env = FakeEnv(mixed_transitions(), [3, 9, 0])
    original_step = env.step
    with pytest.raises(RuntimeError, match="sample counts differ"):
        collect_with_diagnostics(FakeLearner(2, wrong_count=True), env, {})
    assert env.step is original_step
