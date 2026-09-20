"""CPU contracts for the opt-in shared-object world-height penalty."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.marl.g1.core4d_smalltable_reward import (
    g1_29dof_core4d_smalltable_reward,
    validate_object_height_penalty,
    with_interaction_reward_term,
    with_object_height_penalty,
    with_object_z_error_weight,
)
from holosoma.managers.reward.manager import RewardManager
from holosoma.managers.reward.terms import marl
from tests.managers.reward.test_plan5_reward import make_reward_env


_NAME = "object_height_error_penalty"


def _config(weight: float = 1.0, scale_m: float = 0.05) -> RewardTermCfg:
    return with_object_height_penalty(RewardManagerCfg(), weight, scale_m).terms[_NAME]


def _set_error(command, env, error_z) -> None:
    actual = command.object_pos_w.clone()
    actual[:, 2] += torch.as_tensor(error_z)
    env.simulator.all_root_states[command.object_indices_in_simulator, :3] = actual


def test_height_penalty_is_one_raw_positive_cost_per_shared_environment(tmp_path):
    command, env = make_reward_env(tmp_path)
    term = marl.ObjectHeightErrorPenalty(_config(weight=3.0), env)
    torch.testing.assert_close(term(env), torch.zeros(env.num_envs))

    _set_error(command, env, [0.05, -0.05])
    raw = term(env)
    assert raw.shape == (env.num_envs,)
    torch.testing.assert_close(raw, torch.ones(env.num_envs))
    torch.testing.assert_close(term.last_error_z_m, torch.tensor([0.05, -0.05]))

    # The term has no contact gate, clipping, weight, dt, or agent multiplier.
    env.simulator.agent_contact_forces_history.fill_(1.0e6)
    env.dt = 0.5
    _set_error(command, env, [0.10, -0.50])
    torch.testing.assert_close(term(env), torch.tensor([4.0, 100.0]))
    torch.testing.assert_close(term.last_raw_penalty, torch.tensor([4.0, 100.0]))


def test_height_uses_the_same_world_frame_and_actor_origin_as_position_reward(tmp_path):
    command, env = make_reward_env(tmp_path)
    # Nonzero environment heights must be included in reference and actual.
    env.simulator.scene.env_origins[:, 2] = torch.tensor([2.0, -3.0])
    _set_error(command, env, [0.05, -0.10])
    before = env.simulator.all_root_states.clone()
    term = marl.ObjectHeightErrorPenalty(_config(), env)
    torch.testing.assert_close(term(env), torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(env.simulator.all_root_states, before)
    expected = torch.exp(-torch.tensor([0.05**2, 0.10**2]) / 0.3**2)
    torch.testing.assert_close(marl.object_global_ref_position_error_exp(env, sigma=0.3), expected)


def test_height_constructor_does_not_access_command_before_it_exists():
    env = SimpleNamespace(num_envs=2, device="cpu")
    term = marl.ObjectHeightErrorPenalty(_config(), env)
    assert term.get_iteration_diagnostics() == {}
    assert term.scale_m == 0.05


def test_height_iteration_diagnostics_cover_every_call_and_survive_episode_resets(tmp_path):
    command, env = make_reward_env(tmp_path)
    term = marl.ObjectHeightErrorPenalty(_config(), env)
    _set_error(command, env, [0.05, -0.10])
    term(env)
    term.reset(torch.tensor([0]))
    assert term.last_error_z_m[0] == 0
    assert term.last_raw_penalty[0] == 0
    _set_error(command, env, [0.0, 0.20])
    term(env)
    term.reset()
    assert not term.last_error_z_m.any()
    assert not term.last_raw_penalty.any()

    errors = torch.tensor([0.05, -0.10, 0.0, 0.20])
    expected = {
        "Height/error_z_signed_mean_m": errors.mean(),
        "Height/error_z_abs_mean_m": errors.abs().mean(),
        "Height/error_z_rmse_m": errors.square().mean().sqrt(),
        "Height/raw_penalty_mean": (errors / 0.05).square().mean(),
        "Height/raw_penalty_max": (errors / 0.05).square().max(),
        "Height/sample_count": torch.tensor(4.0),
    }
    snapshot = term.get_iteration_diagnostics(reset=False)
    drained = term.get_iteration_diagnostics(reset=True)
    assert snapshot.keys() == expected.keys()
    for name, value in expected.items():
        assert snapshot[name].shape == ()
        torch.testing.assert_close(snapshot[name], value)
        torch.testing.assert_close(drained[name], value)
    assert term.get_iteration_diagnostics() == {}

    # A new iteration starts fresh and does not alter previously read values.
    _set_error(command, env, [0.0, 0.0])
    term(env)
    next_iteration = term.get_iteration_diagnostics()
    assert next_iteration["Height/sample_count"] == 2
    assert next_iteration["Height/raw_penalty_max"] == 0
    for name, value in expected.items():
        torch.testing.assert_close(drained[name], value)


def test_reward_manager_applies_negative_lambda_and_dt_once(tmp_path):
    command, env = make_reward_env(tmp_path)
    _set_error(command, env, [0.05, -0.10])
    manager = RewardManager(RewardManagerCfg(terms={_NAME: _config(weight=2.0)}), env, "cpu")
    torch.testing.assert_close(manager.compute(dt=0.02), torch.tensor([-0.04, -0.16]))
    torch.testing.assert_close(manager.episode_sums_raw[_NAME], torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(manager.compute(dt=0.10), torch.tensor([-0.20, -0.80]))
    diagnostics = manager.get_term(_NAME).get_iteration_diagnostics()
    assert diagnostics["Height/sample_count"] == 2 * env.num_envs
    assert diagnostics["Height/raw_penalty_mean"] == 2.5


def test_default_zero_lambda_preserves_baseline_and_deep_copies_terms():
    base = g1_29dof_core4d_smalltable_reward
    before = copy.deepcopy(base)
    disabled = with_object_height_penalty(base)
    assert disabled == base == before
    assert _NAME not in disabled.terms
    assert disabled is not base
    for name in base.terms:
        assert disabled.terms[name] is not base.terms[name]
        assert disabled.terms[name].params is not base.terms[name].params
    disabled.terms["object_global_ref_position_error_exp"].params["sigma"] = 0.7
    assert base == before


def test_height_variant_preserves_interaction_and_wz2_and_only_adds_one_term():
    base = with_object_z_error_weight(
        with_interaction_reward_term(g1_29dof_core4d_smalltable_reward, "/tmp/interaction.npz"),
        2.0,
    )
    before = copy.deepcopy(base)
    variant = with_object_height_penalty(base, weight=1.0, scale_m=0.05)
    assert base == before
    assert list(variant.terms) == [*base.terms, _NAME]
    for name in base.terms:
        assert variant.terms[name] == base.terms[name]
        assert variant.terms[name] is not base.terms[name]
    added = variant.terms[_NAME]
    assert added.func == "holosoma.managers.reward.terms.marl:ObjectHeightErrorPenalty"
    assert added.weight == -1.0
    assert added.params == {"scale_m": 0.05}
    with pytest.raises(ValueError, match="already enabled"):
        with_object_height_penalty(variant, weight=1.0)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_invalid_height_lambda_is_rejected(weight):
    with pytest.raises(ValueError, match="weight.*nonnegative and finite"):
        validate_object_height_penalty(weight, 0.05)
    with pytest.raises(ValueError, match="weight.*nonnegative and finite"):
        with_object_height_penalty(g1_29dof_core4d_smalltable_reward, weight=weight)


@pytest.mark.parametrize("scale", [0.0, -0.05, float("nan"), float("inf"), True, "0.05", None])
def test_invalid_height_scale_is_rejected_even_when_disabled(scale):
    with pytest.raises(ValueError, match="scale.*positive and finite"):
        validate_object_height_penalty(0.0, scale)
    with pytest.raises(ValueError, match="scale.*positive and finite"):
        with_object_height_penalty(g1_29dof_core4d_smalltable_reward, weight=0.0, scale_m=scale)
    with pytest.raises(ValueError, match="scale.*positive and finite"):
        marl.ObjectHeightErrorPenalty(
            RewardTermCfg(
                func="holosoma.managers.reward.terms.marl:ObjectHeightErrorPenalty",
                params={"scale_m": scale}, weight=-1.0,
            ),
            SimpleNamespace(num_envs=2, device="cpu"),
        )


def test_validation_normalizes_valid_values_to_floats():
    result = validate_object_height_penalty(0, 1)
    assert result == (0.0, 1.0)
    assert all(type(value) is float for value in result)
