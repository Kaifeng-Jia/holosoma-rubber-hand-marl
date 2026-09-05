"""CPU checks for fresh CORE4D small-table shared-policy MAPPO."""

import copy

import pytest
import torch

from holosoma.agents.mappo.core4d_smalltable_evaluation import (
    deterministic_core4d_smalltable_actions,
)
from holosoma.agents.mappo.core4d_smalltable_initialization import (
    CORE4D_SMALLTABLE_INITIALIZATION,
    initialize_core4d_smalltable_model_bundle,
)
from holosoma.agents.mappo.core4d_smalltable_ppo import (
    CORE4D_SMALLTABLE_OBJECT_URDF_SHA256,
    CORE4D_SMALLTABLE_PHYSICS_CONTRACT,
    CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256,
    CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256,
    Core4DSmallTablePPO,
    validate_core4d_smalltable_checkpoint,
)
from holosoma.agents.mappo.core4d_smalltable_runner import (
    Core4DSmallTablePolicyRunner,
)
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import (
    g1_29dof_core4d_smalltable_baseline,
)


def _observations(num_envs: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(721)
    return {
        "actor_obs": torch.randn(num_envs, 2, 154, generator=generator),
        "teammate_obs": torch.randn(num_envs, 2, 4, generator=generator),
        "table_obs": torch.randn(num_envs, 2, 6, generator=generator),
        "critic_obs": torch.randn(num_envs, 527, generator=generator),
    }


def _first_linear_width(module: torch.nn.Module) -> int:
    return next(layer.in_features for layer in module.modules() if isinstance(layer, torch.nn.Linear))


def test_core4d_initialization_is_fully_fresh_and_has_reviewed_dimensions() -> None:
    torch.manual_seed(721)
    bundle = initialize_core4d_smalltable_model_bundle(
        g1_29dof_core4d_smalltable_baseline.algo.config,
        device="cpu",
    )

    assert _first_linear_width(bundle.actor) == 164
    assert _first_linear_width(bundle.critic) == 527
    assert bundle.source_iteration == 0
    assert bundle.source_sha256 == CORE4D_SMALLTABLE_INITIALIZATION
    assert bundle.actor_optimizer.state == {}
    assert bundle.critic_optimizer.state == {}
    assert bundle.actor_obs_normalizer.count.item() == 0
    assert bundle.critic_obs_normalizer.count.item() == 0


def test_core4d_runner_routes_two_actor_rows_and_one_team_critic_row() -> None:
    bundle = initialize_core4d_smalltable_model_bundle(
        g1_29dof_core4d_smalltable_baseline.algo.config,
        device="cpu",
    )
    runner = Core4DSmallTablePolicyRunner(bundle)
    decision = runner.sample(_observations())

    assert decision.actor_observations.shape == (3, 2, 164)
    assert decision.actions.shape == (3, 2, 29)
    assert decision.values.shape == (3, 1)
    assert decision.action_log_probs.shape == (3, 2, 1)
    assert bundle.actor_obs_normalizer.count.item() == 6
    assert bundle.critic_obs_normalizer.count.item() == 3


def test_core4d_evaluation_calls_only_actor() -> None:
    bundle = initialize_core4d_smalltable_model_bundle(
        g1_29dof_core4d_smalltable_baseline.algo.config,
        device="cpu",
    )

    class _ForbiddenCritic:
        def evaluate(self, _state):
            raise AssertionError("Actor-only evaluation must not call the Critic")

    bundle.critic = _ForbiddenCritic()
    actions = deterministic_core4d_smalltable_actions(bundle, _observations())
    assert actions.shape == (3, 2, 29)
    assert torch.isfinite(actions).all()


def test_core4d_checkpoint_is_isolated_and_strictly_resumable() -> None:
    config = g1_29dof_core4d_smalltable_baseline.algo.config
    first_bundle = initialize_core4d_smalltable_model_bundle(config, device="cpu")
    first = Core4DSmallTablePPO(
        first_bundle,
        config,
        num_envs=2,
        num_steps_per_env=2,
        device="cpu",
    )
    state = first.training_state_dict(iteration=1000)

    assert validate_core4d_smalltable_checkpoint(state) == 1000
    assert set(state).isdisjoint({"plan5_mappo", "demo3_mappo", "demo4_mappo"})
    assert state["core4d_smalltable_mappo"]["actor_obs_dim"] == 164
    assert state["core4d_smalltable_mappo"]["critic_obs_dim"] == 527
    assert (
        state["core4d_smalltable_mappo"]["runtime_reference_sha256"]
        == CORE4D_SMALLTABLE_RUNTIME_REFERENCE_SHA256
    )
    assert (
        state["core4d_smalltable_mappo"]["object_urdf_sha256"]
        == CORE4D_SMALLTABLE_OBJECT_URDF_SHA256
    )
    assert (
        state["core4d_smalltable_mappo"]["physics_contract"]
        == CORE4D_SMALLTABLE_PHYSICS_CONTRACT
    )
    assert (
        state["core4d_smalltable_mappo"]["training_promotion_sha256"]
        == CORE4D_SMALLTABLE_TRAINING_PROMOTION_SHA256
    )

    second_bundle = initialize_core4d_smalltable_model_bundle(config, device="cpu")
    second = Core4DSmallTablePPO(
        second_bundle,
        config,
        num_envs=2,
        num_steps_per_env=2,
        device="cpu",
    )
    assert second.load_training_state_dict(state) == 1000
    for key, value in first.models.actor.state_dict().items():
        torch.testing.assert_close(value, second.models.actor.state_dict()[key])

    contaminated = copy.deepcopy(state)
    contaminated["demo4_mappo"] = {}
    with pytest.raises(ValueError, match="cross-demo metadata"):
        second.load_training_state_dict(contaminated)

    incomplete = copy.deepcopy(state)
    incomplete.pop("actor_optimizer_state_dict")
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        second.load_training_state_dict(incomplete)

    wrong_reference = copy.deepcopy(state)
    wrong_reference["core4d_smalltable_mappo"]["runtime_reference_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata mismatch"):
        second.load_training_state_dict(wrong_reference)
