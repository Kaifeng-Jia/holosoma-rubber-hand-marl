from __future__ import annotations

import pytest
import torch

from holosoma.agents.mappo import HomogeneousAgentBatchLayout


def test_actor_observation_flattening_preserves_environment_agent_order() -> None:
    layout = HomogeneousAgentBatchLayout()
    observations = torch.arange(3 * 2 * 158, dtype=torch.float32).reshape(3, 2, 158)

    flattened = layout.flatten_actor_observations(observations)

    assert flattened.shape == (6, 158)
    torch.testing.assert_close(flattened[0], observations[0, 0])
    torch.testing.assert_close(flattened[1], observations[0, 1])
    torch.testing.assert_close(flattened[2], observations[1, 0])


def test_action_round_trip_does_not_mix_agents() -> None:
    layout = HomogeneousAgentBatchLayout()
    shared_actor_actions = torch.stack(
        [torch.full((29,), float(index)) for index in range(8)],
        dim=0,
    )

    routed = layout.unflatten_actions(shared_actor_actions, num_envs=4)

    assert routed.shape == (4, 2, 29)
    torch.testing.assert_close(routed[2, 0], torch.full((29,), 4.0))
    torch.testing.assert_close(routed[2, 1], torch.full((29,), 5.0))
    torch.testing.assert_close(layout.flatten_actions(routed), shared_actor_actions)


def test_team_tensor_expansion_shares_values_without_copying_agent_semantics() -> None:
    layout = HomogeneousAgentBatchLayout()
    team_returns = torch.tensor([[1.0], [2.0], [3.0]])

    expanded = layout.expand_team_tensor(team_returns)

    assert expanded.shape == (3, 2, 1)
    torch.testing.assert_close(expanded[:, 0], team_returns)
    torch.testing.assert_close(expanded[:, 1], team_returns)


@pytest.mark.parametrize(
    ("method_name", "bad_shape"),
    [
        ("flatten_actor_observations", (4, 158)),
        ("flatten_actor_observations", (4, 3, 158)),
        ("flatten_actions", (4, 2, 58)),
    ],
)
def test_invalid_agent_shapes_are_rejected(method_name: str, bad_shape: tuple[int, ...]) -> None:
    layout = HomogeneousAgentBatchLayout()
    values = torch.zeros(bad_shape)

    with pytest.raises(ValueError, match="must have shape"):
        getattr(layout, method_name)(values)


def test_non_positive_layout_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="num_agents must be positive"):
        HomogeneousAgentBatchLayout(num_agents=0)
