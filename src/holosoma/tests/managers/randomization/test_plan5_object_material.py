from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError
from holosoma.managers.randomization.terms.marl import (
    set_object_rigid_body_material_startup,
)


class FakePhysxView:
    def __init__(self, materials: torch.Tensor):
        self.materials = materials
        self.indices = None

    def get_material_properties(self) -> torch.Tensor:
        return self.materials.clone()

    def set_material_properties(self, materials: torch.Tensor, indices: torch.Tensor) -> None:
        self.materials[indices] = materials[indices]
        self.indices = indices.clone()


def make_env(num_envs: int = 2, num_shapes: int = 5):
    view = FakePhysxView(torch.ones(num_envs, num_shapes, 3))
    env = SimpleNamespace(
        num_envs=num_envs,
        simulator=SimpleNamespace(_object=SimpleNamespace(root_physx_view=view)),
    )
    return env, view


def test_fixed_material_sets_every_environment_and_shape():
    env, view = make_env()

    set_object_rigid_body_material_startup(
        env,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
    )

    torch.testing.assert_close(view.materials[..., 0], torch.full((2, 5), 0.5))
    torch.testing.assert_close(view.materials[..., 1], torch.full((2, 5), 0.5))
    torch.testing.assert_close(view.materials[..., 2], torch.zeros(2, 5))
    torch.testing.assert_close(view.indices, torch.tensor([0, 1]))


@pytest.mark.parametrize(
    ("static_friction", "dynamic_friction", "restitution"),
    [
        (-0.1, 0.0, 0.0),
        (0.4, 0.5, 0.0),
        (0.5, 0.5, 1.1),
        (float("nan"), 0.5, 0.0),
    ],
)
def test_invalid_material_values_fail_closed(
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
):
    env, _ = make_env()

    with pytest.raises(ValueError):
        set_object_rigid_body_material_startup(
            env,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )


def test_missing_physx_object_fails_closed():
    env = SimpleNamespace(num_envs=1, simulator=SimpleNamespace())

    with pytest.raises(RandomizerNotSupportedError):
        set_object_rigid_body_material_startup(
            env,
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        )
