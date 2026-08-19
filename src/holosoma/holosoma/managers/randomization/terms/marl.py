"""Fixed startup physics terms for Plan 5 multi-agent environments."""

from __future__ import annotations

import math
from typing import Any

import torch

from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError


def set_object_rigid_body_material_startup(
    env: Any,
    *,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
    **_,
) -> None:
    """Set one fixed PhysX material on every collision shape of the shared object."""
    values = {
        "static_friction": float(static_friction),
        "dynamic_friction": float(dynamic_friction),
        "restitution": float(restitution),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Object material values must be finite: {values}")
    if values["static_friction"] < 0.0 or values["dynamic_friction"] < 0.0:
        raise ValueError(f"Object friction values must be non-negative: {values}")
    if values["dynamic_friction"] > values["static_friction"]:
        raise ValueError(f"Dynamic friction cannot exceed static friction: {values}")
    if not 0.0 <= values["restitution"] <= 1.0:
        raise ValueError(f"Object restitution must be in [0, 1]: {values}")

    simulator = getattr(env, "simulator", None)
    rigid_object = getattr(simulator, "_object", None)
    physx_view = getattr(rigid_object, "root_physx_view", None)
    if physx_view is None:
        raise RandomizerNotSupportedError(
            "Fixed Plan 5 object material requires an initialized PhysX rigid object view"
        )

    materials = physx_view.get_material_properties().clone()
    if materials.ndim != 3 or materials.shape[0] != env.num_envs or materials.shape[-1] != 3:
        raise RuntimeError(
            "Unexpected object material tensor shape: "
            f"{tuple(materials.shape)}, expected [num_envs, shapes, 3]"
        )
    materials[..., 0] = values["static_friction"]
    materials[..., 1] = values["dynamic_friction"]
    materials[..., 2] = values["restitution"]
    env_ids_cpu = torch.arange(env.num_envs, device="cpu", dtype=torch.long)
    physx_view.set_material_properties(materials, env_ids_cpu)

    actual = physx_view.get_material_properties()
    if not torch.equal(actual, materials):
        raise RuntimeError("PhysX did not retain the requested fixed object material")


__all__ = ["set_object_rigid_body_material_startup"]
