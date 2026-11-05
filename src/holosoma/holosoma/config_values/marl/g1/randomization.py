"""Fixed startup physics configuration for the Plan 5 baseline."""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg


g1_29dof_plan5_fixed_object_material = RandomizationManagerCfg(
    setup_terms={
        "set_object_rigid_body_material_startup": RandomizationTermCfg(
            func=(
                "holosoma.managers.randomization.terms.marl:"
                "set_object_rigid_body_material_startup"
            ),
            params={
                "static_friction": 0.5,
                "dynamic_friction": 0.5,
                "restitution": 0.0,
            },
        )
    }
)

__all__ = ["g1_29dof_plan5_fixed_object_material"]
