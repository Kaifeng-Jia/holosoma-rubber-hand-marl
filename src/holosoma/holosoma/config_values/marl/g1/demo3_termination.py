"""Minimal physical termination contract for Demo 3 tug-of-war."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg


g1_29dof_demo3_tug_termination = TerminationManagerCfg(
    terms={
        "reference_horizon": TerminationTermCfg(
            func=(
                "holosoma.managers.termination.terms.demo3_tug:"
                "reference_horizon_reached"
            ),
            is_timeout=True,
        ),
        "clear_robot_fall": TerminationTermCfg(
            func=(
                "holosoma.managers.termination.terms.demo3_tug:"
                "any_robot_clearly_fallen"
            ),
            params={
                "minimum_ref_body_height": 0.25,
                "maximum_gravity_z": -0.2,
            },
        ),
    }
)

__all__ = ["g1_29dof_demo3_tug_termination"]
