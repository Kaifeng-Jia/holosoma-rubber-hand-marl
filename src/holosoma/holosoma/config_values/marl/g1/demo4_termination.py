"""Termination contract for cooperative Demo 4 rotation."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg


_TERMS = "holosoma.managers.termination.terms.demo4_rotate"

g1_29dof_demo4_rotate_termination = TerminationManagerCfg(
    terms={
        "yaw_goal_success": TerminationTermCfg(
            func=f"{_TERMS}:yaw_goal_reached",
        ),
        "reference_horizon": TerminationTermCfg(
            func=f"{_TERMS}:reference_horizon_reached",
            is_timeout=True,
        ),
        "clear_robot_fall": TerminationTermCfg(
            func=f"{_TERMS}:any_robot_clearly_fallen",
            params={
                "minimum_ref_body_height": 0.25,
                "maximum_gravity_z": -0.2,
            },
        ),
        "table_physical_safety": TerminationTermCfg(
            func=f"{_TERMS}:table_physical_safety_exceeded",
            params={
                "maximum_tilt_degrees": 60.0,
                "maximum_xy_drift_m": 3.0,
                "minimum_height_m": 0.05,
                "maximum_height_m": 1.5,
            },
        ),
    }
)

__all__ = ["g1_29dof_demo4_rotate_termination"]
