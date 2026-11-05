"""Action presets for the homogeneous two-G1 environment."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

g1_29dof_dual_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.marl:DualJointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["g1_29dof_dual_joint_pos"]
