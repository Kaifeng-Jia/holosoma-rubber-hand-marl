from __future__ import annotations

from dataclasses import replace
from holosoma_retargeting.config_types.retargeter import RetargeterConfig


def test_manual_joint_limit_override_defaults_to_enabled() -> None:
    config = RetargeterConfig()

    assert config.activate_joint_limits is True
    assert config.apply_manual_joint_limit_overrides is True
    assert config.elastic_constraints.enable is False
    assert config.pt_full_arm_orientation.enable is False
    assert config.pt_wrist_dominant_surface.enable is False


def test_manual_joint_limit_override_is_independent_of_physical_limits() -> None:
    default = RetargeterConfig()
    relaxed = replace(default, apply_manual_joint_limit_overrides=False)

    assert default.apply_manual_joint_limit_overrides is True
    assert relaxed.apply_manual_joint_limit_overrides is False
    assert relaxed.activate_joint_limits is True


def test_elastic_constraints_are_opt_in_and_replaceable() -> None:
    default = RetargeterConfig()
    elastic = replace(
        default,
        elastic_constraints=replace(
            default.elastic_constraints,
            enable=True,
            object_collision_weight=2.0e5,
            foot_kinematics_weight=2.0e4,
        ),
    )

    assert default.elastic_constraints.enable is False
    assert elastic.elastic_constraints.enable is True
    assert elastic.elastic_constraints.object_collision_weight == 2.0e5
    assert elastic.elastic_constraints.foot_kinematics_weight == 2.0e4


def test_full_arm_palm_refinement_is_independently_replaceable() -> None:
    default = RetargeterConfig()
    enabled = replace(
        default,
        pt_full_arm_orientation=replace(
            default.pt_full_arm_orientation,
            enable=True,
            hand_position_weight=2.0e4,
        ),
    )

    assert default.pt_full_arm_orientation.enable is False
    assert enabled.pt_wrist_orientation.enable is False
    assert enabled.pt_full_arm_orientation.enable is True
    assert enabled.pt_full_arm_orientation.hand_position_weight == 2.0e4


def test_wrist_dominant_surface_refinement_is_opt_in() -> None:
    default = RetargeterConfig()
    enabled = replace(
        default,
        pt_wrist_dominant_surface=replace(
            default.pt_wrist_dominant_surface,
            enable=True,
            surface_position_weight=2.0e4,
        ),
    )

    assert default.pt_wrist_dominant_surface.enable is False
    assert enabled.pt_wrist_orientation.enable is False
    assert enabled.pt_full_arm_orientation.enable is False
    assert enabled.pt_wrist_dominant_surface.enable is True
    assert enabled.pt_wrist_dominant_surface.surface_position_weight == 2.0e4
