from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# The simulation test environment does not install the optional SMPL-X body
# model package.  These preprocessing helpers do not use it, so provide only
# the import placeholder needed to load the legacy utility module.
if importlib.util.find_spec("smplx") is None:
    sys.modules["smplx"] = ModuleType("smplx")

from holosoma_retargeting.src.utils import (
    ground_and_scale_human_joints,
    preprocess_motion_data,
    scale_object_pose_trajectory,
)


DEMO_JOINTS = ["Pelvis", "L_Foot", "R_Foot", "Head"]
FOOT_NAMES = ["L_Foot", "R_Foot"]


def _human_joints() -> np.ndarray:
    return np.asarray(
        [
            [[1.0, 2.0, 0.50], [0.0, 0.1, 0.20], [0.0, -0.1, 0.25], [1.0, 2.0, 1.80]],
            [[1.2, 2.2, 0.55], [0.1, 0.1, 0.30], [0.1, -0.1, 0.35], [1.2, 2.2, 1.85]],
        ],
        dtype=np.float64,
    )


def _object_poses() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 2.0, -4.0, 0.50],
            [0.0, 1.0, 0.0, 0.0, 3.0, -2.0, 0.70],
            [0.0, 0.0, 1.0, 0.0, 5.0, 1.0, 0.40],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize(("lowest_foot", "mat_height"), [(0.20, 0.10), (0.03, 0.10)])
def test_ground_and_scale_human_joints_matches_legacy_formula_without_mutation(
    lowest_foot: float,
    mat_height: float,
) -> None:
    human_joints = _human_joints()
    human_joints[0, 1, 2] = lowest_foot
    original = human_joints.copy()
    scale = 0.75

    expected = original.copy()
    z_min = expected[:, [1, 2], 2].min()
    if z_min >= mat_height:
        z_min -= mat_height
    expected[:, :, 2] -= z_min
    expected = expected * scale

    actual = ground_and_scale_human_joints(
        human_joints,
        DEMO_JOINTS,
        FOOT_NAMES,
        scale=scale,
        mat_height=mat_height,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(human_joints, original)


def test_scale_object_pose_trajectory_matches_legacy_formula_without_mutation() -> None:
    object_poses = _object_poses()
    original = object_poses.copy()
    scale = 0.6

    expected = original.copy()
    expected[:, -3:-1] = expected[:, -3:-1] * scale
    object_z0 = expected[0, -1]
    expected[:, -1] = object_z0 + (expected[:, -1] - object_z0) * scale

    actual = scale_object_pose_trajectory(object_poses, scale=scale)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, :4], original[:, :4])
    np.testing.assert_array_equal(object_poses, original)


def test_preprocess_helpers_preserve_float32_legacy_results_and_dtype() -> None:
    human_joints = _human_joints().astype(np.float32)
    object_poses = _object_poses().astype(np.float32)
    scale = np.float32(0.74906534)

    expected_human = human_joints.copy()
    z_min = expected_human[:, [1, 2], 2].min()
    if z_min >= 0.1:
        z_min -= 0.1
    expected_human[:, :, 2] -= z_min
    expected_human = expected_human * scale

    expected_object = object_poses.copy()
    expected_object[:, -3:-1] = expected_object[:, -3:-1] * scale
    object_z0 = expected_object[0, -1]
    expected_object[:, -1] = object_z0 + (expected_object[:, -1] - object_z0) * scale

    actual_human = ground_and_scale_human_joints(
        human_joints,
        DEMO_JOINTS,
        FOOT_NAMES,
        scale=scale,
    )
    actual_object = scale_object_pose_trajectory(object_poses, scale=scale)

    assert actual_human.dtype == np.float32
    assert actual_object.dtype == np.float32
    np.testing.assert_array_equal(actual_human, expected_human)
    np.testing.assert_array_equal(actual_object, expected_object)


def test_preprocess_motion_data_delegates_without_mutating_either_input() -> None:
    human_joints = _human_joints()
    object_poses = _object_poses()
    original_human = human_joints.copy()
    original_object = object_poses.copy()
    retargeter = SimpleNamespace(demo_joints=DEMO_JOINTS)
    scale = 0.8

    actual_human, actual_object, moving_frame = preprocess_motion_data(
        human_joints,
        retargeter,
        FOOT_NAMES,
        scale=scale,
        object_poses=object_poses,
    )

    expected_human = ground_and_scale_human_joints(original_human, DEMO_JOINTS, FOOT_NAMES, scale=scale)
    expected_object = scale_object_pose_trajectory(original_object, scale=scale)
    np.testing.assert_array_equal(actual_human, expected_human)
    np.testing.assert_array_equal(actual_object, expected_object)
    assert moving_frame == 0
    np.testing.assert_array_equal(human_joints, original_human)
    np.testing.assert_array_equal(object_poses, original_object)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_preprocess_helpers_reject_non_positive_or_non_finite_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        ground_and_scale_human_joints(_human_joints(), DEMO_JOINTS, FOOT_NAMES, scale=scale)
    with pytest.raises(ValueError, match="scale"):
        scale_object_pose_trajectory(_object_poses(), scale=scale)


def test_preprocess_helpers_validate_shapes_and_finiteness() -> None:
    with pytest.raises(ValueError, match="human_joints"):
        ground_and_scale_human_joints(np.zeros((2, 3)), DEMO_JOINTS, FOOT_NAMES)
    with pytest.raises(ValueError, match="object_poses"):
        scale_object_pose_trajectory(np.zeros((2, 6)))

    non_finite_human = _human_joints()
    non_finite_human[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ground_and_scale_human_joints(non_finite_human, DEMO_JOINTS, FOOT_NAMES)

    non_finite_object = _object_poses()
    non_finite_object[0, -1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        scale_object_pose_trajectory(non_finite_object)
