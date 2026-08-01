from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from math import isclose
from pathlib import Path

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DIR = PACKAGE_ROOT / "holosoma" / "data" / "robots" / "g1"
RUBBER_HAND_URDF = ROBOT_DIR / "main_mesh_collision_rubberhand.urdf"
ROBOT_CONFIG = PACKAGE_ROOT / "holosoma" / "config_values" / "robot.py"
TRAINING_MOTION_DIR = (
    PACKAGE_ROOT
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
)
TRAINING_TABLE_URDF = TRAINING_MOTION_DIR / "objects_largetable.urdf"
TRAINING_TABLE_MESH = TRAINING_MOTION_DIR / "largetable.obj"
RETARGETING_MODEL_DIR = (
    PACKAGE_ROOT.parent
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "models"
)
RETARGETING_ROBOT_XML = (
    RETARGETING_MODEL_DIR / "g1" / "g1_29dof_w_largetable.xml"
)
RETARGETING_TABLE_URDF = (
    RETARGETING_MODEL_DIR / "largetable" / "largetable.urdf"
)
RETARGETING_TABLE_MESH = (
    RETARGETING_MODEL_DIR / "largetable" / "largetable.obj"
)


def _required_element(root: ET.Element, xpath: str) -> ET.Element:
    element = root.find(xpath)
    assert element is not None, f"Missing URDF element: {xpath}"
    return element


def _assert_float_sequence_equal(
    actual: str,
    expected: str,
    *,
    tolerance: float = 1e-9,
) -> None:
    actual_values = tuple(float(value) for value in actual.split())
    expected_values = tuple(float(value) for value in expected.split())
    assert len(actual_values) == len(expected_values)
    assert all(
        isclose(actual_value, expected_value, abs_tol=tolerance)
        for actual_value, expected_value in zip(
            actual_values,
            expected_values,
            strict=True,
        )
    )


def test_object_interaction_g1_uses_rubber_hand_asset() -> None:
    config_source = ROBOT_CONFIG.read_text()
    assert 'urdf_file="g1/main_mesh_collision_rubberhand.urdf"' in config_source
    assert 'urdf_file="g1/main_mesh_collision_halfspherehand.urdf"' not in config_source


def test_wbt_reward_allows_rubber_hand_contacts() -> None:
    pattern = g1_29dof_wbt_reward.terms["undesired_contacts"].params[
        "undesired_contacts_body_names"
    ]

    allowed_contact_bodies = {
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_rubber_hand_link",
        "right_rubber_hand_link",
    }
    for body_name in allowed_contact_bodies:
        assert re.match(pattern, body_name) is None

    for body_name in {"left_knee_link", "right_hip_yaw_link", "torso_link"}:
        assert re.match(pattern, body_name) is not None


def test_rubber_hand_asset_has_matching_visual_and_collision_geometry() -> None:
    urdf_source = RUBBER_HAND_URDF.read_text()
    assert "sphere_hand" not in urdf_source
    assert "half_sphere" not in urdf_source

    root = ET.fromstring(urdf_source)
    hand_specs = {
        "left": ("0.0415 0.003 0", "meshes/left_rubber_hand.STL"),
        "right": ("0.0415 -0.003 0", "meshes/right_rubber_hand.STL"),
    }

    for side, (expected_xyz, expected_mesh) in hand_specs.items():
        link_name = f"{side}_rubber_hand_link"
        joint = _required_element(root, f"./joint[@name='{side}_hand_palm_joint']")
        origin = _required_element(joint, "./origin")
        child = _required_element(joint, "./child")
        assert origin.attrib == {"xyz": expected_xyz, "rpy": "0 0 0"}
        assert child.attrib["link"] == link_name

        link = _required_element(root, f"./link[@name='{link_name}']")
        visual_mesh = _required_element(link, "./visual/geometry/mesh")
        collision_mesh = _required_element(link, "./collision/geometry/mesh")
        assert visual_mesh.attrib["filename"] == expected_mesh
        assert collision_mesh.attrib["filename"] == expected_mesh
        assert (ROBOT_DIR / expected_mesh).is_file()


def test_retargeting_and_training_rubber_hand_kinematics_match() -> None:
    urdf_root = ET.parse(RUBBER_HAND_URDF).getroot()
    mjcf_root = ET.parse(RETARGETING_ROBOT_XML).getroot()

    urdf_revolute_joint_names = [
        joint.attrib["name"]
        for joint in urdf_root.findall("./joint")
        if joint.attrib.get("type") == "revolute"
    ]
    mjcf_joint_names = [
        joint.attrib["name"]
        for joint in mjcf_root.findall(".//joint")
        if "name" in joint.attrib
    ]
    assert len(urdf_revolute_joint_names) == 29
    assert urdf_revolute_joint_names == mjcf_joint_names

    wrist_specs = {
        "roll": {
            "axis": "1 0 0",
            "origin_xyz": {
                "left": "0.100 0.00188791 -0.010",
                "right": "0.100 -0.00188791 -0.010",
            },
            "lower": "-1.972222054",
            "upper": "1.972222054",
        },
        "pitch": {
            "axis": "0 1 0",
            "origin_xyz": {"left": "0.038 0 0", "right": "0.038 0 0"},
            "lower": "-1.614429558",
            "upper": "1.614429558",
        },
        "yaw": {
            "axis": "0 0 1",
            "origin_xyz": {"left": "0.046 0 0", "right": "0.046 0 0"},
            "lower": "-1.614429558",
            "upper": "1.614429558",
        },
    }

    for side in ("left", "right"):
        for suffix, wrist_spec in wrist_specs.items():
            joint_name = f"{side}_wrist_{suffix}_joint"
            child_link_name = f"{side}_wrist_{suffix}_link"
            urdf_joint = _required_element(
                urdf_root,
                f"./joint[@name='{joint_name}']",
            )
            urdf_origin = _required_element(urdf_joint, "./origin")
            urdf_axis = _required_element(urdf_joint, "./axis")
            urdf_limit = _required_element(urdf_joint, "./limit")
            mjcf_joint = _required_element(
                mjcf_root,
                f".//joint[@name='{joint_name}']",
            )
            mjcf_body = _required_element(
                mjcf_root,
                f".//body[@name='{child_link_name}']",
            )

            expected_origin = wrist_spec["origin_xyz"][side]
            _assert_float_sequence_equal(
                urdf_origin.attrib["xyz"],
                expected_origin,
            )
            _assert_float_sequence_equal(
                mjcf_body.attrib["pos"],
                expected_origin,
            )
            _assert_float_sequence_equal(
                urdf_axis.attrib["xyz"],
                wrist_spec["axis"],
            )
            _assert_float_sequence_equal(
                mjcf_joint.attrib["axis"],
                wrist_spec["axis"],
            )
            _assert_float_sequence_equal(
                urdf_limit.attrib["lower"],
                wrist_spec["lower"],
            )
            _assert_float_sequence_equal(
                urdf_limit.attrib["upper"],
                wrist_spec["upper"],
            )
            _assert_float_sequence_equal(
                mjcf_joint.attrib["range"],
                f"{wrist_spec['lower']} {wrist_spec['upper']}",
                tolerance=1e-5,
            )

        hand_link_name = f"{side}_rubber_hand_link"
        hand_joint = _required_element(
            urdf_root,
            f"./joint[@name='{side}_hand_palm_joint']",
        )
        hand_origin = _required_element(hand_joint, "./origin")
        hand_link = _required_element(
            urdf_root,
            f"./link[@name='{hand_link_name}']",
        )
        hand_inertial = _required_element(hand_link, "./inertial")
        hand_mass = _required_element(hand_inertial, "./mass")
        hand_inertial_origin = _required_element(hand_inertial, "./origin")
        hand_collision_mesh = _required_element(
            hand_link,
            "./collision/geometry/mesh",
        )
        mjcf_hand_body = _required_element(
            mjcf_root,
            f".//body[@name='{hand_link_name}']",
        )
        mjcf_hand_inertial = _required_element(
            mjcf_hand_body,
            "./inertial",
        )
        mjcf_hand_collision = _required_element(
            mjcf_hand_body,
            f"./geom[@name='{hand_link_name}']",
        )

        _assert_float_sequence_equal(
            hand_origin.attrib["xyz"],
            mjcf_hand_body.attrib["pos"],
        )
        _assert_float_sequence_equal(
            hand_mass.attrib["value"],
            mjcf_hand_inertial.attrib["mass"],
        )
        _assert_float_sequence_equal(
            hand_inertial_origin.attrib["xyz"],
            mjcf_hand_inertial.attrib["pos"],
        )
        assert hand_collision_mesh.attrib["filename"] == (
            f"meshes/{side}_rubber_hand.STL"
        )
        assert mjcf_hand_collision.attrib == {
            "name": hand_link_name,
            "type": "mesh",
            "mesh": hand_link_name,
        }


def test_training_and_retargeting_largetable_assets_are_identical() -> None:
    assert TRAINING_TABLE_URDF.read_bytes() == RETARGETING_TABLE_URDF.read_bytes()
    assert TRAINING_TABLE_MESH.read_bytes() == RETARGETING_TABLE_MESH.read_bytes()
