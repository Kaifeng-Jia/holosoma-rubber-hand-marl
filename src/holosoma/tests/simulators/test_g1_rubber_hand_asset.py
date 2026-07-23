from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DIR = PACKAGE_ROOT / "holosoma" / "data" / "robots" / "g1"
RUBBER_HAND_URDF = ROBOT_DIR / "main_mesh_collision_rubberhand.urdf"
ROBOT_CONFIG = PACKAGE_ROOT / "holosoma" / "config_values" / "robot.py"


def _required_element(root: ET.Element, xpath: str) -> ET.Element:
    element = root.find(xpath)
    assert element is not None, f"Missing URDF element: {xpath}"
    return element


def test_object_interaction_g1_uses_rubber_hand_asset() -> None:
    config_source = ROBOT_CONFIG.read_text()
    assert 'urdf_file="g1/main_mesh_collision_rubberhand.urdf"' in config_source
    assert 'urdf_file="g1/main_mesh_collision_halfspherehand.urdf"' not in config_source


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
