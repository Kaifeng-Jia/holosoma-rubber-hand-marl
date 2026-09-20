"""CPU-only checks for opt-in object collision settings and the approved chair asset.

The AST forwarding check does not import Isaac Sim or claim to validate its
cooked convex geometry; that remains a separate simulator smoke check.
"""

from __future__ import annotations

import ast
import hashlib
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from holosoma.config_types.robot import ObjectConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACSIM_SOURCE = REPO_ROOT / "src/holosoma/holosoma/simulator/isaacsim/isaacsim.py"
ASSET_DIR = (
    REPO_ROOT
    / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
    / "core4d_chair021_20231020_074_a1"
)
SOURCE_MESH_SHA256 = "7fb3283f1b67c8d08bb1f68c51d15aab5e6c940b8571e704bca265701b776161"
PREVIEW_INERTIA = {
    "ixx": "0.0250620451088",
    "ixy": "0.00630934282315",
    "ixz": "-0.0001306071754",
    "iyy": "0.025359090737",
    "iyz": "-0.00037383479194",
    "izz": "0.0244425616507",
}


def _object_spawn_call() -> ast.Call:
    tree = ast.parse(ISAACSIM_SOURCE.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "UrdfFileCfg"
        and any(
            keyword.arg == "asset_path"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "object_asset_urdf_path"
            for keyword in node.keywords
        )
    ]
    assert len(calls) == 1
    return calls[0]


def test_object_collision_default_preserves_existing_experiments() -> None:
    assert ObjectConfig().collider_type == "convex_hull"
    assert ObjectConfig(object_urdf_path="existing.urdf").collider_type == "convex_hull"


@pytest.mark.parametrize("collider_type", ["convex_hull", "convex_decomposition"])
def test_object_collision_choice_is_forwarded_to_isaac_importer(collider_type: str) -> None:
    config = ObjectConfig(object_urdf_path="chair021_training.urdf", collider_type=collider_type)
    keywords = [keyword for keyword in _object_spawn_call().keywords if keyword.arg == "collider_type"]
    assert len(keywords) == 1
    expression = ast.Expression(body=keywords[0].value)
    context = SimpleNamespace(robot_config=SimpleNamespace(object=config))
    forwarded = eval(compile(expression, str(ISAACSIM_SOURCE), "eval"), {"self": context})
    assert forwarded == collider_type


def test_object_collision_rejects_unsupported_choices() -> None:
    with pytest.raises(ValidationError):
        ObjectConfig(collider_type="triangle_mesh")


def test_new_collision_option_does_not_change_robot_import_configuration() -> None:
    tree = ast.parse(ISAACSIM_SOURCE.read_text(encoding="utf-8"))
    importer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "UrdfFileCfg"
    ]
    overrides = [
        keyword
        for call in importer_calls
        for keyword in call.keywords
        if keyword.arg == "collider_type"
    ]
    assert len(importer_calls) >= 2
    assert len(overrides) == 1
    assert ast.dump(overrides[0]) == ast.dump(
        next(keyword for keyword in _object_spawn_call().keywords if keyword.arg == "collider_type")
    )


def test_chair_mass_and_inertia_follow_approved_five_kg_change() -> None:
    root = ET.parse(ASSET_DIR / "chair021_training.urdf").getroot()
    assert root.attrib == {"name": "chair021"}
    assert len(root.findall("link")) == 1
    assert root.find("joint") is None
    inertial = root.find("./link/inertial")
    assert inertial is not None
    assert inertial.find("mass").get("value") == "5"
    assert inertial.find("origin").attrib == {
        "xyz": "-0.0153408561273 0.0748650554502 -0.00380911442266",
        "rpy": "0 0 0",
    }
    inertia = inertial.find("inertia").attrib
    assert set(inertia) == set(PREVIEW_INERTIA)
    for key, original in PREVIEW_INERTIA.items():
        assert Decimal(inertia[key]) == Decimal(original) * 5


def test_chair_visual_and_collision_keep_preview_scale_and_origin() -> None:
    root = ET.parse(ASSET_DIR / "chair021_training.urdf").getroot()
    for tag in ("visual", "collision"):
        elements = root.findall(f"./link/{tag}")
        assert len(elements) == 1
        assert elements[0].find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}
        assert elements[0].find("geometry/mesh").attrib == {
            "filename": "chair021_m.obj",
            "scale": "0.744193062691 0.744193062691 0.744193062691",
        }


def test_chair_mesh_is_portable_and_matches_original_geometry() -> None:
    root = ET.parse(ASSET_DIR / "chair021_training.urdf").getroot()
    for mesh in root.findall(".//mesh"):
        relative = Path(mesh.get("filename"))
        assert not relative.is_absolute()
        assert relative.parts == ("chair021_m.obj",)
        payload = (ASSET_DIR / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == SOURCE_MESH_SHA256
        assert not any(line.startswith(b"mtllib ") for line in payload.splitlines())
