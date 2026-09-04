from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from holosoma_retargeting.data_utils.core4d_object_assets import (
    BoxCollisionSpec,
    DEFAULT_DESK001_COLLISION_BOXES,
    create_core4d_retarget_assets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_G1_XML = (
    REPO_ROOT
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "models"
    / "g1"
    / "g1_29dof.xml"
)


def _write_tetrahedron_obj(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 3 2",
                "f 1 2 4",
                "f 1 4 3",
                "f 2 3 4",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _named(root: ET.Element, tag: str, name: str) -> ET.Element:
    matches = [element for element in root.iter(tag) if element.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _semantic_signature(element: ET.Element) -> tuple[object, ...]:
    """Compare XML content while ignoring indentation-only whitespace."""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_semantic_signature(child) for child in element),
    )


def test_create_assets_preserves_rubber_hand_model_and_uses_five_boxes(
    tmp_path: Path,
) -> None:
    source_mesh = _write_tetrahedron_obj(tmp_path / "dataset" / "desk001_m.obj")
    output_dir = tmp_path / "generated"
    base_bytes = BASE_G1_XML.read_bytes()
    base_root = _parse(BASE_G1_XML)

    result = create_core4d_retarget_assets(
        mesh_path=source_mesh,
        base_g1_xml_path=BASE_G1_XML,
        object_name="desk001",
        output_dir=output_dir,
    )

    assert BASE_G1_XML.read_bytes() == base_bytes
    assert result.source_mesh_path == source_mesh.resolve()
    assert result.object_urdf_path.parent == output_dir.resolve()
    assert result.scene_xml_path.parent == output_dir.resolve()
    assert {path.name for path in output_dir.iterdir()} == {
        "desk001.urdf",
        "g1_29dof_w_desk001.xml",
    }

    scene_root = _parse(result.scene_xml_path)
    for hand_name in ("left_rubber_hand_link", "right_rubber_hand_link"):
        original = _semantic_signature(_named(base_root, "body", hand_name))
        generated = _semantic_signature(_named(scene_root, "body", hand_name))
        assert generated == original
        assert _named(scene_root, "mesh", hand_name).get("file") == _named(
            base_root, "mesh", hand_name
        ).get("file")

    compiler = scene_root.find("compiler")
    assert compiler is not None
    assert Path(compiler.get("meshdir", "")).is_absolute()

    visual_asset = _named(scene_root, "mesh", "core4d_desk001_visual_mesh")
    assert visual_asset.get("file") == str(source_mesh.resolve())
    object_body = _named(scene_root, "body", "desk001")
    freejoints = list(object_body.iter("freejoint"))
    assert len(freejoints) == 1
    assert freejoints[0].get("name") == "desk001_freejoint"

    visual_geom = _named(scene_root, "geom", "desk001_visual")
    assert visual_geom.get("type") == "mesh"
    assert visual_geom.get("contype") == "0"
    collision_geoms = [
        _named(scene_root, "geom", f"desk001_{box.name}")
        for box in DEFAULT_DESK001_COLLISION_BOXES
    ]
    assert len(collision_geoms) == 5
    assert all(geom.get("type") == "box" for geom in collision_geoms)
    assert all(geom.get("mesh") is None for geom in collision_geoms)
    for geom, box in zip(collision_geoms, DEFAULT_DESK001_COLLISION_BOXES):
        np.testing.assert_allclose(
            np.fromstring(geom.get("pos", ""), sep=" "),
            box.center,
        )
        np.testing.assert_allclose(
            np.fromstring(geom.get("size", ""), sep=" "),
            0.5 * np.asarray(box.size),
        )

    urdf_root = _parse(result.object_urdf_path)
    urdf_mesh = urdf_root.find("./link/visual/geometry/mesh")
    assert urdf_mesh is not None
    assert urdf_mesh.get("filename") == str(source_mesh.resolve())
    urdf_collisions = urdf_root.findall("./link/collision")
    assert len(urdf_collisions) == 5
    assert all(collision.find("./geometry/mesh") is None for collision in urdf_collisions)
    for collision, box in zip(urdf_collisions, DEFAULT_DESK001_COLLISION_BOXES):
        urdf_box = collision.find("./geometry/box")
        assert urdf_box is not None
        np.testing.assert_allclose(
            np.fromstring(urdf_box.get("size", ""), sep=" "),
            box.size,
        )


def test_explicit_collision_boxes_are_used_without_copying_mesh(tmp_path: Path) -> None:
    source_mesh = _write_tetrahedron_obj(tmp_path / "external" / "custom.obj")
    custom_boxes = tuple(
        BoxCollisionSpec(
            name=f"part_{index}",
            center=(float(index), -float(index), 0.25 * index),
            size=(0.1 + index, 0.2 + index, 0.3 + index),
        )
        for index in range(5)
    )
    result = create_core4d_retarget_assets(
        source_mesh,
        BASE_G1_XML,
        "custom_desk",
        tmp_path / "output",
        collision_boxes=custom_boxes,
        mass_kg=2.0,
    )

    scene_root = _parse(result.scene_xml_path)
    object_body = _named(scene_root, "body", "custom_desk")
    for box in custom_boxes:
        geom = _named(object_body, "geom", f"custom_desk_{box.name}")
        np.testing.assert_allclose(
            np.fromstring(geom.get("size", ""), sep=" "),
            0.5 * np.asarray(box.size),
        )
    assert not (result.scene_xml_path.parent / source_mesh.name).exists()
    assert _named(
        scene_root, "mesh", "core4d_custom_desk_visual_mesh"
    ).get("file") == str(source_mesh.resolve())


def test_non_desk_object_uses_source_mesh_for_visual_and_collision(
    tmp_path: Path,
) -> None:
    source_mesh = _write_tetrahedron_obj(tmp_path / "external" / "box026_m.obj")
    result = create_core4d_retarget_assets(
        source_mesh,
        BASE_G1_XML,
        "Box026",
        tmp_path / "output",
    )

    assert result.collision_strategy == "source_mesh_visual_and_collision"
    assert result.collision_boxes == ()

    scene_root = _parse(result.scene_xml_path)
    visual_asset = _named(scene_root, "mesh", "core4d_Box026_visual_mesh")
    assert visual_asset.get("file") == str(source_mesh.resolve())
    scene_visual = _named(scene_root, "geom", "Box026_visual")
    scene_collision = _named(scene_root, "geom", "Box026_mesh_collision")
    assert scene_visual.get("type") == "mesh"
    assert scene_visual.get("mesh") == visual_asset.get("name")
    assert scene_visual.get("contype") == "0"
    assert scene_collision.get("type") == "mesh"
    assert scene_collision.get("mesh") == visual_asset.get("name")
    assert scene_collision.get("contype") == "1"

    urdf_root = _parse(result.object_urdf_path)
    urdf_visual_mesh = urdf_root.find("./link/visual/geometry/mesh")
    urdf_collisions = urdf_root.findall("./link/collision")
    assert urdf_visual_mesh is not None
    assert len(urdf_collisions) == 1
    urdf_collision_mesh = urdf_collisions[0].find("./geometry/mesh")
    assert urdf_collision_mesh is not None
    assert urdf_visual_mesh.get("filename") == str(source_mesh.resolve())
    assert urdf_collision_mesh.get("filename") == str(source_mesh.resolve())

    inertia_node = urdf_root.find("./link/inertial/inertia")
    assert inertia_node is not None
    inertia = np.asarray(
        [
            [
                float(inertia_node.get("ixx")),
                float(inertia_node.get("ixy")),
                float(inertia_node.get("ixz")),
            ],
            [
                float(inertia_node.get("ixy")),
                float(inertia_node.get("iyy")),
                float(inertia_node.get("iyz")),
            ],
            [
                float(inertia_node.get("ixz")),
                float(inertia_node.get("iyz")),
                float(inertia_node.get("izz")),
            ],
        ]
    )
    assert np.all(np.linalg.eigvalsh(inertia) > 0.0)


@pytest.mark.parametrize(
    ("boxes", "error"),
    [
        ((), "at least one"),
        (
            DEFAULT_DESK001_COLLISION_BOXES[:4]
            + (DEFAULT_DESK001_COLLISION_BOXES[0],),
            "names must be unique",
        ),
        (
            (
                BoxCollisionSpec("bad_center", (0.0, 1.0), (1.0, 1.0, 1.0)),
                *DEFAULT_DESK001_COLLISION_BOXES[1:],
            ),
            r"shape \(3,\)",
        ),
        (
            (
                BoxCollisionSpec("bad_size", (0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
                *DEFAULT_DESK001_COLLISION_BOXES[1:],
            ),
            "strictly positive",
        ),
        (
            (
                BoxCollisionSpec("bad_finite", (0.0, np.nan, 0.0), (1.0, 1.0, 1.0)),
                *DEFAULT_DESK001_COLLISION_BOXES[1:],
            ),
            "finite",
        ),
    ],
)
def test_collision_box_validation_is_strict(
    tmp_path: Path,
    boxes: tuple[BoxCollisionSpec, ...],
    error: str,
) -> None:
    source_mesh = _write_tetrahedron_obj(tmp_path / "external" / "desk.obj")
    with pytest.raises(ValueError, match=error):
        create_core4d_retarget_assets(
            source_mesh,
            BASE_G1_XML,
            "desk001",
            tmp_path / "output",
            collision_boxes=boxes,
        )
    assert not (tmp_path / "output").exists()


def test_paths_names_and_overwrite_are_validated_before_writing(tmp_path: Path) -> None:
    source_mesh = _write_tetrahedron_obj(tmp_path / "external" / "desk.obj")
    with pytest.raises(FileNotFoundError, match="mesh does not exist"):
        create_core4d_retarget_assets(
            tmp_path / "missing.obj",
            BASE_G1_XML,
            "desk001",
            tmp_path / "missing_output",
        )
    with pytest.raises(ValueError, match="object_name"):
        create_core4d_retarget_assets(
            source_mesh,
            BASE_G1_XML,
            "../desk001",
            tmp_path / "unsafe_output",
        )
    with pytest.raises(FileNotFoundError, match="Base G1 XML does not exist"):
        create_core4d_retarget_assets(
            source_mesh,
            tmp_path / "missing.xml",
            "desk001",
            tmp_path / "missing_base_output",
        )
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        create_core4d_retarget_assets(
            source_mesh,
            BASE_G1_XML,
            "desk001",
            output_file,
        )

    output_dir = tmp_path / "output"
    create_core4d_retarget_assets(
        source_mesh,
        BASE_G1_XML,
        "desk001",
        output_dir,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_core4d_retarget_assets(
            source_mesh,
            BASE_G1_XML,
            "desk001",
            output_dir,
        )


def test_generated_scene_loads_in_mujoco(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    source_mesh = _write_tetrahedron_obj(tmp_path / "external" / "desk.obj")
    result = create_core4d_retarget_assets(
        source_mesh,
        BASE_G1_XML,
        "desk001",
        tmp_path / "output",
    )

    base_model = mujoco.MjModel.from_xml_path(str(BASE_G1_XML))
    model = mujoco.MjModel.from_xml_path(str(result.scene_xml_path))
    assert model.nq == base_model.nq + 7
    assert model.nv == base_model.nv + 6

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "desk001")
    assert body_id >= 0
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "desk001_freejoint",
    )
    assert joint_id >= 0
    assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE

    visual_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "desk001_visual",
    )
    assert model.geom_type[visual_id] == mujoco.mjtGeom.mjGEOM_MESH
    for box in DEFAULT_DESK001_COLLISION_BOXES:
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"desk001_{box.name}",
        )
        assert geom_id >= 0
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX

    for hand_name in ("left_rubber_hand_link", "right_rubber_hand_link"):
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_name) >= 0
        )
