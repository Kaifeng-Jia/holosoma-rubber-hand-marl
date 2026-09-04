"""Build isolated CORE4D object assets for the existing OmniRetarget model.

The source CORE4D mesh is used for both visual and collision geometry for
ordinary objects, matching the original OMOMO largebox setup.  The verified
desk001 proxy and caller-supplied compound boxes remain available for objects
whose scanned mesh should not be treated as one collision hull.
"""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh


_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_RUBBER_HAND_NAMES = ("left_rubber_hand_link", "right_rubber_hand_link")


@dataclass(frozen=True)
class BoxCollisionSpec:
    """One axis-aligned collision box in the canonical object frame.

    ``size`` stores full side lengths, matching URDF's box convention.  The
    MJCF writer converts them to half extents.
    """

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


# desk001 is Y-up in its canonical CORE4D mesh frame.  These five primitives
# follow the scanned geometry: one tabletop, two central vertical supports at
# the front/rear, and two wide feet at the bottom.  They are deliberately
# explicit and replaceable; they are not inferred physics metadata.
DEFAULT_DESK001_COLLISION_BOXES: tuple[BoxCollisionSpec, ...] = (
    BoxCollisionSpec(
        name="tabletop",
        center=(0.0, 0.3415, 0.0),
        size=(0.5986, 0.0470, 0.7990),
    ),
    BoxCollisionSpec(
        name="vertical_support_negative_z",
        center=(0.0, -0.0050, -0.2900),
        size=(0.0600, 0.6550, 0.0750),
    ),
    BoxCollisionSpec(
        name="vertical_support_positive_z",
        center=(0.0, -0.0050, 0.2900),
        size=(0.0600, 0.6550, 0.0750),
    ),
    BoxCollisionSpec(
        name="bottom_foot_negative_z",
        center=(0.0, -0.3400, -0.2900),
        size=(0.5500, 0.0500, 0.1000),
    ),
    BoxCollisionSpec(
        name="bottom_foot_positive_z",
        center=(0.0, -0.3400, 0.2900),
        size=(0.5500, 0.0500, 0.1000),
    ),
)


@dataclass(frozen=True)
class Core4DObjectAssetPaths:
    """Paths and validated collision metadata for one generated asset pair."""

    object_urdf_path: Path
    scene_xml_path: Path
    source_mesh_path: Path
    collision_boxes: tuple[BoxCollisionSpec, ...]
    collision_strategy: str


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _validate_name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{field} must match {_SAFE_NAME.pattern!r}; got {value!r}"
        )
    return value


def _validate_vector(
    values: Sequence[float],
    *,
    field: str,
    strictly_positive: bool,
) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{field} must have shape (3,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field} must contain only finite values")
    if strictly_positive and np.any(vector <= 0.0):
        raise ValueError(f"{field} must contain strictly positive values")
    return tuple(float(value) for value in vector)


def _validate_collision_boxes(
    boxes: Sequence[BoxCollisionSpec],
) -> tuple[BoxCollisionSpec, ...]:
    if isinstance(boxes, (str, bytes)):
        raise TypeError("collision_boxes must be a sequence of BoxCollisionSpec")
    validated: list[BoxCollisionSpec] = []
    for index, box in enumerate(boxes):
        if not isinstance(box, BoxCollisionSpec):
            raise TypeError(
                "collision_boxes entries must be BoxCollisionSpec, "
                f"got {type(box).__name__} at index {index}"
            )
        name = _validate_name(box.name, field=f"collision_boxes[{index}].name")
        center = _validate_vector(
            box.center,
            field=f"collision_boxes[{index}].center",
            strictly_positive=False,
        )
        size = _validate_vector(
            box.size,
            field=f"collision_boxes[{index}].size",
            strictly_positive=True,
        )
        validated.append(BoxCollisionSpec(name=name, center=center, size=size))

    if not validated:
        raise ValueError("CORE4D collision geometry must contain at least one box")
    names = [box.name for box in validated]
    if len(set(names)) != len(names):
        raise ValueError("collision box names must be unique")
    return tuple(validated)


def _mesh_mass_properties(
    mesh_path: Path,
    mass_kg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated uniform-density mass properties for a closed mesh."""
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"CORE4D object mesh scene is empty: {mesh_path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"Unable to load CORE4D object mesh: {mesh_path}")
    mesh_mass = float(loaded.mass)
    center_of_mass = np.asarray(loaded.center_mass, dtype=np.float64)
    inertia = np.asarray(loaded.moment_inertia, dtype=np.float64)
    if (
        not math.isfinite(mesh_mass)
        or mesh_mass <= 0.0
        or center_of_mass.shape != (3,)
        or inertia.shape != (3, 3)
        or not np.isfinite(center_of_mass).all()
        or not np.isfinite(inertia).all()
    ):
        raise ValueError(
            "CORE4D collision mesh must be a closed volume with finite positive "
            f"mass properties: {mesh_path}"
        )
    inertia = inertia * (mass_kg / mesh_mass)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.any(eigenvalues <= 0.0):
        raise ValueError(f"CORE4D collision mesh has non-positive inertia: {mesh_path}")
    return center_of_mass, inertia


def _compound_box_inertia(
    boxes: Sequence[BoxCollisionSpec],
    mass_kg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return center of mass and inertia tensor for uniform-density boxes."""
    sizes = np.asarray([box.size for box in boxes], dtype=np.float64)
    centers = np.asarray([box.center for box in boxes], dtype=np.float64)
    volumes = np.prod(sizes, axis=1)
    masses = mass_kg * volumes / volumes.sum()
    center_of_mass = np.sum(masses[:, None] * centers, axis=0) / mass_kg

    inertia = np.zeros((3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for size, center, box_mass in zip(sizes, centers, masses):
        x_size, y_size, z_size = size
        local = np.diag(
            [
                box_mass * (y_size**2 + z_size**2) / 12.0,
                box_mass * (x_size**2 + z_size**2) / 12.0,
                box_mass * (x_size**2 + y_size**2) / 12.0,
            ]
        )
        offset = center - center_of_mass
        inertia += local + box_mass * (
            np.dot(offset, offset) * identity - np.outer(offset, offset)
        )

    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        raise ValueError("collision boxes produced a non-positive inertia tensor")
    return center_of_mass, inertia


def _parse_rubber_hand_base_xml(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    try:
        tree = ET.parse(path, parser=parser)
    except ET.ParseError as exc:
        raise ValueError(f"Base G1 XML is not well formed: {path}") from exc
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Base G1 XML root must be <mujoco>, got <{root.tag}>")
    if root.find("worldbody") is None:
        raise ValueError("Base G1 XML is missing <worldbody>")
    if root.find("asset") is None:
        raise ValueError("Base G1 XML is missing <asset>")

    body_names = {body.get("name") for body in root.iter("body")}
    mesh_names = {mesh.get("name") for mesh in root.iter("mesh")}
    missing_bodies = set(_RUBBER_HAND_NAMES).difference(body_names)
    missing_meshes = set(_RUBBER_HAND_NAMES).difference(mesh_names)
    if missing_bodies or missing_meshes:
        raise ValueError(
            "Base G1 XML must contain rubber-hand bodies and meshes; "
            f"missing bodies={sorted(missing_bodies)}, meshes={sorted(missing_meshes)}"
        )
    return tree


def _make_relocatable(tree: ET.ElementTree, base_xml_path: Path) -> None:
    """Keep relative base-model assets valid after writing XML elsewhere."""
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)

    has_asset_root = False
    for attribute in ("assetdir", "meshdir", "texturedir"):
        raw_value = compiler.get(attribute)
        if raw_value is None:
            continue
        has_asset_root = True
        value = Path(raw_value).expanduser()
        if not value.is_absolute():
            value = (base_xml_path.parent / value).resolve()
        compiler.set(attribute, str(value))
    if not has_asset_root:
        compiler.set("assetdir", str(base_xml_path.parent.resolve()))


def _assert_generated_names_are_new(
    root: ET.Element,
    *,
    object_name: str,
    mesh_asset_name: str,
    boxes: Sequence[BoxCollisionSpec],
    mesh_collision: bool,
) -> None:
    existing = {
        element.get("name")
        for tag in ("body", "joint", "freejoint", "geom", "mesh")
        for element in root.iter(tag)
        if element.get("name") is not None
    }
    generated = {
        object_name,
        mesh_asset_name,
        f"{object_name}_freejoint",
        f"{object_name}_visual",
        *([f"{object_name}_mesh_collision"] if mesh_collision else []),
        *(f"{object_name}_{box.name}" for box in boxes),
    }
    collisions = existing.intersection(generated)
    if collisions:
        raise ValueError(
            "Generated CORE4D object names collide with the base G1 XML: "
            f"{sorted(collisions)}"
        )


def _build_object_urdf(
    *,
    object_name: str,
    mesh_path: Path,
    boxes: Sequence[BoxCollisionSpec],
    mesh_collision: bool,
    center_of_mass: np.ndarray,
    inertia: np.ndarray,
    mass_kg: float,
) -> ET.ElementTree:
    robot = ET.Element("robot", {"name": object_name})
    link = ET.SubElement(robot, "link", {"name": f"{object_name}_link"})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": _format_vector(center_of_mass), "rpy": "0 0 0"},
    )
    ET.SubElement(inertial, "mass", {"value": f"{mass_kg:.12g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia[0, 0]:.12g}",
            "ixy": f"{inertia[0, 1]:.12g}",
            "ixz": f"{inertia[0, 2]:.12g}",
            "iyy": f"{inertia[1, 1]:.12g}",
            "iyz": f"{inertia[1, 2]:.12g}",
            "izz": f"{inertia[2, 2]:.12g}",
        },
    )

    visual = ET.SubElement(link, "visual", {"name": f"{object_name}_visual"})
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(
        visual_geometry,
        "mesh",
        {"filename": str(mesh_path), "scale": "1 1 1"},
    )

    if mesh_collision:
        collision = ET.SubElement(
            link,
            "collision",
            {"name": f"{object_name}_mesh_collision"},
        )
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {"filename": str(mesh_path), "scale": "1 1 1"},
        )
    for box in boxes:
        collision = ET.SubElement(
            link,
            "collision",
            {"name": f"{object_name}_{box.name}"},
        )
        ET.SubElement(
            collision,
            "origin",
            {"xyz": _format_vector(box.center), "rpy": "0 0 0"},
        )
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", {"size": _format_vector(box.size)})
    return ET.ElementTree(robot)


def _add_object_to_scene(
    tree: ET.ElementTree,
    *,
    object_name: str,
    mesh_path: Path,
    boxes: Sequence[BoxCollisionSpec],
    mesh_collision: bool,
    center_of_mass: np.ndarray,
    inertia: np.ndarray,
    mass_kg: float,
) -> None:
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:  # guarded by base validation
        raise RuntimeError("Validated G1 XML lost its asset or worldbody section")

    mesh_asset_name = f"core4d_{object_name}_visual_mesh"
    _assert_generated_names_are_new(
        root,
        object_name=object_name,
        mesh_asset_name=mesh_asset_name,
        boxes=boxes,
        mesh_collision=mesh_collision,
    )
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": mesh_asset_name,
            "file": str(mesh_path),
            "scale": "1 1 1",
        },
    )

    object_body = ET.SubElement(
        worldbody,
        "body",
        {"name": object_name, "pos": "0 0 0", "quat": "1 0 0 0"},
    )
    ET.SubElement(
        object_body,
        "inertial",
        {
            "pos": _format_vector(center_of_mass),
            "mass": f"{mass_kg:.12g}",
            "fullinertia": _format_vector(
                (
                    inertia[0, 0],
                    inertia[1, 1],
                    inertia[2, 2],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 2],
                )
            ),
        },
    )
    if mesh_collision:
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": f"{object_name}_mesh_collision",
                "type": "mesh",
                "mesh": mesh_asset_name,
                "contype": "1",
                "conaffinity": "15",
                "group": "3",
                "rgba": "0.3 0.5 0.8 0.15",
            },
        )
    ET.SubElement(object_body, "freejoint", {"name": f"{object_name}_freejoint"})
    ET.SubElement(
        object_body,
        "geom",
        {
            "name": f"{object_name}_visual",
            "type": "mesh",
            "mesh": mesh_asset_name,
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "1",
            "rgba": "0.7 0.8 0.9 0.8",
        },
    )
    for box in boxes:
        half_size = 0.5 * np.asarray(box.size, dtype=np.float64)
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": f"{object_name}_{box.name}",
                "type": "box",
                "pos": _format_vector(box.center),
                "size": _format_vector(half_size),
                "contype": "1",
                "conaffinity": "15",
                "group": "3",
                "rgba": "0.3 0.5 0.8 0.15",
            },
        )


def _write_new_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    payload = ET.tostring(
        tree.getroot(),
        encoding="utf-8",
        xml_declaration=True,
    )
    with path.open("xb") as output:
        output.write(payload)
        output.write(b"\n")


def create_core4d_retarget_assets(
    mesh_path: str | Path,
    base_g1_xml_path: str | Path,
    object_name: str,
    output_dir: str | Path,
    *,
    collision_boxes: Sequence[BoxCollisionSpec] | None = None,
    mass_kg: float = 1.0,
) -> Core4DObjectAssetPaths:
    """Create a temporary object URDF and rubber-hand G1+object MJCF scene.

    The source mesh is never copied.  ``mass_kg`` is a retarget-scene
    placeholder used only to make the free body well formed; downstream
    simulation must define and record its own physical parameters.
    """
    object_name = _validate_name(object_name, field="object_name")
    source_mesh = Path(mesh_path).expanduser().resolve()
    if not source_mesh.is_file():
        raise FileNotFoundError(f"CORE4D object mesh does not exist: {source_mesh}")
    base_xml = Path(base_g1_xml_path).expanduser().resolve()
    if not base_xml.is_file():
        raise FileNotFoundError(f"Base G1 XML does not exist: {base_xml}")
    if not isinstance(mass_kg, (int, float, np.integer, np.floating)):
        raise TypeError(f"mass_kg must be numeric, got {type(mass_kg).__name__}")
    mass_kg = float(mass_kg)
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError(f"mass_kg must be finite and positive, got {mass_kg!r}")

    if collision_boxes is None:
        if object_name.casefold() == "desk001":
            collision_strategy = "desk001_explicit_five_box_proxy"
            candidate_boxes = DEFAULT_DESK001_COLLISION_BOXES
            mesh_collision = False
        else:
            collision_strategy = "source_mesh_visual_and_collision"
            candidate_boxes = ()
            mesh_collision = True
    else:
        collision_strategy = "caller_supplied_box_primitives"
        candidate_boxes = collision_boxes
        mesh_collision = False
    boxes = (
        () if mesh_collision else _validate_collision_boxes(candidate_boxes)
    )
    if mesh_collision:
        center_of_mass, inertia = _mesh_mass_properties(source_mesh, mass_kg)
    else:
        center_of_mass, inertia = _compound_box_inertia(boxes, mass_kg)
    scene_tree = _parse_rubber_hand_base_xml(base_xml)
    scene_tree = copy.deepcopy(scene_tree)
    _make_relocatable(scene_tree, base_xml)
    _add_object_to_scene(
        scene_tree,
        object_name=object_name,
        mesh_path=source_mesh,
        boxes=boxes,
        mesh_collision=mesh_collision,
        center_of_mass=center_of_mass,
        inertia=inertia,
        mass_kg=mass_kg,
    )
    urdf_tree = _build_object_urdf(
        object_name=object_name,
        mesh_path=source_mesh,
        boxes=boxes,
        mesh_collision=mesh_collision,
        center_of_mass=center_of_mass,
        inertia=inertia,
        mass_kg=mass_kg,
    )

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {destination}")
    object_urdf_path = destination / f"{object_name}.urdf"
    scene_xml_path = destination / f"{base_xml.stem}_w_{object_name}.xml"
    existing_outputs = [
        path for path in (object_urdf_path, scene_xml_path) if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite generated CORE4D assets: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    destination.mkdir(parents=True, exist_ok=True)
    try:
        _write_new_xml(urdf_tree, object_urdf_path)
        _write_new_xml(scene_tree, scene_xml_path)
    except Exception:
        # Only remove a file created by this call; never touch caller inputs.
        if object_urdf_path.exists() and not scene_xml_path.exists():
            object_urdf_path.unlink()
        raise

    return Core4DObjectAssetPaths(
        object_urdf_path=object_urdf_path,
        scene_xml_path=scene_xml_path,
        source_mesh_path=source_mesh,
        collision_boxes=boxes,
        collision_strategy=collision_strategy,
    )


__all__ = [
    "BoxCollisionSpec",
    "Core4DObjectAssetPaths",
    "DEFAULT_DESK001_COLLISION_BOXES",
    "create_core4d_retarget_assets",
]
