#!/usr/bin/env python3
"""Inspect a paired CORE4D robot retarget result in Viser."""

from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_URDF = (
    REPO_ROOT
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "models"
    / "g1"
    / "g1_29dof.urdf"
)
ROBOT_COUNT = 2
ROBOT_DOF = 29
ROBOT_QPOS_WIDTH = 7 + ROBOT_DOF
OBJECT_QPOS_WIDTH = 7
RUBBER_HAND_LINKS = {"left_rubber_hand_link", "right_rubber_hand_link"}
COLLISION_PROXY_ROOT = "/object/collision_proxies"
COLLISION_PROXY_COLOR = (255, 90, 30)
COLLISION_PROXY_OPACITY = 0.22
COLLISION_PROXY_INITIAL_VISIBLE = False
Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Pose = tuple[Vector3, Quaternion]
EXPECTED_ROBOT_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class PairReference:
    """Validated arrays used by the paired CORE4D viewer."""

    robot_qpos: np.ndarray
    object_qpos: np.ndarray
    fps: int
    object_name: str


@dataclass(frozen=True)
class CollisionBoxProxy:
    """One URDF collision box expressed in the object's root-link frame."""

    name: str
    position: Vector3
    wxyz: Quaternion
    dimensions: Vector3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Paired CORE4D NPZ.")
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=DEFAULT_ROBOT_URDF,
        help="Rubber-hand G1 URDF.",
    )
    parser.add_argument(
        "--object-urdf",
        type=Path,
        default=None,
        help="Object URDF; defaults to <input_dir>/assets/<object_name>.urdf.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Viser TCP port.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the NPZ and assets without starting Viser.",
    )
    return parser.parse_args(argv)


def _require_unit_quaternions(name: str, quaternions: np.ndarray) -> None:
    norms = np.linalg.norm(np.asarray(quaternions, dtype=np.float64), axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        max_error = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(f"{name} contains non-unit quaternions (max norm error {max_error:.9g})")


def load_pair_reference(path: str | Path) -> PairReference:
    """Load and strictly validate the pickle-free paired NPZ schema."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Paired CORE4D reference does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as archive:
        required = {"robot_qpos", "object_qpos", "fps", "object_name"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Paired CORE4D reference is missing keys: {missing}")
        robot_qpos = np.asarray(archive["robot_qpos"], dtype=np.float64).copy()
        object_qpos = np.asarray(archive["object_qpos"], dtype=np.float64).copy()
        fps_value = np.asarray(archive["fps"])
        object_name = str(np.asarray(archive["object_name"]).item())

    if robot_qpos.ndim != 3 or robot_qpos.shape[1:] != (ROBOT_COUNT, ROBOT_QPOS_WIDTH):
        raise ValueError(
            "robot_qpos must have shape "
            f"[T, {ROBOT_COUNT}, {ROBOT_QPOS_WIDTH}], got {robot_qpos.shape}"
        )
    if object_qpos.shape != (robot_qpos.shape[0], OBJECT_QPOS_WIDTH):
        raise ValueError(
            f"object_qpos must have shape [{robot_qpos.shape[0]}, {OBJECT_QPOS_WIDTH}], "
            f"got {object_qpos.shape}"
        )
    if robot_qpos.shape[0] == 0:
        raise ValueError("Paired CORE4D reference contains no frames")
    if fps_value.shape != ():
        raise ValueError(f"fps must be a scalar, got shape {fps_value.shape}")
    fps_scalar = fps_value.item()
    fps = int(fps_scalar)
    if not np.isfinite(fps_scalar) or float(fps_scalar) != float(fps):
        raise ValueError(f"fps must be a finite integer, got {fps_scalar}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not np.isfinite(robot_qpos).all() or not np.isfinite(object_qpos).all():
        raise ValueError("Paired CORE4D reference contains non-finite pose values")

    _require_unit_quaternions("robot_qpos", robot_qpos[..., 3:7])
    _require_unit_quaternions("object_qpos", object_qpos[:, 3:7])
    return PairReference(
        robot_qpos=robot_qpos,
        object_qpos=object_qpos,
        fps=fps,
        object_name=object_name,
    )


def resolve_object_urdf(
    input_path: str | Path,
    object_name: str,
    explicit_path: str | Path | None,
) -> Path:
    """Resolve the generated object URDF belonging to this retarget run."""
    if Path(object_name).name != object_name or object_name in {"", ".", ".."}:
        raise ValueError(f"object_name must be a plain filename stem, got {object_name!r}")
    path = (
        Path(explicit_path).expanduser().resolve()
        if explicit_path is not None
        else Path(input_path).expanduser().resolve().parent / "assets" / f"{object_name}.urdf"
    )
    if not path.is_file():
        raise FileNotFoundError(f"CORE4D object URDF does not exist: {path}")
    return path


def _parse_urdf_vector(
    raw_value: str | None,
    *,
    default: Vector3,
    field: str,
    strictly_positive: bool = False,
) -> Vector3:
    if raw_value is None:
        return default
    try:
        values = tuple(float(value) for value in raw_value.split())
    except ValueError as exc:
        raise ValueError(f"{field} must contain three finite numbers") from exc
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field} must contain three finite numbers")
    if strictly_positive and any(value <= 0.0 for value in values):
        raise ValueError(f"{field} must contain three strictly positive numbers")
    return values[0], values[1], values[2]


def _rpy_to_wxyz(rpy: Vector3) -> Quaternion:
    """Convert a URDF fixed-axis roll/pitch/yaw rotation to a wxyz quaternion."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _quaternion_multiply(
    left: Quaternion,
    right: Quaternion,
) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    product = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    norm = math.sqrt(sum(value * value for value in product))
    if norm == 0.0:  # pragma: no cover - products of parsed RPY rotations are unit
        raise ValueError("URDF origin rotations produced a zero quaternion")
    return (
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
        product[3] / norm,
    )


def _rotate_vector(
    wxyz: Quaternion,
    vector: Vector3,
) -> Vector3:
    w, x, y, z = wxyz
    vx, vy, vz = vector
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def _compose_pose(
    parent: Pose,
    child: Pose,
) -> Pose:
    parent_position, parent_wxyz = parent
    child_position, child_wxyz = child
    rotated_child_position = _rotate_vector(parent_wxyz, child_position)
    position = (
        parent_position[0] + rotated_child_position[0],
        parent_position[1] + rotated_child_position[1],
        parent_position[2] + rotated_child_position[2],
    )
    return position, _quaternion_multiply(parent_wxyz, child_wxyz)


def _urdf_origin_pose(
    origin: ET.Element | None,
    *,
    field: str,
) -> Pose:
    if origin is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    position = _parse_urdf_vector(
        origin.get("xyz"),
        default=(0.0, 0.0, 0.0),
        field=f"{field} xyz",
    )
    rpy = _parse_urdf_vector(
        origin.get("rpy"),
        default=(0.0, 0.0, 0.0),
        field=f"{field} rpy",
    )
    return position, _rpy_to_wxyz(rpy)


def load_collision_box_proxies(path: str | Path) -> tuple[CollisionBoxProxy, ...]:
    """Read only collision boxes and express them relative to the URDF base link.

    Visual geometry is intentionally never inspected.  Joint origins are
    composed at the URDF's zero configuration so boxes on fixed child links
    are still placed exactly where the object renderer puts them.
    """
    urdf_path = Path(path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"CORE4D object URDF does not exist: {urdf_path}")
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Object URDF is not well formed: {urdf_path}") from exc
    if root.tag != "robot":
        raise ValueError(f"Object URDF root must be <robot>, got <{root.tag}>")

    links: dict[str, ET.Element] = {}
    for index, link in enumerate(root.findall("link")):
        name = link.get("name")
        if not name:
            raise ValueError(f"Object URDF link {index} is missing a name")
        if name in links:
            raise ValueError(f"Object URDF contains duplicate link name {name!r}")
        links[name] = link
    if not links:
        raise ValueError("Object URDF contains no links")

    child_joints: dict[str, tuple[str, Pose]] = {}
    for index, joint in enumerate(root.findall("joint")):
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        parent_name = None if parent_node is None else parent_node.get("link")
        child_name = None if child_node is None else child_node.get("link")
        if parent_name not in links or child_name not in links:
            raise ValueError(f"Object URDF joint {index} references an unknown link")
        assert parent_name is not None and child_name is not None
        if child_name in child_joints:
            raise ValueError(f"Object URDF link {child_name!r} has multiple parent joints")
        child_joints[child_name] = (
            parent_name,
            _urdf_origin_pose(joint.find("origin"), field=f"joint {index} origin"),
        )

    root_links = set(links).difference(child_joints)
    if len(root_links) != 1:
        raise ValueError(
            "Object URDF must have exactly one root link, "
            f"found {sorted(root_links)}"
        )
    root_link = next(iter(root_links))
    identity_pose: Pose = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    link_poses: dict[str, Pose] = {root_link: identity_pose}

    def link_pose(
        link_name: str,
        visiting: frozenset[str] = frozenset(),
    ) -> Pose:
        cached = link_poses.get(link_name)
        if cached is not None:
            return cached
        if link_name in visiting or link_name not in child_joints:
            raise ValueError("Object URDF joint graph is disconnected or cyclic")
        parent_name, joint_pose = child_joints[link_name]
        pose = _compose_pose(
            link_pose(parent_name, visiting.union((link_name,))), joint_pose
        )
        link_poses[link_name] = pose
        return pose

    proxies: list[CollisionBoxProxy] = []
    for link_name, link in links.items():
        base_to_link = link_pose(link_name)
        for collision_index, collision in enumerate(link.findall("collision")):
            box = collision.find("./geometry/box")
            if box is None:
                continue
            collision_name = collision.get("name") or (
                f"{link_name}_collision_{collision_index}"
            )
            dimensions = _parse_urdf_vector(
                box.get("size"),
                default=(0.0, 0.0, 0.0),
                field=f"collision {collision_name!r} box size",
                strictly_positive=True,
            )
            base_to_collision = _compose_pose(
                base_to_link,
                _urdf_origin_pose(
                    collision.find("origin"),
                    field=f"collision {collision_name!r} origin",
                ),
            )
            proxies.append(
                CollisionBoxProxy(
                    name=collision_name,
                    position=base_to_collision[0],
                    wxyz=base_to_collision[1],
                    dimensions=dimensions,
                )
            )
    return tuple(proxies)


def add_collision_box_proxies(scene, proxies: tuple[CollisionBoxProxy, ...]):
    """Add translucent proxy boxes beneath ``/object`` and keep them hidden."""
    root_handle = scene.add_frame(
        COLLISION_PROXY_ROOT,
        show_axes=False,
        visible=COLLISION_PROXY_INITIAL_VISIBLE,
    )
    box_handles = tuple(
        scene.add_box(
            f"{COLLISION_PROXY_ROOT}/box_{index:02d}",
            color=COLLISION_PROXY_COLOR,
            dimensions=proxy.dimensions,
            opacity=COLLISION_PROXY_OPACITY,
            side="double",
            cast_shadow=False,
            receive_shadow=False,
            position=proxy.position,
            wxyz=proxy.wxyz,
        )
        for index, proxy in enumerate(proxies)
    )
    return root_handle, box_handles


def add_collision_proxy_visibility_control(gui, root_handle):
    """Add the default-off visibility control for the collision proxy subtree."""
    checkbox = gui.add_checkbox(
        "Show collision proxies",
        initial_value=COLLISION_PROXY_INITIAL_VISIBLE,
    )

    @checkbox.on_update
    def _(_event) -> None:
        root_handle.visible = bool(checkbox.value)

    return checkbox


def validation_summary(
    reference: PairReference,
    robot_urdf: str | Path,
    object_urdf: str | Path,
) -> dict[str, object]:
    return {
        "frames": int(reference.robot_qpos.shape[0]),
        "fps": reference.fps,
        "robot_count": ROBOT_COUNT,
        "robot_dof": ROBOT_DOF,
        "robot_qpos_shape": list(reference.robot_qpos.shape),
        "object_qpos_shape": list(reference.object_qpos.shape),
        "object_name": reference.object_name,
        "robot_urdf": str(Path(robot_urdf).expanduser().resolve()),
        "object_urdf": str(Path(object_urdf).expanduser().resolve()),
        "quaternion_convention": "wxyz",
        "pickle_free_input_verified": True,
    }


def validate_robot_urdf_contract(robot_model) -> None:
    """Require the frozen rubber-hand G1 link and joint-order contract."""
    link_names = {link.name for link in robot_model.robot.links}
    missing_hands = sorted(RUBBER_HAND_LINKS.difference(link_names))
    if missing_hands:
        raise ValueError(f"Robot URDF is missing rubber-hand links: {missing_hands}")
    joint_names = tuple(
        joint.name
        for joint in robot_model.robot.joints
        if joint.type not in {"fixed", "floating"}
    )
    if joint_names != EXPECTED_ROBOT_JOINT_NAMES:
        raise ValueError("Robot URDF joint order does not match the frozen G1 29-DoF qpos order")


def run_viewer(
    reference: PairReference,
    *,
    robot_urdf: Path,
    object_urdf: Path,
    host: str,
    port: int,
    loop: bool,
) -> None:
    """Start a Viser player for two robots and one shared object."""
    import viser  # type: ignore[import-not-found]
    import yourdfpy  # type: ignore[import-untyped]
    from viser.extras import ViserUrdf  # type: ignore[import-not-found]

    if not 1 <= port <= 65535:
        raise ValueError(f"port must be in [1, 65535], got {port}")
    robot_urdf_model = yourdfpy.URDF.load(
        str(robot_urdf),
        load_meshes=True,
        build_scene_graph=True,
    )
    validate_robot_urdf_contract(robot_urdf_model)
    object_urdf_model = yourdfpy.URDF.load(
        str(object_urdf),
        load_meshes=True,
        build_scene_graph=True,
    )
    collision_box_proxies = load_collision_box_proxies(object_urdf)

    server = viser.ViserServer(host=host, port=port)
    agent_frames = tuple(
        server.scene.add_frame(f"/agent_{index}", show_axes=False)
        for index in range(ROBOT_COUNT)
    )
    object_frame = server.scene.add_frame("/object", show_axes=False)

    robot_models = tuple(
        ViserUrdf(
            server,
            urdf_or_path=robot_urdf_model,
            root_node_name=f"/agent_{index}",
        )
        for index in range(ROBOT_COUNT)
    )
    object_model = ViserUrdf(
        server,
        urdf_or_path=object_urdf_model,
        root_node_name="/object",
    )
    collision_proxy_root, _collision_proxy_boxes = add_collision_box_proxies(
        server.scene, collision_box_proxies
    )
    if any(len(model.get_actuated_joint_limits()) != ROBOT_DOF for model in robot_models):
        raise ValueError(f"The robot URDF must expose exactly {ROBOT_DOF} actuated joints")

    server.scene.add_grid(
        "/ground",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
        plane_opacity=0.05,
    )
    with server.gui.add_folder("Display"):
        show_meshes = server.gui.add_checkbox("Show meshes", initial_value=True)
        add_collision_proxy_visibility_control(server.gui, collision_proxy_root)
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=reference.robot_qpos.shape[0] - 1,
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play / Pause")
        fps_input = server.gui.add_number(
            "FPS",
            initial_value=reference.fps,
            min=1,
            max=240,
            step=1,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=loop)

    state = {"playing": False, "programmatic_slider": False}

    def apply_frame(frame_index: int) -> None:
        index = int(np.clip(frame_index, 0, reference.robot_qpos.shape[0] - 1))
        with server.atomic():
            for agent_index in range(ROBOT_COUNT):
                qpos = reference.robot_qpos[index, agent_index]
                robot_models[agent_index].update_cfg(qpos[7:])
                agent_frames[agent_index].position = qpos[:3]
                agent_frames[agent_index].wxyz = qpos[3:7]
            object_frame.position = reference.object_qpos[index, :3]
            object_frame.wxyz = reference.object_qpos[index, 3:7]

    @frame_slider.on_update
    def _(_event) -> None:
        if not state["programmatic_slider"]:
            state["playing"] = False
        apply_frame(int(frame_slider.value))

    @play_button.on_click
    def _(_event) -> None:
        state["playing"] = not state["playing"]

    @show_meshes.on_update
    def _(_event) -> None:
        visible = bool(show_meshes.value)
        for model in robot_models:
            model.show_visual = visible
        object_model.show_visual = visible

    apply_frame(0)
    print(
        f"[core4d_pair] Ready: {reference.robot_qpos.shape[0]} frames at "
        f"{reference.fps} FPS, robots=2, object={reference.object_name}, "
        f"collision_boxes={len(collision_box_proxies)}."
    )
    print("Open the Viser URL printed above. Press Ctrl+C to stop.")

    next_frame_time = time.perf_counter()
    while True:
        if not state["playing"]:
            next_frame_time = time.perf_counter()
            time.sleep(0.01)
            continue
        now = time.perf_counter()
        if now < next_frame_time:
            time.sleep(min(0.01, next_frame_time - now))
            continue
        next_frame_time = now + 1.0 / max(1.0, float(fps_input.value))
        current_index = int(frame_slider.value)
        if current_index == reference.robot_qpos.shape[0] - 1 and not loop_checkbox.value:
            state["playing"] = False
            continue
        next_index = (current_index + 1) % reference.robot_qpos.shape[0]
        state["programmatic_slider"] = True
        frame_slider.value = next_index
        state["programmatic_slider"] = False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    reference = load_pair_reference(input_path)
    robot_urdf = args.robot_urdf.expanduser().resolve()
    if not robot_urdf.is_file():
        raise FileNotFoundError(f"Rubber-hand G1 URDF does not exist: {robot_urdf}")
    object_urdf = resolve_object_urdf(input_path, reference.object_name, args.object_urdf)
    summary = validation_summary(reference, robot_urdf, object_urdf)
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    run_viewer(
        reference,
        robot_urdf=robot_urdf,
        object_urdf=object_urdf,
        host=args.host,
        port=args.port,
        loop=args.loop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
