#!/usr/bin/env python3
"""Read-only Viser playback of a bounded CORE4D offline interaction audit.

Consumes audit_arrays.npz; never evaluates a policy, runs physics, rebuilds a
graph, or writes an artifact. All archive root/object quaternions are WXYZ.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package in ("holosoma", "holosoma_retargeting"):
    sys.path.insert(0, str(REPO_ROOT / "src" / _package))

DEFAULT_ROBOT_URDF = (
    REPO_ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"
)
DEFAULT_OBJECT_URDF = (
    REPO_ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
    / "objects_core4d_desk001_small_training.urdf"
)
BODY_COUNT = 19
ORIGINAL_BODY_COUNT = 15
BUDGETS = (64, 100)
ORIGINAL_COLOR = np.asarray((50, 180, 255), dtype=np.uint8)
ADDED_COLOR = np.asarray((255, 165, 40), dtype=np.uint8)
OBJECT_COLOR = np.asarray((70, 225, 150), dtype=np.uint8)
EDGE_COLORS = ((65, 150, 245), (215, 100, 235))


@dataclass(frozen=True)
class AuditData:
    arrays: dict[str, np.ndarray]
    frame_count: int
    fps: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--object-urdf", type=Path, default=DEFAULT_OBJECT_URDF)
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate input and asset contracts without a server."
    )
    return parser.parse_args(argv)


def load_audit(audit_dir: Path) -> AuditData:
    """Load the pickle-free archive and reject shape/convention ambiguity."""
    required = {"node_names", "fps"}
    for mode in ("reference", "actual"):
        required.update(f"{mode}_{suffix}" for suffix in ("robot_qpos", "object_qpos", "landmarks_w"))
    for budget in BUDGETS:
        required.update(f"{prefix}_{budget}" for prefix in ("object_points", "adjacency", "residual"))
    path = audit_dir / "audit_arrays.npz"
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Audit archive missing required arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    reference = arrays["reference_robot_qpos"]
    if reference.ndim != 3 or reference.shape[1:] != (2, 36) or not len(reference):
        raise ValueError(f"reference_robot_qpos must have nonempty shape [T,2,36], got {reference.shape}")
    frame_count = len(reference)
    expected_shapes = {"node_names": (BODY_COUNT,), "fps": ()}
    for mode in ("reference", "actual"):
        expected_shapes.update({
            f"{mode}_robot_qpos": (frame_count, 2, 36),
            f"{mode}_object_qpos": (frame_count, 7),
            f"{mode}_landmarks_w": (frame_count, 2, BODY_COUNT, 3),
        })
    for budget in BUDGETS:
        object_points = arrays[f"object_points_{budget}"]
        if object_points.ndim != 2 or object_points.shape[1] != 3 or not len(object_points):
            raise ValueError(f"object_points_{budget} must have nonempty shape [N,3], got {object_points.shape}")
        actual_count = len(object_points)
        node_count = BODY_COUNT + actual_count
        expected_shapes.update({
            f"object_points_{budget}": (actual_count, 3),
            f"adjacency_{budget}": (frame_count, 2, node_count, node_count),
            f"residual_{budget}": (frame_count, 2, node_count),
        })
    for name, expected in expected_shapes.items():
        value = arrays[name]
        if value.shape != expected:
            raise ValueError(f"{name}: expected shape {expected}, got {value.shape}")
        if name != "node_names" and (not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_):
            raise ValueError(f"{name} must be a numeric array")
        if name != "node_names" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    names = arrays["node_names"]
    if names.dtype.kind not in {"U", "S"} or len(set(names.tolist())) != BODY_COUNT:
        raise ValueError("node_names must contain 19 unique pickle-free strings")
    fps = float(arrays["fps"])
    if fps <= 0:
        raise ValueError("fps must be positive")
    for mode in ("reference", "actual"):
        for suffix in ("robot_qpos", "object_qpos"):
            key = f"{mode}_{suffix}"
            norms = np.linalg.norm(arrays[key][..., 3:7], axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-4, rtol=0):
                raise ValueError(f"{key} must contain unit WXYZ quaternions")
    for budget in BUDGETS:
        graph = arrays[f"adjacency_{budget}"]
        if graph.dtype.kind not in {"b", "u", "i"} or not np.isin(graph, (0, 1)).all():
            raise ValueError(f"adjacency_{budget} must be a boolean or 0/1 integer graph")
        if not np.array_equal(graph, graph.swapaxes(-1, -2)):
            raise ValueError(f"adjacency_{budget} must be symmetric")
        if np.diagonal(graph, axis1=-2, axis2=-1).any():
            raise ValueError(f"adjacency_{budget} must have a zero diagonal")
        if (arrays[f"residual_{budget}"] < 0).any():
            raise ValueError(f"residual_{budget} must be nonnegative norms in metres")
    return AuditData(arrays, frame_count, fps)


def validate_assets(robot_urdf: Path, object_urdf: Path) -> tuple[str, ...]:
    """Resolve joint order only from the checkout containing this script."""
    expected_module = REPO_ROOT / "src/holosoma_retargeting/holosoma_retargeting/dual_pull_reference.py"
    if not expected_module.is_file():
        raise FileNotFoundError(f"Checkout-local joint-name module not found: {expected_module}")
    from holosoma_retargeting import dual_pull_reference

    if Path(dual_pull_reference.__file__).resolve() != expected_module.resolve():
        raise RuntimeError("Refusing an editable holosoma_retargeting import from another checkout")
    names = tuple(str(name) for name in dual_pull_reference.G1_29DOF_JOINT_NAMES)
    robot_root = ET.parse(robot_urdf).getroot()
    links = {link.attrib["name"] for link in robot_root.findall("link")}
    if not {"left_rubber_hand_link", "right_rubber_hand_link"}.issubset(links):
        raise ValueError("Robot URDF must contain the accepted left/right rubber-hand links")
    joints = tuple(joint.attrib["name"] for joint in robot_root.findall("joint") if joint.attrib["type"] != "fixed")
    if len(joints) != 29 or set(joints) != set(names):
        raise ValueError("Robot URDF does not match the G1 29-DoF named joint contract")
    object_root = ET.parse(object_urdf).getroot()
    if any(joint.attrib["type"] != "fixed" for joint in object_root.findall("joint")):
        raise ValueError("Audit object URDF must be rigid")
    return names


def object_points_world(points_local: np.ndarray, object_qpos: np.ndarray) -> np.ndarray:
    """Apply the selected recorded pose; input WXYZ is explicitly reordered."""
    rotation = Rotation.from_quat(object_qpos[[4, 5, 6, 3]])
    return rotation.apply(points_local) + object_qpos[:3]


def edge_segments(nodes_world: np.ndarray, adjacency: np.ndarray, cross_only: bool) -> np.ndarray:
    """Draw existing undirected reference edges, once each, on current nodes."""
    rows, cols = np.nonzero(np.triu(adjacency, k=1))
    if cross_only:
        selected = (rows < BODY_COUNT) & (cols >= BODY_COUNT)
        rows, cols = rows[selected], cols[selected]
    return np.stack((nodes_world[rows], nodes_world[cols]), axis=1).astype(np.float32)


def residual_colors(values: np.ndarray, limit_m: float) -> np.ndarray:
    """Fixed linear blue-to-red scale, with saturation at the GUI limit."""
    fraction = np.clip(np.asarray(values) / max(limit_m, 1e-8), 0.0, 1.0)[..., None]
    low = np.asarray((50, 150, 255), dtype=np.float64)
    high = np.asarray((255, 55, 40), dtype=np.float64)
    return np.rint(low + fraction * (high - low)).astype(np.uint8)


def run_viewer(data: AuditData, args: argparse.Namespace, joint_names: tuple[str, ...]) -> None:
    import viser
    from viser.extras import ViserUrdf

    arrays = data.arrays
    server = viser.ViserServer(host=args.host, port=args.port)
    agent_frames = [server.scene.add_frame(f"/agent_{i + 1}", show_axes=False) for i in range(2)]
    object_frame = server.scene.add_frame("/object", show_axes=False)
    models = [ViserUrdf(server, args.robot_urdf, root_node_name=f"/agent_{i + 1}") for i in range(2)]
    table = ViserUrdf(server, args.object_urdf, root_node_name="/object")
    name_to_column = {name: index for index, name in enumerate(joint_names)}
    joint_columns = []
    for model in models:
        viewer_names = tuple(model.get_actuated_joint_names())
        if len(viewer_names) != 29 or set(viewer_names) != set(joint_names):
            raise ValueError("Viser robot actuated joints differ from the recorded G1 joint names")
        joint_columns.append(np.asarray([name_to_column[name] for name in viewer_names]))
    server.scene.add_grid("/ground", width=6.0, height=6.0, plane="xy", plane_opacity=0.05)

    body_colors = np.tile(ORIGINAL_COLOR, (BODY_COUNT, 1))
    body_colors[ORIGINAL_BODY_COUNT:] = ADDED_COLOR
    body_handles = [server.scene.add_point_cloud(
        f"/audit/agent_{i + 1}/body", points=arrays["reference_landmarks_w"][0, i].astype(np.float32),
        colors=body_colors, point_size=0.024, point_shape="circle", precision="float32",
    ) for i in range(2)]
    # This cloud stays in the table's local frame. Mode changes move mesh and
    # points together; only world-space graph endpoints require conversion.
    object_handle = server.scene.add_point_cloud(
        "/object/audit_points", points=arrays["object_points_64"].astype(np.float32),
        colors=np.tile(OBJECT_COLOR, (len(arrays["object_points_64"]), 1)), point_size=0.012,
        point_shape="circle", precision="float32",
    )
    edge_handles = [server.scene.add_line_segments(
        f"/audit/agent_{i + 1}/edges", points=np.zeros((0, 2, 3), dtype=np.float32),
        colors=EDGE_COLORS[i], line_width=1.5,
    ) for i in range(2)]

    server.gui.add_markdown(
        "离线 interaction audit：边仅表示几何邻接，不代表物理接触或接触力。\n\n"
        "Actual 由已保存关节状态经参考/可视化模型 FK 重建，不是直接记录的 PhysX 身体点位；不运行新的物理仿真。\n\n"
        "Reference 与 Actual 使用相同帧的冻结参考图；不会为 Actual 重建邻接。"
    )
    with server.gui.add_folder("Display"):
        mode_control = server.gui.add_dropdown("Mode", options=("Reference", "Actual"), initial_value="Reference")
        budget_control = server.gui.add_dropdown("Requested point budget", options=("64", "100"), initial_value="64")
        agent_control = server.gui.add_dropdown("Agents", options=("Both", "1", "2"), initial_value="Both")
        show_models = server.gui.add_checkbox("Meshes", initial_value=True)
        show_points = server.gui.add_checkbox("Points", initial_value=True)
        show_edges = server.gui.add_checkbox("Edges", initial_value=True)
        cross_only = server.gui.add_checkbox("Only body-object edges", initial_value=True)
        color_residual = server.gui.add_checkbox("Actual residual colors", initial_value=False)
        color_limit = server.gui.add_number("Residual red at (m)", initial_value=0.05, min=0.001, max=5.0, step=0.005)
    with server.gui.add_folder("Playback"):
        frame_control = server.gui.add_slider(
            "Frame", min=0, max=max(1, data.frame_count - 1), step=1, initial_value=0,
            disabled=data.frame_count == 1,
        )
        play_control = server.gui.add_button("Play / Pause")
        fps_control = server.gui.add_number("Playback FPS", initial_value=data.fps, min=1, max=max(240, data.fps), step=1)
        loop_control = server.gui.add_checkbox("Loop", initial_value=False)
        status = server.gui.add_markdown("")
    legend = server.gui.add_markdown("")
    original_names = ", ".join(str(name) for name in arrays["node_names"][:ORIGINAL_BODY_COUNT])
    added_names = ", ".join(str(name) for name in arrays["node_names"][ORIGINAL_BODY_COUNT:])
    with server.gui.add_folder("Landmark names"):
        server.gui.add_markdown(f"Original 15: {original_names}\n\nAdded 4: {added_names}")

    lock = threading.RLock()
    state = {"playing": False, "frame": 0, "anchor_frame": 0, "anchor_time": time.perf_counter()}

    def reset_clock() -> None:
        state["anchor_frame"] = state["frame"]
        state["anchor_time"] = time.perf_counter()

    def draw_frame() -> None:
        index = int(state["frame"])
        mode = str(mode_control.value).lower()
        budget = int(budget_control.value)
        selected = [0, 1] if agent_control.value == "Both" else [int(agent_control.value) - 1]
        robot_qpos = arrays[f"{mode}_robot_qpos"][index]
        object_qpos = arrays[f"{mode}_object_qpos"][index]
        landmarks = arrays[f"{mode}_landmarks_w"][index]
        object_local = arrays[f"object_points_{budget}"]
        object_world = object_points_world(object_local, object_qpos)
        graph = arrays[f"adjacency_{budget}"][index]
        residual = arrays[f"residual_{budget}"][index]
        use_residual = mode == "actual" and bool(color_residual.value)
        edge_count = 0
        with server.atomic():
            for i, model in enumerate(models):
                model.update_cfg(robot_qpos[i, 7:][joint_columns[i]])
                agent_frames[i].position = robot_qpos[i, :3]
                agent_frames[i].wxyz = robot_qpos[i, 3:7]
                agent_frames[i].visible = i in selected
                model.show_visual = bool(show_models.value)
                body_handles[i].points = landmarks[i].astype(np.float32)
                body_handles[i].colors = residual_colors(residual[i, :BODY_COUNT], float(color_limit.value)) if use_residual else body_colors
                body_handles[i].visible = bool(show_points.value) and i in selected
                segments = edge_segments(np.concatenate((landmarks[i], object_world)), graph[i], bool(cross_only.value))
                edge_handles[i].points = segments
                edge_handles[i].colors = np.tile(np.asarray(EDGE_COLORS[i], dtype=np.uint8), (len(segments), 2, 1))
                edge_handles[i].visible = bool(show_edges.value) and i in selected
                if i in selected:
                    edge_count += len(segments)
            object_frame.position = object_qpos[:3]
            object_frame.wxyz = object_qpos[3:7]
            table.show_visual = bool(show_models.value)
            object_handle.points = object_local.astype(np.float32)
            object_handle.colors = residual_colors(
                np.max(residual[selected, BODY_COUNT:], axis=0), float(color_limit.value)
            ) if use_residual else np.tile(OBJECT_COLOR, (len(object_local), 1))
            object_handle.visible = bool(show_points.value)
            status.content = (
                f"{'Playing' if state['playing'] else 'Paused'} · frame {index}/{data.frame_count - 1} · "
                f"source time {index / data.fps:.3f} s @ {data.fps:g} FPS\n\n"
                f"{mode_control.value} · requested {budget}, actual {len(object_local)} object points · "
                f"{edge_count} selected geometric edges"
            )
            legend.content = (
                f"Residual norm (m): blue = 0; red ≥ {float(color_limit.value):g}. "
                "共享物体点显示已选 agent 的最大节点残差；不是接触力。"
                if use_residual else
                "点颜色：蓝色 = 原始 15 身体点；橙色 = 新增 4 身体点；绿色 = 物体表面采样。\n\n"
                "边颜色：蓝色 = agent 1；紫色 = agent 2。残差着色仅在 Actual 模式生效。"
            )

    @frame_control.on_update
    def on_frame(event) -> None:
        # Python assignments have no client id and are rendered by the loop.
        if event.client_id is None:
            return
        with lock:
            state["playing"] = False
            state["frame"] = min(int(frame_control.value), data.frame_count - 1)
            reset_clock()
            draw_frame()

    @play_control.on_click
    def on_play(_event) -> None:
        with lock:
            state["playing"] = not state["playing"]
            if state["playing"] and state["frame"] == data.frame_count - 1:
                state["frame"] = 0
                frame_control.value = 0
            reset_clock()
            draw_frame()

    @fps_control.on_update
    def on_fps(_event) -> None:
        with lock:
            reset_clock()

    def on_display(_event) -> None:
        with lock:
            draw_frame()

    for control in (mode_control, budget_control, agent_control, show_models, show_points, show_edges, cross_only, color_residual, color_limit):
        control.on_update(on_display)

    @server.on_client_connect
    def on_connect(client) -> None:
        center = arrays["reference_object_qpos"][0, :3]
        client.camera.look_at = center
        client.camera.position = center + np.asarray((2.8, -3.2, 2.0))

    draw_frame()
    counts = {budget: len(arrays[f"object_points_{budget}"]) for budget in BUDGETS}
    print(f"Read-only audit: {data.frame_count} aligned frames at {data.fps:g} FPS; default Reference, budget 64.")
    print(f"Requested/actual object-point counts: {counts}; samples are not padded.")
    print(f"Robot: {args.robot_urdf}\nTable: {args.object_urdf}")
    print("Frozen reference adjacency; Actual is recorded-state playback only. Ctrl+C stops this viewer.")
    try:
        while True:
            with lock:
                if state["playing"]:
                    elapsed = time.perf_counter() - float(state["anchor_time"])
                    target = int(state["anchor_frame"]) + int(elapsed * float(fps_control.value))
                    if target >= data.frame_count:
                        if loop_control.value:
                            target %= data.frame_count
                        else:
                            target = data.frame_count - 1
                            state["playing"] = False
                    if target != state["frame"] or not state["playing"]:
                        state["frame"] = target
                        frame_control.value = target
                        draw_frame()
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    args.audit_dir = args.audit_dir.expanduser().resolve()
    args.robot_urdf = args.robot_urdf.expanduser().resolve()
    args.object_urdf = args.object_urdf.expanduser().resolve()
    data = load_audit(args.audit_dir)
    names = validate_assets(args.robot_urdf, args.object_urdf)
    if args.validate_only:
        print(json.dumps({
            "audit_dir": str(args.audit_dir), "frames": data.frame_count, "fps": data.fps,
            "budgets": list(BUDGETS), "body_nodes": BODY_COUNT,
            "actual_object_point_counts": {budget: len(data.arrays[f"object_points_{budget}"]) for budget in BUDGETS},
            "robot_urdf": str(args.robot_urdf), "object_urdf": str(args.object_urdf),
            "quaternion_convention": "wxyz", "graph": "frozen_reference", "read_only": True,
        }, indent=2))
        return 0
    run_viewer(data, args, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
