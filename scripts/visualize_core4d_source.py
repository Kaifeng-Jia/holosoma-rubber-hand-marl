#!/usr/bin/env python3
"""Inspect a canonical two-person CORE4D source sequence in Viser."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.data_utils.core4d_adapter import (  # noqa: E402
    SMPLX_BODY_PARENTS,
    WRIST_BODY_INDICES,
    load_canonical_core4d_sequence,
)


PERSON_COLORS = ((66, 135, 245), (245, 145, 66))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical CORE4D NPZ.")
    parser.add_argument("--port", type=int, default=8080, help="Viser TCP port.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data and mesh without starting a Viser server.",
    )
    return parser.parse_args()


def load_object_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"CORE4D object mesh does not exist: {mesh_path}")
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = tuple(geometry for geometry in loaded.geometry.values())
        if not meshes:
            raise ValueError(f"CORE4D object mesh scene is empty: {mesh_path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"Unable to load CORE4D object mesh: {mesh_path}")
    return loaded


def skeleton_segments(joints: np.ndarray) -> np.ndarray:
    body = np.asarray(joints, dtype=np.float32)
    if body.shape != (len(SMPLX_BODY_PARENTS), 3):
        raise ValueError(f"Expected 22 body joints, got {body.shape}")
    children = np.arange(1, len(SMPLX_BODY_PARENTS), dtype=np.int64)
    return np.stack((body[SMPLX_BODY_PARENTS[children]], body[children]), axis=1)


def validation_summary(sequence, mesh: trimesh.Trimesh) -> dict[str, object]:
    return {
        "frames": int(len(sequence.human_joints)),
        "fps": int(sequence.fps),
        "human_joints_shape": list(sequence.human_joints.shape),
        "human_joints_full_shape": list(sequence.human_joints_full.shape),
        "wrist_quat_xyzw_shape": list(sequence.wrist_quat_xyzw.shape),
        "object_poses_shape": list(sequence.object_poses.shape),
        "object_name": sequence.object_name,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extents_m": np.asarray(mesh.extents, dtype=float).tolist(),
        "first_frame_skeleton_segments": int(
            len(skeleton_segments(sequence.human_joints[0, 0]))
        ),
        "pickle_free_input_verified": True,
    }


def run_viewer(sequence, mesh: trimesh.Trimesh, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"port must be in [1, 65535], got {port}")
    server = viser.ViserServer(port=port)
    server.scene.add_grid(
        "/ground",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
        plane_opacity=0.05,
    )
    server.scene.add_frame(
        "/world_axes",
        axes_length=0.3,
        axes_radius=0.008,
    )

    skeleton_handles = []
    body_joint_handles = []
    full_joint_handles = []
    wrist_handles = []
    for person_index, color in enumerate(PERSON_COLORS):
        prefix = f"/person_{person_index + 1}"
        body_joints = sequence.human_joints[0, person_index]
        skeleton_handles.append(
            server.scene.add_line_segments(
                f"{prefix}/skeleton",
                skeleton_segments(body_joints),
                colors=color,
                line_width=5.0,
            )
        )
        body_joint_handles.append(
            server.scene.add_point_cloud(
                f"{prefix}/body_joints",
                body_joints.astype(np.float32),
                colors=color,
                point_size=0.035,
                point_shape="circle",
                precision="float32",
            )
        )
        full_joint_handles.append(
            server.scene.add_point_cloud(
                f"{prefix}/full_joints",
                sequence.human_joints_full[0, person_index].astype(np.float32),
                colors=color,
                point_size=0.012,
                point_shape="circle",
                precision="float32",
                visible=False,
            )
        )
        person_wrists = []
        for hand_index, hand_name in enumerate(("left_wrist", "right_wrist")):
            quat_xyzw = sequence.wrist_quat_xyzw[0, person_index, hand_index]
            person_wrists.append(
                server.scene.add_frame(
                    f"{prefix}/{hand_name}",
                    axes_length=0.12,
                    axes_radius=0.004,
                    position=body_joints[WRIST_BODY_INDICES[hand_index]],
                    wxyz=quat_xyzw[[3, 0, 1, 2]],
                )
            )
        wrist_handles.append(person_wrists)

    object_pose = sequence.object_poses[0]
    object_handle = server.scene.add_mesh_simple(
        "/object/desk001",
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=(180, 190, 205),
        opacity=0.82,
        side="double",
        position=object_pose[4:7],
        wxyz=object_pose[:4],
    )

    with server.gui.add_folder("Display"):
        show_skeletons = server.gui.add_checkbox("Show 22-joint skeletons", initial_value=True)
        show_full_joints = server.gui.add_checkbox("Show all 127 joints", initial_value=False)
        show_wrist_axes = server.gui.add_checkbox("Show wrist axes", initial_value=True)
        show_object = server.gui.add_checkbox("Show object mesh", initial_value=True)
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(sequence.human_joints) - 1,
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play / Pause")
        fps_input = server.gui.add_number(
            "FPS",
            initial_value=int(sequence.fps),
            min=1,
            max=120,
            step=1,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

    state = {"playing": False, "programmatic_slider": False}

    def apply_frame(frame_index: int) -> None:
        index = int(np.clip(frame_index, 0, len(sequence.human_joints) - 1))
        with server.atomic():
            for person_index in range(2):
                body_joints = sequence.human_joints[index, person_index]
                skeleton_handles[person_index].points = skeleton_segments(body_joints)
                body_joint_handles[person_index].points = body_joints.astype(np.float32)
                full_joint_handles[person_index].points = sequence.human_joints_full[
                    index, person_index
                ].astype(np.float32)
                for hand_index in range(2):
                    wrist_handles[person_index][hand_index].position = body_joints[
                        WRIST_BODY_INDICES[hand_index]
                    ]
                    quat_xyzw = sequence.wrist_quat_xyzw[index, person_index, hand_index]
                    wrist_handles[person_index][hand_index].wxyz = quat_xyzw[[3, 0, 1, 2]]
            pose = sequence.object_poses[index]
            object_handle.wxyz = pose[:4]
            object_handle.position = pose[4:7]

    @frame_slider.on_update
    def _(_event) -> None:
        if not state["programmatic_slider"]:
            state["playing"] = False
        apply_frame(int(frame_slider.value))

    @play_button.on_click
    def _(_event) -> None:
        state["playing"] = not state["playing"]

    @show_skeletons.on_update
    def _(_event) -> None:
        for skeleton, joints in zip(skeleton_handles, body_joint_handles, strict=True):
            skeleton.visible = bool(show_skeletons.value)
            joints.visible = bool(show_skeletons.value)

    @show_full_joints.on_update
    def _(_event) -> None:
        for handle in full_joint_handles:
            handle.visible = bool(show_full_joints.value)

    @show_wrist_axes.on_update
    def _(_event) -> None:
        for person_handles in wrist_handles:
            for handle in person_handles:
                handle.visible = bool(show_wrist_axes.value)

    @show_object.on_update
    def _(_event) -> None:
        object_handle.visible = bool(show_object.value)

    apply_frame(0)
    print(
        f"[core4d_source] Ready: {len(sequence.human_joints)} frames at "
        f"{sequence.fps} FPS, object={sequence.object_name}."
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
        if current_index == len(sequence.human_joints) - 1 and not loop_checkbox.value:
            state["playing"] = False
            continue
        next_index = (current_index + 1) % len(sequence.human_joints)
        state["programmatic_slider"] = True
        frame_slider.value = next_index
        state["programmatic_slider"] = False


def main() -> None:
    args = parse_args()
    sequence = load_canonical_core4d_sequence(args.input)
    mesh = load_object_mesh(sequence.object_mesh_path)
    summary = validation_summary(sequence, mesh)
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(json.dumps(summary, indent=2, sort_keys=True))
    run_viewer(sequence, mesh, args.port)


if __name__ == "__main__":
    main()
