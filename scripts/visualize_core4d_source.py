#!/usr/bin/env python3
"""Inspect a raw or canonical two-person CORE4D source sequence in Viser."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import viser


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_retargeting"))

from holosoma_retargeting.data_utils.core4d_adapter import (  # noqa: E402
    CORE4D_FPS,
    CORE4D_TO_OMNI_ROTATION,
    SMPLX_BODY_PARENTS,
    WRIST_BODY_INDICES,
    _convert_person,
    _load_official_person_npz,
    _resolve_object_mesh,
    load_canonical_core4d_sequence,
)


PERSON_COLORS = ((66, 135, 245), (245, 145, 66))


@dataclass(frozen=True)
class RawCore4DPreview:
    """Display-only data; deliberately not a canonical/exportable motion asset."""

    human_joints: np.ndarray
    human_joints_full: np.ndarray
    wrist_quat_xyzw: np.ndarray
    raw_object_transforms: np.ndarray
    object_name: str
    object_mesh_path: str
    sequence_dir: str
    fps: int = CORE4D_FPS


def load_raw_core4d_sequence(
    sequence_dir: str | Path,
    object_model_root: str | Path | None = None,
) -> RawCore4DPreview:
    """Read trusted official data without making its object matrices rigid.

    Reuse the existing trusted-pickle boundary and human coordinate conversion.
    Raw object matrices are retained verbatim for vertex-based display only.
    No cache, normalization, resampling, or training artifact is written.
    """
    source = Path(sequence_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"CORE4D sequence directory does not exist: {source}")
    if object_model_root is None:
        if source.parent.parent.name != "human_object_motions":
            raise ValueError("Cannot infer object models; provide --object-model-root")
        model_root = source.parent.parent.parent / "object_models"
    else:
        model_root = Path(object_model_root).expanduser().resolve()

    transforms = np.load(source / "smooth_objposes.npy", allow_pickle=False)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or len(transforms) < 1:
        raise ValueError("smooth_objposes.npy must have shape [T, 4, 4], with T > 0")
    if not np.isfinite(transforms).all():
        raise ValueError("smooth_objposes.npy contains non-finite values")
    if not np.allclose(transforms[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7):
        raise ValueError("CORE4D object transforms must be homogeneous matrices")

    metadata = json.loads((source / "object_metadata.json").read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("obj_name"), str):
        raise ValueError("object_metadata.json must contain a string obj_name")
    mesh_path = _resolve_object_mesh(model_root, metadata["obj_name"])
    people = []
    for person_index in (1, 2):
        person = _load_official_person_npz(source / f"person{person_index}_poses.npz")
        if len(person["joints"]) != len(transforms):
            raise ValueError("CORE4D person1, person2, and object motions must have identical frame counts")
        people.append(_convert_person(person))
        del person  # Do not retain the large raw surface vertices while loading the other person.
    return RawCore4DPreview(
        human_joints=np.stack([person[0] for person in people], axis=1),
        human_joints_full=np.stack([person[1] for person in people], axis=1),
        wrist_quat_xyzw=np.stack([person[2] for person in people], axis=1),
        raw_object_transforms=transforms,
        object_name=metadata["obj_name"],
        object_mesh_path=str(mesh_path),
        sequence_dir=str(source),
    )


def raw_rotation_diagnostics(transforms: np.ndarray) -> dict[str, object]:
    """Report, but do not repair/reject, non-rigid matrices in raw previews."""
    rotation = np.asarray(transforms[:, :3, :3], dtype=np.float64)
    gram = np.swapaxes(rotation, 1, 2) @ rotation
    identity = np.eye(3)
    determinant = np.linalg.det(rotation)
    non_rigid = ~np.isclose(gram, identity, atol=1.0e-5).all(axis=(1, 2))
    non_rigid |= ~np.isclose(determinant, 1.0, atol=1.0e-5)
    return {
        "non_rigid_frame_indices": np.flatnonzero(non_rigid).tolist(),
        "max_rotation_orthogonality_error": float(np.max(np.abs(gram - identity))),
        "max_rotation_determinant_error": float(np.max(np.abs(determinant - 1.0))),
    }


def object_vertices_world(
    sequence: RawCore4DPreview, mesh: trimesh.Trimesh, frame_index: int,
) -> np.ndarray:
    """Apply the original affine matrix, then the same Y-up to Z-up basis as humans."""
    transform = sequence.raw_object_transforms[frame_index]
    raw_world = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
    return raw_world @ CORE4D_TO_OMNI_ROTATION.T


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Existing canonical CORE4D NPZ.")
    source.add_argument(
        "--sequence-dir", type=Path,
        help="Raw trusted official human_object_motions/date/id directory; display only.",
    )
    parser.add_argument(
        "--object-model-root", type=Path,
        help="Raw object_models directory; inferred from the standard dataset layout if omitted.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Viser bind address (default: local machine only).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser TCP port.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data and mesh without starting a Viser server.",
    )
    args = parser.parse_args(argv)
    if args.object_model_root is not None and args.sequence_dir is None:
        parser.error("--object-model-root requires --sequence-dir")
    return args


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
    raw_mode = isinstance(sequence, RawCore4DPreview)
    summary = {
        "input_mode": "raw_preview" if raw_mode else "canonical",
        "frames": int(len(sequence.human_joints)),
        "fps": int(sequence.fps),
        "human_joints_shape": list(sequence.human_joints.shape),
        "human_joints_full_shape": list(sequence.human_joints_full.shape),
        "wrist_quat_xyzw_shape": list(sequence.wrist_quat_xyzw.shape),
        "object_name": sequence.object_name,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extents_m": np.asarray(mesh.extents, dtype=float).tolist(),
        "first_frame_skeleton_segments": int(
            len(skeleton_segments(sequence.human_joints[0, 0]))
        ),
        "pickle_free_input_verified": not raw_mode,
    }
    if raw_mode:
        summary.update(
            sequence_dir=sequence.sequence_dir,
            raw_object_transforms_shape=list(sequence.raw_object_transforms.shape),
            object_transform_mode="original_matrix_applied_to_vertices_without_repair",
            preview_only=True,
            training_asset_exported=False,
            **raw_rotation_diagnostics(sequence.raw_object_transforms),
        )
    else:
        summary["object_poses_shape"] = list(sequence.object_poses.shape)
    return summary


def run_viewer(sequence, mesh: trimesh.Trimesh, host: str, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"port must be in [1, 65535], got {port}")
    server = viser.ViserServer(host=host, port=port)
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

    raw_mode = isinstance(sequence, RawCore4DPreview)
    object_pose = (
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if raw_mode else sequence.object_poses[0]
    )
    object_vertices = object_vertices_world(sequence, mesh, 0) if raw_mode else mesh.vertices
    object_handle = server.scene.add_mesh_simple(
        "/object/source",
        vertices=np.asarray(object_vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        color=(180, 190, 205),
        opacity=0.82,
        side="double",
        position=object_pose[4:7],
        wxyz=object_pose[:4],
    )

    if raw_mode:
        diagnostics = raw_rotation_diagnostics(sequence.raw_object_transforms)
        message = "Raw source preview: original object matrices, no repair or training export."
        if diagnostics["non_rigid_frame_indices"]:
            message += (
                f" Non-rigid matrix frames (zero-based): {diagnostics['non_rigid_frame_indices']}."
                " Original scale/shear is displayed as recorded."
            )
        server.gui.add_markdown(message)

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
            if raw_mode:
                # The handle stays at identity; do not apply the object pose twice.
                object_handle.vertices = object_vertices_world(sequence, mesh, index).astype(np.float32)
            else:
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
    sequence = (
        load_raw_core4d_sequence(args.sequence_dir, args.object_model_root)
        if args.sequence_dir is not None
        else load_canonical_core4d_sequence(args.input)
    )
    mesh = load_object_mesh(sequence.object_mesh_path)
    summary = validation_summary(sequence, mesh)
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(json.dumps(summary, indent=2, sort_keys=True))
    run_viewer(sequence, mesh, args.host, args.port)


if __name__ == "__main__":
    main()
