#!/usr/bin/env python3
"""Render saved paired physical rollouts, without policy inference or physics steps.

Uses the visual URDF meshes and yourdfpy FK, just as the Viser playback does.
MuJoCo is ONLY an offscreen rasterizer here: every link is a mocap body and
mj_step is never called. Original states and real-time sampling are preserved.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import yourdfpy

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(values) -> str:
    return " ".join(f"{float(x):.9g}" for x in values)


def pose(position, xyzw):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    out[:3, 3] = position
    return out


def configured_joint_names():
    """Read the G1 declaration without importing training/simulator packages."""
    tree = ast.parse((ROOT / "src/holosoma/holosoma/config_values/robot.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "g1_29dof" for t in node.targets
        ):
            for kw in node.value.keywords:
                if kw.arg == "dof_names":
                    return ast.literal_eval(kw.value)
    raise RuntimeError("Cannot verify G1 recording joint order from robot.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--object-urdf", type=Path, required=True)
    p.add_argument("--robot-urdf", type=Path, default=ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--checkpoint-label", required=True)
    p.add_argument("--ffmpeg", type=Path)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--video-fps", type=float, default=25.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=135.0)
    p.add_argument("--elevation", type=float, default=-22.0)
    p.add_argument("--camera-mode", choices=("fixed", "follow-pair"), default="follow-pair")
    p.add_argument("--frames", help="Comma-separated original recording indices for stills")
    p.add_argument("--stills-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory; completed exports are not overwritten")
    with np.load(args.rollout, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays.get("_metadata_json", "{}")))
    fps = float(metadata.get("fps", args.fps))
    count = len(arrays["dof_pos"])
    expected = {"dof_pos": (count, 2, 29), "root_pos": (count, 2, 3),
                "root_quat_xyzw": (count, 2, 4), "object_pos_w": (count, 3),
                "object_quat_xyzw": (count, 4)}
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key}: expected finite {shape}")
    for key in ("root_quat_xyzw", "object_quat_xyzw"):
        if not np.allclose(np.linalg.norm(arrays[key], axis=-1), 1, atol=1e-3):
            raise ValueError(f"Non-unit rotation in {key}")
    if "episode_step" in arrays and not np.all(np.diff(arrays["episode_step"]) == 1):
        raise ValueError("Nonconsecutive episode steps: explicit timestamp handling required")
    urdfs = [yourdfpy.URDF.load(str(path.resolve()), load_collision_meshes=False, build_collision_scene_graph=False)
             for path in (args.robot_urdf, args.robot_urdf, args.object_urdf)]
    if any(not u.scene.geometry for u in urdfs):
        raise ValueError("URDF visual meshes failed to load; do not export an empty scene")
    names = configured_joint_names()
    if any(u.actuated_joint_names != names for u in urdfs[:2]):
        raise ValueError("URDF and saved simulator configuration DOF order disagree")
    # Verify the actual visual assets against the experiment's saved metadata.
    contract = metadata.get("experiment_contract", {})
    if contract.get("object_urdf_sha256") not in (None, sha(args.object_urdf)):
        raise ValueError("Object URDF does not match this rollout's training contract")
    robot_expected = metadata.get("bucket_training_contract", {}).get("training_robot_urdf_sha256")
    if robot_expected not in (None, sha(args.robot_urdf)):
        raise ValueError("Robot URDF differs from recorded training asset")

    xml = ET.Element("mujoco", model="recorded_rollout_visual_only")
    ET.SubElement(xml, "compiler", angle="radian")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth=str(args.width), offheight=str(args.height))
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "headlight", diffuse="0.5 0.5 0.5", ambient="0.3 0.3 0.3", specular="0.05 0.05 0.05")
    ET.SubElement(visual, "rgba", haze="0.95 0.96 0.98 1")
    asset = ET.SubElement(xml, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.97 0.98 1", rgb2="0.97 0.98 1", width="128", height="128")
    ET.SubElement(asset, "texture", name="ground", type="2d", builtin="checker", rgb1="0.90 0.92 0.94", rgb2="0.96 0.97 0.98", width="256", height="256")
    ET.SubElement(asset, "material", name="ground", texture="ground", texrepeat="80 80", reflectance="0", specular="0", shininess="0")
    world = ET.SubElement(xml, "worldbody")
    points = np.concatenate((arrays["root_pos"].reshape(-1, 3), arrays["object_pos_w"]), axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    center[2] = 0.75
    ET.SubElement(world, "light", pos=numbers(center + [0, 0, 5]), dir="0 0 -1", diffuse="0.5 0.5 0.5", specular="0 0 0", castshadow="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 .01", material="ground", contype="0", conaffinity="0")
    assets, mapping, mesh_keys = {}, [], {}
    for agent, urdf in enumerate(urdfs):
        for node in urdf.scene.graph.nodes_geometry:
            _, geom_name = urdf.scene.graph.get(node)
            mesh = urdf.scene.geometry[geom_name]
            # Deduplicate repeated robot meshes, preserving the original triangles.
            identity = ("robot" if agent < 2 else "object", geom_name)
            if identity not in mesh_keys:
                mesh_id = f"mesh_{len(mesh_keys)}"
                mesh_keys[identity] = mesh_id
                filename = mesh_id + ".stl"
                assets[filename] = mesh.export(file_type="stl")
                ET.SubElement(asset, "mesh", name=mesh_id, file=filename)
            mesh_id = mesh_keys[identity]
            bodyname = f"visual_{len(mapping)}"
            body = ET.SubElement(world, "body", name=bodyname, mocap="true")
            color = np.asarray(mesh.visual.main_color, dtype=float) / 255
            ET.SubElement(body, "geom", type="mesh", mesh=mesh_id, rgba=numbers(color),
                          contype="0", conaffinity="0", group="1", mass="0")
            mapping.append((agent, node, bodyname))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"), assets)
    data = mujoco.MjData(model)
    body_ids = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
                for _, _, name in mapping]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.azimuth, camera.elevation = args.azimuth, args.elevation
    span = np.ptp(points[:, :2], axis=0)
    camera.distance = max(3.4, float(np.linalg.norm(span)) * 1.3 + 1.8)
    if args.camera_mode == "follow-pair":
        frame_points = np.concatenate((arrays["root_pos"], arrays["object_pos_w"][:, None]), axis=1)
        frame_span = np.ptp(frame_points[:, :, :2], axis=1)
        camera.distance = max(2.8, float(np.linalg.norm(frame_span, axis=1).max()) * 0.65 + 1.3)
    stills = ([int(x) for x in args.frames.split(",")] if args.frames
              else np.linspace(0, count - 1, 6).round().astype(int).tolist())
    if any(i < 0 or i >= count for i in stills):
        raise ValueError("Requested still frame outside recording")
    video_indices = np.minimum(np.rint(np.arange(0, count / fps, 1 / args.video_fps) * fps).astype(int), count - 1)
    if args.stills_only:
        video_indices = np.array([], dtype=int)
    elif args.ffmpeg is None:
        raise ValueError("--ffmpeg is required for MP4; alternatively use --stills-only")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20) if Path(font_path).exists() else ImageFont.load_default()
    writer = None
    images = {}
    if not args.stills_only:
        command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(args.video_fps),
                   "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(args.output / "replay.mp4")]
        writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, args.height, args.width) as renderer:
            for frame in sorted(set(stills) | set(video_indices.tolist())):
                for k in range(2):
                    urdfs[k].update_cfg(arrays["dof_pos"][frame, k])
                bases = [pose(arrays["root_pos"][frame, k], arrays["root_quat_xyzw"][frame, k]) for k in range(2)]
                bases.append(pose(arrays["object_pos_w"][frame], arrays["object_quat_xyzw"][frame]))
                for (agent, node, _), mocap_id in zip(mapping, body_ids, strict=True):
                    tf = bases[agent] @ urdfs[agent].scene.graph.get(node)[0]
                    data.mocap_pos[mocap_id] = tf[:3, 3]
                    data.mocap_quat[mocap_id] = Rotation.from_matrix(tf[:3, :3]).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(model, data)  # FK only: never advance the physics.
                if args.camera_mode == "follow-pair":
                    camera.lookat[:2] = frame_points[frame, :, :2].mean(axis=0)
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, args.width, 65), fill=(248, 250, 252))
                draw.text((16, 7), args.title + " | " + args.checkpoint_label, font=font, fill=(20, 35, 55))
                draw.text((16, 35), f"Recorded physical rollout | t={frame/fps:.2f}s | frame {frame}/{count-1}", font=font, fill=(45, 65, 80))
                if frame in stills:
                    image.save(args.output / f"frame_{frame:04d}.png")
                    images[frame] = image.copy()
                if writer is not None:
                    for _ in range(int(np.count_nonzero(video_indices == frame))):
                        writer.stdin.write(image.tobytes())
                if frame % 100 == 0:
                    print(f"Rendered {frame}/{count-1}", flush=True)
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("ffmpeg failed")
    thumb_w, thumb_h = 640, round(640 * args.height / args.width)
    sheet = Image.new("RGB", (thumb_w * 3, thumb_h * ((len(stills) + 2) // 3)), "white")
    for index, frame in enumerate(stills):
        sheet.paste(images[frame].resize((thumb_w, thumb_h), Image.Resampling.LANCZOS),
                    ((index % 3) * thumb_w, (index // 3) * thumb_h))
    sheet.save(args.output / "contact_sheet.jpg", quality=94, subsampling=0)
    record = {
        "title": args.title, "checkpoint_label": args.checkpoint_label,
        "artifact_kind": "re_rendered_saved_physical_rollout_not_new_evaluation",
        "rendering": "URDF visual meshes, yourdfpy FK, MuJoCo offscreen rasterization; no mj_step/no policy inference",
        "rollout_source": str(args.rollout.resolve()), "rollout_sha256": sha(args.rollout),
        "robot_urdf_source": str(args.robot_urdf.resolve()), "robot_urdf_sha256": sha(args.robot_urdf),
        "object_urdf_source": str(args.object_urdf.resolve()), "object_urdf_sha256": sha(args.object_urdf),
        "recorded_metadata": metadata, "joint_names_verified_against_robot_config": names,
        "recording_fps": fps, "recording_frames": count, "video_fps": None if args.stills_only else args.video_fps,
        "video_frame_indices": video_indices.tolist(), "still_frame_indices": stills,
        "time_origin": "first recorded state; no trimming, smoothing, interpolation or pose edits",
        "camera": {"mode": args.camera_mode, "fixed_lookat": center.tolist(), "follow_xy": "mean(two robot roots, object origin)", "distance": camera.distance, "azimuth": args.azimuth, "elevation": args.elevation},
        "resolution": [args.width, args.height], "script_sha256": sha(Path(__file__)),
        "outputs_sha256": {f.name: sha(f) for f in args.output.iterdir() if f.is_file()},
    }
    (args.output / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"Export complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
