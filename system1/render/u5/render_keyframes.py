#!/usr/bin/env python3
"""Keyframe montage of one evaluator episode in the U5 / paper Fig. 4 style (CLAUDE.md figure rules): the same scene,
materials and F3 camera as render_nominal_video.py (imported from it), Unitree mesh colours, light-blue object, checker
ground. Renders --frames evenly spaced saved rows (first and last executed row included; for a completed episode the extra
terminal row is excluded as in U5) at 960x540 and tiles them 3 per row with the step number in the corner.
Kinematic replay of saved states only (no physics, no interpolation).

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl sonic_s1/bin/python -B system1/render/u5/render_keyframes.py \
        --trajectory <dir>/nominal.trajectory.npz --summary <dir>/summary.json --task pull --out <file>.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("render_nominal_video", HERE / "render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--title", default="")
    a = p.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl"
    f, helper_sha = rnv.helper()
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    s = json.loads(a.summary.read_text())
    ep = next(e for e in s["episodes"] if Path(e["trajectory"]).name == a.trajectory.name)
    robot_urdf = rnv.ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    object_urdf = rnv.ROOT / "src/holosoma" / s["object_urdf"]
    cam_rec = json.loads((rnv.SNAP / f"f3_manifest_{a.task}.json").read_text())["records"][0]["camera"]
    st = cam_rec["settings"]
    with np.load(a.trajectory, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    q = np.concatenate([arr["root_pos"], arr["root_quat_xyzw"][..., [3, 0, 1, 2]], arr["dof_pos"]], axis=-1)
    o = np.concatenate([arr["object_pos_w"], arr["object_quat_xyzw"][..., [3, 0, 1, 2]]], axis=-1)
    rows = len(q)
    n = rows - 1 if (ep["completed_reference"] and rows == ep["horizon_steps"] + 1) else rows
    idx = sorted(set(np.linspace(0, n - 1, a.frames).round().astype(int).tolist()))
    W, H = 960, 540
    scene = f.newscene(robot_urdf, object_urdf, 2, "g1_native_execution", W, H)
    f.styling(scene, st["lookat"])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = st["lookat"]
    cam.distance, cam.azimuth, cam.elevation = st["distance"], st["azimuth"], st["elevation"]
    tiles = []
    for i in idx:
        scene.set_pose(q[i], o[i])
        scene.renderer.update_scene(scene.data, camera=cam)
        tiles.append(Image.fromarray(scene.renderer.render().copy()))
    scene.renderer.close()
    cols = 3
    nrow = (len(tiles) + cols - 1) // cols
    top = 44 if a.title else 0
    mont = Image.new("RGB", (cols * W, nrow * H + top), "white")
    d = ImageDraw.Draw(mont)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
    if a.title:
        d.text((12, 8), a.title, fill="black", font=font)
    for k, (i, t) in enumerate(zip(idx, tiles)):
        x, y = (k % cols) * W, top + (k // cols) * H
        mont.paste(t, (x, y))
        d.rectangle([x + 8, y + 8, x + 190, y + 44], fill="white")
        d.text((x + 14, y + 10), f"step {i} / {n - 1}", fill="black", font=font)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    mont.save(a.out)
    rec = dict(task=a.task, trajectory=str(a.trajectory), trajectory_sha256=rnv.sha(a.trajectory), summary_sha256=rnv.sha(a.summary),
               model_id=s["model_id"], state_id=ep["state_id"], completed_reference=ep["completed_reference"], executed_steps=ep["steps"],
               saved_rows=rows, frames=idx, tile=[W, H], camera=cam_rec, robot_colours="g1_native_execution (Unitree mesh colours)",
               helper_executed_sha256=helper_sha, code_sha256=rnv.sha(__file__), sources_snapshot_sha256=rnv.sha(rnv.SNAP / "SOURCES.sha256"),
               image_sha256=rnv.sha(a.out))
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(a.out, idx)


if __name__ == "__main__":
    main()
