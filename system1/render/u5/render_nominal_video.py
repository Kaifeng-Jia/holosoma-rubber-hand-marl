#!/usr/bin/env python3
"""Nominal-episode MP4 in the paper Fig. 4 / V3 U5_media style (CLAUDE.md figure rules).

Copy of the U5 F3 video path (sources_snapshot/render_v3_f3_video.py + figround_common.py, sha256 in
sources_snapshot/SOURCES.sha256), changed only where the inputs live:
  * ROOT and the R0 material / helper paths point at this branch's snapshot instead of holosoma-core4d-base;
  * the osmesa/CPU assert is replaced by MUJOCO_GL=egl, the same replacement the U5 video script made;
  * the trajectory is any evaluate_paper_supplement.py trajectory (here: scheme B nominal episodes);
  * camera = the paper F3 camera of the same task (lookat, distance, azimuth 135, elevation -22), 1920x1080, 50 fps.
Robots keep the Unitree mesh colours of the native preset ('g1_native_execution'), object light blue, checker ground.
Kinematic replay of the saved states only: no physics step, no interpolation, no pose change. Every saved row is drawn;
for a completed episode the extra terminal row (index N repeats reference N-1) is excluded as in U5.

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl sonic_s1/bin/python -B system1/render/u5/render_nominal_video.py \
        --trajectory <dir>/nominal.trajectory.npz --summary <dir>/summary.json --task pull --out <file>.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SNAP = HERE / "sources_snapshot"
ROOT = HERE.parents[2]
sys.path.insert(0, str(SNAP))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def helper():
    code = (SNAP / "figround_common.py").read_text()
    reps = [("ROOT=Path('/home/kevin/holosoma-core4d-base')", "ROOT=Path(" + repr(str(ROOT)) + ")"),
            ("assert os.environ.get('MUJOCO_GL')=='osmesa' and os.environ.get('CUDA_VISIBLE_DEVICES')==''",
             "assert os.environ.get('MUJOCO_GL')=='egl'"),
            ("cfg=read(R0/'R0/material_lighting.json')", "cfg=read(" + repr(str(SNAP / "material_lighting.json")) + ")"),
            ("tree=ast.parse((R0/'sources/previous_r0_helpers.py').read_text())",
             "tree=ast.parse(Path(" + repr(str(SNAP / "previous_r0_helpers.py")) + ").read_text())")]
    for a, b in reps:
        assert code.count(a) == 1, a
        code = code.replace(a, b)
    mod = types.ModuleType("u5_helper")
    exec(compile(code, str(SNAP / "figround_common.py"), "exec"), mod.__dict__)
    return mod, hashlib.sha256(code.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True, help="evaluator summary.json (object URDF, completion)")
    p.add_argument("--task", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=None, help="smoke test only")
    a = p.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl"
    f, helper_sha = helper()
    import imageio_ffmpeg
    import mujoco
    from OpenGL import GL

    s = json.loads(a.summary.read_text())
    ep = next(e for e in s["episodes"] if Path(e["trajectory"]).resolve() == a.trajectory.resolve())
    robot_urdf = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
    object_urdf = ROOT / "src/holosoma" / s["object_urdf"]
    f3 = json.loads((SNAP / f"f3_manifest_{a.task}.json").read_text())
    cam_rec = f3["records"][0]["camera"]
    settings = cam_rec["settings"]
    with np.load(a.trajectory, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    q = np.concatenate([arr["root_pos"], arr["root_quat_xyzw"][..., [3, 0, 1, 2]], arr["dof_pos"]], axis=-1)
    o = np.concatenate([arr["object_pos_w"], arr["object_quat_xyzw"][..., [3, 0, 1, 2]]], axis=-1)
    rows = len(q)
    n = rows - 1 if (ep["completed_reference"] and rows == ep["horizon_steps"] + 1) else rows
    if a.max_frames:
        n = min(n, a.max_frames)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    scene = f.newscene(robot_urdf, object_urdf, 2, "g1_native_execution", 1920, 1080)
    f.styling(scene, settings["lookat"])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = settings["lookat"]
    cam.distance = settings["distance"]
    cam.azimuth = settings["azimuth"]
    cam.elevation = settings["elevation"]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1920x1080", "-r", "50", "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "18", "-preset", "fast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    first = None
    try:
        for i in range(n):
            scene.set_pose(q[i], o[i])
            scene.renderer.update_scene(scene.data, camera=cam)
            if i == 0:
                vendor = GL.glGetString(GL.GL_VENDOR).decode()
            px = scene.renderer.render()
            if i == 0:
                first = px.copy()
            proc.stdin.write(px.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
        scene.renderer.close()
    assert rc == 0, rc
    if first is not None:
        from PIL import Image
        Image.fromarray(first).save(a.out.with_suffix(".frame0.png"))
    rec = dict(task=a.task, trajectory=str(a.trajectory), trajectory_sha256=sha(a.trajectory), summary=str(a.summary),
               summary_sha256=sha(a.summary), model_id=s["model_id"], state_id=ep["state_id"], completed_reference=ep["completed_reference"],
               executed_steps=ep["steps"], saved_rows=rows, frames=n, fps=50, terminal_extra_state_excluded=n == rows - 1,
               camera_source=f"paper F3 camera, U5_F3_EXEC_20261104/outputs/{a.task}/manifest.json records[0]",
               camera=cam_rec, robot_urdf_sha256=sha(robot_urdf), object_urdf=str(object_urdf), object_urdf_sha256=sha(object_urdf),
               gl_vendor=vendor, helper_executed_sha256=helper_sha, code_sha256=sha(__file__),
               sources_snapshot_sha256=sha(SNAP / "SOURCES.sha256"), encoding_command=cmd, width=1920, height=1080,
               no_interpolation=True, no_smoothing=True, no_pose_modification=True, no_physics_steps=True, labels_ui=False,
               video_sha256=sha(a.out))
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
    print(a.out, n, "frames", vendor)


if __name__ == "__main__":
    main()
