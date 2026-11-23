#!/usr/bin/env python3
"""T12-v: 2x2 replay videos of D Table seed 721 (rows: original / randomized pilot; columns: Isaac / MuJoCo), +/-2 test states
perturbed_000, perturbed_001, perturbed_002 (fixed in advance by the task). Render only: the saved per-step states of the final
evaluations are drawn as they are (no policy inference, no physics step, no interpolation, no smoothing, no pose change).

- Scene, materials and lighting: the U5 / paper Fig. 4 style of system1/render/u5/render_nominal_video.py (imported helper):
  Unitree mesh colours (preset 'g1_native_execution'), light-blue object, checker ground.
- Timing (docs/writing_handoff/media): 50 Hz saved states shown at 25 fps, normal speed: video frame j shows saved row 2j.
  All four cells share one clock; a cell whose episode has ended keeps its last executed row and says so. For a completed
  episode the extra terminal row (index N repeats reference N-1) is not drawn, as in U5.
- Each cell 1280x720, tiled 2x2 (2560x1440). Two cameras per state, the same parameters for all four cells of the state:
  * full: the U5 camera_for rule (all rendered robot and object mesh bounds of the four episodes, every 10th row and the last
    row, 7 % border), azimuth 135, elevation -22, fixed;
  * close: looks at the REFERENCE object position of the current reference frame (x, y) at height CLOSE_Z, distance CLOSE_DIST,
    azimuth 180, elevation -30 (the reference is the same for all four cells).
- Label top left of each cell: policy, simulator, success / failure, termination step and reason (T10b finals episodes.csv),
  current time and step.
- CPU rendering: MUJOCO_GL=egl with the Mesa EGL vendor (llvmpipe), CUDA_VISIBLE_DEVICES='' (the GL vendor is recorded).

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
        sonic_s1/bin/python -B system1/t12/render_t12.py --state perturbed_000 --camera full --out exports_system1/T12_videos
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("render_nominal_video", ROOT / "system1/render/u5/render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)

SOURCES = {  # (policy, simulator) -> +/-2 final evaluation directory
    ("original", "isaac"): "logs_system1/post_train/d_curve/D_table_paper_T-07_seed721/it12000_pm2",
    ("original", "mujoco"): "logs_system1/t10/mj_eval/d_table_seed721_pm2",
    ("pilot", "isaac"): "logs_system1/t10b/finals/d_table_seed721_rand_pm2",
    ("pilot", "mujoco"): "logs_system1/t10b/mj_finals/d_table_seed721_rand_pm2",
}
CHECKPOINT = {"original": "f163d11a9629edae5dd72732c4ddec08a43024533a47b349a099a3275a8d1c50",
              "pilot": "2f343369af2865c448dcc0406a9e6c788ea67675a93312dca76f58bf8ded1806"}
STATES_SHA = "389e8da5fb818572a48cbba049254585311079439265f5ab4cfc1523a185dac1"
EPISODES_CSV = ROOT / "results_system1/T10b_friction_margin/retrain/finals/d_table_seed721_rand/episodes.csv"
REFERENCE = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz"
REFERENCE_SHA = "582e76693f877c61b0b09ab3b584922f330aeb85cae6035f6b7cd2ae729ee153"
OBJECT_URDF = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf"
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
W, H = 1280, 720
FPS, VIDEO_FPS = 50, 25
AZ, EL = 135, -22
CLOSE_Z, CLOSE_DIST, CLOSE_AZ, CLOSE_EL = 0.55, 2.0, 180, -30  # chosen on framing stills of perturbed_000 rows 150 / 300 / 320 (sees both robots, the hands and the table)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
POLICY_ZH = {"original": "原始策略（论文 D，it12000）", "pilot": "随机化 pilot（model_14000）"}
SIM_ZH = {"isaac": "Isaac", "mujoco": "MuJoCo"}
REASON_ZH = {"object_position": "物体位置偏差 > 0.25 m", "object_orientation": "物体姿态偏差 > 0.8 rad", "robot_orientation": "躯干倾斜超限"}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def load_segments(state):
    reasons = {}
    for r in csv.DictReader(EPISODES_CSV.open()):
        if r["level_cm"] == "2":
            reasons[(r["policy"], r["simulator"], r["state_id"])] = r
    segs = {}
    for (pol, sim), d in SOURCES.items():
        summ = ROOT / d / "summary.json"
        s = json.loads(summ.read_text())
        if s["checkpoint_sha256"] != CHECKPOINT[pol] or s["states_sha256"] != STATES_SHA:
            raise ValueError(f"{d}: checkpoint or state file differs from the task")
        e = next(x for x in s["episodes"] if x["state_id"] == state)
        traj = Path(e["trajectory"])
        with np.load(traj, allow_pickle=False) as z:
            arr = {k: z[k] for k in ("root_pos", "root_quat_xyzw", "dof_pos", "object_pos_w", "object_quat_xyzw", "reference_frame")}
        rows = len(arr["dof_pos"])
        n = rows - 1 if (e["completed_reference"] and rows == e["horizon_steps"] + 1) else rows
        q = np.concatenate([arr["root_pos"], arr["root_quat_xyzw"][..., [3, 0, 1, 2]], arr["dof_pos"]], axis=-1)
        o = np.concatenate([arr["object_pos_w"], arr["object_quat_xyzw"][..., [3, 0, 1, 2]]], axis=-1)
        rr = reasons[(pol, sim, state)]
        if (rr["completed_reference"] == "True") != e["completed_reference"] or int(rr["executed_steps"]) != e["steps"]:
            raise ValueError(f"{d} {state}: episodes.csv disagrees with the summary")
        segs[(pol, sim)] = dict(q=q, o=o, n=n, rows=rows, ep=e, summary=summ, traj=traj, reason=rr.get("termination_reason") or None,
                                reference_frame=arr["reference_frame"])
    return segs


def label_lines(pol, sim, seg, row):
    e = seg["ep"]
    if e["completed_reference"]:
        res = f"成功：{e['steps']} 步走完"
    else:
        res = f"失败：第 {e['steps']} 步终止，{REASON_ZH.get(seg['reason'], seg['reason'])}"
    r = min(row, seg["n"] - 1)
    ended = row > seg["n"] - 1
    clock = f"t = {r / FPS:.2f} s，第 {r} 步" + ("（回合已结束，停在最后一帧）" if ended else "")
    return [f"{POLICY_ZH[pol]} · {SIM_ZH[sim]}", res, clock]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", choices=("perturbed_000", "perturbed_001", "perturbed_002"), required=True)
    ap.add_argument("--camera", choices=("full", "close"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test only")
    ap.add_argument("--still", type=int, default=None, help="render only this saved row to a PNG (framing check)")
    ap.add_argument("--close", default=None, help="framing check only: z,distance,azimuth,elevation of the close camera")
    a = ap.parse_args()
    global CLOSE_Z, CLOSE_DIST, CLOSE_AZ, CLOSE_EL
    if a.close:
        if a.still is None:
            raise ValueError("--close is for framing stills only")
        CLOSE_Z, CLOSE_DIST, CLOSE_AZ, CLOSE_EL = (float(x) for x in a.close.split(","))
    assert os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    f, helper_sha = rnv.helper()
    import imageio_ffmpeg
    import mujoco
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont

    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError("reference file changed")
    ref = np.load(REFERENCE)["object_pos_w"]
    segs = load_segments(a.state)
    for seg in segs.values():  # the cells' saved reference frames are the row index (clipped at the last frame)
        if not np.array_equal(seg["reference_frame"][: seg["n"]], np.minimum(np.arange(seg["n"]), len(ref) - 1)):
            raise ValueError("saved reference frame is not the row index")
    count = max(s["n"] for s in segs.values())
    video_rows = np.minimum(np.rint(np.arange(0, count / FPS, 1 / VIDEO_FPS) * FPS).astype(int), count - 1)
    if a.max_frames:
        video_rows = video_rows[: a.max_frames]
    scene = f.newscene(ROBOT_URDF, OBJECT_URDF, 2, "g1_native_execution", W, H)
    if a.camera == "full":
        pts = []
        for seg in segs.values():
            rows = sorted(set(range(0, seg["n"], 10)) | {seg["n"] - 1})
            for t in rows:
                scene.set_pose(seg["q"][t], seg["o"][t])
                pts.append(f.bounds(scene))
        cam_rule = f.camera_for(np.concatenate(pts), W, H, AZ, EL)
        f.styling(scene, cam_rule["lookat"])
        camera_record = dict(kind="full", rule="U5 camera_for over mesh bounds of all four episodes (every 10th row + last row), 7 % border",
                             settings=cam_rule, fovy=45)
    else:
        f.styling(scene, [float(ref[:, 0].mean()), float(ref[:, 1].mean()), CLOSE_Z])
        camera_record = dict(kind="close", rule="lookat = (reference object x, y at the current reference frame, CLOSE_Z); fixed distance / azimuth / elevation",
                             close_z=CLOSE_Z, distance=CLOSE_DIST, azimuth=CLOSE_AZ, elevation=CLOSE_EL, fovy=45, reference=str(REFERENCE.relative_to(ROOT)),
                             reference_sha256=REFERENCE_SHA)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = (AZ, EL) if a.camera == "full" else (CLOSE_AZ, CLOSE_EL)

    def set_cam(row):
        if a.camera == "full":
            cam.lookat[:] = cam_rule["lookat"]
            cam.distance = cam_rule["distance"]
        else:
            r = ref[min(row, len(ref) - 1)]
            cam.lookat[:] = [r[0], r[1], CLOSE_Z]
            cam.distance = CLOSE_DIST

    font = ImageFont.truetype(FONT, 26)
    font_small = ImageFont.truetype(FONT, 22)
    order = [("original", "isaac"), ("original", "mujoco"), ("pilot", "isaac"), ("pilot", "mujoco")]
    vendor = None

    def frame(row):
        nonlocal vendor
        canvas = Image.new("RGB", (2 * W, 2 * H), "white")
        set_cam(row)
        for idx, key in enumerate(order):
            seg = segs[key]
            r = min(row, seg["n"] - 1)
            scene.set_pose(seg["q"][r], seg["o"][r])
            scene.renderer.update_scene(scene.data, camera=cam)
            if vendor is None:
                vendor = GL.glGetString(GL.GL_VENDOR).decode() + " / " + GL.glGetString(GL.GL_RENDERER).decode()
            img = Image.fromarray(scene.renderer.render())
            d = ImageDraw.Draw(img, "RGBA")
            lines = label_lines(*key, seg, row)
            fail = not seg["ep"]["completed_reference"]
            d.rounded_rectangle((12, 12, 760, 128), radius=10, fill=(255, 255, 255, 215), outline=(150, 160, 170, 255))
            d.text((26, 18), lines[0], font=font, fill=(20, 35, 55))
            d.text((26, 54), lines[1], font=font, fill=(170, 40, 30) if fail else (25, 110, 45))
            d.text((26, 92), lines[2], font=font_small, fill=(70, 80, 90))
            canvas.paste(img, ((idx % 2) * W, (idx // 2) * H))
        dd = ImageDraw.Draw(canvas)
        dd.line([(W, 0), (W, 2 * H)], fill=(255, 255, 255), width=4)
        dd.line([(0, H), (2 * W, H)], fill=(255, 255, 255), width=4)
        return canvas

    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"d_table_seed721_pm2_{a.state}_{a.camera}" + (f"_{a.close.replace(',', '_')}" if a.close else "")
    if a.still is not None:
        frame(a.still).save(a.out / f"{stem}_still_row{a.still:03d}.png")
        print("still", a.out / f"{stem}_still_row{a.still:03d}.png", vendor)
        return 0
    out = a.out / f"{stem}.mp4"
    if out.exists():
        raise FileExistsError(out)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{2 * W}x{2 * H}", "-r", str(VIDEO_FPS), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "20",
           "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for j, row in enumerate(video_rows):
            img = frame(int(row))
            if j == 0:
                img.save(a.out / f"{stem}.frame0.png")
            proc.stdin.write(img.tobytes())
            if j % 50 == 0:
                print(f"{stem} frame {j}/{len(video_rows)} row {row} {time.time() - t0:.0f}s", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
        scene.renderer.close()
    assert rc == 0, rc
    cells = []
    for idx, key in enumerate(order):
        seg = segs[key]
        e = seg["ep"]
        cells.append(dict(cell=["top-left", "top-right", "bottom-left", "bottom-right"][idx], policy=key[0], simulator=key[1],
                          state_id=a.state, checkpoint_sha256=CHECKPOINT[key[0]], states_sha256=STATES_SHA,
                          trajectory=str(seg["traj"]), trajectory_sha256=sha(seg["traj"]), summary=str(seg["summary"].relative_to(ROOT)),
                          summary_sha256=sha(seg["summary"]), completed_reference=e["completed_reference"], executed_steps=e["steps"],
                          horizon_steps=e["horizon_steps"], termination_reason=seg["reason"], saved_rows=seg["rows"], drawn_rows=seg["n"],
                          rerun=False))
    rec = dict(task="T12-v", video=rel(out), video_sha256=sha(out), state_id=a.state, camera=camera_record,
               layout="2x2: rows original / pilot, columns Isaac / MuJoCo", cell_resolution=[W, H], video_resolution=[2 * W, 2 * H],
               recording_fps=FPS, video_fps=VIDEO_FPS, video_frames=len(video_rows), timeline_rows=count,
               frame_rule="video frame j shows saved row min(rint(j * 50 / 25), count - 1); a cell past its last executed row keeps it",
               video_rows_first_last=[int(video_rows[0]), int(video_rows[-1])], cells=cells,
               scene_style="U5 / paper Fig. 4: preset g1_native_execution (Unitree mesh colours), light-blue object, checker ground",
               robot_urdf=str(ROBOT_URDF.relative_to(ROOT)), robot_urdf_sha256=sha(ROBOT_URDF), object_urdf=str(OBJECT_URDF.relative_to(ROOT)),
               object_urdf_sha256=sha(OBJECT_URDF), gl_vendor=vendor, helper_executed_sha256=helper_sha, code_sha256=sha(__file__),
               u5_sources_snapshot_sha256=sha(rnv.SNAP / "SOURCES.sha256"), episodes_csv=str(EPISODES_CSV.relative_to(ROOT)),
               episodes_csv_sha256=sha(EPISODES_CSV), encoding_command=cmd, no_interpolation=True, no_smoothing=True, no_pose_modification=True,
               no_physics_steps=True, no_policy_inference=True, wall_s=time.time() - t0)
    (a.out / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    print(out, len(video_rows), "frames", vendor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
