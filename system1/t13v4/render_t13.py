#!/usr/bin/env python3
"""T13 v4 section 5 item 1: side-by-side replay videos of one finished T13 run (copy of system1/t12v2/render_t12v2.py, same
rules): +/-2 states nominal and perturbed_000, cameras full / close as T12-v2 plus "hands" (close-up of the four hands: lookat
on the reference object track at hand height, fixed distance / azimuth / elevation per task, chosen on framing stills). Left
Isaac (this run's laptop-5070 final, logs_system1/t13v4/finals/<run>_pm2), right MuJoCo (T10 setting,
logs_system1/t13v4/mj_finals/<run>_pm2). Render only: the saved per-step states are drawn as they are (no inference, no physics
step, no interpolation, no smoothing, no pose change).

- Scene: U5 / paper Fig. 4 style (system1/render/u5 helper, preset g1_native_execution: Unitree colours, light-blue object,
  checker ground); both robots in the same colour.
- Timing: 50 Hz rows at 25 fps, normal speed: video frame j shows row min(rint(2j), count - 1); both cells share the clock; a
  cell whose episode ended keeps its last executed row and says so; a completed episode's extra terminal row is not drawn.
- Each cell 1280x720, side by side (2560x720). Same camera parameters for both cells:
  * full: U5 camera_for over the mesh bounds of both episodes (every 10th row + last row), 7 % border, azimuth 135, elevation -22;
  * close: lookat = (reference object x, y of the current reference frame, CLOSE[task].z), fixed distance / azimuth /
    elevation per task (CLOSE below; chosen on framing stills of the Isaac cell before rendering: both robots, their hands and
    the object in view). The reference object track is the saved reference_object_pos_w (identical in both cells, checked).
- Label top left of each cell: policy, task, simulator, success / failure, termination step and reason (summary
  terminal_terms), current time and step.
- Key-frame sheet per video: 8 video frames evenly spaced over the timeline (j_i = rint(i (J - 1) / 7)), 2 columns x 4 rows,
  each tile the full side-by-side frame scaled to 1240 px wide, with the time above it; JPEG, quality lowered from 90 in steps of
  5 until the file is <= 1 MB.
- CPU rendering: MUJOCO_GL=egl with the Mesa EGL vendor (llvmpipe), CUDA_VISIBLE_DEVICES='' (GL vendor recorded).

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
        sonic_s1/bin/python -B system1/t12v2/render_t12v2.py --task chair --camera full --out exports_system1/T12_videos_v2
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.failure_t10s import causes as failure_causes  # noqa: E402
from system1.t10b.phases import termination_errors, termination_reason  # noqa: E402
spec = importlib.util.spec_from_file_location("render_nominal_video", ROOT / "system1/render/u5/render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)

TASKS = ("push", "pull", "bucket", "chair", "table")
TASK_ZH = dict(push="Table Pushing", pull="Table Pulling", bucket="Bin Transport", chair="Chair Righting", table="Table Transport")
ISAAC_T12 = dict(push="logs_system1/post_train/d_final_5070/D_push_paper_seed721/it13050_pm2",
             pull="logs_system1/post_train/d_curve/D_pull_paper_seed721/it08050_pm2",
             bucket="logs_system1/post_train/d_curve/D_bucket_paper_B-RZ0_seed721/it12000_pm2",
             chair="logs_system1/post_train/d_final_5070/D_chair_paper_seed721/it12000_pm2",
             table="logs_system1/post_train/d_curve/D_table_paper_T-07_seed721/it12000_pm2")  # = contact_stats.runs() Isaac, seed 721, +/-2
RUN_TASK = dict(d_push="push", d_table="table", d_table_pa="table", c_out_table="table", c_out_push="push")
RUN_TASK.update(d_frontback="table", d_frontback_smoke="table")  # T14 front-back training task
RUN_ZH = dict(d_push="D（T13 v4 d_push，种子 721）", d_table="D（T13 v4 d_table，种子 721）", d_table_pa="D（T13 v4 d_table_pa，种子 721）",
              c_out_table="C-out（T13 v4 c_out_table，种子 721）", c_out_push="C-out（T13 v4 c_out_push，种子 721）")
RUN_ZH.update(d_frontback="D（T14 一前一后 d_frontback，种子 721）", d_frontback_smoke="D（一前一后冒烟 checkpoint）")
ISAAC = {r: f"logs_system1/t13v4/finals/{r}_pm2" for r in RUN_TASK}
MUJOCO = {r: f"logs_system1/t13v4/mj_finals/{r}_pm2" for r in RUN_TASK}
STATE = "perturbed_000"  # overwritten by --state
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
W, H = 1280, 720
FPS, VIDEO_FPS = 50, 25
AZ, EL = 135, -22
# z, distance, azimuth, elevation. Table = T12. The others were chosen on stills of rows 35 % / 70 % of the reference (azimuth
# 90 / 135 / 180 compared at T12's 0.55 m / 2.0 m / -30, then distance and height widened so heads and the lifted bin stay in view);
# stills and sheets in logs_system1/t12v2/framing/.
HANDS = dict(push=(0.75, 1.4, 90, -20), table=(0.85, 1.4, 180, -20))  # z, distance, azimuth, elevation (framing stills)
CLOSE = dict(push=(0.55, 2.3, 90, -30), pull=(0.55, 2.3, 90, -30), bucket=(0.75, 2.6, 135, -25), chair=(0.55, 2.3, 90, -30),
             table=(0.55, 2.0, 180, -30))
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SIM_ZH = {"isaac": "Isaac", "mujoco": "MuJoCo"}
TERM_ZH = dict(object_position="物体位置偏差 > 0.25 m", object_orientation="物体姿态偏差 > 0.8 rad", robot0_tilt="机器人 0 躯干倾斜超限",
               robot1_tilt="机器人 1 躯干倾斜超限", robot_orientation="躯干倾斜超限（Isaac 轨迹未存，按排除法）")
KEYFRAMES, TILE_W, MAX_JPEG = 8, 1240, 1_000_000


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def reason(sim, e, z):
    """termination causes as T10 / T10b: the evaluator's single term joint_bad_tracking covers object position (> 0.25 m),
    object orientation (> 0.8 rad) and torso tilt (gravity-z error > 0.8). MuJoCo: from the saved last_termination record
    (system1/sim2sim/failure_t10s.causes, several can hold); Isaac: object errors at the last row against the reference frame
    the term compared with (system1/t10b/phases.termination_errors / termination_reason; torso tilt is not saved, so it is the
    remaining case)."""
    if e["completed_reference"]:
        return []
    if sim == "mujoco":
        return failure_causes(e["last_termination"])
    pe, oe, _ = termination_errors(z)
    return [termination_reason(pe, oe)]


def load_segments(run):
    segs = {}
    ck = st = None
    for sim, d in (("isaac", ISAAC[run]), ("mujoco", MUJOCO[run])):
        summ = ROOT / d / "summary.json"
        s = json.loads(summ.read_text())
        ck = ck or s["checkpoint_sha256"]
        st = st or s["states_sha256"]
        if s["checkpoint_sha256"] != ck or s["states_sha256"] != st:
            raise ValueError(f"{run}: Isaac and MuJoCo checkpoint / state file differ")
        e = next(x for x in s["episodes"] if x["state_id"] == STATE)
        traj = Path(e["trajectory"])
        with np.load(traj, allow_pickle=False) as z:
            arr = {k: z[k] for k in z.files}
        rows = len(arr["dof_pos"])
        n = rows - 1 if (e["completed_reference"] and rows == e["horizon_steps"] + 1) else rows
        q = np.concatenate([arr["root_pos"], arr["root_quat_xyzw"][..., [3, 0, 1, 2]], arr["dof_pos"]], axis=-1)
        o = np.concatenate([arr["object_pos_w"], arr["object_quat_xyzw"][..., [3, 0, 1, 2]]], axis=-1)
        if not np.array_equal(arr["reference_frame"][:n], np.minimum(np.arange(n), e["horizon_steps"] - 1)):
            raise ValueError(f"{traj}: saved reference frame is not min(row, N-1)")
        segs[sim] = dict(q=q, o=o, n=n, rows=rows, ep=e, summary=summ, summary_data=s, traj=traj, terms=reason(sim, e, arr), ref_obj=arr["reference_object_pos_w"][:n])
    a, b = segs["isaac"]["ref_obj"], segs["mujoco"]["ref_obj"]
    m = min(len(a), len(b))
    ref_diff = float(np.abs(a[:m] - b[:m]).max())
    if ref_diff > 1e-4:
        raise ValueError(f"{run}: reference object tracks differ between the cells by {ref_diff}")
    return segs, ck, st, ref_diff


def result_text(seg):
    e = seg["ep"]
    if e["completed_reference"]:
        return f"成功：{e['steps']} 步走完"
    why = "、".join(TERM_ZH.get(t, t) for t in seg["terms"]) or "未记录"
    return f"失败：第 {e['steps']} 步终止，{why}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(RUN_TASK), required=True)
    ap.add_argument("--state", choices=("nominal", "perturbed_000"), required=True)
    ap.add_argument("--camera", choices=("full", "close", "hands"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test only")
    ap.add_argument("--still", type=int, default=None, help="render only this row to a PNG (framing check)")
    ap.add_argument("--close", default=None, help="framing check only: z,distance,azimuth,elevation of the close camera")
    a = ap.parse_args()
    global STATE
    STATE = a.state
    a.task = RUN_TASK[a.run]
    cz, cdist, caz, cel = (HANDS if a.camera == "hands" else CLOSE)[a.task]
    if a.close:
        if a.still is None:
            raise ValueError("--close is for framing stills only")
        cz, cdist, caz, cel = (float(x) for x in a.close.split(","))
    assert os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    f, helper_sha = rnv.helper()
    import imageio_ffmpeg
    import mujoco
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont

    segs, ck, states_sha, ref_diff = load_segments(a.run)
    object_urdf = ROOT / "src/holosoma" / segs["isaac"]["summary_data"]["object_urdf"]
    if segs["mujoco"]["summary_data"].get("object_urdf") not in (None, segs["isaac"]["summary_data"]["object_urdf"]):
        raise ValueError("object URDF differs")
    count = max(s["n"] for s in segs.values())
    ref = max((s["ref_obj"] for s in segs.values()), key=len)
    video_rows = np.minimum(np.rint(np.arange(0, count / FPS, 1 / VIDEO_FPS) * FPS).astype(int), count - 1)
    if a.max_frames:
        video_rows = video_rows[: a.max_frames]
    J = len(video_rows)
    key_j = sorted(set(int(x) for x in np.rint(np.linspace(0, J - 1, KEYFRAMES))))
    scene = f.newscene(ROBOT_URDF, object_urdf, 2, "g1_native_execution", W, H)
    if a.camera == "full":
        pts = []
        for seg in segs.values():
            for t in sorted(set(range(0, seg["n"], 10)) | {seg["n"] - 1}):
                scene.set_pose(seg["q"][t], seg["o"][t])
                pts.append(f.bounds(scene))
        cam_rule = f.camera_for(np.concatenate(pts), W, H, AZ, EL)
        f.styling(scene, cam_rule["lookat"])
        camera_record = dict(kind="full", rule="U5 camera_for over mesh bounds of both episodes (every 10th row + last row), 7 % border",
                             settings=cam_rule, fovy=45)
    else:
        f.styling(scene, [float(ref[:, 0].mean()), float(ref[:, 1].mean()), cz])
        camera_record = dict(kind=a.camera, rule="lookat = (reference object x, y at the current reference frame, z); fixed distance / azimuth / elevation",
                             z=cz, distance=cdist, azimuth=caz, elevation=cel, fovy=45, reference_track="saved reference_object_pos_w",
                             reference_track_max_diff_between_cells_m=ref_diff)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = (AZ, EL) if a.camera == "full" else (caz, cel)

    def set_cam(row):
        if a.camera == "full":
            cam.lookat[:] = cam_rule["lookat"]
            cam.distance = cam_rule["distance"]
        else:
            r = ref[min(row, len(ref) - 1)]
            cam.lookat[:] = [r[0], r[1], cz]
            cam.distance = cdist

    font = ImageFont.truetype(FONT, 26)
    font_small = ImageFont.truetype(FONT, 22)
    order = ("isaac", "mujoco")
    vendor = None

    def frame(row):
        nonlocal vendor
        canvas = Image.new("RGB", (2 * W, H), "white")
        set_cam(row)
        for idx, sim in enumerate(order):
            seg = segs[sim]
            r = min(row, seg["n"] - 1)
            scene.set_pose(seg["q"][r], seg["o"][r])
            scene.renderer.update_scene(scene.data, camera=cam)
            if vendor is None:
                vendor = GL.glGetString(GL.GL_VENDOR).decode() + " / " + GL.glGetString(GL.GL_RENDERER).decode()
            img = Image.fromarray(scene.renderer.render())
            d = ImageDraw.Draw(img, "RGBA")
            ended = row > seg["n"] - 1
            lines = [f"{RUN_ZH[a.run]} · {TASK_ZH[a.task]} · {SIM_ZH[sim]}", result_text(seg),
                     f"t = {r / FPS:.2f} s，第 {r} 步" + ("（回合已结束，停在最后一帧）" if ended else "")]
            fail = not seg["ep"]["completed_reference"]
            d.rounded_rectangle((12, 12, 800, 128), radius=10, fill=(255, 255, 255, 215), outline=(150, 160, 170, 255))
            d.text((26, 18), lines[0], font=font, fill=(20, 35, 55))
            d.text((26, 54), lines[1], font=font, fill=(170, 40, 30) if fail else (25, 110, 45))
            d.text((26, 92), lines[2], font=font_small, fill=(70, 80, 90))
            canvas.paste(img, (idx * W, 0))
        ImageDraw.Draw(canvas).line([(W, 0), (W, H)], fill=(255, 255, 255), width=4)
        return canvas

    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.run}_seed721_pm2_{STATE}_{a.camera}" + (f"_{a.close.replace(',', '_')}" if a.close else "")
    if a.still is not None:
        p = a.out / "stills" / f"{stem}_row{a.still:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        frame(a.still).save(p)
        print("still", p, vendor)
        return 0
    out = a.out / f"{stem}.mp4"
    if out.exists():
        raise FileExistsError(out)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{2 * W}x{H}", "-r", str(VIDEO_FPS), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "20",
           "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    tiles = {}
    try:
        for j, row in enumerate(video_rows):
            img = frame(int(row))
            if j == 0:
                img.save(a.out / f"{stem}.frame0.png")
            if j in key_j:
                tiles[j] = (int(row), img.resize((TILE_W, round(TILE_W * H / (2 * W))), Image.LANCZOS))
            proc.stdin.write(img.tobytes())
            if j % 50 == 0:
                print(f"{stem} frame {j}/{J} row {row} {time.time() - t0:.0f}s", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
        scene.renderer.close()
    assert rc == 0, rc
    # key-frame sheet
    th = round(TILE_W * H / (2 * W))
    cap, head, pad = 40, 120, 12
    cols, nrow = 2, (len(key_j) + 1) // 2
    sheet = Image.new("RGB", (cols * TILE_W + (cols + 1) * pad, head + nrow * (cap + th + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    fh = ImageFont.truetype(FONT, 34)
    fc = ImageFont.truetype(FONT, 26)
    d.text((pad, 14), f"{TASK_ZH[a.task]} · {RUN_ZH[a.run]} · ±2 {STATE} · {dict(full='全景', close='近景', hands='双手近景')[a.camera]}", font=fh, fill=(20, 35, 55))
    d.text((pad, 66), f"每格左：Isaac，{result_text(segs['isaac'])}；右：MuJoCo，{result_text(segs['mujoco'])}", font=fc, fill=(70, 80, 90))
    for i, j in enumerate(key_j):
        row, tile = tiles[j]
        x = pad + (i % cols) * (TILE_W + pad)
        y = head + (i // cols) * (cap + th + pad)
        d.text((x, y + 4), f"t = {row / FPS:.2f} s（第 {row} 步）", font=fc, fill=(20, 35, 55))
        sheet.paste(tile, (x, y + cap))
    jpg = a.out / f"{stem}.keyframes.jpg"
    for qual in range(90, 20, -5):
        sheet.save(jpg, quality=qual, optimize=True)
        if jpg.stat().st_size <= MAX_JPEG:
            break
    assert jpg.stat().st_size <= MAX_JPEG
    cells = []
    for idx, sim in enumerate(order):
        seg = segs[sim]
        e = seg["ep"]
        cells.append(dict(cell=["left", "right"][idx], simulator=sim, state_id=STATE, checkpoint_sha256=ck, states_sha256=states_sha,
                          trajectory=str(seg["traj"]), trajectory_sha256=sha(seg["traj"]), summary=rel(seg["summary"]), summary_sha256=sha(seg["summary"]),
                          completed_reference=e["completed_reference"], executed_steps=e["steps"], horizon_steps=e["horizon_steps"],
                          termination_terms=seg["terms"], saved_rows=seg["rows"], drawn_rows=seg["n"], rerun=False))
    rec = dict(task="T13 v4 section 5 item 1", run=a.run, scene_task=a.task, video=rel(out), video_sha256=sha(out), keyframes=rel(jpg), keyframes_sha256=sha(jpg),
               keyframes_bytes=jpg.stat().st_size, keyframes_jpeg_quality=qual, keyframe_rows=[tiles[j][0] for j in key_j], keyframe_video_frames=key_j,
               state_id=STATE, seed=721, level_cm=2, camera=camera_record, layout="1x2: left Isaac, right MuJoCo", cell_resolution=[W, H],
               video_resolution=[2 * W, H], recording_fps=FPS, video_fps=VIDEO_FPS, video_frames=J, timeline_rows=count,
               frame_rule="video frame j shows saved row min(rint(j * 50 / 25), count - 1); a cell past its last executed row keeps it",
               cells=cells, scene_style="U5 / paper Fig. 4: preset g1_native_execution (Unitree mesh colours), light-blue object, checker ground",
               robot_urdf=rel(ROBOT_URDF), robot_urdf_sha256=sha(ROBOT_URDF), object_urdf=rel(object_urdf), object_urdf_sha256=sha(object_urdf),
               gl_vendor=vendor, helper_executed_sha256=helper_sha, code_sha256=sha(__file__), u5_sources_snapshot_sha256=sha(rnv.SNAP / "SOURCES.sha256"),
               encoding_command=cmd, no_interpolation=True, no_smoothing=True, no_pose_modification=True, no_physics_steps=True,
               no_policy_inference=True, wall_s=round(time.time() - t0))
    (a.out / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    print(out, J, "frames", vendor, "keyframes", jpg.stat().st_size, "q", qual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
