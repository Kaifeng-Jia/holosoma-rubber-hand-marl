#!/usr/bin/env python3
"""T14 items 1-2: size-level grid figure and video per task (kinematic references, not policy replays). Render only, CPU.

Levels: Table / Bin / Chair k = 0.8 ... 1.2 (two-person, uniform scale); Push / Pull m = a, 1.0, 1.5, 2.0 (single person,
stretched table). Each level = a 30 Hz two-robot reference (robot_qpos [T, 2, 36] wxyz) + object pose + that level's object
URDF.

Columns (same for every level): contact start, middle, end = first / middle / last source frame in which any hand of the
person(s) touches the object (any SMPL-X hand vertex within 1.5 cm, the stop-1 rule), as 30 Hz reference frames.

Grid: one row per level, one column per moment, full view (both robots and the whole object). The camera of a column is
the same for every row, fitted (U5 camera_for, 7 % border) to the robot and object mesh bounds of the LARGEST level at that
frame; azimuth = the horizontal robot-0 -> robot-1 direction at that frame - 90 deg + 30 deg, elevation -25 deg. Each cell
is labelled with the level and the time; a header column per row gives that level's numbers (contact retention, max
penetration, solve success, smallest robot-robot distance). U5 / Fig. 4 style (preset g1_native_execution).

Video: the levels one after another (each the whole reference at 30 fps), one fixed camera fitted to the largest level over
its whole motion (every 10th frame), same azimuth rule at frame 0; label = task, level, time.

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      sonic_s1/bin/python -B system1/t14/render_sizes.py --task table --levels levels.json --out results_system1/T14_retargeting/step2_sizes/figures --video-out exports_system1/T14_videos
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("render_nominal_video", ROOT / "system1/render/u5/render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CW, CH, HEADER_W = 720, 480, 330
VW, VH = 1280, 720
AZ_OFF, EL = 30.0, -25.0
COLS = (("接触开始", 0), ("中段", 1), ("接触结束", 2))
OBJECT_RGBA = (0.76, 0.80, 0.86, 1.0)  # mesh objects without a URDF material (Bin, Chair): the Table's light blue (desk001 URDF material)
TASK_ZH = dict(table="Table Transport", bin="Bin Transport", chair="Chair Righting", push="Table Pushing", pull="Table Pulling")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--levels", type=Path, required=True, help="json: {contact_frames_30hz: [start, mid, end], levels: [{label, pair, urdf, numbers}]}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-out", type=Path, default=None)
    a = ap.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    spec = json.loads(a.levels.read_text())
    f, helper_sha = rnv.helper()
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    font, small = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 20)
    levels = spec["levels"]
    data = []
    for L in levels:
        if L.get("failed"):  # no two-robot reference at this level: placeholder row, not in the video
            data.append(dict(q=None, o=None, L=L))
            continue
        z = np.load(ROOT / L["pair"])
        data.append(dict(q=z["robot_qpos"].astype(np.float64), o=z["object_qpos"].astype(np.float64), L=L))
    largest = max((i for i in range(len(levels)) if not levels[i].get("failed")), key=lambda i: levels[i]["size"])
    frames = spec["contact_frames_30hz"]
    scenes = {}

    def scene_for(i, w, h):
        key = (i, w, h)
        if key not in scenes:
            s = f.newscene(ROBOT_URDF, ROOT / levels[i]["urdf"], 2, "g1_native_execution", w, h)
            s.opt = mujoco.MjvOption()
            s.recolor = "<material" not in (ROOT / levels[i]["urdf"]).read_text()
            scenes[key] = s
        return scenes[key]

    def style(s, lookat):
        recs = f.styling(s, lookat)
        if s.recolor:
            for r in recs:
                if r["role"] == "object":
                    s.model.geom_rgba[r["geom_id"]] = OBJECT_RGBA

    def az_at(d, t):
        ax = d["q"][t, 1, :2] - d["q"][t, 0, :2]
        return float(np.degrees(np.arctan2(ax[1], ax[0]))) - 90.0 + AZ_OFF

    def render(scene, cam, q, o):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["distance"], cam["azimuth"], cam["elevation"]
        scene.set_pose(q, o)
        scene.renderer.update_scene(scene.data, camera=c)
        return Image.fromarray(scene.renderer.render())

    # ------------------------------------------------------------------ grid
    cams = []
    big = data[largest]
    for name, ci in COLS:
        t = min(frames[ci], len(big["q"]) - 1)
        s = scene_for(largest, CW, CH)
        s.set_pose(big["q"][t], big["o"][t])
        cam = f.camera_for(f.bounds(s), CW, CH, az_at(big, t), EL)
        cams.append(dict(column=name, frame_30hz=int(t), time_s=round(t / 30, 3), **cam))
    styled = set()
    W = HEADER_W + 3 * CW + 2 * 4
    sheet = Image.new("RGB", (W, len(levels) * (CH + 4)), "white")
    for ri, d in enumerate(data):
        if d["q"] is None:
            hdr = Image.new("RGB", (HEADER_W, CH), (245, 247, 250))
            dh = ImageDraw.Draw(hdr)
            dh.text((16, 14), d["L"]["label"], font=font, fill=(20, 35, 55))
            y = 60
            for line in d["L"]["numbers"]:
                while line:  # wrap to the header width
                    n = len(line)
                    while dh.textlength(line[:n], font=small) > HEADER_W - 28:
                        n -= 1
                    dh.text((16, y), line[:n], font=small, fill=(170, 40, 30))
                    line = line[n:]
                    y += 30
            sheet.paste(hdr, (0, ri * (CH + 4)))
            for ci in range(3):
                ph = Image.new("RGB", (CW, CH), (232, 234, 238))
                dp = ImageDraw.Draw(ph)
                dp.text((24, CH // 2 - 20), f"{d['L']['label']}：求解失败，无双人参考", font=font, fill=(120, 40, 30))
                sheet.paste(ph, (HEADER_W + ci * (CW + 4), ri * (CH + 4)))
            continue
        s = scene_for(ri, CW, CH)
        if ri not in styled:
            style(s, cams[1]["lookat"])
            styled.add(ri)
        hdr = Image.new("RGB", (HEADER_W, CH), (245, 247, 250))
        dh = ImageDraw.Draw(hdr)
        dh.text((16, 14), d["L"]["label"], font=font, fill=(20, 35, 55))
        y = 60
        for line in d["L"]["numbers"]:
            dh.text((16, y), line, font=small, fill=(40, 50, 60))
            y += 32
        sheet.paste(hdr, (0, ri * (CH + 4)))
        for ci, cam in enumerate(cams):
            t = min(cam["frame_30hz"], len(d["q"]) - 1)
            img = render(s, cam, d["q"][t], d["o"][t])
            dr = ImageDraw.Draw(img, "RGBA")
            txt = f"{d['L']['label']} · t = {t / 30:.2f} s（{cam['column']}）"
            dr.rounded_rectangle((8, 8, 18 + dr.textlength(txt, font=small), 44), radius=8, fill=(255, 255, 255, 220), outline=(150, 160, 170, 255))
            dr.text((16, 12), txt, font=small, fill=(20, 35, 55))
            sheet.paste(img, (HEADER_W + ci * (CW + 4), ri * (CH + 4)))
    a.out.mkdir(parents=True, exist_ok=True)
    p = a.out / f"{a.task}_sizes_grid.jpg"
    qual = 90
    while True:
        sheet.save(p, quality=qual)
        if p.stat().st_size <= 1_000_000 or qual <= 40:
            break
        qual -= 5
    rec = dict(task=a.task, grid=p.name, bytes=p.stat().st_size, jpeg_quality=qual, cameras=cams,
               levels=[dict(label=L["label"], size=L["size"], failed=L["failed"], numbers=L["numbers"]) if L.get("failed") else
                       dict(label=L["label"], size=L["size"], pair=L["pair"], pair_sha256=sha(ROOT / L["pair"]), urdf=L["urdf"], urdf_sha256=sha(ROOT / L["urdf"]),
                            numbers=L["numbers"]) for L in levels], contact_frames_30hz=frames, helper_sha256=helper_sha,
               preset="g1_native_execution", rules=__doc__)
    print(json.dumps({k: rec[k] for k in ("grid", "bytes", "jpeg_quality")}), flush=True)
    for sc in scenes.values():
        sc.renderer.close()
    scenes.clear()
    # ------------------------------------------------------------------ video
    if a.video_out is not None:
        import imageio_ffmpeg
        s = scene_for(largest, VW, VH)
        pts = []
        for t in sorted(set(range(0, len(big["q"]), 10)) | {len(big["q"]) - 1}):
            s.set_pose(big["q"][t], big["o"][t])
            pts.append(f.bounds(s))
        vcam = f.camera_for(np.concatenate(pts), VW, VH, az_at(big, 0), EL)
        a.video_out.mkdir(parents=True, exist_ok=True)
        vp = a.video_out / f"{a.task}_sizes.mp4"
        if vp.exists():
            raise FileExistsError(vp)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", "30",
               "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(vp)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        nframes = 0
        try:
            for ri, d in enumerate(data):
                if d["q"] is None:
                    continue
                sv = scene_for(ri, VW, VH)
                if ("v", ri) not in styled:
                    style(sv, vcam["lookat"])
                    styled.add(("v", ri))
                for t in range(len(d["q"])):
                    img = render(sv, vcam, d["q"][t], d["o"][t])
                    dr = ImageDraw.Draw(img, "RGBA")
                    txt = f"{TASK_ZH.get(a.task, a.task)} · {d['L']['label']} · t = {t / 30:.2f} s"
                    dr.rounded_rectangle((12, 12, 24 + dr.textlength(txt, font=font), 54), radius=8, fill=(255, 255, 255, 220), outline=(150, 160, 170, 255))
                    dr.text((22, 16), txt, font=font, fill=(20, 35, 55))
                    proc.stdin.write(img.tobytes())
                    nframes += 1
        finally:
            proc.stdin.close()
            assert proc.wait() == 0
        vrec = dict(task=a.task, file=str(vp), frames=nframes, fps=30, sha256=sha(vp), bytes=vp.stat().st_size, camera=vcam,
                    levels=[L["label"] for L in levels if not L.get("failed")], rules=__doc__)
        (a.out / f"{a.task}_sizes_video.json").write_text(json.dumps(vrec, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({k: vrec[k] for k in ("file", "frames", "sha256")}), flush=True)
    (a.out / f"{a.task}_sizes_grid.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    for s in scenes.values():
        s.renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
