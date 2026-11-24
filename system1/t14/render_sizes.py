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

Close-up column (round 3, --closeup; default none = rounds 1-2 layout): one more column per level showing the hands' vertical
position. Frame: CORE4D (--closeup lift) the frame at which the object origin is highest; Push / Pull (--closeup mid) the
middle contact frame. View: each robot in profile, the camera looking along that robot's own lateral axis (root yaw + 90 deg),
elevation -10 deg, fitted to its two rubber-hand link origins +- 0.15 m, the other robot hidden; CORE4D: two half-width
panels (robot 0, robot 1); Push / Pull: robot 0 only (robot 1 is its copy / mirror). Distance, azimuth and elevation are
those of the largest level; the look-at point is that robot's hand centroid at each level's own frame.

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
    ap.add_argument("--closeup", choices=("none", "lift", "mid"), default="none")
    ap.add_argument("--axes-arrows", action="store_true", help="round 5: draw the palm normal (red) and finger direction (blue) of every hand "
                    "in the close-up panels (the hand metrics' vectors; robot 10 cm, person 10 cm / scale)")
    ap.add_argument("--human-panels", action="store_true", help="round 4: in the close-up column each robot panel is preceded by the person's hand "
                    "(human_render.py; same frame, same azimuth / elevation, distance / human-to-robot scale)")
    ap.add_argument("--grid-name", default=None, help="file stem of the grid (default <task>_sizes_grid)")
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
        s.robot_geoms = [[r["geom_id"] for r in recs if r["role"] == i] for i in (0, 1)]
        if s.recolor:
            for r in recs:
                if r["role"] == "object":
                    s.model.geom_rgba[r["geom_id"]] = OBJECT_RGBA

    def az_at(d, t):
        ax = d["q"][t, 1, :2] - d["q"][t, 0, :2]
        return float(np.degrees(np.arctan2(ax[1], ax[0]))) - 90.0 + AZ_OFF

    def render(scene, cam, q, o, opt=None, arrows=None):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["distance"], cam["azimuth"], cam["elevation"]
        scene.set_pose(q, o)
        if opt is None:
            scene.renderer.update_scene(scene.data, camera=c)
        else:
            scene.renderer.update_scene(scene.data, camera=c, scene_option=opt)
        if arrows:
            import human_render as _hr
            _hr.add_arrows(scene.renderer.scene, arrows)
        return Image.fromarray(scene.renderer.render())

    axes_robot = None
    if a.axes_arrows:
        sys.path.insert(0, str(ROOT / "system1/t14"))
        sys.path.insert(0, str(ROOT / "system1/t12v2"))
        import contact_gap as _cg
        import hand_landmarks as _hl
        import human_render as _hr
        _rob, _offs = _cg.cs.Robot("physx"), _cg.palm_offsets()

        def axes_robot(qpair, r):
            tl = _rob.fk(qpair[None, r, :3], qpair[None, r, [4, 5, 6, 3]], qpair[None, r, 7:])
            out = []
            for side in ("left", "right"):
                M = tl[f"{side}_rubber_hand_link"][0]
                pp = M[:3, :3] @ _offs[side] + M[:3, 3]
                rd = _hl.robot_directions(M[:3, :3], side)
                out += [(pp, pp + 0.10 * rd["palm_normal"], _hr.RED), (pp, pp + 0.10 * rd["finger"], _hr.BLUE)]
            return out

    # close-up column helpers: robot 0 geoms -> group 4, robot 1 -> group 5 (set in style for close-up grids)
    CU_EL, CU_MARGIN = -10.0, 0.15
    opt_show = mujoco.MjvOption()
    opt_show.geomgroup[4] = opt_show.geomgroup[5] = 1
    opt_only = [mujoco.MjvOption(), mujoco.MjvOption()]
    opt_only[0].geomgroup[4], opt_only[0].geomgroup[5] = 1, 0
    opt_only[1].geomgroup[4], opt_only[1].geomgroup[5] = 0, 1
    cu_robots = (0,) if a.task in ("push", "pull") else (0, 1)
    PW = 360 if a.human_panels else CW // len(cu_robots)
    CUW = 2 * PW * len(cu_robots) if a.human_panels else CW
    hs = None
    if a.human_panels:
        import human_render
        hs = human_render.HumanScene(a.task, ROOT / "logs_system1/t14/r4/human_meshes")

    def cu_frame(d):
        return int(np.argmax(d["o"][:, 2])) if a.closeup == "lift" else min(frames[1], len(d["q"]) - 1)

    def cu_hands(s, d, t, r):
        s.set_pose(d["q"][t], d["o"][t])
        return np.array([s.hand_center(r, side) for side in ("left", "right")])

    def cu_az(d, t, r):
        w, x, y, z = d["q"][t, r, 3:7]
        fwd = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w)])  # root x axis, horizontal part
        return float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + 90.0

    # ------------------------------------------------------------------ grid
    cams = []
    big = data[largest]
    for name, ci in COLS:
        t = min(frames[ci], len(big["q"]) - 1)
        s = scene_for(largest, CW, CH)
        s.set_pose(big["q"][t], big["o"][t])
        cam = f.camera_for(f.bounds(s), CW, CH, az_at(big, t), EL)
        cams.append(dict(column=name, frame_30hz=int(t), time_s=round(t / 30, 3), **cam))
    ncol = 3
    cu_cam = None
    if a.closeup != "none":
        ncol = 4
        s = scene_for(largest, PW, CH)
        t = cu_frame(big)
        box = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * CU_MARGIN
        cu_cam = {}
        for r in cu_robots:
            H = cu_hands(s, big, t, r)
            pts = (H[:, None, :] + box[None]).reshape(-1, 3)
            cu_cam[r] = dict(column="手部特写", rule=a.closeup, robot=r, **f.camera_for(pts, PW, CH, cu_az(big, t, r), CU_EL))
    styled = set()
    W = HEADER_W + 3 * CW + (CUW if ncol == 4 else 0) + (ncol - 1) * 4
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
            for ci in range(ncol):
                ph = Image.new("RGB", (CW, CH), (232, 234, 238))
                dp = ImageDraw.Draw(ph)
                dp.text((24, CH // 2 - 20), f"{d['L']['label']}：求解失败，无双人参考", font=font, fill=(120, 40, 30))
                sheet.paste(ph, (HEADER_W + ci * (CW + 4), ri * (CH + 4)))
            continue
        s = scene_for(ri, CW, CH)
        if ri not in styled:
            style(s, cams[1]["lookat"])
            styled.add(ri)
            if a.closeup != "none":  # robot r -> geom group 4 + r: both shown in the three columns, one at a time in the close-up
                for r, gs in enumerate(s.robot_geoms):
                    for g in gs:
                        s.model.geom_group[g] = 4 + r
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
            img = render(s, cam, d["q"][t], d["o"][t], opt_show if a.closeup != "none" else None)
            dr = ImageDraw.Draw(img, "RGBA")
            txt = f"{d['L']['label']} · t = {t / 30:.2f} s（{cam['column']}）"
            dr.rounded_rectangle((8, 8, 18 + dr.textlength(txt, font=small), 44), radius=8, fill=(255, 255, 255, 220), outline=(150, 160, 170, 255))
            dr.text((16, 12), txt, font=small, fill=(20, 35, 55))
            sheet.paste(img, (HEADER_W + ci * (CW + 4), ri * (CH + 4)))
        if cu_cam is not None:
            t = cu_frame(d)
            sp = scene_for(ri, PW, CH)
            if ("cu", ri) not in styled:
                style(sp, cams[1]["lookat"])
                for r, gs in enumerate(sp.robot_geoms):
                    for g in gs:
                        sp.model.geom_group[g] = 4 + r
                styled.add(("cu", ri))
            cell = Image.new("RGB", (CUW, CH), "white")
            d["closeup"] = dict(frame_30hz=t)
            what = "物体最高" if a.closeup == "lift" else "中段"
            if hs is not None:
                if a.task in ("push", "pull"):
                    scales = [float(np.load(d["L"]["values"]["single"])["smpl_scale"])]
                else:
                    scales = [float(x) for x in np.load(ROOT / d["L"]["pair"])["human_to_robot_scales"]]
                d["closeup"]["human_to_robot_scales"] = scales
            for pi, r in enumerate(cu_robots):
                H = cu_hands(sp, d, t, r)
                cam = dict(cu_cam[r], lookat=H.mean(0).tolist())
                img = render(sp, cam, d["q"][t], d["o"][t], opt_only[r], arrows=axes_robot(d["q"][t], r) if axes_robot else None)
                dr = ImageDraw.Draw(img, "RGBA")
                txt = f"机器人 {r}" if len(cu_robots) > 1 else "只画机器人 0"
                dr.rounded_rectangle((8, CH - 44, 18 + dr.textlength(txt, font=small), CH - 8), radius=8, fill=(255, 255, 255, 220), outline=(150, 160, 170, 255))
                dr.text((16, CH - 40), txt, font=small, fill=(20, 35, 55))
                if hs is not None:
                    person = r if a.task in ("table", "bin", "chair") else 0
                    harrows = None
                    if axes_robot:
                        hp_, hd_ = hs.axes(person, t)
                        L_ = 0.10 / scales[person]
                        harrows = [(hp_[i], hp_[i] + L_ * hd_["palm_normal"][i], _hr.RED) for i in range(2)] + [(hp_[i], hp_[i] + L_ * hd_["finger"][i], _hr.BLUE) for i in range(2)]
                    himg = hs.render(person, t, cu_cam[r]["azimuth"], cu_cam[r]["elevation"], cu_cam[r]["distance"] / scales[person], PW, CH, arrows=harrows)
                    dh2 = ImageDraw.Draw(himg, "RGBA")
                    htxt = f"人 {person + 1}" if a.task in ("table", "bin", "chair") else "人（源）"
                    dh2.rounded_rectangle((8, CH - 44, 18 + dh2.textlength(htxt, font=small), CH - 8), radius=8, fill=(255, 255, 255, 220), outline=(150, 160, 170, 255))
                    dh2.text((16, CH - 40), htxt, font=small, fill=(20, 35, 55))
                    cell.paste(himg, (2 * pi * PW, 0))
                    cell.paste(img, ((2 * pi + 1) * PW, 0))
                else:
                    cell.paste(img, (pi * PW, 0))
                d["closeup"][f"lookat_r{r}"] = [round(x, 4) for x in cam["lookat"]]
            dr = ImageDraw.Draw(cell, "RGBA")
            for xline in range(1, CUW // PW):
                dr.line((xline * PW, 0, xline * PW, CH), fill=(255, 255, 255), width=4 if (hs is None or xline % 2 == 0) else 2)
            txt = f"{d['L']['label']} · t = {t / 30:.2f} s（手部特写，{what}）"
            dr.rounded_rectangle((8, 8, 18 + dr.textlength(txt, font=small), 44), radius=8, fill=(255, 255, 255, 220), outline=(150, 160, 170, 255))
            dr.text((16, 12), txt, font=small, fill=(20, 35, 55))
            sheet.paste(cell, (HEADER_W + 3 * (CW + 4), ri * (CH + 4)))
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.grid_name or f"{a.task}_sizes_grid"
    p = a.out / f"{stem}.jpg"
    qual = 90
    while True:
        sheet.save(p, quality=qual)
        if p.stat().st_size <= 1_000_000 or qual <= 40:
            break
        qual -= 5
    rec = dict(task=a.task, grid=p.name, bytes=p.stat().st_size, jpeg_quality=qual, cameras=cams, closeup_camera=None if cu_cam is None else {f"r{r}": c for r, c in cu_cam.items()},
               closeup_frames={d["L"]["label"]: d.get("closeup") for d in data},
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
        (a.out / f"{stem}_video.json").write_text(json.dumps(vrec, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({k: vrec[k] for k in ("file", "frames", "sha256")}), flush=True)
    (a.out / f"{stem}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    for s in scenes.values():
        s.renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
