#!/usr/bin/env python3
"""T14 step 7, layer 1 pictures (round-6 style, Fig. 4 look: U5 g1_native_execution preset, Unitree grey robots, light-blue
object, chequered floor). Render only, CPU (MuJoCo EGL).

Frames: contact start / middle / end = first, middle, last frame in which a hand of the person touches the object (SMPL-X hand
vertex within 1.5 cm of the captured mesh), 30 Hz.
  * side-by-side sheet (<tag>_sidebyside.jpg): one row per frame; columns = the person with the captured object (SMPL-X mesh,
    skin colour, original OMOMO world), robot 0 with the box (level 1.0), both robots with the box (level 1.0). Close-up
    views: look-at = mean of pelvis, neck and the two palm points, azimuth = that body's heading + 90 + 30 deg, elevation
    -15 deg, distance 1.6 m (robot) / 1.6 m / smpl_scale (person); arrows: palm normal (red) and finger direction (blue), 10 cm
    (robot) and 10 cm / smpl_scale (person). Full view of both robots: U5 camera_for fitted to the robots and box, azimuth =
    robot-0 -> robot-1 direction - 90 + 30 deg, elevation -25 deg.
  * level grid (<tag>_levels_grid.jpg): one row per reported level (a, 1.0, 1.25, 1.5, plus the first level without
    intersection when 1.0 already intersects), columns = the three frames, one camera per column fitted to the largest level;
    a header per row with the level's numbers.
  * video (<tag>_levels.mp4, logs only, Drive): left the person, right both robots; the whole sequence once per level.

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      sonic_s1/bin/python -B system1/t14/objects/render_box.py --solve <solve> --levels <levels.json> --out <dir> --video-out <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("render_nominal_video", ROOT / "system1/render/u5/render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
OBJECT_RGBA = (0.76, 0.80, 0.86, 1.0)
SKIN, OBJ = "0.80 0.70 0.62 1", "0.76 0.80 0.86 1"
PW, PH = 480, 400
RED, BLUE = (0.9, 0.1, 0.1, 1.0), (0.1, 0.3, 0.95, 1.0)
CLASS_ZH = dict(a="抱着往前走（一前一后）", b="推/拉/拖", c="原地抬起放下")


def save_jpg(img, path, limit=1_500_000):
    q = 90
    while True:
        img.save(path, quality=q)
        if path.stat().st_size <= limit or q <= 40:
            return q
        q -= 6


class PersonScene:
    def __init__(self, rec):
        import box_assets as ba
        import omomo_motion as om
        import single_shared as ss
        import trimesh
        from render_screen import smplx_vertices
        name, seq = rec["object"], rec["sequence"]
        self.v = om.load_sequences({name})[seq]
        self.V, self.F = smplx_vertices(self.v)
        self.J = om.smplx_joints(self.v)
        self.ob = om.object_track(self.v, name)
        self.mesh = ba.capture_mesh(name)
        self.floor = float(self.V[..., 2].min())
        palms, fit = ss.omomo_palms(seq)
        Rf, tf = np.asarray(fit["rotation"]), np.asarray(fit["translation"])
        self.palms = (palms - tf) @ Rf  # back to the original world
        import hand_landmarks as hl
        Jn, tips = ss.omomo_palms.last["joints_omomo_new"], ss.omomo_palms.last["tips_omomo_new"]
        tfm = hl.template_frames()
        self.dirs = {}
        for side in ("left", "right"):
            Vt = np.zeros((len(Jn), max(hl.TIP_VERTEX.values()) + 1, 3))
            Vt[:, hl.TIP_VERTEX[side]] = tips[side]
            d = hl.human_directions(Jn, Vt, side, tfm)
            self.dirs[side] = {k: d[k] @ Rf for k in ("palm_normal", "finger")}
        self.scale = float(rec["smpl_scale"])
        self.trimesh = trimesh
        self.n = 0

    def heading(self, t):
        import omomo_motion as om
        h = om.heading(self.J[t:t + 1])[0]
        return float(np.degrees(np.arctan2(h[1], h[0])))

    def render(self, t, lookat, az, dist, el, W, H, arrows=True):
        import mujoco
        from human_render import add_arrows
        m = self.mesh.copy()
        T = np.eye(4)
        T[:3, :3] = self.ob["R"][t] * self.ob["s"][t]
        T[:3, 3] = self.ob["t"][t]
        m.apply_transform(T)
        self.n += 1
        with tempfile.TemporaryDirectory(dir=ROOT / "logs_system1/t14") as tmp:
            pp, po = Path(tmp) / f"p{self.n}.obj", Path(tmp) / f"o{self.n}.obj"  # unique names: MuJoCo caches mesh files by name
            self.trimesh.Trimesh(self.V[t], self.F, process=False).export(pp)
            m.export(po)
            xml = f"""<mujoco><visual><global offwidth="{W}" offheight="{H}"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/><quality shadowsize="2048"/></visual>
            <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.82" rgb2="0.62 0.62 0.62" width="512" height="512"/>
            <material name="grid" texture="grid" texrepeat="16 16"/><mesh name="p" file="{pp}" inertia="shell"/><mesh name="o" file="{po}" inertia="shell"/></asset>
            <worldbody><light pos="{lookat[0]} {lookat[1]} {self.floor + 4}" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
            <geom type="plane" size="8 8 0.01" pos="{lookat[0]} {lookat[1]} {self.floor}" material="grid" contype="0" conaffinity="0"/>
            <geom type="mesh" mesh="p" rgba="{SKIN}" contype="0" conaffinity="0"/><geom type="mesh" mesh="o" rgba="{OBJ}" contype="0" conaffinity="0"/></worldbody></mujoco>"""
            model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, H, W) as r:
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
            r.update_scene(data, cam)
            if arrows:
                L = 0.10 / self.scale
                arr = []
                for hi, side in enumerate(("left", "right")):
                    p = self.palms[t, hi]
                    arr += [(p, p + L * self.dirs[side]["palm_normal"][t], RED), (p, p + L * self.dirs[side]["finger"][t], BLUE)]
                add_arrows(r.scene, arr, width=0.006 / self.scale)
            return Image.fromarray(r.render().copy())

    def lookat(self, t):
        return (self.J[t, 0] + self.J[t, 12] + self.palms[t, 0] + self.palms[t, 1]) / 4  # pelvis, neck, two palms


def caption(img, lines, font):
    d = ImageDraw.Draw(img, "RGBA")
    h = 26 * len(lines) + 6
    d.rectangle([0, 0, img.width, h], fill=(255, 255, 255, 215))
    for i, s in enumerate(lines):
        d.text((8, 3 + 26 * i), s, fill=(0, 0, 0), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", type=Path, required=True)
    ap.add_argument("--levels", type=Path, required=True)
    ap.add_argument("--cls", required=True, choices=("a", "b", "c"))
    ap.add_argument("--clip-start", type=int, default=0, help="step 9 clips: solve frame 0 = sequence frame clip-start (the person is drawn there)")
    ap.add_argument("--side-level", type=float, default=1.0, help="level of the side-by-side sheet (step 9: the lowest solved level when 1.0 has no solve)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-out", type=Path, default=None)
    a = ap.parse_args()
    import mujoco
    import contact_gap as cg
    import contact_stats as cs
    import hand_landmarks as hl
    from human_render import add_arrows
    rec = json.loads((a.solve / "run_record.json").read_text())
    lv = json.loads(a.levels.read_text())
    tag, name = rec["tag"], rec["object"]
    f, _ = rnv.helper()
    font, small = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 17)
    flags = np.load(a.solve / "human_contact.npz")["h_sd"] >= -0.015
    cf = np.flatnonzero(flags.any(1))
    frames = [int(cf[0]), int(cf[len(cf) // 2]), int(cf[-1])]
    show = [x for x in lv["report_levels"]] + ([lv["first_free_level_above_1"]] if lv.get("first_free_level_above_1") else [])
    L = {m: lv["levels"][f"{m:.2f}"] for m in show}
    data = {m: (np.load(L[m]["pair"])["robot_qpos"].astype(np.float64), np.load(L[m]["pair"])["object_qpos"].astype(np.float64)) for m in show}
    scenes = {}

    def scene_for(m, w, h):
        if (m, w, h) not in scenes:
            s = f.newscene(ROBOT_URDF, Path(L[m]["urdf"]), 2, "g1_native_execution", w, h)
            recs = f.styling(s, data[m][1][0, :3])
            for r in recs:
                if r["role"] == "object":
                    s.model.geom_rgba[r["geom_id"]] = OBJECT_RGBA
            scenes[(m, w, h)] = s
        return scenes[(m, w, h)]

    def render_robot(s, cam, q, o, arrows=None, hide_robot1=False):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["distance"], cam["azimuth"], cam["elevation"]
        q = q.copy()
        if hide_robot1:
            q[1, :3] += np.array([0.0, 0.0, -50.0])  # robot 1 moved out of view (below the floor) for the robot-0 close-up
        s.set_pose(q, o)
        s.renderer.update_scene(s.data, camera=c)
        if arrows:
            add_arrows(s.renderer.scene, arrows)
        return Image.fromarray(s.renderer.render())

    robot = cs.Robot("physx")
    offs = cg.palm_offsets()
    person = PersonScene(rec)

    def robot_arrows(q, t):
        tl = robot.fk(q[t:t + 1, :1, :3], np.concatenate([q[t:t + 1, :1, 4:7], q[t:t + 1, :1, 3:4]], -1), q[t:t + 1, :1, 7:])
        arr = []
        for side in ("left", "right"):
            M = tl[f"{side}_rubber_hand_link"][0, 0]
            p = M[:3, :3] @ offs[side] + M[:3, 3]
            d = hl.robot_directions(M[:3, :3], side)
            arr += [(p, p + 0.10 * d["palm_normal"], RED), (p, p + 0.10 * d["finger"], BLUE)]
        return arr

    def robot_close_cam(q, t):
        tl = robot.fk(q[t:t + 1, :1, :3], np.concatenate([q[t:t + 1, :1, 4:7], q[t:t + 1, :1, 3:4]], -1), q[t:t + 1, :1, 7:])
        neck = tl["torso_link"][0, 0][:3, :3] @ np.array([0.003, -0.0005, 0.290]) + tl["torso_link"][0, 0][:3, 3]
        palms = [tl[f"{s}_rubber_hand_link"][0, 0][:3, :3] @ offs[s] + tl[f"{s}_rubber_hand_link"][0, 0][:3, 3] for s in ("left", "right")]
        look = (q[t, 0, :3] + neck + palms[0] + palms[1]) / 4
        w, x, y, z = q[t, 0, 3:7]
        yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return dict(lookat=look, distance=1.6, azimuth=yaw + 90 + 30, elevation=-15)

    # ---------------------------------------------------------------- side-by-side sheet (level 1.0)
    sl = a.side_level
    q1, o1 = data[sl]
    rows = []
    s_close = scene_for(sl, PW, PH)
    s_full = scene_for(sl, PW + 120, PH)
    off = a.clip_start
    for t in frames:
        ph = person.render(t + off, person.lookat(t + off), person.heading(t + off) + 90 + 30, 1.6 / person.scale, -15, PW, PH)
        ph = caption(ph, [f"人 · 帧 {t}（{t / 30:.2f} s）"], small)
        cam = robot_close_cam(q1, t)
        rc = caption(render_robot(s_close, cam, q1[t], o1[t], robot_arrows(q1, t), hide_robot1=True), [f"机器人 0 · {sl:.2f} 倍 · 帧 {t}"], small)
        s_full.set_pose(q1[t], o1[t])
        fc = f.camera_for(f.bounds(s_full), PW + 120, PH, float(np.degrees(np.arctan2(*(q1[t, 1, 1::-1] - q1[t, 0, 1::-1])))) - 90 + 30, -25)
        fu = caption(render_robot(s_full, fc, q1[t], o1[t]), [f"两台 · {sl:.2f} 倍 · 帧 {t}"], small)
        row = Image.new("RGB", (PW * 2 + PW + 120, PH), "white")
        for i, im in enumerate((ph, rc, fu)):
            row.paste(im.convert("RGB"), (i * PW, 0))
        rows.append(row)
    l1 = L[sl]
    ce = l1["contact_position_error_3d"]
    head = Image.new("RGB", (rows[0].width, 60), "white")
    d = ImageDraw.Draw(head)
    clip = f" · 片段 {off}–{off + len(q1) - 1}（帧号从片段起点算）" if off else ""
    d.text((10, 4), f"{CLASS_ZH[a.cls]} · {rec['sequence']}{clip} · 盒子 {' × '.join(f'{x:.2f}' for x in lv['source_box_m'])} m · 角色检查 {lv['role_status']}", fill=(0, 0, 0), font=font)
    d.text((10, 32), f"{sl:.2f} 倍：接触保持率 {l1['contact_retention']['pooled']}，最大穿透 {max(l1['max_penetration_cm'].values()):.2f} cm，两机最小距离 {l1['robot_robot_min_distance_cm']:.1f} cm，"
                     f"三维接触误差 r0 {ce['r0']['distance']['mean_cm'] if ce['r0']['distance'] else '—'} cm · 红 = 手心法向，蓝 = 手指方向", fill=(0, 0, 0), font=small)
    sheet = Image.new("RGB", (rows[0].width, 60 + PH * len(rows)), "white")
    sheet.paste(head, (0, 0))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, 60 + PH * i))
    a.out.mkdir(parents=True, exist_ok=True)
    q_side = save_jpg(sheet, a.out / f"{tag}_sidebyside.jpg")
    # ---------------------------------------------------------------- level grid
    big = max(show, key=lambda m: L[m]["length_along_axis_m"])
    GW, GH, HW = 560, 400, 300
    cams = []
    for t in frames:
        s = scene_for(big, GW, GH)
        s.set_pose(data[big][0][t], data[big][1][t])
        qb = data[big][0]
        az = float(np.degrees(np.arctan2(qb[t, 1, 1] - qb[t, 0, 1], qb[t, 1, 0] - qb[t, 0, 0]))) - 90 + 30
        cams.append(f.camera_for(f.bounds(s), GW, GH, az, -25))
    grid = Image.new("RGB", (HW + GW * 3, GH * len(show)), "white")
    for i, m in enumerate(show):
        x = L[m]
        hd = Image.new("RGB", (HW, GH), "white")
        dd = ImageDraw.Draw(hd)
        lab = f"{m:.2f} 倍" + ("（a）" if m == lv["a"] else "") + ("（1.0 以上第一个不相交档）" if m == lv.get("first_free_level_above_1") else "")
        lines = [lab, f"盒长 {x['length_along_axis_m']:.2f} m", f"保持率 {x['contact_retention']['pooled']}", f"最大穿透 {max(x['max_penetration_cm'].values()):.2f} cm",
                 f"两机最小距离 {x['robot_robot_min_distance_cm']:.1f} cm", f"相交帧 {x['robot_robot_intersect_frames']}",
                 f"脚漂移最大 r0 {max((v['drift_max_cm'] or 0) for k, v in x['foot_slide'].items() if k.startswith('r0')):.1f} / r1 {max((v['drift_max_cm'] or 0) for k, v in x['foot_slide'].items() if k.startswith('r1')):.1f} cm"]
        for j, s_ in enumerate(lines):
            dd.text((10, 10 + 32 * j), s_, fill=(0, 0, 0), font=font if j == 0 else small)
        grid.paste(hd, (0, GH * i))
        s = scene_for(m, GW, GH)
        for j, t in enumerate(frames):
            im = caption(render_robot(s, cams[j], data[m][0][t], data[m][1][t]), [f"{m:.2f} 倍 · 帧 {t}（{t / 30:.2f} s）"], small)
            grid.paste(im.convert("RGB"), (HW + GW * j, GH * i))
    q_grid = save_jpg(grid, a.out / f"{tag}_levels_grid.jpg")
    rec_out = dict(rules=__doc__, frames=frames, levels=show, sidebyside=dict(file=f"{tag}_sidebyside.jpg", jpeg_quality=q_side),
                   grid=dict(file=f"{tag}_levels_grid.jpg", jpeg_quality=q_grid))
    # ---------------------------------------------------------------- video
    if a.video_out is not None:
        import imageio_ffmpeg
        a.video_out.mkdir(parents=True, exist_ok=True)
        vp = a.video_out / f"{tag}_levels.mp4"
        VW, VH = 640, 480
        sv = {m: scene_for(m, VW, VH) for m in show}
        pts = []
        for t in range(0, len(data[big][0]), 10):
            sv[big].set_pose(data[big][0][t], data[big][1][t])
            pts.append(f.bounds(sv[big]))
        qb = data[big][0]
        vcam = f.camera_for(np.concatenate(pts), VW, VH, float(np.degrees(np.arctan2(qb[0, 1, 1] - qb[0, 0, 1], qb[0, 1, 0] - qb[0, 0, 0]))) - 90 + 30, -25)
        from scipy.ndimage import gaussian_filter1d
        plook = gaussian_filter1d(np.array([person.lookat(t) for t in range(len(person.V))]), 10, axis=0, mode="nearest")
        paz = person.heading(frames[1] + off) + 90 + 30
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * VW}x{VH}", "-r", "30",
               "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "22", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(vp)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        pcache = {}
        for m in show:
            q, o = data[m]
            for t in range(len(q)):
                if t not in pcache:
                    pcache[t] = caption(person.render(t + off, plook[t + off], paz, 2.6, -18, VW, VH, arrows=False), [f"人 · {rec['sequence']} · {t / 30:.2f} s"], small).convert("RGB")
                rimg = caption(render_robot(sv[m], vcam, q[t], o[t]), [f"两台机器人 · {m:.2f} 倍 · {t / 30:.2f} s"], small).convert("RGB")
                frame = Image.new("RGB", (2 * VW, VH))
                frame.paste(pcache[t], (0, 0))
                frame.paste(rimg, (VW, 0))
                proc.stdin.write(np.asarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        rec_out["video"] = dict(file=str(vp), bytes=vp.stat().st_size, levels=show)
    (a.out / f"{tag}_render.json").write_text(json.dumps(rec_out, indent=1, ensure_ascii=False) + "\n")
    for s in scenes.values():
        s.renderer.close()
    print("RENDER_DONE", tag, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
