#!/usr/bin/env python3
"""T14 step 7, part 3 item 1: pictures of the screen's typical picks (screen.json "picks", 3 per class): the OMOMO person
(SMPL-X mesh from the stored parameters, flat hands, skin colour) and the captured object (light blue, captured mesh at its
stored pose), chequered floor at the person's lowest vertex of the sequence. Read only, CPU (MuJoCo EGL, software GL).

Camera (fixed before rendering): side view, azimuth = the episode's mean heading + 90 deg + 30 deg (oblique from the person's
right-front), elevation -18 deg, look-at = the midpoint of the pelvis joint and the object centre (Gaussian-smoothed over time,
sigma 10 frames, for the video), distance 3.0 m (class a, walking) / 2.4 m (b, c).
  * key-frame sheet per class: one row per pick, 5 frames at 0 / 25 / 50 / 75 / 100 % of the episode, 400 x 300 px each,
    caption = sequence, frame, time, lift; jpg <= 1 MB;
  * video per pick: the whole sequence, 30 fps, 640 x 480, H.264 (logs only; Drive).

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      sonic_s1/bin/python -B system1/t14/objects/render_screen.py --screen results_system1/T14_retargeting/step7/screen/screen.json \
      --out results_system1/T14_retargeting/step7/screen --video-out exports_system1/T14_videos/step7/screen
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omomo_motion as om  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CLASS_ZH = dict(a="a 抱起往前走", b="b 推/拉/拖", c="c 原地抬起放下")
SKIN, OBJ = "0.80 0.70 0.62 1", "0.62 0.80 0.95 1"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(size):
    for p in (CJK, FONT):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def smplx_vertices(v):
    import smplx
    import torch
    T = len(v["trans"])
    m = smplx.create(str(om.SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        o = m(betas=torch.tensor(np.repeat(v["betas"], T, 0), dtype=torch.float32), global_orient=torch.tensor(v["root_orient"], dtype=torch.float32),
              body_pose=torch.tensor(v["pose_body"], dtype=torch.float32), transl=torch.tensor(v["trans"], dtype=torch.float32))
    return o.vertices.numpy().astype(np.float64), m.faces.astype(np.int64)


class Seq:
    def __init__(self, v, name):
        self.v, self.name = v, name
        self.V, self.F = smplx_vertices(v)
        self.J = om.smplx_joints(v)
        self.ob = om.object_track(v, name)
        self.mesh = trimesh.load(om.OMOMO / f"captured_objects/{name}_cleaned_simplified.obj", force="mesh", process=True)
        self.floor = float(self.V[..., 2].min())
        bottom = self.ob["bottom"]
        self.lift = bottom - bottom.min()

    def object_at(self, t):
        m = self.mesh.copy()
        T = np.eye(4)
        T[:3, :3] = self.ob["R"][t] * self.ob["s"][t]
        T[:3, 3] = self.ob["t"][t]
        m.apply_transform(T)
        return m


def render_frame(seq, t, lookat, az, dist, W, H, tmpdir):
    import mujoco
    person = trimesh.Trimesh(seq.V[t], seq.F, process=False)
    render_frame.n = getattr(render_frame, "n", 0) + 1  # unique file names: MuJoCo caches mesh assets by file name
    pp, po = Path(tmpdir) / f"p{render_frame.n}.obj", Path(tmpdir) / f"o{render_frame.n}.obj"
    person.export(pp)
    seq.object_at(t).export(po)
    xml = f"""<mujoco><visual><global offwidth="{W}" offheight="{H}"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/><quality shadowsize="2048"/></visual>
    <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.82" rgb2="0.62 0.62 0.62" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="16 16"/><mesh name="p" file="{pp}" inertia="shell"/><mesh name="o" file="{po}" inertia="shell"/></asset>
    <worldbody><light pos="{lookat[0]} {lookat[1]} {seq.floor + 4}" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom type="plane" size="8 8 0.01" pos="{lookat[0]} {lookat[1]} {seq.floor}" material="grid" contype="0" conaffinity="0"/>
    <geom type="mesh" mesh="p" rgba="{SKIN}" contype="0" conaffinity="0"/><geom type="mesh" mesh="o" rgba="{OBJ}" contype="0" conaffinity="0"/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, H, W) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance = dist
        cam.azimuth = az
        cam.elevation = -18
        r.update_scene(data, cam)
        img = r.render().copy()
    pp.unlink()  # our own temporary files from this call
    po.unlink()
    return img


def caption(img, lines, size=15):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    f = font(size)
    h = (size + 6) * len(lines) + 6
    d.rectangle([0, 0, im.width, h], fill=(255, 255, 255, 220))
    for i, t in enumerate(lines):
        d.text((6, 3 + (size + 6) * i), t, fill=(0, 0, 0), font=f)
    return np.asarray(im)


def view(seq, ep, cls):
    s, e = ep["start"], ep["stop"]
    hd = om.heading(seq.J)[s:e, :2].sum(0)
    hd /= np.linalg.norm(hd)
    az = float(np.degrees(np.arctan2(hd[1], hd[0]))) + 90 + 30
    mid = (seq.J[:, 0] + seq.ob["centre"]) / 2
    return az, mid, 3.0 if cls == "a" else 2.4


def save_jpg(arr, path, limit=1_000_000):
    im = Image.fromarray(arr)
    q = 90
    while True:
        im.save(path, quality=q)
        if path.stat().st_size <= limit or q <= 40:
            return q
        q -= 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-out", type=Path, default=None)
    ap.add_argument("--classes", default="abc")
    a = ap.parse_args()
    import imageio_ffmpeg
    sc = json.loads(a.screen.read_text())
    want = {p["sequence"] for k in a.classes for p in sc["picks"][k]}
    seqs = {k: v for k, v in om.load_sequences({s.split("_")[1] for s in want}).items() if k in want}
    rec = dict(rules=__doc__, sheets={}, videos={})
    a.out.mkdir(parents=True, exist_ok=True)
    for cls in a.classes:
        rows = []
        for p in sc["picks"][cls]:
            name = p["sequence"]
            seq = Seq(seqs[name], name.split("_")[1])
            az, mid, dist = view(seq, p, cls)
            s, e = p["start"], p["stop"]
            frames = [int(round(s + f * (e - 1 - s))) for f in (0, 0.25, 0.5, 0.75, 1.0)]
            tiles = []
            with tempfile.TemporaryDirectory(dir=ROOT / "logs_system1/t14") as tmp:
                lookat = mid[s:e].mean(0)
                for t in frames:
                    img = render_frame(seq, t, lookat, az, dist, 400, 300, tmp)
                    tiles.append(caption(img, [f"{name}  帧 {t}  t = {t / 30:.2f} s", f"抬高 {100 * seq.lift[t]:.1f} cm"], 13))
            row = np.concatenate(tiles, 1)
            extra = (f"抓握 {p['grip']}，步数 {p['steps']}，骨盆净位移 {p['pelvis_net_m']:.2f} m" if cls == "a" else
                     f"物体移动 {p['travel_m']:.2f} m，{p['push_pull']}，两手同侧 {p['same_side']}（{p['left_face']} / {p['right_face']}）" if cls == "b" else
                     f"抓握 {p['grip']}，抬高峰值 {100 * p['peak_lift_m']:.1f} cm，骨盆离起点最远 {p['pelvis_max_from_start_m']:.2f} m")
            head = caption(np.full((28, row.shape[1], 3), 255, np.uint8), [f"{CLASS_ZH[cls]} · {name} · 帧 {s}–{e - 1} · {extra}"], 15)
            rows.append(np.concatenate([head, row], 0))
            if a.video_out is not None:
                a.video_out.mkdir(parents=True, exist_ok=True)
                vp = a.video_out / f"class_{cls}_{name}.mp4"
            if a.video_out is not None and vp.exists():  # finished by an earlier run (resume)
                rec["videos"][name] = dict(file=str(vp), frames=len(seq.V), bytes=vp.stat().st_size, resumed=True)
            elif a.video_out is not None:
                look = gaussian_filter1d(mid, 10, axis=0, mode="nearest")
                cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "30",
                       "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "22", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(vp)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                with tempfile.TemporaryDirectory(dir=ROOT / "logs_system1/t14") as tmp:
                    for t in range(len(seq.V)):
                        img = render_frame(seq, t, look[t], az, dist, 640, 480, tmp)
                        tag = "  [片段内]" if s <= t < e else ""
                        img = caption(img, [f"{CLASS_ZH[cls]} · {name}", f"帧 {t}  t = {t / 30:.2f} s  抬高 {100 * seq.lift[t]:.1f} cm{tag}"], 15)
                        proc.stdin.write(np.ascontiguousarray(img[..., :3]).tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed {vp}")
                rec["videos"][name] = dict(file=str(vp), frames=len(seq.V), bytes=vp.stat().st_size)
            print(cls, name, flush=True)
        sheet = np.concatenate(rows, 0)
        path = a.out / f"class_{cls}_keyframes.jpg"
        q = save_jpg(sheet, path)
        rec["sheets"][cls] = dict(file=path.name, bytes=path.stat().st_size, jpeg_quality=q, picks=[p["sequence"] for p in sc["picks"][cls]])
    (a.out / "render_screen.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
