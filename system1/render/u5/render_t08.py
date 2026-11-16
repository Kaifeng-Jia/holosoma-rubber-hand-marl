#!/usr/bin/env python3
"""T08 slide assets D7 / D8 (docs/system1/T08_slides_assets_zh.md): GIF + mid-episode PNG of saved physics trajectories in the
U5 / paper Fig. 4 style. Same scene, materials and helper as render_nominal_video.py (imported from it): Unitree mesh colours
(preset 'g1_native_execution'), light-blue object (the URDF material), checker ground, azimuth 135, elevation -22, fovy 45,
rendered at 1920x1080. Kinematic replay of saved states only: no physics step, no policy, no interpolation, no pose change.

GIF: width 960 (1920x1080 frame downscaled, Lanczos), 20 fps, loop, whole episode:
  * episode longer than 8 s -> 2x speed, "2x speed" top right;
  * shorter than 4 s -> 1x, last frame held to 4 s, the held frames labelled "episode ended" top right;
  * GIF frame k shows saved row floor(k * speed * 50 / 20); the last saved row is always included.
PNG: the middle saved row floor((n - 1) / 2), 1920x1080, same camera.

  --d7  single-robot frozen A1 push policy at the two ends of the 1.4 m table (Stage 1B retention check, 2026-08-15):
        the recordings plan5_preflight_v1/formal_wide_{right,left}_trajectory.npz (read only), attempt chosen by the rule of
        the task book (smallest-index attempt whose outcome equals the end's majority). One camera for both: lookat x, y =
        centre of the table's bounding box over both rendered episodes, z = middle of the height range of every robot and table
        mesh bound of both episodes; distance by the U5 camera_for rule (7 % border) over those bounds.
  --d8  scheme A on Pull +/-2 perturbed_000 (T01b main), camera = the U5 F3 Pull camera (sources_snapshot/f3_manifest_pull.json
        records[0], the same settings the U5 Pull video used).

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl sonic_s1/bin/python -B system1/render/u5/render_t08.py --d7 --d8 --out results_system1/T08_slides
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

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("render_nominal_video", HERE / "render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)
ROOT = rnv.ROOT
SIM_HZ, GIF_FPS, GIF_W = 50, 20, 960
MIN_S, MAX_S = 4.0, 8.0
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
MARL = Path("/home/kevin/holosoma-rubber-hand-marl")
D7_DIR = MARL / "logs/WholeBodyTracking/plan5_preflight_v1"
D7_CKPT = MARL / "logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt"
D7_TABLE = ROOT / "system1/t08/assets/objects_widetable_a1_retention.urdf"  # git show d04b312a:<path> of holosoma-rubber-hand-marl
D8_A = ROOT / "logs_system1/T01b/schemeA_main/pull_pm2"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def playback_plan(n_rows: int, sim_hz: int = SIM_HZ, fps: int = GIF_FPS):
    """Rows shown by each GIF frame, playback speed, number of held (labelled) frames. Whole episode, never cut."""
    dur = n_rows / sim_hz
    speed = 2 if dur > MAX_S else 1
    step = speed * sim_hz / fps
    rows = []
    k = 0
    while int(np.floor(k * step)) <= n_rows - 1:
        rows.append(int(np.floor(k * step)))
        k += 1
    if rows[-1] != n_rows - 1:
        rows.append(n_rows - 1)
    hold = 0
    if speed == 1 and len(rows) < MIN_S * fps:
        hold = int(round(MIN_S * fps)) - len(rows)
    return dict(rows=rows, speed=speed, hold_frames=hold, gif_frames=len(rows) + hold, gif_seconds=(len(rows) + hold) / fps,
                episode_seconds=dur)


def camera_for_center(points, center, width, height, azimuth, elevation):
    """camera_for of the U5 helper (previous_r0_helpers.py: 7 % border, fovy 45) with the look-at point given instead of the
    bounding-box centre; distance = the smallest that keeps every point inside the frame."""
    center = np.asarray(center, float)
    az, el = np.deg2rad([azimuth, elevation])
    back = np.array([-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), -np.sin(el)])
    right = np.cross(-back, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, -back)
    q = points - center
    tan = np.tan(np.deg2rad(22.5))
    factor = .86
    distance = max(np.max(q @ back + np.abs(q @ right) / (tan * width / height * factor)), np.max(q @ back + np.abs(q @ up) / (tan * factor)))
    return dict(lookat=center.tolist(), distance=float(distance), azimuth=azimuth, elevation=elevation)


def label(img, text, corner, scale=1.0):
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(FONT, int(round(22 * scale)))
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    w, h = x1 - x0, y1 - y0
    pad = int(round(8 * scale))
    W = img.size[0]
    x = W - w - 2 * pad - int(12 * scale) if corner == "right" else int(12 * scale)
    y = int(12 * scale)
    d.rectangle([x, y, x + w + 2 * pad, y + h + 2 * pad], fill=(255, 255, 255, 215))
    d.text((x + pad - x0, y + pad - y0), text, fill=(31, 35, 40, 255), font=font)
    return img


class Renderer:
    def __init__(self, count, object_urdf, settings):
        import mujoco
        self.f, self.helper_sha = rnv.helper()
        self.scene = self.f.newscene(ROBOT_URDF, object_urdf, count, "g1_native_execution", 1920, 1080)
        self.f.styling(self.scene, settings["lookat"])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = settings["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = settings["distance"], settings["azimuth"], settings["elevation"]
        self.settings = settings

    def frame(self, q, o):
        from PIL import Image
        self.scene.set_pose(q, o)
        self.scene.renderer.update_scene(self.scene.data, camera=self.cam)
        return Image.fromarray(self.scene.renderer.render().copy())

    def close(self):
        self.scene.renderer.close()


def write_gif(frames, dest):
    import imageio_ffmpeg
    w, h = frames[0].size
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(GIF_FPS), "-i", "-", "-filter_complex", "split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
           "-loop", "0", "-r", str(GIF_FPS), str(dest)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for im in frames:
            p.stdin.write(im.tobytes())
    finally:
        p.stdin.close()
        rc = p.wait()
    assert rc == 0, rc
    return cmd


def render_asset(r, q, o, name, out, corner_text=None):
    """q: [n, count, 36] (root xyz, quat wxyz, 29 dof), o: [n, 7]; writes <name>.gif and <name>_mid.png; returns record."""
    from PIL import Image
    n = len(q)
    plan = playback_plan(n)
    frames = []
    for i in plan["rows"]:
        im = r.frame(q[i], o[i]).resize((GIF_W, GIF_W * 1080 // 1920), Image.LANCZOS)
        if corner_text:
            label(im, corner_text, "left")
        if plan["speed"] == 2:
            label(im, "2× speed", "right")
        frames.append(im)
    last = frames[-1]
    for _ in range(plan["hold_frames"]):  # only at 1x speed (episode shorter than 4 s)
        frames.append(label(last.copy(), "episode ended", "right"))
    gif = out / f"{name}.gif"
    cmd = write_gif(frames, gif)
    mid = (n - 1) // 2
    png = out / f"{name}_mid.png"
    im = r.frame(q[mid], o[mid])
    if corner_text:
        label(im, corner_text, "left", scale=2.0)
    im.save(png)
    return dict(gif=gif.name, gif_sha256=sha(gif), gif_bytes=gif.stat().st_size, gif_width=GIF_W, gif_height=GIF_W * 1080 // 1920, gif_fps=GIF_FPS,
                gif_loop=True, gif_frames=plan["gif_frames"], gif_seconds=plan["gif_seconds"], playback_speed=plan["speed"],
                held_end_frames=plan["hold_frames"], saved_rows=n, episode_seconds=plan["episode_seconds"], gif_rows=plan["rows"],
                png=png.name, png_sha256=sha(png), png_row=mid, png_time_s=mid / SIM_HZ, png_size=[1920, 1080], corner_label=corner_text,
                camera_settings=r.settings, encoding_command=cmd, render="MuJoCo kinematic replay of saved physics states (U5 helper, EGL)",
                helper_executed_sha256=r.helper_sha, no_physics_steps=True, no_policy=True, no_interpolation=True, no_pose_modification=True)


def pick(attempts):
    """Task-book rule: the end's majority outcome, then the smallest attempt index with that outcome."""
    done = sum(bool(a["completed"]) for a in attempts)
    major = done * 2 > len(attempts)
    a = min((x for x in attempts if bool(x["completed"]) == major), key=lambda x: x["attempt"])
    return a, done


def d7(out):
    import json as _j
    recs, eps = {}, {}
    for side in ("right", "left"):
        npz, js = D7_DIR / f"formal_wide_{side}_trajectory.npz", D7_DIR / f"formal_wide_{side}_trajectory.json"
        rep = _j.loads(js.read_text())
        att = rep["attempt_details"][:20]
        assert rep["attempts"]["closed"] == 20 and len(att) == 20
        a, done = pick(att)
        with np.load(npz, allow_pickle=False) as z:
            meta = _j.loads(str(z["_metadata_json"]))
            s, e = a["start"], a["end"]
            sl = slice(s, e + 1)
            q = np.concatenate([z["root_pos"][sl], z["root_quat_xyzw"][sl][:, [3, 0, 1, 2]], z["dof_pos"][sl]], axis=-1)[:, None, :]
            o = np.concatenate([z["object_pos_w"][sl], z["object_quat_xyzw"][sl][:, [3, 0, 1, 2]]], axis=-1)
            obj_err_last = float(np.linalg.norm(z["object_pos_w"][e] - z["ref_object_pos_w"][e]))
            term_rows = np.flatnonzero(z["terminated"][sl]).tolist()
        assert len(q) == a["length_steps"]
        eps[side] = (q, o)
        recs[side] = dict(side=side, attempt=a, end_completed=done, end_attempts=20, npz=str(npz), npz_sha256=sha(npz), json=str(js),
                          json_sha256=sha(js), metadata=dict((k, meta[k]) for k in ("teammate_provider_class", "teammate_observer_side",
                          "teammate_lateral_spacing_m", "motion_names", "motion_files", "urdf_path", "dt", "sim_dt", "control_decimation")),
                          last_row_object_position_error_m=obj_err_last, terminated_rows_in_episode=term_rows)
    # one camera for both ends
    f, _ = rnv.helper()
    sc = f.newscene(ROBOT_URDF, D7_TABLE, 1, "g1_native_execution", 1920, 1080)
    obj_rows = np.concatenate([np.arange(8 * k, 8 * k + 8) for k, m in enumerate(sc.mapping) if m[0] == sc.count])
    pts, tab = [], []
    for side, (q, o) in eps.items():
        for i in range(len(q)):
            sc.set_pose(q[i], o[i])
            b = f.bounds(sc)  # 8 corners per mapped visual, in scene.mapping order
            pts.append(b)
            tab.append(b[obj_rows])
    sc.renderer.close()
    pts, tab = np.concatenate(pts), np.concatenate(tab)
    center = (tab.min(0) + tab.max(0)) / 2
    center[2] = (pts[:, 2].min() + pts[:, 2].max()) / 2  # height: middle of all robot + table bounds (no empty floor band)
    settings = camera_for_center(pts, center, 1920, 1080, 135, -22)
    r = Renderer(1, D7_TABLE, settings)
    rows = []
    try:
        for side in ("right", "left"):
            q, o = eps[side]
            rec = recs[side]
            txt = f"{side} end: {rec['end_completed']}/20 completed"
            res = render_asset(r, q, o, f"d7_{side}_end_attempt{rec['attempt']['attempt']:02d}", out, corner_text=txt)
            rows.append(dict(asset="D7", **rec, **res, camera_rule="lookat x, y = centre of the table bounding box over both rendered episodes, "
                             "z = middle of the height range of every robot and table mesh bound of both episodes; distance = U5 camera_for "
                             "rule (7 % border) over those bounds; the same camera for both ends", lookat=center.tolist()))
    finally:
        r.close()
    return rows


def d8(out):
    s = json.loads((D8_A / "summary.json").read_text())
    ep = next(e for e in s["episodes"] if e["state_id"] == "perturbed_000")
    traj = Path(ep["trajectory"])
    with np.load(traj, allow_pickle=False) as z:
        q = np.concatenate([z["root_pos"], z["root_quat_xyzw"][..., [3, 0, 1, 2]], z["dof_pos"]], axis=-1)
        o = np.concatenate([z["object_pos_w"], z["object_quat_xyzw"][..., [3, 0, 1, 2]]], axis=-1)
    rows_saved = len(q)
    n = rows_saved - 1 if (ep["completed_reference"] and rows_saved == ep["horizon_steps"] + 1) else rows_saved  # as U5
    f3 = json.loads((rnv.SNAP / "f3_manifest_pull.json").read_text())
    settings = f3["records"][0]["camera"]["settings"]
    obj = ROOT / "src/holosoma" / s["object_urdf"]
    r = Renderer(2, obj, settings)
    try:
        res = render_asset(r, q[:n], o[:n], "d8_pull_schemeA_perturbed_000", out)
    finally:
        r.close()
    return [dict(asset="D8", summary=str(D8_A / "summary.json"), summary_sha256=sha(D8_A / "summary.json"), trajectory=str(traj),
                 trajectory_sha256=sha(traj), episode=ep, checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                 object_urdf=str(obj), object_urdf_sha256=sha(obj), camera_source="sources_snapshot/f3_manifest_pull.json records[0] "
                 "(the U5 F3 Pull camera; render_v3_f3_video.py used the same settings)", **res)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d7", action="store_true")
    p.add_argument("--d8", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl"
    a.out.mkdir(parents=True, exist_ok=True)
    recs = []
    if a.d7:
        recs += d7(a.out)
    if a.d8:
        recs += d8(a.out)
    common = dict(robot_urdf=str(ROBOT_URDF), robot_urdf_sha256=sha(ROBOT_URDF), code_sha256=sha(__file__),
                  sources_snapshot_sha256=sha(rnv.SNAP / "SOURCES.sha256"), command=[sys.executable, *sys.argv])
    for r in recs:
        r.update(common)
        if r["asset"] == "D7":
            r.update(table_urdf=str(D7_TABLE), table_urdf_sha256=sha(D7_TABLE), checkpoint=str(D7_CKPT), checkpoint_sha256=sha(D7_CKPT))
        name = Path(r["gif"]).stem
        (a.out / f"{name}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
        print(name, r["gif_bytes"], "bytes", r["gif_seconds"], "s", "speed", r["playback_speed"], "held", r["held_end_frames"], flush=True)


if __name__ == "__main__":
    main()
