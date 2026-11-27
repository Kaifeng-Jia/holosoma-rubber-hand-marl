#!/usr/bin/env python3
"""T14 front-back training task, item 2.6 (Kevin 2026-11-26): pictures of the training reference before training. Render only,
CPU (MuJoCo EGL), step-9 picture style (render_box.py: Fig. 4 look, Unitree grey robots, light-blue box, chequered floor).

What is drawn: the 50 Hz training reference itself (system1_t14fb/table/core4d_pair_runtime_fps50.npz: both robots' root pose
and joints, the box pose) with the training URDF box (5 kg, size of the construction's level-1.0 box), next to the person of
the OMOMO clip (sub8_suitcase_020, sequence frames 103-162, 30 Hz; person frame = 103 + round(0.6 k) for reference frame k).
Three moments, fixed by rule on the reference before looking at the pictures:
  * 抱起 (lift): first frame whose box height exceeds the first frame's by 2 cm;
  * 走路中段 (mid-walk): first frame at which the mean horizontal pelvis displacement of the two robots (from frame 0, along the
    net displacement direction) reaches half of its value at the last frame;
  * 放下 (put down): last frame whose box height exceeds the last frame's by 2 cm.
Sheet: one row per moment; columns = the person with the captured object (close-up as render_box), robot 0 close-up (robot 1
hidden), both robots (full view). Video (--video-out): left the person, right both robots, the whole reference at 50 Hz.

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      sonic_s1/bin/python -B system1/t14/frontback_train/render_reference.py --out results_system1/T14_frontback/reference \
      --video-out exports_system1/T14_videos/frontback_train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "system1/t14/objects"))
import render_box as rb  # noqa: E402

A = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14fb/table"
SOLVE = ROOT / "logs_system1/t14/step9g2/sub8_suitcase_020/m1p00/r0"
CLIP_START = 103
LIFT_DZ = 0.02


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def moments(z):
    oz = z["object_pos_w"][:, 2]
    lift = int(np.flatnonzero(oz - oz[0] > LIFT_DZ)[0])
    put = int(np.flatnonzero(oz - oz[-1] > LIFT_DZ)[-1])
    pi = list(z["body_names"]).index("pelvis")
    p = z["agent_body_pos_w"][:, :, pi, :2].mean(1)
    d = p[-1] - p[0]
    s = (p - p[0]) @ (d / np.linalg.norm(d))
    mid = int(np.flatnonzero(s >= 0.5 * s[-1])[0])
    return dict(lift=lift, mid_walk=mid, put_down=put)


def qpos(z, k):
    pi = list(z["body_names"]).index("pelvis")
    q = np.zeros((2, 36))
    q[:, :3] = z["agent_body_pos_w"][k, :, pi]
    q[:, 3:7] = z["agent_body_quat_w"][k, :, pi]  # wxyz
    q[:, 7:] = z["agent_joint_pos"][k]
    return q, np.concatenate([z["object_pos_w"][k], z["object_quat_w"][k]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-out", type=Path, default=None)
    a = ap.parse_args()
    import mujoco
    man = json.loads((A / "t14_reference.json").read_text())
    runtime, urdf = A / "core4d_pair_runtime_fps50.npz", A / "frontback_box5kg_training.urdf"
    z = {k: v for k, v in np.load(runtime, allow_pickle=False).items()}
    T = len(z["agent_joint_pos"])
    mo = moments(z)
    rec = json.loads((SOLVE / "run_record.json").read_text())
    person = rb.PersonScene(rec)
    f, _ = rb.rnv.helper()
    font, small = ImageFont.truetype(rb.FONT, 20), ImageFont.truetype(rb.FONT, 17)
    scenes = {}

    def scene(w, h):
        if (w, h) not in scenes:
            s = f.newscene(rb.ROBOT_URDF, urdf, 2, "g1_native_execution", w, h)
            for r in f.styling(s, z["object_pos_w"][0]):
                if r["role"] == "object":
                    s.model.geom_rgba[r["geom_id"]] = rb.OBJECT_RGBA
            scenes[(w, h)] = s
        return scenes[(w, h)]

    def render(s, cam, q, o, hide_robot1=False):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["distance"], cam["azimuth"], cam["elevation"]
        q = q.copy()
        if hide_robot1:
            q[1, :3] += np.array([0.0, 0.0, -50.0])
        s.set_pose(q, o)
        s.renderer.update_scene(s.data, camera=c)
        return Image.fromarray(s.renderer.render())

    bn = list(z["body_names"])
    ix = {n: bn.index(n) for n in ("pelvis", "torso_link", "left_rubber_hand_link", "right_rubber_hand_link")}

    def close_cam(k):
        P = z["agent_body_pos_w"][k, 0]
        look = np.mean([P[ix[n]] for n in ix], axis=0)
        w, x, y, zz = z["agent_body_quat_w"][k, 0, ix["pelvis"]]
        yaw = np.degrees(np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz)))
        return dict(lookat=look, distance=1.6, azimuth=yaw + 90 + 30, elevation=-15)

    PW, PH = rb.PW, rb.PH
    names = dict(lift="抱起", mid_walk="走路中段", put_down="放下")
    rows = []
    for key, k in mo.items():
        tp = CLIP_START + int(round(0.6 * k))
        ph = person.render(tp, person.lookat(tp), person.heading(tp) + 90 + 30, 1.6 / person.scale, -15, PW, PH)
        ph = rb.caption(ph, [f"{names[key]} · 人 · 序列帧 {tp}（{(tp - CLIP_START) / 30:.2f} s）"], small)
        q, o = qpos(z, k)
        rc = rb.caption(render(scene(PW, PH), close_cam(k), q, o, hide_robot1=True), [f"{names[key]} · 机器人 0 · 参考帧 {k}（{k / 50:.2f} s）"], small)
        sf = scene(PW + 120, PH)
        sf.set_pose(q, o)
        az = float(np.degrees(np.arctan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0]))) - 90 + 30
        fu = rb.caption(render(sf, f.camera_for(f.bounds(sf), PW + 120, PH, az, -25), q, o), [f"{names[key]} · 两台 · 参考帧 {k}（{k / 50:.2f} s）"], small)
        row = Image.new("RGB", (PW * 2 + PW + 120, PH), "white")
        for i, im in enumerate((ph, rc, fu)):
            row.paste(im.convert("RGB"), (i * PW, 0))
        rows.append(row)
    ob = man["object_replacement"]
    head = Image.new("RGB", (rows[0].width, 60), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(head)
    d.text((10, 4), f"一前一后训练参考 · sub8_suitcase_020 · 1.0 倍（冻结的 9g2 构造）· 50 Hz，{T} 帧（{(T - 1) / 50:.2f} s）· 箱子 "
                    f"{' × '.join(f'{x:.2f}' for x in ob['box_size_m'])} m，{ob['object_mass_kg']:.0f} kg", fill=(0, 0, 0), font=font)
    d.text((10, 32), f"时刻按规则取：抱起 = 箱高首次比第 0 帧高 2 cm；走路中段 = 两机骨盆平均水平位移首次达到全程一半；放下 = 箱高最后一次比末帧高 2 cm",
           fill=(0, 0, 0), font=small)
    sheet = Image.new("RGB", (rows[0].width, 60 + PH * len(rows)), "white")
    sheet.paste(head, (0, 0))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, 60 + PH * i))
    a.out.mkdir(parents=True, exist_ok=True)
    q_sheet = rb.save_jpg(sheet, a.out / "frontback_reference_sidebyside.jpg")
    out = dict(rules=__doc__, moments_reference_frame=mo, moments_person_sequence_frame={k: CLIP_START + int(round(0.6 * v)) for k, v in mo.items()},
               object_height_m={k: float(z["object_pos_w"][v, 2]) for k, v in mo.items()},
               object_height_first_last_m=[float(z["object_pos_w"][0, 2]), float(z["object_pos_w"][-1, 2])],
               sheet=dict(file="frontback_reference_sidebyside.jpg", jpeg_quality=q_sheet),
               inputs={str(p.relative_to(ROOT)): sha(p) for p in (runtime, urdf, A / "t14_reference.json", SOLVE / "run_record.json")})
    if a.video_out is not None:
        import imageio_ffmpeg
        a.video_out.mkdir(parents=True, exist_ok=True)
        vp = a.video_out / "frontback_reference_sub8_suitcase_020_m1p00.mp4"
        VW, VH = 640, 480
        sv = scene(VW, VH)
        pts = []
        for k in range(0, T, 5):
            sv.set_pose(*qpos(z, k))
            pts.append(f.bounds(sv))
        q0, _ = qpos(z, 0)
        vcam = f.camera_for(np.concatenate(pts), VW, VH, float(np.degrees(np.arctan2(q0[1, 1] - q0[0, 1], q0[1, 0] - q0[0, 0]))) - 90 + 30, -25)
        from scipy.ndimage import gaussian_filter1d
        plook = gaussian_filter1d(np.array([person.lookat(t) for t in range(len(person.V))]), 10, axis=0, mode="nearest")
        paz = person.heading(CLIP_START + int(round(0.6 * mo["mid_walk"]))) + 90 + 30
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * VW}x{VH}",
               "-r", "50", "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-crf", "22", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(vp)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k in range(T):
            tp = CLIP_START + int(round(0.6 * k))
            pim = rb.caption(person.render(tp, plook[tp], paz, 2.6, -18, VW, VH, arrows=False), [f"人 · sub8_suitcase_020 · {(tp - CLIP_START) / 30:.2f} s"], small).convert("RGB")
            q, o = qpos(z, k)
            rim = rb.caption(render(sv, vcam, q, o), [f"两台机器人 · 训练参考 · {k / 50:.2f} s"], small).convert("RGB")
            fr = Image.new("RGB", (2 * VW, VH))
            fr.paste(pim, (0, 0))
            fr.paste(rim, (VW, 0))
            proc.stdin.write(np.asarray(fr).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        out["video"] = dict(file=str(vp.resolve().relative_to(ROOT)), bytes=vp.stat().st_size, sha256=sha(vp), fps=50, frames=T)
    (a.out / "frontback_reference_render.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for s in scenes.values():
        s.renderer.close()
    print(json.dumps({k: out[k] for k in ("moments_reference_frame", "moments_person_sequence_frame", "object_height_m")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
