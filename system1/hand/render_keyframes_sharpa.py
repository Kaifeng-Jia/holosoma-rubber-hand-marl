#!/usr/bin/env python3
"""T06 step 2 keyframe montage of a Sharpa finger reference (kinematic replay of the reference, no physics).

Scene, materials, ground, lighting and the overview camera are the U5 / paper Fig. 4 ones (system1/render/u5, the
NativeScene of the frozen figround_common helper, F3 Bin camera). The NativeScene source is reused with two changes
only: no rubber-hand / 29-joint assertions, and all 73 actuated joints (29 body + 44 Sharpa) are set per agent.
Robot colours: G1 meshes keep the Unitree colours; the Sharpa visual meshes (native light blue-grey, close to the
light-blue object, and green fingertip elastomers) are shown in the G1 grey (0.7) and the elastomers in the G1 dark
grey (0.2) - stated in the caption JSON.

Outputs (--out-dir):
  * overview.png  - 6 frames evenly spaced over the clip, F3 Bin camera (U5 keyframe layout, 3 per row);
  * closeups.png  - per hand, 3 frames at 25 / 50 / 75 % of its contact interval (no contact frames: the closest
                    approach), camera at the palm point, 0.55 m, view picked automatically (see the code comment);
  * closeups_xray_diagnostic.png - the same views with the object at 30 % opacity (diagnostic, not the paper style);
  * render.json   - inputs and sha256, frames, cameras, per-tile finger / rigid-part penetration from per_frame.csv.

CPU only: Mesa llvmpipe through EGL (no GPU):
    CUDA_VISIBLE_DEVICES= __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json EGL_PLATFORM=surfaceless \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl taskset -c 0-15 sonic_s1/bin/python -B system1/hand/render_keyframes_sharpa.py \
        --reference logs_system1/t06/finger_ref/A_track --out-dir results_system1/T06/step2_A/montage
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
U5 = ROOT / "system1/render/u5"
spec = importlib.util.spec_from_file_location("render_nominal_video", U5 / "render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)

BIN_DIR = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1"
REFERENCE = BIN_DIR / "core4d_pair_runtime_fps50.npz"
OBJECT_URDF = BIN_DIR / "bucket003_training.urdf"
ROBOT_URDF = ROOT / "system1/hand/assets/g1_29dof_sharpa.urdf"
G1_GREY, G1_DARK = (0.7, 0.7, 0.7, 1.0), (0.2, 0.2, 0.2, 1.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sharpa_scene_class(f):
    src = f.native
    for a, b in [("            assert tuple(u.actuated_joint_names) == EXPECTED_ROBOT_JOINT_NAMES\n", "            pass\n"),
                 ("            assert {'left_rubber_hand_link', 'right_rubber_hand_link'} <= set(u.link_map)\n", ""),
                 ("            u.update_cfg(q[7:36])", "            u.update_cfg(q[7:])"),
                 ("class NativeScene:", "class SharpaNativeScene:")]:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    exec(compile(src, "sharpa_native_scene", "exec"), f.__dict__)
    return f.__dict__["SharpaNativeScene"]


def recolor_sharpa(scene):
    """Sharpa visual geoms -> G1 grey / dark grey (elastomer); returns the number of geoms changed."""
    n = 0
    for (agent, node, body), rec in zip(scene.mapping, scene.mapping_records):
        if agent >= scene.count:
            continue
        link = rec["link"]
        if any(link.startswith(f"{s}_{p}") for s in ("left", "right") for p in ("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_")):
            scene.model.geom_rgba[rec["geom_id"]] = G1_DARK if "elastomer" in link else G1_GREY
            n += 1
    return n


def camera_at(mujoco, lookat, distance, az, el):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, az, el
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True, help="finger_ref.py output dir (finger_reference.npz, per_frame.csv)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl"
    f, helper_sha = rnv.helper()
    import mujoco
    from OpenGL import GL
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    from system1.hand.sdf import GridSDF
    from system1.hand.sharpa import SharpaHand, mount_transform
    from system1.hand.finger_ref import palm_point_flange, quat_wxyz_to_matrix

    ref = dict(np.load(REFERENCE, allow_pickle=True))
    fr = dict(np.load(a.reference / "finger_reference.npz", allow_pickle=True))
    rows = list(csv.DictReader((a.reference / "per_frame.csv").open()))
    T = fr["agent_joint_pos"].shape[0]
    body_names = [str(n) for n in ref["body_names"]]
    ip = body_names.index("pelvis")  # body 0 is "world"
    jn = [str(n) for n in fr["joint_names"]]
    fn = [str(n) for n in fr["finger_joint_names"]]
    Scene = sharpa_scene_class(f)
    W, H = 960, 540
    scene = Scene(ROBOT_URDF, OBJECT_URDF, 2, W, H)
    scene.model.vis.global_.ipd = 0
    scene.preset, scene.width, scene.height, scene.mapping_records = "g1_native_execution", W, H, []
    cam_rec = json.loads((U5 / "sources_snapshot/f3_manifest_bucket.json").read_text())["records"][0]["camera"]
    st = cam_rec["settings"]
    f.styling(scene, st["lookat"])
    n_recolor = recolor_sharpa(scene)
    order = list(scene.urdfs[0].actuated_joint_names)
    assert len(order) == 73

    def qvec(t):
        out = []
        for ag in range(2):
            cfg = {n: float(fr["agent_joint_pos"][t, ag, i]) for i, n in enumerate(jn)}
            cfg.update({n: float(fr["agent_finger_joint_pos"][t, ag, i]) for i, n in enumerate(fn)})
            out.append(np.concatenate([ref["agent_body_pos_w"][t, ag, ip], ref["agent_body_quat_w"][t, ag, ip], [cfg[n] for n in order]]))
        return out, np.concatenate([ref["object_pos_w"][t], ref["object_quat_w"][t]])

    def shot(t, cam):
        q, o = qvec(t)
        scene.set_pose(q, o)
        scene.renderer.update_scene(scene.data, camera=cam)
        px = scene.renderer.render()
        assert px.shape == (H, W, 3)
        stds.append(float(px.std()))
        return Image.fromarray(px)

    def label(im, lines):
        d = ImageDraw.Draw(im)
        y = 6
        for s in lines:
            d.rectangle([4, y - 2, 8 + 7 * len(s), y + 12], fill=(255, 255, 255))
            d.text((7, y), s, fill=(0, 0, 0))
            y += 16
        return im

    def tile(images, cols):
        r = (len(images) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * W, r * H), "white")
        for i, im in enumerate(images):
            canvas.paste(im, ((i % cols) * W, (i // cols) * H))
        return canvas

    per = {(int(r["agent"]), r["side"], int(r["frame"])): r for r in rows}
    stds = []
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(reference_dir=str(a.reference), finger_reference_sha256=sha(a.reference / "finger_reference.npz"),
               per_frame_sha256=sha(a.reference / "per_frame.csv"), body_reference=str(REFERENCE.relative_to(ROOT)),
               body_reference_sha256=sha(REFERENCE), robot_urdf=str(ROBOT_URDF.relative_to(ROOT)), robot_urdf_sha256=sha(ROBOT_URDF),
               object_urdf=str(OBJECT_URDF.relative_to(ROOT)), object_urdf_sha256=sha(OBJECT_URDF), helper_executed_sha256=helper_sha,
               code_sha256=sha(__file__), gl_vendor=None, gl_renderer=None, width=W, height=H,
               colours=dict(g1="Unitree mesh colours (unchanged)", sharpa=f"G1 grey {G1_GREY[:3]}; fingertip elastomer G1 dark grey {G1_DARK[:3]}",
                            sharpa_geoms_recoloured=n_recolor, object="U5 object colour (unchanged)"),
               no_physics_steps=True, kinematic_reference_replay=True, overview=[], closeups=[])

    # overview: 6 frames evenly spaced; F3 Bin distance / azimuth / elevation, lookat = centroid of the two pelvises and
    # the object (xy) at the F3 lookat height (the F3 lookat itself is in the evaluator's world, not the reference's)
    ims = []
    for t in sorted(set(np.linspace(0, T - 1, 6).round().astype(int).tolist())):
        c = np.mean([ref["agent_body_pos_w"][t, 0, ip], ref["agent_body_pos_w"][t, 1, ip], ref["object_pos_w"][t]], axis=0)
        look = [float(c[0]), float(c[1]), float(st["lookat"][2])]
        cam = camera_at(mujoco, look, st["distance"], st["azimuth"], st["elevation"])
        im = shot(t, cam)
        if rec["gl_vendor"] is None:
            rec["gl_vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
            rec["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        ims.append(label(im, [f"frame {t} ({t / 50:.2f} s)"]))
        rec["overview"].append(dict(frame=t, pixel_std=stds[-1], camera=dict(lookat=look, distance=st["distance"], azimuth=st["azimuth"], elevation=st["elevation"])))
    tile(ims, 3).save(a.out_dir / "overview.png")

    # close-ups: per hand, 25 / 50 / 75 % of its contact interval (a hand without contact frames: its closest approach).
    # Camera: 0.55 m from the robot palm point; azimuth every 20 deg x elevation -20 / -40 / -60; candidates whose camera
    # is inside or within 5 cm of the object, or whose line of sight (first 80 %) crosses the object or passes within
    # 12 cm of a reference body origin (both robots, except this arm's elbow / wrist / hand), are penalised; among the
    # rest the one looking most along the object's inward surface normal at the palm wins.
    import torch
    sdf = GridSDF(BIN_DIR / "bucket003_m.obj", resolution=0.003, cache_dir=ROOT / "logs_system1/t06/cache")
    Robj = quat_wxyz_to_matrix(ref["object_quat_w"])
    obj_ids = [r["geom_id"] for r in scene.mapping_records if r["role"] == "object"]
    obj_rgba = {g: scene.model.geom_rgba[g].copy() for g in obj_ids}
    ims, xray = [], []

    def sdf_w(t, pts):
        po = (np.asarray(pts, float) - ref["object_pos_w"][t]) @ Robj[t]
        return sdf(torch.as_tensor(po)).numpy()

    def pick_camera(t, ag, side, palm):
        bodies = [ref["agent_body_pos_w"][t, k, b] for k in range(2) for b, n in enumerate(body_names)
                  if n != "world" and not (k == ag and n.startswith(side) and any(x in n for x in ("elbow", "wrist", "hand")))]
        bodies = np.asarray(bodies)
        e = 0.01
        grad = np.array([(sdf_w(t, [palm + e * np.eye(3)[i]])[0] - sdf_w(t, [palm - e * np.eye(3)[i]])[0]) / (2 * e) for i in range(3)])
        nrm = grad / max(np.linalg.norm(grad), 1e-9)
        best = None
        for el in (-20.0, -40.0, -60.0):
            for az in range(0, 360, 20):
                fwd = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)), math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
                cam = palm - 0.55 * fwd
                seg = cam[None] + np.linspace(0, 0.8, 17)[:, None] * (palm - cam)[None]
                if sdf_w(t, [cam])[0] < 0.05:
                    continue
                obj_block = bool((sdf_w(t, seg) < 0).any())
                body_block = bool(np.min(np.linalg.norm(seg[:, None] - bodies[None], axis=-1)) < 0.12)
                score = -10 * obj_block - 5 * body_block + float(-fwd @ nrm) - 0.3 * abs(el + 40) / 20
                if best is None or score > best[0]:
                    best = (score, az, el, obj_block, body_block)
        return best

    for ag in range(2):
        for side in ("left", "right"):
            hand = SharpaHand(side)
            cf = [t for t in range(T) if per[(ag + 1, side, t)]["contact"] == "True"]
            if cf:
                picks = [(cf[int(round(fr_ * (len(cf) - 1)))], fr_) for fr_ in (0.25, 0.5, 0.75)]
            else:
                t_close = int(np.argmin([float(per[(ag + 1, side, t)]["robot_palm_point_sd_m"]) for t in range(T)]))
                picks = [(t_close, None)]
            u = scene.urdfs[0]
            for t, frac in picks:
                q, o = qvec(t)
                u.update_cfg(q[ag][7:])
                Tb = np.eye(4)
                Tb[:3, :3] = quat_wxyz_to_matrix(q[ag][3:7]); Tb[:3, 3] = q[ag][:3]
                Tfl = Tb @ u.get_transform(hand.root, u.base_link)
                palm = (Tfl[:3, :3] @ palm_point_flange(hand)) + Tfl[:3, 3]
                score, az, el, ob, bb = pick_camera(t, ag, side, palm)
                cam = camera_at(mujoco, palm, 0.55, az, el)
                r = per[(ag + 1, side, t)]
                lines = [f"agent {ag + 1} {side} hand, frame {t} ({t / 50:.2f} s)" + ("" if frac is not None else ", no contact frame: closest approach"),
                         f"finger pen {1000 * float(r['finger_penetration_m']):.1f} mm, palm/wrist pen {1000 * float(r['rigid_penetration_m']):.1f} mm, "
                         f"palm point sd {1000 * float(r['robot_palm_point_sd_m']):.0f} mm, fingers touching {r['fingers_touching']}"]
                ims.append(label(shot(t, cam), lines))
                std_opaque = stds[-1]
                for g in obj_ids:
                    scene.model.geom_rgba[g][3] = 0.3
                xray.append(label(shot(t, cam), lines + ["DIAGNOSTIC: object at 30% opacity"]))
                for g in obj_ids:
                    scene.model.geom_rgba[g] = obj_rgba[g]
                rec["closeups"].append(dict(agent=ag + 1, side=side, frame=t, contact_fraction=frac, contact_frames=len(cf),
                                            camera=dict(lookat=palm.tolist(), distance=0.55, azimuth=az, elevation=el, score=score,
                                                        line_of_sight_crosses_object=ob, line_of_sight_near_body=bb),
                                            finger_penetration_m=float(r["finger_penetration_m"]), rigid_penetration_m=float(r["rigid_penetration_m"]),
                                            robot_palm_point_sd_m=float(r["robot_palm_point_sd_m"]),
                                            fingers_touching=int(r["fingers_touching"]), palm_touching=r["palm_touching"] == "True",
                                            pixel_std=std_opaque, pixel_std_xray=stds[-1]))
    tile(ims, 3).save(a.out_dir / "closeups.png")
    tile(xray, 3).save(a.out_dir / "closeups_xray_diagnostic.png")
    rec["closeups_xray_sha256"] = sha(a.out_dir / "closeups_xray_diagnostic.png")
    rec["overview_sha256"] = sha(a.out_dir / "overview.png")
    rec["closeups_sha256"] = sha(a.out_dir / "closeups.png")
    (a.out_dir / "render.json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    scene.renderer.close()
    print(a.out_dir, rec["gl_vendor"], rec["gl_renderer"], len(rec["closeups"]), "close-ups")


if __name__ == "__main__":
    raise SystemExit(main())
