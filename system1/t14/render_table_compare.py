#!/usr/bin/env python3
"""T14: Table reference comparison stills (kinematic reference itself, not a policy replay). Render only, CPU.

Rows: t = 4.0 s (lift), 6.5 s (carry), 9.5 s (lower) -> 50 Hz runtime frames 200, 325, 475. Columns: left the current
reference (core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz, sha 582e7669...), right the T14 1.0x reference
(exports_system1/T14_references/table_k1p0/core4d_pair_runtime_fps50.npz). Robots = reference pelvis pose + 29 joint
angles (URDF order, checked), table = reference object pose; training Table URDF.

Scene: U5 / paper Fig. 4 style (system1/render/u5 helper, preset g1_native_execution: Unitree colours, light-blue object,
chequered ground). Both cells of a row use the same camera parameters, computed from the current reference at that frame:
  * --fig side:  lookat = (table x, y, table-top height); azimuth = robot-0 -> robot-1 horizontal direction - 90 deg
                 (looking across the line between the robots), elevation SIDE_EL, distance SIDE_DIST;
  * --fig hands: lookat = midpoint of robot 0's two rubber-hand palm points; the camera looks from robot 1's side over the
                 table at robot 0 (azimuth = robot-1 -> robot-0 direction), elevation HANDS_EL, distance HANDS_DIST. Robot 1 is
                 not drawn in this figure (its geoms are put in a hidden render group; otherwise it sits between the camera and
                 robot 0's hands) and the cell label says so.
Each cell is labelled (old / new, time); jpg <= 1 MB (quality lowered from 90 in steps of 5).

    CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      sonic_s1/bin/python -B system1/t14/render_table_compare.py --fig side --out results_system1/T14_retargeting/step1_table/figures
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("render_nominal_video", ROOT / "system1/render/u5/render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)

ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
OBJ_URDF = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf"
REFS = dict(old=ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz",
            new=ROOT / "exports_system1/T14_references/table_k1p0/core4d_pair_runtime_fps50.npz")
TIMES = ((4.0, "抬起"), (6.5, "搬运中"), (9.5, "放下"))
SIZE = dict(side=(960, 640), hands=(1280, 853))  # cell pixels
SIDE_EL, SIDE_DIST, TOP_ABOVE_ORIGIN = -10.0, 1.75, 0.27
HANDS_AZ_OFF, HANDS_EL, HANDS_DIST = 270.0, -40.0, 0.6
PALM = dict(left=np.array([0.063785, -0.042639, 0.008036]), right=np.array([0.06378, 0.042687, 0.008938]))  # rubber-hand palm points (link frame)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
LABEL = dict(old="旧参考（现有）", new="新参考（T14，1.0 倍）")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quat_wxyz_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def state(z, k):
    b = list(z["body_names"])
    pel = b.index("pelvis")
    q = np.concatenate([z["agent_body_pos_w"][k, :, pel], z["agent_body_quat_w"][k, :, pel], z["agent_joint_pos"][k]], axis=-1)  # [2, 36], quat wxyz
    o = np.concatenate([z["object_pos_w"][k], z["object_quat_w"][k]])
    palms = []
    for side in ("left", "right"):
        i = b.index(f"{side}_rubber_hand_link")
        palms.append(z["agent_body_pos_w"][k, 0, i] + quat_wxyz_to_mat(z["agent_body_quat_w"][k, 0, i]) @ PALM[side])
    return q, o, np.array(palms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", choices=("side", "hands"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--override", default=None, help="framing check only: el,dist[,azoff]")
    a = ap.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    W, H = SIZE[a.fig]
    f, helper_sha = rnv.helper()
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    refs = {k: np.load(v) for k, v in REFS.items()}
    assert list(refs["old"]["joint_names"]) == list(refs["new"]["joint_names"])
    scene = f.newscene(ROBOT_URDF, OBJ_URDF, 2, "g1_native_execution", W, H)
    assert [str(x) for x in refs["old"]["joint_names"]] == list(scene.urdfs[0].actuated_joint_names)
    el, dist, azoff = (SIDE_EL, SIDE_DIST, 0.0) if a.fig == "side" else (HANDS_EL, HANDS_DIST, HANDS_AZ_OFF)
    if a.override:
        v = [float(x) for x in a.override.split(",")]
        el, dist = v[0], v[1]
        azoff = v[2] if len(v) > 2 else azoff
    font = ImageFont.truetype(FONT, 26)
    rows, cams = [], []
    styled = False
    for t, name in TIMES:
        k = int(round(t * 50))
        q0, o0, palms0 = state(refs["old"], k)
        axis = q0[1, :2] - q0[0, :2]
        az = float(np.degrees(np.arctan2(axis[1], axis[0]))) - 90.0 + azoff
        lookat = [float(o0[0]), float(o0[1]), float(o0[2] + TOP_ABOVE_ORIGIN)] if a.fig == "side" else palms0.mean(0).tolist()
        if not styled:
            recs = f.styling(scene, lookat)
            opt = mujoco.MjvOption()
            if a.fig == "hands":  # hide robot 1
                hidden = [r["geom_id"] for r in recs if r["role"] == 1]
                assert hidden
                for g in hidden:
                    scene.model.geom_group[g] = 5
                opt.geomgroup[5] = 0
            styled = True
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        cams.append(dict(time_s=t, frame_50hz=k, lookat=[round(x, 4) for x in lookat], distance=dist, azimuth=round(az, 2), elevation=el, fovy=45))
        cells = []
        for col in ("old", "new"):
            q, o, _ = state(refs[col], k)
            scene.set_pose(q, o)
            scene.renderer.update_scene(scene.data, camera=cam, scene_option=opt)
            img = Image.fromarray(scene.renderer.render())
            d = ImageDraw.Draw(img, "RGBA")
            txt = f"{'旧' if col == 'old' else '新'} · {LABEL[col]} · t = {t:.1f} s（{name}）" + ("· 只画机器人 0" if a.fig == "hands" else "")
            d.rounded_rectangle((10, 10, 20 + d.textlength(txt, font=font), 54), radius=8, fill=(255, 255, 255, 220), outline=(150, 160, 170, 255))
            d.text((20, 14), txt, font=font, fill=(20, 35, 55))
            cells.append(img)
        row = Image.new("RGB", (2 * W + 6, H), "white")
        row.paste(cells[0], (0, 0))
        row.paste(cells[1], (W + 6, 0))
        rows.append(row)
    sheet = Image.new("RGB", (2 * W + 6, 3 * H + 12), "white")
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * (H + 6)))
    a.out.mkdir(parents=True, exist_ok=True)
    stem = "table_ref_old_vs_new_side" if a.fig == "side" else "table_ref_old_vs_new_robot0_hands"
    if a.override:
        stem += "_check_" + a.override.replace(",", "_")
    p = a.out / f"{stem}.jpg"
    qual = 90
    while True:
        sheet.save(p, quality=qual)
        if p.stat().st_size <= 1_000_000 or qual <= 40:
            break
        qual -= 5
    rec = dict(figure=a.fig, file=p.name, bytes=p.stat().st_size, jpeg_quality=qual, cameras=cams, refs={k: dict(path=str(v.relative_to(ROOT)), sha256=sha(v)) for k, v in REFS.items()},
               object_urdf=str(OBJ_URDF.relative_to(ROOT)), robot_urdf=str(ROBOT_URDF.relative_to(ROOT)), helper_sha256=helper_sha, preset="g1_native_execution", cell_px=[W, H],
               script=str(Path(__file__).relative_to(ROOT)), rules=__doc__)
    if not a.override:
        (a.out / f"{stem}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("file", "bytes", "jpeg_quality")}), flush=True)
    scene.renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
