#!/usr/bin/env python3
"""T14 section 10 item 5: draw the palm normal (red) and finger direction (blue) arrows that the hand metrics use, on the
person's hands and the robot's hands, side by side (same frame, same azimuth / elevation; human distance / human-to-robot
scale), for a few contact frames, and print the per-frame angles the metric computes for those frames.

Arrows start at the palm point (person: mean of the 67 palm-side SMPL-X vertices; robot: palm point of the simulator hand
hull); length 10 cm on the robot, 10 cm / scale on the person. Directions are the metric's (hand_landmarks.human_directions /
robot_directions) mapped to world.

    sonic_s1/bin/python -B system1/t14/verify_hand_axes.py --task push --ref A=<single qpos npz>@1.0 --ref B=... --out <jpg>
    sonic_s1/bin/python -B system1/t14/verify_hand_axes.py --task table --ref A=<pair npz>@1.0 ... --out <jpg>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
import contact_gap as cg  # noqa: E402
import hand_landmarks as hl  # noqa: E402
import human_render  # noqa: E402

spec = importlib.util.spec_from_file_location("rnv", ROOT / "system1/render/u5/render_nominal_video.py")
rnv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rnv)
W, H = 420, 480
RED, BLUE = (0.9, 0.1, 0.1, 1.0), (0.1, 0.3, 0.95, 1.0)


def add_arrows(scene, arrows):
    import mujoco
    for p0, p1, rgba in arrows:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, np.asarray(p0, float), np.asarray(p1, float))
        scene.ngeom += 1


def human_axes(hs, person, t30):
    """world palm points [2, 3] and directions {palm_normal, finger}: [2, 3] of the person at frame t30 (human_render world)."""
    from table_shared import palm_vertex_ids
    t = min(hs.frame(t30), len(hs.V[person]) - 1)
    V = hs.V[person][t]
    J = hs.J[person][t]
    pv = palm_vertex_ids()
    palms = np.stack([V[pv[s]].mean(0) for s in ("left", "right")])
    d = {k: np.zeros((2, 3)) for k in ("palm_normal", "finger")}
    for hi, side in enumerate(("left", "right")):
        dd = hl.human_directions(J[None], V[None], side)
        for k in d:
            d[k][hi] = dd[k][0]
    return palms, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("push", "pull", "table", "bin", "chair"))
    ap.add_argument("--ref", action="append", required=True, help="name=path@size (CORE4D pair npz / OMOMO single qpos npz)")
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="30 Hz frames (default: start / middle / end of contact)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert os.environ.get("MUJOCO_GL") == "egl"
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    import dual_stretch as ds
    f, _ = rnv.helper()
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
    robot = cg.cs.Robot("physx")
    offs = cg.palm_offsets()
    omomo = a.task in ("push", "pull")
    hs = human_render.HumanScene(a.task, ROOT / "logs_system1/t14/r5/verify_meshes")
    # full joints of the person(s) in the HumanScene world (for the directions)
    if omomo:
        import single_shared as ss
        hs.J = [ss.omomo_palms.last["joints_omomo_new"]]
    else:
        import table_shared as ts
        from holosoma_retargeting.data_utils.core4d_adapter import CORE4D_TO_OMNI_ROTATION as C
        seq = ts.CORE4D / "CORE4D_Real/human_object_motions" / ts.TASKS[a.task]["raw"]
        hs.J = [np.einsum("ij,tkj->tki", C, np.load(seq / f"person{p}_poses.npz", allow_pickle=True)["arr_0"].item()["joints"].astype(np.float64)) for p in (1, 2)]
    refs = []
    for spec_ in a.ref:
        name, rest = spec_.split("=", 1)
        path, size = rest.split("@")
        if omomo:
            q = np.load(path)["qpos"].astype(np.float64)
            pair, oq = ds.make_pair(a.task, q, float(size))
            scale = float(np.load(path)["smpl_scale"])
            urdf = ds.stretch_table(ds.TABLE[a.task][0], ds.TABLE[a.task][1], float(size), ROOT / f"logs_system1/t14/r5/verify_urdf/{a.task}_{name}.urdf")
            scales = [scale]
        else:
            z = np.load(path)
            pair, oq = z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64)
            scales = [float(x) for x in z["human_to_robot_scales"]]
            kk = "k" + f"{float(size):.1f}".replace(".", "p")
            urdf = next(Path(path).parent.glob("*.urdf"), None) or ROOT / f"exports_system1/T14_references/r3/{a.task}_{kk}/{a.task}_{kk}.urdf"
        refs.append(dict(name=name, pair=pair, oq=oq, scales=scales, urdf=urdf))
    # frames: start / middle / end of the person's hand contact (source rule), as 30 Hz frames
    if a.frames:
        frames = a.frames
    else:
        if omomo:
            h, _ = ds.human_hand_sd(ds.SEQ[a.task])
            idx = np.flatnonzero((h >= -0.015).any(1))
        else:
            import eval_core4d_hulls as eh
            h, _ = eh.human_hand_sd(a.task)
            idx = 2 * np.flatnonzero((h >= -0.015).any(axis=(1, 2)))
        frames = [int(idx[0] + 0.1 * (idx[-1] - idx[0])), int((idx[0] + idx[-1]) / 2), int(idx[-1] - 0.1 * (idx[-1] - idx[0]))]
    persons = (0,) if omomo else (0, 1)
    cols = [("human", p, None) for p in persons for _ in [0]]
    rows_img, report = [], []
    for t in frames:
        row = []
        for p in persons:
            # robot camera: profile of robot p of the first reference (as the grid close-ups), human camera with the same angles
            r0 = refs[0]
            w_, x_, y_, z_ = r0["pair"][t, p, 3:7]
            fwd = np.array([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ + z_ * w_)])
            az = float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + 90.0
            el = -10.0
            # human panel
            palms, d = human_axes(hs, p, t)
            sc = r0["scales"][p if not omomo else 0]
            L = 0.10 / sc
            img = hs.render(p, t, az, el, 0.75 / sc, W, H)
            # re-render with arrows (HumanScene caches the renderer)
            m, dd, rr = hs.cache[(p, min(hs.frame(t), len(hs.V[p]) - 1), W, H)]
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = palms.mean(0), 0.75 / sc, az, el
            rr.update_scene(dd, camera=cam)
            add_arrows(rr.scene, [(palms[i], palms[i] + L * d["palm_normal"][i], RED) for i in range(2)] + [(palms[i], palms[i] + L * d["finger"][i], BLUE) for i in range(2)])
            img = Image.fromarray(rr.render())
            ImageDraw.Draw(img).text((8, 8), f"人 {p + 1} · t = {t / 30:.2f} s", font=font, fill=(20, 20, 20))
            row.append(img)
            for ref in refs:
                s = f.newscene(rnv.ROBOT_URDF if hasattr(rnv, "ROBOT_URDF") else ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf", ref["urdf"], 2, "g1_native_execution", W, H)
                recs = f.styling(s, ref["oq"][t, :3].tolist())
                for rr_ in recs:
                    if rr_["role"] == 1 - p if not omomo else rr_["role"] == 1:
                        s.model.geom_group[rr_["geom_id"]] = 5
                s.set_pose(ref["pair"][t], ref["oq"][t])
                tl = robot.fk(ref["pair"][t:t + 1, p, :3], ref["pair"][t:t + 1, p, [4, 5, 6, 3]], ref["pair"][t:t + 1, p, 7:])
                arrows, rp = [], []
                for side in ("left", "right"):
                    M = tl[f"{side}_rubber_hand_link"][0]
                    pp = M[:3, :3] @ offs[side] + M[:3, 3]
                    rd = hl.robot_directions(M[:3, :3], side)
                    rp.append(pp)
                    arrows += [(pp, pp + 0.10 * rd["palm_normal"], RED), (pp, pp + 0.10 * rd["finger"], BLUE)]
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = np.mean(rp, 0), 0.75, az, el
                opt = mujoco.MjvOption()
                s.renderer.update_scene(s.data, camera=cam, scene_option=opt)
                add_arrows(s.renderer.scene, arrows)
                rimg = Image.fromarray(s.renderer.render())
                ImageDraw.Draw(rimg).text((8, 8), f"{ref['name']} · 机器人 {p} · t = {t / 30:.2f} s", font=font, fill=(20, 20, 20))
                row.append(rimg)
                s.renderer.close()
                # the metric's angles at this frame (object frame), both hands
                Ro = cg.cs.pose_mat(ref["oq"][t:t + 1, :3], ref["oq"][t:t + 1, [4, 5, 6, 3]])[0, :3, :3]
                To = None
                ang = {}
                for hi, side in enumerate(("left", "right")):
                    M = tl[f"{side}_rubber_hand_link"][0]
                    rd = hl.robot_directions(M[:3, :3], side)
                    # human directions are compared in the object frame of the person's object (human_render world object pose)
                    Rh = hs.obj_R[min(hs.frame(t), len(hs.obj_R) - 1)]
                    hn, hf = Rh.T @ d["palm_normal"][hi], Rh.T @ d["finger"][hi]
                    rn, rf = Ro.T @ rd["palm_normal"], Ro.T @ rd["finger"]
                    ang[side] = dict(palm_deg=round(float(np.degrees(np.arccos(np.clip(hn @ rn, -1, 1)))), 1),
                                     finger_deg=round(float(np.degrees(np.arccos(np.clip(hf @ rf, -1, 1)))), 1))
                report.append(dict(ref=ref["name"], frame=t, robot=p, angles_vs_person=ang))
        rows_img.append(row)
    sheet = Image.new("RGB", (W * len(rows_img[0]) + 4 * (len(rows_img[0]) - 1), (H + 4) * len(rows_img)), "white")
    for ri, row in enumerate(rows_img):
        for ci, im in enumerate(row):
            sheet.paste(im, (ci * (W + 4), ri * (H + 4)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    q = 90
    while True:
        sheet.save(a.out, quality=q)
        if a.out.stat().st_size <= 1_000_000 or q <= 40:
            break
        q -= 5
    (a.out.with_suffix(".json")).write_text(json.dumps(dict(rules=__doc__, frames=frames, report=report), indent=1, ensure_ascii=False) + "\n")
    for r in report:
        print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
