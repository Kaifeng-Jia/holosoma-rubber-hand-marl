#!/usr/bin/env python3
"""Offline MuJoCo replay of recorded holosoma states (no physics; poses only).

Robots are our rubber-hand G1 URDF (one copy per recorded robot). Optional object URDF.
Reference body positions are drawn as small green spheres, the robot's tracked bodies
as red spheres, so drift is visible. Output: H.264 mp4 at the recording rate.

Inputs (npz):
  --single : validate_single.py trajectory.npz (root_pos, root_quat_wxyz, dof_pos_h,
             body_pos14, ref_body_pos14)
  --paired : evaluate_paper_supplement.py <state>.trajectory.npz (root_pos [K,2,3],
             root_quat_xyzw, dof_pos [K,2,29], object_pos_w, object_quat_xyzw,
             reference_object_pos_w) with --object-urdf
Run with an env that has mujoco + imageio_ffmpeg, MUJOCO_GL=egl (or osmesa).
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"


def urdf_joint_order(path):
    return [j.get("name") for j in ET.parse(path).getroot().findall("joint") if j.get("type") in ("revolute", "continuous")]


def build_model(n_robots, object_urdf=None):
    import mujoco

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    w = spec.worldbody
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], rgba=[0.82, 0.84, 0.86, 1])
    w.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    colors = ([0.35, 0.55, 0.95, 1], [0.95, 0.6, 0.25, 1])
    for r in range(n_robots):
        rs = mujoco.MjSpec.from_file(str(ROBOT_URDF))
        rs.compiler.meshdir = str(ROBOT_URDF.parent)
        for g in rs.geoms:
            g.rgba = colors[r % 2]
            g.contype = g.conaffinity = 0
        body = rs.worldbody.first_body()
        body.add_freejoint()
        frame = w.add_frame()
        frame.attach_body(body, f"r{r}_", "")
    if object_urdf is not None:
        object_urdf = Path(object_urdf).resolve()  # absolute, so mesh files resolve independent of cwd
        os_ = mujoco.MjSpec.from_file(str(object_urdf))
        os_.compiler.meshdir = str(object_urdf.parent)
        for g in os_.geoms:
            g.rgba = [0.75, 0.75, 0.7, 0.8]
            g.contype = g.conaffinity = 0
        ob = os_.worldbody.first_body()
        ob.add_freejoint()
        w.add_frame().attach_body(ob, "obj_", "")
    return spec.compile()


def main():
    import imageio_ffmpeg
    import mujoco

    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", type=Path)
    g.add_argument("--paired", type=Path)
    p.add_argument("--object-urdf", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--title", default="")
    a = p.parse_args()
    z = np.load(a.single or a.paired)
    if a.single:
        root = z["root_pos"][:, None]
        quat = z["root_quat_wxyz"][:, None]
        q = z["dof_pos_h"][:, None]
        ref_pts, rob_pts = z["ref_body_pos14"][1 : len(root) + 1], z["body_pos14"]
        obj = None
    else:
        root = z["root_pos"]
        quat = z["root_quat_xyzw"][..., [3, 0, 1, 2]]
        q = z["dof_pos"]
        ref_pts = rob_pts = None
        obj = (z["object_pos_w"], z["object_quat_xyzw"][..., [3, 0, 1, 2]]) if a.object_urdf else None
    K, R = root.shape[0], root.shape[1]
    m = build_model(R, a.object_urdf if obj is not None else None)
    d = mujoco.MjData(m)
    names = urdf_joint_order(ROBOT_URDF)
    adr = [[m.joint(f"r{r}_{n}").qposadr[0] for n in names] for r in range(R)]
    free = [m.joint(m.body(f"r{r}_pelvis").jntadr[0]).qposadr[0] for r in range(R)]
    obj_free = None
    if obj is not None:
        ob = [i for i in range(m.nbody) if m.body(i).name.startswith("obj_")][0]
        obj_free = m.joint(m.body(ob).jntadr[0]).qposadr[0]
    renderer = mujoco.Renderer(m, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 4.0, -18.0, 135.0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio_ffmpeg.write_frames(str(a.out), (a.width, a.height), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p", quality=7)
    wr.send(None)
    for k in range(K):
        for r in range(R):
            d.qpos[free[r] : free[r] + 3] = root[k, r]
            d.qpos[free[r] + 3 : free[r] + 7] = quat[k, r]
            d.qpos[adr[r]] = q[k, r]
        if obj is not None:
            d.qpos[obj_free : obj_free + 3] = obj[0][k]
            d.qpos[obj_free + 3 : obj_free + 7] = obj[1][k]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = root[k].mean(axis=0) * np.array([1, 1, 0]) + np.array([0, 0, 0.8])
        renderer.update_scene(d, camera=cam)
        if ref_pts is not None:
            sc = renderer.scene
            for pts, rgba in ((ref_pts[k], [0.1, 0.8, 0.2, 0.9]), (rob_pts[k], [0.9, 0.15, 0.15, 0.9])):
                for pt in pts:
                    if sc.ngeom >= sc.maxgeom:
                        break
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                                        np.asarray(pt, dtype=np.float64), np.eye(3).reshape(-1), np.array(rgba, dtype=np.float32))
                    sc.ngeom += 1
        wr.send(np.ascontiguousarray(renderer.render()))
    wr.close()
    print(json.dumps(dict(out=str(a.out), frames=K, robots=R)))


if __name__ == "__main__":
    main()
