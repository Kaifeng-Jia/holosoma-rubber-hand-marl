#!/usr/bin/env python3
"""T15 (docs/system1/T15_paper_figures_zh.md): clean render material for the draft-6 figures. Common scene / render code.

Style = the U5_media pipeline (system1/render/u5/sources_snapshot: figround_common.py's NativeScene = the archived
render_wrist_reference_pairs.VisualScene with the U5 checker ground): every robot / object visual mesh of the URDFs as a mocap
body (yourdfpy forward kinematics of the stored joint values; no physics step), Unitree G1 mesh colours, the object's URDF
colour; here EVERY object gets the Table's opaque light blue (0.76, 0.80, 0.86) = the desk001 URDF material (the Push / Pull
training-table URDF material is 0.7 0.8 0.9 with alpha 0.7, translucent; Bin / Chair have none), checker ground
(.90 .92 .94 / .96 .97 .98, 0.25 m cells), headlight, fovy 45, ipd 0. Changes for T15 (task book, general spec 1-2):
  * 2560 x 1440 PNG; anti-aliasing: MSAA 8 at 2x resolution (5120 x 2880), downsampled (Lanczos) = 2x2 supersampling;
  * no dark corners: the U5 overhead light is a spot light (cone around the subject, the floor outside it is lit by the
    headlight only); here it is a DIRECTIONAL light (same downward direction, slight tilt for shadows) so the ground is lit
    evenly; background (skybox) light grey (0.96, 0.96, 0.97);
  * a person = the SMPL-X mesh as a static geom, grey (0.66, 0.67, 0.69);
  * every render also gives a segmentation image (class per pixel, CLASSES), a depth image (camera z, metres) and the pinhole
    camera (K, world-to-camera; top-left pixel centre (0, 0), u right, v down) for key-point pixels.
Classes are geom groups as well (robot 0 -> 1, robot 1 -> 2, object -> 3, person 1 -> 4, person 2 -> 5, ground -> 0), so any
of them can be hidden per render (the other robot / person in the close-ups).
Key points (project()): occluded: true when the point is outside the image / behind the camera, or at its pixel another
entity (not the point's own, not ground / background) is in front of the point, or anything (also its own surface) is more
than `tol` metres in front of it (`tol` per point: palms 0.06 m robot / 0.04 m person, pelvis 0.15 m, object centre half the
object's largest extent + 0.05 m, so points inside their own mesh are not counted as hidden by it). A point in empty space
(e.g. a table's bounding-box centre under the top) with only the ground behind it is visible.

Environment: MUJOCO_GL=egl (NVIDIA EGL; the 5070 may be shared with evaluations), sonic_s1 python.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
W, H, SS = 2560, 1440, 2
CLASSES = dict(background=0, robot0=1, robot1=2, object=3, person1=4, person2=5, ground=6)
GROUP = dict(ground=0, robot0=1, robot1=2, object=3, person1=4, person2=5)
OBJECT_RGBA = (0.76, 0.80, 0.86, 1.0)
HUMAN_RGBA = (0.66, 0.67, 0.69, 1.0)
BG = (0.96, 0.96, 0.97)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v)) + "\n")


def pose(p, quat_xyzw):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = p
    return T


def camera_for(points, width, height, azimuth, elevation, border=0.86):
    """U5 camera_for (previous_r0_helpers.py): fit points with a 7 % border (factor 0.86); fovy 45."""
    center = (points.min(0) + points.max(0)) / 2
    az, el = np.deg2rad([azimuth, elevation])
    back = np.array([-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), -np.sin(el)])
    right = np.cross(-back, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, -back)
    q = points - center
    tan = np.tan(np.deg2rad(22.5))
    distance = max(np.max(q @ back + np.abs(q @ right) / (tan * width / height * border)), np.max(q @ back + np.abs(q @ up) / (tan * border)))
    return dict(lookat=center.tolist(), distance=float(distance), azimuth=float(azimuth), elevation=float(elevation))


class T15Scene:
    """robots: number of G1 (0-2); object_urdf: URDF of the object (or None); object_mesh: a mesh file placed by a pose (person
    scenes: the captured object); persons: [(V [n, 3] world, faces)] static meshes (person 1, person 2)."""

    def __init__(self, robots=0, object_urdf=None, object_mesh=None, persons=(), object_rgba=None, width=W, height=H, ss=SS):
        import mujoco
        import trimesh
        import yourdfpy
        self.w, self.h, self.ss = width, height, ss
        self.n_robots = robots
        self.urdfs = [yourdfpy.URDF.load(str(ROBOT_URDF), load_collision_meshes=False, build_collision_scene_graph=False) for _ in range(robots)]
        self.obj_urdf = yourdfpy.URDF.load(str(object_urdf), load_collision_meshes=False, build_collision_scene_graph=False) if object_urdf else None
        xml = ET.Element("mujoco", model="t15_visual_only")
        vis = ET.SubElement(xml, "visual")
        ET.SubElement(vis, "global", offwidth=str(width * ss), offheight=str(height * ss), fovy="45", ipd="0")
        ET.SubElement(vis, "quality", shadowsize="8192", offsamples="8")
        ET.SubElement(vis, "headlight", diffuse=".42 .42 .42", ambient=".36 .36 .36", specular=".04 .04 .04")
        ET.SubElement(vis, "map", znear="0.01", zfar="60")
        asset = ET.SubElement(xml, "asset")
        ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1=" ".join(map(str, BG)), rgb2=" ".join(map(str, BG)), width="64", height="64")
        ET.SubElement(asset, "texture", name="ground", type="2d", builtin="checker", rgb1=".90 .92 .94", rgb2=".96 .97 .98", width="256", height="256")
        ET.SubElement(asset, "material", name="neutral_ground", texture="ground", texrepeat="80 80", reflectance="0", specular="0", shininess="0")
        world = ET.SubElement(xml, "worldbody")
        ET.SubElement(world, "light", directional="true", pos="0 0 6", dir="0.25 0.15 -1", diffuse=".42 .42 .42", specular="0 0 0", castshadow="true")
        self.ground_z = 0.0
        ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 .01", material="neutral_ground", contype="0", conaffinity="0", group=str(GROUP["ground"]))
        assets, self.mapping, mesh_ids, self.mesh_records = {}, [], {}, []

        def add_urdf(u, who, src):
            for node in u.scene.graph.nodes_geometry:
                _, g = u.scene.graph.get(node)
                mesh = u.scene.geometry[g]
                ident = (who if who == "object" else "robot", g)
                if ident not in mesh_ids:
                    mid = f"mesh_{len(mesh_ids)}"
                    mesh_ids[ident] = mid
                    assets[mid + ".stl"] = mesh.export(file_type="stl")
                    ET.SubElement(asset, "mesh", name=mid, file=mid + ".stl")
                if hasattr(mesh.visual, "main_color"):
                    color = np.asarray(mesh.visual.main_color) / 255
                else:
                    mat = mesh.visual.material
                    color = np.asarray(mat.diffuse) / 255 if getattr(mat, "diffuse", None) is not None else np.asarray(OBJECT_RGBA)
                if who == "object":  # every object the Table's opaque light blue (the Push / Pull training-table URDF is 0.7 0.8 0.9 alpha 0.7)
                    color = np.asarray(object_rgba or OBJECT_RGBA)
                body = ET.SubElement(world, "body", name=f"visual_{len(self.mapping)}", mocap="true")
                ET.SubElement(body, "geom", type="mesh", mesh=mesh_ids[ident], rgba=" ".join(f"{c:.6g}" for c in color), contype="0", conaffinity="0",
                              mass="0", group=str(GROUP[who]))
                self.mapping.append((who, node, body.attrib["name"]))
            for nd in ET.parse(src).findall(".//visual/geometry/mesh"):
                p = Path(nd.attrib["filename"])
                p = p if p.is_absolute() else Path(src).parent / p
                self.mesh_records.append(dict(urdf=str(src), file=str(p.resolve()), sha256=sha(p), scale=nd.attrib.get("scale", "1 1 1")))
        for r, u in enumerate(self.urdfs):
            add_urdf(u, f"robot{r}", ROBOT_URDF)
        if self.obj_urdf is not None:
            add_urdf(self.obj_urdf, "object", object_urdf)
        self.obj_mesh_body = None
        if object_mesh is not None:
            assets["capture_obj.stl"] = trimesh.load(str(object_mesh), force="mesh").export(file_type="stl")
            ET.SubElement(asset, "mesh", name="capture_obj", file="capture_obj.stl")
            b = ET.SubElement(world, "body", name="capture_obj_body", mocap="true")
            ET.SubElement(b, "geom", type="mesh", mesh="capture_obj", rgba=" ".join(map(str, object_rgba or OBJECT_RGBA)), contype="0", conaffinity="0", mass="0",
                          group=str(GROUP["object"]))
            self.obj_mesh_body = "capture_obj_body"
            self.mesh_records.append(dict(mesh=str(object_mesh), sha256=sha(object_mesh)))
        for i, (V, F) in enumerate(persons):
            name = f"person{i + 1}"
            assets[name + ".stl"] = trimesh.Trimesh(np.asarray(V, float), np.asarray(F), process=False).export(file_type="stl")
            ET.SubElement(asset, "mesh", name=name, file=name + ".stl")
            ET.SubElement(world, "geom", name=name, type="mesh", mesh=name, rgba=" ".join(map(str, HUMAN_RGBA)), contype="0", conaffinity="0", group=str(GROUP[name]))
        self.n_persons = len(persons)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"), assets)
        self.data = mujoco.MjData(self.model)
        self.mocap = [self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)] for _, _, n in self.mapping]
        gcls = np.full(self.model.ngeom, CLASSES["background"], np.uint8)
        for g in range(self.model.ngeom):
            grp = int(self.model.geom_group[g])
            gcls[g] = {0: CLASSES["ground"], 1: CLASSES["robot0"], 2: CLASSES["robot1"], 3: CLASSES["object"], 4: CLASSES["person1"], 5: CLASSES["person2"]}[grp]
        self.geom_class = gcls
        self.r_ss = mujoco.Renderer(self.model, height * ss, width * ss)
        self.r_1 = mujoco.Renderer(self.model, height, width)
        self.bases = {}

    def set_ground(self, z):
        gid = [g for g in range(self.model.ngeom) if self.model.geom_group[g] == 0][0]
        self.model.geom_pos[gid][2] = z
        self.ground_z = float(z)

    def set_pose(self, robots=(), obj=None, obj_mesh_pose=None):
        """robots: [q (36,) = root xyz, root quat wxyz, 29 joints] per robot; obj: (7,) xyz + quat wxyz (URDF object);
        obj_mesh_pose: 4x4 of the captured-object mesh."""
        import mujoco
        from scipy.spatial.transform import Rotation
        for r, q in enumerate(robots):
            q = np.asarray(q, float)
            self.bases[f"robot{r}"] = pose(q[:3], q[3:7][[1, 2, 3, 0]])
            self.urdfs[r].update_cfg(q[7:36])
        if obj is not None:
            obj = np.asarray(obj, float)
            self.bases["object"] = pose(obj[:3], obj[3:7][[1, 2, 3, 0]])
        for (who, node, _), mid in zip(self.mapping, self.mocap):
            u = self.urdfs[int(who[-1])] if who.startswith("robot") else self.obj_urdf
            tf = self.bases[who] @ u.scene.graph.get(node)[0]
            self.data.mocap_pos[mid] = tf[:3, 3]
            self.data.mocap_quat[mid] = Rotation.from_matrix(tf[:3, :3]).as_quat()[[3, 0, 1, 2]]
        if obj_mesh_pose is not None and self.obj_mesh_body is not None:
            mid = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.obj_mesh_body)]
            self.data.mocap_pos[mid] = obj_mesh_pose[:3, 3]
            self.data.mocap_quat[mid] = Rotation.from_matrix(obj_mesh_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            self.bases["capture_object"] = np.asarray(obj_mesh_pose, float)
        mujoco.mj_forward(self.model, self.data)

    def link(self, r, name):
        return self.bases[f"robot{r}"] @ self.urdfs[r].get_transform(name)

    def mesh_bounds(self, who=None):
        """world corner points of every visual mesh's bounds (U5 bounds()), optionally of one entity."""
        pts = []
        for w_, node, _ in self.mapping:
            if who is not None and w_ not in who:
                continue
            u = self.urdfs[int(w_[-1])] if w_.startswith("robot") else self.obj_urdf
            tf = self.bases[w_] @ u.scene.graph.get(node)[0]
            b = u.scene.geometry[u.scene.graph.get(node)[1]].bounds
            c = np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]])
            pts.append(c @ tf[:3, :3].T + tf[:3, 3])
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    def object_center(self):
        """centre of the object's visual-mesh bounding box (object frame) in the world, and the object frame origin."""
        if self.obj_urdf is None:
            return None, None
        pts = []
        for w_, node, _ in self.mapping:
            if w_ == "object":
                tf = self.obj_urdf.scene.graph.get(node)[0]
                b = self.obj_urdf.scene.geometry[self.obj_urdf.scene.graph.get(node)[1]].bounds
                c = np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]])
                pts.append(c @ tf[:3, :3].T + tf[:3, 3])
        pts = np.concatenate(pts)
        local = (pts.min(0) + pts.max(0)) / 2
        B = self.bases["object"]
        return B[:3, :3] @ local + B[:3, 3], B[:3, 3].copy()

    def render(self, cam, show, stem=None):
        """cam: lookat / distance / azimuth / elevation; show: set of class names drawn (others hidden).
        Saves <stem>.png and <stem>.seg.png when stem is given; returns dict(rgb, seg, depth, camera)."""
        import mujoco
        from PIL import Image
        opt = mujoco.MjvOption()
        for name, g in GROUP.items():
            opt.geomgroup[g] = 1 if (name in show or name == "ground") else 0
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["distance"], cam["azimuth"], cam["elevation"]
        self.r_ss.update_scene(self.data, camera=c, scene_option=opt)
        big = self.r_ss.render()
        rgb = np.asarray(Image.fromarray(big).resize((self.w, self.h), Image.LANCZOS))
        self.r_1.update_scene(self.data, camera=c, scene_option=opt)
        gl = self.r_1.scene.camera[0]
        f = np.array(gl.forward)
        f /= np.linalg.norm(f)
        up = np.array(gl.up)
        up /= np.linalg.norm(up)
        right = np.cross(f, up)
        E = np.eye(4)
        E[:3, :3] = np.stack([right, -up, f])
        E[:3, 3] = -E[:3, :3] @ np.array(gl.pos)
        fy = self.h * gl.frustum_near / (gl.frustum_top - gl.frustum_bottom)
        K = np.array([[fy, 0, (self.w - 1) / 2], [0, fy, (self.h - 1) / 2], [0, 0, 1]])
        self.r_1.enable_depth_rendering()
        depth = self.r_1.render().copy()
        self.r_1.disable_depth_rendering()
        self.r_1.enable_segmentation_rendering()
        sraw = self.r_1.render().copy()
        self.r_1.disable_segmentation_rendering()
        seg = np.zeros((self.h, self.w), np.uint8)
        isg = (sraw[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (sraw[..., 0] >= 0)
        seg[isg] = self.geom_class[sraw[..., 0][isg]]
        camera = dict(settings=dict(cam), resolution=[self.w, self.h], K=K.tolist(), world_to_camera=E.tolist(), near=float(gl.frustum_near),
                      far=float(gl.frustum_far), fovy_deg=45.0, projection="perspective",
                      pixel_convention="top-left pixel centre (0, 0), u right, v down; camera x right, y down, z forward",
                      supersampling=f"{self.ss}x{self.ss} (rendered {self.w * self.ss} x {self.h * self.ss}, MSAA 8, Lanczos downsample)")
        if stem is not None:
            stem = Path(stem)
            stem.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(stem.with_suffix(".png"), optimize=False)
            Image.fromarray(seg).save(stem.with_name(stem.name + ".seg.png"))
        subj = np.isin(seg, [CLASSES[k] for k in ("robot0", "robot1", "object", "person1", "person2")])
        rows = np.flatnonzero(subj.any(1))
        cols = np.flatnonzero(subj.any(0))
        frac = dict(height=round(float((rows[-1] - rows[0] + 1) / self.h), 4) if len(rows) else 0.0,
                    width=round(float((cols[-1] - cols[0] + 1) / self.w), 4) if len(cols) else 0.0,
                    bbox_uv=[int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])] if len(rows) else None,
                    pixels=round(float(subj.mean()), 4))
        return dict(rgb=rgb, seg=seg, depth=depth, camera=camera, subject_extent=frac)

    @staticmethod
    def project(out, points, owners, tols):
        """points [N, 3] world; owners: class name per point; tols: metres per point. -> list of dicts."""
        E, K = np.asarray(out["camera"]["world_to_camera"]), np.asarray(out["camera"]["K"])
        h, w = out["seg"].shape
        res = []
        for p, who, tol in zip(np.asarray(points, float), owners, tols):
            c = E[:3, :3] @ p + E[:3, 3]
            front = bool(c[2] > 0)
            uv = (K @ c)[:2] / c[2] if front else np.array([np.nan, np.nan])
            inside = front and 0 <= uv[0] <= w - 1 and 0 <= uv[1] <= h - 1
            occ, why = True, "outside the image or behind the camera"
            if inside:
                u, v = int(round(uv[0])), int(round(uv[1]))
                cls_here = int(out["seg"][v, u])
                dz = float(out["depth"][v, u])
                name_here = next(k for k, x in CLASSES.items() if x == cls_here)
                other = cls_here not in (CLASSES[who], CLASSES["ground"], CLASSES["background"]) and dz < c[2]
                far = dz < c[2] - tol
                occ = bool(other or far)
                why = None if not occ else (f"{name_here} in front ({c[2] - dz:.3f} m)" if other else f"{name_here} surface {c[2] - dz:.3f} m in front (> {tol} m)")
            res.append(dict(world_xyz_m=p.round(6).tolist(), uv=None if not front else [round(float(uv[0]), 2), round(float(uv[1]), 2)],
                            camera_z_m=round(float(c[2]), 4), in_image=bool(inside), occluded=bool(occ), occlusion_reason=why, owner=who))
        return res

    def close(self):
        self.r_ss.close()
        self.r_1.close()


def save_preview(png, out_jpg, width=1280, quality=85):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    h = round(im.height * width / im.width)
    Path(out_jpg).parent.mkdir(parents=True, exist_ok=True)
    im.resize((width, h), Image.LANCZOS).save(out_jpg, quality=quality)


def versions():
    import importlib.metadata as md
    return {x: md.version(x) for x in ("mujoco", "numpy", "yourdfpy", "trimesh", "Pillow")} | dict(python=sys.version.split()[0])


def gl_info():
    from OpenGL import GL
    return {k: GL.glGetString(v).decode() for k, v in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION))}


if __name__ == "__main__":  # smoke: one robot at the origin and a box, the four corners' ground brightness
    import os
    assert os.environ.get("MUJOCO_GL") == "egl"
    s = T15Scene(robots=1, object_urdf=None)
    q = np.zeros(36)
    q[2], q[3] = 0.793, 1.0
    s.set_pose([q])
    cam = camera_for(s.mesh_bounds(), W, H, 135, -22)
    out = s.render(cam, {"robot0"}, stem=Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    g = out["rgb"].astype(float).mean(-1)
    corners = [g[:200, :200].mean(), g[:200, -200:].mean(), g[-200:, :200].mean(), g[-200:, -200:].mean()]
    print("corners", np.round(corners, 1), "centre ground", round(float(g[out["seg"] == CLASSES["ground"]].mean()), 1), "subject", out["subject_extent"], gl_info())
    s.close()
