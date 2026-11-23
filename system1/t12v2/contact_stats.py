#!/usr/bin/env python3
"""T12-v2 part 1: what the D policies actually do with the object. Read only, CPU, saved per-step states of the final
evaluations only (no simulation, no policy inference).

Runs: D (paper), Push / Pull / Bin (bucket) / Chair / Table, seeds 721, 722, 723, +/-2 and +/-8, Isaac (the laptop-5070 final
evaluation of T03: c2_main, for Push / Chair d_final_5070) and MuJoCo (T10 setting, logs_system1/t10/mj_eval[_other]); every
episode of every run (nominal + the 50 perturbed states). The summary table uses the 50 perturbed states of each run (the
+/-2 / +/-8 test sets, as the completion counts); nominal episodes are in the per-episode table only.

Rules (fixed here before any number was looked at):
1. Geometry. Robot: main_mesh_collision_rubberhand.urdf collision geometry (meshes, cylinders, spheres as yourdfpy builds the
   collision scene), placed by forward kinematics from the saved root pose (pelvis, xyzw) and the 29 joint angles (URDF joint
   order = the saved dof order, checked). Object: the collision geometry of the run's object URDF (summary.json object_urdf),
   placed by the saved object pose.
2. Signed distance of a link to the object = max over the link's sample points of the object's signed distance at the point
   (positive inside, the trimesh convention of T12). A link is in contact when it is >= -1.5 cm (within 1.5 cm or penetrating).
   * Hand links (left_rubber_hand_link, right_rubber_hand_link): sample points as T12 = every k-th vertex of the hand collision
     mesh, k = max(1, n_vertices // 1500).
   * Non-hand links (legs: hip / knee / ankle links; torso: pelvis, torso_link, head_link; arms: shoulder_yaw, elbow, wrist
     roll / pitch / yaw links): points spread evenly over every collision surface of the link: 20 random surface samples
     per cm2 (seed 0), then one kept per 7 mm voxel (the first one), plus every k-th mesh vertex (k = n // 50); every surface
     point is then within about 1.2 cm (the voxel diagonal) of a sample point, typically 0.5 cm.
   * Object signed distance: Push / Pull / Table objects are unions of axis-aligned boxes in the object frame: exact union-of-
     boxes distance outside (min over boxes), max of the boxes' inside depths inside. Bin / Chair objects are single watertight
     meshes: trimesh signed distance precomputed on a 5 mm grid over the object bounds + 3 cm, trilinear interpolation;
     outside the grid = not in contact. A link whose bounding sphere is farther than 1.5 cm from the object's bounding box is
     not in contact (exact, only skips work).
3. Executed rows of an episode: as T12 / U5: steps + 1 rows for a terminated episode, steps rows for a completed one (the extra
   terminal row is not used). Row k is reference frame min(k, N-1) (the evaluator alignment; the saved reference_frame
   array is checked).
4. Statistics window: from the reference's first contact to the last executed row. Reference first contact = first reference
   frame in which any hand of either reference robot is in contact with the reference object by rules 1-2 (reference pelvis
   pose + reference joint angles through the same URDF, reference object pose). An episode that ended before it has an empty
   window (fractions empty).
5. Per episode, over the window rows:
   * frac_r{0,1}_hand: fraction of rows in which at least one hand of that robot is in contact;
   * frac_both_hands: fraction of rows in which both robots have at least one hand in contact at the same time;
   * frac_r{0,1}_nonhand: fraction of rows in which at least one non-hand link of that robot is in contact; also split into
     legs / torso / arms, arms further into upper_arm (shoulder_yaw, elbow) / wrist (wrist roll / pitch / yaw, the links next
     to the hand), frac_r{0,1}_nonhand_excl_wrist (legs, torso, upper arm), and per hand (left / right).
   Over all executed rows: object_max_rise_cm = max object height - row-0 height; reference_max_rise_cm = the same for the
   full reference; reference_max_rise_executed_cm = the reference over the executed rows only.
6. The same window statistics for the reference itself (task constant) are given as a separate row.

Switch --geometry physx (default urdf = the rules above, unchanged): the same statistics on the collision shapes both simulators
actually used, from the T10 Isaac export (logs_system1/t10/isaac/d_<task>/dump: scene.json + collision_convexes.npz, the shapes
the T10 MuJoCo scene was built from): robot mesh links = the convex hull PhysX cooked for each, URDF cylinders = PhysX capsules,
pelvis sphere; Bin / Chair = the union of the 16 convex pieces of PhysX's convex decomposition (5 mm grid of the max over the
pieces' exact signed distances); box objects unchanged (checked equal to the exported cubes). All links, hands included, are
sampled by the voxel rule. Output in <out>/physx_geometry/.

    OMP_NUM_THREADS=1 hssim_s1/bin/python -B system1/t12v2/contact_stats.py --out results_system1/T12_videos_v2 --workers 16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src/holosoma"
ROBOT_URDF = DATA / "holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
TASKS = ("push", "pull", "bucket", "chair", "table")
TASK_ZH = dict(push="Table Pushing", pull="Table Pulling", bucket="Bin Transport", chair="Chair Righting", table="Table Transport")
SEEDS, LEVELS = (721, 722, 723), (2, 8)
HANDS = ("left_rubber_hand_link", "right_rubber_hand_link")
GROUPS = dict(legs=("left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
                    "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link"),
              torso=("pelvis", "torso_link", "head_link"),
              arms=("left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
                    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"))
ARM_SPLIT = dict(upper_arm=("left_shoulder_yaw_link", "left_elbow_link", "right_shoulder_yaw_link", "right_elbow_link"),
                 wrist=("left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
                        "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"))
CONTACT_M = -0.015
VOXEL = 0.007
GRID_H, GRID_MARGIN = 0.005, 0.03
CACHE = ROOT / "logs_system1/t12v2/cache"
DUMP = ROOT / "logs_system1/t10/isaac"


def dump_shapes(task, match):
    """(body name, trimesh in the body frame) of the exported PhysX collision shapes whose path contains match."""
    d = DUMP / f"d_{task}" / "dump"
    j = json.loads((d / "scene.json").read_text())
    conv = np.load(d / "collision_convexes.npz")
    out = []
    for sh in j["shapes"]:
        if match not in sh["path"]:
            continue
        T = np.array(sh["prim_in_body"])
        R = T[:3, :3]
        sc = np.linalg.norm(R, axis=0)
        Rn = R / sc
        body = sh["body"].split("/")[-1]
        if sh["type"] == "Xform":
            for mi, m in enumerate(sh["meshes"]):
                for hi in range(m["n_hulls"]):
                    v = conv[f"{sh['npz_key']}_mesh{mi}_hull{hi}"]
                    out.append((body, trimesh.convex.convex_hull((R @ v.T).T + T[:3, 3])))
        elif sh["type"] == "Capsule":
            ax = {"X": 0, "Y": 1, "Z": 2}[sh["axis"]]
            c = trimesh.creation.capsule(height=sh["height"] * sc[ax], radius=sh["radius"] * max(sc[(ax + 1) % 3], sc[(ax + 2) % 3]), count=[24, 24])
            A = trimesh.geometry.align_vectors([0, 0, 1], Rn[:, ax])
            A[:3, 3] = T[:3, 3]
            out.append((body, c.apply_transform(A)))
        elif sh["type"] == "Sphere":
            out.append((body, trimesh.creation.icosphere(subdivisions=4, radius=sh["radius"] * sc.max()).apply_translation(T[:3, 3])))
        elif sh["type"] == "Cube":
            A = np.eye(4)
            A[:3, :3] = Rn
            A[:3, 3] = T[:3, 3]
            out.append((body, trimesh.creation.box(extents=sh["size"] * sc, transform=A)))
        else:
            raise ValueError(sh["type"])
    return out, sha(d / "scene.json"), sha(d / "collision_convexes.npz")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- robot
def quat_xyzw_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = (q[..., i] for i in range(4))
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)  # noqa: E702
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)  # noqa: E702
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)  # noqa: E702
    return m


def pose_mat(pos, quat_xyzw):
    T = np.zeros(np.shape(pos)[:-1] + (4, 4))
    T[..., :3, :3] = quat_xyzw_to_mat(quat_xyzw)
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1
    return T


def axis_angle(axis, ang):
    """(..., 4, 4) rotation about a fixed unit axis by angles ang (...)."""
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(ang)[..., None, None], np.cos(ang)[..., None, None]
    T = np.zeros(np.shape(ang) + (4, 4))
    T[..., :3, :3] = np.eye(3) + s * K + (1 - c) * (K @ K)
    T[..., 3, 3] = 1
    return T


def voxel_samples(m, voxel=None, density_per_cm2=20, seed=0):
    voxel = VOXEL if voxel is None else voxel
    p, _ = trimesh.sample.sample_surface(m, count=int(math.ceil(m.area * 1e4 * density_per_cm2)), seed=seed)
    _, keep = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)
    return np.concatenate([p[np.sort(keep)], np.asarray(m.vertices)[:: max(1, len(m.vertices) // 50)]])


class Robot:
    def __init__(self, geometry="urdf"):
        self.u = yourdfpy.URDF.load(str(ROBOT_URDF), load_meshes=True, build_scene_graph=True, load_collision_meshes=True,
                                    build_collision_scene_graph=True)
        self.dofs = list(self.u.actuated_joint_names)
        self.root = self.u.base_link
        self.joints = []  # topological order
        done = {self.root}
        pending = list(self.u.robot.joints)
        while pending:
            for j in list(pending):
                if j.parent in done:
                    self.joints.append(j)
                    done.add(j.child)
                    pending.remove(j)
        cs = self.u.collision_scene
        geo = {}
        self.geometry = geometry
        if geometry == "urdf":
            for node in cs.graph.nodes_geometry:
                link = cs.graph.transforms.parents[node]
                T, g = cs.graph.get(frame_to=node, frame_from=link)
                geo.setdefault(link, []).append(cs.geometry[g].copy().apply_transform(T))
        else:
            shapes, self.dump_scene_sha256, self.dump_convex_sha256 = dump_shapes("table", "/World/envs/env_0/Robot/")
            for link, m in shapes:
                geo.setdefault(link, []).append(m)
        self.points, self.sphere = {}, {}
        nonhand = [x for g in GROUPS.values() for x in g]
        assert set(geo) == set(HANDS) | set(nonhand), sorted(set(geo) ^ (set(HANDS) | set(nonhand)))
        for link, ms in geo.items():
            m = ms[0] if len(ms) == 1 else trimesh.util.concatenate(ms)
            if link in HANDS and geometry == "urdf":
                assert len(ms) == 1
                v = np.asarray(m.vertices)
                p = v[:: max(1, len(v) // 1500)]
            else:
                p = voxel_samples(m)
            c = p.mean(axis=0)
            self.points[link] = np.asarray(p, dtype=np.float64)
            self.sphere[link] = (c, float(np.linalg.norm(p - c, axis=1).max()))
        self.links = list(HANDS) + nonhand

    def fk(self, root_pos, root_quat_xyzw, dof):
        """world transforms (..., 4, 4) of every collision link; leading dims of the inputs are kept."""
        lead = np.shape(root_pos)[:-1]
        q = {n: np.asarray(dof)[..., i] for i, n in enumerate(self.dofs)}
        T = {self.root: pose_mat(root_pos, root_quat_xyzw)}
        for j in self.joints:
            M = T[j.parent] @ np.broadcast_to(np.eye(4) if j.origin is None else j.origin, lead + (4, 4))
            if j.type in ("revolute", "continuous"):
                M = M @ axis_angle(j.axis, q[j.name])
            elif j.type != "fixed":
                raise ValueError(j.type)
            T[j.child] = M
        return {k: T[k] for k in self.links}


# ---------------------------------------------------------------------------------------------------------------- object
class Object:
    def __init__(self, urdf: Path, workers=1, geometry="urdf", task=None):
        o = yourdfpy.URDF.load(str(urdf), load_meshes=True, build_scene_graph=True, load_collision_meshes=True, build_collision_scene_graph=True)
        self.urdf = urdf
        self.mesh = o.collision_scene.to_geometry()
        self.lo, self.hi = self.mesh.bounds
        self.geometry, self.hulls, self.grid_file = geometry, None, None
        cols = [c for l in o.robot.links for c in l.collisions]
        if geometry == "physx":
            shapes, self.dump_scene_sha256, self.dump_convex_sha256 = dump_shapes(task, "/World/envs/env_0/Object/")
            if all(c.geometry.box is not None for c in cols):
                ex = trimesh.util.concatenate([m for _, m in shapes])
                assert len(shapes) == len(cols) and np.allclose(np.sort(ex.vertices, axis=0), np.sort(self.mesh.vertices, axis=0), atol=1e-5), "exported cubes differ from the URDF boxes"
            else:
                self.hulls = [m for _, m in shapes]
                self.mesh = trimesh.util.concatenate(self.hulls)
                self.lo, self.hi = self.mesh.bounds
        if all(c.geometry.box is not None for c in cols):
            self.kind = "boxes"
            for c in cols:
                assert np.allclose(c.origin[:3, :3], np.eye(3)), "box rotated in the object frame"
            self.box_c = np.array([c.origin[:3, 3] for c in cols])
            self.box_h = np.array([np.asarray(c.geometry.box.size) / 2 for c in cols])
        else:
            assert len(cols) == 1 and cols[0].geometry.mesh is not None and (self.hulls is not None or self.mesh.is_watertight)
            self.kind = "grid"
            self.grid_lo = self.lo - GRID_MARGIN
            shape = np.ceil((self.hi - self.lo + 2 * GRID_MARGIN) / GRID_H).astype(int) + 1
            key = hashlib.sha256(self.mesh.vertices.tobytes() + self.mesh.faces.tobytes() + str((GRID_H, GRID_MARGIN)).encode()).hexdigest()[:16]
            f = CACHE / f"{urdf.stem}_{key}.npz"
            if f.exists():
                self.grid = np.load(f)["sdf"]
            else:
                ax = [self.grid_lo[i] + GRID_H * np.arange(shape[i]) for i in range(3)]
                pts = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
                chunks = np.array_split(pts, max(1, len(pts) // 20000))
                t0 = time.time()
                with Pool(workers) as pool:
                    if self.hulls is None:
                        sd = pool.map(_sd_chunk, [(self.mesh, c) for c in chunks])
                    else:  # union of convex pieces: max of the pieces' signed distances (positive inside)
                        sd = pool.map(_sd_union_chunk, [(self.hulls, c) for c in chunks])
                self.grid = np.concatenate(sd).reshape(shape)
                CACHE.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f, sdf=self.grid)
                print(f"grid {urdf.name} {shape} {time.time() - t0:.0f}s", flush=True)
            self.grid_file = f

    def sdf(self, p):
        """signed distance (m, positive inside) at points (..., 3) in the object frame."""
        if self.kind == "boxes":
            d = np.abs(p[..., None, :] - self.box_c) - self.box_h  # (..., B, 3)
            out = np.linalg.norm(np.maximum(d, 0), axis=-1)
            ins = np.minimum(d.max(axis=-1), 0)
            return (-(out + ins)).max(axis=-1)
        g = (p - self.grid_lo) / GRID_H
        flat = g.reshape(-1, 3).T
        v = map_coordinates(self.grid, flat, order=1, mode="constant", cval=-1.0)
        return v.reshape(p.shape[:-1])


def _sd_chunk(args):
    mesh, pts = args
    return trimesh.proximity.signed_distance(mesh, pts)


def _sd_union_chunk(args):
    hulls, pts = args
    return np.max([trimesh.proximity.signed_distance(h, pts) for h in hulls], axis=0)


def link_distances(robot: Robot, obj: Object, root_pos, root_quat, dof, obj_pos, obj_quat):
    """signed distance (m) of every collision link to the object, shape (F, 2, n_links); -inf where the link's bounding sphere
    is farther than 1.5 cm from the object's bounding box (cannot be in contact)."""
    F = len(dof)
    Tl = robot.fk(root_pos, root_quat, dof)  # (F, 2, 4, 4)
    To_inv = np.linalg.inv(pose_mat(obj_pos, obj_quat))[:, None]  # (F, 1, 4, 4)
    out = np.full((F, 2, len(robot.links)), -np.inf)
    for li, link in enumerate(robot.links):
        M = To_inv @ Tl[link]  # link -> object frame
        c, r = robot.sphere[link]
        cc = M[..., :3, :3] @ c + M[..., :3, 3]
        gap = np.linalg.norm(np.maximum(np.maximum(obj.lo - cc, cc - obj.hi), 0), axis=-1)
        cand = np.argwhere(gap <= r - CONTACT_M + 1e-6)
        if not len(cand):
            continue
        P = robot.points[link]
        step = max(1, 4_000_000 // len(P))
        for s in range(0, len(cand), step):
            cs = cand[s:s + step]
            Mc = M[cs[:, 0], cs[:, 1]]
            pts = np.einsum("cij,pj->cpi", Mc[:, :3, :3], P) + Mc[:, None, :3, 3]
            out[cs[:, 0], cs[:, 1], li] = obj.sdf(pts).max(axis=1)
    return out


# ---------------------------------------------------------------------------------------------------------------- runs
def runs():
    rows = {f: list(csv.DictReader((ROOT / f"results_system1/T03/{f}/per_seed_summary.csv").open())) for f in ("c2_main", "d_final_5070")}
    out = []
    for t in TASKS:
        f = "d_final_5070" if t in ("push", "chair") else "c2_main"
        for s in SEEDS:
            for lvl in LEVELS:
                r = [x for x in rows[f] if x["variant"] == "D" and x["task"] == t and int(x["training_seed"]) == s and x["perturbation_level_cm"] == str(lvl)]
                assert len(r) == 1, (t, s, lvl)
                mj = ROOT / "logs_system1/t10" / ("mj_eval" if t in ("table", "bucket") else "mj_eval_other") / f"d_{t}_seed{s}_pm{lvl}"
                out.append(dict(task=t, seed=s, level=lvl, simulator="isaac", eval_dir=ROOT / r[0]["eval_dir"], expected_completed=int(r[0]["n_completed"]),
                                checkpoint_sha256=r[0]["checkpoint_sha256"], states_sha256=r[0]["states_sha256"], t03_table=f))
                out.append(dict(task=t, seed=s, level=lvl, simulator="mujoco", eval_dir=mj, expected_completed=None,
                                checkpoint_sha256=r[0]["checkpoint_sha256"], states_sha256=r[0]["states_sha256"], t03_table=f))
    t10 = {(r["task"], r["scheme"], int(r["training_seed"]), int(r["perturbation_level_cm"])): int(r["mujoco_n_completed"])
           for r in csv.DictReader((ROOT / "results_system1/T10_sim2sim/isaac_vs_mujoco_all.csv").open()) if r["scheme"] == "D"}
    for r in out:
        if r["simulator"] == "mujoco":
            r["expected_completed"] = t10[(r["task"], "D", r["seed"], r["level"])]
    return out


def executed_rows(e, length):
    return min(e["steps"] if e["completed_reference"] else e["steps"] + 1, length)


def reason(e):
    return "+".join(k for k, v in (e.get("terminal_terms") or {}).items() if v and k != "reference_horizon") or None


def window_stats(D, robot, k0, n):
    """D: (F, 2, n_links) signed distances; window rows k0..n-1."""
    C = D[k0:n] >= CONTACT_M
    w = len(C)
    if w == 0:
        return dict(window_rows=0)
    li = {l: i for i, l in enumerate(robot.links)}
    hand = C[:, :, [li[h] for h in HANDS]]  # (w, 2, 2)
    out = dict(window_rows=w)
    for r in range(2):
        out[f"frac_r{r}_hand"] = float(hand[:, r].any(axis=1).mean())
    out["frac_both_hands"] = float((hand[:, 0].any(axis=1) & hand[:, 1].any(axis=1)).mean())
    for r in range(2):
        nh = C[:, r, [li[x] for g in GROUPS.values() for x in g]]
        out[f"frac_r{r}_nonhand"] = float(nh.any(axis=1).mean())
    for r in range(2):
        for g, names in list(GROUPS.items()) + list(ARM_SPLIT.items()):
            out[f"frac_r{r}_{g}"] = float(C[:, r, [li[x] for x in names]].any(axis=1).mean())
        ex = GROUPS["legs"] + GROUPS["torso"] + ARM_SPLIT["upper_arm"]
        out[f"frac_r{r}_nonhand_excl_wrist"] = float(C[:, r, [li[x] for x in ex]].any(axis=1).mean())
        for side, h in zip(("left", "right"), HANDS):
            out[f"frac_r{r}_{side}_hand"] = float(C[:, r, li[h]].mean())
    return out


G = {}  # robot, objects, references; set before the worker pool forks

def run_job(job):
    robot, objs, refs = G["robot"], G["objs"], G["refs"]
    s = json.loads((job["eval_dir"] / "summary.json").read_text())
    assert s["checkpoint_sha256"] == job["checkpoint_sha256"] and s["states_sha256"] == job["states_sha256"] and s.get("complete", True)
    obj = objs[job["task"]]
    ref = refs[job["task"]]
    rows, per_step = [], {}
    for e in s["episodes"]:
        tr = Path(e["trajectory"])
        with np.load(tr, allow_pickle=False) as z:
            a = {k: z[k] for k in ("root_pos", "root_quat_xyzw", "dof_pos", "object_pos_w", "object_quat_xyzw", "reference_object_pos_w", "reference_frame")}
        n = executed_rows(e, len(a["dof_pos"]))
        if not np.array_equal(a["reference_frame"][:n], np.minimum(np.arange(n), ref["N"] - 1)):
            raise ValueError(f"{tr}: saved reference frame is not min(row, N-1)")
        D = link_distances(robot, obj, a["root_pos"][:n], a["root_quat_xyzw"][:n], a["dof_pos"][:n], a["object_pos_w"][:n], a["object_quat_xyzw"][:n])
        z0 = a["object_pos_w"][0, 2]
        st = window_stats(D, robot, ref["k0"], n)
        rows.append(dict(task=job["task"], simulator=job["simulator"], level_cm=job["level"], seed=job["seed"], state_id=e["state_id"], kind=e["kind"],
                         completed=bool(e["completed_reference"]), executed_steps=e["steps"], termination=reason(e), executed_rows=n,
                         window_start_row=ref["k0"], **st,
                         object_max_rise_cm=round(100 * float(a["object_pos_w"][:n, 2].max() - z0), 2),
                         reference_max_rise_cm=ref["rise_cm"],
                         reference_max_rise_executed_cm=round(100 * float(a["reference_object_pos_w"][:n, 2].max() - a["reference_object_pos_w"][0, 2]), 2),
                         trajectory=str(tr), trajectory_sha256=sha(tr)))
        per_step[e["state_id"]] = (D >= CONTACT_M)
    tag = f"{job['task']}_seed{job['seed']}_pm{job['level']}_{job['simulator']}"
    f = G["per_step"] / f"{tag}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, links=np.array(robot.links), **per_step)
    return job, rows, s


def load_references():
    sys.path.insert(0, str(ROOT))
    import torch
    from system1.sonic.task_refs import load_task_reference

    out = {}
    for t in TASKS:
        ref, bodies, dofs, meta = load_task_reference(t)
        smp = ref.sample(torch.arange(ref.num_frames))
        pel = bodies.index("pelvis")
        out[t] = dict(N=int(ref.num_frames), root_pos=smp["agent_body_pos_w"][:, :, pel].numpy().astype(np.float64),
                      root_quat=smp["agent_body_quat_w"][:, :, pel].numpy().astype(np.float64), dof=smp["agent_joint_pos"].numpy().astype(np.float64),
                      obj_pos=smp["object_pos_w"].numpy().astype(np.float64), obj_quat=smp["object_quat_w"].numpy().astype(np.float64), meta=meta, dofs=dofs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None, help="smoke: TASK,SEED,LEVEL,SIM")
    ap.add_argument("--geometry", choices=("urdf", "physx"), default="urdf")
    a = ap.parse_args()
    if a.geometry == "physx":
        a.out = a.out / "physx_geometry"
    t0 = time.time()
    refs = load_references()
    robot = Robot(a.geometry)
    assert robot.dofs == list(refs["push"]["dofs"]), "URDF joint order differs from the saved dof order"
    summaries0 = {}
    objs = {}
    for t in TASKS:
        s0 = json.loads((next(r for r in runs() if r["task"] == t)["eval_dir"] / "summary.json").read_text())
        objs[t] = Object(DATA / s0["object_urdf"], workers=a.workers, geometry=a.geometry, task=t)
        summaries0[t] = s0["object_urdf"]
    ref_rows = []
    for t in TASKS:
        R = refs[t]
        D = link_distances(robot, objs[t], R["root_pos"], R["root_quat"], R["dof"], R["obj_pos"], R["obj_quat"])
        hand_any = (D[:, :, :2] >= CONTACT_M).any(axis=(1, 2))
        k0 = int(np.flatnonzero(hand_any)[0]) if hand_any.any() else None
        assert k0 is not None, f"{t}: the reference hands never touch the object"
        R["k0"] = k0
        R["rise_cm"] = round(100 * float(R["obj_pos"][:, 2].max() - R["obj_pos"][0, 2]), 2)
        ref_rows.append(dict(task=t, frames=R["N"], first_contact_frame=k0, first_contact_s=round(k0 / 50, 2), reference_max_rise_cm=R["rise_cm"],
                             **window_stats(D, robot, k0, R["N"]), reference_file=R["meta"]["file"], reference_sha256=R["meta"]["sha256"],
                             loader=R["meta"]["loader"]))
        print("reference", json.dumps(ref_rows[-1]), flush=True)
    G.update(robot=robot, objs=objs, refs=refs, per_step=ROOT / "logs_system1/t12v2" / ("per_step" if a.geometry == "urdf" else "per_step_physx"))
    jobs = runs()
    if a.only:
        t, s, l, sim = a.only.split(",")
        jobs = [j for j in jobs if (j["task"], j["seed"], j["level"], j["simulator"]) == (t, int(s), int(l), sim)]
    eps, run_rec = [], []
    with Pool(a.workers) as pool:
        for job, rows, s in pool.imap_unordered(run_job, jobs):
            done = sum(r["completed"] for r in rows if r["kind"] == "perturbed")
            if done != job["expected_completed"]:
                raise ValueError(f"{job}: {done} completed perturbed episodes, expected {job['expected_completed']}")
            eps += rows
            run_rec.append(dict(task=job["task"], seed=job["seed"], level_cm=job["level"], simulator=job["simulator"], eval_dir=str(job["eval_dir"].relative_to(ROOT)),
                                summary_sha256=sha(job["eval_dir"] / "summary.json"), checkpoint_sha256=s["checkpoint_sha256"], states_sha256=s["states_sha256"],
                                episodes=len(rows), completed_perturbed=done, source_table=job["t03_table"] if job["simulator"] == "isaac" else "T10 isaac_vs_mujoco_all.csv"))
            print(f"{time.time() - t0:.0f}s done {job['task']} {job['seed']} pm{job['level']} {job['simulator']} completed {done}", flush=True)
    order = {t: i for i, t in enumerate(TASKS)}
    eps.sort(key=lambda r: (order[r["task"]], r["simulator"], r["level_cm"], r["seed"], r["state_id"]))
    a.out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in eps for k in r))
    with (a.out / "contact_episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(eps)
    with (a.out / "reference_contact.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref_rows[0]))
        w.writeheader()
        w.writerows(ref_rows)
    metrics = [k for k in keys if k.startswith("frac_")] + ["object_max_rise_cm", "reference_max_rise_executed_cm"]
    summ = []
    for t in TASKS:
        for sim in ("isaac", "mujoco"):
            for lvl in LEVELS:
                sel = [r for r in eps if r["task"] == t and r["simulator"] == sim and r["level_cm"] == lvl and r["kind"] == "perturbed"]
                for grp, rs in (("completed", [r for r in sel if r["completed"]]), ("failed", [r for r in sel if not r["completed"]]), ("all", sel)):
                    row = dict(task=t, simulator=sim, level_cm=lvl, episodes=grp, n=len(rs), n_empty_window=sum(1 for r in rs if r["window_rows"] == 0))
                    for m in metrics:
                        v = np.array([r[m] for r in rs if r.get(m) is not None], dtype=float)
                        row[f"{m}_mean"] = round(float(v.mean()), 4) if len(v) else None
                        row[f"{m}_median"] = round(float(np.median(v)), 4) if len(v) else None
                    row["reference_max_rise_cm"] = refs[t]["rise_cm"]
                    summ.append(row)
    with (a.out / "contact_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    prov = dict(geometry=a.geometry, script=str(Path(__file__).relative_to(ROOT)), script_sha256=sha(__file__),
                physx_dump=None if a.geometry == "urdf" else dict(robot=dict(dump="logs_system1/t10/isaac/d_table/dump", scene_sha256=robot.dump_scene_sha256,
                                                                             convexes_sha256=robot.dump_convex_sha256),
                                                                  objects={t: dict(dump=f"logs_system1/t10/isaac/d_{t}/dump", scene_sha256=objs[t].dump_scene_sha256,
                                                                                   convexes_sha256=objs[t].dump_convex_sha256, n_pieces=len(objs[t].hulls) if objs[t].hulls else None)
                                                                           for t in TASKS}), robot_urdf=str(ROBOT_URDF.relative_to(ROOT)), robot_urdf_sha256=sha(ROBOT_URDF),
                objects={t: dict(object_urdf=summaries0[t], sha256=sha(DATA / summaries0[t]), kind=objs[t].kind,
                                 grid_cache=str(objs[t].grid_file.relative_to(ROOT)) if objs[t].kind == "grid" else None,
                                 grid_cache_sha256=sha(objs[t].grid_file) if objs[t].kind == "grid" else None) for t in TASKS},
                references={t: dict(refs[t]["meta"], frames=refs[t]["N"], first_contact_frame=refs[t]["k0"]) for t in TASKS},
                rules=dict(contact_m=CONTACT_M, nonhand_voxel_m=VOXEL, grid_h_m=GRID_H, grid_margin_m=GRID_MARGIN, hand_vertex_cap=1500),
                sample_points={l: int(len(p)) for l, p in robot.points.items()}, runs=sorted(run_rec, key=lambda r: (order[r["task"]], r["simulator"], r["level_cm"], r["seed"])),
                wall_s=round(time.time() - t0))
    (a.out / "contact_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"{time.time() - t0:.0f}s wrote {len(eps)} episodes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
