#!/usr/bin/env python3
"""T14 item 2: per-level metrics of two-person CORE4D references (Bin, Chair; also usable for Table) with exact convex
distances. Read only, CPU.

Geometry = the simulator's: object = the PhysX convex pieces of the T10 Isaac dump (d_<task>: Bin / Chair 16 pieces of the
training object, Table 5 boxes), scaled by k about the object frame origin (k = level / 1.0x object); robot links = the
PhysX link shapes of the same dump. MuJoCo convex collision with a 0.1 m margin, robot-object pairs only:
  * hand signed distance = - (smallest distance between that hand's shape and any object piece), positive = penetration;
    below -0.1 m when nothing is within the margin;
  * penetration of a frame = largest depth over all robot-object pairs (any link), per robot;
  * contact retention (CoCarry, as Table): source 15 Hz frame t <-> reference 30 Hz frame 2t; condition = the person's hand
    (any SMPL-X hand vertex) within 0.10 m of the raw CORE4D mesh and the object moving faster than 0.03 m/s; retained when
    the robot hand signed distance >= -0.02 m; robot r <-> person r + 1;
  * robot-robot: dual_stretch.PairCollision (smallest distance between the two robots' shapes, < 0 = intersection).

    sonic_s1/bin/python -B system1/t14/eval_core4d_hulls.py --task chair --ref k1.0=<pair npz>@1.0 ... --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "system1/t14"))
import contact_stats as cs  # noqa: E402
import contact_parts as cp  # noqa: E402
import dual_stretch as ds  # noqa: E402
from table_shared import TASKS  # noqa: E402

DUMP_TASK = dict(table="table", bin="bucket", chair="chair")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump_object_meshes(task, k):
    """The PhysX pieces of the task's training object (T10 dump), scaled by k about the object frame origin."""
    out = []
    for _, mesh in cs.dump_shapes(DUMP_TASK[task], "/World/envs/env_0/Object/")[0]:
        m = mesh.copy()
        m.vertices = m.vertices * k
        out.append(m)
    return out


def urdf_box_meshes(urdf):
    """The collision boxes of a box-union URDF as meshes in the object frame (exact for boxes)."""
    import trimesh
    import yourdfpy
    u = yourdfpy.URDF.load(str(urdf), load_collision_meshes=True, build_collision_scene_graph=True)
    out = []
    for link in u.robot.links:
        for c in link.collisions:
            assert c.geometry.box is not None
            out.append(trimesh.creation.box(extents=c.geometry.box.size, transform=c.origin))
    return out


class ObjectRobotDistance:
    """depth="contact" (rounds 1-2): signed distance of a pair = the deepest mj_collision contact; this is not the minimum
    separating distance for some box-hull configurations (T14 round 3: 1.19 cm reported where the exact depth is 0.45 cm).
    depth="exact" (round 3): mj_collision only proposes the candidate pairs (margin); each pair's signed distance is
    mj_geomDistance, the function of the retargeter's constraint, equal to the Minkowski-difference depth (test_sim_scene)."""

    def __init__(self, object_meshes, robot, margin=0.1, robot_dump="table", depth="contact"):
        import mujoco
        assert depth in ("contact", "exact")
        self.depth = depth
        self.mujoco, self.robot = mujoco, robot
        obj = [(None, m) for m in object_meshes]
        rob, _, _ = cs.dump_shapes(robot_dump, "/World/envs/env_0/Robot/")
        self.links = sorted({b for b, _ in rob})
        tmp = Path(tempfile.mkdtemp(dir=ROOT / "logs_system1/t14"))
        assets, bodies = [], []
        for i, (_, m) in enumerate(obj):
            m.export(tmp / f"o{i}.obj")
            assets.append(f'<mesh name="o{i}" file="{tmp / f"o{i}.obj"}"/>')
        for i, (_, mesh) in enumerate(rob):
            mesh.export(tmp / f"s{i}.obj")
            assets.append(f'<mesh name="s{i}" file="{tmp / f"s{i}.obj"}"/>')
        og = "".join(f'<geom type="mesh" mesh="o{i}" contype="4" conaffinity="3" margin="{margin}"/>' for i in range(len(obj)))
        bodies.append(f'<body name="object"><freejoint/>{og}</body>')
        self.geom_owner = []
        for r in (0, 1):
            for link in self.links:
                gs = "".join(f'<geom name="r{r}_{link}_{i}" type="mesh" mesh="s{i}" contype="{1 << r}" conaffinity="4" margin="{margin}"/>'
                             for i, (b, _) in enumerate(rob) if b == link)
                bodies.append(f'<body name="r{r}_{link}"><freejoint/>{gs}</body>')
        xml = f'<mujoco><size memory="128M"/><asset>{"".join(assets)}</asset><worldbody>{"".join(bodies)}</worldbody></mujoco>'
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        for f in tmp.iterdir():
            f.unlink()  # temporary meshes written above
        tmp.rmdir()
        self.obj_geoms = set(range(len(obj)))
        self.geom_robot, self.geom_link = {}, {}
        for g in range(self.m.ngeom):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name:
                self.geom_robot[g] = int(name[1])
                self.geom_link[g] = name[3:].rsplit("_", 1)[0]
        self.margin = margin

    def frames(self, pair, obj_q):
        """pair [T, 2, 36] wxyz, obj_q [T, 7] wxyz -> hand signed distance [T, 2, 2], penetration depth [T, 2]."""
        T = len(pair)
        tl = self.robot.fk(pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:])
        H = np.full((T, 2, 2), -self.margin)
        P = np.zeros((T, 2))
        fromto = np.zeros(6)
        for t in range(T):
            self.d.qpos[0:3], self.d.qpos[3:7] = obj_q[t, :3], obj_q[t, 3:7]
            k = 1
            for r in (0, 1):
                for link in self.links:
                    M = tl[link][t, r]
                    self.d.qpos[7 * k:7 * k + 3] = M[:3, 3]
                    self.d.qpos[7 * k + 3:7 * k + 7] = Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]]
                    k += 1
            self.mujoco.mj_kinematics(self.m, self.d)
            self.mujoco.mj_collision(self.m, self.d)
            if self.depth == "contact":
                pairs = [(c.geom1, c.geom2, c.dist) for c in self.d.contact[:self.d.ncon]]
            else:
                cand = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in self.d.contact[:self.d.ncon]}
                pairs = [(g1, g2, self.mujoco.mj_geomDistance(self.m, self.d, g1, g2, self.margin, fromto)) for g1, g2 in cand]
            for g1, g2, dist in pairs:
                g = g1 if g1 in self.geom_robot else g2
                r, link = self.geom_robot[g], self.geom_link[g]
                P[t, r] = max(P[t, r], -dist)
                for hi, h in enumerate(cs.HANDS):
                    if link == h:
                        H[t, r, hi] = max(H[t, r, hi], -dist)
        return H, P


def human_hand_sd(task):
    """[15 Hz frames, 2 persons, 2 hands] largest signed distance of the SMPL-X hand vertices to the raw CORE4D mesh; object
    origin speed (m/s)."""
    seq = Path("/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions") / TASKS[task]["raw"]
    hv = cp.hand_vertex_ids()
    To = np.load(seq / "smooth_objposes.npy")
    o = cp.obj(TASKS[task]["obj_key"])
    out = np.full((len(To), 2, 2), -np.inf)
    for pi in (1, 2):
        V = np.load(seq / f"person{pi}_poses.npz", allow_pickle=True)["arr_0"].item()["vertices"].astype(np.float64)
        for hi, side in enumerate(("left", "right")):
            for t in range(min(len(V), len(To))):
                R, x = To[t, :3, :3], To[t, :3, 3]
                pl = (V[t, hv[side]] - x) @ R
                d, j = o["tree"].query(pl)
                out[t, pi - 1, hi] = (-np.sign(np.einsum("ij,ij->i", pl - o["p"][j], o["n"][j])) * d).max()
    return out, np.linalg.norm(np.gradient(To[:, :3, 3], axis=0), axis=1) * 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(DUMP_TASK), required=True)
    ap.add_argument("--ref", action="append", required=True, help="name=pair npz@k")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    robot = cs.Robot("physx") if a.task == "table" else None
    if robot is None:  # the robot shapes of this task's dump (identical robot, checked by T10)
        robot = cs.Robot("physx")
    pc = ds.PairCollision(robot)
    h_sd, speed = human_hand_sd(a.task)
    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for spec in a.ref:
        name, rest = spec.split("=", 1)
        path, k = rest.split("@")
        k = float(k)
        z = np.load(path)
        pair, obj_q = z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64)
        ev = ObjectRobotDistance(dump_object_meshes(a.task, k), robot)
        H, P = ev.frames(pair, obj_q)
        T = len(pair)
        t15 = np.arange(min(len(h_sd), (T + 1) // 2))
        cond = (h_sd[t15] >= -0.10) & (speed[t15, None, None] > 0.03)
        robo = H[2 * t15] >= -0.02
        ret = {f"r{r}": (round(float(robo[:, r][cond[:, r]].mean()), 4) if cond[:, r].any() else None) for r in (0, 1)}
        ret["pooled"] = round(float(robo[cond].mean()), 4) if cond.any() else None
        rr = pc.min_distance(pair)
        res[name] = dict(path=str(path), sha256=sha(path), k=k, frames=T, condition_frames=int(cond.sum()), contact_retention=ret,
                         max_penetration_cm={f"r{r}": round(100 * float(P[:, r].max()), 2) for r in (0, 1)},
                         frames_penetrating={f"r{r}": round(float((P[:, r] > 0).mean()), 4) for r in (0, 1)},
                         hand_median_signed_distance_cm_in_condition_frames={f"r{r}": round(100 * float(np.median(H[2 * t15][:, r][cond[:, r]])), 2) if cond[:, r].any() else None for r in (0, 1)},
                         robot_robot_min_distance_cm=round(100 * float(rr.min()), 3), robot_robot_intersect_frames=int((rr < 0).sum()), finite=bool(np.isfinite(pair).all()))
        print(name, json.dumps(res[name]), flush=True)
    (a.out / "metrics.json").write_text(json.dumps(dict(task=a.task, rules=__doc__, refs=res, physx_dump_task=DUMP_TASK[a.task]), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
