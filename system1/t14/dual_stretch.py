#!/usr/bin/env python3
"""T14 item 1 part B: single person -> two robots on the stretched table (Push / Pull), size levels m, lower bound a.

Role isometry (existing constructions, exact for any length): 1.0x table = the current training table (1.4 m along the
stretch axis); a table of m x has its top 1.4 m x m long and its legs at +-(0.7 m - 0.0257264) along the axis (all else
unchanged). The source table (0.5219528 m wide) end region is mapped to each end, so each role is offset by
c(m) = 0.7 m - 0.5219528 / 2 along the axis (c(1) = 0.4390236, the registered Push spacing / 2 and the Pull leg offset).
  * Push: both robots keep the source pose; root shifted by -c / +c along the table's local x (dual_a1_layout semantics).
  * Pull: dual_pull_reference.synthesize_dual_pull_reference(single qpos, lateral_leg_offset_m = c) (object symmetrised,
    robot B = G1 sagittal mirror across the table's local XY plane, both moved outward along local z).
Single robot = system1/t14/single_shared.py (T14 step 1). Robot 0 = the source role, robot 1 = the partner.

Per level (30 Hz, all frames), geometry = the simulator's (robot links: PhysX shapes of the T10 dump; table: boxes):
  * robot-robot: smallest distance between the two robots' link shapes (MuJoCo convex collision of cross-robot pairs only,
    margin 0.5 m); "intersect" when any cross-robot pair has distance < 0;
  * robot-table penetration: largest link signed distance > 0 (any link; hands separately);
  * contact retention (CoCarry, as Table): source frames where the person's hand (SMPL-X hand vertices, original OMOMO
    parameters, captured largetable mesh) is within 0.10 m of the object and the object moves faster than 0.03 m/s; robot
    hand signed distance >= -0.02 m; robot 1 uses robot 0's (the source person's) labels and hands (mirrored for Pull:
    the partner's left hand mirrors the source's right hand);
  * lower bound a: levels m = 1.0, 0.95, ... (step 0.05); a = the last level before the first one in which the two
    robots intersect in any frame (T14 section 7 item 5).

    sonic_s1/bin/python -B system1/t14/dual_stretch.py --task push --out exports_system1/T14_references/push_sizes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import contact_stats as cs  # noqa: E402

WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
TABLE = dict(push=(WBT / "objects_widetable_plan5_training.urdf", 0), pull=(WBT / "objects_widetable_plan5_pull_training.urdf", 2))
SEQ = dict(push="sub6_largetable_033", pull="sub3_largetable_010")
HALF_SRC, LEG_IN, HALF_LEN = 0.5219528 / 2, 0.0257264, 0.7


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def offset_c(m):
    return HALF_LEN * m - HALF_SRC


def stretch_table(src, axis, m, dst):
    """Tabletop length along `axis` x m; legs at +-(0.7 m - 0.0257264) along it; everything else unchanged."""
    tree = ET.parse(src)
    root = tree.getroot()
    for el in root.iter():
        if el.tag not in ("visual", "collision"):
            continue
        name = el.get("name", "")
        o, box = el.find("origin"), el.find("geometry/box")
        xyz = [float(x) for x in o.get("xyz").split()]
        size = [float(x) for x in box.get("size").split()]
        if "tabletop" in name:
            assert abs(size[axis] - 1.4) < 1e-9, size
            size[axis] = 1.4 * m
        elif "leg" in name:
            assert abs(abs(xyz[axis]) - (HALF_LEN - LEG_IN)) < 1e-6, xyz
            xyz[axis] = np.sign(xyz[axis]) * (HALF_LEN * m - LEG_IN)
        else:
            raise ValueError(name)
        o.set("xyz", " ".join(f"{x:.9g}" for x in xyz))
        box.set("size", " ".join(f"{x:.9g}" for x in size))
    root.set("name", f"{root.get('name')}_m{m:g}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def quat_wxyz_to_mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def make_pair(task, q, m):
    """q: single [T, 43] (robot 36 + object 7, wxyz). Returns robots [T, 2, 36], object [T, 7]."""
    c = offset_c(m)
    if task == "push":
        axis = quat_wxyz_to_mat(q[:, 39:43])[:, :, 0]
        r0, r1 = q[:, :36].copy(), q[:, :36].copy()
        r0[:, :3] -= c * axis
        r1[:, :3] += c * axis
        return np.stack([r0, r1], 1), q[:, 36:43].copy()
    from holosoma_retargeting.dual_pull_reference import synthesize_dual_pull_reference
    d = synthesize_dual_pull_reference(q, fps=30, lateral_leg_offset_m=c)
    return np.stack([d.robot_a_qpos, d.robot_b_qpos], 1), d.shared_object_qpos


class PairCollision:
    """MuJoCo scene of both robots' simulator link shapes on free bodies (poses written to qpos; mocap bodies are static
    and MuJoCo does not collide two static bodies); only cross-robot pairs collide (contype / conaffinity)."""

    def __init__(self, robot, margin=0.5, depth="contact"):
        """depth: "contact" (rounds 1-2) = deepest mj_collision contact; "exact" (round 3) = mj_geomDistance per candidate pair
        (see eval_core4d_hulls.ObjectRobotDistance)."""
        import mujoco
        assert depth in ("contact", "exact")
        self.depth, self.margin = depth, margin
        import tempfile
        shapes, _, _ = cs.dump_shapes("table", "/World/envs/env_0/Robot/")
        self.links = sorted({b for b, _ in shapes})
        tmp = Path(tempfile.mkdtemp(dir=ROOT / "logs_system1/t14"))
        assets, bodies = [], []
        for i, (b, mesh) in enumerate(shapes):
            p = tmp / f"s{i}.obj"
            mesh.export(p)
            assets.append(f'<mesh name="s{i}" file="{p}"/>')
        for r in (0, 1):
            for li, link in enumerate(self.links):
                geoms = "".join(f'<geom type="mesh" mesh="s{i}" contype="{1 << r}" conaffinity="{1 << (1 - r)}" margin="{margin}"/>'
                                for i, (b, _) in enumerate(shapes) if b == link)
                bodies.append(f'<body name="r{r}_{link}"><freejoint/>{geoms}</body>')
        xml = f'<mujoco><option><flag gravity="disable"/></option><size memory="64M"/><asset>{"".join(assets)}</asset><worldbody>{"".join(bodies)}</worldbody></mujoco>'
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.robot = robot
        self.mujoco = mujoco
        for f in tmp.iterdir():
            f.unlink()  # temporary meshes written above
        tmp.rmdir()

    def min_distance(self, pair_q):
        """pair_q [T, 2, 36] (wxyz) -> per-frame smallest cross-robot distance (m, negative = intersection)."""
        T = len(pair_q)
        q = pair_q
        tl = self.robot.fk(q[:, :, :3], np.concatenate([q[:, :, 4:7], q[:, :, 3:4]], -1), q[:, :, 7:])
        out = np.full(T, np.inf)
        fromto = np.zeros(6)
        from scipy.spatial.transform import Rotation
        for t in range(T):
            k = 0
            for r in (0, 1):
                for link in self.links:
                    M = tl[link][t, r]
                    self.d.qpos[7 * k:7 * k + 3] = M[:3, 3]
                    self.d.qpos[7 * k + 3:7 * k + 7] = Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]]
                    k += 1
            self.mujoco.mj_kinematics(self.m, self.d)
            self.mujoco.mj_collision(self.m, self.d)
            if self.d.ncon and self.depth == "contact":
                out[t] = min(float(self.d.contact[i].dist) for i in range(self.d.ncon))
            elif self.d.ncon:
                cand = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in self.d.contact[:self.d.ncon]}
                out[t] = min(self.mujoco.mj_geomDistance(self.m, self.d, g1, g2, self.margin, fromto) for g1, g2 in cand)
        return out


def human_hand_sd(seq):
    """[T, 2 hands] largest signed distance of the person's SMPL-X hand vertices to the captured largetable mesh, and the
    object speed (m/s), original OMOMO data at 30 Hz."""
    import joblib
    import smplx
    import torch
    import contact_parts as cp
    for split in ("train", "test"):
        d = joblib.load(cp.OMOMO / f"{split}_diffusion_manip_seq_joints24.p")
        c = [x for x in d.values() if x["seq_name"] == seq]
        del d
        if c:
            v = c[0]
            break
    T = len(v["trans"])
    mdl = smplx.create(str(cp.SMPLX_DIR), model_type="smplx", gender=str(v["gender"]), num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        V = mdl(betas=torch.tensor(np.repeat(v["betas"], T, 0), dtype=torch.float32), global_orient=torch.tensor(v["root_orient"], dtype=torch.float32),
                body_pose=torch.tensor(v["pose_body"], dtype=torch.float32), transl=torch.tensor(v["trans"], dtype=torch.float32)).vertices.numpy().astype(np.float64)
    hv = cp.hand_vertex_ids()
    o = cp.obj("omomo_largetable")
    out = np.full((T, 2), -np.inf)
    for hi, side in enumerate(("left", "right")):
        for t in range(T):
            pl = ((V[t, hv[side]] - v["obj_trans"][t, :, 0]) @ v["obj_rot"][t]) / v["obj_scale"][t]
            dd, j = o["tree"].query(pl)
            sgn = -np.sign(np.einsum("ij,ij->i", pl - o["p"][j], o["n"][j]))
            out[t, hi] = (sgn * dd * v["obj_scale"][t]).max()
    speed = np.linalg.norm(np.gradient(v["obj_trans"][:, :, 0], axis=0), axis=1) * 30.0
    return out, speed


def level_metrics(task, robot, pc, pair, obj_q, urdf, h_sd, speed):
    obj = cs.Object(urdf, workers=1, geometry="urdf")
    T = len(pair)
    root_pos, root_quat, dof = pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:]
    op, oq = obj_q[:, :3], np.concatenate([obj_q[:, 4:7], obj_q[:, 3:4]], -1)
    Tl = robot.fk(root_pos, root_quat, dof)
    To_inv = np.linalg.inv(cs.pose_mat(op, oq))[:, None]
    H = []
    for l in cs.HANDS:
        M = To_inv @ Tl[l]
        p = robot.points[l]
        qq = np.einsum("...ij,nj->...ni", M[..., :3, :3], p) + M[..., None, :3, 3]
        H.append(obj.sdf(qq.reshape(-1, 3)).reshape(qq.shape[:-1]).max(-1))
    H = np.stack(H, -1)  # [T, 2 robots, 2 hands (L, R)]
    D = cs.link_distances(robot, obj, root_pos, root_quat, dof, op, oq)
    allmax = np.maximum(H.max(-1), D[:, :, 2:].max(-1))
    n = min(T, len(h_sd))
    cond = (h_sd[:n] >= -0.10) & (speed[:n, None] > 0.03)  # [n, 2 hands] of the source person
    ret = {}
    for r in (0, 1):
        # robot 1 inherits robot 0's labels; for the mirrored Pull partner the source left hand is its right hand
        hands = [0, 1] if (r == 0 or task == "push") else [1, 0]
        ok = np.stack([H[:n, r, hands[0]], H[:n, r, hands[1]]], 1) >= -0.02
        ret[f"r{r}"] = round(float(ok[cond].mean()), 4) if cond.any() else None
    ret["pooled"] = round(float(np.mean([ret["r0"], ret["r1"]])), 4)
    rr = pc.min_distance(pair)
    return dict(contact_retention=ret, condition_frames=int(cond.sum()),
                max_penetration_cm={f"r{r}": round(100 * float(max(allmax[:, r].max(), 0)), 2) for r in (0, 1)},
                hand_max_penetration_cm={f"r{r}": round(100 * float(max(H[:, r].max(), 0)), 2) for r in (0, 1)},
                robot_robot_min_distance_cm=round(100 * float(rr.min()), 2), robot_robot_intersect_frames=int((rr < 0).sum()),
                first_intersect_frame=int(np.argmax(rr < 0)) if (rr < 0).any() else None, palm_distance=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("push", "pull"), required=True)
    ap.add_argument("--single", type=Path, default=None, help="single-robot qpos npz (default logs_system1/t14/single/<task>/<task>_new_qpos.npz)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    single = a.single or ROOT / f"logs_system1/t14/single/{a.task}/{a.task}_new_qpos.npz"
    q = np.load(single)["qpos"].astype(np.float64)
    assert q.shape[1] == 43 and np.isfinite(q).all()
    robot = cs.Robot("physx")
    pc = PairCollision(robot)
    h_sd, speed = human_hand_sd(SEQ[a.task])
    src_urdf, axis = TABLE[a.task]
    a.out.mkdir(parents=True, exist_ok=True)
    levels, search = {}, []

    def run(m):
        tag = f"m{m:.2f}".replace(".", "p")
        urdf = stretch_table(src_urdf, axis, m, a.out / tag / f"table_{tag}.urdf")
        pair, obj_q = make_pair(a.task, q, m)
        r = level_metrics(a.task, robot, pc, pair, obj_q, urdf, h_sd, speed)
        H = r.pop("palm_distance")
        np.savez(a.out / tag / f"{a.task}_{tag}_pair_fps30.npz", robot_qpos=pair, object_qpos=obj_q, fps=np.int64(30), m=m, c=offset_c(m), palm_sd_m=H)
        r.update(m=m, c_m=round(offset_c(m), 7), urdf=str(urdf), urdf_sha256=sha(urdf), finite=bool(np.isfinite(pair).all()))
        print(a.task, json.dumps(r), flush=True)
        return r

    m = 1.0
    while m > 0.399:
        r = run(round(m, 2))
        search.append(dict(m=r["m"], robot_robot_min_distance_cm=r["robot_robot_min_distance_cm"], intersect_frames=r["robot_robot_intersect_frames"]))
        levels[r["m"]] = r
        if r["robot_robot_intersect_frames"]:
            break
        m -= 0.05
    a_level = round(search[-2]["m"], 2) if search[-1]["intersect_frames"] and len(search) > 1 else None
    for m in (1.5, 2.0):
        levels[m] = run(m)
    rec = dict(task=a.task, sequence=SEQ[a.task], single=str(single), single_sha256=sha(single), rules=__doc__, a=a_level, search=search,
               report_levels=[x for x in (a_level, 1.0, 1.5, 2.0) if x is not None], levels={str(k): v for k, v in levels.items()},
               physx_dump_scene_sha256=robot.dump_scene_sha256)
    (a.out / "sizes.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("a =", a_level, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
