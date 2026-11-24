#!/usr/bin/env python3
"""T14 round 4 (section 9 item 5): Push leg check. For a Push single-robot reference, made into the two-robot pair at m:
  * robot: per frame and per simulator link except the two rubber hands, the smallest exact distance (PhysX link shapes,
    mj_geomDistance per candidate pair within 0.3 m) to the stretched training table of that level; per robot;
  * human: per frame and per body part (body_parts.LINK_TO_SMPLX, SMPL-X skinning segments, flat-hand OMOMO SMPL-X mapped
    into the OMOMO_new world by the palm fit), the smallest distance of the part's vertices to the captured table
    (InterMimic largetable.obj, the human's object, dense surface samples);
  * per link: robot minimum (cm, both robots; robot 1 is the copy of robot 0) and the human part minimum (cm);
    flagged when a robot non-hand link comes within 1 cm or its human part within 10 cm (frames listed).

    sonic_s1/bin/python -B system1/t14/push_legs.py --single <push qpos npz> --m 1.0 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import body_parts as bp  # noqa: E402
import contact_stats as cs  # noqa: E402
import dual_stretch as ds  # noqa: E402
import eval_core4d_hulls as eh  # noqa: E402

MARGIN = 0.3


def robot_link_distances(pair, obj_q, urdf):
    import mujoco
    from scipy.spatial.transform import Rotation
    robot = cs.Robot("physx")
    ev = eh.ObjectRobotDistance(eh.urdf_box_meshes(urdf), robot, margin=MARGIN, depth="exact")
    tl = robot.fk(pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:])
    T = len(pair)
    D = {(r, l): np.full(T, MARGIN) for r in (0, 1) for l in ev.links}
    ft = np.zeros(6)
    for t in range(T):
        ev.d.qpos[0:3], ev.d.qpos[3:7] = obj_q[t, :3], obj_q[t, 3:7]
        k = 1
        for r in (0, 1):
            for link in ev.links:
                M = tl[link][t, r]
                ev.d.qpos[7 * k:7 * k + 3] = M[:3, 3]
                ev.d.qpos[7 * k + 3:7 * k + 7] = Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]]
                k += 1
        mujoco.mj_kinematics(ev.m, ev.d)
        mujoco.mj_collision(ev.m, ev.d)
        for g1, g2 in {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in ev.d.contact[:ev.d.ncon]}:
            g = g1 if g1 in ev.geom_robot else g2
            key = (ev.geom_robot[g], ev.geom_link[g])
            D[key][t] = min(D[key][t], mujoco.mj_geomDistance(ev.m, ev.d, g1, g2, MARGIN, ft))
    return D


def human_part_distances(task="push"):
    import single_shared as ss
    import trimesh
    from scipy.spatial import cKDTree
    from holosoma_retargeting.src.utils import load_intermimic_data
    seq = ds.SEQ[task]
    ss.omomo_palms(seq)
    V = ss.omomo_palms.last["vertices_omomo_new"]
    _, obj = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))
    To = cs.pose_mat(obj[:, 4:7], obj[:, [1, 2, 3, 0]])
    mesh = trimesh.load(ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/largetable/largetable.obj", force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, int(mesh.area / 0.003 ** 2), seed=0)
    tree = cKDTree(pts)
    parts = bp.smplx_part_vertices()
    out = {}
    for name, vid in parts.items():
        loc = np.einsum("tji,tvj->tvi", To[:, :3, :3], V[:, vid] - To[:, None, :3, 3])
        d, _ = tree.query(loc.reshape(-1, 3))
        out[name] = d.reshape(len(V), len(vid)).min(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=Path, required=True)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    q = np.load(a.single)["qpos"].astype(np.float64)
    pair, obj_q = ds.make_pair("push", q, a.m)
    src, axis = ds.TABLE["push"]
    urdf = ds.stretch_table(src, axis, a.m, ROOT / f"logs_system1/t14/r4/push_legs_urdf/table_m{a.m:g}.urdf")
    D = robot_link_distances(pair, obj_q, urdf)
    Hd = human_part_distances()
    n = min(len(pair), len(next(iter(Hd.values()))))
    rows, flagged = [], []
    for link, (part, _) in bp.LINK_TO_SMPLX.items():
        rmin = {r: float(D[(r, link)][:n].min()) for r in (0, 1) if (r, link) in D}
        if not rmin:
            continue
        h = Hd[part][:n]
        row = dict(link=link, human_part=part, robot_min_cm={f"r{r}": round(100 * v, 2) for r, v in rmin.items()},
                   robot_min_frame={f"r{r}": int(D[(r, link)][:n].argmin()) for r in rmin}, human_min_cm=round(100 * float(h.min()), 2), human_min_frame=int(h.argmin()))
        rows.append(row)
        for r in rmin:
            close = np.flatnonzero(D[(r, link)][:n] < 0.01)
            if len(close):
                flagged.append(dict(kind="robot link within 1 cm", robot=r, link=link, human_part=part, frames=[int(close[0]), int(close[-1])], count=int(len(close)),
                                    robot_min_cm=round(100 * rmin[r], 2), human_part_min_cm_same_frames=round(100 * float(h[close].min()), 2)))
        hc = np.flatnonzero(h < 0.10)
        if len(hc):
            flagged.append(dict(kind="human part within 10 cm", link=link, human_part=part, frames=[int(hc[0]), int(hc[-1])], count=int(len(hc)),
                                human_min_cm=round(100 * float(h.min()), 2), robot_min_cm_same_frames={f"r{r}": round(100 * float(D[(r, link)][hc].min()), 2) for r in rmin}))
    res = dict(rules=__doc__, single=str(a.single), m=a.m, urdf=str(urdf), frames=n, links=rows, flagged=flagged, margin_m=MARGIN,
               note="distances >= 30 cm are reported as 30 cm (candidate margin)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for r in rows:
        print(r["link"], r["human_part"], r["robot_min_cm"], r["human_min_cm"])
    print("flagged", json.dumps(flagged, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
