#!/usr/bin/env python3
"""T14 step 9g item 1 (Kevin 2026-11-26): the verdict's non-hand penetration measured on the actual collision shapes.

Old measure (shared_eval.py, V0 "additional_nonhand_proxy_penetration_m", round-2 CollisionProxy): every robot link except the
two rubber hands is replaced by the box aligned with its link frame that bounds its collision mesh in the simulator URDF
(main_mesh_collision_rubberhand.urdf); depth = obb_sat against the object boxes.

New measure (this module): the same links, with the shapes the Stage-2 non-penetration constraint of the solver and the training
simulator use, i.e. the PhysX link shapes of the T10 Isaac dump (23 links: 21 without the hands, the head shape is on
torso_link; sim_scene.py puts exactly these shapes into the solver scene, eval_core4d_hulls.ObjectRobotDistance into the
evaluation), placed with the simulator's kinematics (contact_stats.Robot("physx")), against the object's collision boxes
(urdf_box_meshes, exact for boxes). Per frame and robot: depth = max over the pairs (non-hand link shape, object box) proposed by
mj_collision with a 0.1 m margin of -mj_geomDistance (the solver constraint's function), > 0 = penetration.

Quantity and limit unchanged: additional penetration of robot r = max over frames of
    max(0, depth_r(t) vs the shared target) - max(0, depth of the source robot at frame t vs the source object),
passed when <= 0.01 m (V0 additional_nonhand_proxy_penetration_m), the same structure, frames (T = shortest of the two solves)
and source (the evaluation's --source solve at the same frame) as the old item. Read only, CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared_eval as se  # noqa: E402  (sets the import paths)
import contact_stats as cs  # noqa: E402
import eval_core4d_hulls as eh  # noqa: E402
import partner as pt  # noqa: E402

LIMIT = 0.01
EV = {}


def distance_model(urdf):
    key = str(urdf)
    if key not in EV:
        EV[key] = eh.ObjectRobotDistance(eh.urdf_box_meshes(urdf), se.ctx()["robot"], depth="exact")
    return EV[key]


def nonhand_depth(ev, pair, obj_q, robots=(0, 1)):
    """pair [T, 2, 36] (root pos, wxyz, 29 joints), obj_q [T, 7] (pos, wxyz) -> depth [T, 2] (m, > 0 = penetration; -margin when no
    non-hand pair is within the margin) and the link of the deepest pair [T, 2] (None when none)."""
    mj = ev.mujoco
    T = len(pair)
    tl = ev.robot.fk(pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:])
    D = np.full((T, 2), -ev.margin)
    W = np.full((T, 2), None, dtype=object)
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
        mj.mj_kinematics(ev.m, ev.d)
        mj.mj_collision(ev.m, ev.d)
        cand = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in ev.d.contact[:ev.d.ncon]}
        for g1, g2 in cand:
            r1, r2 = g1 in ev.geom_robot, g2 in ev.geom_robot
            if r1 == r2:  # robot-robot or object-object: not produced by the masks, skipped for safety
                continue
            g = g1 if r1 else g2
            r, link = ev.geom_robot[g], ev.geom_link[g]
            if r not in robots or link in cs.HANDS:
                continue
            dep = -mj.mj_geomDistance(ev.m, ev.d, g1, g2, ev.margin, ft)
            if dep > D[t, r]:
                D[t, r], W[t, r] = dep, link
    return D, W


def measure(cand_dir, source_dir):
    """the new non-hand item of one evaluated level / candidate (same inputs and frames as shared_eval.evaluate)."""
    cand_dir, source_dir = Path(cand_dir), Path(source_dir)
    recs = [json.loads((cand_dir / f"r{r}" / "run_record.json").read_text()) for r in (0, 1)]
    zs = [np.load(cand_dir / f"r{r}" / f"{recs[r]['tag']}_new_qpos.npz") for r in (0, 1)]
    q = [z["qpos"].astype(np.float64)[:, :36] for z in zs]
    obj = zs[0]["object_qpos_mj"].astype(np.float64)
    T = min(len(q[0]), len(q[1]))
    pair = np.stack([x[:T] for x in q], 1)
    obj = obj[:T]
    rec = recs[0]
    srec = json.loads((source_dir / "run_record.json").read_text())
    if rec["family"] == "box":
        sz = np.load(source_dir / f"{srec['tag']}_new_qpos.npz")
        src_urdf = srec["target"]["urdf"]
    else:
        sz = np.load(source_dir / f"{srec['task']}_new_qpos.npz")
        src_urdf = pt.SOURCE_TABLE
    sq = sz["qpos"].astype(np.float64)[:T]
    s_robot, s_obj = sq[:, :36], sq[:, 36:43]
    tgt_urdf = rec["target"]["urdf"]
    D, W = nonhand_depth(distance_model(tgt_urdf), pair, obj)
    sp = np.stack([s_robot, s_robot], 1).copy()
    sp[:, 1, :3] += 1000.0  # second slot far away: only the source robot is measured
    Ds, Ws = nonhand_depth(distance_model(src_urdf), sp, s_obj, robots=(0,))
    src_pen = np.maximum(0.0, Ds[:, 0])
    out = dict(frames=int(T), target_urdf=str(tgt_urdf), source_urdf=str(src_urdf),
               source_max_depth_m=round(float(src_pen.max()), 5), source_link_at_max=Ws[int(src_pen.argmax()), 0] if src_pen.max() > 0 else None)
    for r in (0, 1):
        pen = np.maximum(0.0, D[:, r])
        add = pen - src_pen
        t = int(add.argmax())
        out[f"r{r}"] = dict(value_m=round(float(add.max()), 5), limit=LIMIT, passed=bool(add.max() <= LIMIT + 1e-10), frame=t,
                            link=W[t, r] if pen[t] > 0 else None,
                            depth_at_frame_m=round(float(pen[t]), 5), source_depth_at_frame_m=round(float(src_pen[t]), 5),
                            max_depth_m=round(float(pen.max()), 5), frames_added_over_limit=int((add > LIMIT + 1e-10).sum()),
                            links_over_limit=sorted({W[i, r] for i in np.flatnonzero(add > LIMIT + 1e-10)}))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(measure(a.cand_dir, a.source), indent=1))
