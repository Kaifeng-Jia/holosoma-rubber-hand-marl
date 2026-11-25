#!/usr/bin/env python3
"""T14 step 9c: where do the two robots intersect? Diagnostic only (read only, CPU). For a pair npz (step9_eval --pair-out):
per frame the exact smallest cross-robot distance (dual_stretch.PairCollision, depth "exact", the evaluation's quantity) and
the two links of the closest geom pair; summary per level: intersecting frames, their ranges, and the link pairs counted over
the intersecting frames. The walking segment of the level's gates file is given for reference.

    sonic_s1/bin/python -B system1/t14/objects/step9_intersect.py --pairs <tag>=<pair npz>[@<gates json>] ... --out <json>
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shared_eval as se  # noqa: E402  (import paths)


def per_frame(pc, pair_q):
    from scipy.spatial.transform import Rotation
    mj = pc.mujoco
    q = pair_q
    tl = pc.robot.fk(q[:, :, :3], np.concatenate([q[:, :, 4:7], q[:, :, 3:4]], -1), q[:, :, 7:])
    T = len(q)
    dist, links = np.full(T, np.inf), [None] * T
    fromto = np.zeros(6)
    names = [mj.mj_id2name(pc.m, mj.mjtObj.mjOBJ_BODY, i) for i in range(pc.m.nbody)]
    for t in range(T):
        k = 0
        for r in (0, 1):
            for link in pc.links:
                M = tl[link][t, r]
                pc.d.qpos[7 * k:7 * k + 3] = M[:3, 3]
                pc.d.qpos[7 * k + 3:7 * k + 7] = Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]]
                k += 1
        mj.mj_kinematics(pc.m, pc.d)
        mj.mj_collision(pc.m, pc.d)
        if not pc.d.ncon:
            continue
        cand = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in pc.d.contact[:pc.d.ncon]}
        best = min(((mj.mj_geomDistance(pc.m, pc.d, g1, g2, pc.margin, fromto), g1, g2) for g1, g2 in cand))
        dist[t] = best[0]
        b = sorted(names[pc.m.geom_bodyid[g]] for g in best[1:])
        links[t] = " / ".join(b)
    return dist, links


def ranges(idx):
    out, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append([int(start), int(prev)])
            start = prev = i
    if start is not None:
        out.append([int(start), int(prev)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pc = se.ctx()["pc"]
    res = {}
    for item in a.pairs:
        tag, rest = item.split("=", 1)
        pair, _, gates = rest.partition("@")
        z = np.load(pair)
        d, links = per_frame(pc, z["robot_qpos"].astype(np.float64))
        bad = np.flatnonzero(d < 0)
        walk = json.loads(Path(gates).read_text())["diag"].get("walk") if gates else None
        res[tag] = dict(pair=pair, frames=int(len(d)), intersect_frames=int(len(bad)), min_distance_cm=round(100 * float(d.min()), 3),
                        intersect_ranges=ranges(bad), link_pairs=Counter(links[i] for i in bad).most_common(8),
                        walk=None if walk is None else [walk["w0"], walk["w1"]], distance_cm_every5=np.round(100 * np.where(np.isfinite(d), d, 50.0)[::5], 1).tolist())
        print(tag, res[tag]["intersect_frames"], res[tag]["intersect_ranges"], res[tag]["link_pairs"][:4], "walk", res[tag]["walk"], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(rules=__doc__, levels=res), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
