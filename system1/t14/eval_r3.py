#!/usr/bin/env python3
"""T14 round 3: per-level metrics of two-robot references, one measure for every task and every compared reference
(round 3, round 2 re-measured, current, OmniRetarget as is). Read only, CPU.

Geometry = the simulator's: robot = PhysX link shapes of the T10 dump with the simulator URDF FK; object = Table / Bin /
Chair PhysX pieces x k (eval_core4d_hulls.dump_object_meshes), Push / Pull the stretched training-table boxes of the level.
Distances = exact convex signed distances (depth="exact": mj_collision proposes candidate pairs within the margin,
mj_geomDistance gives each pair's distance; equal to the Minkowski-difference depth, test_sim_scene.py). Rounds 1-2 used
the deepest mj_collision contact, which overstates some box-hull depths (Table k = 1: 1.19 cm reported, exact 0.53 cm).

  * contact retention (CoCarry): source frames where the person's hand is within 0.10 m of the captured mesh and the
    object moves faster than 0.03 m/s; retained when the robot hand's signed distance >= -0.02 m. CORE4D: robot r <->
    person r + 1, 15 Hz source frame t <-> reference frame 2t, pooled over both robots' frames; Push / Pull: 30 Hz, robot 1
    inherits the source labels (Pull: left / right swapped by the mirror), pooled = mean of the two robots;
  * max penetration: largest penetration depth over all robot-object pairs (any link), per robot, cm; frames penetrating;
  * robot-robot: smallest exact distance between the two robots' shapes (dual_stretch.PairCollision, depth exact);
  * solve success: every solve of the level finished with a finite trajectory and the object locked;
  * contact-position error: contact_gap.py (report only).
Push / Pull lower bound a (T14 section 7 item 5): from m = 1.0 down in 0.05 steps, the last level before the first one at
which the two robots' shapes intersect in any frame.

    sonic_s1/bin/python -B system1/t14/eval_r3.py --task bin --ref r3_k1.0=<pair npz>@1.0 ... --out <json>
    sonic_s1/bin/python -B system1/t14/eval_r3.py --task push --single r3=<single qpos npz> --search-a --levels 1.0 1.5 2.0 --pairs-out <dir> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
import contact_gap as cg  # noqa: E402
import dual_stretch as ds  # noqa: E402
import eval_core4d_hulls as eh  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pen_summary(P):
    return dict(max_penetration_cm={f"r{r}": round(100 * float(max(P[:, r].max(), 0.0)), 3) for r in (0, 1)},
                frames_penetrating={f"r{r}": round(float((P[:, r] > 0).mean()), 4) for r in (0, 1)})


def core4d_level(task, ctx, pair, obj_q, k):
    robot, pc, h_sd, speed, human, offs = ctx
    ev = eh.ObjectRobotDistance(eh.dump_object_meshes(task, k), robot, depth="exact")
    H, P = ev.frames(pair, obj_q)
    T = len(pair)
    t15 = np.arange(min(len(h_sd), (T + 1) // 2))
    cond = (h_sd[t15] >= -0.10) & (speed[t15, None, None] > 0.03)
    robo = H[2 * t15] >= -0.02
    ret = {f"r{r}": (round(float(robo[:, r][cond[:, r]].mean()), 4) if cond[:, r].any() else None) for r in (0, 1)}
    ret["pooled"] = round(float(robo[cond].mean()), 4) if cond.any() else None
    rr = pc.min_distance(pair)
    gap, _ = cg.core4d_arrays(task, robot, offs, human, pair, obj_q, k)
    return dict(frames=T, condition_frames=int(cond.sum()), contact_retention=ret, **pen_summary(P),
                robot_robot_min_distance_cm=round(100 * float(rr.min()), 3), robot_robot_intersect_frames=int((rr < 0).sum()),
                finite=bool(np.isfinite(pair).all()), contact_position_error=gap)


def omomo_level(task, ctx, q, m, out_dir=None):
    robot, pc, h_sd, speed, human, offs = ctx
    src_urdf, axis = ds.TABLE[task]
    tag = f"m{m:.2f}".replace(".", "p")
    udir = out_dir / tag if out_dir is not None else ROOT / f"logs_system1/t14/r3/eval_urdf/{task}_{tag}"
    urdf = ds.stretch_table(src_urdf, axis, m, udir / f"table_{tag}.urdf")
    pair, obj_q = ds.make_pair(task, q, m)
    ev = eh.ObjectRobotDistance(eh.urdf_box_meshes(urdf), robot, depth="exact")
    H, P = ev.frames(pair, obj_q)
    n = min(len(pair), len(h_sd))
    cond = (h_sd[:n] >= -0.10) & (speed[:n, None] > 0.03)
    ret = {}
    for r in (0, 1):
        hands = [0, 1] if (r == 0 or task == "push") else [1, 0]
        ok = np.stack([H[:n, r, hands[0]], H[:n, r, hands[1]]], 1) >= -0.02
        ret[f"r{r}"] = round(float(ok[cond].mean()), 4) if cond.any() else None
    ret["pooled"] = round(float(np.mean([ret["r0"], ret["r1"]])), 4)
    rr = pc.min_distance(pair)
    gap, _ = cg.omomo_arrays(task, robot, offs, human, pair, obj_q, m, urdf)
    res = dict(m=m, c_m=round(ds.offset_c(m), 7), frames=len(pair), condition_frames=int(cond.sum()), contact_retention=ret, **pen_summary(P),
               robot_robot_min_distance_cm=round(100 * float(rr.min()), 3), robot_robot_intersect_frames=int((rr < 0).sum()),
               finite=bool(np.isfinite(pair).all()), contact_position_error=gap, urdf=str(urdf), urdf_sha256=sha(urdf))
    if out_dir is not None:
        p = udir / f"{task}_{tag}_pair_fps30.npz"
        np.savez(p, robot_qpos=pair, object_qpos=obj_q, fps=np.int64(30), m=m, c=ds.offset_c(m))
        res.update(pair=str(p), pair_sha256=sha(p))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("table", "bin", "chair", "push", "pull"), required=True)
    ap.add_argument("--ref", action="append", default=[], help="CORE4D: name=pair npz@k ; Push / Pull: name=single npz@m[@fps50]")
    ap.add_argument("--single", default=None, help="Push / Pull: name=single qpos npz whose levels are made here (pairs saved to --pairs-out)")
    ap.add_argument("--search-a", action="store_true")
    ap.add_argument("--levels", type=float, nargs="*", default=[])
    ap.add_argument("--pairs-out", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    robot = eh.cs.Robot("physx")
    pc = ds.PairCollision(robot, depth="exact")
    offs = cg.palm_offsets()
    omomo = a.task in ("push", "pull")
    if omomo:
        h_sd, speed = ds.human_hand_sd(ds.SEQ[a.task])
        human = cg.human_omomo(a.task)
    else:
        h_sd, speed = eh.human_hand_sd(a.task)
        human = cg.human_core4d(a.task)
    ctx = (robot, pc, h_sd, speed, human, offs)
    res, search = {}, []
    for spec in a.ref:
        name, rest = spec.split("=", 1)
        parts = rest.split("@")
        if omomo:
            q = cg.omomo_single(parts[0], len(parts) > 2 and parts[2] == "fps50")
            r = omomo_level(a.task, ctx, q, float(parts[1]))
        else:
            z = np.load(parts[0])
            r = core4d_level(a.task, ctx, z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64), float(parts[1]))
        r.update(path=parts[0], sha256=sha(parts[0]))
        res[name] = r
        print(name, json.dumps({k: r[k] for k in ("contact_retention", "max_penetration_cm", "robot_robot_min_distance_cm")}), flush=True)
    a_level = None
    if a.single:
        name, path = a.single.split("=", 1)
        q = np.load(path)["qpos"].astype(np.float64)
        done = {}
        if a.search_a:
            m = 1.0
            while m > 0.399:
                mm = round(m, 2)
                done[mm] = omomo_level(a.task, ctx, q, mm, a.pairs_out)
                search.append(dict(m=mm, robot_robot_min_distance_cm=done[mm]["robot_robot_min_distance_cm"], intersect_frames=done[mm]["robot_robot_intersect_frames"]))
                print("search", search[-1], flush=True)
                if done[mm]["robot_robot_intersect_frames"]:
                    break
                m -= 0.05
            a_level = round(search[-2]["m"], 2) if search[-1]["intersect_frames"] and len(search) > 1 else None
        for m in ([a_level] if a_level is not None else []) + list(a.levels):
            r = done.get(round(m, 2)) or omomo_level(a.task, ctx, q, round(m, 2), a.pairs_out)
            r.update(single=path, single_sha256=sha(path))
            res[f"{name}_m{m:.2f}"] = r
            print(f"{name}_m{m:.2f}", json.dumps({k: r[k] for k in ("contact_retention", "max_penetration_cm", "robot_robot_min_distance_cm")}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(task=a.task, rules=__doc__, a=a_level, a_search=search, refs=res, physx_dump_scene_sha256=robot.dump_scene_sha256), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
