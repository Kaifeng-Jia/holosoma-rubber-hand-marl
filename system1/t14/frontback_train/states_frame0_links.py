#!/usr/bin/env python3
"""Kevin 2026-11-27 item 5 (report only): for every +/-2 and +/-8 test-state file of the current training tasks (T13 v4 Table and
Push on the T14 round 6 C references, T14 front-back), the largest depth by which any robot link (hands included) is inside the
object at frame 0, measured on the simulator collision shapes. Same geometry as make_states_t14.py's frame-0 check (MuJoCo
mj_collision proposes the pairs with a 0.1 m margin, each pair's depth = mj_geomDistance, robot shapes of the Isaac dump, object
= the training object: Table dump meshes / Push box URDF / front-back box URDF), extended with the deepest link and state.
Read only; one row per file -> logs_system1/t14/frontback_train/states_frame0/frame0_links.csv/.json (+ per state csv).

    sonic_s1/bin/python -B system1/t14/frontback_train/states_frame0_links.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
for p in ("system1/t14", "system1/t12v2", "src/holosoma_retargeting"):
    sys.path.insert(0, str(ROOT / p))
import system1.t13v4.make_states_t14 as ms  # noqa: E402
from paper_evaluation_utils import yaw_transform  # noqa: E402

WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
OUT = ROOT / "logs_system1/t14/frontback_train/states_frame0"
FILES = [
    ("table", "T13 v4 Table (T14 r6C)", WBT / "system1_t14r6C/table/t14_reference.json", None, "system1/t13v4/states_t14r6C"),
    ("push", "T13 v4 Push (T14 r6C)", WBT / "system1_t14r6C/push/t14_reference.json", None, "system1/t13v4/states_t14r6C"),
    ("table", "T14 front-back", WBT / "system1_t14fb/table/t14_reference.json", WBT / "system1_t14fb/table/frontback_box5kg_training.urdf",
     "system1/t14/frontback_train/states"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def deepest(ev, pair, objq):
    """per robot: (depth m, link) of the deepest robot-object pair; depth 0 / None when nothing is inside."""
    tl = ev.robot.fk(pair[None, :, :3], np.concatenate([pair[None, :, 4:7], pair[None, :, 3:4]], -1), pair[None, :, 7:])
    m, d = ev.m, ev.d
    d.qpos[0:3], d.qpos[3:7] = objq[:3], objq[3:7]
    k = 1
    for r in (0, 1):
        for link in ev.links:
            M = tl[link][0, r]
            d.qpos[7 * k:7 * k + 3] = M[:3, 3]
            d.qpos[7 * k + 3:7 * k + 7] = Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]]
            k += 1
    ev.mujoco.mj_kinematics(m, d)
    ev.mujoco.mj_collision(m, d)
    cand = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d.contact[:d.ncon]}
    out = [(0.0, None), (0.0, None)]
    fromto = np.zeros(6)
    for g1, g2 in cand:
        dist = ev.mujoco.mj_geomDistance(m, d, g1, g2, ev.margin, fromto)
        g = g1 if g1 in ev.geom_robot else g2
        r, link = ev.geom_robot[g], ev.geom_link[g]
        if -dist > out[r][0]:
            out[r] = (-dist, link)
    return out


def main():
    import contact_stats as cs
    import dual_stretch as ds
    import eval_core4d_hulls as eh
    OUT.mkdir(parents=True, exist_ok=True)
    robot = cs.Robot("physx")
    rows, per_state = [], []
    for case, task, manifest, urdf, sdir in FILES:
        meshes = eh.urdf_box_meshes(urdf) if urdf is not None else (eh.dump_object_meshes("table", 1.0) if case == "table" else eh.urdf_box_meshes(ds.TABLE["push"][0]))
        ev = eh.ObjectRobotDistance(meshes, robot, depth="exact")
        rt = ROOT / "src/holosoma" / json.loads(manifest.read_text())["files"]["runtime_reference" if case == "table" else "motion"]["path"]
        root, dof, jn, obj = ms.frame0(case, rt)
        perm = [jn.index(n) for n in robot.dofs]
        for lv in (2, 8):
            sf = ROOT / sdir / f"{case}_pm{lv}_states.json"
            states = json.loads(sf.read_text())["states"]
            best, n_in, n_1cm, hand_best = (0.0, None, None, None), 0, 0, 0.0
            for s in states:
                rr = yaw_transform(root, s["robot_xy_m"], s["robot_yaw_rad"])
                q = np.zeros((2, 36))
                q[:, :3], q[:, 3:7], q[:, 7:] = rr[:, :3], rr[:, [6, 3, 4, 5]], dof[:, perm]
                dp = deepest(ev, q, obj)
                for r in (0, 1):
                    per_state.append(dict(task=task, file=str(sf.relative_to(ROOT)), state_id=s.get("state_id", s.get("id")), robot=f"r{r}",
                                          depth_cm=round(100 * dp[r][0], 3), link=dp[r][1]))
                    if dp[r][1] is not None and dp[r][1] in cs.HANDS:
                        hand_best = max(hand_best, dp[r][0])
                m = max(dp, key=lambda x: x[0])
                n_in += m[0] > 0
                n_1cm += m[0] > 0.01
                if m[0] > best[0]:
                    r = 0 if dp[0][0] >= dp[1][0] else 1
                    best = (m[0], f"r{r}", m[1], s.get("state_id", s.get("id")))
            rows.append(dict(task=task, file=str(sf.relative_to(ROOT)), sha256=sha(sf), states=len(states), object=("box URDF " + urdf.name) if urdf is not None else
                             ("Table dump meshes" if case == "table" else "Push box URDF"), max_depth_cm=round(100 * best[0], 3), robot=best[1], link=best[2],
                             state=best[3], states_any_link_inside=int(n_in), states_deeper_than_1cm=int(n_1cm),
                             max_depth_deepest_link_is_hand_cm=round(100 * hand_best, 3), runtime=str(rt.relative_to(ROOT)), runtime_sha256=sha(rt)))
            print(json.dumps(rows[-1]), flush=True)
    for name, rr in (("frame0_links.csv", rows), ("frame0_links_per_state.csv", per_state)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    (OUT / "frame0_links.json").write_text(json.dumps(dict(rules=__doc__, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
