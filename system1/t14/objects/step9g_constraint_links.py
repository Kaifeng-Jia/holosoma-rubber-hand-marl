#!/usr/bin/env python3
"""T14 step 9g item 0 (Kevin 2026-11-26 01:40): which robot links the solver's "no penetration into the object" constraint
covers. Report only (read only, CPU); nothing in the solver is changed.

The constraint (src/holosoma_retargeting/.../src/interaction_mesh_retargeter.py, _update_jacobians_and_phis_from_q and the
"Non-penetration constraints" block): every SQP iteration, mj_collision with all geom margins set to
collision_detection_threshold (default 0.1 m) gives candidate geom pairs; a pair is kept when one geom belongs to the object
body (<name>_link) and neither geom has contype = conaffinity = 0; for kept pairs within 0.1 m, the linearized distance
must stay >= -penetration_tolerance (hard in Stage 1; in Stage 2 role_solve enables elastic constraints: L1 slack with weight
1e5 per metre). So a robot geom is covered iff MuJoCo's own contact filter lets it touch an object geom:
(contype_r & conaffinity_o) or (contype_o & conaffinity_r), the two bodies are not excluded (<exclude>), and they are not
parent / child (the object body is a child of the world). This script applies exactly that filter to the scene files of
actual step-9f solves (Stage 2: <level>/r<k>/scene_sim.xml; Stage 1 of the partner: the source-box scene in
<level>/r1/assets_source/), and lists for every robot body: its collision geoms, contype / conaffinity, covered or not.
It also checks the filter empirically: mj_collision with margin 0.1 m at the solved poses of the level, counting the
object pairs that actually appear per robot body.

    hsretargeting_s1/bin/python -B system1/t14/objects/step9g_constraint_links.py --level logs_system1/t14/step9f/suitcasea/m1p00 --out <json> [--csv <csv>]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np


def body_name(m, b):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


def geom_name(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"


def object_body(m, name):
    for n in (name, f"{name}_link"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        if b >= 0:
            return b
    raise SystemExit(f"object body {name} not in the scene")


def coverage(scene, obj):
    m = mujoco.MjModel.from_xml_path(str(scene))
    ob = object_body(m, obj)
    og = [g for g in range(m.ngeom) if m.geom_bodyid[g] == ob]
    excl = {(int(m.exclude_signature[i]) >> 16, int(m.exclude_signature[i]) & 0xFFFF) for i in range(m.nexclude)}
    rows = []
    for b in range(1, m.nbody):
        if b == ob:
            continue
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        for g in geoms:
            ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
            enabled = not (ct == 0 and ca == 0)
            hits = [o for o in og if (ct & int(m.geom_conaffinity[o])) or (int(m.geom_contype[o]) & ca)]
            excluded = any((min(b, ob), max(b, ob)) == (min(x, y), max(x, y)) for x, y in excl)
            parent = m.body_parentid[b] == ob or m.body_parentid[ob] == b
            rows.append(dict(body=body_name(m, b), geom=geom_name(m, g), type=int(m.geom_type[g]), contype=ct, conaffinity=ca,
                             group=int(m.geom_group[g]), collision_enabled=enabled,
                             covered=bool(enabled and hits and not excluded and not parent)))
    obj_geoms = [dict(geom=geom_name(m, o), contype=int(m.geom_contype[o]), conaffinity=int(m.geom_conaffinity[o]), type=int(m.geom_type[o]),
                      size=np.round(m.geom_size[o], 4).tolist()) for o in og]
    return m, ob, og, rows, obj_geoms


def empirical(m, ob, og, qpos, thr=0.1, every=1):
    """mj_collision with margin thr at the solved poses (robot qpos[:36], object = the last 7 entries if present): object pairs per robot body."""
    d = mujoco.MjData(m)
    saved = m.geom_margin.copy()
    m.geom_margin[:] = thr
    seen = defaultdict(int)
    nq = m.nq
    try:
        for t in range(0, len(qpos), every):
            d.qpos[:] = qpos[t, :nq]
            mujoco.mj_forward(m, d)
            mujoco.mj_collision(m, d)
            bodies = set()
            for k in range(d.ncon):
                c = d.contact[k]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 in og and g2 not in og:
                    bodies.add(body_name(m, m.geom_bodyid[g2]))
                elif g2 in og and g1 not in og:
                    bodies.add(body_name(m, m.geom_bodyid[g1]))
            for b in bodies:
                seen[b] += 1
    finally:
        m.geom_margin[:] = saved
    return dict(seen)


def solver_depth(level):
    """what the Stage-2 constraint itself saw on a solved level: per robot, the deepest penetration (-distance, m) of the covered
    non-hand geoms into the object geoms (mj_geomDistance on the level's scene_sim.xml at the solved qpos, all frames), its body and
    frame, next to the verdict's additional non-hand proxy penetration (eval.json)."""
    ev = json.loads((level / "eval.json").read_text())
    res = {}
    for r in (0, 1):
        rec = json.loads((level / f"r{r}/run_record.json").read_text())
        scene = Path(rec["scene"])
        m0 = mujoco.MjModel.from_xml_path(str(scene))
        objname = [body_name(m0, int(m0.jnt_bodyid[j])) for j in range(m0.njnt) if m0.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_FREE)
                   and body_name(m0, int(m0.jnt_bodyid[j])) != "pelvis"][0]
        m, ob, og, rows, _ = coverage(scene, objname)
        cov = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, x["geom"]) for x in rows if x["covered"] and "rubber_hand" not in x["geom"]]
        z = np.load(level / f"r{r}/{rec['tag']}_new_qpos.npz")["qpos"]
        d = mujoco.MjData(m)
        ft = np.zeros(6)
        best = (0.0, None, None)
        frames_over_1mm = 0
        for t in range(len(z)):
            d.qpos[:] = z[t, :m.nq]
            mujoco.mj_forward(m, d)
            worst = 0.0
            for g in cov:
                for o in og:
                    dist = mujoco.mj_geomDistance(m, d, g, o, 0.05, ft)
                    if -dist > worst:
                        worst = -dist
                        if worst > best[0]:
                            best = (worst, body_name(m, m.geom_bodyid[g]), t)
            frames_over_1mm += worst > 1e-3 + 1e-9
        key = f"nonhand_penetration_r{r}"
        res[f"r{r}"] = dict(solver_shape_max_penetration_m=round(best[0], 5), body=best[1], frame=best[2], frames_over_1mm=int(frames_over_1mm),
                            verdict_additional_proxy_penetration_m=ev["V0"][key]["value_m"] if key in ev.get("V0", {}) else None)
    return dict(level=str(level), applicable=ev["applicable"], failed=ev["failed"], robots=res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=Path, required=True, help="a step-9f level dir with r0/ r1/ (solved)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--solver-depth", type=Path, nargs="*", default=None,
                    help="instead of the coverage table: solver_depth() of these level dirs (json list to --out)")
    a = ap.parse_args()
    if a.solver_depth is not None:
        rows = []
        for lv in a.solver_depth:
            x = solver_depth(lv)
            rows.append(x)
            print(lv, {k: (v["solver_shape_max_penetration_m"], v["body"], v["verdict_additional_proxy_penetration_m"]) for k, v in x["robots"].items()}, flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(rules=solver_depth.__doc__, rows=rows), indent=1) + "\n")
        return 0
    out = dict(rules=__doc__, level=str(a.level), scenes={})
    csv_rows = []
    for r in (0, 1):
        rec = json.loads((a.level / f"r{r}/run_record.json").read_text())
        obj = rec["object"] if rec.get("family") != "box" else None
        scene = Path(rec["scene"])
        # the object body name of the target scene = <tag-name>_link; find the free body that is not the robot
        m0 = mujoco.MjModel.from_xml_path(str(scene))
        free_bodies = [body_name(m0, int(m0.jnt_bodyid[j])) for j in range(m0.njnt) if m0.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_FREE)]
        objname = [b for b in free_bodies if b != "pelvis"][0]
        m, ob, og, rows, obj_geoms = coverage(scene, objname)
        z = np.load(a.level / f"r{r}/{rec['tag']}_new_qpos.npz")["qpos"]
        emp = empirical(m, ob, og, z) if z.shape[1] == m.nq else {"note": f"qpos width {z.shape[1]} != model nq {m.nq}"}
        out["scenes"][f"stage2_r{r}"] = dict(scene=str(scene), object_body=objname, object_geoms=obj_geoms, geoms=rows, empirical_object_pairs_frames=emp,
                                             frames=int(len(z)))
        for x in rows:
            csv_rows.append(dict(scene=f"stage2_r{r}", **x))
    # Stage 1 of the partner (default RetargeterConfig, source-box scene)
    src_scenes = sorted((a.level / "r1/assets_source").glob("*.xml"))
    for sc in src_scenes:
        m0 = mujoco.MjModel.from_xml_path(str(sc))
        free_bodies = [body_name(m0, int(m0.jnt_bodyid[j])) for j in range(m0.njnt) if m0.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_FREE)]
        objname = [b for b in free_bodies if b != "pelvis"][0]
        m, ob, og, rows, obj_geoms = coverage(sc, objname)
        out["scenes"]["stage1_r1"] = dict(scene=str(sc), object_body=objname, object_geoms=obj_geoms, geoms=rows)
        for x in rows:
            csv_rows.append(dict(scene="stage1_r1", **x))
    # per-body summary
    summ = {}
    for k, v in out["scenes"].items():
        by = defaultdict(lambda: dict(geoms=0, covered=0, enabled=0))
        for x in v["geoms"]:
            by[x["body"]]["geoms"] += 1
            by[x["body"]]["enabled"] += x["collision_enabled"]
            by[x["body"]]["covered"] += x["covered"]
        summ[k] = dict(by)
    out["per_body"] = summ
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    for k, by in summ.items():
        print("==", k)
        for b, x in by.items():
            print(f"  {b:28s} geoms {x['geoms']:2d} enabled {x['enabled']:2d} covered {x['covered']:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
