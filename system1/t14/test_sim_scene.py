#!/usr/bin/env python3
"""Smoke test of sim_scene.simify_scene on the round-2 Table k = 1 scene of person 2 (robot 1):
  1. kinematics: MuJoCo body frames of the patched scene == the simulator URDF FK (contact_stats.Robot) at every frame of
     the round-2 reference, < 1e-9 m / 1e-9;
  2. joint axes and limits of the 29 dofs: MJCF == simulator URDF;
  3. collision filter: original robot geoms reach the ground but not the object; PhysX robot geoms reach the object but
     not the ground;
  4. geometry: the per-frame robot-object penetration that the retargeter's constraint sees in the patched scene (mj_collision
     candidates with a 0.1 m margin, then mj_geomDistance on object pairs, as _update_jacobians_and_phis_from_q) equals the
     evaluation measure (eval_core4d_hulls on the training URDF boxes; diag npz written by diag/pen_dump_eval.py), and the
     unpatched scene does not;
  5. Minkowski-difference depth == mj_geomDistance on the deepest frames;
  6. mj_geomDistance == exact convex distance for every robot-object candidate pair of every frame (fails on mujoco 3.10.0).

    hsretargeting/bin/python -B system1/t14/test_sim_scene.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
import contact_stats as cs  # noqa: E402
from sim_scene import SIM_URDF, simify_scene  # noqa: E402

R2 = ROOT / "logs_system1/t14/table/new/person2"
EVAL = ROOT / "logs_system1/t14/diag3/eval_k1_exact.npz"  # diag/pen_dump_eval.py ... exact (sonic_s1, MuJoCo 4.0)
EVAL_CONTACT = ROOT / "logs_system1/t14/diag3/eval_k1.npz"  # the round-2 measure (deepest mj_collision contact)


def constraint_view(m, d, q):
    """per frame: max penetration over robot-object pairs as the retargeter's constraint sees them."""
    obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "desk001")
    obj = {g for g in range(m.ngeom) if m.geom_bodyid[g] == obj_body}
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    out = np.zeros(len(q))
    fromto = np.zeros(6)
    hits = dict(orig_obj=0, physx_obj=0, physx_ground=0)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    saved = m.geom_margin.copy()
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        m.geom_margin[:] = 0.1
        mujoco.mj_collision(m, d)
        m.geom_margin[:] = saved
        cand = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d.contact[:d.ncon]}
        for g1, g2 in cand:
            o1, o2 = g1 in obj, g2 in obj
            if o1 or o2:
                r = g2 if o1 else g1
                if names[r].startswith("physx_"):
                    hits["physx_obj"] += 1
                else:
                    hits["orig_obj"] += 1
                dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.1, fromto)
                out[t] = max(out[t], -dist)
            elif ground in (g1, g2) and (names[g1].startswith("physx_") or names[g2].startswith("physx_")):
                hits["physx_ground"] += 1
    return out, hits


def main():
    q = np.load(R2 / "person2_new_qpos.npz")["qpos"]
    tmp = ROOT / "logs_system1/t14/r3/test_sim_scene"  # kept (overwritten by the next run)
    tmp.mkdir(parents=True, exist_ok=True)
    out, rec = simify_scene(R2 / "assets/shared_scene_simhands.xml", tmp / "scene_sim.xml", tmp / "sim_meshes", "desk001")
    print("changed bodies:", sorted(rec["kinematics_changed"]), "skipped:", rec["kinematics_skipped"])
    assert set(rec["kinematics_changed"]) == {"waist_roll_link", "torso_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link"}, rec["kinematics_changed"]
    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    # 1. kinematics
    rob = cs.Robot("physx")
    tl = rob.fk(q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:36])
    err_p = err_r = 0.0
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        for link, T in tl.items():
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            if b < 0:
                continue
            err_p = max(err_p, float(np.abs(d.xpos[b] - T[t, :3, 3]).max()))
            err_r = max(err_r, float(np.abs(d.xmat[b].reshape(3, 3) - T[t, :3, :3]).max()))
    print(f"1. FK vs simulator URDF over {len(q)} frames: pos {err_p:.2e} m, rot {err_r:.2e}")
    assert err_p < 1e-9 and err_r < 1e-9
    # 2. axes and limits
    import yourdfpy
    u = yourdfpy.URDF.load(str(SIM_URDF), load_meshes=False, build_scene_graph=False)
    uj = {j.name: j for j in u.robot.joints}
    for jn in rob.dofs:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert np.allclose(m.jnt_axis[jid], uj[jn].axis, atol=1e-9), jn
        assert np.allclose(m.jnt_range[jid], [uj[jn].limit.lower, uj[jn].limit.upper], atol=1e-6), (jn, m.jnt_range[jid])
    print("2. joint axes and limits identical (29 dofs)")
    # 3 + 4. collision filter and geometry
    pen, hits = constraint_view(m, d, q)
    print("3. candidate pairs:", hits)
    assert hits["orig_obj"] == 0 and hits["physx_ground"] == 0 and hits["physx_obj"] > 0
    ev = np.load(EVAL)["P"][:, 1]
    m0 = mujoco.MjModel.from_xml_path(str(R2 / "assets/shared_scene_simhands.xml"))
    pen0, _ = constraint_view(m0, mujoco.MjData(m0), q)
    print(f"4. max penetration: evaluation {100 * ev.max():.3f} cm at frame {ev.argmax()}; patched scene {100 * pen.max():.3f} cm at frame {pen.argmax()}; "
          f"unpatched (round-2 constraint view) {100 * pen0.max():.3f} cm")
    print(f"   per-frame |patched - evaluation| max {np.abs(pen - ev).max():.2e} m; |unpatched - evaluation| max {np.abs(pen0 - ev).max():.2e} m")
    evc = np.load(EVAL_CONTACT)["P"][:, 1]
    print(f"   round-2 measure (deepest mj_collision contact, MuJoCo 4.0): {100 * evc.max():.3f} cm at frame {evc.argmax()}")
    assert np.abs(pen - ev).max() < 1e-6
    # 5. exactness: Minkowski-difference depth of the deepest hand-tabletop pairs == mj_geomDistance (patched scene)
    from scipy.spatial import ConvexHull
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)

    def verts(g):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            c = np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * m.geom_size[g]
        else:
            mid = m.geom_dataid[g]
            c = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return c @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    fromto = np.zeros(6)
    top = gid("desk001_tabletop")
    worst = 0.0
    for t in np.argsort(-pen)[:20]:
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not nm.startswith("physx_") or "rubber_hand" not in nm:
                continue
            dist = mujoco.mj_geomDistance(m, d, g, top, 0.1, fromto)
            if dist >= 0:
                continue
            h = ConvexHull((verts(g)[:, None] - verts(top)[None]).reshape(-1, 3))
            exact = float(np.min(-h.equations[:, 3]))
            worst = max(worst, abs(exact + dist))
    print(f"5. Minkowski depth vs mj_geomDistance on the 20 deepest frames: max |diff| {worst:.2e} m")
    assert worst < 1e-6
    # 6. every robot-object candidate pair of every frame: mj_geomDistance == exact convex distance (MuJoCo 3.10.0 fails this:
    #    some separated pairs come back as 0 or negative; the retargeting interpreter must be hsretargeting_s1, mujoco 4.0.0)
    import json
    import subprocess
    r = subprocess.run([sys.executable, "-B", str(ROOT / "system1/t14/diag/geomdist_audit.py"), str(out), str(R2 / "person2_new_qpos.npz"), "desk001"],
                       capture_output=True, text=True, check=True)
    au = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"6. mj_geomDistance audit (MuJoCo {au['mujoco']}): {au['pairs_checked']} robot-object pairs, {au['pairs_wrong']} wrong, worst {au['worst_abs_err_m']:.2e} m")
    assert au["pairs_wrong"] == 0, au["wrong_examples"]
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
