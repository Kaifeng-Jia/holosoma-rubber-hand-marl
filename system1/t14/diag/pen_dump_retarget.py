"""T14 round 3 item 3 diagnosis, retargeter side (hsretargeting): load the retargeter's own scene of one person, set the
solved qpos frame by frame and measure (a) mj_geomDistance hand-geom vs object box geoms exactly as
_update_jacobians_and_phis_from_q does, (b) the world vertices of the hand geoms and object boxes, for comparison with
pen_dump_eval.py."""
import sys, json
from pathlib import Path
import numpy as np
import mujoco
scene, qpath, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
q = np.load(qpath)["qpos"]
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
obj = [g for g in range(m.ngeom) if names[g].startswith("desk001_") and m.geom_contype[g]]
hand = {s: names.index(f"{s}_rubber_hand_link") for s in ("left", "right")}
D = np.full((len(q), 2), np.inf)
fromto = np.zeros(6)
HV = {s: [] for s in hand}
BV = []
for t in range(len(q)):
    d.qpos[:] = q[t]
    mujoco.mj_forward(m, d)
    for hi, s in enumerate(hand):
        g = hand[s]
        for o in obj:
            D[t, hi] = min(D[t, hi], mujoco.mj_geomDistance(m, d, g, o, 0.1, fromto))
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        V = m.mesh_vert[a:a + n]
        HV[s].append(V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    bv = []
    for o in obj:
        hs = m.geom_size[o]
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hs
        bv.append(c @ d.geom_xmat[o].reshape(3, 3).T + d.geom_xpos[o])
    BV.append(np.stack(bv))
np.savez(out, D=D, HV_left=np.stack(HV["left"]), HV_right=np.stack(HV["right"]), BV=np.stack(BV), obj_names=np.array([names[o] for o in obj]))
print(json.dumps(dict(mujoco=mujoco.__version__, min_dist_cm=(100 * D.min(0)).round(3).tolist(), argmin=D.argmin(0).tolist())))
