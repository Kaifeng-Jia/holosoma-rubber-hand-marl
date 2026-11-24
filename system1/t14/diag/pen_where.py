"""T14 round 3: link, frame and depth of the deepest robot-object penetration of a two-robot reference (exact measure)."""
import sys, json
from pathlib import Path
import numpy as np, mujoco
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "system1/t14")); sys.path.insert(0, str(ROOT / "system1/t12v2"))
import eval_core4d_hulls as eh
from scipy.spatial.transform import Rotation
pair_path, obj = sys.argv[1], sys.argv[2]
z = np.load(pair_path); pair, oq = z["robot_qpos"].astype(float), z["object_qpos"].astype(float)
robot = eh.cs.Robot("physx")
meshes = eh.dump_object_meshes(obj.split(":")[1], float(obj.split(":")[2])) if obj.startswith("dump:") else eh.urdf_box_meshes(obj)
ev = eh.ObjectRobotDistance(meshes, robot, depth="exact")
tl = robot.fk(pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:])
best = {0: (0, None, None), 1: (0, None, None)}
ft = np.zeros(6)
for t in range(len(pair)):
    ev.d.qpos[0:3], ev.d.qpos[3:7] = oq[t, :3], oq[t, 3:7]
    k = 1
    for r in (0, 1):
        for link in ev.links:
            M = tl[link][t, r]; ev.d.qpos[7 * k:7 * k + 3] = M[:3, 3]; ev.d.qpos[7 * k + 3:7 * k + 7] = Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]]; k += 1
    mujoco.mj_kinematics(ev.m, ev.d); mujoco.mj_collision(ev.m, ev.d)
    for g1, g2 in {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in ev.d.contact[:ev.d.ncon]}:
        dist = mujoco.mj_geomDistance(ev.m, ev.d, g1, g2, 0.1, ft)
        g = g1 if g1 in ev.geom_robot else g2
        r = ev.geom_robot[g]
        if -dist > best[r][0]:
            best[r] = (-dist, t, ev.geom_link[g])
print(json.dumps({f"r{r}": dict(depth_cm=round(100 * v[0], 3), frame=v[1], time_s=None if v[1] is None else round(v[1] / 30, 2), link=v[2]) for r, v in best.items()}))
