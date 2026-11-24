"""T14 round 3 item 3 diagnosis, eval side (sonic_s1): per frame and per hand the convex penetration of the Table k=1 new
reference as hull_penetration measures it, plus the world vertices of the hand shapes and object boxes used, for the
side-by-side with the retargeter's own scene (pen_dump_retarget.py)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "system1/t14"))
import eval_core4d_hulls as eh
pair_path, urdf, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
depth = sys.argv[4] if len(sys.argv) > 4 else "contact"
z = np.load(pair_path)
pair, objq = z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64)
robot = eh.cs.Robot("physx")
meshes = eh.urdf_box_meshes(urdf)
ev = eh.ObjectRobotDistance(meshes, robot, depth=depth)
H, P = ev.frames(pair, objq)
tl = robot.fk(pair[:, :, :3], np.concatenate([pair[:, :, 4:7], pair[:, :, 3:4]], -1), pair[:, :, 7:])
rob, _, _ = eh.cs.dump_shapes("table", "/World/envs/env_0/Robot/")
hands = {b: m for b, m in rob if "rubber_hand" in b}
np.savez(out, H=H, P=P, **{f"T_{r}_{b}": tl[b][:, r] for b in hands for r in (0, 1)}, **{f"V_{b}": hands[b].vertices for b in hands},
         box_vertices=np.stack([m.vertices for m in meshes]))
i = int(P[:, 1].argmax())
print(json.dumps(dict(r1_max_cm=100 * P[:, 1].max(), frame=i, H_frame=(100 * H[i]).round(3).tolist(), r0_max_cm=100 * P[:, 0].max(), frame0=int(P[:, 0].argmax()))))
