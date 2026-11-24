"""T14 round 4: render the person's hand (SMPL-X mesh with the captured object) for the human / robot close-up panels.

World: Z-up, metres, human scale (CORE4D: the raw Y-up capture rotated by CORE4D_TO_OMNI_ROTATION; OMOMO: the OMOMO_new world
of the palm fit, flat-hand SMPL-X). Object: the captured mesh at its captured pose (CORE4D object_models <name>_m.obj;
OMOMO, round 6: the OMOMO capture table the hands touch, omomo_table.py; before round 6 the InterMimic largetable.obj). Floor = checker plane at the lowest vertex of the person at that frame. Colours:
person (0.87, 0.72, 0.62), object (0.76, 0.80, 0.86) as the robot panels' light-blue object, other person not drawn.
Camera: the robot panel's azimuth and elevation, look-at = the person's two wrist joints' midpoint, distance = the robot
panel's distance / (human-to-robot scale), so the hand appears at the same size as the robot's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
SKIN, OBJ = (0.87, 0.72, 0.62, 1.0), (0.76, 0.80, 0.86, 1.0)


RED, BLUE = (0.9, 0.1, 0.1, 1.0), (0.1, 0.3, 0.95, 1.0)


def add_arrows(scene, arrows, width=0.006):
    """append arrow geoms (p0, p1, rgba) to an MjvScene after update_scene."""
    import mujoco
    for p0, p1, rgba in arrows:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float), np.asarray(p1, float))
        scene.ngeom += 1


class HumanScene:
    def __init__(self, task, tmp):
        from hand_landmarks import SMPLX_NEUTRAL
        self.task, self.tmp = task, Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.faces = np.load(SMPLX_NEUTRAL)["f"]
        if task in ("table", "bin", "chair"):
            import table_shared as ts
            from holosoma_retargeting.data_utils.core4d_adapter import CORE4D_TO_OMNI_ROTATION as C
            seq = ts.CORE4D / "CORE4D_Real/human_object_motions" / ts.TASKS[task]["raw"]
            To = np.load(seq / "smooth_objposes.npy")
            self.V, self.W, self.J = [], [], []
            for p in (1, 2):
                P = np.load(seq / f"person{p}_poses.npz", allow_pickle=True)["arr_0"].item()
                self.V.append(np.einsum("ij,tvj->tvi", C, P["vertices"].astype(np.float64)))
                self.J.append(np.einsum("ij,tvj->tvi", C, P["joints"].astype(np.float64)))
                self.W.append(self.J[-1][:, [20, 21]])
            self.obj_R = np.einsum("ij,tjk->tik", C, To[:, :3, :3])
            self.obj_x = np.einsum("ij,tj->ti", C, To[:, :3, 3])
            self.mesh = next((ts.CORE4D / "CORE4D_Real/object_models").rglob(f"{ts.TASKS[task]['obj_key'].split('_', 1)[1]}_m.obj"))
            self.fps = 15
        else:
            import dual_stretch as ds
            import single_shared as ss
            from scipy.spatial.transform import Rotation
            from holosoma_retargeting.src.utils import load_intermimic_data
            seq = ds.SEQ[task]
            ss.omomo_palms(seq)
            self.V = [ss.omomo_palms.last["vertices_omomo_new"]]
            self.J = [ss.omomo_palms.last["joints_omomo_new"]]
            self.W = [self.J[0][:, [20, 21]]]
            _, obj = load_intermimic_data(str(ss.OMOMO_NEW / f"{seq}.pt"))
            self.obj_R = Rotation.from_quat(obj[:, [1, 2, 3, 0]]).as_matrix()
            self.obj_x = obj[:, 4:7]
            import omomo_table as ot  # the table the person touches (OMOMO capture) in the InterMimic object frame
            self.mesh = self.tmp / f"{task}_capture_table.obj"
            ot.relation(seq)["mesh"].export(self.mesh)
            self.fps = 30
        self.cache = {}

    def frame(self, t30):
        return min(int(round(t30 * self.fps / 30)), len(self.obj_x) - 1)

    def wrist_mid(self, person, t30):
        return self.W[person][min(self.frame(t30), len(self.W[person]) - 1)].mean(0)

    def axes(self, person, t30):
        """palm points [2, 3] and {palm_normal, finger}: [2, 3] (hand_landmarks.human_directions) of the person at frame t30."""
        from hand_landmarks import human_directions
        from table_shared import palm_vertex_ids
        t = min(self.frame(t30), len(self.V[person]) - 1)
        V, J = self.V[person][t], self.J[person][t]
        pv = palm_vertex_ids()
        palms = np.stack([V[pv[s]].mean(0) for s in ("left", "right")])
        d = {k: np.zeros((2, 3)) for k in ("palm_normal", "finger")}
        for hi, side in enumerate(("left", "right")):
            dd = human_directions(J[None], V[None], side)
            for k in d:
                d[k][hi] = dd[k][0]
        return palms, d

    def render(self, person, t30, azimuth, elevation, distance, w, h, arrows=None):
        import mujoco
        from PIL import Image
        from scipy.spatial.transform import Rotation
        t = min(self.frame(t30), len(self.V[person]) - 1)
        key = (person, t, w, h)
        if key not in self.cache:
            V = self.V[person][t]
            hp = self.tmp / f"{self.task}_p{person}_f{t}.obj"
            with open(hp, "w") as f:
                f.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
                f.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in self.faces))
            q = Rotation.from_matrix(self.obj_R[t]).as_quat()[[3, 0, 1, 2]]
            x = self.obj_x[t]
            z0 = float(V[:, 2].min())
            xml = f"""<mujoco><compiler angle="degree"/><visual><headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55"/><quality shadowsize="4096"/></visual>
<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="64" height="64"/><texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.87 0.89" rgb2="0.70 0.72 0.75" width="512" height="512"/>
<material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
<mesh name="obj" file="{self.mesh}" inertia="shell"/><mesh name="human" file="{hp}" inertia="shell"/></asset>
<worldbody><light pos="0 0 4" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
<geom type="plane" size="20 20 0.1" pos="0 0 {z0:.5f}" material="grid"/>
<body pos="{x[0]:.6f} {x[1]:.6f} {x[2]:.6f}" quat="{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"><geom type="mesh" mesh="obj" rgba="{' '.join(map(str, OBJ))}" contype="0" conaffinity="0"/></body>
<geom type="mesh" mesh="human" rgba="{' '.join(map(str, SKIN))}" contype="0" conaffinity="0"/></worldbody></mujoco>"""
            m = mujoco.MjModel.from_xml_string(xml)
            self.cache[key] = (m, mujoco.MjData(m), mujoco.Renderer(m, h, w))
        m, d, r = self.cache[key]
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.wrist_mid(person, t30)
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r.update_scene(d, camera=cam)
        if arrows:
            add_arrows(r.scene, arrows)
        return Image.fromarray(r.render())

    def close(self):
        for _, _, r in self.cache.values():
            r.close()
        self.cache.clear()
