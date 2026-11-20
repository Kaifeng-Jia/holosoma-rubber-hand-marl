"""T10 sim2sim: MuJoCo scene and evaluation-time environment for the paired CORE4D tasks (Table, Bin), built from what Isaac
simulates (system1/sim2sim/isaac_dump.py export) and the holosoma code paths it mirrors.

Scene (built with mujoco.MjSpec):
* two rubber-hand G1 from the training URDF (kinematic tree, joint axes); per Isaac rigid body the PhysX mass / COM / full
  inertia; joints with the PhysX limits and armature, no damping / frictionloss (Isaac: drive stiffness / damping 0, friction 0);
  one motor per joint (torques computed by the holosoma PD below, clipped at the effort limits);
  collision geometry = exactly the Isaac collision shapes: capsules (URDF cylinders converted by the Isaac importer, its radius /
  height / axis), spheres, and the convex hulls PhysX cooked for the mesh links;
* the object from the same export: Table = 5 boxes, Bin = the 16 convex pieces of PhysX's convex decomposition; PhysX mass, COM,
  inertia; free joint;
* ground plane z = 0; friction combined as PhysX does (ground material multiply 1.0 x other; robot 1.0; object 0.5;
  robot-object average 0.75): robot geoms friction 1.0, object geoms friction 0.75 with priority 1 (robot-object 0.75),
  object-ground as explicit pairs with friction 0.5 (no dynamic object-ground contact), condim 3, restitution 0;
* timestep 0.005 s (200 Hz), gravity 9.81; everything else MuJoCo defaults (as the official SONIC sim2sim scene).
State conventions follow Isaac Lab / holosoma: root position / orientation of the link frame, root linear velocity of the
centre of mass, angular velocity in world; quaternions (x, y, z, w) unless named wxyz; joint vectors in holosoma DOF order.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma"))
ROBOT_URDF = ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
TASK = dict(
    table=dict(runtime="holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz",
               object_urdf="holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf"),
    bucket=dict(runtime="holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/core4d_pair_runtime_fps50.npz",
                object_urdf="holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1/bucket003_training.urdf"),
)
GRAVITY = 9.81
SIM_DT = 0.005
DECIMATION = 4
FRICTION = dict(robot=1.0, ground=1.0, object=0.5)  # PhysX materials (Isaac export); combine: ground multiply, others average


def xyzw_to_wxyz(q):
    return np.asarray(q)[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q):
    return np.asarray(q)[..., [1, 2, 3, 0]]


def quat_to_mat_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ----------------------------------------------------------------------------------------------------------------------------
class SceneSpec:
    """Isaac export of one task (scene.json + collision_convexes.npz)."""

    def __init__(self, dump_dir):
        dump_dir = Path(dump_dir)
        self.dir = dump_dir
        self.scene = json.loads((dump_dir / "scene.json").read_text())
        self.conv = np.load(dump_dir / "collision_convexes.npz")
        self.control = self.scene["control"]
        self.dof_names = list(self.control["dof_names"])
        self.body_list = list(self.control["body_list"])


def _body_of(path):
    return path.rstrip("/").split("/")[-1]


def _robot_spec(ss: SceneSpec, robot_key: str):
    """One robot as an MjSpec: URDF kinematics, Isaac inertia / joints / collision geometry."""
    spec = mujoco.MjSpec.from_file(str(ROBOT_URDF))
    spec.compiler.fusestatic = False
    for g in list(spec.geoms):
        spec.delete(g)
    for m in list(spec.meshes):
        spec.delete(m)
    bod = ss.scene["bodies"][robot_key]
    names = bod["names"]
    isaac_bodies = set(names)
    for b in spec.bodies:
        if b.name == "world":
            continue
        b.explicitinertial = True
        if b.name in isaac_bodies:
            i = names.index(b.name)
            com = bod["coms_xyzw"][i]
            I = np.array(bod["inertias"][i]).reshape(3, 3)
            b.mass = float(bod["masses"][i])
            b.ipos = com[:3]
            b.iquat = [1, 0, 0, 0]
            b.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        else:  # fixed link merged into its parent by the Isaac importer: its mass is part of the parent's PhysX mass
            b.mass = 0.0
            b.ipos = [0, 0, 0]
            b.inertia = [0, 0, 0]
    jt = ss.scene["joints"][robot_key]
    for j in spec.joints:
        if j.name in jt["names"]:
            i = jt["names"].index(j.name)
            j.range = jt["limits"][i]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.armature = float(jt["armature"][i])
            j.damping = 0.0
            j.frictionloss = float(jt["friction"][i])
    eff = dict(zip(ss.control["dof_names"], ss.control["torque_limits"][:len(ss.control["dof_names"])]))
    for n in ss.control["dof_names"]:  # one torque motor per joint; torques come from the holosoma PD, clipped at the effort limit
        act = spec.add_actuator(name=n, target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gainprm[0] = 1.0
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-eff[n], eff[n]]
    root = "/World/envs/env_0/Robot" if robot_key == "robot" else "/World/envs/env_0/Robot_1"
    bodies = {b.name: b for b in spec.bodies}
    for k, sh in enumerate(ss.scene["shapes"]):
        if sh["root"] != root or not sh.get("collision_enabled", True):
            continue
        body = bodies[_body_of(sh["body"])]
        T = np.array(sh["prim_in_body"])
        R = T[:3, :3]
        sc = np.linalg.norm(R, axis=0)
        Rn = R / sc
        t = sh["type"]
        g = None
        if t == "Capsule":
            ax = {"X": 0, "Y": 1, "Z": 2}[sh["axis"]]
            d = Rn[:, ax] * (sh["height"] * sc[ax] / 2)
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=np.concatenate([T[:3, 3] - d, T[:3, 3] + d]),
                              size=[sh["radius"] * max(sc[(ax + 1) % 3], sc[(ax + 2) % 3]), 0, 0])
        elif t == "Sphere":
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=T[:3, 3], size=[sh["radius"] * sc.max(), 0, 0])
        elif t == "Xform":
            key = sh["npz_key"]
            for mi, m in enumerate(sh["meshes"]):
                for hi in range(m["n_hulls"]):
                    v = ss.conv[f"{key}_mesh{mi}_hull{hi}"]
                    vb = (T[:3, :3] @ v.T).T + T[:3, 3]
                    mname = f"{robot_key}_h{k}_{mi}_{hi}"
                    spec.add_mesh(name=mname, uservert=vb.reshape(-1).tolist())
                    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname)
                    _robot_geom(g)
            continue
        else:
            raise ValueError(f"unsupported robot collision prim {t}: {sh['path']}")
        _robot_geom(g)
    return spec


def _robot_geom(g):
    g.contype, g.conaffinity = 3, 3
    g.friction = [FRICTION["robot"], 0, 0]
    g.condim = 3
    g.rgba = [0.7, 0.7, 0.7, 1]


def build_model(ss: SceneSpec, task: str):
    """World spec with two robots (prefixes r0_, r1_), the object and the ground. Returns (MjModel, info)."""
    world = mujoco.MjSpec()
    world.option.timestep = SIM_DT
    world.option.gravity = [0, 0, -GRAVITY]
    world.compiler.degree = False
    world.visual.global_.offwidth, world.visual.global_.offheight = 1920, 1080
    ground = world.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1])
    ground.contype, ground.conaffinity = 1, 1
    ground.friction = [FRICTION["ground"], 0, 0]
    for r, key in enumerate(("robot", "robot_1")):
        rs = _robot_spec(ss, key)
        pelvis = rs.worldbody.first_body()
        pelvis.add_freejoint(name="root")
        frame = world.worldbody.add_frame()
        frame.attach_body(pelvis, f"r{r}_", "")
    ob = world.worldbody.add_body(name="object")
    ob.add_freejoint(name="object_root")
    od = ss.scene["object"]
    com = od["coms_xyzw"][0]
    I = np.array(od["inertias"][0]).reshape(3, 3)
    ob.explicitinertial = True
    ob.mass = float(od["mass"][0][0] if isinstance(od["mass"][0], list) else od["mass"][0])
    ob.ipos = com[:3]
    ob.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    obj_geoms = []
    for k, sh in enumerate(ss.scene["shapes"]):
        if not sh["root"].endswith("/Object") or not sh.get("collision_enabled", True):
            continue
        T = np.array(sh["prim_in_body"])
        R = T[:3, :3]
        sc = np.linalg.norm(R, axis=0)
        Rn = R / sc
        if sh["type"] == "Cube":
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, Rn.reshape(-1))
            g = ob.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=T[:3, 3], quat=q, size=(sh["size"] * sc / 2).tolist())
            obj_geoms.append(g)
        elif sh["type"] == "Xform":
            for mi, m in enumerate(sh["meshes"]):
                for hi in range(m["n_hulls"]):
                    v = ss.conv[f"{sh['npz_key']}_mesh{mi}_hull{hi}"]
                    vb = (T[:3, :3] @ v.T).T + T[:3, 3]
                    mname = f"obj_h{k}_{mi}_{hi}"
                    world.add_mesh(name=mname, uservert=vb.reshape(-1).tolist())
                    obj_geoms.append(ob.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname))
        else:
            raise ValueError(f"unsupported object collision prim {sh['type']}")
    ob_mu = (FRICTION["robot"] + FRICTION["object"]) / 2  # PhysX average combine robot-object
    for i, g in enumerate(obj_geoms):
        g.name = f"obj_geom{i}"
        g.contype, g.conaffinity = 2, 2
        g.friction = [ob_mu, 0, 0]
        g.priority = 1
        g.condim = 3
        g.rgba = [0.7, 0.8, 0.9, 1]
        p = world.add_pair(geomname1="ground", geomname2=g.name, condim=3)
        p.friction = [FRICTION["ground"] * FRICTION["object"], FRICTION["ground"] * FRICTION["object"], 0, 0, 0]
    model = world.compile()
    info = dict(n_obj_geoms=len(obj_geoms))
    return model, world, info


# ----------------------------------------------------------------------------------------------------------------------------
class PairEnv:
    """Evaluation-time paired environment on MuJoCo (one environment, two agents), mirroring the holosoma Core4D managers."""

    def __init__(self, dump_dir, task, sonic_pd=False):
        from holosoma.envs.marl.paired_motion_reference import PairedMotionReference
        self.ss = SceneSpec(dump_dir)
        self.task = task
        self.model, self.spec, self.info = build_model(self.ss, task)
        self.data = mujoco.MjData(self.model)
        m = self.model
        c = self.ss.control
        self.dof_names = list(c["dof_names"])
        self.nd = len(self.dof_names)
        self.body_list = list(c["body_list"])
        self.qadr = np.array([[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{r}_{n}")] for n in self.dof_names] for r in range(2)])
        self.vadr = np.array([[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{r}_{n}")] for n in self.dof_names] for r in range(2)])
        self.act = np.array([[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"r{r}_{n}") for n in self.dof_names] for r in range(2)])
        self.root_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{r}_root")] for r in range(2)]
        self.root_v = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{r}_root")] for r in range(2)]
        self.pelvis = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"r{r}_pelvis") for r in range(2)]
        self.body_ids = np.array([[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"r{r}_{n}") for n in self.body_list] for r in range(2)])
        assert (self.body_ids >= 0).all(), "Isaac body missing in MuJoCo model"
        self.obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obj_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_root")]
        self.obj_v = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_root")]
        self.kp = np.array(c["p_gains"], dtype=np.float64)
        self.kd = np.array(c["d_gains"], dtype=np.float64)
        self.tau_lim = np.array(c["torque_limits"], dtype=np.float64)
        self.default = np.array(c["default_dof_pos"], dtype=np.float64)
        self.scales = np.array(c["action_scales"], dtype=np.float64)
        self.clip_a = float(c["action_clip_value"]) if c["clip_actions"] else None
        from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES  # the paired command's alias map
        aliases = [FAKE_BODY_NAME_ALIASES.get(n, n) for n in self.body_list]
        self.ref = PairedMotionReference(str(ROOT / "src/holosoma" / TASK[task]["runtime"]), aliases, self.dof_names, device="cpu")
        self.n = int(self.ref.num_frames)
        self.ref_body = self.body_list.index("torso_link")
        self.t = 0
        self.substep_torques = None

    # -- reference
    def sample(self, t):
        s = self.ref.sample(torch.tensor([t]))
        return {k: v[0].numpy().astype(np.float64) for k, v in s.items()}

    # -- state io (Isaac conventions)
    def set_state(self, root13, dof_pos, dof_vel, obj_pos, obj_quat_xyzw, obj_lin_vel=None, obj_ang_vel=None):
        """root13 [2,13] = pos, quat xyzw, COM lin vel world, ang vel world (Isaac Lab root_state_w)."""
        d, m = self.data, self.model
        d.qvel[:] = 0
        for r in range(2):
            d.qpos[self.root_q[r]:self.root_q[r] + 3] = root13[r, :3]
            d.qpos[self.root_q[r] + 3:self.root_q[r] + 7] = xyzw_to_wxyz(root13[r, 3:7])
            d.qpos[self.qadr[r]] = dof_pos[r]
            d.qvel[self.vadr[r]] = dof_vel[r]
        d.qpos[self.obj_q:self.obj_q + 3] = obj_pos
        d.qpos[self.obj_q + 3:self.obj_q + 7] = xyzw_to_wxyz(obj_quat_xyzw)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        for r in range(2):
            b = self.pelvis[r]
            R = d.xmat[b].reshape(3, 3)
            w = root13[r, 10:13]
            off = d.xipos[b] - d.xpos[b]
            d.qvel[self.root_v[r]:self.root_v[r] + 3] = root13[r, 7:10] - np.cross(w, off)
            d.qvel[self.root_v[r] + 3:self.root_v[r] + 6] = R.T @ w
        R = d.xmat[self.obj].reshape(3, 3)
        w = np.zeros(3) if obj_ang_vel is None else np.asarray(obj_ang_vel)
        v = np.zeros(3) if obj_lin_vel is None else np.asarray(obj_lin_vel)
        off = d.xipos[self.obj] - d.xpos[self.obj]
        d.qvel[self.obj_v:self.obj_v + 3] = v - np.cross(w, off)
        d.qvel[self.obj_v + 3:self.obj_v + 6] = R.T @ w
        mujoco.mj_forward(m, d)

    def state(self):
        d = self.data
        root = np.zeros((2, 13))
        for r in range(2):
            b = self.pelvis[r]
            R = d.xmat[b].reshape(3, 3)
            w = R @ d.qvel[self.root_v[r] + 3:self.root_v[r] + 6]
            v0 = d.qvel[self.root_v[r]:self.root_v[r] + 3]
            root[r, :3] = d.xpos[b]
            root[r, 3:7] = wxyz_to_xyzw(d.xquat[b])
            root[r, 7:10] = v0 + np.cross(w, d.xipos[b] - d.xpos[b])
            root[r, 10:13] = w
        out = dict(root=root, dof_pos=np.stack([d.qpos[self.qadr[r]] for r in range(2)]), dof_vel=np.stack([d.qvel[self.vadr[r]] for r in range(2)]),
                   body_pos=d.xpos[self.body_ids].copy(), body_quat_xyzw=wxyz_to_xyzw(d.xquat[self.body_ids]).copy(),
                   obj_pos=d.xpos[self.obj].copy(), obj_quat_xyzw=wxyz_to_xyzw(d.xquat[self.obj]).copy())
        return out

    # -- reset exactly as Core4D reset + evaluator yaw transform
    def reset(self, init):
        sys.path.insert(0, str(ROOT / "scripts"))
        from paper_evaluation_utils import yaw_transform  # the evaluator's own transform
        s = self.sample(0)
        root = np.zeros((2, 13))
        root[:, :3] = s["agent_body_pos_w"][:, 0]
        root[:, 3:7] = s["agent_body_quat_w"][:, 0]
        root[:, 7:10] = s["agent_body_lin_vel_w"][:, 0]
        root[:, 10:13] = s["agent_body_ang_vel_w"][:, 0]
        root = yaw_transform(root, init["robot_xy_m"], init["robot_yaw_rad"])
        self.set_state(root, s["agent_joint_pos"], s["agent_joint_vel"], s["object_pos_w"], s["object_quat_w"], obj_lin_vel=s["object_lin_vel_w"])
        self.t = 0
        return root

    # -- PD (holosoma marl joint_control, control_type P) and physics
    def physics_step(self, target):
        """target [2, 29] PD joint targets; 4 substeps of 200 Hz with torques recomputed each substep and clipped."""
        d, m = self.data, self.model
        taus = np.zeros((DECIMATION, 2, self.nd))
        for k in range(DECIMATION):
            for r in range(2):
                q = d.qpos[self.qadr[r]]
                qd = d.qvel[self.vadr[r]]
                tau = self.kp * (target[r] - q) - self.kd * qd
                tau = np.clip(tau, -self.tau_lim, self.tau_lim)
                d.ctrl[self.act[r]] = tau
                taus[k, r] = tau
            mujoco.mj_step(m, d)
        self.substep_torques = taus

    def d_target(self, actions):
        a = np.asarray(actions, dtype=np.float64)
        if self.clip_a is not None:
            a = np.clip(a, -self.clip_a, self.clip_a)
        return a * self.scales + self.default

    # -- observation (plan5 actor obs + real teammate obs), Isaac order per agent
    def observation(self, last_action, st=None, t=None):
        from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms, yaw_quat
        st = self.state() if st is None else st
        t = self.t if t is None else t
        s = self.sample(t)
        cmd = np.concatenate([s["agent_joint_pos"], s["agent_joint_vel"]], axis=-1)  # [2, 58]
        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)  # noqa: E731
        ap = T(st["body_pos"][:, self.ref_body])
        aq = T(st["body_quat_xyzw"][:, self.ref_body])
        rp = T(s["agent_body_pos_w"][:, self.ref_body])
        rq = T(s["agent_body_quat_w"][:, self.ref_body])
        _, rel = subtract_frame_transforms(ap, aq, rp, rq)
        ori = quaternion_to_matrix(rel, w_last=True)[..., :2].reshape(2, 6).numpy()
        root = st["root"]
        angb = quat_rotate_inverse(T(root[:, 3:7]), T(root[:, 10:13]), w_last=True).numpy()
        # holosoma's observation manager concatenates the group's terms in alphabetical order (managers/observation/manager.py):
        # actions, base_ang_vel, dof_pos, dof_vel, motion_command, motion_ref_ori_b
        actor = np.concatenate([np.asarray(last_action), angb, st["dof_pos"] - self.default, st["dof_vel"], cmd, ori], axis=-1)
        ids = [1, 0]
        heading = yaw_quat(T(root[:, 3:7]), w_last=True)
        rp_w = T(root[ids, :3] - root[:, :3])
        rv_w = T(root[ids, 7:10] - root[:, 7:10])
        mate = np.concatenate([quat_rotate_inverse(heading, rp_w, w_last=True).numpy()[:, :2],
                               quat_rotate_inverse(heading, rv_w, w_last=True).numpy()[:, :2]], axis=-1)
        mate = np.clip(mate, -1.0, 1.0)
        return actor, mate

    # -- termination (JointBadTrackingZOnly: robot orientation by torso gravity z, object position / orientation)
    def termination(self, st=None, t=None):
        from holosoma.utils.rotations import quat_rotate_inverse
        st = self.state() if st is None else st
        t = self.t if t is None else t
        s = self.sample(t)
        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)  # noqa: E731
        g = T([[0, 0, -1.0], [0, 0, -1.0]])
        rg = quat_rotate_inverse(T(s["agent_body_quat_w"][:, self.ref_body]), g, w_last=True).numpy()[:, 2]
        agz = quat_rotate_inverse(T(st["body_quat_xyzw"][:, self.ref_body]), g, w_last=True).numpy()[:, 2]
        bad_robot = bool(np.any(np.abs(rg - agz) > 0.8))
        pos_err = float(np.linalg.norm(s["object_pos_w"] - st["obj_pos"]))
        from holosoma.utils.rotations import quat_error_magnitude
        ori_err = float(quat_error_magnitude(T(s["object_quat_w"])[None], T(st["obj_quat_xyzw"])[None], w_last=True)[0])
        return dict(bad_robot=bad_robot, bad_object_pos=pos_err > 0.25, bad_object_ori=ori_err > 0.8, object_pos_err=pos_err, object_ori_err=ori_err,
                    gravity_z_err=np.abs(rg - agz).tolist())
