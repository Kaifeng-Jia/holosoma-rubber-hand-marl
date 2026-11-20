"""T10 sim2sim: export what Isaac / PhysX actually simulates, from inside an evaluation run (read only).

Used by scripts/evaluate_paper_supplement.py only when ``--sim2sim-dump DIR`` is given (off: the evaluator is unchanged).

* :func:`dump_scene` writes DIR/scene.json and DIR/collision_convexes.npz: every collision prim under env 0 (both robots, the
  object, the ground plane) with its USD type, dimensions, physics material (friction, restitution, combine modes), pose relative
  to its rigid body, and - for meshes - the convex hulls PhysX cooked for it (convex hull / convex decomposition, via the PhysX
  cooking interface ``request_convex_collision_representation``); per-body mass / COM / inertia and per-joint limits, armature,
  friction, max velocity / force, drive stiffness / damping as read back from the PhysX tensor views; solver settings; the
  holosoma control constants (PD gains, action scales, torque limits, default pose, decimation).
* :class:`EpisodeRecorder` records per control step the policy observations, raw actions, PD targets, per-substep torques and
  joint states, all agent body poses, and (scheme C) SONIC's inputs / outputs, by wrapping functions without changing results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _matrix(gf_matrix):
    return np.array([[gf_matrix[i][j] for j in range(4)] for i in range(4)], dtype=np.float64).T  # USD row-vector -> column form


def _material(prim, stage):
    from pxr import PhysxSchema, UsdPhysics, UsdShade
    out = None
    api = UsdShade.MaterialBindingAPI(prim)
    mat, _ = api.ComputeBoundMaterial("physics")
    if mat:
        mp = mat.GetPrim()
        m = UsdPhysics.MaterialAPI(mp)
        out = dict(path=str(mp.GetPath()), static_friction=m.GetStaticFrictionAttr().Get(), dynamic_friction=m.GetDynamicFrictionAttr().Get(),
                   restitution=m.GetRestitutionAttr().Get())
        if mp.HasAPI(PhysxSchema.PhysxMaterialAPI):
            px = PhysxSchema.PhysxMaterialAPI(mp)
            out.update(friction_combine_mode=px.GetFrictionCombineModeAttr().Get(), restitution_combine_mode=px.GetRestitutionCombineModeAttr().Get())
    return out


def _rigid_ancestor(prim):
    from pxr import UsdPhysics
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


def _cooked_convexes(stage, prim):
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    from pxr import PhysicsSchemaTools
    stage_id = omni.usd.get_context().get_stage_id()
    prim_id = PhysicsSchemaTools.sdfPathToInt(prim.GetPath())
    got = []

    def cb(result, convexes):
        got.append((str(result), [np.array([[v[0], v[1], v[2]] for v in c.vertices], dtype=np.float64) for c in convexes]))

    get_physx_cooking_interface().request_convex_collision_representation(stage_id, prim_id, False, cb)
    return got[0] if got else ("no_callback", [])


def _physx_collision_attrs(prim):
    """Every authored physx collision attribute (convex hull / decomposition / collision offsets) on a prim."""
    out = {}
    for a in prim.GetAttributes():
        n = a.GetName()
        if n.startswith(("physxConvexHullCollision:", "physxConvexDecompositionCollision:", "physxCollision:", "physics:approximation")):
            v = a.Get()
            out[n] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return out, [str(x) for x in prim.GetAppliedSchemas()]


def _cook_copy(coll_prim, mesh_prim, mesh_in_coll):
    """PhysX cooks only real (non-instance-proxy) collision meshes: copy the mesh (points, faces, its transform relative to the
    collision prim) and every physx collision attribute of the collision prim into a separate in-memory stage (not the simulated
    stage) and cook there. Returns (result, hulls) with hull vertices as returned by the cooking interface."""
    from omni.physx import get_physx_cooking_interface
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    sid = UsdUtils.StageCache.Get().Insert(st).ToLongInt()
    UsdGeom.Xform.Define(st, "/cook")
    m = UsdGeom.Mesh.Define(st, "/cook/mesh")
    src = UsdGeom.Mesh(mesh_prim)
    m.CreatePointsAttr(src.GetPointsAttr().Get())
    m.CreateFaceVertexCountsAttr(src.GetFaceVertexCountsAttr().Get())
    m.CreateFaceVertexIndicesAttr(src.GetFaceVertexIndicesAttr().Get())
    m.AddTransformOp().Set(Gf.Matrix4d(np.asarray(mesh_in_coll).T.tolist()))
    p = m.GetPrim()
    for schema in coll_prim.GetAppliedSchemas():
        p.AddAppliedSchema(schema)
    for a in coll_prim.GetAttributes():
        n = a.GetName()
        if n.startswith(("physxConvexHullCollision:", "physxConvexDecompositionCollision:", "physxCollision:", "physics:approximation",
                         "physics:collisionEnabled")):
            v = a.Get()
            if v is not None:
                p.CreateAttribute(n, a.GetTypeName()).Set(v)
    if not p.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(p)
    got = []

    def cb(result, convexes):
        got.append((str(result), [np.array([[v[0], v[1], v[2]] for v in c.vertices], dtype=np.float64) for c in convexes]))

    get_physx_cooking_interface().request_convex_collision_representation(sid, PhysicsSchemaTools.sdfPathToInt(p.GetPath()), False, cb)
    UsdUtils.StageCache.Get().Erase(st)
    return got[0] if got else ("no_callback", [])


def dump_scene(env, out_dir, extra=None):
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    shapes, conv = [], {}
    roots = ["/World/envs/env_0/Robot", "/World/envs/env_0/Robot_1", "/World/envs/env_0/Object", "/World/ground"]
    for root in roots:
        rp = stage.GetPrimAtPath(root)
        if not rp or not rp.IsValid():
            continue
        for prim in Usd.PrimRange(rp, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body = _rigid_ancestor(prim)
            l2w = _matrix(xc.GetLocalToWorldTransform(prim))
            b2w = _matrix(xc.GetLocalToWorldTransform(body)) if body else np.eye(4)
            rec = dict(path=str(prim.GetPath()), root=root, type=prim.GetTypeName(), body=str(body.GetPath()) if body else None,
                       prim_in_body=(np.linalg.inv(b2w) @ l2w).tolist(), prim_world=l2w.tolist(), body_world=b2w.tolist(),
                       collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(), material=_material(prim, stage))
            if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                pc = PhysxSchema.PhysxCollisionAPI(prim)
                rec.update(contact_offset=pc.GetContactOffsetAttr().Get(), rest_offset=pc.GetRestOffsetAttr().Get())
            t = prim.GetTypeName()
            if t == "Capsule":
                g = UsdGeom.Capsule(prim)
                rec.update(radius=g.GetRadiusAttr().Get(), height=g.GetHeightAttr().Get(), axis=g.GetAxisAttr().Get())
            elif t == "Sphere":
                rec.update(radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            elif t == "Cube":
                rec.update(size=UsdGeom.Cube(prim).GetSizeAttr().Get())
            elif t == "Cylinder":
                g = UsdGeom.Cylinder(prim)
                rec.update(radius=g.GetRadiusAttr().Get(), height=g.GetHeightAttr().Get(), axis=g.GetAxisAttr().Get())
            elif t == "Mesh":
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
                pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                result, hulls = _cooked_convexes(stage, prim)
                key = f"s{len(shapes)}"
                conv[key + "_points"] = pts
                for i, h in enumerate(hulls):
                    conv[f"{key}_hull{i}"] = h
                rec.update(approximation=approx, n_points=len(pts), cook_result=result, n_hulls=len(hulls), npz_key=key,
                           hull_vertex_counts=[len(h) for h in hulls])
            elif t == "Xform":  # URDF mesh collisions: CollisionAPI on an Xform whose child Mesh holds the geometry
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
                key = f"s{len(shapes)}"
                result, hulls = _cooked_convexes(stage, prim)
                for i, h in enumerate(hulls):
                    conv[f"{key}_hull{i}"] = h
                meshes = []
                for k2, ch in enumerate(q for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if q.GetTypeName() == "Mesh"):
                    m2w = _matrix(xc.GetLocalToWorldTransform(ch))
                    pts = np.array(UsdGeom.Mesh(ch).GetPointsAttr().Get(), dtype=np.float64)
                    conv[f"{key}_mesh{k2}_points"] = pts
                    r2, h2 = _cook_copy(prim, ch, np.linalg.inv(l2w) @ m2w)
                    for i, h in enumerate(h2):
                        conv[f"{key}_mesh{k2}_hull{i}"] = h
                    meshes.append(dict(path=str(ch.GetPath()), mesh_in_body=(np.linalg.inv(b2w) @ m2w).tolist(), mesh_in_prim=(np.linalg.inv(l2w) @ m2w).tolist(),
                                       n_points=len(pts), has_collision_api=ch.HasAPI(UsdPhysics.CollisionAPI),
                                       approximation=UsdPhysics.MeshCollisionAPI(ch).GetApproximationAttr().Get() if ch.HasAPI(UsdPhysics.MeshCollisionAPI) else None,
                                       cook_result=r2, n_hulls=len(h2), hull_vertex_counts=[len(h) for h in h2]))
                attrs, schemas = _physx_collision_attrs(prim)
                rec.update(approximation=approx, cook_result=result, n_hulls=len(hulls), npz_key=key, hull_vertex_counts=[len(h) for h in hulls],
                           meshes=meshes, physx_attrs=attrs, applied_schemas=schemas, hull_frame="cooked in a copy whose root is the collision prim: "
                           "vertices in the collision prim frame (mesh transform applied); body frame = prim_in_body @ v")
            elif t == "Plane":
                g = UsdGeom.Plane(prim)
                rec.update(axis=g.GetAxisAttr().Get())
            shapes.append(rec)
    np.savez_compressed(out_dir / "collision_convexes.npz", **conv)

    sim = env.simulator
    arts = {"robot": sim._robot, "robot_1": getattr(sim, "_robot_1", None)}
    bodies, joints = {}, {}
    for name, art in arts.items():
        if art is None:
            continue
        v = art.root_physx_view
        bodies[name] = dict(names=list(art.body_names), masses=_np(v.get_masses())[0].tolist(), coms_xyzw=_np(v.get_coms())[0].tolist(),
                            inertias=_np(v.get_inertias())[0].tolist(), material_per_shape=_np(v.get_material_properties())[0].tolist(),
                            body_pos_w=_np(art.data.body_pos_w)[0].tolist(), body_quat_w_wxyz=_np(art.data.body_quat_w)[0].tolist())
        joints[name] = dict(names=list(art.joint_names), limits=_np(v.get_dof_limits())[0].tolist(), armature=_np(v.get_dof_armatures())[0].tolist(),
                            friction=_np(v.get_dof_friction_coefficients())[0].tolist(), max_velocity=_np(v.get_dof_max_velocities())[0].tolist(),
                            max_force=_np(v.get_dof_max_forces())[0].tolist(), stiffness=_np(v.get_dof_stiffnesses())[0].tolist(),
                            damping=_np(v.get_dof_dampings())[0].tolist(), joint_pos=_np(art.data.joint_pos)[0].tolist())
    ov = sim._object.root_physx_view
    obj = dict(mass=_np(ov.get_masses()).tolist(), coms_xyzw=_np(ov.get_coms()).tolist(), inertias=_np(ov.get_inertias()).tolist(),
               material_per_shape=_np(ov.get_material_properties()).tolist())
    solver = {}
    for path in ("/World/envs/env_0/Robot", "/World/envs/env_0/Robot_1", "/World/envs/env_0/Object"):
        p = stage.GetPrimAtPath(path)
        for q in Usd.PrimRange(p, Usd.TraverseInstanceProxies()):
            if q.HasAPI(PhysxSchema.PhysxArticulationAPI):
                a = PhysxSchema.PhysxArticulationAPI(q)
                solver[str(q.GetPath())] = dict(kind="articulation", pos_iters=a.GetSolverPositionIterationCountAttr().Get(),
                                                vel_iters=a.GetSolverVelocityIterationCountAttr().Get(), self_collisions=a.GetEnabledSelfCollisionsAttr().Get())
            if q.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                r = PhysxSchema.PhysxRigidBodyAPI(q)
                solver.setdefault(str(q.GetPath()), {}).update(lin_damping=r.GetLinearDampingAttr().Get(), ang_damping=r.GetAngularDampingAttr().Get(),
                                                               max_depenetration_velocity=r.GetMaxDepenetrationVelocityAttr().Get(),
                                                               rb_pos_iters=r.GetSolverPositionIterationCountAttr().Get(),
                                                               rb_vel_iters=r.GetSolverVelocityIterationCountAttr().Get())
    scene = {}
    for q in stage.Traverse():
        if q.HasAPI(PhysxSchema.PhysxSceneAPI):
            s = PhysxSchema.PhysxSceneAPI(q)
            u = UsdPhysics.Scene(q)
            scene = dict(path=str(q.GetPath()), solver_type=s.GetSolverTypeAttr().Get(), gravity_dir=list(u.GetGravityDirectionAttr().Get() or []),
                         gravity_mag=u.GetGravityMagnitudeAttr().Get(), bounce_threshold=s.GetBounceThresholdAttr().Get(),
                         friction_offset_threshold=s.GetFrictionOffsetThresholdAttr().Get(), friction_correlation_distance=s.GetFrictionCorrelationDistanceAttr().Get(),
                         enable_stabilization=s.GetEnableStabilizationAttr().Get(), enable_ccd=s.GetEnableCCDAttr().Get(), time_steps_per_second=s.GetTimeStepsPerSecondAttr().Get())
            break
    term = env.action_manager.get_term("joint_control")
    rc = env.robot_config
    control = dict(dof_names=list(env.dof_names), default_dof_pos=_np(env.default_dof_pos).reshape(-1).tolist(), p_gains=_np(env.p_gains).reshape(-1).tolist(),
                   d_gains=_np(env.d_gains).reshape(-1).tolist(), action_scales=_np(env.action_scales).reshape(-1).tolist(),
                   torque_limits=_np(env.torque_limits).reshape(-1).tolist(), clip_actions=bool(rc.control.clip_actions), action_clip_value=float(rc.control.action_clip_value),
                   clip_torques=bool(rc.control.clip_torques), control_type=rc.control.control_type, decimation=int(sim.simulator_config.sim.control_decimation),
                   sim_dt=float(env.sim_dt), dt=float(env.dt), dof_pos_lower=list(rc.dof_pos_lower_limit_list), dof_pos_upper=list(rc.dof_pos_upper_limit_list),
                   effort_limits=list(rc.dof_effort_limit_list), vel_limits=list(rc.dof_vel_limit_list), armature=list(rc.dof_armature_list),
                   joint_friction=list(rc.dof_joint_friction_list), kp_scale=_np(getattr(term, "_kp_scale", 1.0)).reshape(-1)[:3].tolist(),
                   kd_scale=_np(getattr(term, "_kd_scale", 1.0)).reshape(-1)[:3].tolist(), rfi=bool(getattr(term, "_randomize_torque_rfi", False)),
                   ctrl_delay=bool(getattr(env, "_randomize_ctrl_delay", False)), body_list=list(sim._body_list), env_origin=_np(sim.scene.env_origins)[0].tolist())
    rec = dict(shapes=shapes, bodies=bodies, joints=joints, object=obj, solver=solver, physx_scene=scene, control=control, extra=extra or {})
    (out_dir / "scene.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


class EpisodeRecorder:
    """Per-step record of one evaluation episode. Wraps (does not replace) the SONIC controller's act to read its inputs."""

    def __init__(self, env, driver=None):
        self.env, self.driver = env, driver
        self.rows = []
        self.sonic = None
        self._orig = None
        if driver is not None:
            ctl = driver.ctl
            self._orig = ctl.act
            rec = self

            def act(*args, **kwargs):
                action, target, obs = rec._orig(*args, **kwargs)
                rec.sonic = dict(sonic_actor_obs=_np(obs["actor_obs"]).copy(), sonic_tokenizer=_np(obs["tokenizer"]).copy(),
                                 sonic_action=_np(action).copy(), sonic_target_s=_np(target).copy())
                return action, target, obs
            ctl.act = act

    def close(self):
        if self._orig is not None:
            self.driver.ctl.act = self._orig

    def pre(self, observations):
        """Before env.step: the observations the actions are computed from, and the state they were computed from."""
        sim = self.env.simulator
        self._cur = dict(actor_obs=_np(observations["actor_obs"]).copy(), teammate_obs=_np(observations["teammate_obs"]).copy(),
                         body_pos_w=_np(sim.agent_rigid_body_pos).copy(), body_quat_w_xyzw=_np(sim.agent_rigid_body_rot).copy(),
                         root_states=_np(sim.agent_root_states).copy(), dof_pos=_np(sim.agent_dof_pos).copy(), dof_vel=_np(sim.agent_dof_vel).copy())

    def post(self, actions):
        """After env.step: raw actions, PD target, per-substep torques / joint states, SONIC inputs (scheme C)."""
        term = self.env.action_manager.get_term("joint_control")
        r = self._cur
        r.update(actions=_np(actions).copy(), pd_target=_np(term.last_pd_target).copy(), torques_substep=_np(term.torques_substep).copy(),
                 dof_pos_substep=_np(term.dof_pos_substep).copy(), dof_vel_substep=_np(term.dof_vel_substep).copy())
        pol = getattr(self.env.action_manager, "policy_action", None)
        if pol is not None:
            r["policy_action"] = _np(pol).copy()
        if self.sonic is not None:
            r.update(self.sonic)
            self.sonic = None
        self.rows.append(r)
        self._cur = None

    def save(self, path):
        keys = self.rows[0].keys() if self.rows else []
        out = {k: np.stack([r[k] for r in self.rows]) for k in keys}
        np.savez_compressed(path, **out)
