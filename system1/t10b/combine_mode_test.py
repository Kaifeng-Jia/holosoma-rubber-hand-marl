#!/usr/bin/env python3
"""T10b A addendum: empirical check of the PhysX friction combine mode between the hand and the object.

In the evaluation, the rubber-hand shapes (articulation links) and the object shapes (an Isaac Lab RigidObject) have no USD-bound
physics material; their friction is written with the PhysX tensor API (hand: system1/t10b/hand_material.py, object: holosoma
set_object_rigid_body_material_startup). The combine mode of those materials cannot be read from the --sim2sim-dump.

This script measures it in a minimal scene that uses the same two write paths:
- "hand": a fixed-base single-link articulation (a flat plate, no USD material), material written with
  Articulation.root_physx_view.set_material_properties, exactly as hand_material.py does;
- "object": a free RigidObject box resting on the plate (no USD material), material written with
  RigidObject.root_physx_view.set_material_properties, exactly as holosoma does, before the hand (same order as the evaluation);
- scene-level physics as in the evaluation dump (logs_system1/t10b/verify_hand_material/dump/scene.json): dt 1/200, TGS,
  scene default material 1.0 / 1.0 average, ground plane 1.0 / 1.0 multiply, gravity 9.81, bounce threshold 0.5, friction offset
  threshold 0.04, friction correlation distance 0.025, stabilization and CCD off, articulation solver 8 / 4, rigid-body solver
  16 / 1, GPU PhysX.
Measurement: after settling, the box gets a horizontal velocity; mu_eff = -(dv/dt) / g from a linear fit of the box velocity
over the sliding window. Static and dynamic friction are set equal, so mu_eff is the combined dynamic friction.
Controls (method check): plate and box with USD-bound materials of a known combine mode (average, min, multiply, max) and no
tensor-API write; hand 0.75 / object 0.5 so that the four modes predict four different values (0.625, 0.5, 0.375, 0.75).

    hssim_s1/bin/python -B system1/t10b/combine_mode_test.py --out logs_system1/t10b/combine_mode_test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = ("average", "min", "multiply", "max")
G = 9.81
DT = 1.0 / 200.0
PLATE = (3.0, 1.0, 0.05)  # x, y, z size (m)
BOX = (0.3, 0.3, 0.1)  # flat box: tipping needs mu > 3
PLATE_TOP = 0.3
SPACING = 2.0
SETTLE_STEPS = 100
SLIDE_STEPS = 400
START_X = -1.3


def predictions(a: float, b: float) -> dict:
    return dict(average=(a + b) / 2.0, min=min(a, b), multiply=a * b, max=max(a, b))


def fit_mu(v: np.ndarray, v0: float, skip: int = 3, stop_frac: float = 0.1, dt: float = DT, g: float = G) -> dict:
    """Linear fit of the box velocity over the sliding window: from `skip` steps after the push to the last step with
    v > stop_frac * v0. Returns mu = -slope / g, the window and the fit residual."""
    v = np.asarray(v, dtype=float)
    above = np.flatnonzero(v > stop_frac * v0)
    if above.size == 0 or above[-1] < skip + 3:
        return dict(mu=float("nan"), first_step=skip, last_step=int(above[-1]) if above.size else -1, n=0, rms_residual=float("nan"))
    k = np.arange(skip, int(above[-1]) + 1)
    t = k * dt
    slope, icpt = np.polyfit(t, v[k], 1)
    res = v[k] - (slope * t + icpt)
    return dict(mu=float(-slope / g), first_step=int(k[0]), last_step=int(k[-1]), n=int(k.size), rms_residual=float(np.sqrt(np.mean(res**2))))


def closest_mode(mu: float, hand: float, obj: float) -> tuple[str, dict]:
    p = predictions(hand, obj)
    err = {m: abs(mu - p[m]) for m in MODES}
    best = min(err.values())
    return "+".join(m for m in MODES if err[m] - best < 1e-6), err


def conditions(hand_frictions, object_friction, speeds, control_hand_friction=0.75):
    out = []
    for hf in hand_frictions:
        for s in speeds:
            out.append(dict(kind="tensor_api", hand_friction=hf, object_friction=object_friction, speed=s, usd_combine=None))
    for mode in MODES:
        for s in speeds:
            out.append(dict(kind=f"usd_bound_{mode}", hand_friction=control_hand_friction, object_friction=object_friction, speed=s,
                            usd_combine=mode))
    return out


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hand-frictions", default="1.0,0.75,0.5,0.25")
    ap.add_argument("--object-friction", type=float, default=0.5)
    ap.add_argument("--speeds", default="1.0,2.0")
    ap.add_argument("--control-hand-friction", type=float, default=0.75,
                    help="hand friction of the USD-bound controls; 0.75 with object 0.5 gives four distinct predictions")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    args.headless = True
    launcher = AppLauncher(args)
    app = launcher.app

    import torch
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext

    hand_frictions = [float(x) for x in args.hand_frictions.split(",")]
    speeds = [float(x) for x in args.speeds.split(",")]
    conds = conditions(hand_frictions, args.object_friction, speeds, args.control_hand_friction)
    n = len(conds)
    args.out.mkdir(parents=True, exist_ok=True)

    sim = SimulationContext(SimulationCfg(
        dt=DT, render_interval=1, device=args.device, gravity=(0.0, 0.0, -G),
        physx=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.5, friction_offset_threshold=0.04, friction_correlation_distance=0.025,
                       enable_stabilization=False, enable_ccd=False),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    ))
    stage = sim.stage
    ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="multiply", restitution_combine_mode="multiply"))
    ground.func("/World/ground", ground)

    rb = dict(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0,
              linear_damping=0.0, angular_damping=0.0)
    for i, c in enumerate(conds):
        y = SPACING * i
        mat_h = mat_o = None
        if c["usd_combine"]:
            mat_h = sim_utils.RigidBodyMaterialCfg(static_friction=c["hand_friction"], dynamic_friction=c["hand_friction"], restitution=0.0,
                                                   friction_combine_mode=c["usd_combine"], restitution_combine_mode="average")
            mat_o = sim_utils.RigidBodyMaterialCfg(static_friction=c["object_friction"], dynamic_friction=c["object_friction"], restitution=0.0,
                                                   friction_combine_mode=c["usd_combine"], restitution_combine_mode="average")
        root = f"/World/C{i:02d}/Plate"
        UsdGeom.Xform.Define(stage, root)
        link = f"{root}/link"
        plate_cfg = sim_utils.CuboidCfg(size=PLATE, rigid_props=sim_utils.RigidBodyPropertiesCfg(**rb), mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                                        collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat_h)
        plate_cfg.func(link, plate_cfg, translation=(0.0, y, PLATE_TOP - PLATE[2] / 2))
        rp = stage.GetPrimAtPath(root)
        UsdPhysics.ArticulationRootAPI.Apply(rp)
        pa = PhysxSchema.PhysxArticulationAPI.Apply(rp)
        pa.CreateSolverPositionIterationCountAttr(8)
        pa.CreateSolverVelocityIterationCountAttr(4)
        pa.CreateEnabledSelfCollisionsAttr(False)
        j = UsdPhysics.FixedJoint.Define(stage, f"{root}/root_joint")
        j.CreateBody1Rel().SetTargets([link])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(y), float(PLATE_TOP - PLATE[2] / 2)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        box_cfg = sim_utils.CuboidCfg(size=BOX, rigid_props=sim_utils.RigidBodyPropertiesCfg(**rb), mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                      collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat_o)
        box_cfg.func(f"/World/C{i:02d}/Box", box_cfg, translation=(START_X, y, PLATE_TOP + BOX[2] / 2 + 0.002))

    plate = Articulation(ArticulationCfg(prim_path="/World/C.*/Plate", spawn=None, actuators={},
                                         init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={})))
    box = RigidObject(RigidObjectCfg(prim_path="/World/C.*/Box", spawn=None))
    sim.reset()

    pp = [p.split("/")[2] for p in plate.root_physx_view.prim_paths]
    bp = [p.split("/")[2] for p in box.root_physx_view.prim_paths]
    if pp != bp or len(pp) != n:
        raise RuntimeError(f"view order mismatch: {pp} vs {bp}")
    order = [int(p[1:]) for p in pp]

    tensor_ids = [k for k, ci in enumerate(order) if conds[ci]["kind"] == "tensor_api"]
    ids = torch.tensor(tensor_ids, dtype=torch.long, device="cpu")
    mat_before = dict(hand=plate.root_physx_view.get_material_properties().clone(), object=box.root_physx_view.get_material_properties().clone())
    mo = mat_before["object"].clone()
    mo[ids, :, 0] = args.object_friction
    mo[ids, :, 1] = args.object_friction
    mo[ids, :, 2] = 0.0
    box.root_physx_view.set_material_properties(mo, ids)  # object first, as holosoma does at startup
    mh = mat_before["hand"].clone()
    for k in tensor_ids:
        hf = conds[order[k]]["hand_friction"]
        mh[k, :, 0] = hf
        mh[k, :, 1] = hf
        mh[k, :, 2] = 0.0
    plate.root_physx_view.set_material_properties(mh, ids)  # then the hand, as the evaluator does
    mat_after = dict(hand=plate.root_physx_view.get_material_properties().clone(), object=box.root_physx_view.get_material_properties().clone())
    if not (torch.equal(mat_after["hand"], mh) and torch.equal(mat_after["object"], mo)):
        raise RuntimeError("PhysX did not retain the requested materials")

    for _ in range(SETTLE_STEPS):
        sim.step(render=False)
        box.update(DT)
        plate.update(DT)
    z_settled = box.data.root_pos_w[:, 2].cpu().numpy().copy()
    vel = torch.zeros(n, 6, device=box.device)
    vel[:, 0] = torch.tensor([conds[order[k]]["speed"] for k in range(n)], device=box.device)
    box.write_root_velocity_to_sim(vel)
    vx, pz, wy, px = [], [], [], []
    for _ in range(SLIDE_STEPS):
        sim.step(render=False)
        box.update(DT)
        d = box.data
        vx.append(d.root_lin_vel_w[:, 0].cpu().numpy().copy())
        px.append(d.root_pos_w[:, 0].cpu().numpy().copy())
        pz.append(d.root_pos_w[:, 2].cpu().numpy().copy())
        wy.append(d.root_ang_vel_w[:, 1].cpu().numpy().copy())
    vx, px, pz, wy = (np.stack(a, 1) for a in (vx, px, pz, wy))
    plate_pos = plate.data.root_pos_w.cpu().numpy()

    rows = []
    for k in range(n):
        c = conds[order[k]]
        f = fit_mu(vx[k], c["speed"])
        pred = predictions(c["hand_friction"], c["object_friction"])
        best, err = closest_mode(f["mu"], c["hand_friction"], c["object_friction"])
        rows.append(dict(condition=order[k], **{kk: c[kk] for kk in ("kind", "hand_friction", "object_friction", "speed", "usd_combine")},
                         hand_material_read_back=mat_after["hand"][k].tolist(), object_material_read_back=mat_after["object"][k].tolist(),
                         hand_material_before=mat_before["hand"][k].tolist(), object_material_before=mat_before["object"][k].tolist(),
                         mu_measured=f["mu"], fit_first_step=f["first_step"], fit_last_step=f["last_step"], fit_n=f["n"],
                         fit_rms_residual_m_s=f["rms_residual"], **{f"mu_pred_{m}": pred[m] for m in MODES},
                         **{f"abs_err_{m}": err[m] for m in MODES}, closest_mode=best,
                         box_z_settled_m=float(z_settled[k]), box_z_range_while_sliding_m=float(pz[k].max() - pz[k].min()),
                         box_max_abs_pitch_rate_rad_s=float(np.abs(wy[k]).max()), box_travel_m=float(px[k, -1] - START_X),
                         plate_root_pos_w=plate_pos[k].tolist()))
    rows.sort(key=lambda r: r["condition"])
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    try:
        from isaacsim.core.version import get_version
        isaac_version = ".".join(str(x) for x in get_version()[:3])
    except Exception as e:  # noqa: BLE001
        isaac_version = f"unavailable: {e}"
    meta = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), git_head=git("rev-parse", "HEAD"), git_dirty=git("status", "--porcelain"),
                script_sha256=sha256(Path(__file__)), device=args.device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                isaac_sim_version=isaac_version, dt=DT, g=G, plate_size_m=PLATE, box_size_m=BOX, box_mass_kg=1.0, plate_top_m=PLATE_TOP,
                settle_steps=SETTLE_STEPS, slide_steps=SLIDE_STEPS, start_x=START_X,
                scene=dict(default_material=dict(static=1.0, dynamic=1.0, combine="average"), ground=dict(static=1.0, dynamic=1.0, combine="multiply"),
                           solver="TGS", articulation_iters=(8, 4), rigid_body_iters=(16, 1), bounce_threshold=0.5, friction_offset_threshold=0.04,
                           friction_correlation_distance=0.025, enable_stabilization=False, enable_ccd=False),
                view_prim_order=pp)
    (args.out / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
    np.savez_compressed(args.out / "traces.npz", vx=vx, px=px, pz=pz, wy=wy, order=np.array(order))
    for r in rows:
        print(f"{r['kind']:<22} hand {r['hand_friction']:.2f} obj {r['object_friction']:.2f} v0 {r['speed']:.1f}  mu {r['mu_measured']:.4f}  "
              f"pred avg {r['mu_pred_average']:.3f} min {r['mu_pred_min']:.3f} mult {r['mu_pred_multiply']:.3f} max {r['mu_pred_max']:.3f}  -> {r['closest_mode']}")
    sys.stdout.flush()
    os._exit(0)  # app.close() hangs after the run in this setup (try 2); outputs are already written


if __name__ == "__main__":
    sys.exit(main())
