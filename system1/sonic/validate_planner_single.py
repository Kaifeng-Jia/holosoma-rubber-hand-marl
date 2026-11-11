#!/usr/bin/env python3
"""T02 gate "规划器单独验证": one rubber-hand G1, no object, planner + teleop encoder following a reference
pelvis route (SONIC's own sample walk). Pass: no fall and final pelvis horizontal error <= 0.30 m.

Same environment and low-level settings as validate_single.py (holosoma WBT env, GPU PhysX, TF32 off,
hip_pitch PD -> SONIC, frozen action scales, PD gets q* verbatim). Per control step s (reference frame s):
  * planner (system1/scheme_b.py): initialised from the robot state at s = 0, replanned every 5 steps;
    waypoints = reference pelvis at gen frame + i*H/4 (i = 1..4), headings = reference pelvis yaw;
    --feedback shift adds (planner root - robot pelvis) to the waypoints (xy) and the yaw difference to the
    headings, --feedback none uses the reference values as they are;
  * teleop encoder: lower-body window from the planner motion, 3-point targets from the reference at frame s
    relative to the reference pelvis, motion_anchor_ori_b against the planner root.
Fall rule and MPJPE as validate_single.py (SONIC im_eval definitions, vs the reference).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "holosoma"))
sys.path.insert(0, str(ROOT))
from system1.sonic.validate_single import build_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-prefix", type=Path, required=True, help="prefix of *.holosoma.npz / *.sonic.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--feedback", choices=("shift", "none"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--diag-fall-rule", choices=("sonic", "height_only"), default="sonic",
                   help="diagnostic only: height_only drops the orientation term of the fall rule (never the gate)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sref = np.load(f"{a.ref_prefix}.sonic.npz")
    href = np.load(f"{a.ref_prefix}.holosoma.npz", allow_pickle=True)
    N = sref["dof_pos"].shape[0] if not a.max_steps else min(a.max_steps + 1, sref["dof_pos"].shape[0])
    config = build_config(str(Path(f"{a.ref_prefix}.holosoma.npz").resolve()), N, True)

    from holosoma.utils.eval_utils import init_sim_imports
    app = init_sim_imports(config)
    import torch
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app
    from system1 import scheme_b as sb
    from system1 import sonic_controller as sc

    env = None
    try:
        env = get_class(config.env_class)(get_tyro_env_config(config), device=a.device)
        env.set_is_evaluating()
        dev = env.device
        sim = env.simulator
        h_names = list(sim.dof_names)
        ctl = sc.SonicController(batch=1, device=dev, holosoma_joint_names=h_names, tf32=False)
        term = env.action_manager.get_term("joint_control")
        cmd_bodies = ctl.spec["cmd_body_names"]
        b_idx = [sim._body_list.index(b) for b in cmd_bodies]
        pelvis = cmd_bodies.index("pelvis")
        ee = [cmd_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
        R = {k: torch.tensor(sref[k], dtype=torch.float32, device=dev) for k in ("root_quat_w", "body_pos_w")}
        hb = [str(x) for x in href["body_names"]]
        ref_pel_pos = href["body_pos_w"][:, hb.index("pelvis")].astype(np.float64)
        ref_pel_quat = href["body_quat_w"][:, hb.index("pelvis")].astype(np.float64)
        pts_i = [hb.index(n) for n in sc.VR3_BODIES]
        ref_pts_pos = href["body_pos_w"][:, pts_i].astype(np.float64)
        ref_pts_quat = href["body_quat_w"][:, pts_i].astype(np.float64)
        Nref = len(ref_pel_pos)
        env.reset_all()
        ids = torch.arange(env.num_envs, device=dev)
        env.reset_envs_idx(ids)
        sim.set_actor_root_state_tensor_robots(ids, sim.robot_root_states)
        sim.set_dof_state_tensor_robots(ids, sim.dof_state)
        sim.refresh_sim_tensors()
        origin = sim.scene.env_origins[0]
        org = origin.cpu().numpy().astype(np.float64)
        default_h = env.default_dof_pos.reshape(-1)
        scale_h = env.action_scales.reshape(-1)
        planner = sb.Planner()
        track = sb.PlannerTrack(planner, ctl.joint_names)
        H = sb.MAX_FRAMES_30 / 30.0 * 50.0

        def state():
            rs = sim.robot_root_states
            q = sc.xyzw_to_wxyz(rs[:, 3:7])
            w_b = sc.quat_apply(sc.quat_conj(q), rs[:, 10:13])
            return rs[0, :3].cpu().numpy().astype(np.float64) - org, q, w_b, sim.dof_pos, sim.dof_vel

        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        rec = {k: [] for k in ("mpjpe_mm", "pelvis_dz", "pelvis_ori_err2", "ee_dz_max", "root_pos", "planner_root_pos",
                               "ref_pelvis_pos", "d_xy", "waypoint_last", "action",
                               # logging only (added after the 2026-11-10 gate failure; behaviour unchanged)
                               "robot_yaw", "planner_yaw", "ref_root_yaw", "waypoint_last_yaw")}
        fall_step = None
        k_last = 0
        t_start = time.perf_counter()
        for k in range(1, N):
            s = k - 1
            pos_r, q, w_b, jp_h, jv_h = state()
            q_np = q[0].cpu().numpy().astype(np.float64)
            if s == 0:
                track.initialize(pos_r, q_np, ctl.to_sonic(jp_h)[0].cpu().numpy().astype(np.float64))
            if s % sb.PLANNER_TICK == 0:
                proot, pquat = track.root()
                shift = (proot[:2] - pos_r[:2]) if a.feedback == "shift" else np.zeros(2)
                dyaw = sb.wrap(sb.yaw_of(pquat) - sb.yaw_of(q_np)) if a.feedback == "shift" else 0.0
                wp, wy, _ = sb.waypoints(ref_pel_pos, ref_pel_quat, s, Nref, H, shift, dyaw)
                d = wp[-1, :2] - proot[:2]
                mdir = np.r_[d / max(np.linalg.norm(d), 1e-9), 0.0]
                facing = (np.cos(wy[-1]), np.sin(wy[-1]), 0.0)
                track.replan(sb.MODE_WALK, sb.ref_planar_speed(ref_pel_pos, s, Nref), mdir, facing,
                             float(ref_pel_pos[s, 2]), 1, wp, wy)
                rec["waypoint_last"].append(np.r_[s, wp[-1]])
                rec["waypoint_last_yaw"].append(np.r_[s, wy[-1]])
            lp, lv = track.future_window(ctl.n_future, ctl.skip)
            proot, pquat = track.root()
            action, q_target_s, _ = ctl.act_teleop(
                q, w_b, ctl.to_sonic(jp_h), ctl.to_sonic(jv_h), T(lp)[None], T(lv)[None],
                T(ref_pel_pos[s])[None], T(ref_pel_quat[s])[None], T(ref_pts_pos[s])[None], T(ref_pts_quat[s])[None],
                ori_anchor_quat=T(pquat)[None])
            q_target_h = ctl.to_holosoma(q_target_s)
            term.joint_target_override = q_target_h
            env.step({"actions": (q_target_h - default_h) / scale_h})
            track.advance()
            body = sim._rigid_body_pos[:, b_idx] - origin
            ref = R["body_pos_w"][k][None]
            err = torch.linalg.norm(body - ref, dim=-1)
            rq = sc.xyzw_to_wxyz(sim.robot_root_states[:, 3:7])
            dq = sc.quat_mul(sc.quat_conj(R["root_quat_w"][k][None]), rq)
            ang = 2 * torch.atan2(torch.linalg.norm(dq[:, 1:], dim=-1), dq[:, 0].abs())
            pel_dz = (body[:, pelvis, 2] - ref[:, pelvis, 2]).abs()
            ee_dz = (body[:, ee, 2] - ref[:, ee, 2]).abs().max(dim=-1).values
            rp = (sim.robot_root_states[0, :3] - origin).cpu().numpy().astype(np.float64)
            rec["mpjpe_mm"].append(err.mean().item() * 1000)
            rec["pelvis_dz"].append(pel_dz.item())
            rec["pelvis_ori_err2"].append((ang ** 2).item())
            rec["ee_dz_max"].append(ee_dz.item())
            rec["root_pos"].append(rp)
            rec["planner_root_pos"].append(track.root()[0].copy())
            rec["ref_pelvis_pos"].append(ref_pel_pos[min(k, Nref - 1)])
            rec["d_xy"].append(float(np.linalg.norm(rp[:2] - ref_pel_pos[min(k, Nref - 1), :2])))
            rec["action"].append(action[0].cpu().numpy())
            rec["robot_yaw"].append(float(sb.yaw_of(rq[0].cpu().numpy().astype(np.float64))))
            rec["planner_yaw"].append(float(sb.yaw_of(track.root()[1])))
            rec["ref_root_yaw"].append(float(sb.yaw_of(R["root_quat_w"][k].cpu().numpy().astype(np.float64))))
            k_last = k
            ori_fail = ang.item() ** 2 > 1.0 and a.diag_fall_rule == "sonic"
            if pel_dz.item() > 0.25 or ori_fail or ee_dz.item() > 0.25:
                fall_step = k
                break
        wall = time.perf_counter() - t_start
        d = np.array(rec["d_xy"])
        final_err = float(d[-1])
        passed = fall_step is None and k_last == N - 1 and final_err <= 0.30 and a.diag_fall_rule == "sonic"
        result = dict(ref_prefix=str(a.ref_prefix), feedback=a.feedback, steps_executed=k_last, horizon_steps=N - 1,
                      fell=fall_step is not None, fall_step=fall_step, final_pelvis_xy_err_m=final_err,
                      max_pelvis_xy_err_m=float(d.max()), mean_pelvis_xy_err_m=float(d.mean()),
                      mpjpe_g_mm=float(np.mean(rec["mpjpe_mm"])), planner_calls=planner.calls,
                      planner_seconds=planner.seconds, wall_seconds=wall, gate_threshold_m=0.30, gate_passed=passed,
                      constants=sb.preregistered_constants(), tf32=False, device=a.device, diag_fall_rule=a.diag_fall_rule,
                      max_ori_err_rad=float(np.sqrt(np.max(rec["pelvis_ori_err2"]))))
        np.savez_compressed(a.out / "trajectory.npz", **{k: np.asarray(v) for k, v in rec.items()})
        (a.out / "result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return 0
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
