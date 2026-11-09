#!/usr/bin/env python3
"""SONIC validation in holosoma: one rubber-hand G1, no object, track one reference.

* Simulator: holosoma IsaacSim single-robot WBT env (``g1_29dof_wbt``) with our
  rubber-hand URDF, no randomization, no observation noise, start at frame 0.
* Controller: system1.sonic_controller.SonicController (released checkpoint, G1 encoder).
  Joint targets q* = SONIC default + SONIC action_scale * a are passed through the
  existing holosoma action term as a_h = (q* - default_h) / scale_h, i.e. the same
  q* reaches the PD. With ``--sonic-pd`` (default) the PD gains are SONIC's
  (only hip_pitch differs from ours); the robot model (effort/velocity limits,
  armature, masses) stays ours.
* Metrics (SONIC im_eval definitions): per post-step k, MPJPE = mean over the 14
  cmd bodies of ||p_robot - p_ref[k]|| in mm; episode MPJPE = mean over k=1..N-1.
  Fall = SONIC eval termination: |dz pelvis| > 0.25 or quat_err(pelvis)^2 > 1.0 or
  any of {ankle_roll, wrist_yaw} x2 |dz| > 0.25. Episode stops at the first fall.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "holosoma"))
sys.path.insert(0, str(ROOT))

SONIC_HIP_PITCH_KP = 99.098427777  # STIFFNESS_7520_22 (SONIC g1.py actuators "legs")
SONIC_HIP_PITCH_KD = 6.308801854  # DAMPING_7520_22


def build_config(motion_file: str, n_frames: int, sonic_pd: bool, device_headless=True):
    from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
    from holosoma.config_types.curriculum import CurriculumManagerCfg
    from holosoma.config_types.randomization import RandomizationManagerCfg
    from holosoma.config_types.reward import RewardManagerCfg
    from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
    from holosoma.config_values.wbt.g1.command import g1_29dof_wbt_command, motion_config
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt

    mc = replace(motion_config, motion_file=motion_file, use_adaptive_timesteps_sampler=False,
                 start_at_timestep_zero_prob=1.0, noise_to_initial_pose=NoiseToInitialPoseConfig())
    cmd = replace(g1_29dof_wbt_command, setup_terms={
        "motion_command": replace(g1_29dof_wbt_command.setup_terms["motion_command"], params={"motion_config": mc})})
    ctl = g1_29dof_wbt.robot.control
    if sonic_pd:
        ctl = replace(ctl, stiffness={**ctl.stiffness, "hip_pitch": SONIC_HIP_PITCH_KP},
                      damping={**ctl.damping, "hip_pitch": SONIC_HIP_PITCH_KD})
    robot = replace(g1_29dof_wbt.robot, control=ctl,
                    asset=replace(g1_29dof_wbt.robot.asset, urdf_file="g1/main_mesh_collision_rubberhand.urdf"))
    sim = replace(g1_29dof_wbt.simulator, config=replace(g1_29dof_wbt.simulator.config,
                  sim=replace(g1_29dof_wbt.simulator.config.sim, max_episode_length_s=(n_frames + 10) / 50.0)))
    term = TerminationManagerCfg(terms={"timeout": TerminationTermCfg(
        func="holosoma.managers.termination.terms.common:timeout_exceeded", is_timeout=True)})
    obs = g1_29dof_wbt.observation
    obs = replace(obs, groups={k: replace(v, enable_noise=False) for k, v in obs.groups.items()})
    return replace(g1_29dof_wbt, training=replace(g1_29dof_wbt.training, num_envs=1, headless=device_headless, seed=241003),
                   robot=robot, command=cmd, simulator=sim, termination=term, observation=obs,
                   randomization=_no_randomization(), reward=RewardManagerCfg(), curriculum=g1_29dof_wbt.curriculum)


def _no_randomization():
    """Only the two setup terms the action term requires, both disabled (no delay, no torque RFI)."""
    from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

    T = "holosoma.managers.randomization.terms.locomotion"
    return RandomizationManagerCfg(setup_terms={
        "setup_action_delay_buffers": RandomizationTermCfg(func=f"{T}:setup_action_delay_buffers",
                                                           params={"ctrl_delay_step_range": [0, 1], "enabled": False}),
        "setup_torque_rfi": RandomizationTermCfg(func=f"{T}:setup_torque_rfi", params={"enabled": False}),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-prefix", type=Path, required=True, help="prefix of *.holosoma.npz / *.sonic.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-sonic-pd", action="store_true")
    p.add_argument("--max-steps", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sref = np.load(f"{a.ref_prefix}.sonic.npz")
    N = sref["dof_pos"].shape[0] if not a.max_steps else min(a.max_steps + 1, sref["dof_pos"].shape[0])
    config = build_config(str(Path(f"{a.ref_prefix}.holosoma.npz").resolve()), N, not a.no_sonic_pd)

    from holosoma.utils.eval_utils import init_sim_imports
    app = init_sim_imports(config)
    import torch
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app
    from system1 import sonic_controller as sc

    env = None
    try:
        env = get_class(config.env_class)(get_tyro_env_config(config), device=a.device)
        env.set_is_evaluating()
        dev = env.device
        sim = env.simulator
        h_names = list(sim.dof_names)
        ctl = sc.SonicController(batch=1, device=dev, holosoma_joint_names=h_names)
        cmd_bodies = ctl.spec["cmd_body_names"]
        b_idx = [sim._body_list.index(b) for b in cmd_bodies]
        pelvis = cmd_bodies.index("pelvis")
        ee = [cmd_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
        R = {k: torch.tensor(sref[k], dtype=torch.float32, device=dev) for k in ("dof_pos", "dof_vel", "root_quat_w", "body_pos_w")}
        Nref = R["dof_pos"].shape[0]
        env.reset_all()  # initializes manager buffers (it also takes one zero-action step)
        # Reset again without stepping, as the paper evaluator does: reference frame 0 written to sim.
        ids = torch.arange(env.num_envs, device=dev)
        env.reset_envs_idx(ids)
        sim.set_actor_root_state_tensor_robots(ids, sim.robot_root_states)
        sim.set_dof_state_tensor_robots(ids, sim.dof_state)
        sim.refresh_sim_tensors()
        origin = sim.scene.env_origins[0]
        default_h = env.default_dof_pos.reshape(-1)
        scale_h = env.action_scales.reshape(-1)
        cmdterm = env.command_manager.get_state("motion_command")
        if int(cmdterm.time_steps[0].item()) != 0:
            raise RuntimeError("reference not at frame 0 after reset")

        def state():
            rs = sim.robot_root_states  # [1,13] xyzw
            q = sc.xyzw_to_wxyz(rs[:, 3:7])
            w_b = sc.quat_apply(sc.quat_conj(q), rs[:, 10:13])
            return q, w_b, sim.dof_pos, sim.dof_vel

        rec = {k: [] for k in ("mpjpe_mm", "pelvis_dz", "pelvis_ori_err2", "ee_dz_max", "root_pos", "root_quat_wxyz", "dof_pos_h", "body_pos14", "action")}
        fall_step = None
        k_last = 0
        for k in range(1, N):
            q, w_b, jp_h, jv_h = state()
            t = torch.tensor([k - 1], device=dev)
            idx = sc.future_indices(t, Nref, ctl.n_future, ctl.skip)
            action, q_target_s, _ = ctl.act(q, w_b, ctl.to_sonic(jp_h), ctl.to_sonic(jv_h),
                                            R["dof_pos"][idx], R["dof_vel"][idx], R["root_quat_w"][idx])
            q_target_h = ctl.to_holosoma(q_target_s)
            a_h = (q_target_h - default_h) / scale_h
            env.step({"actions": a_h})
            # post-step state vs reference frame k (SONIC im_eval / holosoma eval alignment)
            body = sim._rigid_body_pos[:, b_idx] - origin
            ref = R["body_pos_w"][k][None]
            err = torch.linalg.norm(body - ref, dim=-1)
            rq = sc.xyzw_to_wxyz(sim.robot_root_states[:, 3:7])
            dq = sc.quat_mul(sc.quat_conj(R["root_quat_w"][k][None]), rq)
            ang = 2 * torch.atan2(torch.linalg.norm(dq[:, 1:], dim=-1), dq[:, 0].abs())
            pel_dz = (body[:, pelvis, 2] - ref[:, pelvis, 2]).abs()
            ee_dz = (body[:, ee, 2] - ref[:, ee, 2]).abs().max(dim=-1).values
            rec["mpjpe_mm"].append(err.mean().item() * 1000)
            rec["pelvis_dz"].append(pel_dz.item())
            rec["pelvis_ori_err2"].append((ang ** 2).item())
            rec["ee_dz_max"].append(ee_dz.item())
            rec["root_pos"].append((sim.robot_root_states[0, :3] - origin).cpu().numpy())
            rec["root_quat_wxyz"].append(rq[0].cpu().numpy())
            rec["dof_pos_h"].append(sim.dof_pos[0].cpu().numpy())
            rec["body_pos14"].append(body[0].cpu().numpy())
            rec["action"].append(action[0].cpu().numpy())
            k_last = k
            if pel_dz.item() > 0.25 or ang.item() ** 2 > 1.0 or ee_dz.item() > 0.25:
                fall_step = k
                break
        mp = np.array(rec["mpjpe_mm"])
        result = dict(
            ref_prefix=str(a.ref_prefix), provenance=json.loads(str(sref["provenance"])), steps_executed=k_last,
            horizon_steps=N - 1, fell=fall_step is not None, fall_step=fall_step,
            fall_time_s=None if fall_step is None else fall_step * 0.02,
            mpjpe_g_mm=float(mp.mean()), mpjpe_g_mm_first_200=float(mp[:200].mean()),
            max_pelvis_dz_m=float(np.max(rec["pelvis_dz"])), max_ee_dz_m=float(np.max(rec["ee_dz_max"])),
            sonic_pd=not a.no_sonic_pd, device=a.device,
            holosoma_action_scales=scale_h.tolist(), holosoma_p_gains=env.p_gains.reshape(-1).tolist(),
        )
        np.savez_compressed(a.out / "trajectory.npz", **{k: np.asarray(v) for k, v in rec.items()},
                            ref_body_pos14=sref["body_pos_w"][: k_last + 1])
        (a.out / "result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if not k.startswith("holosoma_")}, indent=1))
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
