#!/usr/bin/env python3
"""Drift analysis (iii) control group: a single-person prior (D's actor initialization) tracks a
role reference in the same single-robot setting as validate_single.py (no object, frame 0,
no randomization, GPU PhysX), without SONIC.

The prior was trained in holosoma's single-robot WBT env (GhostTeammate manager: the four
teammate channels are reserved and zero). Here: actor_obs (WBT terms, noise off) + teammate_obs
(zeros), its own frozen normalizer, deterministic action mean, original PD and action scales.
The critic group is dropped (it reads the object). Output format = validate_single.py so the
same drift metrics apply.
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
PRIORS = dict(push="marl_compat_a1_v1", pull="marl_compat_pull_v1", kick="marl_compat_kick_v1")
PRIOR_ROOT = Path("/home/kevin/holosoma-rubber-hand-marl/logs/WholeBodyTracking")
PRIOR_SHA = dict(push="11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224",
                 pull="f63a697a9e3d5d316ef88e7c5c8a94e04a4f340b563abe67e7be27ae411f2364",
                 kick="1555968f678c2b69fcd6f09c64d0a6252683eab902edd773a84acc205d0f5491")


def build_config(motion_file, n_frames):
    from holosoma.config_types.observation import ObservationManagerCfg
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt
    from holosoma.config_values.wbt.g1.observation import actor_obs_shared, teammate_obs_marl_compat
    from system1.sonic.validate_single import build_config as sonic_build

    cfg = sonic_build(motion_file, n_frames, sonic_pd=False)  # same env, original PD / scales
    obs = ObservationManagerCfg(groups={"actor_obs": replace(actor_obs_shared, enable_noise=False),
                                        "teammate_obs": teammate_obs_marl_compat})
    return replace(cfg, env_class="holosoma.envs.wbt.wbt_marl_compat_manager.GhostTeammateWholeBodyTrackingManager",
                   observation=obs, robot=replace(cfg.robot, control=g1_29dof_wbt.robot.control))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=tuple(PRIORS), required=True)
    p.add_argument("--ref-prefix", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sref = np.load(f"{a.ref_prefix}.sonic.npz")
    N = sref["dof_pos"].shape[0]
    config = build_config(str(Path(f"{a.ref_prefix}.holosoma.npz").resolve()), N)
    from holosoma.utils.eval_utils import init_sim_imports

    app = init_sim_imports(config)
    import hashlib

    import torch
    from holosoma.agents.mappo.initialization import initialize_plan5_model_bundle
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_w_object
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app
    from system1 import sonic_controller as sc

    env = None
    try:
        ck = PRIOR_ROOT / PRIORS[a.task] / "model_07999_actor158.pt"
        if hashlib.sha256(ck.read_bytes()).hexdigest() != PRIOR_SHA[a.task]:
            raise RuntimeError("prior sha256 mismatch")
        env = get_class(config.env_class)(get_tyro_env_config(config), device=a.device)
        env.set_is_evaluating()
        dev = env.device
        models = initialize_plan5_model_bundle(ck, g1_29dof_wbt_w_object.algo.config, device=dev, expected_sha256=PRIOR_SHA[a.task])
        models.actor.eval()
        sim = env.simulator
        cmd = json.loads(sc.DEFAULT_SPEC.read_text())["cmd_body_names"]
        b_idx = [sim._body_list.index(b) for b in cmd]
        pel = cmd.index("pelvis")
        ee = [cmd.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
        R = {k: torch.tensor(sref[k], dtype=torch.float32, device=dev) for k in ("root_quat_w", "body_pos_w")}
        env.reset_all()
        ids = torch.arange(env.num_envs, device=dev)
        env.reset_envs_idx(ids)
        sim.set_actor_root_state_tensor_robots(ids, sim.robot_root_states)
        sim.set_dof_state_tensor_robots(ids, sim.dof_state)
        sim.refresh_sim_tensors()
        env._compute_observations()
        env._clip_observations()
        obs = env.obs_buf_dict
        origin = sim.scene.env_origins[0]
        if int(env.command_manager.get_state("motion_command").time_steps[0].item()) != 0:
            raise RuntimeError("reference not at frame 0")
        rs0 = sim.robot_root_states
        init_xy = float(torch.linalg.norm((rs0[0, :2] - origin[:2]) - torch.tensor(sref["root_pos_w"][0, :2], device=dev)).item())
        rec = {k: [] for k in ("mpjpe_mm", "pelvis_dz", "pelvis_ori_err2", "ee_dz_max", "root_pos", "body_pos14")}
        fall_step, k_last = None, 0
        for k in range(1, N):
            x = torch.cat([obs["actor_obs"], obs["teammate_obs"]], dim=-1)
            with torch.no_grad():
                act = models.actor.act_inference({"actor_obs": models.actor_obs_normalizer(x)})
            obs, _, _, _ = env.step({"actions": act})
            body = sim._rigid_body_pos[:, b_idx] - origin
            ref = R["body_pos_w"][k][None]
            err = torch.linalg.norm(body - ref, dim=-1)
            rq = sc.xyzw_to_wxyz(sim.robot_root_states[:, 3:7])
            dq = sc.quat_mul(sc.quat_conj(R["root_quat_w"][k][None]), rq)
            ang = 2 * torch.atan2(torch.linalg.norm(dq[:, 1:], dim=-1), dq[:, 0].abs())
            pz = (body[:, pel, 2] - ref[:, pel, 2]).abs()
            ez = (body[:, ee, 2] - ref[:, ee, 2]).abs().max(dim=-1).values
            rec["mpjpe_mm"].append(err.mean().item() * 1000)
            rec["pelvis_dz"].append(pz.item())
            rec["pelvis_ori_err2"].append((ang ** 2).item())
            rec["ee_dz_max"].append(ez.item())
            rec["root_pos"].append((sim.robot_root_states[0, :3] - origin).cpu().numpy())
            rec["body_pos14"].append(body[0].cpu().numpy())
            k_last = k
            if pz.item() > 0.25 or ang.item() ** 2 > 1.0 or ez.item() > 0.25:
                fall_step = k
                break
        res = dict(task=a.task, prior=str(ck), prior_sha256=PRIOR_SHA[a.task], ref_prefix=str(a.ref_prefix),
                   provenance=json.loads(str(sref["provenance"])), steps_executed=k_last, horizon_steps=N - 1,
                   fell=fall_step is not None, fall_step=fall_step, mpjpe_g_mm=float(np.mean(rec["mpjpe_mm"])),
                   initial_pelvis_xy_err_m=init_xy, initial_yaw_err_deg=None, pd_target_minus_qstar_max_abs=None,
                   controller="single-person prior (no SONIC), deterministic mean, original PD/scales", device=a.device)
        np.savez_compressed(a.out / "trajectory.npz", **{k: np.asarray(v) for k, v in rec.items()},
                            ref_body_pos14=sref["body_pos_w"][: k_last + 1])
        (a.out / "result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
        return 0
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        return 1
    finally:
        if env is not None and hasattr(env.simulator, "close"):
            env.simulator.close()
        close_simulation_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
