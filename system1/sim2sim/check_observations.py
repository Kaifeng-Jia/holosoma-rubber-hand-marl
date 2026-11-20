#!/usr/bin/env python3
"""T10 step 2.2: observation construction. For every control step of the recorded Isaac nominal episodes (D and C-in, Table and
Bin, seed 721; system1/sim2sim/run_isaac_dumps.py), the Isaac state the observation was computed from is written into MuJoCo, the
158-D actor input is built by the MuJoCo-side code (mj_env.PairEnv.observation) and compared per dimension with the observation
Isaac recorded. Also compared: MuJoCo forward kinematics of all 32 Isaac bodies (the torso pose enters the observation), and for
C-in SONIC's two inputs (930-D proprioception history, 1761-D tokenizer) built by sonic_mj from the same states, the recorded C-in
actions and SONIC's recorded previous action. Threshold 1e-4 (task book).

    sim2sim_s1/bin/python -B system1/sim2sim/check_observations.py --out results_system1/T10_sim2sim/consistency
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.mj_env import PairEnv  # noqa: E402
from system1.sim2sim.sonic_mj import SonicOnnx  # noqa: E402

RUNS = dict(d_table=("table", "D"), cin_table=("table", "C-in"), d_bucket=("bucket", "D"), cin_bucket=("bucket", "C-in"))
NAMES = ([f"actions_{i}" for i in range(29)] + [f"base_ang_vel_{i}" for i in range(3)] + [f"dof_pos_{i}" for i in range(29)] + [f"dof_vel_{i}" for i in range(29)]
         + [f"motion_command_jpos_{i}" for i in range(29)] + [f"motion_command_jvel_{i}" for i in range(29)] + [f"motion_ref_ori_b_{i}" for i in range(6)]
         + ["teammate_relative_position_b_x", "teammate_relative_position_b_y", "teammate_relative_velocity_b_x", "teammate_relative_velocity_b_y"])
TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    per_dim, summary = [], {}
    for run, (task, scheme) in RUNS.items():
        base = ROOT / "logs_system1/t10/isaac" / run / "dump"
        z = np.load(base / "nominal.sim2sim.npz")
        env = PairEnv(base, task)
        K = z["actor_obs"].shape[0]
        err = np.zeros((K, 2, 158))
        fk_pos, fk_ang = 0.0, 0.0
        sonic = SonicOnnx(env.dof_names) if scheme == "C-in" else None
        s_err_a, s_err_t, s_err_act = 0.0, 0.0, 0.0
        sc_d = np.array(env.ss.control["action_scales"])
        lo, hi = np.array(env.ss.control["dof_pos_lower"]), np.array(env.ss.control["dof_pos_upper"])
        for k in range(K):
            root = z["root_states"][k, 0].astype(np.float64)
            env.set_state(root, z["dof_pos"][k, 0], z["dof_vel"][k, 0], np.zeros(3), np.array([0, 0, 0, 1.0]))
            st = env.state()
            fk_pos = max(fk_pos, float(np.abs(st["body_pos"] - z["body_pos_w"][k, 0]).max()))
            dq = np.abs(np.sum(st["body_quat_xyzw"] * z["body_quat_w_xyzw"][k, 0], axis=-1))
            fk_ang = max(fk_ang, float((2 * np.arccos(np.clip(dq, -1, 1))).max()))
            if scheme == "D":
                last = z["actions"][k - 1, 0].reshape(2, 29) if k > 0 else np.zeros((2, 29))
            else:
                last = z["policy_action"][k - 1, 0].reshape(2, 29) if k > 0 else np.zeros((2, 29))
            env.t = k
            actor, mate = env.observation(last, st=st)
            mine = np.concatenate([actor, mate], axis=1)
            isaac = np.concatenate([z["actor_obs"][k, 0], z["teammate_obs"][k, 0]], axis=1)
            err[k] = np.abs(mine - isaac)
            if sonic is not None:
                from system1.sim2sim.sonic_mj import CInController
                cin = CInController.__new__(CInController)
                cin.env, cin.sonic, cin.scale_d, cin.lo, cin.hi = env, sonic, sc_d, lo, hi
                a_used = np.clip(z["actions"][k, 0].reshape(2, 29), -1.0, 1.0)  # scheme_c a_used (policy_action is zeroed by the reset after the last step)
                if k == 0:
                    sonic.reset()
                else:
                    sonic.last_action = torch.as_tensor(np.clip(z["sonic_action"][k - 1], -sonic.clip, sonic.clip), dtype=torch.float32)  # [2, 29]
                act, _, sobs = sonic.act(*cin.sonic_inputs(st, a_used, k))
                s_err_a = max(s_err_a, float(np.abs(sobs["sonic_actor_obs"] - z["sonic_actor_obs"][k]).max()))
                s_err_t = max(s_err_t, float(np.abs(sobs["sonic_tokenizer"] - z["sonic_tokenizer"][k]).max()))
                s_err_act = max(s_err_act, float(np.abs(act - z["sonic_action"][k]).max()))
        # actor ONNX (system1/sim2sim/onnx_check.py export) on the recorded Isaac observations vs the actions Isaac computed (GPU)
        from system1.sim2sim.sonic_mj import ActorOnnx
        name = f"{scheme.replace('-', '').lower()}_{task}_seed721"
        act_onnx = ActorOnnx(ROOT / "exports_system1/t10/onnx" / f"{name}.onnx")
        obs_rec = np.concatenate([z["actor_obs"][:, 0], z["teammate_obs"][:, 0]], axis=-1).reshape(-1, 158)
        a_err = float(np.abs(act_onnx(obs_rec) - z["actions"][:, 0].reshape(-1, 29)).max())
        mx = err.max(axis=(0, 1))
        for i, n in enumerate(NAMES):
            per_dim.append(dict(run=run, task=task, scheme=scheme, dim=i, name=n, max_abs_diff=float(mx[i]), passed=bool(mx[i] < TOL)))
        summary[run] = dict(steps=int(K), obs_max_abs_diff=float(mx.max()), obs_worst_dims=[NAMES[i] for i in np.argsort(-mx)[:5]],
                            obs_passed=bool(mx.max() < TOL), fk_body_pos_max_m=fk_pos, fk_body_ori_max_rad=fk_ang,
                            actor_onnx_on_recorded_obs_vs_isaac_actions_max_abs=a_err)
        if sonic is not None:
            summary[run].update(sonic_actor_obs_max_abs_diff=s_err_a, sonic_tokenizer_max_abs_diff=s_err_t, sonic_inputs_passed=bool(max(s_err_a, s_err_t) < TOL),
                                sonic_action_onnx_vs_isaac_max_abs_diff=s_err_act)
        print(run, json.dumps(summary[run]), flush=True)
    with (a.out / "observations_per_dim.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_dim[0]))
        w.writeheader()
        w.writerows(per_dim)
    (a.out / "observations_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
