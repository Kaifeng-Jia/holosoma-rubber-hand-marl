#!/usr/bin/env python3
"""F1 offline check: SonicDualDriver path == single-robot path (TF32 off).

For each task, the same robot states (reference pose + fixed-seed perturbation) and the
same frames go through
  (dual)   system1.scheme_a.SonicDualDriver.actions() on a mocked paired env whose
           command.reference is the paper loader's reference object, and
  (single) one SonicController(batch=1) per robot fed exactly as validate_single.py
           does, from the exported single-robot reference (P.sonic.npz).
Checked per step: tokenizer/actor obs, raw actions, q*; the reference joint vector the
dual driver gives robot k equals the npz agent-k row bit for bit; the holosoma action
layout [E, A*29] puts robot k in columns k*29:(k+1)*29. Gate: max diff < 1e-6.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "holosoma"))
from system1 import scheme_a, sonic_controller as sc  # noqa: E402
from system1.sonic.task_refs import TASKS, load_task_reference, reference_file  # noqa: E402

GATE = 1e-6


class _Term:
    joint_target_override = None
    last_pd_target = None


def mock_env(ref, bodies, dofs, task, device):
    from holosoma.config_values.robot import g1_29dof_w_object as R

    E, A = 1, 2
    sim = SimpleNamespace(dof_names=dofs, _body_list=bodies, scene=SimpleNamespace(env_origins=torch.zeros(E, 3, device=device)),
                          agent_root_states=torch.zeros(E, A, 13, device=device), agent_dof_pos=torch.zeros(E, A, 29, device=device),
                          agent_dof_vel=torch.zeros(E, A, 29, device=device))
    term = _Term()
    frozen = torch.tensor(scheme_a.paper_action_scales(R), device=device)
    default = torch.tensor([R.init_state.default_joint_angles[n] for n in dofs], device=device)[None]
    rel = reference_file(task)
    command = SimpleNamespace(reference=ref, time_steps=torch.zeros(E, dtype=torch.long, device=device),
                              paired_reference_file=None if task == "push" else rel, motion_cfg=SimpleNamespace(motion_file=rel))
    env = SimpleNamespace(simulator=sim, num_envs=E, num_agents=A, dof_names=dofs, default_dof_pos=default, action_scales=frozen,
                          action_manager=SimpleNamespace(get_term=lambda name: term),
                          command_manager=SimpleNamespace(get_state=lambda name: command))
    return env, sim, command, term


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refs", type=Path, default=ROOT / "logs_system1/T01b/refs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device
    rng = np.random.default_rng(20261110)
    report = {}
    for task in TASKS:
        ref, bodies, dofs, meta = load_task_reference(task, device=dev)
        env, sim, command, term = mock_env(ref, bodies, dofs, task, dev)
        drv = scheme_a.SonicDualDriver(env, dev)  # TF32 off by default; runs the F4 assertions
        cap = {}
        orig_act = drv.ctl.act

        def wrapped(*args, **kw):
            out = orig_act(*args, **kw)
            cap["obs"] = out[2]
            return out

        drv.ctl.act = wrapped
        singles = [sc.SonicController(batch=1, device=dev, holosoma_joint_names=dofs) for _ in range(2)]
        R = [{k: torch.tensor(np.load(a.refs / f"{task}_agent{r}.sonic.npz")[k], dtype=torch.float32, device=dev)
              for k in ("dof_pos", "dof_vel", "root_quat_w")} for r in range(2)]
        npz = [np.load(a.refs / f"{task}_agent{r}.sonic.npz") for r in range(2)]
        N = ref.num_frames
        pel = bodies.index("pelvis")
        from holosoma.utils.path import resolve_data_file_path

        with np.load(resolve_data_file_path(str(reference_file(task))), allow_pickle=False) as raw:
            fj = [str(x) for x in raw["joint_names"]]
            permf = [fj.index(n) for n in drv.ctl.joint_names]
            if "agent_joint_pos" in raw.files:
                raw_q = raw["agent_joint_pos"][:, :, permf].astype(np.float32)  # [T, A, 29] SONIC order
            else:  # push: both agents replay the single A1 joints (joint_pos[:, 7:])
                raw_q = np.repeat(raw["joint_pos"][:, None, 7:][:, :, permf], 2, axis=1).astype(np.float32)
        worst = dict(tokenizer=0.0, actor_obs=0.0, action=0.0, q_target=0.0, ref_joint_vs_npz=0.0, ref_joint_vs_raw_file=0.0, layout=0.0)
        frames = list(range(min(a.steps, N)))
        if N > a.steps:
            frames = frames[: a.steps - 20] + list(range(N - 20, N))  # include the clamped tail
        for t in frames:
            smp = ref.sample(torch.tensor([t], device=dev))
            rs = torch.zeros(1, 2, 13, device=dev)
            rs[0, :, :3] = smp["agent_body_pos_w"][0, :, pel] + torch.tensor(rng.normal(0, 0.03, (2, 3)), dtype=torch.float32, device=dev)
            q = smp["agent_body_quat_w"][0, :, pel] + torch.tensor(rng.normal(0, 0.02, (2, 4)), dtype=torch.float32, device=dev)
            rs[0, :, 3:7] = q / q.norm(dim=-1, keepdim=True)
            rs[0, :, 7:13] = torch.tensor(rng.normal(0, 0.3, (2, 6)), dtype=torch.float32, device=dev)
            sim.agent_root_states = rs
            sim.agent_dof_pos = smp["agent_joint_pos"] + torch.tensor(rng.normal(0, 0.05, (1, 2, 29)), dtype=torch.float32, device=dev)
            sim.agent_dof_vel = smp["agent_joint_vel"] + torch.tensor(rng.normal(0, 0.3, (1, 2, 29)), dtype=torch.float32, device=dev)
            command.time_steps[:] = t
            a_h, act_d = drv.actions(command)
            jp_win, _, _, _ = drv.reference_window(command)
            obs_d = cap["obs"]
            for r in range(2):
                rr = rs[:, r]
                qw = sc.xyzw_to_wxyz(rr[:, 3:7])
                w_b = sc.quat_apply(sc.quat_conj(qw), rr[:, 10:13])
                ctl = singles[r]
                idx = sc.future_indices(torch.tensor([t], device=dev), R[r]["dof_pos"].shape[0], ctl.n_future, ctl.skip)
                act_s, qs_s, obs_s = ctl.act(qw, w_b, ctl.to_sonic(sim.agent_dof_pos[:, r]), ctl.to_sonic(sim.agent_dof_vel[:, r]),
                                             R[r]["dof_pos"][idx], R[r]["dof_vel"][idx], R[r]["root_quat_w"][idx])
                worst["tokenizer"] = max(worst["tokenizer"], (obs_d["tokenizer"][r] - obs_s["tokenizer"][0]).abs().max().item())
                worst["actor_obs"] = max(worst["actor_obs"], (obs_d["actor_obs"][r] - obs_s["actor_obs"][0]).abs().max().item())
                worst["action"] = max(worst["action"], (act_d[0, r] - act_s[0]).abs().max().item())
                q_h_s = ctl.to_holosoma(qs_s)[0]
                worst["q_target"] = max(worst["q_target"], (drv.last_q_target_h[0, r] - q_h_s).abs().max().item())
                ref_np = torch.tensor(npz[r]["dof_pos"][t], dtype=torch.float32, device=dev)
                worst["ref_joint_vs_npz"] = max(worst["ref_joint_vs_npz"], (jp_win[r, 0] - ref_np).abs().max().item())
                raw_t = torch.tensor(raw_q[t, r], device=dev)
                worst["ref_joint_vs_raw_file"] = max(worst["ref_joint_vs_raw_file"], (jp_win[r, 0] - raw_t).abs().max().item())
                expect = (q_h_s - drv.default_h) / drv.scale_h
                worst["layout"] = max(worst["layout"], (a_h[0, r * 29 : (r + 1) * 29] - expect).abs().max().item())
        exact = ("ref_joint_vs_npz", "ref_joint_vs_raw_file")
        passed = all(v < GATE for k, v in worst.items() if k not in exact) and all(worst[k] == 0.0 for k in exact)
        report[task] = dict(steps=len(frames), reference=meta, max_abs=worst, passed=passed)
        print(task, json.dumps(worst), "PASS" if passed else "FAIL", flush=True)
    out = dict(gate=GATE, tf32=False, infer_block=sc.INFER_BLOCK, tasks=report, passed=all(v["passed"] for v in report.values()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("ALL PASS" if out["passed"] else "SOME FAIL")


if __name__ == "__main__":
    main()
