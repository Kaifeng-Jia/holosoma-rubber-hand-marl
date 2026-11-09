#!/usr/bin/env python3
"""Offline check of system1/sonic_controller.py against SONIC's own environment.

Input: a native dump from native_eval_dump.py (G1-only encoder run). For each dumped
env and control step, the controller rebuilds the tokenizer (G1 encoder part) and
proprioceptive observations from the dumped robot states and reference arrays, and
runs the policy. Reported:
  * max |obs_rebuilt - obs_sonic| for the G1 encoder slice and the 930-D actor obs;
  * max |action(obs_sonic) - action_sonic| (model loading / inference path);
  * max |action(obs_rebuilt) - action_sonic| (full controller path).
No simulator is involved. Run with hssim_s1 from the repository root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1 import sonic_controller as sc  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    z = np.load(a.dump)
    dev = a.device
    T = min(a.steps, z["action"].shape[0])
    E = z["action"].shape[1]
    ctl = sc.SonicController(batch=E, device=dev)
    # SONIC's eval env randomizes each env's default joint pos at startup (add_joint_default_pos,
    # U(-0.01, 0.01)); the per-env values equal the action offsets stored in the dump.
    ctl.default_joint_pos = torch.tensor(z["static/action_offset"].reshape(-1, 29)[:E], device=dev)
    lay = ctl.layout
    off = dict(zip(lay.names, np.cumsum([0] + lay.dims[:-1])))
    sl = {n: slice(off[n], off[n] + d) for n, d in zip(lay.names, lay.dims)}
    g = lambda k: torch.tensor(z[k], device=dev)  # noqa: E731
    refs = [{k: torch.tensor(z[f"ref{e}/{k}"], device=dev) for k in ("dof_pos", "dof_vel", "root_quat_w")} for e in range(E)]
    tok_dump, act_obs_dump, act_dump = g("obs/tokenizer"), g("obs/actor_obs"), g("action")
    tsteps = g("time_steps").long() + g("motion_start_time_steps").long()
    stats = {k: 0.0 for k in ("tok_cmd", "tok_ori", "tok_encidx", "actor_obs", "action_from_sonic_obs", "action_from_rebuilt_obs")}
    per_term = {n: 0.0 for n, _ in sc.PROPRIO_TERMS}
    ctl.reset(torch.arange(E, device=dev))
    ctl_ref = sc.SonicController.__new__(sc.SonicController)  # second policy instance for the sonic-obs path
    ctl_ref.policy = sc.load_sonic_policy(sc.DEFAULT_CHECKPOINT, ctl.spec, dev)
    ctl_ref.policy.init_rollout()
    for t in range(T):
        idx = torch.stack([sc.future_indices(tsteps[t, e : e + 1], refs[e]["dof_pos"].shape[0], ctl.n_future, ctl.skip)[0] for e in range(E)])
        pos_f = torch.stack([refs[e]["dof_pos"][idx[e]] for e in range(E)])
        vel_f = torch.stack([refs[e]["dof_vel"][idx[e]] for e in range(E)])
        quat_f = torch.stack([refs[e]["root_quat_w"][idx[e]] for e in range(E)])
        dones = torch.zeros(E, device=dev)
        action, _, obs = ctl.act(g("pre/root_quat_w")[t], g("pre/root_ang_vel_b")[t], g("pre/joint_pos")[t], g("pre/joint_vel")[t],
                                 pos_f, vel_f, quat_f, dones)
        tok = obs["tokenizer"]
        stats["tok_cmd"] = max(stats["tok_cmd"], (tok[:, sl["command_multi_future_nonflat"]] - tok_dump[t][:, sl["command_multi_future_nonflat"]]).abs().max().item())
        stats["tok_ori"] = max(stats["tok_ori"], (tok[:, sl["motion_anchor_ori_b_mf_nonflat"]] - tok_dump[t][:, sl["motion_anchor_ori_b_mf_nonflat"]]).abs().max().item())
        stats["tok_encidx"] = max(stats["tok_encidx"], (tok[:, sl["encoder_index"]] - tok_dump[t][:, sl["encoder_index"]]).abs().max().item())
        d = (obs["actor_obs"] - act_obs_dump[t]).abs()
        stats["actor_obs"] = max(stats["actor_obs"], d.max().item())
        o = 0
        for n, w in sc.PROPRIO_TERMS:
            per_term[n] = max(per_term[n], d[:, o : o + w * sc.HISTORY].max().item())
            o += w * sc.HISTORY
        a_ref = ctl_ref.policy.act_inference(obs_dict={"actor_obs": act_obs_dump[t], "tokenizer": tok_dump[t]}, cur_dones=dones, skip_episode_attnmask=True)
        stats["action_from_sonic_obs"] = max(stats["action_from_sonic_obs"], (a_ref - act_dump[t]).abs().max().item())
        stats["action_from_rebuilt_obs"] = max(stats["action_from_rebuilt_obs"], (action - act_dump[t]).abs().max().item())
        # Feed back SONIC's own action so later histories stay aligned even if a tiny difference appears.
        ctl.last_action = act_dump[t].clamp(-ctl.action_clip, ctl.action_clip).clone()
    report = dict(dump=str(a.dump), steps=T, envs=E, max_abs=stats, actor_obs_per_term=per_term)
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
