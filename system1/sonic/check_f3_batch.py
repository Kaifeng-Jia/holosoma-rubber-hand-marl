#!/usr/bin/env python3
"""F3 batch-size invariance: the same SONIC inputs give bit-identical actions at batch 1, 2, 128.

Inputs are dumped native observations (TF32 off). For each of ``--steps`` time steps,
env-0 / env-1 observations are evaluated (a) alone (batch 1), (b) together (batch 2), and
(c) tiled 64x into a batch of 128; the max |difference| of each row against batch 1 is
reported. Run with hssim_s1 from the repository root.
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
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = json.loads(sc.DEFAULT_SPEC.read_text())
    pol = sc.load_sonic_policy(sc.DEFAULT_CHECKPOINT, spec, a.device)
    z = np.load(a.dump)
    tok, act = torch.tensor(z["obs/tokenizer"], device=a.device), torch.tensor(z["obs/actor_obs"], device=a.device)

    ctl = sc.SonicController.__new__(sc.SonicController)
    ctl.policy = pol

    def run_plain(t_obs, a_obs):
        pol.init_rollout()
        return pol.act_inference(obs_dict={"actor_obs": a_obs, "tokenizer": t_obs}, cur_dones=torch.zeros(a_obs.shape[0], device=a.device),
                                 skip_episode_attnmask=True)

    def run_block(t_obs, a_obs):  # the controller's inference path
        return ctl.infer({"actor_obs": a_obs, "tokenizer": t_obs}, torch.zeros(a_obs.shape[0], device=a.device))

    results = {}
    for mode, run in (("plain_batched", run_plain), ("controller_fixed_block", run_block)):
        results[mode] = check(run, tok, act, a.steps)
    rep = dict(dump=str(a.dump), steps=min(a.steps, tok.shape[0]), tf32=False, infer_block=sc.INFER_BLOCK, results=results,
               controller_identical=results["controller_fixed_block"]["identical"], device=a.device,
               gpu=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def check(run, tok, act, steps):

    worst = {"b2_vs_b1": 0.0, "b128_vs_b1": 0.0}
    for t in range(min(steps, tok.shape[0])):
        b1 = torch.cat([run(tok[t, e : e + 1], act[t, e : e + 1]) for e in range(2)])
        b2 = run(tok[t], act[t])
        b128 = run(tok[t].repeat(64, 1), act[t].repeat(64, 1)).reshape(64, 2, -1)
        worst["b2_vs_b1"] = max(worst["b2_vs_b1"], (b2 - b1).abs().max().item())
        worst["b128_vs_b1"] = max(worst["b128_vs_b1"], (b128 - b1[None]).abs().max().item())
    return dict(max_abs=worst, identical=all(v == 0.0 for v in worst.values()))


if __name__ == "__main__":
    main()
