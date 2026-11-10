#!/usr/bin/env python3
"""T02 gate "坐标口径": rebuild the teleop-encoder inputs from a SONIC native dump and compare with the
tokenizer SONIC's own environment computed (obs/tokenizer slices). Threshold: max abs diff < 1e-5.

Inputs per step t / env e (same indexing as check_obs_equivalence.py, frame = time_steps + motion_start_time_steps):
reference dof pos/vel (future window), reference pelvis / torso / wrist body poses (14 cmd bodies), robot pelvis quat.
Terms checked: command_multi_future_lower_body, vr_3point_local_target, vr_3point_local_orn_target, motion_anchor_ori_b.
No simulator. Run with hssim_s1.
"""
from __future__ import annotations

import argparse
import hashlib
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
    a = p.parse_args()
    z = np.load(a.dump)
    spec = json.loads(sc.DEFAULT_SPEC.read_text())
    names = spec["cmd_body_names"]
    pidx = [names.index(n) for n in sc.VR3_BODIES]
    aidx = names.index(spec["anchor_body"])
    lay = sc.TokenizerLayout.from_spec(spec)
    off = dict(zip(lay.names, np.cumsum([0] + lay.dims[:-1])))
    sl = {n: slice(int(off[n]), int(off[n]) + d) for n, d in zip(lay.names, lay.dims)}
    T = min(a.steps, z["action"].shape[0])
    E = z["action"].shape[1]
    d64 = torch.float64
    g = lambda k: torch.tensor(z[k], dtype=d64)  # noqa: E731
    refs = [{k: g(f"ref{e}/{k}") for k in ("dof_pos", "dof_vel", "body_pos_w", "body_quat_w")} for e in range(E)]
    tok = g("obs/tokenizer")
    ts = (g("time_steps") + g("motion_start_time_steps")).long()
    terms = ("command_multi_future_lower_body", "vr_3point_local_target", "vr_3point_local_orn_target", "motion_anchor_ori_b")
    worst = {k: 0.0 for k in terms}
    where = {k: None for k in terms}
    nonzero = {k: 0.0 for k in terms}
    for t in range(T):
        for e in range(E):
            r = refs[e]
            idx = sc.future_indices(ts[t, e:e + 1], r["dof_pos"].shape[0], int(spec["num_future_frames"]), int(spec["frame_skips"]))[0]
            f = ts[t, e]
            cmd, local, orn, ori = sc.teleop_encoder_terms(
                r["dof_pos"][idx][None], r["dof_vel"][idx][None], r["body_pos_w"][f, aidx][None], r["body_quat_w"][f, aidx][None],
                r["body_pos_w"][f, pidx][None], r["body_quat_w"][f, pidx][None], g("pre/root_quat_w")[t, e][None])
            for k, v in zip(terms, (cmd, local, orn, ori)):
                ref = tok[t, e, sl[k]]
                dmax = float((v[0] - ref).abs().max())
                nonzero[k] = max(nonzero[k], float(ref.abs().max()))
                if dmax > worst[k]:
                    worst[k], where[k] = dmax, (t, e)
    passed = all(v < 1e-5 for v in worst.values())
    rep = dict(dump=str(a.dump), dump_sha256=hashlib.sha256(a.dump.read_bytes()).hexdigest(), steps=T, envs=E,
               threshold=1e-5, max_abs_diff=worst, at_step_env=where, max_abs_value_in_sonic_obs=nonzero,
               vr3_bodies=list(sc.VR3_BODIES), vr3_offsets=[list(x) for x in sc.VR3_OFFSETS],
               lower_body_sonic_indices=list(sc.LOWER_BODY_SONIC), passed=passed,
               note="float64 recomputation from float32 dump arrays; SONIC computed in float32")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
