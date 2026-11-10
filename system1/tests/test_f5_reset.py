#!/usr/bin/env python3
"""F5: SonicController.reset(ids) clears only the given environments (TF32 off).

Batch of 8 robots, 8 steps of fixed-seed inputs. Run A never resets; run B resets
ids {2, 5} before step 3 (and passes dones=1 for them). Gate: every action of the six
other robots is bit-identical between A and B. Also reported: the reset robots' actions
from step 3 on equal those of a fresh controller started at step 3 (expected identical).
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
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev, B, T, RESET_AT, IDS = a.device, 8, 8, 3, [2, 5]
    rng = np.random.default_rng(5)
    t = lambda *s: torch.tensor(rng.normal(0, 1, s), dtype=torch.float32, device=dev)  # noqa: E731
    steps = []
    for _ in range(T):
        q = t(B, 4)
        steps.append(dict(q=q / q.norm(dim=-1, keepdim=True), w=t(B, 3) * 0.3, jp=t(B, 29) * 0.3, jv=t(B, 29),
                          fp=t(B, 10, 29) * 0.3, fv=t(B, 10, 29), fq=torch.nn.functional.normalize(t(B, 10, 4), dim=-1)))

    def run(reset, start=0, batch_ids=None):
        n = B if batch_ids is None else len(batch_ids)
        ctl = sc.SonicController(batch=n, device=dev)
        out = []
        for k in range(start, T):
            s = {kk: (v if batch_ids is None else v[batch_ids]) for kk, v in steps[k].items()}
            dones = torch.zeros(n, device=dev)
            if reset and k == RESET_AT:
                ctl.reset(torch.tensor(IDS, device=dev))
                dones[IDS] = 1.0
            act, _, _ = ctl.act(s["q"], s["w"], s["jp"], s["jv"], s["fp"], s["fv"], s["fq"], dones)
            out.append(act.clone())
        return torch.stack(out)

    A, Bres = run(False), run(True)
    others = [i for i in range(B) if i not in IDS]
    diff_others = (A[:, others] - Bres[:, others]).abs().max().item()
    fresh = run(False, start=RESET_AT, batch_ids=IDS)
    diff_fresh = (Bres[RESET_AT:, IDS] - fresh).abs().max().item()
    changed = (A[RESET_AT:, IDS] - Bres[RESET_AT:, IDS]).abs().max().item()
    rep = dict(batch=B, steps=T, reset_at=RESET_AT, reset_ids=IDS, tf32=False,
               max_abs_other_envs_vs_no_reset=diff_others, max_abs_reset_envs_vs_fresh_start=diff_fresh,
               reset_envs_changed_by=changed, passed=diff_others == 0.0)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
