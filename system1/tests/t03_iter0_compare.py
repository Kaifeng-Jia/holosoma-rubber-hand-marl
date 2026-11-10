#!/usr/bin/env python3
"""T03 iteration-0 check: scheme C at iteration 0 (zero output layer, a = 0) must reproduce
scheme A on the same frozen states. Compares every trajectory array and the episode rows of
<root>/<case>_sonic vs <root>/<case>_sonic_residual; writes one JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SKIP_EPISODE_KEYS = ("model_id", "trajectory", "controller", "checkpoint", "checkpoint_sha256")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cases", default="pull,table,bucket")
    p.add_argument("--checkpoints", nargs="*", default=[], help="case=path of the model_00000.pt used")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ck = dict(x.split("=", 1) for x in a.checkpoints)
    res, ok = {}, True
    for c in a.cases.split(","):
        A, C = a.root / f"{c}_sonic", a.root / f"{c}_sonic_residual"
        trajs = {}
        for f in sorted(A.glob("*.trajectory.npz")):
            x, y = np.load(f), np.load(C / f.name)
            d = {}
            for k in sorted(set(x.files) | set(y.files)):
                if k not in x.files or k not in y.files or x[k].shape != y[k].shape:
                    d[k] = "missing_or_shape_mismatch"
                else:
                    d[k] = float(np.abs(x[k].astype(np.float64) - y[k].astype(np.float64)).max()) if x[k].size else 0.0
            trajs[f.name] = d
        ea, ec = json.loads((A / "episodes.json").read_text()), json.loads((C / "episodes.json").read_text())
        ep_diff = []
        for ra, rc in zip(ea, ec):
            for k in ra:
                if k in SKIP_EPISODE_KEYS or k.startswith("sonic_residual"):
                    continue
                if ra.get(k) != rc.get(k):
                    ep_diff.append(dict(state_id=ra.get("state_id"), key=k, scheme_a=ra.get(k), scheme_c0=rc.get(k)))
        max_traj = max((v for d in trajs.values() for v in d.values() if isinstance(v, float)), default=None)
        bad = any(isinstance(v, str) for d in trajs.values() for v in d.values())
        case_ok = (not bad) and max_traj == 0.0 and not ep_diff and len(ea) == len(ec)
        ok &= case_ok
        res[c] = dict(episodes=len(ea), trajectories=len(trajs), max_abs_trajectory_diff=max_traj, episode_field_diffs=ep_diff,
                      per_trajectory=trajs, identical=case_ok,
                      checkpoint=ck.get(c), checkpoint_sha256=hashlib.sha256(Path(ck[c]).read_bytes()).hexdigest() if c in ck else None)
    out = dict(check="T03 iteration 0: scheme C (zero output layer) vs scheme A, TF32 off, same frozen states", root=str(a.root),
               passed=bool(ok), cases=res)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({c: dict(identical=r["identical"], max=r["max_abs_trajectory_diff"], ep_diffs=len(r["episode_field_diffs"]), n=r["episodes"])
                      for c, r in res.items()}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
