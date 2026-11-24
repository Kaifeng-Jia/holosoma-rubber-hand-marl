#!/usr/bin/env python3
"""T13 v4 section 4 A: compare two --t13v4-trace directories iteration by iteration (actions, rewards, dones, values,
log-probs; the update metrics: losses, entropy, KL, grad norms)."""
import json
import sys
from pathlib import Path

import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])
out = dict(a=str(a), b=str(b), iterations=[])
for fa in sorted(a.glob("iter_*.npz")):
    fb = b / fa.name
    za, zb = np.load(fa), np.load(fb)
    ja, jb = json.loads(fa.with_suffix(".json").read_text()), json.loads(fb.with_suffix(".json").read_text())
    arr = {k: float(np.abs(za[k].astype(np.float64) - zb[k].astype(np.float64)).max()) for k in za.files if k in zb.files}
    met = {k: abs(ja[k] - jb[k]) for k in ("surrogate_loss", "value_loss", "entropy", "kl", "actor_grad_norm", "critic_grad_norm", "reward_mean", "reward_min", "reward_max")
           if isinstance(ja.get(k), (int, float)) and isinstance(jb.get(k), (int, float))}
    out["iterations"].append(dict(iteration=fa.stem, arrays_max_abs_diff=arr, metrics_abs_diff=met,
                                  bitwise_equal=all(np.array_equal(za[k], zb[k]) for k in za.files if k in zb.files)))
out["all_bitwise_equal"] = all(r["bitwise_equal"] for r in out["iterations"])
out["max_abs_diff"] = max((max(r["arrays_max_abs_diff"].values(), default=0.0) for r in out["iterations"]), default=None)
print(json.dumps(out, indent=1))
