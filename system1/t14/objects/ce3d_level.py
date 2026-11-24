#!/usr/bin/env python3
"""T14 step 8 item 6: 3-D contact-position error (contact_error3d.py, round-6 definition: the person's target on the OMOMO
capture table moved to the training table by the similarity, then the role isometry of the level) of one Push / Pull pair."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "system1/t14"))
sys.path.insert(0, str(ROOT / "system1/t12v2"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
import contact_error3d as ce  # noqa: E402
import contact_gap as cg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=("push", "pull"), required=True)
ap.add_argument("--pair", type=Path, required=True)
ap.add_argument("--m", type=float, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
z = np.load(a.pair)
res = ce.omomo(a.task, cg.cs.Robot("physx"), cg.palm_offsets(), ce.human_omomo(a.task), z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64), a.m)
a.out.write_text(json.dumps(dict(task=a.task, m=a.m, pair=str(a.pair), pair_sha256=hashlib.sha256(a.pair.read_bytes()).hexdigest(), rules=ce.__doc__, result=res), indent=1) + "\n")
print(json.dumps(res)[:400])
