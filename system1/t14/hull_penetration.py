#!/usr/bin/env python3
"""T14: robot-object penetration and hand signed distance of two-robot references with exact convex distances (the same
measure for every task; eval_core4d_hulls.ObjectRobotDistance): object = the box-union URDF of that level (--ref
name=pair@urdf) or the task's PhysX pieces x k (--ref name=pair@dump:<task>:<k>); robot = PhysX link shapes; MuJoCo convex
collision, robot-object pairs only. Penetration depth of a pair = the convex penetration depth (minimum separating
translation), which also counts an object edge pushed into the side of a hand; the earlier point measure (largest depth of
the hand's surface samples) is reported next to it where available.

    sonic_s1/bin/python -B system1/t14/hull_penetration.py --ref k1.0=<pair>@<urdf> ... --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t14"))
import eval_core4d_hulls as eh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    robot = eh.cs.Robot("physx")
    res = {}
    for spec in a.ref:
        name, rest = spec.split("=", 1)
        path, src = rest.split("@", 1)
        if src.startswith("dump:"):
            _, task, k = src.split(":")
            meshes = eh.dump_object_meshes(task, float(k))
        else:
            meshes = eh.urdf_box_meshes(ROOT / src if not Path(src).is_absolute() else src)
        z = np.load(path)
        H, P = eh.ObjectRobotDistance(meshes, robot).frames(z["robot_qpos"].astype(np.float64), z["object_qpos"].astype(np.float64))
        res[name] = dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), object=src,
                         max_penetration_cm={f"r{r}": round(100 * float(P[:, r].max()), 2) for r in (0, 1)},
                         frames_penetrating={f"r{r}": round(float((P[:, r] > 0).mean()), 4) for r in (0, 1)},
                         hand_max_penetration_cm={f"r{r}": round(100 * float(max(H[:, r].max(), 0)), 2) for r in (0, 1)})
        print(name, json.dumps(res[name]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(rules=__doc__, refs=res), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
