#!/usr/bin/env python3
"""Smoke test of system1/t10b/package_friction.py helpers on synthetic records (no files, CPU): the 'hands only' read-back check and
the argv comparison with the friction-1.0 final (CPU slot and per-run arguments ignored, anything else compared)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t10b.package_friction import hand_only, strip_run_args  # noqa: E402


def rec(hf, changed=("left_rubber_hand_link", "right_rubber_hand_link"), same=True, val=None):
    after = [[1.0, 1.0, 0.0]] * 4
    after[2] = after[3] = val or [hf, hf, 0.0]
    r = dict(changed_bodies=list(changed), unchanged_shapes_identical=same, after=after, hand_shape_indices=[2, 3])
    return dict(per_robot=dict(robot=r, robot_1=dict(r)))


def main():
    assert hand_only(rec(0.5), 0.5) and hand_only(rec(0.75), 0.75)
    assert not hand_only(rec(0.5), 0.75)  # wrong value
    assert not hand_only(rec(0.5, changed=("left_rubber_hand_link", "right_rubber_hand_link", "torso_link")), 0.5)
    assert not hand_only(rec(0.5, same=False), 0.5)
    assert not hand_only(dict(per_robot=dict(robot=rec(0.5)["per_robot"]["robot"])), 0.5)  # one robot only
    a = ["taskset", "-c", "16-23", "py", "eval.py", "--case", "table", "--model-id", "x", "--output-dir", "o1", "--hand-friction", "0.5"]
    b = ["taskset", "-c", "24-31", "py", "eval.py", "--case", "table", "--model-id", "y", "--output-dir", "o2"]
    assert strip_run_args(a) == strip_run_args(b) == ["py", "eval.py", "--case", "table"]
    assert strip_run_args(a) != strip_run_args(b[:6] + ["bucket"] + b[7:])
    print("ok package_friction helpers")


if __name__ == "__main__":
    main()
