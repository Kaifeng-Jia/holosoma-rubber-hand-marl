#!/usr/bin/env python3
"""Offline smoke tests of the T10b helpers (no simulator): the pilot stopping rule, the reference phase labels and the
validation-state files (count, sha256 recorded in the retrain README, no overlap with the frozen test states).

    sim2sim_s1/bin/python -B system1/t10b/test_t10b_offline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t10b.make_validation_states import TEST, key  # noqa: E402
from system1.t10b.phases import divergence_step, label_frames, segments  # noqa: E402
from system1.t10b.pilot_cloud import stop_decision  # noqa: E402

SHA = dict(table="87a716562282bb838e253fa7578c28d9839fba6ac0a981f4f01368f06e50670d",
           bucket="a2222d1fb1c7ba325bfcc87f16fa4619fa10dc8cc212d0dbe1e9e0a5cc502d0d")


def test_stop_rule():
    c = {2000: 10, 4000: 20, 6000: 30, 8000: 30, 10000: 29}
    assert not stop_decision(c)  # window (6000, 10000] not entirely after 8000
    assert stop_decision({**c, 12000: 30})  # 10000, 12000 <= best before (30 at 6000/8000)
    assert not stop_decision({**c, 12000: 31})  # improvement
    assert not stop_decision({2000: 1, 4000: 2, 6000: 3, 8000: 4, 10000: 5, 12000: 6})
    assert stop_decision({2000: 1, 4000: 2, 6000: 3, 8000: 9, 10000: 5, 12000: 6, 14000: 7, 16000: 9})


def test_phases():
    n = 200
    pos = np.zeros((n, 3))
    pos[:, 2] = 0.5
    pos[50:100, 2] = np.linspace(0.5, 1.0, 50)  # lift
    pos[100:150, 2] = 1.0  # carry
    pos[150:, 2] = 1.0
    pos[150:, 0] = np.linspace(0, 1, 50)  # moving while held -> carry (height above z0)
    q = np.tile([0, 0, 0, 1.0], (n, 1))
    lab = label_frames(pos, q)
    assert lab[0] == "still" and lab[75] == "lift" and lab[125] == "carry" and lab[175] == "carry", segments(lab)
    ref = pos.copy()
    obj = pos.copy()
    obj[120:, 0] += 0.2
    assert divergence_step(obj, ref, 0.1) == 120 and divergence_step(ref, ref) is None


def test_validation_states():
    for task, tests in TEST.items():
        p = ROOT / f"system1/t10b/validation_states/{task}_validation_states.json"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == SHA[task], task
        val = [s for s in json.loads(p.read_text())["states"] if s["kind"] != "nominal"]
        assert len(val) == 50
        test_keys = {key(s) for t in tests for s in json.loads(Path(t).read_text())["states"] if s["kind"] != "nominal"}
        assert not any(key(s) in test_keys for s in val), task


if __name__ == "__main__":
    for f in (test_stop_rule, test_phases, test_validation_states):
        f()
        print("ok", f.__name__)
